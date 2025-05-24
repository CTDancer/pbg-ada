import pdb
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler, Sampler
from datasets import load_from_disk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, \
    Timer, TQDMProgressBar, LearningRateMonitor, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import _LRScheduler
from transformers.optimization import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import os
import math
import esm
import numpy as np
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Adam, AdamW
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import gc
import esm

from models.graph import NodeGraph, EdgeGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class EdgeAwareMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(EdgeAwareMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_heads)
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, edge_representation, mask=None):
        # query, key, value: (batch_size, L, d_model)
        # edge_representation: (batch_size, L, L)
        # mask: (batch_size, L) binary mask with 1 for valid tokens and 0 for padding

        batch_size, L, _ = query.size()

        # pdb.set_trace()

        Q = self.q_proj(query).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)

        edge_representation = edge_representation.unsqueeze(-1)  # Shape: (batch_size, L, L, 1)
        edge_bias = self.edge_mlp(edge_representation).squeeze(-1)  # Shape: (batch_size, L, L, num_heads)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)   # Shape: (batch_size, num_heads, L, L)

        # Incorporate edge bias (added directly after expanding)
        attn_scores = attn_scores + edge_bias.permute(0, 3, 1, 2)  # Shape: (batch_size, num_heads, L, L)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, L)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, L, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = EdgeAwareMultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_representation, mask=None):
        # x: (batch_size, L, d_model)
        # edge_representation: (batch_size, L, L)
        # mask: (batch_size, L)

        # pdb.set_trace()
        x2 = self.self_attn(x, x, x, edge_representation, mask)
        x = x + x2
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + x2
        x = self.norm2(x)
        return x


class ProteinDecoder(pl.LightningModule):
    def __init__(self, d_model=1296, nhead=8, num_layers=6, dim_feedforward=2048, vocab_size=24, dropout=0.2, lr=1e-3):
        super(ProteinDecoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.learning_rate = lr
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, node_representation, edge_representation, mask=None):
        # node_representation: (batch_size, L, d_model)
        # edge_representation: (batch_size, L, L)
        # mask: (batch_size, L)

        for layer in self.layers:
            node_representation = layer(node_representation, edge_representation, mask)

        logits = self.fc_out(node_representation)  # Shape: (batch_size, L, vocab_size)

        return logits


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", type=str, default='AHTGR')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-d_node", type=int, default=256, help="Node representation dimension")
    parser.add_argument("-output_dim", type=int, default=26, help="Decoder output dimension")
    parser.add_argument("-n_layers", type=int, default=6, help="num decoder layers")
    parser.add_argument("-n_heads", type=int, default=8, help="num attention heads")
    parser.add_argument("-d_ff", type=int, default=2048, help="Feed forward layer dimension")
    parser.add_argument("-sm", default=None, help="File containing initial params", type=str)
    parser.add_argument("-max_epochs", type=int, default=15, help="Max number of epochs to train")
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("-grad_clip", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    print(f"Loading model from {args.sm}")

    decoder = ProteinDecoder.load_from_checkpoint(args.sm,
                                            d_model = args.d_node, 
                                            nhead = args.n_heads, 
                                            num_layers = args.n_layers, 
                                            dim_feedforward = args.d_ff,
                                            vocab_size = 24,
                                            dropout = args.dropout,
                                            lr = args.lr)
    device = decoder.device
    decoder.eval()

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    nodegraph = NodeGraph(8)
    edgegraph = EdgeGraph()

    idx_to_aa = {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}

    sequence = tokenizer(args.s, return_tensors='pt')
    seq_tokens = sequence['input_ids'][:, 1:-1]

    node_representations = nodegraph(seq_tokens, esm_model, alphabet)
    edge_representations = edgegraph(seq_tokens, esm_model)

    # pdb.set_trace()

    logits = decoder(node_representations.to(device), edge_representations.to(device))

    # pdb.set_trace()
    predicted_indices = torch.argmax(logits.squeeze(0), dim=1)
    predicted_sequence = [idx_to_aa[idx.item()] for idx in predicted_indices]
    predicted_sequence_str = ''.join(predicted_sequence)
    print("Predicted Amino Acid Sequence:", predicted_sequence_str)
    print(f"{predicted_sequence_str == args.s}")

    # pdb.set_trace()



if __name__ == "__main__":
    main()
