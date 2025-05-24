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
import uuid
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

from models.graph import NodeGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class GPTDecoder(pl.LightningModule):
    def __init__(self, d_node, output_dim=26, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, lr=1e-5):
        super(GPTDecoder, self).__init__()
        self.d_node = d_node
        self.output_dim = output_dim

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_node, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Final linear layer to predict amino acids
        self.fc_out = nn.Linear(d_node, output_dim)

        self.learning_rate = lr
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, node_representations):
        # node_representations: (batch_size, L, d_node)

        # pdb.set_trace()
        batch_size, seq_len, _ = node_representations.size()    # shape: (B, L, d_node)
        
        # Transpose to match Transformer layer input (seq_len, batch_size, d_node)
        tgt_embedding = node_representations.transpose(0, 1)
        
        # Pass through the GPT layers
        for layer in self.layers:
            tgt_embedding = layer(tgt_embedding, tgt_embedding)  # Decoding is done with self-attention
        
        # Transpose back to (batch_size, seq_len, d_node)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        
        # Predict the next amino acid in the sequence
        logits = self.fc_out(tgt_embedding)  # shape: (batch_size, L, output_dim)
        
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

    decoder = GPTDecoder.load_from_checkpoint(args.sm,
                                            d_node = args.d_node, 
                                            output_dim = args.output_dim, 
                                            n_layers = args.n_layers, 
                                            n_heads = args.n_heads, 
                                            d_ff = args.d_ff,
                                            dropout = args.dropout,
                                            lr = args.lr)

    device = decoder.device
    decoder.eval()

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    graph = NodeGraph(8)
    idx_to_aa = {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}

    sequence = tokenizer(args.s, return_tensors='pt', padding=True, truncation=True, max_length=500)
    seq_tokens = sequence['input_ids'][:, 1:-1]

    node_representations= graph(seq_tokens, esm_model, alphabet)

    # pdb.set_trace()

    logits = decoder(node_representations.to(device))

    # pdb.set_trace()
    predicted_indices = torch.argmax(logits.squeeze(0), dim=1)
    predicted_sequence = [idx_to_aa[idx.item()] for idx in predicted_indices]
    predicted_sequence_str = ''.join(predicted_sequence)
    print("Predicted Amino Acid Sequence:", predicted_sequence_str)

    # pdb.set_trace()



if __name__ == "__main__":
    main()
