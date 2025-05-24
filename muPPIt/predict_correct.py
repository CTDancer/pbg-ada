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
from tqdm import tqdm

from models.graph import ProteinGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class muPPIt(pl.LightningModule):
    def __init__(self, d_node, d_edge, d_cross_edge, d_position, num_heads, 
                num_intra_layers, num_mim_layers, num_cross_layers, lr):
        super(muPPIt, self).__init__()

        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.esm.parameters():
            param.requires_grad = False

        self.graph = ProteinGraph(d_node, d_edge, d_position)

        self.intra_graph_att_layers = nn.ModuleList([
            IntraGraphAttention(d_node, d_edge, num_heads) for _ in range(num_intra_layers)
        ])

        self.diff_layer = DiffEmbeddingLayer(d_node)

        self.mim_layers = nn.ModuleList([
            MIM(d_node, d_edge, d_node, num_heads) for _ in range(num_mim_layers)
        ])

        self.cross_graph_att_layers = nn.ModuleList([
            CrossGraphAttention(d_node, d_cross_edge, d_node, num_heads) for _ in range(num_cross_layers)
        ])

        self.affinity_linear = nn.Sequential(
            nn.Linear(d_cross_edge, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 15)
        )

        self.d_cross_edge = d_cross_edge
        self.learning_rate = lr

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('muppit.', ''): v for k, v in state_dict.items() if 'muppit' in k}

        self.load_state_dict(state_dict, strict=True)

        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, binder_tokens, wt_tokens, mut_tokens):
        device = binder_tokens.device

        # Construct Graph
        binder_node, binder_edge, _, _ = self.graph(binder_tokens, self.esm, self.alphabet)
        wt_node, wt_edge, _, _ = self.graph(wt_tokens, self.esm, self.alphabet)
        mut_node, mut_edge, _, _ = self.graph(mut_tokens, self.esm, self.alphabet)

        # Intra-Graph Attention
        for layer in self.intra_graph_att_layers:
            binder_node, binder_edge = layer(binder_node, binder_edge)
            wt_node, wt_edge = layer(wt_node, wt_edge)
            mut_node, mut_edge = layer(mut_node, mut_edge)

        # Differential Embedding Layer
        diff_vec = self.diff_layer(wt_node, mut_node)

        # Mutation Impact Module
        for layer in self.mim_layers:
            wt_node, wt_edge = layer(wt_node, wt_edge, diff_vec)
            mut_node, mut_edge = layer(mut_node, mut_edge, diff_vec)

        # Initialize cross-graph edges
        B = mut_node.shape[0]
        L_mut = mut_node.shape[1]
        L_wt = wt_node.shape[1]
        L_binder = binder_node.shape[1]

        mut_binder_edges = torch.randn(B, L_mut, L_binder, self.d_cross_edge).to(device)
        wt_binder_edges = torch.randn(B, L_wt, L_binder, self.d_cross_edge).to(device)

        # Cross-Graph Attention
        for layer in self.cross_graph_att_layers:
            wt_node, binder_node, wt_binder_edges = layer(wt_node, binder_node, wt_binder_edges, diff_vec)
            binder_node_wt = binder_node

            mut_node, binder_node, mut_binder_edges = layer(mut_node, binder_node, mut_binder_edges, diff_vec)
            binder_node_mut = binder_node

            binder_node = (binder_node_wt + binder_node_mut) / 2

        wt_binder_edges = torch.mean(wt_binder_edges, dim=(1,2))
        mut_binder_edges = torch.mean(mut_binder_edges, dim=(1,2))

        mut_affinity = self.affinity_linear(mut_binder_edges)
        wt_affinity = self.affinity_linear(wt_binder_edges)

        # mut_binder_distance = F.cosine_similarity(torch.mean(mut_node, dim=1), torch.mean(binder_node, dim=1), dim=1)
        # wt_binder_distance = F.cosine_similarity(torch.mean(wt_node, dim=1), torch.mean(binder_node, dim=1))

        return mut_affinity, wt_affinity


def main(args):
    print(args.sm)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    muppit = muPPIt(args.d_node, args.d_edge, args.d_cross_edge, args.d_position, args.n_heads,
                    args.n_intra_layers, args.n_mim_layers, args.n_cross_layers, args.lr)
    muppit.load_weights(args.sm)

    device = muppit.device
    muppit.eval()

    with torch.no_grad():
        binder_tokens = tokenizer(args.binder, return_tensors='pt')['input_ids'][:, 1:-1].to(device)
        mut_tokens = tokenizer(args.mut, return_tensors='pt')['input_ids'][:, 1:-1].to(device)
        wt_tokens = tokenizer(args.wt, return_tensors='pt')['input_ids'][:, 1:-1].to(device)
        
        mut_pred_affinity, wt_pred_affinity = muppit(binder_tokens, wt_tokens, mut_tokens)
    
    print(f"Binder: {args.binder}")
    print(f"Mutant Affinity = {torch.argmax(mut_pred_affinity) - 16}")
    print(f"Wildtype Affinity = {torch.argmax(wt_pred_affinity) - 16}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-wt", type=str)
    parser.add_argument("-mut", type=str)
    parser.add_argument("-binder", type=str)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-d_node", type=int, default=1024, help="Node Representation Dimension")
    parser.add_argument("-d_edge", type=int, default=512, help="Intra-Graph Edge Representation Dimension")
    parser.add_argument("-d_cross_edge", type=int, default=512, help="Cross-Graph Edge Representation Dimension")
    parser.add_argument("-d_position", type=int, default=8, help="Positional Embedding Dimension")
    parser.add_argument("-n_heads", type=int, default=8)
    parser.add_argument("-n_intra_layers", type=int, default=1)
    parser.add_argument("-n_mim_layers", type=int, default=1)
    parser.add_argument("-n_cross_layers", type=int, default=1)
    parser.add_argument("-sm", default=None, help="File containing initial params", type=str)
    parser.add_argument("-max_epochs", type=int, default=15, help="Max number of epochs to train")
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("-grad_clip", type=float, default=0.5)

    args = parser.parse_args()
    main(args)