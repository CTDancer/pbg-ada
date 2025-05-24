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



def collate_fn(batch):
    # Unpack the batch
    binders = []
    mutants = []
    wildtypes = []
    deltas = []
    nums = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    for b in batch:
        binders.append(torch.tensor(b['binder_tokens']).squeeze(0))  # shape: 1*L1 -> L1
        mutants.append(torch.tensor(b['mutant_tokens']).squeeze(0))  # shape: 1*L2 -> L2
        wildtypes.append(torch.tensor(b['wildtype_tokens']).squeeze(0))  # shape: 1*L3 -> L3
        nums.append(torch.tensor(b['delta']))

        if b['delta'] <= -2:
            deltas.append(torch.tensor(1))
        else:
            deltas.append(torch.tensor(0))

    
    # Collate the tensors using torch's pad_sequence
    binder_input_ids = torch.nn.utils.rnn.pad_sequence(binders, batch_first=True, padding_value=tokenizer.pad_token_id)

    mutant_input_ids = torch.nn.utils.rnn.pad_sequence(mutants, batch_first=True, padding_value=tokenizer.pad_token_id)

    wildtype_input_ids = torch.nn.utils.rnn.pad_sequence(wildtypes, batch_first=True, padding_value=tokenizer.pad_token_id)

    deltas = torch.stack(deltas)    
    nums = torch.stack(nums)

    # Return the collated batch
    return {
        'binder_input_ids': binder_input_ids.int(),
        'mutant_input_ids': mutant_input_ids.int(),
        'wildtype_input_ids': wildtype_input_ids.int(),
        'delta': deltas,
        'num': nums
    }



class CustomDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.testloader = self.test_dataloader()

    def test_dataloader(self):
        test_dataset = load_from_disk('/home/tc415/muPPIt/dataset/val/delta_correct_2')
        return DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, num_workers=8, pin_memory=True)


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

        # self.cross_graph_edge_mapping = nn.Linear(1, d_cross_edge)
        # self.delta_mapping = nn.Linear(1, 1)
        # self.mapping = nn.Linear(d_cross_edge, 1)
        self.delta_linear = nn.Sequential(
            nn.Linear(2*d_cross_edge, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.d_cross_edge = d_cross_edge
        self.learning_rate = lr

        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7/3]))
        # self.criterion = nn.L1Loss()


    def forward(self, binder_tokens, wt_tokens, mut_tokens):
        device = binder_tokens.device

        # Construct Graph
        binder_node, binder_edge, binder_node_mask, binder_edge_mask = self.graph(binder_tokens, self.esm, self.alphabet)
        wt_node, wt_edge, wt_node_mask, wt_edge_mask = self.graph(wt_tokens, self.esm, self.alphabet)
        mut_node, mut_edge, mut_node_mask, mut_edge_mask = self.graph(mut_tokens, self.esm, self.alphabet)

        # Intra-Graph Attention
        for layer in self.intra_graph_att_layers:
            binder_node, binder_edge = layer(binder_node, binder_edge)
            binder_node = binder_node * binder_node_mask.unsqueeze(-1)
            binder_edge = binder_edge * binder_edge_mask.unsqueeze(-1)

            wt_node, wt_edge = layer(wt_node, wt_edge)
            wt_node = wt_node * wt_node_mask.unsqueeze(-1)
            wt_edge = wt_edge * wt_edge_mask.unsqueeze(-1)

            mut_node, mut_edge = layer(mut_node, mut_edge)
            mut_node = mut_node * mut_node_mask.unsqueeze(-1)
            mut_edge = mut_edge * mut_edge_mask.unsqueeze(-1)

        # Differential Embedding Layer
        diff_vec = self.diff_layer(wt_node, mut_node)

        # Mutation Impact Module
        for layer in self.mim_layers:
            wt_node, wt_edge = layer(wt_node, wt_edge, diff_vec)
            wt_node = wt_node * wt_node_mask.unsqueeze(-1)
            wt_edge = wt_edge * wt_edge_mask.unsqueeze(-1)

            mut_node, mut_edge = layer(mut_node, mut_edge, diff_vec)
            mut_node = mut_node * mut_node_mask.unsqueeze(-1)
            mut_edge = mut_edge * mut_edge_mask.unsqueeze(-1)

        # Initialize cross-graph edges
        B = mut_node.shape[0]
        L_mut = mut_node.shape[1]
        L_wt = wt_node.shape[1]
        L_binder = binder_node.shape[1]

        mut_binder_edges = torch.randn(B, L_mut, L_binder, self.d_cross_edge).to(device)
        wt_binder_edges = torch.randn(B, L_wt, L_binder, self.d_cross_edge).to(device)

        mut_binder_mask = mut_node_mask.unsqueeze(-1) * binder_node_mask.unsqueeze(1).to(device)
        wt_binder_mask = wt_node_mask.unsqueeze(-1) * binder_node_mask.unsqueeze(1).to(device)

        # Cross-Graph Attention
        for layer in self.cross_graph_att_layers:
            wt_node, binder_node, wt_binder_edges = layer(wt_node, binder_node, wt_binder_edges, diff_vec)
            wt_node = wt_node * wt_node_mask.unsqueeze(-1)
            binder_node = binder_node * binder_node_mask.unsqueeze(-1)
            wt_binder_edges = wt_binder_edges * wt_binder_mask.unsqueeze(-1)

            mut_node, binder_node, mut_binder_edges = layer(mut_node, binder_node, mut_binder_edges, diff_vec)
            mut_node = mut_node * mut_node_mask.unsqueeze(-1)
            binder_node = binder_node * binder_node_mask.unsqueeze(-1)
            mut_binder_edges = mut_binder_edges * mut_binder_mask.unsqueeze(-1)

        wt_binder_edges = torch.mean(wt_binder_edges, dim=(1,2))
        mut_binder_edges = torch.mean(mut_binder_edges, dim=(1,2))

        concat_edges = torch.concat([mut_binder_edges, wt_binder_edges], dim=-1)
        # pdb.set_trace()
        delta_pred = self.delta_linear(concat_edges)
        # delta_pred = torch.sigmoid(delta_pred)
        
        return delta_pred.squeeze(-1)


def main():
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
    print(args.sm)

    # Initialize the process group for distributed training
    # dist.init_process_group(backend='nccl')

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    test_dataloader = CustomDataModule(tokenizer, batch_size=1).testloader

    muppit = muPPIt.load_from_checkpoint(args.sm,
                                        d_node = args.d_node, 
                                        d_edge = args.d_edge, 
                                        d_cross_edge = args.d_cross_edge, 
                                        d_position = args.d_position, 
                                        num_heads = args.n_heads,
                                        num_intra_layers = args.n_intra_layers, 
                                        num_mim_layers = args.n_mim_layers, 
                                        num_cross_layers = args.n_cross_layers, 
                                        lr = args.lr)
    
    device = muppit.device
    muppit.eval()

    correct = 0
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    with torch.no_grad():
        with open("val_delta.txt", 'w') as f:
            for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                binder_tokens = batch['binder_input_ids'].to(device)
                mut_tokens = batch['mutant_input_ids'].to(device)
                wt_tokens = batch['wildtype_input_ids'].to(device)
                delta = batch['delta'].item()
                num = batch['num'].item()
                
                delta_pred = muppit(binder_tokens, wt_tokens, mut_tokens).item()

                if delta == 1:
                    positive_total += 1
                    if round(delta_pred) == delta:
                        positive_correct += 1
                
                if delta == 0:
                    negative_total += 1
                    if round(delta_pred) == delta:
                        negative_correct += 1
                
                f.write(f"{num},{delta},{round(delta_pred)},{(delta == round(delta_pred))}\n")

            f.write(f"Positive Accuracy = {positive_correct / positive_total}\n")
            f.write(f"Negative Accuracy = {negative_correct / negative_total}\n")
    
    print(f"Positive Accuracy = {positive_correct / positive_total}")
    print(f"Negative Accuracy = {negative_correct / negative_total}")
    

    




if __name__ == "__main__":
    main()