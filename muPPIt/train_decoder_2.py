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

from models.graph import ProteinGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



def collate_fn(batch):
    # Unpack the batch
    seqs = []
    nodes = []
    edges = []

    for b in batch:
        seqs.append(torch.tensor(b['Sequence']).squeeze(0))
        nodes.append(torch.tensor(b['Graph']).squeeze(0))
        edges.append(torch.tensor(b['Edge']).squeeze(0))

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    seq_input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
    seq_mask = (seq_input_ids != tokenizer.pad_token_id).int()

    node_representations = torch.nn.utils.rnn.pad_sequence(nodes, batch_first=True, padding_value=0)

    max_L = max([edge.shape[0] for edge in edges])
    edge_representations = []
    for edge in edges:
        pad_size = max_L - edge.shape[0]
        padded_edge = F.pad(edge, (0, pad_size, 0, pad_size)) 
        edge_representations.append(padded_edge)
    
    edge_representations = torch.stack(edge_representations)

    return {
        'seqs': seq_input_ids,
        'seq_mask': seq_mask,
        'node_representations': node_representations,
        'edge_representations': edge_representations,
    }


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tokenizer, batch_size: int = 128):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def train_dataloader(self):
        # batch_sampler = LengthAwareDistributedSampler(self.train_dataset, 'mutant_tokens', self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        # batch_sampler = LengthAwareDistributedSampler(self.val_dataset, 'mutant_tokens', self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=8, 
                          pin_memory=True)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            pass


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
        print(f"SELF BASE LRS = {self.base_lrs}")

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase from base_lr to max_lr
            return [self.base_lr + (self.max_lr - self.base_lr) * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]

        # Cosine annealing phase from max_lr to min_lr
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return [decayed_lr for base_lr in self.base_lrs]


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


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        seqs = batch['seqs']    # shape: (B, L)
        mask = batch['seq_mask']    # shape: (B, L)
        node_representations = batch['node_representations']
        edge_representations = batch['edge_representations']

        # pdb.set_trace()

        logits = self.forward(node_representations, edge_representations, mask) # shape: (batch_size, L, vocab_size)
        
        mask_flat = mask.view(-1)
        loss = self.criterion(logits.view(-1, self.vocab_size), seqs.view(-1)) * mask_flat
        loss = loss.sum() / mask_flat.sum()

        # pdb.set_trace()
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs = batch['seqs']    # shape: (B, L)
        mask = batch['seq_mask']    # shape: (B, L)
        node_representations = batch['node_representations']
        edge_representations = batch['edge_representations']
        
        # pdb.set_trace()

        logits = self.forward(node_representations, edge_representations, mask) # shape: (batch_size, L, vocab_size)

        # pdb.set_trace()
        
        mask_flat = mask.view(-1)
        loss = self.criterion(logits.view(-1, self.vocab_size), seqs.view(-1)) * mask_flat
        loss = loss.sum() / mask_flat.sum()

        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))

        base_lr = 1e-5
        max_lr = self.learning_rate
        min_lr = 0.1 * self.learning_rate

        schedulers = CosineAnnealingWithWarmup(optimizer, warmup_steps=6382, total_steps=63816,
                                              base_lr=base_lr, max_lr=max_lr, min_lr=min_lr)

        lr_schedulers = {
            "scheduler": schedulers,
            "name": 'learning_rate_logs',
            "interval": 'step',  # The scheduler updates the learning rate at every step (not epoch)
            'frequency': 1  # The scheduler updates the learning rate after every batch
        }
        return [optimizer], [lr_schedulers]

    def on_training_epoch_end(self, outputs):
        gc.collect()
        torch.cuda.empty_cache()
        super().training_epoch_end(outputs)

    # def on_validation_epoch_end(self, outputs):
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     super().validation_epoch_end(outputs)


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", dest="output_file", help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-d_node", type=int, default=1296, help="Node representation dimension")
    parser.add_argument("-n_layers", type=int, default=6, help="num decoder layers")
    parser.add_argument("-n_heads", type=int, default=8, help="num attention heads")
    parser.add_argument("-d_ff", type=int, default=2048, help="Feed forward layer dimension")
    parser.add_argument("-sm", default=None, help="File containing initial params", type=str)
    parser.add_argument("-max_epochs", type=int, default=15, help="Max number of epochs to train")
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("-grad_clip", type=float, default=0.5)
    args = parser.parse_args()

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')

    train_dataset = load_from_disk('/home/tc415/muPPIt/dataset/train/decoder_2')
    val_dataset = load_from_disk('/home/tc415/muPPIt/dataset/val/decoder_2')

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_module = CustomDataModule(train_dataset, val_dataset, tokenizer=tokenizer, batch_size=args.batch_size)

    model = ProteinDecoder(args.d_node, args.n_heads, args.n_layers, args.d_ff, 24, args.dropout, args.lr)
    if args.sm:
        model = ProteinDecoder.load_from_checkpoint(args.sm,
                                            d_model = args.d_node, 
                                            nhead = args.n_heads, 
                                            num_layers = args.n_layers, 
                                            dim_feedforward = args.d_ff,
                                            vocab_size = 24,
                                            dropout = args.dropout,
                                            lr = args.lr)
    else:   
        print("Train from scratch!")

    run_id = str(uuid.uuid4())

    logger = WandbLogger(project=f"muppit_decoder",
                        #  name="debug",
                         name=f"lr={args.lr}_layers={args.n_layers}_heads={args.n_heads}_dff={args.d_ff}_dropout={args.dropout}",
                         job_type='model-training',
                         id=run_id)

    print(f"Saving to {args.output_file}")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.output_file,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        # every_n_train_steps=2000,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    accumulator = GradientAccumulationScheduler(scheduling={0: 8, 3: 4, 20: 2})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        precision='bf16',
        logger=logger,
        devices=[0,1,2,3],
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=args.grad_clip,
        # val_check_interval=100,
    )

    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)


if __name__ == "__main__":
    main()
