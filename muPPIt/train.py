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
    mutant_binders = []
    wildtype_binders = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    for b in batch:
        binders.append(torch.tensor(b['binder_tokens']).squeeze(0))  # shape: 1*L1 -> L1
        mutants.append(torch.tensor(b['mutant_tokens']).squeeze(0))  # shape: 1*L2 -> L2
        wildtypes.append(torch.tensor(b['wildtype_tokens']).squeeze(0))  # shape: 1*L3 -> L3
        mutant_binders.append(torch.tensor(b['mutant_binder']).squeeze(0,1))   # shape: 1*1*L2*L1 -> L2*L1
        wildtype_binders.append(torch.tensor(b['wildtype_binder']).squeeze(0,1))    # shape: 1*1*L3*L1 -> L3*L1

    # Collate the tensors using torch's pad_sequence
    binder_input_ids = torch.nn.utils.rnn.pad_sequence(binders, batch_first=True, padding_value=tokenizer.pad_token_id)

    mutant_input_ids = torch.nn.utils.rnn.pad_sequence(mutants, batch_first=True, padding_value=tokenizer.pad_token_id)

    wildtype_input_ids = torch.nn.utils.rnn.pad_sequence(wildtypes, batch_first=True, padding_value=tokenizer.pad_token_id)

    def pad_tensor(tensor, target_size):
        """Pads a tensor to the target size with zeros."""
        rows, cols = tensor.size()
        pad_rows = target_size[0] - rows
        pad_cols = target_size[1] - cols
        # Padding: (left, right, top, bottom)
        return F.pad(tensor, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

    max_row = max([tensor.shape[0] for tensor in mutant_binders])
    max_col = max([tensor.shape[1] for tensor in mutant_binders])
    padded_tensors = [pad_tensor(tensor, (max_row, max_col)) for tensor in mutant_binders]
    batch_mutant_binders = torch.stack(padded_tensors)

    max_row = max(tensor.shape[0] for tensor in wildtype_binders)
    max_col = max(tensor.shape[1] for tensor in wildtype_binders)
    padded_tensors = [pad_tensor(tensor, (max_row, max_col)) for tensor in wildtype_binders]
    batch_wildtype_binders = torch.stack(padded_tensors)

    # Return the collated batch
    return {
        'binder_input_ids': binder_input_ids.int(),
        'mutant_input_ids': mutant_input_ids.int(),
        'wildtype_input_ids': wildtype_input_ids.int(),
        'mutant_binders': batch_mutant_binders,
        'wildtype_binders': batch_wildtype_binders
    }


class LengthAwareDistributedSampler(DistributedSampler):
    def __init__(self, dataset, key, batch_size, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.dataset = dataset
        self.key = key
        self.batch_size = batch_size

        # Sort indices by the length of the mutant sequence
        self.indices = sorted(range(len(self.dataset)), key=lambda i: len(self.dataset[i][key]))

    def __iter__(self):
        # Divide indices among replicas
        indices = self.indices[self.rank::self.num_replicas]
        
        if self.shuffle:
            torch.manual_seed(self.epoch)
            indices = torch.randperm(len(indices)).tolist()

        # Yield indices in batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i+self.batch_size]

    def __len__(self):
        return len(self.indices) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tokenizer, batch_size: int = 128):
        super().__init__()
        self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def train_dataloader(self):
        # batch_sampler = LengthAwareDistributedSampler(self.train_dataset, 'mutant_tokens', self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=8, pin_memory=True)

    # def val_dataloader(self):
    #     batch_sampler = LengthAwareDistributedSampler(self.val_dataset, 'mutant_tokens', self.batch_size)
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=8,
    #                       batch_sampler=batch_sampler, pin_memory=True)

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

class muPPIt(pl.LightningModule):
    def __init__(self, d_node, d_edge, d_cross_edge, d_position, num_heads, 
                num_intra_layers, num_mim_layers, num_cross_layers, lr, delta=1.0):
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

        self.cross_graph_edge_mapping = nn.Linear(1, d_cross_edge)
        self.mapping = nn.Linear(d_cross_edge, 1)

        self.learning_rate = lr
        self.delta = delta

    def forward(self, binder_tokens, wt_tokens, mut_tokens, wt_binder_edges, mut_binder_edges):
        # Construct Graph
        # print("Graph")
        binder_node, binder_edge, binder_node_mask, binder_edge_mask = self.graph(binder_tokens, self.esm, self.alphabet)
        wt_node, wt_edge, wt_node_mask, wt_edge_mask = self.graph(wt_tokens, self.esm, self.alphabet)
        mut_node, mut_edge, mut_node_mask, mut_edge_mask = self.graph(mut_tokens, self.esm, self.alphabet)

        # Intra-Graph Attention
        # print("Intra Graph")
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
        # print("Diff")
        diff_vec = self.diff_layer(wt_node, mut_node)

        # Mutation Impact Module
        # print("MIM")
        for layer in self.mim_layers:
            wt_node, wt_edge = layer(wt_node, wt_edge, diff_vec)
            wt_node = wt_node * wt_node_mask.unsqueeze(-1)
            wt_edge = wt_edge * wt_edge_mask.unsqueeze(-1)

            mut_node, mut_edge = layer(mut_node, mut_edge, diff_vec)
            mut_node = mut_node * mut_node_mask.unsqueeze(-1)
            mut_edge = mut_edge * mut_edge_mask.unsqueeze(-1)

        # Mapping cross-graph edge representation
        wt_binder_mask = (wt_binder_edges != 0)
        mut_binder_mask = (mut_binder_edges != 0)

        wt_binder_edges = self.cross_graph_edge_mapping(wt_binder_edges.unsqueeze(-1))
        mut_binder_edges = self.cross_graph_edge_mapping(mut_binder_edges.unsqueeze(-1))

        # Cross-Graph Attention
        # print("Cross")
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

        wt_pred = torch.sigmoid(self.mapping(wt_binder_edges))
        mut_pred = torch.sigmoid(self.mapping(mut_binder_edges))

        return wt_pred, mut_pred

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        binder_tokens = batch['binder_input_ids'].to(self.device)
        mut_tokens = batch['mutant_input_ids'].to(self.device)
        wt_tokens = batch['wildtype_input_ids'].to(self.device)
        mutant_binders = batch['mutant_binders'].to(self.device)
        wildtype_binders = batch['wildtype_binders'].to(self.device)

        wt_pred, mut_pred = self.forward(binder_tokens, wt_tokens, mut_tokens, mutant_binders, wildtype_binders)

        wt_loss = F.smooth_l1_loss(wt_pred, torch.ones_like(wt_pred) * 0.5, beta=self.delta, reduction='none')
        mut_loss = F.smooth_l1_loss(mut_pred, torch.ones_like(mut_pred) * 0.5, beta=self.delta, reduction='none')
        loss = (wt_loss + mut_loss).mean()

        # pdb.set_trace()
        self.log('train_wt_loss', wt_loss.mean().item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_mut_loss', mut_loss.mean().item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        binder_tokens = batch['binder_input_ids'].to(self.device)
        mut_tokens = batch['mutant_input_ids'].to(self.device)
        wt_tokens = batch['wildtype_input_ids'].to(self.device)
        mutant_binders = batch['mutant_binders'].to(self.device)
        wildtype_binders = batch['wildtype_binders'].to(self.device)

        wt_pred, mut_pred = self.forward(binder_tokens, wt_tokens, mut_tokens, mutant_binders, wildtype_binders)

        wt_loss = F.smooth_l1_loss(wt_pred, torch.ones_like(wt_pred) * 0.5, beta=self.delta, reduction='none')
        mut_loss = F.smooth_l1_loss(mut_pred, torch.ones_like(mut_pred) * 0.5, beta=self.delta, reduction='none')
        loss = (wt_loss + mut_loss) / wt_loss.shape[0]

        self.log('val_wt_loss', wt_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mut_loss', mut_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))

        base_lr = 1e-5
        max_lr = self.learning_rate
        min_lr = 0.1 * self.learning_rate

        schedulers = CosineAnnealingWithWarmup(optimizer, warmup_steps=600, total_steps=15390,
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
    parser.add_argument("-delta", type=float, default=1)
    args = parser.parse_args()

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')

    train_dataset = load_from_disk('/home/tc415/muPPIt/dataset/train/ppiref_0_3')
    val_dataset = load_from_disk('/home/tc415/muPPIt/dataset/train/ppiref_0_3')
    # val_dataset = None
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_module = CustomDataModule(train_dataset, val_dataset, tokenizer=tokenizer, batch_size=args.batch_size)

    model = muPPIt(args.d_node, args.d_edge, args.d_cross_edge, args.d_position, args.n_heads,
                   args.n_intra_layers, args.n_mim_layers, args.n_cross_layers, args.lr, args.delta)
    if args.sm:
        model = muPPIt.load_from_checkpoint(args.sm,args.d_node, args.d_edge, args.d_cross_edge, args.d_position, args.n_heads,
                                            args.n_intra_layers, args.n_mim_layers, args.n_cross_layers, args.lr, args.delta)
    else:   
        print("Train from scratch!")

    run_id = str(uuid.uuid4())

    logger = WandbLogger(project=f"muppit",
                         name="debug",
                        #  name=f"lr={args.lr}_dnode={args.d_node}_dedge={args.d_edge}_dcross={args.d_cross_edge}_dposition={args.d_position}",
                         job_type='model-training',
                         id=run_id)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.output_file,
        filename='model-{epoch:02d}-{val_mcc:.2f}',
        save_top_k=-1,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_mcc',
        patience=5,
        verbose=True,
        mode='max'
    )

    accumulator = GradientAccumulationScheduler(scheduling={0: 8, 3: 4, 20: 2})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        precision='bf16',
        # logger=logger,
        devices=[0,1,2,3,4,5,6,7],
        callbacks=[checkpoint_callback, accumulator, early_stopping_callback],
        gradient_clip_val=args.grad_clip,
        # num_sanity_val_steps=0
    )

    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)


if __name__ == "__main__":
    main()
