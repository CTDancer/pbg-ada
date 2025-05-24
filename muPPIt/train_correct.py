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
import math

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
    mut_affinities = []
    wt_affinities = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    for b in batch:
        binders.append(torch.tensor(b['binder_tokens']).squeeze(0))  # shape: 1*L1 -> L1
        mutants.append(torch.tensor(b['mutant_tokens']).squeeze(0))  # shape: 1*L2 -> L2
        wildtypes.append(torch.tensor(b['wildtype_tokens']).squeeze(0))  # shape: 1*L3 -> L3

        mut_affinities.append(torch.tensor(int(b['mut_affinity']) + 16))
        wt_affinities.append(torch.tensor(int(b['wt_affinity']) + 16))
    
    # Collate the tensors using torch's pad_sequence
    binder_input_ids = torch.nn.utils.rnn.pad_sequence(binders, batch_first=True, padding_value=tokenizer.pad_token_id)

    mutant_input_ids = torch.nn.utils.rnn.pad_sequence(mutants, batch_first=True, padding_value=tokenizer.pad_token_id)

    wildtype_input_ids = torch.nn.utils.rnn.pad_sequence(wildtypes, batch_first=True, padding_value=tokenizer.pad_token_id)

    mut_affinities = torch.stack(mut_affinities) 
    wt_affinities = torch.stack(wt_affinities) 

    binder_mask = (binder_input_ids != tokenizer.pad_token_id).int()

    # Return the collated batch
    return {
        'binder_input_ids': binder_input_ids.int(),
        'mutant_input_ids': mutant_input_ids.int(),
        'wildtype_input_ids': wildtype_input_ids.int(),
        'mut_affinity': mut_affinities,
        'wt_affinity': wt_affinities,
        'binder_mask': binder_mask
    }


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tokenizer, batch_size: int = 128):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        print(len(train_dataset))
        print(len(val_dataset))

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
        state_dict = {k: v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=False)

        for name, param in self.named_parameters():
            param.requires_grad = True
            if 'esm' in name:
                param.requires_grad = False


class MLPMsg(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(MLPMsg, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate)

        nn.init.kaiming_normal_(self.fc1.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        return x

class MLPUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(MLPUpdate, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate)

        nn.init.kaiming_normal_(self.fc.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.activation(self.fc(x))
        x = self.dropout(x)
        return x

class MLPOutput(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPOutput, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

        nn.init.kaiming_normal_(self.fc.weight, a=0.01, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class ProteinDecoder(nn.Module):
    def __init__(self, node_dim=256, edge_dim=128, hidden_dim=256, num_layers=3, num_classes=24, dropout=0.2):
        super(ProteinDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define MLPs for message passing and node updates
        self.msg_mlps = nn.ModuleList([
            MLPMsg(input_dim=2 * node_dim + edge_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.update_mlps = nn.ModuleList([
            MLPUpdate(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Define Output MLP
        self.output_mlp = MLPOutput(input_dim=hidden_dim, num_classes=num_classes)

        self.vocab_size = num_classes

    def compute_messages(self, x, E, mlp_msg):
        """
        Compute messages between nodes using their features and edge features.
        x: Node features of shape [B, L, hidden_dim]
        E: Edge features of shape [B, L, L, edge_dim]
        mlp_msg: MLP for message computation
        """
        B, L, hidden_dim = x.size()
        # Expand node features to match edge dimensions
        x_i = x.unsqueeze(2).expand(B, L, L, hidden_dim)  # Shape: [B, L, L, hidden_dim]
        x_j = x.unsqueeze(1).expand(B, L, L, hidden_dim)  # Shape: [B, L, L, hidden_dim]
        e_ij = E  # Shape: [B, L, L, edge_dim]
        # Concatenate features
        inputs = torch.cat([x_i, x_j, e_ij], dim=-1)  # Shape: [B, L, L, 2*hidden_dim + edge_dim]
        # Compute messages
        m_ij = mlp_msg(inputs)  # Shape: [B, L, L, hidden_dim]
        return m_ij


class muPPIt_decoder(pl.LightningModule):
    def __init__(self, muppit, decoder, margin, lr):
        super(muPPIt_decoder, self).__init__()

        self.muppit = muppit
        self.decoder = decoder

        self.criterion = nn.CrossEntropyLoss()
        self.margin = margin

        self.learning_rate = lr

    def forward(self, binder_tokens, wt_tokens, mut_tokens, mask):
        device = binder_tokens.device

        # Construct Graph
        binder_node, binder_edge, binder_node_mask, binder_edge_mask = self.muppit.graph(binder_tokens, self.muppit.esm, self.muppit.alphabet)
        wt_node, wt_edge, wt_node_mask, wt_edge_mask = self.muppit.graph(wt_tokens, self.muppit.esm, self.muppit.alphabet)
        mut_node, mut_edge, mut_node_mask, mut_edge_mask = self.muppit.graph(mut_tokens, self.muppit.esm, self.muppit.alphabet)

        # Intra-Graph Attention
        for layer in self.muppit.intra_graph_att_layers:
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
        diff_vec = self.muppit.diff_layer(wt_node, mut_node)

        # Mutation Impact Module
        for layer in self.muppit.mim_layers:
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

        mut_binder_edges = torch.randn(B, L_mut, L_binder, self.muppit.d_cross_edge).to(device)
        wt_binder_edges = torch.randn(B, L_wt, L_binder, self.muppit.d_cross_edge).to(device)

        mut_binder_mask = mut_node_mask.unsqueeze(-1) * binder_node_mask.unsqueeze(1).to(device)
        wt_binder_mask = wt_node_mask.unsqueeze(-1) * binder_node_mask.unsqueeze(1).to(device)

        # Cross-Graph Attention
        for layer in self.muppit.cross_graph_att_layers:
            wt_node, binder_node, wt_binder_edges = layer(wt_node, binder_node, wt_binder_edges, diff_vec)
            wt_node = wt_node * wt_node_mask.unsqueeze(-1)
            binder_node_wt = binder_node * binder_node_mask.unsqueeze(-1)
            wt_binder_edges = wt_binder_edges * wt_binder_mask.unsqueeze(-1)

            mut_node, binder_node, mut_binder_edges = layer(mut_node, binder_node, mut_binder_edges, diff_vec)
            mut_node = mut_node * mut_node_mask.unsqueeze(-1)
            binder_node_mut = binder_node * binder_node_mask.unsqueeze(-1)
            mut_binder_edges = mut_binder_edges * mut_binder_mask.unsqueeze(-1)

            binder_node = (binder_node_wt + binder_node_mut) / 2

        wt_binder_edges = torch.mean(wt_binder_edges, dim=(1,2))
        mut_binder_edges = torch.mean(mut_binder_edges, dim=(1,2))

        mut_affinity = self.muppit.affinity_linear(mut_binder_edges)
        wt_affinity = self.muppit.affinity_linear(wt_binder_edges)

        mut_binder_distance = F.cosine_similarity(torch.mean(mut_node, dim=1), torch.mean(binder_node, dim=1), dim=1)
        wt_binder_distance = F.cosine_similarity(torch.mean(wt_node, dim=1), torch.mean(binder_node, dim=1), dim=1)

        # pdb.set_trace()

        # Decoder
        x = binder_node
        E = binder_edge
        for k in range(self.decoder.num_layers):
            # Compute messages
            m_ij = self.decoder.compute_messages(x, E, self.decoder.msg_mlps[k])  # Shape: [B, L, L, hidden_dim]
            # Aggregate messages for each node
            m_i = m_ij.sum(dim=2)  # Shape: [B, L, hidden_dim]
            # Update node features
            m_i = self.decoder.update_mlps[k](m_i)  # Shape: [B, L, hidden_dim]
            x = self.decoder.layer_norms[k](x + m_i)  # Residual connection and LayerNorm

        # Predict amino acid identities
        binder_logits = self.decoder.output_mlp(x)  # Shape: [B, L, num_classes]

        # pdb.set_trace()
        
        return mut_affinity, wt_affinity, mut_binder_distance, wt_binder_distance, binder_logits

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        binder_tokens = batch['binder_input_ids'].to(self.device)   # shape: (B, L)
        mut_tokens = batch['mutant_input_ids'].to(self.device)
        wt_tokens = batch['wildtype_input_ids'].to(self.device)
        mut_affinity = batch['mut_affinity'].to(self.device)
        wt_affinity = batch['wt_affinity'].to(self.device)
        binder_mask = batch['binder_mask']    # shape: (B, L)

        mut_pred_affinity, wt_pred_affinity, mut_binder_distance, wt_binder_distance, binder_logits = self.forward(binder_tokens, wt_tokens, mut_tokens, binder_mask)

        mut_loss = self.criterion(mut_pred_affinity, mut_affinity)
        wt_loss = self.criterion(wt_pred_affinity, wt_affinity)
        triplet_loss = torch.clamp(self.margin - mut_binder_distance + wt_binder_distance, min=0.0).mean()
        
        mask_flat = binder_mask.view(-1)
        decoder_loss = self.criterion(binder_logits.view(-1, self.decoder.vocab_size), binder_tokens.view(-1).long()) * mask_flat
        decoder_loss = decoder_loss.sum() / mask_flat.sum()

        loss = mut_loss + wt_loss + triplet_loss + decoder_loss

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        binder_tokens = batch['binder_input_ids'].to(self.device)   # shape: (B, L)
        mut_tokens = batch['mutant_input_ids'].to(self.device)
        wt_tokens = batch['wildtype_input_ids'].to(self.device)
        mut_affinity = batch['mut_affinity'].to(self.device)
        wt_affinity = batch['wt_affinity'].to(self.device)
        binder_mask = batch['binder_mask']    # shape: (B, L)

        mut_pred_affinity, wt_pred_affinity, mut_binder_distance, wt_binder_distance, binder_logits = self.forward(binder_tokens, wt_tokens, mut_tokens, binder_mask)


        mut_loss = self.criterion(mut_pred_affinity, mut_affinity)
        wt_loss = self.criterion(wt_pred_affinity, wt_affinity)
        triplet_loss = torch.clamp(self.margin - mut_binder_distance + wt_binder_distance, min=0.0).mean()
        
        # pdb.set_trace()
        mask_flat = binder_mask.view(-1)
        decoder_loss = self.criterion(binder_logits.view(-1, self.decoder.vocab_size), binder_tokens.view(-1).long()) * mask_flat
        decoder_loss = decoder_loss.sum() / mask_flat.sum()

        loss = mut_loss + wt_loss + triplet_loss + decoder_loss

        mut_accuracy = (torch.argmax(mut_pred_affinity, dim=1) == mut_affinity).sum() / mut_affinity.shape[0]
        wt_accuracy = (torch.argmax(wt_pred_affinity, dim=1) == wt_affinity).sum() / wt_affinity.shape[0]
        accuracy = (mut_accuracy + wt_accuracy) / 2

        self.log('val_mut_acc', mut_accuracy.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_wt_acc', wt_accuracy.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', accuracy.item(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mut_loss', mut_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_wt_loss', wt_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_decoder_loss', decoder_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_triplet_loss', triplet_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))

        base_lr = 0.1 * self.learning_rate
        max_lr = self.learning_rate
        min_lr = 0.1 * self.learning_rate

        schedulers = CosineAnnealingWithWarmup(optimizer, warmup_steps=2031, total_steps=20310,
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


def main(args):
    dist.init_process_group(backend='nccl')

    train_dataset = load_from_disk('/home/tc415/muPPIt/dataset/train/affinity_correct')
    val_dataset = load_from_disk('/home/tc415/muPPIt/dataset/val/affinity_correct')
    # val_dataset = None
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_module = CustomDataModule(train_dataset, val_dataset, tokenizer=tokenizer, batch_size=args.batch_size)

    muppit = muPPIt(args.d_node_muppit, args.d_edge_muppit, args.d_cross_edge_muppit, args.d_position_muppit, args.n_heads_muppit,
                   args.n_intra_layers_muppit, args.n_mim_layers_muppit, args.n_cross_layers_muppit, args.lr)
    
    muppit.load_weights(args.sm_muppit)
    print(f"muPPIt Loading checkpoint from {args.sm_muppit}")

    decoder = ProteinDecoder(args.node_dim_decoder, args.edge_dim_decoder, args.hidden_dim_decoder, args.n_layers_decoder, 24, args.dropout_decoder)

    model = muPPIt_decoder(muppit=muppit, decoder=decoder, margin=args.margin, lr=args.lr)

    run_id = str(uuid.uuid4())

    logger = WandbLogger(project=f"muppit_decoder",
                        #  name="debug",
                         name=f"affinity_lr={args.lr}_gradclip={args.grad_clip}",
                         job_type='model-training',
                         id=run_id)

    print(f"Saving to {args.output_file}")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=args.output_file,
        filename='model-{epoch:02d}-{val_acc:.2f}',
        # filename='muppit',
        save_top_k=-1,
        mode='max',
        # every_n_train_steps=1000,
        # save_on_train_epoch_end=False
    )

    # early_stopping_callback = EarlyStopping(
    #     monitor='val_acc',
    #     patience=10,
    #     verbose=True,
    #     mode='max'
    # )

    accumulator = GradientAccumulationScheduler(scheduling={0: 4})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        precision='bf16',
        logger=logger,
        devices=[0,1,2],
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.grad_clip,
        # val_check_interval=100,
    )

    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-o", dest="output_file", help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-grad_clip", type=float, default=0.5)
    parser.add_argument("-margin", type=float, default=0.5)
    parser.add_argument("-max_epochs", type=int, default=30)

    parser.add_argument("-d_node_muppit", type=int, default=1024, help="Node Representation Dimension")
    parser.add_argument("-d_edge_muppit", type=int, default=512, help="Intra-Graph Edge Representation Dimension")
    parser.add_argument("-d_cross_edge_muppit", type=int, default=512, help="Cross-Graph Edge Representation Dimension")
    parser.add_argument("-d_position_muppit", type=int, default=8, help="Positional Embedding Dimension")
    parser.add_argument("-n_heads_muppit", type=int, default=8)
    parser.add_argument("-n_intra_layers_muppit", type=int, default=1)
    parser.add_argument("-n_mim_layers_muppit", type=int, default=1)
    parser.add_argument("-n_cross_layers_muppit", type=int, default=1)
    parser.add_argument("-sm_muppit", default=None, help="Pre-trained muPPIt", type=str)

    parser.add_argument("-node_dim_decoder", type=int, default=256, help="Node representation dimension")
    parser.add_argument("-edge_dim_decoder", type=int, default=128, help="Edge representation dimension")
    parser.add_argument("-hidden_dim_decoder", type=int, default=256, help="Decoder hidden dimension")
    parser.add_argument("-n_layers_decoder", type=int, default=4, help="Decoder layer counts")
    parser.add_argument("-dropout_decoder", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
