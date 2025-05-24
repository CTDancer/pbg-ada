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

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

def collate_fn(batch):
    # Unpack the batch
    binders = []
    targets = []
    
    global tokenizer

    for b in batch:
        binder = torch.tensor(b['binder_input_ids']['input_ids'][1:-1])
        target = torch.tensor(b['target_input_ids']['input_ids'][1:-1])

        if binder.dim() == 0 or binder.numel() == 0 or target.dim() == 0 or target.numel() == 0:
            continue
        binders.append(binder)  # shape: 1*L1 -> L1
        targets.append(target)  # shape: 1*L2 -> L2

    # Collate the tensors using torch's pad_sequence
    try:
        binder_input_ids = torch.nn.utils.rnn.pad_sequence(binders, batch_first=True, padding_value=tokenizer.pad_token_id)

        target_input_ids = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    except:
        pdb.set_trace()

    # Return the collated batch
    return {
        'binder_input_ids': binder_input_ids.long(),
        'target_input_ids': target_input_ids.long(),
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=8, 
                          pin_memory=True)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            pass


def RoPE(x, seq_dim=0):
    """
    Applies Rotary Positional Encoding to the input embeddings.
    :param x: Input tensor (seq_len, batch_size, embed_dim)
    :param seq_dim: The sequence dimension, usually 0 (first dimension in (seq_len, batch_size, embed_dim))
    :return: Tensor with RoPE applied (seq_len, batch_size, embed_dim)
    """
    seq_len = x.shape[seq_dim]
    d_model = x.shape[-1]
    
    # Create the positions and the sine-cosine rotational matrices
    theta = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    theta = 10000 ** (-theta)  # scaling factor for RoPE
    seq_idx = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    
    # Compute sine and cosine embedding for each position
    sin_emb = torch.sin(seq_idx * theta)
    cos_emb = torch.cos(seq_idx * theta)
    
    sin_emb = sin_emb.unsqueeze(1)  # [seq_len, 1, embed_dim//2]
    cos_emb = cos_emb.unsqueeze(1)  # [seq_len, 1, embed_dim//2]
    
    x1, x2 = x[..., ::2], x[..., 1::2]  # Split embedding into even and odd indices

    cos_emb = cos_emb.to(x1.device)
    sin_emb = sin_emb.to(x1.device)
    
    # Apply rotary transformation
    x_rotated = torch.cat([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)
    return x_rotated

def probability_decay(step, k):
    return k / (k + torch.exp(torch.tensor(step / k, dtype=torch.float)))

class BinderGenerator(pl.LightningModule):
    def __init__(self, vocab_size=24, embed_dim=1280, num_heads=4, num_layers=4, dropout=0.1, lr=1e-4, k=100, passes=5):
        super(BinderGenerator, self).__init__()
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.esm.parameters():
            param.requires_grad = False

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim // 2), 
            torch.nn.SiLU(),
            nn.Dropout(dropout),
            torch.nn.Linear(embed_dim // 2, vocab_size)
        )

        for layer in self.fc_out:
            if isinstance(layer, nn.Linear): 
                nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.alphabet.padding_idx)
        # self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
        self.learning_rate = lr
        self.k = k
        self.passes = passes
    
    def standard_forward(self, input_tokens, output_tokens):
        with torch.no_grad():
            input_pad_mask = (input_tokens != self.alphabet.padding_idx).int()
            input_embed = self.esm(input_tokens, repr_layers=[33], return_contacts=True)["representations"][33] * input_pad_mask.unsqueeze(-1)

            output_pad_mask = (output_tokens != self.alphabet.padding_idx).int()
            output_embed = self.esm(output_tokens, repr_layers=[33], return_contacts=True)["representations"][33] * output_pad_mask.unsqueeze(-1)

        input_embed = input_embed.transpose(0, 1)
        output_embed = output_embed.transpose(0, 1)

        input_embed = RoPE(input_embed)  # [src_len, batch_size, embed_dim]
        output_embed = RoPE(output_embed)  # [tgt_len, batch_size, embed_dim]

        # Define padding masks
        src_key_padding_mask = (input_tokens == self.alphabet.padding_idx)  # [batch_size, src_len]
        tgt_key_padding_mask = (output_tokens == self.alphabet.padding_idx)  # [batch_size, tgt_len]

        # Causal mask for autoregressive decoding (target mask)
        tgt_len = output_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(output_tokens.device)  # [tgt_len, tgt_len]

        # Forward through the transformer
        prediction = self.transformer(
            input_embed, output_embed, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,  # memory_key_padding_mask is the same as src_key_padding_mask
            tgt_is_causal=True,
        )  # [tgt_len, batch_size, embed_dim]

        return self.fc_out(prediction).transpose(0, 1)  # [batch_size, tgt_len, vocab_size]

    def forward(self, input_tokens, output_tokens):
        device = input_tokens.device
        batch_size, tgt_len = output_tokens.shape  # output_tokens is of shape [batch_size, tgt_len]
        final_prediction = None

        # Initialize generated tokens with the ground-truth
        true_tokens = output_tokens.clone()
        mixed_tokens = true_tokens.clone()

        # First pass: Generate predictions using ground-truth history
        next_token_logits = self.standard_forward(input_tokens, mixed_tokens)  # Shape: [batch_size, tgt_len, vocab_size]

        # Step 2: Mix tokens
        epsilon = probability_decay(0, self.k)
        rand_values = torch.rand(batch_size, tgt_len)
        mask = (rand_values > epsilon).long().to(device)  # Shape: [batch_size, tgt_len]

        next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: [batch_size, tgt_len]

        mixed_tokens = mask * next_token + (1 - mask) * mixed_tokens  # [batch_size, tgt_len]

        # Step 3: Additional passes for resampling
        for p in range(self.passes):
            # Generate predictions for the current mixed tokens
            next_token_logits = self.standard_forward(input_tokens, mixed_tokens)  # Shape: [batch_size, tgt_len, vocab_size]

            # Generate a new random mask for the current pass
            epsilon = probability_decay(p+1, self.k)
            rand_values = torch.rand(batch_size, tgt_len)
            mask = (rand_values > epsilon).long().to(device)  # Shape: [batch_size, tgt_len]

            next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: [batch_size, tgt_len]

            mixed_tokens = mask * next_token + (1 - mask) * true_tokens  # [batch_size, tgt_len]

            if p == self.passes - 1:
                final_prediction = next_token_logits

        return final_prediction 

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        state_dict = checkpoint['state_dict']

        self.load_state_dict(state_dict, strict=True)

        for name, param in self.named_parameters():
            param.requires_grad = False


def main(args):
    model = BinderGenerator(vocab_size=24, embed_dim=1280, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout, lr=args.lr, k=args.k, passes=args.passes)
    model.load_weights(args.sm)

    device = 'cuda:0'
    model = model.to(device)
    model.eval()

    _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    global tokenizer
    target_tokens = torch.tensor(tokenizer(args.target)['input_ids']).unsqueeze(0).to(device)
    
    binder_tokens = torch.full((1, args.max_length), alphabet.padding_idx, dtype=torch.long).to(device) 
    binder_tokens[:, 0] = 0

    with torch.no_grad():
        for t in range(1, args.max_length):
            output = model(target_tokens, binder_tokens[:, :t])
            next_token_logits = output[:, t - 1, :] 
            # pdb.set_trace()
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Update the binder tokens
            binder_tokens[:, t] = next_token

    predicted_sequence = tokenizer.decode(binder_tokens.squeeze(0).cpu().numpy().tolist())  # Convert to a list
    print(predicted_sequence)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-target", type=str, required=True)
    parser.add_argument("-max_length", type=int, required=True)
    parser.add_argument("-sm", help="Saved model", required=True, type=str)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-num_heads", type=int, default=4)
    parser.add_argument("-num_layers", type=int, default=4)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-k", type=int, default=100)
    parser.add_argument("-passes", type=int, default=5)
    args = parser.parse_args()

    main(args)
