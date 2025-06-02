import pdb
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
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
import numpy as np
import torch.distributed as dist
from models import *
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, EsmForMaskedLM
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Adam, AdamW
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import gc
from train_bindevaluator import PeptideModel

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def differentiable_argmax(one_hot):
    indices = torch.arange(one_hot.size(-1))
    return torch.sum(one_hot * indices, dim=-1)


class PepMLM(nn.Module):
    def __init__(self):
        super(PepMLM, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
        self.pepmlm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M")

        # Unfreeze the PepMLM weights
        for param in self.pepmlm.parameters():
            param.requires_grad = True

    def forward(self, pep_prot_tokens):
        logits = self.pepmlm(**pep_prot_tokens).logits
        mask_token_indices = (pep_prot_tokens["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]
        one_hot = F.gumbel_softmax(logits_at_masks, tau=1, hard=True)
        peptide_tokens = differentiable_argmax(one_hot)

        return peptide_tokens


class BindEvaluator(nn.Module):
    def __init__(self, args):
        self.bindevaluator = PeptideModel.load_from_checkpoint(args.sm,
                                                               n_layers=args.n_layers,
                                                               d_model=args.d_model,
                                                               d_hidden=args.d_hidden,
                                                               n_head=args.n_head,
                                                               d_k=64,
                                                               d_v=128,
                                                               d_inner=args.d_inner)
        # Freeze the BindEvaluator weights
        for param in self.bindevaluator.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        self.class_weights = torch.tensor([3.000471363174231, 0.5999811490272925])

    def forward(self,binder_tokens, target_tokens):
        outputs_nodes = self.bindevaluator(binder_tokens, target_tokens)
        return outputs_nodes


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running on rank {rank}.")

    device_id = randk % torch.cuda.device_count()
    pepmlm = PepMLM().to(device_id)
    bind_evaluator = BindEvaluator(args).to(device_id)
    ddp_pepmlm = DDP(pepmlm, device_ids=[device_id])
    ddp_bind_evaluator = DDP(bind_evaluator, device_ids=[device_id])

    # Load data
    train_dataloader, val_dataloader = get_dataloader(args)

    for epoch in range(args.max_epoch):
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            masked_peptide_tokens = batch['masked_peptide']     # B*l
            protein_tokens = batch['protein']       # B*L
            peptide_tokens = batch['peptide']       # B*l

            pep_protein_tokens = masked_peptide_tokens + protein_tokens
            predicted_peptide_tokens = ddp_pepmlm(pep_protein_tokens)   # B*l
            int_predicted_peptide_tokens = predicted_peptide_tokens.clone().detach().to(torch.int64)
            output_nodes = ddp_bind_evaluator(predicted_peptide_tokens, protein_tokens)



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-o", dest="output_file", help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-d", dest="dataset", required=False, type=str, default="pepnn",
                        help="Which dataset to train on, pepnn, pepbind, or interpep")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("-sm", default=None, help="File containing initial params", type=str)
    parser.add_argument("--max_epochs", type=int, default=15, help="Max number of epochs to train")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=1)
    args = parser.parse_args()
    main(args)
