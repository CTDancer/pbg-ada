import pdb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time

from .bindevaluator_modules import *


class BindEvaluator(pl.LightningModule):
    def __init__(self, n_layers, d_model, d_hidden, n_head,
                 d_k, d_v, d_inner, dropout=0.2,
                 learning_rate=0.00001, max_epochs=15, kl_weight=1):
        super(BindEvaluator, self).__init__()

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # freeze all the esm_model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.repeated_module = RepeatedModule3(n_layers, d_model, d_hidden,
                                               n_head, d_k, d_v, d_inner, dropout=dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)

        self.final_ffn = FFN(d_model, d_inner, dropout=dropout)

        self.output_projection_prot = nn.Linear(d_model, 1)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.kl_weight = kl_weight

        self.classification_threshold = nn.Parameter(torch.tensor(0.5))  # Initial threshold
        self.historical_memory = 0.9
        self.class_weights = torch.tensor([3.000471363174231, 0.5999811490272925])  # binding_site weights, non-bidning site weights

    def forward(self, binder_tokens, target_tokens):
        peptide_sequence = self.esm_model(**binder_tokens).last_hidden_state
        protein_sequence = self.esm_model(**target_tokens).last_hidden_state

        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list, \
            seq_prot_attention_list, seq_prot_attention_list = self.repeated_module(peptide_sequence,
                                                                                    protein_sequence)

        prot_enc, final_prot_seq_attention = self.final_attention_layer(prot_enc, sequence_enc, sequence_enc)

        prot_enc = self.final_ffn(prot_enc)

        prot_enc = self.output_projection_prot(prot_enc)

        return prot_enc

    def get_probs(self, xt, target_sequence):
        '''
        Inputs:
        - xt: Shape (bsz*seq_len*vocab_size, seq_len)
        - target_sequence: Shape (bsz*seq_len*vocab_size, tgt_len)
        '''
        binder_attention_mask = torch.ones_like(xt)
        target_attention_mask = torch.ones_like(target_sequence)

        binder_attention_mask[:, 0] = binder_attention_mask[:, -1] = 0
        target_attention_mask[:, 0] = target_attention_mask[:, -1] = 0

        binder_tokens = {'input_ids': xt, 'attention_mask': binder_attention_mask.to(xt.device)}
        target_tokens = {'input_ids': target_sequence, 'attention_mask': target_attention_mask.to(target_sequence.device)}
        
        
        start = time.time()
        logits = self.forward(binder_tokens, target_tokens).squeeze(-1)
        # print(f"Time: {time.time() - start} seconds")
        
        logits[:, 0] = logits[:, -1] = -100 # float('-inf')
        probs = F.softmax(logits, dim=-1)

        return probs    # shape (bsz*seq_len*vocab_size, tgt_len)
