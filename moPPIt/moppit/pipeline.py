import pandas as pd
from Bio import SeqIO
import io
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import os
from argparse import ArgumentParser
# from predict_bindevaluator import *
import random
from models import * 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

AA = 'ARNDCEQGHILKMFPSTWYV'

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Load the model and tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
model = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)

def calculate_score(target_sequence, binder_sequence, model, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    anchor_tokens = tokenizer(target_sequence, return_tensors='pt', padding=True, truncation=True, max_length=40000)
    positive_tokens = tokenizer(binder_sequence, return_tensors='pt', padding=True, truncation=True, max_length=40000)

    anchor_tokens['attention_mask'][0][0] = 0
    anchor_tokens['attention_mask'][0][-1] = 0
    positive_tokens['attention_mask'][0][0] = 0
    positive_tokens['attention_mask'][0][-1] = 0

    target_tokens = {'input_ids': anchor_tokens["input_ids"].to(device),
                     'attention_mask': anchor_tokens["attention_mask"].to(device)}
    binder_tokens = {'input_ids': positive_tokens['input_ids'].to(device),
                     'attention_mask': positive_tokens['attention_mask'].to(device)}

    model.eval()

    # pdb.set_trace()

    prediction = model(binder_tokens, target_tokens).squeeze(-1)[0][1:-1]
    prediction = torch.sigmoid(prediction)

    return prediction, model.classification_threshold

class PeptideModel(pl.LightningModule):
    def __init__(self, n_layers, d_model, d_hidden, n_head,
                 d_k, d_v, d_inner, dropout=0.2,
                 learning_rate=0.00001, max_epochs=15, kl_weight=1):
        super(PeptideModel, self).__init__()

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

def generate_random_seq(length):
    global AA
    s = ''
    for i in range(length):
        s += random.choice(AA)
    return s


def cal_score(binder_seq, protein_seq, motif, model, args):
    prediction, threshold = calculate_score(protein_seq, binder_seq, model, args)
    binding_score = 0
    score = 0
    for pos in motif:
        if prediction[pos] < 0.5:
            score += 10

    for i in range(len(prediction)):
        if i not in motif and prediction[i] >= 0.5:
            score += 0.5

    return score


def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    """
    For alternative computation of PPL (in batch/matrix format), please check our GitHub repo:
    https://github.com/programmablebio/pepmlm/blob/main/scripts/generation.py
    """
    sequence = protein_seq + binder_seq
    tensor_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)
    total_loss = 0

    # Loop through each token in the binder sequence
    for i in range(-len(binder_seq)-1, -1):
        # Create a copy of the original tensor
        masked_input = tensor_input.clone()

        # Mask one token at a time
        masked_input[0, i] = tokenizer.mask_token_id
        # Create labels
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        # Get model prediction and loss
        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()

    # Calculate the average loss
    avg_loss = total_loss / len(binder_seq)

    # Calculate pseudo perplexity
    pseudo_perplexity = np.exp(avg_loss)
    return pseudo_perplexity


def generate_peptide_for_single_sequence(protein_seq, peptide_length = 15, top_k = 3, num_binders = 4):

    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    binders_with_ppl = []

    for _ in range(num_binders):
        # Generate binder
        masked_peptide = '<mask>' * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        # Apply top-k sampling
        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')

        # Compute PPL for the generated binder
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)

        # Add the generated binder and its PPL to the results list
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):  # Single sequence
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])

    elif isinstance(input_seqs, list):  # List of sequences
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])

class Binder(object):
    def __init__(self, binder_seq, model, pepmlm, tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = args.motif
        self.model = model
        self.args = args
        self.pepmlm = pepmlm
        self.tokenizer = tokenizer
        self.score = cal_score(binder_seq, self.protein_seq, self.motif, self.model, self.args)
        self.ppl = compute_pseudo_perplexity(self.pepmlm, self.tokenizer, self.protein_seq, self.binder_seq)

    def mutated_aa(self):
        global AA
        mutated_aa = random.choice(AA)
        return mutated_aa

    def mutate_seq(self, binder_seq):
        position = random.randint(0, len(binder_seq) - 1)
        mutated_seq = binder_seq[:position] + self.mutated_aa() + binder_seq[position + 1:]
        return mutated_seq

    def mate(self, par2):
        assert len(par2.binder_seq) == len(self.binder_seq)

        child = ''
        for i in range(len(par2.binder_seq)):
            a1 = self.binder_seq[i]
            a2 = par2.binder_seq[i]

            prob = random.random()
            if prob < 0.45:
                child += a1
            elif prob < 0.90:
                child += a2
            else:
                child += self.mutated_aa()

        return Binder(child, self.model, self.pepmlm, self.tokenizer, args)

    def simulated_annealing(self, T_max, T_min, alpha):
        def accept(delta_score, T):
            delta_score = delta_score.item()
            if delta_score < 0:
                return True
            else:
                r = random.random()
                if r < np.exp(-delta_score / T):
                    return True
                else:
                    return False

        T = T_max
        score = self.score
        binder_seq = self.binder_seq

        print(f"Initial Binder Sequence: {binder_seq}")

        while T > T_min and score > 1:
            mutated_seq = self.mutate_seq(binder_seq)

            new_score = cal_score(mutated_seq, self.protein_seq, self.motif, self.model, self.args)

            delta_score = new_score - score

            if accept(delta_score, T):
                binder_seq = mutated_seq
                score = new_score
                print(f"New Sequence: {binder_seq}, Score: {score}")

            T = T * alpha
            print(f"New T: {T}")

        return binder_seq, score

def find_contigs(lst):
    if not lst:
        return []
    
    contigs = []
    start = lst[0]
    end = lst[0]

    for i in range(1, len(lst)):
        if lst[i] == end + 1:
            end = lst[i]
        else:
            contigs.append((start, end))
            start = lst[i]
            end = lst[i]
    contigs.append((start, end))  # append the last contig

    contigs = [contig for contig in contigs if contig[0] != contig[1]]

    return contigs


def remove_redundant_elements(lst):
    seen = set()
    new_lst = []
    for d in lst:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_lst.append(d)
    return new_lst


def main(args):
    random.seed(args.seed)
    if args.csv is None:
        entries = [args.entry]
        seqs = [args.seq]
    else:
        df = pd.read_csv(args.csv)
        entries = df['Entry']
        seqs = df['Sequence']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PeptideModel.load_from_checkpoint(args.sm,
                                              n_layers=args.n_layers,
                                              d_model=args.d_model,
                                              d_hidden=args.d_hidden,
                                              n_head=args.n_head,
                                              d_k=64,
                                              d_v=128,
                                              d_inner=64).to(device)

    tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
    pepmlm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)

    print(len(entries))
    for i in range(len(entries)):
        # print(f"################ Processing Entry {entries[i]} ################")
        # protein_seq = seqs[i]

        # if len(protein_seq) >= 600:
        #     print(f"Protein Length for {entries[i]} is longer than 600")
        #     continue

        # peptide_length = 8
        # peptide_df = generate_peptide(protein_seq, peptide_length, args.top_k, args.num_binders)
        # peptide_df = peptide_df.drop_duplicates(subset='Binder')
        # peptide_df = peptide_df.sort_values(by='Pseudo Perplexity')
        # print(peptide_df[:10])

        # candidate_seqs = [peptide_df.iloc[0]['Binder'], 
        #                 #   peptide_df.iloc[2]['Binder'],
        #                   peptide_df.iloc[2]['Binder']]
        # binding_sites = []
        # candidate_binding_site = []

        # for binder_seq in candidate_seqs:
        #     prediction, _ = calculate_score(protein_seq, binder_seq, model, args)
        #     binding_site = []
        #     for i in range(len(prediction)):
        #         if prediction[i] >= 0.5:
        #             binding_site.append(i)

        #     binding_sites.append(binding_site)
        #     print(binding_site)

        # for i in binding_sites[0]:
        #     if i in binding_sites[1]:
        #         candidate_binding_site.append(i)

        # # pdb.set_trace()
        # # candidate_binding_site = candidate_binding_site.sort()
        # print(f"Original Candidate Binding Site = {candidate_binding_site}")
        # if len(candidate_binding_site) == 0:
        #     continue
        # # pdb.set_trace()
        # contigs = find_contigs(candidate_binding_site)
        # if len(contigs) > 2:
        #     selected_contigs = random.sample(contigs, 2)
        #     expanded_selected_contigs = [list(range(start, end + 1)) for start, end in selected_contigs]
        #     candidate_binding_site = [num for sublist in expanded_selected_contigs for num in sublist]

        # print(f"Candidate Binding Site = {candidate_binding_site}")
    
        args.protein_seq = seqs[i]
        protein_seq = args.protein_seq
        candidate_binding_site = list(range(210,225))
        print(candidate_binding_site)
        args.motif = candidate_binding_site
        # candidate_binders = []

        for peptide_length in range(6,26):
            print(f"********* Starting peptide length = {peptide_length} *********")
            generation = 0
            binders = []
            temp_binders = []
            count = 5
            candidate_binders = []

            while len(temp_binders) < args.num_binders and count >= 0:
                # pdb.set_trace()
                temp_binders += generate_peptide(protein_seq, peptide_length, args.top_k, args.num_binders-len(temp_binders))['Binder'].tolist()
                temp_binders = [seq for seq in temp_binders if 'X' not in seq]
                temp_binders = list(set(temp_binders))
                count -= 1

            for n in range(0, args.num_binders-len(temp_binders)):
                temp_binders.append(generate_random_seq(peptide_length))
            temp_binders = list(set(temp_binders))

            for binder_seq in temp_binders:
                binders.append(Binder(binder_seq, model, pepmlm, tokenizer, args))

            binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
            for m in range(min(args.num_display, len(binders))):
                print(f"Generation: -1\tBinder: {binders[m].binder_seq}\tScore: {binders[m].score}\tPPL: {binders[m].ppl}")

            # for m in range(min(args.num_display, len(binders))):
            #         print(f"{binders[m].binder_seq}")

            no_improvement_generations = 0
            max_tolerance = 20
            previous_score = 10000
            previous_ppl = 10000
            threshold = int(0.1*len(candidate_binding_site))

            # print(f"Threshold: {threshold}")

            while no_improvement_generations < max_tolerance:
                previous_score = binders[0].score
                previous_ppl = binders[0].ppl

                # if binders[0].score <= threshold:
                #     break

                new_binders = []

                s = int((10*len(binders))/100)
                new_binders.extend(binders[:s])

                s = int((90*len(binders))/100)
                half = int(len(binders) / 2)
                for _ in range(s):
                    par1 = random.choice(binders[:half])
                    par2 = random.choice(binders[:half])
                    new_binders.append(par1.mate(par2))

                for k in range(1, len(new_binders)):
                    if new_binders[k].binder_seq == new_binders[k-1].binder_seq:
                        new_seq = new_binders[k].mutate_seq(new_binders[k].binder_seq)
                        new_binders[k] = Binder(new_seq, model, pepmlm, tokenizer, args)

                new_binders = sorted(new_binders, key=lambda binder: (binder.score, binder.ppl))

                for m in range(min(args.num_display, len(new_binders))):
                    print(f"Generation: {generation}\tBinder: {new_binders[m].binder_seq}\tScore: {new_binders[m].score}\tPPL: {new_binders[m].ppl}")

                if new_binders[0].score < previous_score or new_binders[0].ppl < previous_ppl:
                    no_improvement_generations = 0
                    print(f"Generation: {generation}\tImproved!")
                else:
                    no_improvement_generations += 1
                    print(f"Generation: {generation}\tNo improvement {no_improvement_generations} generations")

                # for m in range(min(args.num_display, len(new_binders))):
                #     print(f"{new_binders[m].binder_seq}")

                binders = new_binders
                generation += 1
                for i in range(10):
                    candidate_binders.append({'Binder': new_binders[i].binder_seq, 'PPL': new_binders[i].ppl})

            print(f"Generation: {generation}\tBinder: {binders[0].binder_seq}\tScore: {binders[0].score}\tPPL: {binders[0].ppl}")

            candidate_binders = remove_redundant_elements(candidate_binders)
            candidate_binders = sorted(candidate_binders, key=lambda x: x['PPL'])
            print(f"$$$$$$$$$$ Final Binders for GFAP of length {peptide_length} $$$$$$$$$$")
            for j in range(min(20, len(candidate_binders))):
                print(candidate_binders[j]['Binder'])
            for j in range(min(20, len(candidate_binders))):
                print(candidate_binders[j]['PPL'])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-sm", default='/home/tc415/muPPIt/muppit/train_base_1/model-epoch=14-val_loss=0.40.ckpt',
                        help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument('--protein_seq', type=str)
    parser.add_argument('--motif', type=str)
    parser.add_argument("-target", type=str)
    parser.add_argument("-binder", type=str)
    parser.add_argument("-gt", type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--csv", type=str)
    parser.add_argument('--num_binders', type=int, default=50)
    parser.add_argument("--num_display", type=int, default=1)
    parser.add_argument("-seq", type=str)
    parser.add_argument("-entry", type=str)
    args = parser.parse_args()

    main(args)

