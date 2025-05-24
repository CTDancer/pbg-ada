import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.distributions import Categorical
from tqdm import tqdm
import random
import pdb
import esm
import gc

from models.graph import ProteinGraph, NodeGraph, EdgeGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention
from pepmlm import generate_peptide
from models.policy_network import *

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

        self.criterion = nn.CrossEntropyLoss()


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
            mut_node, binder_node, mut_binder_edges = layer(mut_node, binder_node, mut_binder_edges, diff_vec)

        wt_binder_edges = torch.mean(wt_binder_edges, dim=(1,2))
        mut_binder_edges = torch.mean(mut_binder_edges, dim=(1,2))

        # pdb.set_trace()

        mut_affinity = self.affinity_linear(mut_binder_edges)
        wt_affinity = self.affinity_linear(wt_binder_edges)
        
        return mut_affinity.squeeze(-1), wt_affinity.squeeze(-1)


def compute_pseudo_perplexity(pepmlm, binder_tokens, protein_tokens):
    sequence_tokens = torch.cat([protein_tokens, binder_tokens], dim=1).to(pepmlm.device)
    total_loss = 0

    # Loop through each token in the binder sequence
    binder_len = binder_tokens.size(1)
    for i in range(-binder_len - 1, -1):
        masked_input = sequence_tokens.clone()

        # Mask one token at a time
        masked_input[0, i] = pepmlm.config.mask_token_id

        labels = torch.full(sequence_tokens.shape, -100).to(pepmlm.device)
        labels[0, i] = sequence_tokens[0, i]

        outputs = pepmlm(masked_input, labels=labels)
        total_loss += outputs.loss

    avg_loss = total_loss / binder_len

    pseudo_perplexity = torch.exp(avg_loss)
    return pseudo_perplexity


def generate_random_binders(num, length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
    return [''.join(random.choice(amino_acids) for _ in range(length)) for _ in range(num)]


def softargmax1d(input, beta=10):
    *_, n = input.shape
    input = (input - input.mean()) / input.std()
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def evaluate_binder(binder_sequence, mutant_sequence, wild_type_sequence, muppit, pepmlm):
    mut_pred, wt_pred = muppit(binder_sequence, wild_type_sequence, mutant_sequence)
    # pdb.set_trace()

    mut_affinity = softargmax1d(mut_pred.squeeze(0))
    wt_affinity = softargmax1d(wt_pred.squeeze(0))

    # delta_affinity = torch.abs(wt_affinity - mut_affinity)
    delta_mut = mut_affinity - 0
    delta_wt = 14 - wt_affinity

    # pdb.set_trace()
    
    ppl = compute_pseudo_perplexity(pepmlm, binder_sequence, mutant_sequence)

    # preference_score = 2 * (w1 * ppl * (w2 / delta_affinity)) / (w1 * ppl + w2 / delta_affinity)
    # preference_score = w1 * ppl + w2 / delta_affinity # The smaller, the better
    # preference_score = delta_affinity ** 2 + ppl / 10   # The bigger, the better
    preference_score = delta_mut + delta_wt + ppl / 2  # The smaller, the better

    if torch.argmax(wt_pred) <= torch.argmax(mut_pred):
        preference_score *= 10

    return preference_score, torch.argmax(mut_pred), torch.argmax(wt_pred)


def evaluate_binder_kl(binder_sequence, mutant_sequence, wild_type_sequence, muppit, pepmlm):
    mut_pred, wt_pred = muppit(binder_sequence, mutant_sequence, wild_type_sequence)
    
    mut_target = torch.zeros(15)
    mut_target[0] = 1

    wt_target = torch.zeros(15)
    wt_target[-1] = 1

    similarity = F.kl_div(mut_pred.squeeze(0), mut_target.to(mut_pred.device)) + F.kl_div(wt_pred.squeeze(0), wt_target.to(wt_pred.device))

    ppl = compute_pseudo_perplexity(pepmlm, binder_sequence, mutant_sequence)

    preference_score = similarity + ppl / 10

    return preference_score, torch.argmax(mut_pred), torch.argmax(wt_pred)


def compute_loss(trajectories, policy_network):
    loss = 0.0
    entropy_loss = 0.0  # To accumulate the entropy losses

    for binder_sequence, action_index, mutated_binder, preference_score, _, _ in trajectories:
        action_probs = policy_network(binder_sequence)

        action_probs_flat = action_probs.view(-1)  # Shape: [L*20]

        # Log probability for the chosen action
        log_prob_action = action_probs_flat[action_index]  # Already log probability

        trajectory_loss = torch.exp(log_prob_action) * preference_score 

        loss += trajectory_loss

        # Entropy loss aims to encourage exploration
        entropy = -torch.sum(torch.exp(action_probs_flat) * action_probs_flat)
        entropy_loss += entropy 

    # Normalize the loss by the number of trajectories
    loss = loss / len(trajectories)

    loss += 0.1 * (entropy_loss / len(trajectories))

    return loss


def compute_loss_kl(trajectories, policy_network):
    loss = 0.0
    entropy_loss = 0.0

    for binder_sequence, action_index, mutated_binder, preference_score, _, _ in trajectories:
        action_probs = policy_network(binder_sequence)

        action_probs_flat = action_probs.view(-1)  # Shape: [L*20]

        log_prob_action = action_probs_flat[action_index]  # Already log probability
        
        trajectory_loss = torch.exp(log_prob_action) * preference_score 

        loss += trajectory_loss

        # Compute entropy using log probabilities (P(x) * log P(x) with P(x) = exp(log P(x)))
        entropy = -torch.sum(torch.exp(action_probs_flat) * action_probs_flat)
        entropy_loss += entropy

    loss = loss / len(trajectories)

    # Add the entropy regularization term to the final loss
    # loss += 0.1 * (entropy_loss / len(trajectories)) 
    return loss


def collect_trajectories(binders, mutant, wildtype, muppit, policy_network, pepmlm):
    trajectories = []
    
    for binder_idx, binder_sequence in tqdm(enumerate(binders), total=len(binders)):
        # Get action probabilities from the policy network
        action_probs = policy_network(binder_sequence)  # Input shape: [1, L], Output shape: [L, 20]
        
        # pdb.set_trace()

        # Flatten the action probabilities to shape [L*20] for sampling an action
        action_probs_flat = action_probs.view(-1)  # Shape: [L*20]
        
        # Sample an action (position and new amino acid) from the probability distribution
        action_distribution = Categorical(logits=action_probs_flat)
        action_index = action_distribution.sample().item()
        
        # Decode the action (position in the sequence and new amino acid)
        position = action_index // 20
        new_amino_acid = action_index % 20 + 4  # amino acid tokens range from 4~23
        
        # Apply the mutation to the binder sequence
        new_binder = binder_sequence.clone()
        new_binder[:, position] = new_amino_acid
        
        # Evaluate the mutated binder sequence using the pre-trained model
        preference_score, mut_affinity, pre_affinity = evaluate_binder(new_binder, mutant, wildtype, muppit, pepmlm)
        
        # Penalize redundant mutations (mutating an amino acid to itself)
        penalty = 0
        if binder_sequence[:, position] == new_amino_acid:
            penalty += 10
        adjusted_preference_score = preference_score + penalty
        
        trajectories.append((binder_sequence, action_index, new_binder, adjusted_preference_score, mut_affinity, pre_affinity))

        del binder_sequence
        del new_binder
        
        gc.collect()

    return trajectories


def random_mutate(binder):
    mutated_binder = binder.clone()

    random_position = random.randint(0, len(binder) - 1) 

    # Choose a random new amino acid (0-19) different from the current one
    current_amino_acid = binder[:, random_position].item()
    
    # Randomly select a new amino acid, excluding the current one
    new_amino_acid = current_amino_acid
    while new_amino_acid == current_amino_acid:
        new_amino_acid = random.randint(4, 23)  # Random amino acid index (4-23)

    mutated_binder[:, random_position] = new_amino_acid

    return mutated_binder


def select_top_candidates_and_mutate(trajectories, top_k):
    current_pool = [traj[2] for traj in trajectories]
    top_candidates = [traj[2] for traj in trajectories[:top_k]]

    new_pool = [random_mutate(binder) for binder in top_candidates] + current_pool
    
    return new_pool


def decode(tokens, alphabet):
    return "".join([alphabet.get_tok(i) for i in tokens.tolist()])


def main(args):
    '''
    Model Initialization
    '''
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    muppit = muPPIt.load_from_checkpoint(args.sm_muppit,
                                        d_node = args.d_node_muppit, 
                                        d_edge = args.d_edge_muppit, 
                                        d_cross_edge = args.d_cross_edge_muppit, 
                                        d_position = args.d_position_muppit, 
                                        num_heads = args.n_heads_muppit,
                                        num_intra_layers = args.n_intra_layers_muppit, 
                                        num_mim_layers = args.n_mim_layers_muppit, 
                                        num_cross_layers = args.n_cross_layers_muppit,
                                        lr=None)

    device = muppit.device
    muppit.eval()
    for params in muppit.parameters():
        params.requires_grad = False

    policy_network = PolicyNetwork(
        amino_acid_vocab_size=20, 
        embed_size=args.pn_embed_size, 
        num_layers=args.pn_num_layers, 
        heads=args.pn_num_heads, 
        forward_expansion=args.pn_forward_expansion, 
        dropout=args.pn_dropout, 
        max_length=args.binder_length,
    )

    policy_network.to(device)
    policy_network.device = device
    for param in policy_network.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(policy_network.parameters(), lr=args.pn_lr)

    _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    pepmlm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)
    for param in pepmlm.parameters():
        param.requires_grad = False

    '''
    Binder Initialization
    '''
    pepmlm_results = generate_peptide(args.mut, args.binder_length, args.top_k, args.num_binders)
    pepmlm_results = pepmlm_results.sort_values(by='Pseudo Perplexity')
    pepmlm_results_nox = pepmlm_results[~pepmlm_results['Binder'].str.contains('X')]

    if len(pepmlm_results_nox) == 0:
        binders = [binder.replace('X', 'A') for binder in pepmlm_results['Binder'].tolist()]
    else:
        binders = pepmlm_results_nox['Binder'].tolist()

    # Also a portion of random binders to add diversity
    if args.num_random == 0 and len(binders) < args.num_binders:
        args.num_random = args.num_binders - len(binders)

    if args.num_random > 0:
        binders += generate_random_binders(args.num_random, args.binder_length)

    # Tokenization
    binders = [tokenizer(binder, return_tensors='pt')['input_ids'][:, 1:-1].to(device) for binder in binders]
    mutant = tokenizer(args.mut, return_tensors='pt')['input_ids'][:, 1:-1].to(device)
    wildtype = tokenizer(args.wt, return_tensors='pt')['input_ids'][:, 1:-1].to(device)

    '''
    Optimization Loop
    '''
    # Main optimization loop
    for epoch in range(args.max_epochs):
        # current_binders = binders

        # Collect trajectories
        trajectories = collect_trajectories(binders, mutant, wildtype, muppit, policy_network, pepmlm)
        
        # Update the policy network
        loss = compute_loss(trajectories, policy_network)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        trajectories = sorted(trajectories, key=lambda x: x[3], reverse=False)
        # pdb.set_trace()
        print(f"Best Binder: {decode(binders[0].squeeze(0), alphabet)}\tScore: {trajectories[0][-3].item()}\tMut: {trajectories[0][-2].item()}\tWt: {trajectories[0][-1].item()}")
        
        # Select the top candidates and generate new mutations for the next iteration
        # This step is to maintain a diverse pool while focusing on high-performing binders
        new_binders = select_top_candidates_and_mutate(trajectories, top_k=20)
        # binders = list(set(current_binders + new_binders))
        binders = new_binders


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-wt", type=str, required=True)
    parser.add_argument("-mut", type=str, required=True)

    parser.add_argument("-d_node_muppit", type=int, default=1024, help="Node Representation Dimension")
    parser.add_argument("-d_edge_muppit", type=int, default=512, help="Intra-Graph Edge Representation Dimension")
    parser.add_argument("-d_cross_edge_muppit", type=int, default=512, help="Cross-Graph Edge Representation Dimension")
    parser.add_argument("-d_position_muppit", type=int, default=8, help="Positional Embedding Dimension")
    parser.add_argument("-n_heads_muppit", type=int, default=8)
    parser.add_argument("-n_intra_layers_muppit", type=int, default=1)
    parser.add_argument("-n_mim_layers_muppit", type=int, default=1)
    parser.add_argument("-n_cross_layers_muppit", type=int, default=1)
    parser.add_argument("-sm_muppit", default=None, help="Pre-trained muPPIt", type=str)

    parser.add_argument("-pn_embed_size", type=int, default=1280)
    parser.add_argument("-pn_num_layers", type=int, default=4)
    parser.add_argument("-pn_num_heads", type=int, default=4)
    parser.add_argument("-pn_forward_expansion", type=int, default=2)
    parser.add_argument("-pn_dropout", type=float, default=0.2)
    parser.add_argument("-pn_lr", type=float, default=1e-3)

    parser.add_argument("-max_epochs", type=int, default=15, help="Update rounds for binder")
    parser.add_argument("-top_k", type=int, default=3, help="top k sampling of PepMLM")
    parser.add_argument("-num_binders", type=int, default=50, help="candidate binder pool size of PepMLM")
    parser.add_argument("-binder_length", type=int, default=12)
    parser.add_argument("-num_random", type=int, default=10)
    
    args = parser.parse_args()
    main(args)



