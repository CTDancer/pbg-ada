import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
import random
import esm
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

from models.graph import ProteinGraph, NodeGraph, EdgeGraph
from models.modules_vec import IntraGraphAttention, DiffEmbeddingLayer, MIM, CrossGraphAttention
from pepmlm import generate_peptide
from predict_decoder_2 import ProteinDecoder

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

vhse8_values = {
    'A': [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
    'R': [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83],
    'N': [-0.99, 0.00, 0.69, -0.37, -0.55, 0.85, 0.73, -0.80],
    'D': [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
    'C': [0.18, -1.67, -0.21, 0.00, 1.20, -1.61, -0.19, -0.41],
    'Q': [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41],
    'E': [-1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.36],
    'G': [-0.20, -1.53, -2.63, 2.28, -0.53, -1.18, -1.34, 1.10],
    'H': [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
    'I': [1.27, 0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13],
    'L': [1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62],
    'K': [-1.17, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13, -0.01],
    'M': [1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68],
    'F': [1.52, 0.61, 0.95, -0.16, 0.25, 0.28, -1.33, -0.65],
    'P': [0.22, -0.17, -0.50, -0.05, 0.01, -1.34, 0.19, 3.56],
    'S': [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
    'T': [-0.34, -0.51, -0.55, -1.06, 0.01, -0.01, -0.79, 0.39],
    'W': [1.50, 2.06, 1.79, 0.75, 0.75, 0.13, -1.06, -0.85],
    'Y': [0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
    'V': [0.76, -0.92, 0.17, -1.91, 0.22, -1.40, -0.24, -0.03],
}

AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

vhse8_tensor = torch.stack([torch.tensor(v) for v in vhse8_values.values()])


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

    def forward(self, binder_node_representation, binder_edge_representation, wt_tokens, mut_tokens):
        device = binder_node_representation.device

        # Construct Graph
        binder_node, binder_edge, _, _ = self.graph(None, self.esm, self.alphabet, binder_node_representation, binder_edge_representation)
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

        return mut_affinity, wt_affinity, binder_node, binder_edge

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

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if 'decoder' in k}

        self.load_state_dict(state_dict, strict=True)

        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, binder_node, binder_edge):
        x = binder_node
        E = binder_edge
        for k in range(self.num_layers):
            # Compute messages
            m_ij = self.compute_messages(x, E, self.msg_mlps[k])  # Shape: [B, L, L, hidden_dim]
            # Aggregate messages for each node
            m_i = m_ij.sum(dim=2)  # Shape: [B, L, hidden_dim]
            # Update node features
            m_i = self.update_mlps[k](m_i)  # Shape: [B, L, hidden_dim]
            x = self.layer_norms[k](x + m_i)  # Residual connection and LayerNorm

        # Predict amino acid identities
        binder_logits = self.output_mlp(x)  # Shape: [B, L, num_classes]

        return binder_logits


# def decode(node_representation, edge_representation, decoder):
#     idx_to_aa = {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}
    
#     if node_representation.device != decoder.device:
#         node_representation = node_representation.to(decoder.device)
    
#     # pdb.set_trace()
#     with torch.no_grad():
#         logits = decoder(node_representation, edge_representation)
#     predicted_indices = torch.argmax(logits.squeeze(0), dim=1)
#     predicted_sequence = [idx_to_aa[idx.item()] for idx in predicted_indices]
#     predicted_sequence_str = ''.join(predicted_sequence)

#     return predicted_sequence_str


def decode(logits):
    idx_to_aa = {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}

    predicted_indices = torch.argmax(logits.squeeze(0), dim=1)
    predicted_sequence = [idx_to_aa[idx.item()] for idx in predicted_indices]
    predicted_sequence_str = ''.join(predicted_sequence)

    return predicted_sequence_str


def get_nearest_vhse8(node_representation, vhse8_tensor, use_cosine_similarity=False):
    vhse8_embeddings = node_representation[:, :, -16:-8]  # Shape (1, L, 8)
    vhse8_embeddings = vhse8_embeddings.view(-1, 8)  # Shape (L, 8)

    vhse8_tensor = vhse8_tensor.to(vhse8_embeddings.device)

    if use_cosine_similarity:
        vhse8_embeddings_normalized = F.normalize(vhse8_embeddings, p=2, dim=1)  # Shape (L, 8)
        vhse8_tensor_normalized = F.normalize(vhse8_tensor, p=2, dim=1)  # Shape (20, 8)

        similarity = torch.matmul(vhse8_embeddings_normalized, vhse8_tensor_normalized.T)  # Shape (L, 20)
        closest_indices = torch.argmax(similarity, dim=1)  # Shape (L,)
    else:
        distances = torch.cdist(vhse8_embeddings, vhse8_tensor)  # Shape (L, 20)
        closest_indices = torch.argmin(distances, dim=1)  # Shape (L,)

    # pdb.set_trace()

    closest_embeddings = vhse8_tensor[closest_indices]  # Shape (L, 8)
    closest_embeddings = closest_embeddings.view(1, -1, 8)  # Shape (1, L, 8)

    node_representation[:, :, -16:-8] = closest_embeddings

    return node_representation


def generate_random_peptide(length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
    return ''.join(random.choice(amino_acids) for _ in range(length))


def softargmax1d(input, beta=10):
    *_, n = input.shape
    input = (input - input.mean()) / input.std()
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    muppit = muPPIt(args.d_node_muppit, args.d_edge_muppit, args.d_cross_edge_muppit, args.d_position_muppit, args.n_heads_muppit,
                            args.n_intra_layers_muppit, args.n_mim_layers_muppit, args.n_cross_layers_muppit, args.lr)
    muppit.load_weights(args.sm)

    decoder = ProteinDecoder(args.node_dim_decoder, args.edge_dim_decoder, args.hidden_dim_decoder, args.n_layers_decoder, 24, args.dropout_decoder)
    decoder.load_weights(args.sm)
    
    device = muppit.device
    muppit.eval()
    decoder.eval()

    pepmlm_results = generate_peptide(args.mut, args.binder_length, args.top_k, args.num_binders)
    pepmlm_results_nox = pepmlm_results[~pepmlm_results['Binder'].str.contains('X')]

    if len(pepmlm_results_nox) == 0:
        binder = pepmlm_results.sort_values(by='Pseudo Perplexity')['Binder'].iloc[0].replace('X', 'A')
    else:
        binder = pepmlm_results_nox.sort_values(by='Pseudo Perplexity')['Binder'].iloc[0]

    # binder = 'WPDWQSRLVRPDDDRLNFIW'

    # binder = generate_random_peptide(args.binder_length)
    print(f"Initial Binder: {binder}")
    current_binder = binder
    
    binder = tokenizer(binder, return_tensors='pt')
    mutant = tokenizer(args.mut, return_tensors='pt')
    wildtype = tokenizer(args.wt, return_tensors='pt')

    binder_tokens = binder['input_ids'][:, 1:-1].to(device)
    mut_tokens = mutant['input_ids'][:, 1:-1].to(device)
    wt_tokens = wildtype['input_ids'][:, 1:-1].to(device)

    # Get binder node representation and edge representation
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    
    nodegraph = NodeGraph(args.d_position_muppit)
    binder_node_representation = nodegraph(binder_tokens, esm_model, alphabet)
    
    positional_embedding = binder_node_representation[:, :, -8:]    # Store the positional embeddings

    edgegraph = EdgeGraph()
    binder_edge_representation = edgegraph(binder_tokens, esm_model)
    
    mut_tokens.requires_grad = False
    wt_tokens.requires_grad = False
    binder_node_representation.requires_grad = True
    binder_edge_representation.requires_grad = True
    positional_embedding.requires_grad = False

    for epoch in tqdm(range(args.num_updates)):
        mut_pred, wt_pred, _, _ = muppit(binder_node_representation, binder_edge_representation, wt_tokens, mut_tokens)

        mut_pred_affinity = softargmax1d(mut_pred)
        wt_pred_affinity = softargmax1d(wt_pred)

        # mut_pred_affinity.retain_grad()
        # wt_pred_affinity.retain_grad()

        print(f"Mutant affinity: {torch.argmax(mut_pred) - 16}")
        print(f"Wildtype affinity: {torch.argmax(wt_pred) - 16}")

        loss = (mut_pred_affinity - 0) + (14 - wt_pred_affinity)

        # mut_pred = (mut_pred_res - mut_pred_res.mean()) / mut_pred_res.std()
        # wt_pred = (wt_pred_res - wt_pred_res.mean()) / wt_pred_res.std()

        # pdb.set_trace()

        # mut_pred = torch.log_softmax(mut_pred, dim=-1)
        # wt_pred = torch.log_softmax(wt_pred, dim=-1)

        # mut_target = torch.zeros(15)
        # mut_target[0] = 1

        # wt_target = torch.zeros(15)
        # wt_target[-1] = 1

        # loss = nn.KLDivLoss()(mut_pred.squeeze(0), mut_target.to(mut_pred.device)) + nn.KLDivLoss()(wt_pred.squeeze(0), wt_target.to(wt_pred.device)) + (mut_pred.mean() - wt_pred.mean())

        loss.backward()

        # pdb.set_trace()

        with torch.no_grad():
            # pdb.set_trace()
            binder_node_representation -= binder_node_representation.grad * args.step_size 
            binder_node_representation[:, :, -8:] = positional_embedding    # Keep the positional embeddings unchanged (last 8 dimensions)
            binder_node_representation = get_nearest_vhse8(binder_node_representation, vhse8_tensor, use_cosine_similarity=False)
            binder_edge_representation -= args.step_size * binder_edge_representation.grad

            _, _, binder_node, binder_edge = muppit(binder_node_representation, binder_edge_representation, wt_tokens, mut_tokens)

            logits = decoder(binder_node, binder_edge)
            new_binder = decode(logits)
            
            while current_binder == new_binder:
                binder_node_representation -= binder_node_representation.grad * args.step_size 
                binder_node_representation[:, :, -8:] = positional_embedding    # Keep the positional embeddings unchanged (last 8 dimensions)
                binder_node_representation = get_nearest_vhse8(binder_node_representation, vhse8_tensor, use_cosine_similarity=False)
                binder_edge_representation -= args.step_size * binder_edge_representation.grad

                _, _, binder_node, binder_edge = muppit(binder_node_representation, binder_edge_representation, wt_tokens, mut_tokens)
                logits = decoder(binder_node, binder_edge)
                new_binder = decode(logits)

            current_binder = new_binder
            print(f"Updated Binder: {new_binder}")

            binder_tokens = tokenizer(new_binder, return_tensors='pt')['input_ids'][:, 1:-1].to(device)
            binder_node_representation = nodegraph(binder_tokens, esm_model, alphabet)
            binder_edge_representation = edgegraph(binder_tokens, esm_model)
            binder_node_representation.requires_grad = True
            binder_edge_representation.requires_grad = True
            

        # Zero the gradients after the update
        # binder_node_representation.grad.zero_()
        # binder_edge_representation.grad.zero_()

    # Decode
    # final_binder = decode(binder_node_representation, binder_edge_representation, decoder)
    # print("Final Binder sequence: ", final_binder)


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
    parser.add_argument("-sm", default=None, help="Pre-trained muPPIt_decoder", type=str)
    parser.add_argument("-lr", type=float, default=1e-3)

    parser.add_argument("-node_dim_decoder", type=int, default=256, help="Node representation dimension")
    parser.add_argument("-edge_dim_decoder", type=int, default=128, help="Edge representation dimension")
    parser.add_argument("-hidden_dim_decoder", type=int, default=256, help="Decoder hidden dimension")
    parser.add_argument("-n_layers_decoder", type=int, default=4, help="Decoder layer counts")
    parser.add_argument("-dropout_decoder", type=float, default=0.2)


    parser.add_argument("-num_updates", type=int, default=15, help="Update rounds for binder")
    parser.add_argument("-step_size", type=float, default=1e-3, help="update step size for binder")
    parser.add_argument("-top_k", type=int, default=3, help="top k sampling of PepMLM")
    parser.add_argument("-num_binders", type=int, default=50, help="candidate binder pool size of PepMLM")
    parser.add_argument("-binder_length", type=int, default=12)
    
    args = parser.parse_args()
    main(args)
