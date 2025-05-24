import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import pdb

class ProteinGraph(nn.Module):
    def __init__(self, d_node, d_edge, d_position):
        super(ProteinGraph, self).__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_position = d_position

        d_node_original = 1280 + 8 + d_position
        self.node_mapping = nn.Linear(d_node_original, self.d_node)
        self.linear_edge = nn.Linear(1, d_edge)

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

        aa_to_idx = {'A': 5, 'R': 10, 'N': 17, 'D': 13, 'C': 23, 'Q': 16, 'E': 9, 'G': 6, 'H': 21, 'I': 12, 'L': 4, 'K': 15, 'M': 20, 'F': 18, 'P': 14, 'S': 8, 'T': 11, 'W': 22, 'Y': 19, 'V': 7}

        self.vhse8_tensor = torch.zeros(24, 8)
        for aa, values in vhse8_values.items():
            aa_index = aa_to_idx[aa]
            self.vhse8_tensor[aa_index] = torch.tensor(values)
        self.vhse8_tensor.requires_grad = False
        # self.position_embedding = nn.Embedding(seq_len, self.d_position)

    # def one_hot_encoding(self, seq_len):
    #     positions = torch.arange(seq_len).unsqueeze(1)
    #     one_hot = torch.nn.functional.one_hot(positions, num_classes=seq_len).squeeze(1)
    #     return one_hot

    def create_sinusoidal_embeddings(self, seq_len, d_position):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_position, 2) * -(math.log(10000.0) / d_position))
        pe = torch.zeros(seq_len, d_position)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_position)
        return pe


    def add_cls_eos(self, tensor):
        modified_tensor = []
        
        for row in tensor:
            new_row = [0]  # Start with 0 at the beginning
            ones_indices = (row == 1).nonzero(as_tuple=True)[0]
            
            if len(ones_indices) > 0:
                # Add 2 before the first occurrence of 1
                first_one_idx = ones_indices[0].item()
                new_row.extend(row[:first_one_idx].tolist())  # Add elements before the first 1
                new_row.append(2)  # Add 2 before the first 1
                new_row.extend(row[first_one_idx:].tolist())  # Add the rest of the row
            else:
                # No 1 in the row, add 2 at the end
                new_row.extend(row.tolist())
                new_row.append(2)  # Add 2 at the end
            
            modified_tensor.append(torch.tensor(new_row))
        
        return torch.stack(modified_tensor)

    def forward(self, tokens, esm, alphabet, node_representation=None, contact_map=None):
        # pdb.set_trace()
        if tokens is None:
            assert node_representation is not None  # shape: 1*L*1296
            assert contact_map is not None  # shape: 1*L*L*d_edge
            device = node_representation.device
            pad_mask = torch.ones(node_representation.shape[0], node_representation.shape[1]).to(device)
            expanded_pad_mask = torch.ones(contact_map.shape[0], contact_map.shape[1], contact_map.shape[2]).to(device)
            
        else:
            batch_size, seq_len = tokens.size()
            pad_mask = (tokens != alphabet.padding_idx).int()   # B*L
            expanded_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
            device = tokens.device

        if node_representation is None:
            # ESM-2 embedding
            with torch.no_grad():
                esm_results = esm(tokens, repr_layers=[33], return_contacts=True) 
            esm_embedding = esm_results["representations"][33] # shape: B*L*1280
            esm_embedding = esm_embedding * pad_mask.unsqueeze(-1)

            # VSHE embedding
            vhse8_tensor = self.vhse8_tensor.to(device)
            vshe8_embedding = vhse8_tensor[tokens]

            # Sinual positional embedding
            sin_embedding = self.create_sinusoidal_embeddings(seq_len, self.d_position).repeat(batch_size, 1, 1).to(device)  # shape: B*L*d_position
            sin_embedding = sin_embedding * pad_mask.unsqueeze(-1)

            node_representation = torch.cat((esm_embedding, vshe8_embedding, sin_embedding), dim=-1) # B*L*(1280+8+d_position)
        
        node_representation = self.node_mapping(node_representation)    # B*L*d_node

        # pdb.set_trace()

        if contact_map is None:
            # Edge represntation
            with torch.no_grad():
                esm_results = esm(self.add_cls_eos(tokens.cpu()).to(device), repr_layers=[33], return_contacts=True) # add <cls> and <eos> back to the tokens for predicting contact maps

            # pdb.set_trace()
            contact_map = esm_results["contacts"] # shape: B*L*L
        
        edge_representation = self.linear_edge(contact_map.unsqueeze(-1))   # shape: B*L*L*d_edge
        # expanded_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        edge_representation = edge_representation * expanded_pad_mask.unsqueeze(-1)

        # pdb.set_trace()
        return node_representation, edge_representation, pad_mask, expanded_pad_mask


class NodeGraph(nn.Module):
    def __init__(self, d_position):
        super(NodeGraph, self).__init__()
        self.d_position = d_position

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

        aa_to_idx = {'A': 5, 'R': 10, 'N': 17, 'D': 13, 'C': 23, 'Q': 16, 'E': 9, 'G': 6, 'H': 21, 'I': 12, 'L': 4, 'K': 15, 'M': 20, 'F': 18, 'P': 14, 'S': 8, 'T': 11, 'W': 22, 'Y': 19, 'V': 7}

        self.vhse8_tensor = torch.zeros(24, 8)
        for aa, values in vhse8_values.items():
            aa_index = aa_to_idx[aa]
            self.vhse8_tensor[aa_index] = torch.tensor(values)
        self.vhse8_tensor.requires_grad = False

    def create_sinusoidal_embeddings(self, seq_len, d_position):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_position, 2) * -(math.log(10000.0) / d_position))
        pe = torch.zeros(seq_len, d_position)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_position)
        return pe

    def forward(self, tokens, esm, alphabet):
        # pdb.set_trace()
        batch_size, seq_len = tokens.size()
        pad_mask = (tokens != alphabet.padding_idx).int()   # B*L
        device = tokens.device
        
        # ESM-2 embedding
        with torch.no_grad():
            esm_results = esm(tokens, repr_layers=[33], return_contacts=True) 
        esm_embedding = esm_results["representations"][33] # shape: B*L*1280
        esm_embedding = esm_embedding * pad_mask.unsqueeze(-1)

        # VSHE embedding
        vhse8_tensor = self.vhse8_tensor.to(device)
        vshe8_embedding = vhse8_tensor[tokens]

        sin_embedding = self.create_sinusoidal_embeddings(seq_len, self.d_position).repeat(batch_size, 1, 1).to(device)  # shape: B*L*d_position
        sin_embedding = sin_embedding * pad_mask.unsqueeze(-1)

        node_representation = torch.cat((esm_embedding, vshe8_embedding, sin_embedding), dim=-1)

        return node_representation
    

class EdgeGraph(nn.Module):
    def __init__(self):
        super(EdgeGraph, self).__init__()

    def add_cls_eos(self, tensor):
        modified_tensor = []
        
        for row in tensor:
            new_row = [0]  # Start with 0 at the beginning
            ones_indices = (row == 1).nonzero(as_tuple=True)[0]
            
            if len(ones_indices) > 0:
                # Add 2 before the first occurrence of 1
                first_one_idx = ones_indices[0].item()
                new_row.extend(row[:first_one_idx].tolist())  # Add elements before the first 1
                new_row.append(2)  # Add 2 before the first 1
                new_row.extend(row[first_one_idx:].tolist())  # Add the rest of the row
            else:
                # No 1 in the row, add 2 at the end
                new_row.extend(row.tolist())
                new_row.append(2)  # Add 2 at the end
            
            modified_tensor.append(torch.tensor(new_row))
        
        return torch.stack(modified_tensor)

    def forward(self, tokens, esm):
        device = tokens.device
        with torch.no_grad():
            esm_results = esm(self.add_cls_eos(tokens.cpu()).to(device), repr_layers=[33], return_contacts=True) # add <cls> and <eos> back to the tokens for predicting contact maps

        # pdb.set_trace()
        contact_map = esm_results["contacts"] # shape: B*L*L

        return contact_map


if __name__ == '__main__':
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    tokens = torch.tensor([[5,5,5,1]])
    graph = ProteinGraph(256, 128, 8)
    node, edge, pad = graph(tokens, model, alphabet)
    print(node.shape, edge.shape, pad.shape)