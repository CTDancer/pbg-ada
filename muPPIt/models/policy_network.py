import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import esm
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


def RoPE(embedding, seq_len, dim):
    assert dim % 2 == 0, "Embedding dimension must be even."

    # Step 1: Split embedding dimension into pairs
    half_dim = dim // 2
    embedding_2d = embedding.view(embedding.shape[0], embedding.shape[1], half_dim, 2)  # Shape: B * L * (D/2) * 2
    
    # Step 2: Generate sinusoidal positional encodings
    position_ids = torch.arange(seq_len, dtype=torch.float32, device=embedding.device).unsqueeze(1)
    # Compute the scaling factor for each embedding pair (logarithmically scaled across half_dim)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=embedding.device) / half_dim))
    
    # Compute the angles for RoPE using sine and cosine
    angle_rates = position_ids * inv_freq  # Shape: (L, D/2)
    sin_embeddings = torch.sin(angle_rates)  # Shape: (L, D/2)
    cos_embeddings = torch.cos(angle_rates)  # Shape: (L, D/2)
    
    # Expand sin and cos embeddings to match the embedding pairs
    sin_embeddings = sin_embeddings.unsqueeze(0)  # Shape: (1, L, D/2)
    cos_embeddings = cos_embeddings.unsqueeze(0)  # Shape: (1, L, D/2)
    
    # pdb.set_trace()

    # Step 3: Apply RoPE (rotation matrix) to embedding pairs
    rotated_embedding = torch.cat(
        [embedding_2d[..., 0] * cos_embeddings - embedding_2d[..., 1] * sin_embeddings,  # Apply RoPE rotation
         embedding_2d[..., 0] * sin_embeddings + embedding_2d[..., 1] * cos_embeddings], dim=-1
    )  # Shape: B * L * D
    
    # Reshape back to original B*L*D format
    rotated_embedding = rotated_embedding.view(embedding.shape)
    
    return rotated_embedding


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PolicyNetwork(nn.Module):
    def __init__(self, amino_acid_vocab_size, embed_size, num_layers, heads, 
                 forward_expansion, dropout, max_length):
        super(PolicyNetwork, self).__init__()

        self.esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, amino_acid_vocab_size)  # Output L*20 actions, including all 20 amino acids

    def forward(self, x, mask=None):
        B, L = x.shape
        with torch.no_grad():
            esm_results = self.esm_model(x, repr_layers=[33], return_contacts=False)
        esm_embedding = esm_results["representations"][33]  # shape: 1*L*1280

        
        out = RoPE(esm_embedding, L, esm_embedding.shape[-1])

        # pdb.set_trace()

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)  # Shape: B*L*20 (predicting for each amino acid position)
        return F.log_softmax(out, dim=-1)  # Output probabilities for each position