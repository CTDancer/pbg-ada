import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class IntraGraphAttention(nn.Module):
    def __init__(self, d_node, d_edge, num_heads, negative_slope=0.2):
        super(IntraGraphAttention, self).__init__()
        assert d_node % num_heads == 0, "d_node must be divisible by num_heads"
        assert d_edge % num_heads == 0, "d_edge must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_node // num_heads
        self.d_edge_head = d_edge // num_heads

        self.Wn = nn.Linear(d_node, d_node)
        self.Wh = nn.Linear(self.d_k, self.d_k)
        self.We = nn.Linear(d_edge, d_edge)
        self.Wn_2 = nn.Linear(d_node, d_node)
        self.We_2 = nn.Linear(d_edge, d_edge)
        self.attn_linear = nn.Linear(self.d_k * 2 + self.d_edge_head, 1, bias=False)
        self.edge_linear = nn.Linear(self.d_k * 2 + self.d_edge_head, self.d_edge_head)

        self.out_proj_node = nn.Linear(d_node, d_node)
        self.out_proj_edge = nn.Linear(d_edge, d_edge)

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        for layer in [self.Wn, self.Wh, self.We, self.Wn_2, self.We_2, self.attn_linear, self.edge_linear, self.out_proj_node, self.out_proj_edge]:
            nn.init.kaiming_uniform_(layer.weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, node_representation, edge_representation):
        # node_representation: (B, L, d_node)
        # edge_representation: (B, L, L, d_edge)
        # pdb.set_trace()
        B, L, d_node = node_representation.size()
        d_edge = edge_representation.size(-1)
        
        # Multi-head projection
        node_proj = self.Wn(node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj = self.We(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        
        # Node representation update
        new_node_representation = self.single_head_attention_node(node_proj, edge_proj)
        
        concatenated_node_rep = new_node_representation.view(B, L, -1)  # Shape: (B, L, num_heads * d_k)
        new_node_representation = self.out_proj_node(concatenated_node_rep)

        # Edge representation update
        node_proj_2 = self.Wn_2(new_node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj_2 = self.We_2(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        
        new_edge_representation = self.single_head_attention_edge(node_proj_2, edge_proj_2)
        
        concatenated_edge_rep = new_edge_representation.view(B, L, L, -1)  # Shape: (B, L, L, num_heads * d_edge_head)
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)
        
        return new_node_representation, new_edge_representation

    def single_head_attention_node(self, node_representation, edge_representation):
        B, L, num_heads, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)

        hi = node_representation.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_k)
        hj = node_representation.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_k)

        hi_hj_concat = torch.cat([hi.expand(-1, -1, L, -1, -1), 
                                hj.expand(-1, L, -1, -1, -1), 
                                edge_representation], dim=-1)  # shape: (B, L, L, num_heads, 2*d_k + d_edge_head)

        attention_scores = self.attn_linear(hi_hj_concat).squeeze(-1)  # shape: (B, L, L, num_heads)
        
        # Mask the diagonal (self-attention) by setting it to a large negative value
        mask = torch.eye(L).bool().unsqueeze(0).unsqueeze(-1).to(node_representation.device)  # shape: (1, L, L, 1)
        attention_scores.masked_fill_(mask, float('-inf'))
        
        attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=2)  # shape: (B, L, L, num_heads)

        # Aggregating features correctly along the L dimension
        node_representation_Wh = self.Wh(node_representation)  # shape: (B, L, num_heads, d_k)
        node_representation_Wh = node_representation_Wh.permute(0, 2, 1, 3)  # shape: (B, num_heads, L, d_k)
        
        aggregated_features = torch.matmul(attention_probs.permute(0, 3, 1, 2), node_representation_Wh)  # shape: (B, num_heads, L, d_k)
        aggregated_features = aggregated_features.permute(0, 2, 1, 3)  # shape: (B, L, num_heads, d_k)

        new_node_representation = node_representation + self.leaky_relu(aggregated_features)  # shape: (B, L, num_heads, d_k)

        return new_node_representation

    def single_head_attention_edge(self, node_representation, edge_representation):
        # Update edge representation
        B, L, num_heads, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)

        hi = node_representation.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_k)
        hj = node_representation.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_k)

        hi_hj_concat = torch.cat([edge_representation, hi.expand(-1, -1, L, -1, -1), hj.expand(-1, L, -1, -1, -1)], dim=-1)  # shape: (B, L, L, num_heads, 2*d_k + d_edge_head)

        new_edge_representation = self.edge_linear(hi_hj_concat)  # shape: (B, L, L, num_heads, d_edge_head)

        return new_edge_representation


class DiffEmbeddingLayer(nn.Module):
    def __init__(self, d_node):
        super(DiffEmbeddingLayer, self).__init__()
        self.W_delta = nn.Linear(d_node, d_node)

    def forward(self, wt_node, mut_node):
        delta_h = mut_node - wt_node  # (B, L, d_node)
        diff_vec = torch.relu(self.W_delta(delta_h))  # (B, L, d_node)
        return diff_vec


class MIM(nn.Module):
    def __init__(self, d_node, d_edge, d_diff, num_heads, negative_slope=0.2):
        super(MIM, self).__init__()
        assert d_node % num_heads == 0, "d_node must be divisible by num_heads"
        assert d_edge % num_heads == 0, "d_edge must be divisible by num_heads"
        assert d_diff % num_heads == 0, "d_diff must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_node // num_heads
        self.d_edge_head = d_edge // num_heads
        self.d_diff_head = d_diff // num_heads

        self.Wn = nn.Linear(d_node, d_node)
        self.Wh = nn.Linear(self.d_k, self.d_k)
        self.We = nn.Linear(d_edge, d_edge)
        self.Wn_2 = nn.Linear(d_node, d_node)
        self.We_2 = nn.Linear(d_edge, d_edge)
        self.Wd = nn.Linear(d_diff, d_diff)
        self.Wd_2 = nn.Linear(d_diff, d_diff)
        self.attn_linear = nn.Linear(self.d_k * 2 + self.d_edge_head + 2 * self.d_diff_head, 1, bias=False)
        self.edge_linear = nn.Linear(self.d_k * 2 + self.d_edge_head + 2 * self.d_diff_head, self.d_edge_head)

        self.out_proj_node = nn.Linear(d_node, d_node)
        self.out_proj_edge = nn.Linear(d_edge, d_edge)

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        for layer in [self.Wn, self.Wh, self.We, self.Wn_2, self.We_2, self.Wd, self.Wd_2, self.attn_linear, self.edge_linear, self.out_proj_node, self.out_proj_edge]:
            nn.init.kaiming_uniform_(layer.weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, node_representation, edge_representation, diff_vec):
        # node_representation: (B, L, d_node)
        # edge_representation: (B, L, L, d_edge)
        # diff_vec: (B, L, d_diff)

        B, L, d_node = node_representation.size()
        d_edge = edge_representation.size(-1)
        
        # Multi-head projection
        node_proj = self.Wn(node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj = self.We(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        diff_proj = self.Wd(diff_vec).view(B, L, self.num_heads, self.d_diff_head)  # (B, L, num_heads, d_diff_head)

        # Node representation update
        new_node_representation = self.single_head_attention_node(node_proj, edge_proj, diff_proj)
        
        concatenated_node_rep = new_node_representation.view(B, L, -1)  # Shape: (B, L, num_heads * d_k)
        new_node_representation = self.out_proj_node(concatenated_node_rep)

        # Edge representation update
        node_proj_2 = self.Wn_2(new_node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj_2 = self.We_2(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        diff_proj_2 = self.Wd_2(diff_vec).view(B, L, self.num_heads, self.d_diff_head)  # (B, L, num_heads, d_diff_head)

        new_edge_representation = self.single_head_attention_edge(node_proj_2, edge_proj_2, diff_proj_2)
        
        concatenated_edge_rep = new_edge_representation.view(B, L, L, -1)  # Shape: (B, L, L, num_heads * d_edge_head)
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)
        
        return new_node_representation, new_edge_representation

    def single_head_attention_node(self, node_representation, edge_representation, diff_vec):
        # Update node representation
        B, L, num_heads, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        d_diff_head = diff_vec.size(-1)

        hi = node_representation.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_k)
        hj = node_representation.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_k)
        diff_i = diff_vec.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_diff_head)
        diff_j = diff_vec.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_diff_head)
        
        hi_hj_concat = torch.cat([
            hi.expand(-1, -1, L, -1, -1), 
            hj.expand(-1, L, -1, -1, -1), 
            edge_representation, 
            diff_i.expand(-1, -1, L, -1, -1), 
            diff_j.expand(-1, L, -1, -1, -1)
        ], dim=-1)  # shape: (B, L, L, num_heads, 2*d_k + d_edge_head + 2*d_diff_head)

        attention_scores = self.attn_linear(hi_hj_concat).squeeze(-1)  # shape: (B, L, L, num_heads)
        
        # Mask the diagonal (self-attention) by setting it to a large negative value
        mask = torch.eye(L).bool().unsqueeze(0).unsqueeze(-1).to(node_representation.device)  # shape: (1, L, L, 1)
        attention_scores.masked_fill_(mask, float('-inf'))
        
        attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=2)  # shape: (B, L, L, num_heads)

        # Aggregating features correctly along the L dimension
        node_representation_Wh = self.Wh(node_representation)  # shape: (B, L, num_heads, d_k)
        node_representation_Wh = node_representation_Wh.permute(0, 2, 1, 3)  # shape: (B, num_heads, L, d_k)
        
        aggregated_features = torch.matmul(attention_probs.permute(0, 3, 1, 2), node_representation_Wh)  # shape: (B, num_heads, L, d_k)
        aggregated_features = aggregated_features.permute(0, 2, 1, 3)  # shape: (B, L, num_heads, d_k)

        new_node_representation = node_representation + self.leaky_relu(aggregated_features)  # shape: (B, L, num_heads, d_k)

        return new_node_representation


    def single_head_attention_edge(self, node_representation, edge_representation, diff_vec):
        # Update edge representation
        B, L, num_heads, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        d_diff_head = diff_vec.size(-1)

        hi = node_representation.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_k)
        hj = node_representation.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_k)
        diff_i = diff_vec.unsqueeze(2)  # shape: (B, L, 1, num_heads, d_diff_head)
        diff_j = diff_vec.unsqueeze(1)  # shape: (B, 1, L, num_heads, d_diff_head)

        hi_hj_concat = torch.cat([edge_representation, 
                                  hi.expand(-1, -1, L, -1, -1), 
                                  hj.expand(-1, L, -1, -1, -1),
                                  diff_i.expand(-1, -1, L, -1, -1), 
                                  diff_j.expand(-1, L, -1, -1, -1)], dim=-1)  # shape: (B, L, L, num_heads, 2*d_k + d_edge_head + 2*d_diff_head)

        new_edge_representation = self.edge_linear(hi_hj_concat)  # shape: (B, L, L, num_heads, d_edge_head)

        return new_edge_representation


class CrossGraphAttention(nn.Module):
    def __init__(self, d_node, d_cross_edge, d_diff, num_heads, negative_slope=0.2):
        super(CrossGraphAttention, self).__init__()
        assert d_node % num_heads == 0, "d_node must be divisible by num_heads"
        assert d_cross_edge % num_heads == 0, "d_edge must be divisible by num_heads"
        assert d_diff % num_heads == 0, "d_diff must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_node // num_heads
        self.d_edge_head = d_cross_edge // num_heads
        self.d_diff_head = d_diff // num_heads

        self.Wn = nn.Linear(d_node, d_node)
        self.Wh = nn.Linear(self.d_k, self.d_k)
        self.We = nn.Linear(d_cross_edge, d_cross_edge)
        self.Wn_2 = nn.Linear(d_node, d_node)
        self.We_2 = nn.Linear(d_cross_edge, d_cross_edge)
        self.Wd = nn.Linear(d_diff, d_diff)
        self.Wd_2 = nn.Linear(d_diff, d_diff)
        self.attn_linear_target = nn.Linear(self.d_k * 2 + self.d_edge_head + self.d_diff_head, 1, bias=False)
        self.attn_linear_binder = nn.Linear(self.d_k * 2 + self.d_edge_head, 1, bias=False)
        self.edge_linear = nn.Linear(self.d_k * 2 + self.d_edge_head + self.d_diff_head, self.d_edge_head)

        self.out_proj_node = nn.Linear(d_node, d_node)
        self.out_proj_edge = nn.Linear(d_cross_edge, d_cross_edge)

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        for layer in [self.Wn, self.Wh, self.We, self.Wn_2, self.We_2, self.Wd, self.Wd_2, self.attn_linear_target, self.attn_linear_binder, self.edge_linear, self.out_proj_node, self.out_proj_edge]:
            nn.init.kaiming_uniform_(layer.weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, target_representation, binder_representation, edge_representation, diff_vec):
        B, L1, d_node = target_representation.size()
        L2 = binder_representation.size()[1]
        d_edge = edge_representation.size(-1)

        # pdb.set_trace()
        
        # Multi-head projection
        target_proj = self.Wn(target_representation).view(B, L1, self.num_heads, self.d_k)  
        binder_proj = self.Wn(binder_representation).view(B, L2, self.num_heads, self.d_k)          
        edge_proj = self.We(edge_representation).view(B, L1, L2, self.num_heads, self.d_edge_head)  
        diff_proj = self.Wd(diff_vec).view(B, L1, self.num_heads, self.d_diff_head)   

        # Edge representation update
        new_edge_representation = self.single_head_attention_edge(target_proj, binder_proj, edge_proj, diff_proj)

        concatenated_edge_rep = new_edge_representation.view(B, L1, L2, -1)  
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)

        # Node representation update
        target_proj_2 = self.Wn_2(target_representation).view(B, L1, self.num_heads, self.d_k)  
        binder_proj_2 = self.Wn_2(binder_representation).view(B, L2, self.num_heads, self.d_k)  
        edge_proj_2 = self.We_2(new_edge_representation).view(B, L1, L2, self.num_heads, self.d_edge_head)  
        diff_proj_2 = self.Wd_2(diff_vec).view(B, L1, self.num_heads, self.d_diff_head)   
        
        new_target_representation = self.single_head_attention_target(target_proj_2, binder_proj_2, edge_proj_2, diff_proj_2)
        new_binder_representation = self.single_head_attention_binder(binder_proj_2, target_proj_2, edge_proj_2)
        
        concatenated_target_rep = new_target_representation.view(B, L1, -1)  
        new_target_representation = self.out_proj_node(concatenated_target_rep)

        concatenated_binder_rep = new_binder_representation.view(B, L2, -1)  
        new_binder_representation = self.out_proj_node(concatenated_binder_rep)

        return new_target_representation, new_binder_representation, new_edge_representation

    def single_head_attention_target(self, target_representation, binder_representation, edge_representation, diff_vec):
        # Update target node representation
        # pdb.set_trace()
        B, L1, num_heads, d_k = target_representation.size()
        L2 = binder_representation.size(1)
        d_edge_head = edge_representation.size(-1)
        d_diff_head = diff_vec.size(-1)

        hi = target_representation.unsqueeze(2)  # shape: (B, L1, 1, num_heads, d_k)
        hj = binder_representation.unsqueeze(1)  # shape: (B, 1, L2, num_heads, d_k)
        diff_i = diff_vec.unsqueeze(2)  # shape: (B, L1, 1, num_heads, d_diff_head)

        # Concatenate hi, hj, edge_representation, and diff_i
        hi_hj_concat = torch.cat([
            hi.expand(-1, -1, L2, -1, -1), 
            hj.expand(-1, L1, -1, -1, -1), 
            edge_representation, 
            diff_i.expand(-1, -1, L2, -1, -1)
        ], dim=-1)  # shape: (B, L1, L2, num_heads, 2*d_k + d_edge_head + d_diff_head)

        # Calculate attention scores
        attention_scores = self.attn_linear_target(hi_hj_concat).squeeze(-1)  # shape: (B, L1, L2, num_heads)
        attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=2)  # shape: (B, L1, L2, num_heads)

        # Aggregating features correctly along the L2 dimension
        binder_representation_Wh = self.Wh(binder_representation)  # shape: (B, L2, num_heads, d_k)
        binder_representation_Wh = binder_representation_Wh.permute(0, 2, 1, 3)  # shape: (B, num_heads, L2, d_k)
        
        aggregated_features = torch.matmul(attention_probs.permute(0, 3, 1, 2), binder_representation_Wh)  # shape: (B, num_heads, L1, d_k)
        aggregated_features = aggregated_features.permute(0, 2, 1, 3)  # shape: (B, L1, num_heads, d_k)

        # Update target representation
        new_target_representation = target_representation + self.leaky_relu(aggregated_features)  # shape: (B, L1, num_heads, d_k)

        return new_target_representation


    def single_head_attention_binder(self, target_representation, binder_representation, edge_representation):
        # Update target node representation
        # pdb.set_trace()
        B, L1, num_heads, d_k = target_representation.size()
        L2 = binder_representation.size(1)
        d_edge_head = edge_representation.size(-1)

        hi = target_representation.unsqueeze(2)  # shape: (B, L1, 1, num_heads, d_k)
        hj = binder_representation.unsqueeze(1)  # shape: (B, 1, L2, num_heads, d_k)
        edge_representation = edge_representation.transpose(1,2)

        # Concatenate hi, hj, edge_representation, and diff_i
        hi_hj_concat = torch.cat([
            hi.expand(-1, -1, L2, -1, -1), 
            hj.expand(-1, L1, -1, -1, -1), 
            edge_representation, 
        ], dim=-1)  # shape: (B, L1, L2, num_heads, 2*d_k + d_edge_head)

        # Calculate attention scores
        attention_scores = self.attn_linear_binder(hi_hj_concat).squeeze(-1)  # shape: (B, L1, L2, num_heads)
        attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=2)  # shape: (B, L1, L2, num_heads)

        # Aggregating features correctly along the L2 dimension
        binder_representation_Wh = self.Wh(binder_representation)  # shape: (B, L2, num_heads, d_k)
        binder_representation_Wh = binder_representation_Wh.permute(0, 2, 1, 3)  # shape: (B, num_heads, L2, d_k)
        
        aggregated_features = torch.matmul(attention_probs.permute(0, 3, 1, 2), binder_representation_Wh)  # shape: (B, num_heads, L1, d_k)
        aggregated_features = aggregated_features.permute(0, 2, 1, 3)  # shape: (B, L1, num_heads, d_k)

        # Update target representation
        new_target_representation = target_representation + self.leaky_relu(aggregated_features)  # shape: (B, L1, num_heads, d_k)

        return new_target_representation


    def single_head_attention_edge(self, target_representation, binder_representation, edge_representation, diff_vec):
        # Update edge representation
        # pdb.set_trace()
        B, L1, num_heads, d_k = target_representation.size()
        L2 = binder_representation.size(1)
        d_edge_head = edge_representation.size(-1)
        d_diff_head = diff_vec.size(-1)

        hi = target_representation.unsqueeze(2)  # shape: (B, L1, 1, num_heads, d_k)
        hj = binder_representation.unsqueeze(1)  # shape: (B, 1, L2, num_heads, d_k)
        diff_i = diff_vec.unsqueeze(2)  # shape: (B, L1, 1, num_heads, d_diff_head)

        hi_hj_concat = torch.cat([edge_representation, 
                                  hi.expand(-1, -1, L2, -1, -1), 
                                  hj.expand(-1, L1, -1, -1, -1), 
                                  diff_i.expand(-1, -1, L2, -1, -1)], dim=-1)  # shape: (B, L1, L2, num_heads, 2*d_k + d_edge_head + d_diff_head)

        new_edge_representation = self.edge_linear(hi_hj_concat)  # shape: (B, L1, L2, num_heads, d_edge_head)

        return new_edge_representation
