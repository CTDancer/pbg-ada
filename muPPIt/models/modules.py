import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, node_representation, edge_representation):
        # node_representation: (B, L, d_node)
        # edge_representation: (B, L, L, d_edge)

        B, L, d_node = node_representation.size()
        d_edge = edge_representation.size(-1)
        
        # Multi-head projection
        node_proj = self.Wn(node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj = self.We(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        
        head_outputs_node = []
        head_outputs_edge = []
        
        # Node representation update
        for head in range(self.num_heads):
            head_node_rep = node_proj[:, :, head, :]
            head_edge_rep = edge_proj[:, :, :, head, :]
            
            head_new_node_rep = self.single_head_attention_node(head_node_rep, head_edge_rep)
            
            head_outputs_node.append(head_new_node_rep)

        concatenated_node_rep = torch.cat(head_outputs_node, dim=-1)  # Shape: (B, L, num_heads * d_k)
        new_node_representation = self.out_proj_node(concatenated_node_rep)

        # Edge representation update
        node_proj_2 = self.Wn_2(new_node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj_2 = self.We_2(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        
        for head in range(self.num_heads):
            head_node_rep = node_proj_2[:, :, head, :]
            head_edge_rep = edge_proj_2[:, :, :, head, :]

            head_new_edge_rep = self.single_head_attention_edge(head_node_rep, head_edge_rep)
            head_outputs_edge.append(head_new_edge_rep)

        concatenated_edge_rep = torch.cat(head_outputs_edge, dim=-1)  # Shape: (B, L, L, num_heads * d_edge_head)
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)
        
        return new_node_representation, new_edge_representation

    def single_head_attention_node(self, node_representation, edge_representation):
        # Update node representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        
        new_node_representation = torch.zeros_like(node_representation) # Shape: (B, L, d_k)
        
        for i in range(L):
            hi = node_representation[:, i, :]  # shape: (B, d_k)
            
            attention_scores = []
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                
                concat_features = torch.cat([hi, hj, eij], dim=-1)  # shape: (B, 2*d_k + d_edge_head)
                attention_score = self.attn_linear(concat_features).squeeze(-1)  # shape: (B,)
                attention_scores.append(attention_score)
            
            # Convert attention scores to probabilities
            attention_scores = torch.stack(attention_scores, dim=-1)  # shape: (B, L-1)
            attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=-1)  # shape: (B, L-1)
            
            aggregated_features = torch.zeros_like(hi)  # shape: (B, d_k)
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                attention_prob = attention_probs[:, j-1]  # shape: (B,) (skip self)
                
                aggregated_features += attention_prob.unsqueeze(-1) * self.Wh(hj)  # shape: (B, d_k)
            
            new_hi = hi + self.leaky_relu(aggregated_features)  # shape: (B, d_k)
            new_node_representation[:, i, :] = new_hi
        
        return new_node_representation
        

    def single_head_attention_edge(self, node_representation, edge_representation):
        # Update edge representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        new_edge_representation = torch.zeros_like(edge_representation) # Shape: (B, L, L, d_edge_head)

        for i in range(L):
            for j in range(L):
                if i == j:
                    continue

                hi = node_representation[:, i, :]  # shape: (B, d_k)
                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                
                concat_features = torch.cat([eij, hi, hj], dim=-1)  # shape: (B, 2*d_k + d_edge_head)
                new_eij = self.edge_linear(concat_features)  # shape: (B, d_edge_head)
                
                new_edge_representation[:, i, j, :] = new_eij
        
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

    def forward(self, node_representation, edge_representation, diff_vec):
        # node_representation: (B, L, d_node)
        # edge_representation: (B, L, L, d_edge)

        B, L, d_node = node_representation.size()
        d_edge = edge_representation.size(-1)
        
        # Multi-head projection
        node_proj = self.Wn(node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj = self.We(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        diff_proj = self.Wd(diff_vec).view(B, L, self.num_heads, self.d_diff_head)   # (B, L, num_heads, d_diff_head)

        head_outputs_node = []
        head_outputs_edge = []
        
        # Node representation update
        for head in range(self.num_heads):
            head_node_rep = node_proj[:, :, head, :]
            head_edge_rep = edge_proj[:, :, :, head, :]
            diff_rep = diff_proj[:, :, head, :]
            
            head_new_node_rep = self.single_head_attention_node(head_node_rep, head_edge_rep, diff_rep)
            
            head_outputs_node.append(head_new_node_rep)

        concatenated_node_rep = torch.cat(head_outputs_node, dim=-1)  # Shape: (B, L, num_heads * d_k)
        new_node_representation = self.out_proj_node(concatenated_node_rep)

        # Edge representation update
        node_proj_2 = self.Wn_2(new_node_representation).view(B, L, self.num_heads, self.d_k)  # (B, L, num_heads, d_k)
        edge_proj_2 = self.We_2(edge_representation).view(B, L, L, self.num_heads, self.d_edge_head)  # (B, L, L, num_heads, d_edge_head)
        diff_proj_2 = self.Wd_2(diff_vec).view(B, L, L, self.num_heads, self.d_diff_head)   # (B, L, num_heads, d_diff_head)

        for head in range(self.num_heads):
            head_node_rep = node_proj_2[:, :, head, :]
            head_edge_rep = edge_proj_2[:, :, :, head, :]
            diff_rep = diff_proj_2[:, :, head, :]

            head_new_edge_rep = self.single_head_attention_edge(head_node_rep, head_edge_rep, diff_rep)
            head_outputs_edge.append(head_new_edge_rep)

        concatenated_edge_rep = torch.cat(head_outputs_edge, dim=-1)  # Shape: (B, L, L, num_heads * d_edge_head)
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)
        
        return new_node_representation, new_edge_representation

    def single_head_attention_node(self, node_representation, edge_representation, diff_vec):
        # Update node representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        
        new_node_representation = torch.zeros_like(node_representation) # Shape: (B, L, d_k)
        
        for i in range(L):
            hi = node_representation[:, i, :]  # shape: (B, d_k)
            diff_i = diff_vec[:, i, :]    # shape: (B, d_diff_head)
            
            attention_scores = []
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                diff_j = diff_vec[:, j, :]
                
                concat_features = torch.cat([hi, hj, eij, diff_i, diff_j], dim=-1)  # shape: (B, 2*d_k + d_edge_head + 2*d_diff_head)
                attention_score = self.attn_linear(concat_features).squeeze(-1)  # shape: (B,)
                attention_scores.append(attention_score)
            
            # Convert attention scores to probabilities
            attention_scores = torch.stack(attention_scores, dim=-1)  # shape: (B, L-1)
            attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=-1)  # shape: (B, L-1)
            
            aggregated_features = torch.zeros_like(hi)  # shape: (B, d_k)
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                attention_prob = attention_probs[:, j-1]  # shape: (B,) (skip self)
                
                aggregated_features += attention_prob.unsqueeze(-1) * self.Wh(hj)  # shape: (B, d_k)
            
            new_hi = hi + self.leaky_relu(aggregated_features)  # shape: (B, d_k)
            new_node_representation[:, i, :] = new_hi
        
        return new_node_representation
        

    def single_head_attention_edge(self, node_representation, edge_representation, diff_vec):
        # Update edge representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        new_edge_representation = torch.zeros_like(edge_representation) # Shape: (B, L, L, d_edge_head)

        for i in range(L):
            for j in range(L):
                if i == j:
                    continue

                hi = node_representation[:, i, :]  # shape: (B, d_k)
                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                diff_i = diff_vec[:, i, :]
                diff_j = diff_vec[:, j, :]
                
                concat_features = torch.cat([eij, hi, hj, diff_i, diff_j], dim=-1)  # shape: (B, 2*d_k + d_edge_head + 2*d_diff_head)
                new_eij = self.edge_linear(concat_features)  # shape: (B, d_edge_head)
                
                new_edge_representation[:, i, j, :] = new_eij
        
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

    def forward(self, target_representation, binder_representation, edge_representation, diff_vec):
        B, L1, d_node = target_representation.size()
        L2 = binder_representation.size()[1]
        d_edge = edge_representation.size(-1)
        
        # Multi-head projection
        target_proj = self.Wn(target_representation).view(B, L1, self.num_heads, self.d_k)  
        binder_proj = self.Wn(binder_representation).view(B, L2, self.num_heads, self.d_k)          
        edge_proj = self.We(edge_representation).view(B, L1, L2, self.num_heads, self.d_edge_head)  
        diff_proj = self.Wd(diff_vec).view(B, L1, self.num_heads, self.d_diff_head)   

        head_outputs_target = []
        head_outputs_binder = []
        head_outputs_edge = []
        
        # Edge representation update
        for head in range(self.num_heads):
            head_target_rep = target_proj[:, :, head, :]
            head_binder_rep = binder_proj[:, :, head, :]
            head_edge_rep = edge_proj[:, :, :, head, :]
            diff_rep = diff_proj[:, :, head, :]

            head_new_edge_rep = self.single_head_attention_edge(head_target_rep, head_binder_rep, head_edge_rep, diff_rep)
            head_outputs_edge.append(head_new_edge_rep)

        concatenated_edge_rep = torch.cat(head_outputs_edge, dim=-1)  
        new_edge_representation = self.out_proj_edge(concatenated_edge_rep)

        # Node representation update
        target_proj_2 = self.Wn_2(target_representation).view(B, L1, self.num_heads, self.d_k)  
        binder_proj_2 = self.Wn_2(binder_representation).view(B, L2, self.num_heads, self.d_k)  
        edge_proj_2 = self.We_2(new_edge_representation).view(B, L1, L2, self.num_heads, self.d_edge_head)  
        diff_proj_2 = self.Wd_2(diff_vec).view(B, L1, self.num_heads, self.d_diff_head)   
        
        for head in range(self.num_heads):
            head_target_rep = target_proj_2[:, :, head, :]
            head_edge_rep = edge_proj_2[:, :, :, head, :]
            head_binder_rep = binder_proj[:, :, head, :]
            diff_rep = diff_proj_2[:, :, head, :]
            
            head_new_target_rep = self.single_head_attention_target(head_target_rep, head_binder_rep, head_edge_rep, diff_rep)
            head_new_binder_rep = self.single_head_attention_binder(head_target_rep, head_binder_rep, head_edge_rep)
            
            head_outputs_target.append(head_new_target_rep)
            head_outputs_binder.append(head_new_binder_rep)

        concatenated_target_rep = torch.cat(head_outputs_target, dim=-1)  
        new_target_representation = self.out_proj_node(concatenated_target_rep)

        concatenated_binder_rep = torch.cat(head_outputs_binder, dim=-1)  
        new_binder_representation = self.out_proj_node(concatenated_binder_rep)

        return new_target_representation, new_binder_representation, new_edge_representation

    def single_head_attention_target(self, node_representation, edge_representation, diff_vec):
        # Update node representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        
        new_node_representation = torch.zeros_like(node_representation) # Shape: (B, L, d_k)
        
        for i in range(L):
            hi = node_representation[:, i, :]  # shape: (B, d_k)
            diff_i = diff_vec[:, i, :]    # shape: (B, d_diff_head)
            
            attention_scores = []
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                
                concat_features = torch.cat([hi, hj, eij, diff_i], dim=-1)  # shape: (B, 2*d_k + d_edge_head + d_diff_head)
                attention_score = self.attn_linear_target(concat_features).squeeze(-1)  # shape: (B,)
                attention_scores.append(attention_score)
            
            # Convert attention scores to probabilities
            attention_scores = torch.stack(attention_scores, dim=-1)  # shape: (B, L-1)
            attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=-1)  # shape: (B, L-1)
            
            aggregated_features = torch.zeros_like(hi)  # shape: (B, d_k)
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                attention_prob = attention_probs[:, j-1]  # shape: (B,) (skip self)
                
                aggregated_features += attention_prob.unsqueeze(-1) * self.Wh(hj)  # shape: (B, d_k)
            
            new_hi = hi + self.leaky_relu(aggregated_features)  # shape: (B, d_k)
            new_node_representation[:, i, :] = new_hi
        
        return new_node_representation

    def single_head_attention_binder(self, node_representation, edge_representation):
        # Update node representation
        B, L, d_k = node_representation.size()
        d_edge_head = edge_representation.size(-1)
        
        new_node_representation = torch.zeros_like(node_representation) # Shape: (B, L, d_k)
        
        for i in range(L):
            hi = node_representation[:, i, :]  # shape: (B, d_k)
            
            attention_scores = []
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                
                concat_features = torch.cat([hi, hj, eij], dim=-1)  # shape: (B, 2*d_k + d_edge_head)
                attention_score = self.attn_linear_binder(concat_features).squeeze(-1)  # shape: (B,)
                attention_scores.append(attention_score)
            
            # Convert attention scores to probabilities
            attention_scores = torch.stack(attention_scores, dim=-1)  # shape: (B, L-1)
            attention_probs = F.softmax(self.leaky_relu(attention_scores), dim=-1)  # shape: (B, L-1)
            
            aggregated_features = torch.zeros_like(hi)  # shape: (B, d_k)
            for j in range(L):
                if i == j:
                    continue

                hj = node_representation[:, j, :]  # shape: (B, d_k)
                attention_prob = attention_probs[:, j-1]  # shape: (B,) (skip self)
                
                aggregated_features += attention_prob.unsqueeze(-1) * self.Wh(hj)  # shape: (B, d_k)
            
            new_hi = hi + self.leaky_relu(aggregated_features)  # shape: (B, d_k)
            new_node_representation[:, i, :] = new_hi
        
        return new_node_representation

    def single_head_attention_edge(self, target_representation, binder_representation, edge_representation, diff_vec):
        # Update edge representation
        B, L1, d_k = target_representation.size()
        L2 = binder_representation.size()[1]
        d_edge_head = edge_representation.size(-1)
        new_edge_representation = torch.zeros_like(edge_representation) # Shape: (B, L, L, d_edge_head)

        for i in range(L1):
            for j in range(L2):
                if i == j:
                    continue

                hi = target_representation[:, i, :]  # shape: (B, d_k)
                hj = binder_representation[:, j, :]  # shape: (B, d_k)
                eij = edge_representation[:, i, j, :]  # shape: (B, d_edge_head)
                diff_i = diff_vec[:, i, :]
                
                concat_features = torch.cat([eij, hi, hj, diff_i], dim=-1)  # shape: (B, 2*d_k + d_edge_head + d_diff_head)
                new_eij = self.edge_linear(concat_features)  # shape: (B, d_edge_head)
                
                new_edge_representation[:, i, j, :] = new_eij
        
        return new_edge_representation
        