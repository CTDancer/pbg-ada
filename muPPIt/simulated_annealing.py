import torch
import random
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def embedding_similarity(embedding1, embedding2):
    return -torch.norm(embedding1 - embedding2, p=2)


def mutate_sequence(sequence):
    mutated_sequence = list(sequence)
    mutation_position = random.randint(0, len(sequence) - 1)
    new_residue = random.choice(amino_acids)
    mutated_sequence[mutation_position] = new_residue
    return ''.join(mutated_sequence)


def compute_embedding(seq, tokenizer, esm_model, alphabet, nodegraph):
    tokens = tokenizer(seq, return_tensors='pt')['input_ids'][:, 1:-1]
    node_representation = nodegraph(tokens, esm_model, alphabet)
    return node_representation.squeeze(0)


def simulated_annealing(initial_sequence, target_embedding, esm_model, alphabet, nodegraph, max_iters=5000, initial_temp=1.0, cooling_rate=0.99):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    print(f"Initial Sequence: {initial_sequence}")
    current_sequence = initial_sequence
    current_embedding = compute_embedding(current_sequence, tokenizer, esm_model, alphabet, nodegraph)
    
    current_similarity = embedding_similarity(current_embedding, target_embedding)
    
    best_sequence = current_sequence
    best_similarity = current_similarity
    
    temperature = initial_temp
    
    for iteration in range(max_iters):
        new_sequence = mutate_sequence(current_sequence)
        new_embedding = compute_embedding(new_sequence, tokenizer, esm_model, alphabet, nodegraph)
        
        new_similarity = embedding_similarity(new_embedding, target_embedding)
        
        delta_similarity = new_similarity - current_similarity
        acceptance_prob = torch.exp(delta_similarity / temperature)
        
        # Decide whether to accept the new sequence
        if delta_similarity > 0 or torch.rand(1).item() < acceptance_prob.item():
            current_sequence = new_sequence
            current_similarity = new_similarity
            
            # Update best sequence if new one is better
            if current_similarity > best_similarity:
                print(f"New Sequence: {current_sequence}")
                best_sequence = current_sequence
                best_similarity = current_similarity
        
        # Cool down the temperature
        temperature *= cooling_rate
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best Similarity = {best_similarity.item()} | Temperature = {temperature}")
        
    return best_sequence, best_similarity

