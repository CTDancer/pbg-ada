import torch
import pandas as pd
from transformers import AutoTokenizer
import pdb

def compute_empirical_shannon_entropy(generated_peptides, alphabet, tokenizer):
    """
    Computes the empirical Shannon entropy (in bits) of the amino acid distribution
    across all generated peptides.

    Args:
        generated_peptides (list of str): A list of generated peptide sequences.
        alphabet (str): ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 
                        'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 
                        'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
        tokenizer: facebook/esm2_t33_650M_UR50D

    Returns:
        float: The Shannon entropy in bits.
    """
    # Create a frequency-count tensor
    freq = torch.zeros(len(alphabet), dtype=torch.float32)

    # Count occurrences of each amino acid
    for peptide in generated_peptides:
        tokens = tokenizer.decode(tokenizer.encode(peptide)).split(' ')[1:-1]
        for token in tokens:
            idx = alphabet.index(token)  
            freq[idx] += 1

    # Convert frequency counts to probabilities
    total_count = freq.sum()
    if total_count == 0:
        # Handle edge case: no data
        return 0.0

    probs = freq / total_count

    # Compute Shannon entropy = - sum_i p_i log2(p_i)
    # Add a small epsilon to avoid log(0). 
    eps = 1e-12
    entropy = -(probs * torch.log2(probs + eps)).sum().item()

    return entropy

entropy_values = []
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
alphabet = list(tokenizer.get_vocab().keys())

# df = pd.read_csv('/home/tc415/discrete-diffusion-guidance/dataset/peptide/test.csv')
# generated_peptides = df['Sequence'].tolist()
# entropy_value = compute_empirical_shannon_entropy(generated_peptides, alphabet, tokenizer)
# print(entropy_value)

# for i in range(6,50):
#     df = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/samples/{i}.csv')
#     generated_peptides = df['sequence'].tolist()

#     entropy_value = compute_empirical_shannon_entropy(generated_peptides, alphabet, tokenizer)
#     entropy_values.append(entropy_value)

# print(entropy_values)
# print(sum(entropy_values)/len(entropy_values))

entropy_values = {}

for length in range(6,50):
    df_samples = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/samples/step_32/{length}.csv')
    generated_peptides = df_samples['sequence'].tolist()

    df_test = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/dataset/peptide/test.csv')
    df_test = df_test[df_test['Length'] == length]
    test_peptides = df_test['Sequence'].tolist()

    entropy_value1 = compute_empirical_shannon_entropy(generated_peptides, alphabet, tokenizer)
    entropy_value2 = compute_empirical_shannon_entropy(test_peptides, alphabet, tokenizer)
    entropy_values[length] = (entropy_value1, entropy_value2)

print(entropy_values)