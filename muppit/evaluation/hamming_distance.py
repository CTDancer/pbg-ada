import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import pdb

def compute_average_hamming_distance(gen_peptides, train_peptides, tokenizer, alphabet):
    # 1) Encode peptides into torch tensors of shape (N, L) and (M, L).
    # pdb.set_trace()
    gen_tensor = tokenizer(gen_peptides, return_tensors='pt')['input_ids']
    train_tensor = tokenizer(train_peptides, return_tensors='pt')['input_ids']

    # 2) Compute element-wise equality. 
    #    - gen_tensor[:, None, :] => (N, 1, L)
    #    - train_tensor[None, :, :] => (1, M, L)
    #    => broadcasting => shape (N, M, L)
    same_positions = (gen_tensor[:, None, :] == train_tensor[None, :, :])

    # 3) Convert 'True'/'False' to integer mismatch count along dim=-1
    #    Hamming distance is the number of positions that are different
    hamming_matrix = (~same_positions).sum(dim=-1)  # (N, M)

    # 4) Average Hamming distance across all pairs
    avg_hamming_distance = hamming_matrix.float().mean().item()
    return avg_hamming_distance


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
alphabet = list(tokenizer.get_vocab().keys())
average_hds = {}

for length in range(6,50):
    df_samples = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/samples/step_32/{length}.csv')
    generated_peptides = df_samples['sequence'].tolist()

    df_test = pd.read_csv(f'/home/tc415/discrete-diffusion-guidance/dataset/peptide/test.csv')
    df_test = df_test[df_test['Length'] == length]
    test_peptides = df_test['Sequence'].tolist()

    average_hd = compute_average_hamming_distance(generated_peptides, test_peptides, tokenizer, alphabet)
    average_hds[length] = average_hd

print(average_hds)
# print(sum(average_hds) / len(average_hds))