import pandas as pd
from transformers import AutoTokenizer
from models.graph import NodeGraph, EdgeGraph
import esm
from datasets import Dataset as HFDataset
import pdb
from tqdm import tqdm
import torch

df_train = pd.read_csv('/home/tc415/muPPIt/dataset/train_ppiref.csv')
df_val = pd.read_csv('/home/tc415/muPPIt/dataset/val_ppiref.csv')
df_test = pd.read_csv('/home/tc415/muPPIt/dataset/test_ppiref.csv')

df = pd.concat([df_train, df_val, df_test], ignore_index=True)

binders = df['Binder'].tolist()
wts = df['Wt'].tolist()
muts = df['Mutant'].tolist()

sequences = list(set(binders + wts + muts))
sequences = [seq for seq in sequences if len(seq)<=200 and 'X' not in seq and 'B' not in seq]
print(len(sequences))

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

seq_dataset = {
    'Sequence': [],
    'Graph': [],
    'Edge': []
}

nodegraph = NodeGraph(8)
edgegraph = EdgeGraph()

with torch.no_grad():
    for seq in tqdm(sequences):
        sequence = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=500)
        seq_tokens = sequence['input_ids'][:, 1:-1]    # shape: 1*L

        # pdb.set_trace()
        
        node = nodegraph(seq_tokens, model, alphabet) # shape: 1*L*1296
        edge = edgegraph(seq_tokens, model) # shape: 1 * L * L

        pdb.set_trace()

        seq_dataset['Sequence'].append(seq_tokens)
        seq_dataset['Graph'].append(node)
        seq_dataset['Edge'].append(edge)


# Convert seq_dataset to a DataFrame
df_seq = pd.DataFrame(seq_dataset)

# Shuffle the dataset
df_seq = df_seq.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split indices
train_size = int(0.8 * len(df_seq))
val_size = int(0.1 * len(df_seq))
test_size = len(df_seq) - train_size - val_size

# Split the dataset
train_df = df_seq[:train_size]
val_df = df_seq[train_size:train_size + val_size]
test_df = df_seq[train_size + val_size:]

# Convert the splits back to dictionaries (if needed)
train_set = {
    'Sequence': train_df['Sequence'].tolist(),
    'Graph': train_df['Graph'].tolist(),
    'Edge': train_df['Edge'].tolist(),
}

val_set = {
    'Sequence': val_df['Sequence'].tolist(),
    'Graph': val_df['Graph'].tolist(),
    'Edge': val_df['Edge'].tolist(),
}

test_set = {
    'Sequence': test_df['Sequence'].tolist(),
    'Graph': test_df['Graph'].tolist(),
    'Edge': test_df['Edge'].tolist(),
}

# Convert to Hugging Face Datasets
train_hf_dataset = HFDataset.from_dict(train_set)
val_hf_dataset = HFDataset.from_dict(val_set)
test_hf_dataset = HFDataset.from_dict(test_set)

# Save the datasets to disk
train_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/train/decoder_2')
val_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/val/decoder_2')
test_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/test/decoder_2')
