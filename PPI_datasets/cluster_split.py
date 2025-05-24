import os
import subprocess
import pandas as pd
import numpy as np
import pdb

# File paths
input_csv = "/home/tc415/PPI_datasets/muPPIt_ppiref_dataset_noX.csv"
output_dir = "/home/tc415/PPI_datasets/mmseqs_output"
seq_file = os.path.join(output_dir, "sequences.fasta")
db_file = os.path.join(output_dir, "sequences_db")
cluster_file = os.path.join(output_dir, "clusters.tsv")

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Load the CSV file and extract 'Sequence' column
df = pd.read_csv(input_csv)
print(f"Total sequences: {len(df)}")

# # Write sequences to a FASTA file
# with open(seq_file, "w") as f:
#     for i, seq in enumerate(df["Sequence"]):
#         f.write(f">seq{i}\n{seq}\n")

# # Run MMseqs clustering
# subprocess.run(["mmseqs", "createdb", seq_file, db_file], check=True)
# subprocess.run([
#     "mmseqs", "cluster", db_file, os.path.join(output_dir, "clusters"), output_dir,
#     "--min-seq-id", "0.3", "--cov-mode", "0", "-c", "0.8", "--threads", "4"
# ], check=True)
# subprocess.run([
#     "mmseqs", "createtsv", db_file, db_file, os.path.join(output_dir, "clusters"), cluster_file
# ], check=True)

clusters_df = pd.read_csv(cluster_file, sep='\t', header=None, names=['query', 'target'])
clusters_df['query'] = clusters_df['query'].str.replace('seq', '').astype(int)
clusters_df['target'] = clusters_df['target'].str.replace('seq', '').astype(int)

# Group targets by query to form clusters as lists of lists
cluster_groups = clusters_df.groupby('query')['target'].apply(list).tolist()

# Shuffle clusters randomly
np.random.seed(42)
np.random.shuffle(cluster_groups)

# Calculate the total number of sequences and target sizes for each split
total_sequences = sum(len(group) for group in cluster_groups)
train_limit = int(total_sequences * 0.8)
val_limit = int(total_sequences * 0.1)
test_limit = total_sequences - train_limit - val_limit  # Remaining sequences for test

# Initialize lists to hold the sequence IDs for each split
train_ids, val_ids, test_ids = [], [], []

# Distribute clusters to train, val, and test sets
for group in cluster_groups:
    if len(train_ids) < train_limit:
        train_ids.extend(group)
    elif len(val_ids) < val_limit:
        val_ids.extend(group)
    else:
        test_ids.extend(group)

# Ensure that any remaining clusters are added to test if limits were reached early
# test_ids.extend([seq_id for group in cluster_groups if seq_id not in train_ids and seq_id not in val_ids for seq_id in group])

# Retrieve the sequences and lengths from the original DataFrame for each split
train_df = df.iloc[train_ids]
val_df = df.iloc[val_ids]
test_df = df.iloc[test_ids]

pdb.set_trace()

# Save the splits to new CSV files
train_csv = os.path.join(output_dir, "train.csv")
val_csv = os.path.join(output_dir, "val.csv")
test_csv = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("Datasets have been split and saved.")