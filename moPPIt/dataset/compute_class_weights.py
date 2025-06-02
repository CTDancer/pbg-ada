import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
import pdb

def main():

    data = pd.read_csv('dataset_drop_500.csv')

    print(len(data))

    binding_sites = data['mutTarget_motifs'].tolist()
    targets = data['Target'].tolist()

    # No need for padding the first position of binding sites for class weight calculations
    binding_sites = [ast.literal_eval(binding_site) for binding_site in binding_sites]
    binding_sites = [len(binding_site) for binding_site in binding_sites]
    targets = [len(seq) for seq in targets]
    pdb.set_trace()

    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

    train_index = train_data.index.to_numpy()
    print(len(train_index))
    return

    train_binding_dataset = [binding_sites[i] for i in train_index]
    train_targets = [targets[i] for i in train_index]

    num_binding_sites = sum(train_binding_dataset)
    num_total = sum(train_targets)
    num_non_binding_sites = num_total - num_binding_sites
    weight_for_binding = num_total / (2 * num_binding_sites)
    weight_for_non_binding = num_total / (2 * num_non_binding_sites)

    print(weight_for_binding, weight_for_non_binding)


if __name__ == "__main__":
    main()