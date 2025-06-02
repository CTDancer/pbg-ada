import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
from argparse import ArgumentParser
import pdb
import os


class TripletDataset(Dataset):
    def __init__(self, anchors, positives, binding_sites, tokenizer, max_sequence_length=40000):
        self.anchors = anchors
        self.positives = positives
        self.binding_sites = binding_sites
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.triplets = []
        self.precompute_triplets()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]

    def precompute_triplets(self):
        self.triplets = []
        for anchor, positive, binding_site in zip(self.anchors, self.positives, self.binding_sites):
            anchor_tokens = self.tokenizer(anchor, return_tensors='pt', padding=True, truncation=True,
                                           max_length=self.max_sequence_length)
            positive_tokens = self.tokenizer(positive, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)

            # mask out the first and last tokens due to being <bos> and <eos>
            anchor_tokens['attention_mask'][0][0] = 0
            anchor_tokens['attention_mask'][0][-1] = 0
            positive_tokens['attention_mask'][0][0] = 0
            positive_tokens['attention_mask'][0][-1] = 0

            self.triplets.append((anchor_tokens, positive_tokens, binding_site))
            # pdb.set_trace()
        return self.triplets


def main(args):

    data = pd.read_csv(args.dataset_pth)

    print(len(data))
    print(os.path.join(args.output_dir, 'test', 'correct_pepnn_biolip_test'))

    positives = data['Binder'].tolist()
    anchors = data['Target'].tolist()
    binding_sites = data['Motif'].tolist()

    # We should plus 1 because there will be a start token after embedded by ESM-2
    binding_sites = [binding_site.split(',') for binding_site in binding_sites]
    binding_sites = [[int(site.strip()) + 1 for site in binding_site] for binding_site in binding_sites]
    # pdb.set_trace()

    train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_index = train_data.index.to_numpy()
    val_index = val_data.index.to_numpy()
    test_index = test_data.index.to_numpy()

    # # Specific indices to move from train to test
    # specific_indices = [5, 33, 38, 1049, 1939, 2200, 3462, 4557, 4666, 4994, 5077, 6086, 8651, 9077]

    # # Initialize list to keep track of swaps
    # indices_to_swap = []

    # # pdb.set_trace()
    # # Find indices in train_index that need to be swapped
    # for idx in specific_indices:
    #     if idx in train_index:
    #         # Find a replacement index from test_index
    #         for replacement in test_index:
    #             # Check if the replacement index is not already swapped and not in specific_indices
    #             if (replacement not in specific_indices and 
    #                 replacement not in [swap[1] for swap in indices_to_swap]):
    #                 indices_to_swap.append((idx, replacement))
    #                 break

    # # pdb.set_trace()
    # # Update train_index and test_index with the swaps
    # for train_idx, test_idx in indices_to_swap:
    #     # Remove the train_idx from train_index
    #     train_index = np.delete(train_index, np.where(train_index == train_idx))
    #     # Add the test_idx to train_index
    #     train_index = np.append(train_index, test_idx)

    #     # Remove the test_idx from test_index
    #     test_index = np.delete(test_index, np.where(test_index == test_idx))
    #     # Add the train_idx to test_index
    #     test_index = np.append(test_index, train_idx)

    train_anchor_dataset = np.array(anchors)[train_index]
    train_positive_dataset = np.array(positives)[train_index]
    train_binding_dataset = [binding_sites[i] for i in train_index]

    # pdb.set_trace()

    # with open('test_indices.txt', 'w') as f:
    #     for i in np.sort(test_index):
    #         f.write(str(i) + '\n')

    # pdb.set_trace()

    val_anchor_dataset = np.array(anchors)[val_index]
    val_positive_dataset = np.array(positives)[val_index]
    val_binding_dataset = [binding_sites[i] for i in val_index]

    test_anchor_dataset = np.array(anchors)[test_index]
    test_positive_dataset = np.array(positives)[test_index]
    test_binding_dataset = [binding_sites[i] for i in test_index]

    num_binding_sites = sum([len(binding_site) for binding_site in train_binding_dataset])
    total = sum([len(target) for target in train_anchor_dataset])
    weight_for_binding = total / (2 * num_binding_sites)
    weight_for_non_binding = total / (2 * (total-num_binding_sites))
    print("Weights for binding and non-binding site, which will be used in the BCE Loss during finetuning: ", torch.tensor([weight_for_binding, weight_for_non_binding]))
    pdb.set_trace()

    # Create an instance of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize the TripletDataset
    train_dataset = TripletDataset(train_anchor_dataset, train_positive_dataset, train_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)
    val_dataset = TripletDataset(val_anchor_dataset, val_positive_dataset, val_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)
    test_dataset = TripletDataset(test_anchor_dataset, test_positive_dataset, test_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)

    train_prebatched_data_dict = {
        'anchors': [batch[0] for batch in train_dataset.triplets],
        'positives': [batch[1] for batch in train_dataset.triplets],
        'binding_site': [batch[2] for batch in train_dataset.triplets]
    }

    val_prebatched_data_dict = {
        'anchors': [batch[0] for batch in val_dataset.triplets],
        'positives': [batch[1] for batch in val_dataset.triplets],
        'binding_site': [batch[2] for batch in val_dataset.triplets]
    }

    test_prebatched_data_dict = {
        'anchors': [batch[0] for batch in test_dataset.triplets],
        'positives': [batch[1] for batch in test_dataset.triplets],
        'binding_site': [batch[2] for batch in test_dataset.triplets]
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'train', 'correct_pepnn_biolip_train'))

    val_hf_dataset = HFDataset.from_dict(val_prebatched_data_dict)
    val_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'val', 'correct_pepnn_biolip_val'))

    test_hf_dataset = HFDataset.from_dict(test_prebatched_data_dict)
    test_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'test', 'correct_pepnn_biolip_test'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dataset_pth", default='/home/tc415/moPPIt/dataset/pep_prot/finetune_dataset.csv', help="The path for the dataset to be processed")
    parser.add_argument("-output_dir", default='/home/tc415/moPPIt/dataset/', help="The directory for storing the processed huggingface dataset")
    args =  parser.parse_args()
    main(args)

