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


def main():

    data = pd.read_csv('/home/tc415/muPPIt/dataset/pep_prot/pep_prot_test.csv')

    print(len(data))

    positives = data['Binder'].tolist()
    anchors = data['Target'].tolist()
    binding_sites = data['Motif'].tolist()

    # We should plus 1 because there will be a start token after embedded by ESM-2
    binding_sites = [binding_site.split(',') for binding_site in binding_sites]
    binding_sites = [[int(site) + 1 for site in binding_site] for binding_site in binding_sites]

    train_anchor_dataset = np.array(anchors)
    train_positive_dataset = np.array(positives)
    train_binding_dataset = binding_sites

    # Create an instance of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize the TripletDataset
    train_dataset = TripletDataset(train_anchor_dataset, train_positive_dataset, train_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)
    train_prebatched_data_dict = {
        'anchors': [batch[0] for batch in train_dataset.triplets],
        'positives': [batch[1] for batch in train_dataset.triplets],
        'binding_site': [batch[2] for batch in train_dataset.triplets]
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/pep_prot_test')


if __name__ == "__main__":
    main()
