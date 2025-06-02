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
    def __init__(self, anchors, positives, negatives, binding_sites, tokenizer, max_sequence_length=40000):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
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
        for anchor, positive, negative, binding_site in zip(self.anchors, self.positives, self.negatives,
                                                            self.binding_sites):
            anchor_tokens = self.tokenizer(anchor, return_tensors='pt', padding=True, truncation=True,
                                           max_length=self.max_sequence_length)
            positive_tokens = self.tokenizer(positive, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)
            negative_tokens = self.tokenizer(negative, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)

            # mask out the first and last tokens due to being <bos> and <eos>
            anchor_tokens['attention_mask'][0][0] = 0
            anchor_tokens['attention_mask'][0][-1] = 0
            positive_tokens['attention_mask'][0][0] = 0
            positive_tokens['attention_mask'][0][-1] = 0
            negative_tokens['attention_mask'][0][0] = 0
            negative_tokens['attention_mask'][0][-1] = 0

            self.triplets.append((anchor_tokens, positive_tokens, negative_tokens, binding_site))
            # pdb.set_trace()
        return self.triplets


def main(args):

    data = pd.read_csv(args.dataset_pth)

    print(len(data))

    negatives = data['mutTarget'].tolist()
    positives = data['Binder'].tolist()
    anchors = data['Target'].tolist()
    binding_sites = data['mutTarget_motifs'].tolist()

    # We should plus 1 because there will be a start token after embedded by ESM-2
    binding_sites = [ast.literal_eval(binding_site) for binding_site in binding_sites]
    binding_sites = [[int(site.split('_')[1]) + 1 for site in binding_site] for binding_site in binding_sites]

    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

    train_index = train_data.index.to_numpy()
    val_index = val_data.index.to_numpy()
    test_index = test_data.index.to_numpy()

    train_anchor_dataset = np.array(anchors)[train_index]
    train_negative_dataset = np.array(negatives)[train_index]
    train_positive_dataset = np.array(positives)[train_index]
    train_binding_dataset = [binding_sites[i] for i in train_index]

    val_anchor_dataset = np.array(anchors)[val_index]
    val_negative_dataset = np.array(negatives)[val_index]
    val_positive_dataset = np.array(positives)[val_index]
    val_binding_dataset = [binding_sites[i] for i in val_index]

    test_anchor_dataset = np.array(anchors)[test_index]
    test_negative_dataset = np.array(negatives)[test_index]
    test_positive_dataset = np.array(positives)[test_index]
    test_binding_dataset = [binding_sites[i] for i in test_index]

    # Create an instance of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize the TripletDataset
    train_dataset = TripletDataset(train_anchor_dataset, train_positive_dataset, train_negative_dataset, train_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)
    val_dataset = TripletDataset(val_anchor_dataset, val_positive_dataset, val_negative_dataset, val_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)
    test_dataset = TripletDataset(test_anchor_dataset, test_positive_dataset, test_negative_dataset, test_binding_dataset, tokenizer=tokenizer, max_sequence_length=50000)

    train_prebatched_data_dict = {
        'anchors': [batch[0] for batch in train_dataset.triplets],
        'positives': [batch[1] for batch in train_dataset.triplets],
        # 'negatives': [batch[2] for batch in train_dataset.triplets],
        'binding_site': [batch[3] for batch in train_dataset.triplets]
    }

    val_prebatched_data_dict = {
        'anchors': [batch[0] for batch in val_dataset.triplets],
        'positives': [batch[1] for batch in val_dataset.triplets],
        # 'negatives': [batch[2] for batch in val_dataset.triplets],
        'binding_site': [batch[3] for batch in val_dataset.triplets]
    }

    test_prebatched_data_dict = {
        'anchors': [batch[0] for batch in test_dataset.triplets],
        'positives': [batch[1] for batch in test_dataset.triplets],
        # 'negatives': [batch[2] for batch in test_dataset.triplets],
        'binding_site': [batch[3] for batch in test_dataset.triplets]
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'train', 'correct_train_dataset_drop_500'))

    val_hf_dataset = HFDataset.from_dict(val_prebatched_data_dict)
    val_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'val', 'correct_val_dataset_drop_500'))

    test_hf_dataset = HFDataset.from_dict(test_prebatched_data_dict)
    test_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'test', 'correct_test_dataset_drop_500'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dataset_pth", default='/home/tc415/moPPIt/dataset/pretrain_dataset.csv', help="The path for the dataset to be processed")
    parser.add_argument("-output_dir", default='/home/tc415/moPPIt/dataset/', help="The directory for storing the processed huggingface dataset")
    args =  parser.parse_args()
    main(args)
