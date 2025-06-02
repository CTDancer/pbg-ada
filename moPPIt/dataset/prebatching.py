import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
from lightning.pytorch import seed_everything
import pdb


class TripletDataset(Dataset):
    def __init__(self, anchors, positives, negatives, binding_sites, tokenizer, max_sequence_length=40000):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.binding_sites = binding_sites
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.triplets = self.precompute_triplets()
        self.batch_indices = self.get_batch_indices()
        self.prebatched_data = self.create_prebatched_data()

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, index):
        batch = self.prebatched_data[index]
        return batch

    def precompute_triplets(self):
        triplets = []
        for anchor, positive, negative, binding_site in zip(self.anchors, self.positives, self.negatives,
                                                            self.binding_sites):
            triplets.append((anchor, positive, negative, binding_site))
        return triplets

    def get_batch_indices(self):
        sizes = [(len(anchor) + len(positive) + len(negative), i) for i, (anchor, positive, negative, _) in
                 enumerate(self.triplets)]
        sizes.sort()
        batches = []
        buf = []
        current_buf_len = 0

        def _flush_current_buf():
            nonlocal current_buf_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            current_buf_len = 0

        for sz, i in sizes:
            if current_buf_len + sz > self.max_sequence_length:
                _flush_current_buf()
            buf.append(i)
            current_buf_len += sz

        _flush_current_buf()
        return batches

    def create_prebatched_data(self):
        prebatched_data = []
        for batch_indices in self.batch_indices:
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            binding_site_batch = []

            for index in batch_indices:
                anchor, positive, negative, binding_site = self.triplets[index]
                anchor_batch.append(anchor)
                positive_batch.append(positive)
                negative_batch.append(negative)
                binding_site_batch.append(binding_site)

            anchor_tokens = self.tokenizer(anchor_batch, return_tensors='pt', padding=True, truncation=True,
                                           max_length=self.max_sequence_length)
            positive_tokens = self.tokenizer(positive_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)
            negative_tokens = self.tokenizer(negative_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)

            n, max_length = negative_tokens['input_ids'].shape[0], negative_tokens['input_ids'].shape[1]
            site = torch.zeros(n, max_length)
            for i in range(len(binding_site_batch)):
                binding_site = binding_site_batch[i]
                site[i, binding_site] = 1

            # mask out the first column because it corresponds to the start token
            anchor_tokens['attention_mask'][:, 0] = 0
            positive_tokens['attention_mask'][:, 0] = 0
            negative_tokens['attention_mask'][:, 0] = 0

            prebatched_data.append({
                'anchor_input_ids': anchor_tokens['input_ids'],
                'anchor_attention_mask': anchor_tokens['attention_mask'],
                'positive_input_ids': positive_tokens['input_ids'],
                'positive_attention_mask': positive_tokens['attention_mask'],
                'negative_input_ids': negative_tokens['input_ids'],
                'negative_attention_mask': negative_tokens['attention_mask'],
                'binding_site': site
            })

        return prebatched_data

def main():
    seed_everything(42)

    data = pd.read_csv('dataset.csv')

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

    # Convert the prebatched data to a dictionary with each batch as an entry
    train_prebatched_data_dict = {
        'anchor_input_ids': [batch['anchor_input_ids'].numpy() for batch in train_dataset.prebatched_data],
        'anchor_attention_mask': [batch['anchor_attention_mask'].numpy() for batch in train_dataset.prebatched_data],
        'positive_input_ids': [batch['positive_input_ids'].numpy() for batch in train_dataset.prebatched_data],
        'positive_attention_mask': [batch['positive_attention_mask'].numpy() for batch in train_dataset.prebatched_data],
        'negative_input_ids': [batch['negative_input_ids'].numpy() for batch in train_dataset.prebatched_data],
        'negative_attention_mask': [batch['negative_attention_mask'].numpy() for batch in train_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in train_dataset.prebatched_data]
    }

    val_prebatched_data_dict = {
        'anchor_input_ids': [batch['anchor_input_ids'].numpy() for batch in val_dataset.prebatched_data],
        'anchor_attention_mask': [batch['anchor_attention_mask'].numpy() for batch in val_dataset.prebatched_data],
        'positive_input_ids': [batch['positive_input_ids'].numpy() for batch in val_dataset.prebatched_data],
        'positive_attention_mask': [batch['positive_attention_mask'].numpy() for batch in val_dataset.prebatched_data],
        'negative_input_ids': [batch['negative_input_ids'].numpy() for batch in val_dataset.prebatched_data],
        'negative_attention_mask': [batch['negative_attention_mask'].numpy() for batch in val_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in val_dataset.prebatched_data]
    }
    test_prebatched_data_dict = {
        'anchor_input_ids': [batch['anchor_input_ids'].numpy() for batch in test_dataset.prebatched_data],
        'anchor_attention_mask': [batch['anchor_attention_mask'].numpy() for batch in test_dataset.prebatched_data],
        'positive_input_ids': [batch['positive_input_ids'].numpy() for batch in test_dataset.prebatched_data],
        'positive_attention_mask': [batch['positive_attention_mask'].numpy() for batch in test_dataset.prebatched_data],
        'negative_input_ids': [batch['negative_input_ids'].numpy() for batch in test_dataset.prebatched_data],
        'negative_attention_mask': [batch['negative_attention_mask'].numpy() for batch in test_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in test_dataset.prebatched_data]
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/train_dataset_50000')

    val_hf_dataset = HFDataset.from_dict(val_prebatched_data_dict)
    val_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/val_dataset_bind_50000')

    test_hf_dataset = HFDataset.from_dict(test_prebatched_data_dict)
    test_hf_dataset.save_to_disk('/home/tc415/muPPIt/dataset/test_dataset_bind_50000')



if __name__ == "__main__":
    main()
