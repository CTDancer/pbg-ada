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
    def __init__(self, targets, binders, binding_sites, tokenizer, max_sequence_length=40000):
        self.targets = targets
        self.binders = binders
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
        for target, binder, binding_site in zip(self.targets, self.binders, self.binding_sites):
            triplets.append((target, binder, binding_site))
        return triplets

    def get_batch_indices(self):
        sizes = [(len(target) + len(binder), i) for i, (target, binder, _) in enumerate(self.triplets)]
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
            target_batch = []
            binder_batch = []
            binding_site_batch = []

            for index in batch_indices:
                target, binder, binding_site = self.triplets[index]
                target_batch.append(target)
                binder_batch.append(binder)
                binding_site_batch.append(binding_site)

            target_tokens = self.tokenizer(target_batch, return_tensors='pt', padding=True, truncation=True,
                                           max_length=self.max_sequence_length)
            binder_tokens = self.tokenizer(binder_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_sequence_length)

            n, max_length = target_tokens['input_ids'].shape[0], target_tokens['input_ids'].shape[1]
            site = torch.zeros(n, max_length)
            for i in range(len(binding_site_batch)):
                binding_site = binding_site_batch[i]
                site[i, binding_site] = 1

            # mask out the first column because it corresponds to the start token
            target_tokens['attention_mask'][:, 0] = 0
            binder_tokens['attention_mask'][:, 0] = 0

            prebatched_data.append({
                'target_input_ids': target_tokens['input_ids'],
                'target_attention_mask': target_tokens['attention_mask'],
                'binder_input_ids': binder_tokens['input_ids'],
                'binder_attention_mask': binder_tokens['attention_mask'],
                'binding_site': site
            })

        return prebatched_data


def main():
    seed_everything(42)

    df = pd.read_csv('dataset.csv')
    data = df.drop_duplicates(subset=['Binder', 'Target', 'Binder_motifs'])
    data.reset_index(drop=True, inplace=True)

    binders = data['Binder'].tolist()
    targets = data['Target'].tolist()
    binding_sites = data['mutTarget_motifs'].tolist()
    # pdb.set_trace()

    # We should plus 1 because there will be a start token after embedded by ESM-2
    binding_sites = [ast.literal_eval(binding_site) for binding_site in binding_sites]
    binding_sites = [[int(site.split('_')[1]) + 1 for site in binding_site] for binding_site in binding_sites]

    # pdb.set_trace()
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

    train_index = train_data.index.to_numpy()
    val_index = val_data.index.to_numpy()
    test_index = test_data.index.to_numpy()

    # pdb.set_trace()

    train_target_dataset = np.array(targets)[train_index]
    train_binder_dataset = np.array(binders)[train_index]
    train_binding_dataset = [binding_sites[i] for i in train_index]

    val_target_dataset = np.array(targets)[val_index]
    val_binder_dataset = np.array(binders)[val_index]
    val_binding_dataset = [binding_sites[i] for i in val_index]

    test_target_dataset = np.array(targets)[test_index]
    test_binder_dataset = np.array(binders)[test_index]
    test_binding_dataset = [binding_sites[i] for i in test_index]

    # Create an instance of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize the TripletDataset
    train_dataset = TripletDataset(train_target_dataset, train_binder_dataset, train_binding_dataset, tokenizer=tokenizer, max_sequence_length=1000)
    val_dataset = TripletDataset(val_target_dataset, val_binder_dataset, val_binding_dataset, tokenizer=tokenizer, max_sequence_length=1000)
    test_dataset = TripletDataset(test_target_dataset, test_binder_dataset, test_binding_dataset, tokenizer=tokenizer, max_sequence_length=1000)

    # Convert the prebatched data to a dictionary with each batch as an entry
    train_prebatched_data_dict = {
        'target_input_ids': [batch['target_input_ids'].numpy() for batch in train_dataset.prebatched_data],
        'target_attention_mask': [batch['target_attention_mask'].numpy() for batch in train_dataset.prebatched_data],
        'binder_input_ids': [batch['binder_input_ids'].numpy() for batch in train_dataset.prebatched_data],
        'binder_attention_mask': [batch['binder_attention_mask'].numpy() for batch in train_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in train_dataset.prebatched_data]
    }

    val_prebatched_data_dict = {
        'target_input_ids': [batch['target_input_ids'].numpy() for batch in val_dataset.prebatched_data],
        'target_attention_mask': [batch['target_attention_mask'].numpy() for batch in val_dataset.prebatched_data],
        'binder_input_ids': [batch['binder_input_ids'].numpy() for batch in val_dataset.prebatched_data],
        'binder_attention_mask': [batch['binder_attention_mask'].numpy() for batch in val_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in val_dataset.prebatched_data]
    }
    test_prebatched_data_dict = {
        'target_input_ids': [batch['target_input_ids'].numpy() for batch in test_dataset.prebatched_data],
        'target_attention_mask': [batch['target_attention_mask'].numpy() for batch in test_dataset.prebatched_data],
        'binder_input_ids': [batch['binder_input_ids'].numpy() for batch in test_dataset.prebatched_data],
        'binder_attention_mask': [batch['binder_attention_mask'].numpy() for batch in test_dataset.prebatched_data],
        'binding_site': [batch['binding_site'].numpy() for batch in test_dataset.prebatched_data]
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk('train_raw')
    print("Finished training dataset")

    val_hf_dataset = HFDataset.from_dict(val_prebatched_data_dict)
    val_hf_dataset.save_to_disk('val_raw')
    print("Finished validation dataset")

    test_hf_dataset = HFDataset.from_dict(test_prebatched_data_dict)
    test_hf_dataset.save_to_disk('test_raw')
    print("Finished test dataset")



if __name__ == "__main__":
    main()
