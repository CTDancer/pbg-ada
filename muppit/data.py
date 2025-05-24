import sys
import glob
import lightning.pytorch as pl
import torch
import pickle
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from functools import partial
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
import random
import gc
import h5py
import numpy as np
from multiprocessing import Pool
torch.cuda.empty_cache()
from transformers import AutoTokenizer, EsmModel
global_tokenizer = None


def init_pool(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer


def standalone_tokenize_function(s1, extra_toks_per_seq=1):
    global global_tokenizer
    try:
        tokens_1 = global_tokenizer.encode()
        return tokens_1
    except Exception as e:
        raise ValueError(f"Error during tokenization of string {s1} : {e}")

class BatchedPPIDataset(object):
    """inspired by esm2, but instead of sorting the original sequences,
    we should really sorting based on tokenized sequences
    """

    def __init__(self, sequence_strs, tokenizer, max_sequence_length):
        self.batch_indices = None
        self.sequence_str_1 = sequence_strs['sequence_1']
        self.sequence_str_2 = sequence_strs['sequence_2']
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenized_sequences = []
        self.accumulated_length = 0
        # automatically tokenize sequences upon creation of the object
        # if need manual, change it to call object.process_all() in a separate line
        # self.tokenize_sequences()

    def tokenize_sequences_forward(self):
        prot_tuples = list(zip(self.sequence_str_1, self.sequence_str_2))

        with Pool(processes=16, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            tokenized_pairs = list(
                tqdm(pool.imap(partial(standalone_tokenize_function),
                               prot_tuples),
                     total=len(prot_tuples)))

        for tokens_1, tokens_2 in tokenized_pairs:
            seq_length = len(tokens_1) + len(tokens_2) + 3  # for both bos, eos, sep tokens
            if seq_length <= self.max_sequence_length:
                forward_sequence = [self.tokenizer.bos_id()] + tokens_1 + [self.tokenizer.piece_to_id('<sep>')] + tokens_2 + [self.tokenizer.eos_id()]
                self.tokenized_sequences.append(forward_sequence)

    def process_all(self, base_path, split_name):
        self.tokenize_sequences_forward()
        forward_batches = self.process_chunk(self.tokenized_sequences, self.get_batch_indices())
        offset = len(self.tokenized_sequences)
        self.tokenize_sequences_backward()
        backward_batches = self.process_chunk(self.tokenized_sequences, self.get_batch_indices(offset))
        self.tokenized_sequences = []
        combined_dataset = concatenate_datasets([forward_batches, backward_batches])
        #combined_dataset = forward_batches
        # shuffle the datasets overall again
        shuffled_dataset = combined_dataset.shuffle()
        self.save_checkpoint(shuffled_dataset, base_path,
                             split_name=split_name)
        return shuffled_dataset

    def process_chunk(self, tokenized_sequences, batch_indices):
        print(f'Start padding and masking for sequences {len(batch_indices)} batches')

        token_batch_fn = TokenizeBatch(self.tokenizer)
        processed_batches = [
            token_batch_fn([tokenized_sequences[i] for i in batch]) for batch
            in batch_indices]
        assert len(processed_batches) == len(batch_indices)

        # Shuffle together using a permutation
        permutation = list(torch.randperm(len(processed_batches)))
        processed_batches = [processed_batches[i] for i in permutation]

        all_attention_masks = []
        all_input_ids = []
        all_labels = []

        all_attention_masks.extend([batch['attention_mask'] for batch in processed_batches])
        all_input_ids.extend([batch['input_ids'] for batch in processed_batches])
        all_labels.extend([batch['labels'] for batch in processed_batches])

        combined_dataset = Dataset.from_dict({
            'attention_mask': all_attention_masks,
            'input_ids': all_input_ids,
            'labels': all_labels
        })
        del token_batch_fn, processed_batches, batch_indices, tokenized_sequences, all_attention_masks, all_input_ids, all_labels
        gc.collect()

        return combined_dataset

    def save_checkpoint(self, shuffled_dataset, base_path, split_name=None):
        print(f'Start generating tokens for shuffled_dataset sequences')
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f'{base_path}/{split_name}_combined_reversed_ppi_tokenized_sequences.hf'

        # clear up memory for the multiprocessing
        shuffled_dataset.save_to_disk(output_file)
        del shuffled_dataset
        print(f'successfully written {split_name} processed datasets into disc!')
        self.tokenized_sequences.clear()
        gc.collect()

    def get_batch_indices(self, offset=0, end=None):
        if end is None:
            end = len(self.tokenized_sequences)
        # list splice to isolate processing of forward and backward sequences who
        # are both stored within self.tokenized_sequences
        sizes = [(len(tokens), i) for i, tokens in enumerate(self.tokenized_sequences[offset:end])]
        sizes = [(sz, idx + offset) for sz, idx in sizes]
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
            # print('my batches is:')
            # print(batches)

        for sz, i in sizes:
            # sz already has the length of special tokens, handled at tokenization level
            # check accumulative seq length in the buffer
            if current_buf_len + sz > self.max_sequence_length:
                _flush_current_buf()
            buf.append(i)
            current_buf_len += sz
            # print('my buffer is:')
            # print(buf)

        _flush_current_buf()
        return batches


class DynamicBatchingDataset(Dataset):
    """
    Process dynamically batched datasets of Huggingface Datasets object. Need special handling since in the previous
    steps, each batch (row in the Datasets object) is already processed for per batch loading
    Usage:
    train_dataset = DynamicBatchingDataset(small_dataset_dict['train'], batch_indices_train)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False,
                        collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)
    """

    def __init__(self, dataset_dict):
        print('Initializing dataset...')
        #self.dataset_dict = dataset_dict
        self.dataset_dict = {
            'attention_mask': [torch.tensor(item) for item in dataset_dict['attention_mask']],
            'input_ids': [torch.tensor(item) for item in dataset_dict['input_ids']],
            'labels': [torch.tensor(item) for item in dataset_dict['labels']]
        }

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])  # assuming each entry in dataset_dict represents a batch

    def __getitem__(self, idx):
        # Check if idx is an integer or a list
        if isinstance(idx, int):
            return {
                'attention_mask': self.dataset_dict['attention_mask'][idx],
                'input_ids': self.dataset_dict['input_ids'][idx],
                'labels': self.dataset_dict['labels'][idx]
            }
        elif isinstance(idx, list):
            return {
                'attention_mask': [self.dataset_dict['attention_mask'][i] for i in idx],
                'input_ids': [self.dataset_dict['input_ids'][i] for i in idx],
                'labels': [self.dataset_dict['labels'][i] for i in idx]
            }   
        else:
            raise ValueError(f"Expected idx to be int or list, but got {type(idx)}")    
        
    #if isinstance(idx, int):
         #   indices = [idx]
        #else:
        #    indices = idx
        
        #attention_masks = []
        #input_ids = []
        #labels = []
        #for index in indices:
         #   attention_masks.append(torch.tensor(self.dataset_dict['attention_mask'][index]))
         #   input_ids.append(torch.tensor(self.dataset_dict['input_ids'][index]))
         #   labels.append(torch.tensor(self.dataset_dict['labels'][index]))

        #return {
         #   'attention_mask': attention_masks,
          #  'input_ids': input_ids,
          #  'labels': labels
        #}

    @staticmethod
    def collate_fn(batch, verbose=False):
        # Since DataLoader's batch_size is 1, batch[0] contains your pre-batched data
        item = batch[0]
        #if verbose:
         #   print(f"collate_fn batch shape: {item['input_ids'].shape}")

       # attention_mask = item['attention_mask']
       # input_ids = item['input_ids']
       # if verbose:
        #    print(f"collate_fn input_ids shape after indexing: {input_ids.shape}")
        #labels = item['labels']

        # These are already pre-padded, so you can directly return
        return {
            'attention_mask': item['attention_mask'],
            'input_ids': item['input_ids'],
            'labels': item['labels']
        }

import sys
import lightning.pytorch as pl
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
import sentencepiece as spm
import os
from multiprocessing import Pool
from tqdm import tqdm
import gc
import pandas as pd

def standalone_tokenize_function(sequence, tokenizer):
    try:
        tokens = tokenizer.encode(sequence)
        return tokens
    except Exception as e:
        raise ValueError(f"Error during tokenization: {e}")

class TokenizeBatch:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batches):
        print(f"Processing batch of size {len(batches)}")
        data_tokens = [torch.tensor(token_list) for token_list in batches]
        data_tokens_padded = torch.nn.utils.rnn.pad_sequence(data_tokens, batch_first=True, padding_value=self.pad_token_id)
        attention_masks = (data_tokens_padded != self.pad_token_id).long()
        labels = data_tokens_padded.clone()
        labels[data_tokens_padded == self.pad_token_id] = -100
        
        print(f"Batch processed. Shape: {data_tokens_padded.shape}")
        return {
            'input_ids': data_tokens_padded,
            'attention_mask': attention_masks,
            'labels': labels
        }
class SequenceDataset:
    def __init__(self, sequences, tokenizer, max_sequence_length):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenized_sequences = []

    def tokenize_sequences(self):
        print(f"Starting tokenization of {len(self.sequences)} sequences")
        for i, seq in enumerate(tqdm(self.sequences)):
            # ESM tokenizer handles special tokens automatically
            tokens = self.tokenizer.encode(seq)
            if len(tokens) <= self.max_sequence_length:
                self.tokenized_sequences.append(tokens)
        
            if i % 10000 == 0:
                print(f"Processed {i} sequences. Current tokenized count: {len(self.tokenized_sequences)}")
        
        print(f"Tokenization complete. Final count: {len(self.tokenized_sequences)}")

    def process_sequences(self, batch_size):
        print("Starting sequence processing")
        self.tokenize_sequences()
        
        print("Sorting sequences by length")
        lengths = [(len(seq), i) for i, seq in enumerate(self.tokenized_sequences)]
        lengths.sort()
        
        batches = []
        current_batch = []
        current_length = 0
        
        print("Creating batches")
        for length, idx in tqdm(lengths):
            if current_length + length > self.max_sequence_length or len(current_batch) == batch_size:
                if current_batch:
                    batches.append([self.tokenized_sequences[i] for i in current_batch])
                current_batch = [idx]
                current_length = length
            else:
                current_batch.append(idx)
                current_length += length
                
        if current_batch:
            batches.append([self.tokenized_sequences[i] for i in current_batch])
        
        print(f"Created {len(batches)} batches")
            
        token_batch_fn = TokenizeBatch(self.tokenizer)
        print("Processing batches")
        processed_batches = [token_batch_fn(batch) for batch in tqdm(batches)]
        
        print("Creating final dataset")
        all_attention_masks = [batch['attention_mask'] for batch in processed_batches]
        all_input_ids = [batch['input_ids'] for batch in processed_batches]
        all_labels = [batch['labels'] for batch in processed_batches]
        
        dataset = Dataset.from_dict({
            'attention_mask': all_attention_masks,
            'input_ids': all_input_ids,
            'labels': all_labels
        })
        
        print(f"Final dataset size: {len(dataset)}")
        return dataset

class PretrainSequenceDataModule(pl.LightningDataModule):
    def __init__(self,
                 input_dataset_path,
                 output_dataset_path,
                 num_workers,
                 batch_size,
                 max_sequence_length=512,
                 model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        print(f"Initializing tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.input_path = input_dataset_path
        self.output_path = output_dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
    def prepare_data(self):
        if not os.path.exists(self.output_path):
            print("Loading CSV files")
            train_df = pd.read_csv(f"{self.input_path}/train.csv")
            val_df = pd.read_csv(f"{self.input_path}/val.csv")
            test_df = pd.read_csv(f"{self.input_path}/test.csv")
            
            print("Processing training data")
            train_dataset = SequenceDataset(train_df['Sequence'].tolist(), 
                                          self.tokenizer,
                                          self.max_sequence_length)
            print("Processing validation data")
            val_dataset = SequenceDataset(val_df['Sequence'].tolist(),
                                        self.tokenizer,
                                        self.max_sequence_length)
            print("Processing test data")
            test_dataset = SequenceDataset(test_df['Sequence'].tolist(),
                                         self.tokenizer,
                                         self.max_sequence_length)
            
            processed_train = train_dataset.process_sequences(self.batch_size)
            processed_val = val_dataset.process_sequences(self.batch_size)
            processed_test = test_dataset.process_sequences(self.batch_size)
            
            print("Combining datasets")
            combined_dataset = DatasetDict({
                'train': processed_train,
                'val': processed_val,
                'test': processed_test
            })
            
            print(f"Saving dataset to {self.output_path}")
            combined_dataset.save_to_disk(self.output_path)
    
    def setup(self, stage: str):
        print("Loading processed dataset")
        dataset = load_from_disk(self.output_path)
        self.train_dataset = DynamicBatchingDataset(dataset['train'])
        self.val_dataset = DynamicBatchingDataset(dataset['val'])
        self.test_dataset = DynamicBatchingDataset(dataset['test'])
        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        print("Creating training dataloader")
        return DataLoader(self.train_dataset, 
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=DynamicBatchingDataset.collate_fn,
                        pin_memory=True)
    
    def val_dataloader(self):
        print("Creating validation dataloader")
        return DataLoader(self.val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=DynamicBatchingDataset.collate_fn,
                        pin_memory=True)
    
    def test_dataloader(self):
        print("Creating test dataloader")
        return DataLoader(self.test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=DynamicBatchingDataset.collate_fn,
                        pin_memory=True)

if __name__ == '__main__':
    dm = PretrainSequenceDataModule(
        input_dataset_path='/home/tc415/discrete-diffusion-guidance/dataset/peptide',
        output_dataset_path='/home/tc415/discrete-diffusion-guidance/dataset/tokenized_peptide',
        num_workers=8,
        batch_size=50,
        max_sequence_length=100,
        model_name="facebook/esm2_t33_650M_UR50D"
    )
    dm.prepare_data()
    dm.setup('fit')
    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()
