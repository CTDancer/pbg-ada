import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from protein_lm.tokenizer.tokenizer import PTMTokenizer
from argparse import ArgumentParser
import pdb
from tqdm import tqdm
import os
from transformers import AutoTokenizer

from collections import namedtuple
import esm
from typing import List, Union, Optional
from protein_lm.modeling.scripts.train import compute_esm_embedding, compute_saprot_embedding, load_ckpt, make_esm_input_ids
from torch.nn.utils.rnn import pad_sequence

Output = namedtuple("output", ["logits", "hidden_states"])


# class PTMMamba:
#     def __init__(self, ckpt_path, device='cuda', use_esm=True) -> None:
#         self.use_esm = use_esm
#         self._tokenizer = PTMTokenizer()
#         self._model = load_ckpt(ckpt_path, self.tokenizer, device)
#         self._device = device
#         self._model.to(device)
#         self._model.eval()
#         self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.esm_model.eval()
#
#     @property
#     def model(self) -> torch.nn.Module:
#         return self._model
#
#     @property
#     def tokenizer(self) -> PTMTokenizer:
#         return self._tokenizer
#
#     @property
#     def device(self) -> torch.device:
#         return self._device
#
#     def infer(self, seq: str) -> Output:
#         input_id = self.tokenizer(seq)
#         input_ids = torch.tensor(input_id, device=self.device).unsqueeze(0)
#         outputs = self._infer(input_ids)
#         return outputs
#
#     # @torch.no_grad()
#     # def _infer(self, input_ids):
#     #     if torch.max(input_ids) >= 33:
#     #         is_ptm = True
#     #     else:
#     #         is_ptm = False
#     #     if is_ptm:
#     #         if self.use_esm:
#     #             esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
#     #             embedding = compute_esm_embedding(
#     #                 self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
#     #             )
#     #         else:
#     #             embedding = None
#     #         outputs = self.model(input_ids, embedding=embedding)
#     #     else:
#     #         outputs = self.model(input_ids, is_ptm=False)
#     #     return outputs
#
#     @torch.no_grad()
#     def _infer(self, input_ids):
#         if self.use_esm:
#             esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
#             embedding = compute_esm_embedding(
#                 self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
#             )
#         else:
#             embedding = None
#         outputs = self.model(input_ids, embedding=embedding)
#         return outputs
#
#     def infer_batch(self, seqs: list) -> Output:
#         input_ids = self.tokenizer(seqs)
#         input_ids = pad_sequence(
#             [torch.tensor(x) for x in input_ids],
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id,
#         )
#         input_ids = torch.tensor(input_ids, device=self.device)
#         outputs = self._infer(input_ids)
#         return outputs
#
#     def __call__(self, seq: Union[str, List]) -> Output:
#         if isinstance(seq, str):
#             return self.infer(seq)
#         elif isinstance(seq, list):
#             return self.infer_batch(seq)
#         else:
#             raise ValueError("Input must be a string or a list of strings, got {}".format(type(seq)))


class PTMMamba:
    def __init__(self, ckpt_path, device='cuda', ) -> None:
        self._tokenizer = PTMTokenizer()
        train_config = torch.load(ckpt_path, map_location='cpu')['config']
        self.use_esm = train_config.get('use_esm', True)
        self.use_saprot = train_config.get('use_saprot', False)
        assert not (self.use_esm and self.use_saprot), "Only one of use_esm and use_saprot can be True"
        saprot_path = train_config.get('saprot_path', "SaProt_650M_AF2/")
        if not os.path.exists(saprot_path):
            raise ValueError(f"Invalid saprot_path {saprot_path}")
        self._model = load_ckpt(ckpt_path, self.tokenizer, device)
        self._device = device
        self._model.to(device)
        self._model.eval()

        if self.use_esm:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.esm_model.eval()
        elif self.use_saprot:
            self.saprot_tokenizer = EsmTokenizer.from_pretrained(saprot_path)
            self.saprot_model = EsmForMaskedLM.from_pretrained(saprot_path)
            self.esmfold_model = esm.pretrained.esmfold_v1().eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def tokenizer(self) -> PTMTokenizer:
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(self, seq: str) -> Output:
        input_id = self.tokenizer(seq)
        input_ids = torch.tensor(input_id, device=self.device).unsqueeze(0)
        outputs = self._infer(input_ids)
        return outputs

    @torch.no_grad()
    def _infer(self, input_ids):
        if self.use_esm:
            esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_esm_embedding(
                self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
            )
        elif self.use_saprot:
            saprot_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_saprot_embedding(
                self.saprot_tokenizer, self.saprot_model, self.esmfold_model, self.tokenizer, saprot_input_ids
            )
        else:
            embedding = None
        outputs = self.model(input_ids, embedding=embedding)
        return outputs

    def infer_batch(self, seqs: list) -> Output:
        input_ids = self.tokenizer(seqs)
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = torch.tensor(input_ids, device=self.device)
        outputs = self._infer(input_ids)
        return outputs

    def __call__(self, seq: Union[str, List]) -> Output:
        if isinstance(seq, str):
            return self.infer(seq)
        elif isinstance(seq, list):
            return self.infer_batch(seq)
        else:
            raise ValueError("Input must be a string or a list of strings, got {}".format(type(seq)))


def one_hot_encode_from_indices(indices):
    # Initialize one-hot encoding matrix
    one_hot_matrix = np.zeros((len(indices), 59), dtype=int)
    
    for idx, token_index in enumerate(indices):
        one_hot_matrix[idx, token_index] = 1
        
    return one_hot_matrix


class TripletDataset(Dataset):
    def __init__(self, binders, wts, ptms, effects, mamba, esm_model, esm_tokenizer, ptm_tokenizer):
        self.binders = binders
        self.wts = wts
        self.ptms = ptms
        self.effects = effects
        self.mamba = mamba
        self.esm = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.ptm_tokenizer = ptm_tokenizer
        self.triplets = []
        self.precompute_triplet_embeddings()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]

    def precompute_triplet_embeddings(self):
        self.triplets = []
        for binder_seq, wt_seq, ptm_seq, effect in tqdm(zip(self.binders, self.wts, self.ptms, self.effects), total=len(self.wts)):
            with torch.no_grad():
                tokenized_binder = self.esm_tokenizer(binder_seq, return_tensors='pt')['input_ids'][:, 1:-1].to('cuda')
                binder_embedding = self.esm(tokenized_binder, repr_layers=[33])["representations"][33].mean(-1)
                wt_embedding = self.mamba(wt_seq).hidden_states.mean(-1)
                ptm_embedding = self.mamba(ptm_seq).hidden_states.mean(-1)
                # tokenized_wt = self.tokenizer(wt_seq, return_tensors='pt')['input_ids'][:, 1:-1].to('cuda')
                # wt_embedding = self.esm(tokenized_wt, repr_layers=[33])["representations"][33].mean(-1)

                # tokenized_ptm = self.tokenizer(ptm_seq, return_tensors='pt')['input_ids'][:, 1:-1].to('cuda')
                # ptm_embedding = self.esm(tokenized_ptm, repr_layers=[33])["representations"][33].mean(-1)
                # pdb.set_trace()
                wt_embedding = one_hot_encode_from_indices(self.ptm_tokenizer.encode(wt_seq)).mean(-1)
                ptm_embedding = one_hot_encode_from_indices(self.ptm_tokenizer.encode(ptm_seq)).mean(-1)
            self.triplets.append((wt_embedding, ptm_embedding, effect, binder_embedding))
        return self.triplets


def main(args):
    cluster_df = pd.read_csv(args.cluster_file, sep="\t", header=None, names=["rep_seq", "seq_id"])
    cluster_df["original_index"] = cluster_df["seq_id"].apply(lambda x: int(x[3:]))

    train_index, test_index = train_test_split(cluster_df["original_index"].unique(), test_size=0.2, random_state=42)
    val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=42)

    data = pd.read_csv(args.dataset_pth)

    print(len(data))

    binders = []
    wts = []
    ptms = []
    effects = []

    for i, row in data.iterrows():
        binder_seq = row['binder_seq']
        wt_seq = row['wt_seq']
        ptm_seq = row['ptm_seq']
        effect = row['Effect'].lower()

        binders.append(binder_seq)
        wts.append(wt_seq)  
        ptms.append(ptm_seq)
        if effect == 'enhance' or effect == 'induce':
            effects.append(1)
        elif effect == 'inhibit':
            effects.append(0)
        else:
            raise NotImplementedError

    train_binder_dataset = np.array(binders)[train_index]
    train_wt_dataset = np.array(wts)[train_index]
    train_ptm_dataset = np.array(ptms)[train_index]
    train_effect_dataset = np.array(effects)[train_index]

    val_binder_dataset = np.array(binders)[val_index]
    val_wt_dataset = np.array(wts)[val_index]
    val_ptm_dataset = np.array(ptms)[val_index]
    val_effect_dataset = np.array(effects)[val_index]

    test_binder_dataset = np.array(binders)[test_index]
    test_wt_dataset = np.array(wts)[test_index]
    test_ptm_dataset = np.array(ptms)[test_index]
    test_effect_dataset = np.array(effects)[test_index]

    ptm_tokenizer = PTMTokenizer()
    ckpt_path = "/home/tc415/ptm-mamba/ckpt/ptm_mamba_saprot/best.ckpt"
    pdb.set_trace()
    mamba = PTMMamba(ckpt_path, device='cuda')
    esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.to('cuda')
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize the TripletDataset
    train_dataset = TripletDataset(train_binder_dataset, train_wt_dataset, train_ptm_dataset, train_effect_dataset, mamba, esm_model, esm_tokenizer, ptm_tokenizer)
    val_dataset = TripletDataset(val_binder_dataset, val_wt_dataset, val_ptm_dataset, val_effect_dataset, mamba, esm_model, esm_tokenizer, ptm_tokenizer)
    test_dataset = TripletDataset(test_binder_dataset, test_wt_dataset, test_ptm_dataset, test_effect_dataset, mamba, esm_model, esm_tokenizer, ptm_tokenizer)

    train_prebatched_data_dict = {
        'wt': [batch[0] for batch in train_dataset.triplets],
        'ptm': [batch[1] for batch in train_dataset.triplets],
        'effect': [batch[2] for batch in train_dataset.triplets],
        'binder': [batch[3] for batch in train_dataset.triplets],
    }

    val_prebatched_data_dict = {
        'wt': [batch[0] for batch in val_dataset.triplets],
        'ptm': [batch[1] for batch in val_dataset.triplets],
        'effect': [batch[2] for batch in val_dataset.triplets],
        'binder': [batch[3] for batch in val_dataset.triplets],
    }

    test_prebatched_data_dict = {
        'wt': [batch[0] for batch in test_dataset.triplets],
        'ptm': [batch[1] for batch in test_dataset.triplets],
        'effect': [batch[2] for batch in test_dataset.triplets],
        'binder': [batch[3] for batch in test_dataset.triplets],
    }

    # Convert the dictionary to a HuggingFace Dataset
    train_hf_dataset = HFDataset.from_dict(train_prebatched_data_dict)
    train_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'train', 'discriminator_onehot'))

    val_hf_dataset = HFDataset.from_dict(val_prebatched_data_dict)
    val_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'val', 'discriminator_onehot'))

    test_hf_dataset = HFDataset.from_dict(test_prebatched_data_dict)
    test_hf_dataset.save_to_disk(os.path.join(args.output_dir, 'test', 'discriminator_onehot'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dataset_pth", default="/home/tc415/ptm-mamba/dataset/ptm_correct.csv",
                        help="The path for the dataset to be processed")
    parser.add_argument("-output_dir", default='/home/tc415/ptm-mamba/dataset',
                        help="The directory for storing the processed huggingface dataset")
    parser.add_argument("-cluster_file", default="/home/tc415/ptm-mamba/dataset/mmseqs_output_ptm_correct/clusters.tsv",
                        help="The path for the mmseqs2 clustering result")
    args = parser.parse_args()
    main(args)

