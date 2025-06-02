import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import esm
from tqdm import tqdm
from protein_lm.modeling.scripts.infer import PTMMamba

res_to_id = {
    "*": 0,
    "K": 1,
    "A": 2,
    "R": 3,
    "N": 4,
    "D": 5,
    "C": 6,
    "Q": 7,
    "E": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "L": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "U": 6
}

esm_models = {
    'esm2_t48_15B_UR50D': esm.pretrained.esm2_t48_15B_UR50D,
    'esm2_t36_3B_UR50D': esm.pretrained.esm2_t36_3B_UR50D,
    'esm2_t33_650M_UR50D': esm.pretrained.esm2_t33_650M_UR50D,
    'esm2_t30_150M_UR50D': esm.pretrained.esm2_t30_150M_UR50D,
    'esm2_t12_35M_UR50D': esm.pretrained.esm2_t12_35M_UR50D,
    'esm2_t6_8M_UR50D': esm.pretrained.esm2_t6_8M_UR50D
}

def get_esm_embedding(model, batch_converter, seq, device):
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][0, 1:-1]
    return token_representations.cpu()

def get_ptm_mamba_embedding(mamba, seq):
    output = mamba(seq)
    return output.hidden_states.squeeze(0).cpu()

class Processor:
    def __init__(self, info_df, length, embed_dir, device, esm_model=None, batch_converter=None, mamba=None):
        self.info = info_df
        self.length = length
        self.embed_dir = embed_dir
        self.device = device
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.mamba = mamba

    def run(self):
        train_ls, val_ls, test_ls = [], [], []
        for _, record in self.info.iterrows():
            seq = record[f'seq_{self.length}']
            unique_id = record['unique_id'].split(';')
            uniprot, start_idx, prot_len, label = unique_id[0], int(unique_id[1]), int(unique_id[2]), int(unique_id[3])
            embed_path = f'{self.embed_dir}/{seq}.pt'
            if not os.path.exists(embed_path):
                if self.mamba:
                    emb = get_ptm_mamba_embedding(self.mamba, seq)
                else:
                    emb = get_esm_embedding(self.esm_model, self.batch_converter, seq, self.device)
                torch.save(emb, embed_path)
            else:
                emb = torch.load(embed_path)
            
            x = [res_to_id[res] for res in seq]
            x_one_hot = torch.zeros(len(x), 21)
            x_one_hot[range(len(x)), x] = 1
            x = torch.tensor(x, dtype=torch.int32).unsqueeze(1)
            
            n = len(seq)
            edge_index1 = dense_to_sparse(torch.ones((n, n)))[0]
            a = torch.zeros((n, n))
            a[range(n), np.arange(n)] = 1
            a[range(n-1), np.arange(n-1) + 1] = 1
            a[np.arange(n-1) + 1, np.arange(n-1)] = 1
            idx = int(n / 2)
            a[[idx]*n, range(n)] = 1
            edge_index2 = dense_to_sparse(a)[0]
            edge_index = torch.cat([edge_index1, edge_index2], dim=1)
            
            data = Data(
                x=x,
                x_one_hot=x_one_hot,
                edge_index1=edge_index1,
                edge_index2=edge_index2,
                edge_index=edge_index,
                emb=emb,
                seq=seq,
                uniprot=uniprot,
                start_idx=start_idx,
                prot_len=prot_len,
                unique_id=';'.join(unique_id),
                y=torch.tensor(label, dtype=torch.float32)
            )
            
            group = record['set']
            if group == 'train':
                train_ls.append(data)
            elif group == 'val':
                val_ls.append(data)
            elif group == 'test':
                test_ls.append(data)
            else:
                raise Exception('Unknown data group')

        return train_ls, val_ls, test_ls

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process protein sequences and generate embeddings.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing protein sequences.')
    parser.add_argument('--embed_dir', type=str, default='embeddings', help='Directory to save embeddings.')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory to save processed data.')
    parser.add_argument('--seq_lens', type=str, default='11,15,21,25,31,35,41,45,51,55,61', help='Comma-separated list of sequence lengths.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--esm_model', type=str, help='ESM model to use. Choose one of: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D, esm2_t33_650M_UR50D, esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D.')
    parser.add_argument('--ptm_mamba_ckpt', type=str, help='Path to the PTM-Mamba checkpoint.')

    args = parser.parse_args()
    
    if args.esm_model and args.ptm_mamba_ckpt:
        raise ValueError("You must specify either an ESM model or a PTM-Mamba checkpoint, not both.")
    if not args.esm_model and not args.ptm_mamba_ckpt:
        raise ValueError("You must specify either an ESM model or a PTM-Mamba checkpoint.")
    
    df = pd.read_csv(args.input_csv)
    embed_dir = args.embed_dir
    data_dir = args.data_dir
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    device = args.device
    
    esm_model, batch_converter, mamba = None, None, None
    if args.esm_model:
        esm_model_name = args.esm_model
        esm_model, alphabet = esm_models[esm_model_name]()
        batch_converter = alphabet.get_batch_converter()
        esm_model = esm_model.to(device).eval()
    
    if args.ptm_mamba_ckpt:
        mamba = PTMMamba(args.ptm_mamba_ckpt, device=device)
    
    for seq_len in tqdm(seq_lens):
        save_path = f'{data_dir}/{seq_len}.pt'
        processor = Processor(df, length=seq_len, embed_dir=embed_dir, device=device, esm_model=esm_model, batch_converter=batch_converter, mamba=mamba)
        data_ls = processor.run()
        torch.save(data_ls, save_path)
