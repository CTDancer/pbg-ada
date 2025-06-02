import pandas as pd
import numpy as np
import os
import torch
from Bio import SeqIO
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import esm
from tqdm import tqdm
from types import SimpleNamespace
from protein_lm.modeling.scripts.infer import PTMMamba
from utils import TrainProcessor
from torch_geometric.loader import DataLoader
from model import GNNTrans
import argparse
import random

random.seed(42)
np.random.seed(42)

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
    'esm2_t48_15B_UR50D': (esm.pretrained.esm2_t48_15B_UR50D, 48),
    'esm2_t36_3B_UR50D': (esm.pretrained.esm2_t36_3B_UR50D, 36),
    'esm2_t33_650M_UR50D': (esm.pretrained.esm2_t33_650M_UR50D, 33),
    'esm2_t30_150M_UR50D': (esm.pretrained.esm2_t30_150M_UR50D, 30),
    'esm2_t12_35M_UR50D': (esm.pretrained.esm2_t12_35M_UR50D, 12),
    'esm2_t6_8M_UR50D': (esm.pretrained.esm2_t6_8M_UR50D, 6)
}

esm_embedding_dims = {
    'esm2_t48_15B_UR50D': 5120,
    'esm2_t36_3B_UR50D': 2560,
    'esm2_t33_650M_UR50D': 1280,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t6_8M_UR50D': 320
}

def get_esm_embedding(model, batch_converter, seq, device, num_layers):
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
    token_representations = results["representations"][num_layers][0, 1:-1]
    return token_representations.cpu()

def get_ptm_mamba_embedding(mamba, seq):
    output = mamba(seq)
    return output.hidden_states.squeeze(0).cpu()

class Processor:
    def __init__(self, info_df, length, embed_dir, device, esm_model=None, batch_converter=None, mamba=None, num_layers=None):
        self.info = info_df
        self.length = length
        self.embed_dir = embed_dir
        self.device = device
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.mamba = mamba
        self.num_layers = num_layers

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
                    emb = get_esm_embedding(self.esm_model, self.batch_converter, seq, self.device, self.num_layers)
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

def save_metrics_to_csv(result_dir, metrics, model_type, seq_len):
    df = pd.DataFrame([metrics])
    df['Model'] = model_type
    df['Seq_Len'] = seq_len
    csv_path = os.path.join(result_dir, 'test_results.csv')
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process protein sequences and generate embeddings.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing protein sequences.')
    parser.add_argument('--embed_dir', type=str, default='embeddings', help='Directory to save embeddings.')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory to save processed data.')
    parser.add_argument('--seq_lens', type=str, default='11,15,21,25,31,35,41,45,51,55,61', help='Comma-separated list of sequence lengths.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--esm_model', type=str, help='ESM model to use. Choose one of: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D, esm2_t33_650M_UR50D, esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D.')
    parser.add_argument('--ptm_mamba_ckpt', type=str, help='Path to the PTM-Mamba checkpoint.')
    parser.add_argument('--embedding_type', type=str, required=True, choices=['ptm_mamba', 'esm'], help='Type of embeddings to use: "ptm_mamba" or "esm".')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory to save results.')

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
    
    esm_model, batch_converter, mamba, num_layers = None, None, None, None
    if args.esm_model:
        esm_model_name = args.esm_model
        esm_model, num_layers = esm_models[esm_model_name]
        esm_model, alphabet = esm_model()
        batch_converter = alphabet.get_batch_converter()
        esm_model = esm_model.to(device).eval()
    
    if args.ptm_mamba_ckpt:
        mamba = PTMMamba(args.ptm_mamba_ckpt, device=device)
    
    for seq_len in tqdm(seq_lens):
        save_path = f'{data_dir}/{seq_len}.pt'
        processor = Processor(df, length=seq_len, embed_dir=embed_dir, device=device, esm_model=esm_model, batch_converter=batch_converter, mamba=mamba, num_layers=num_layers)
        data_ls = processor.run()
        torch.save(data_ls, save_path)

    # Now train the model using the generated embeddings
    if args.embedding_type == 'ptm_mamba':
        data_dir = args.data_dir if args.data_dir else "processed3_ptm_mamba"
        result_dir = args.result_dir if args.result_dir else "ptm_mamba_results"
        input_dim = 768
    elif args.embedding_type == 'esm':
        if not args.esm_model:
            raise ValueError("You must specify an ESM model when using ESM embeddings.")
        data_dir = args.data_dir if args.data_dir else "processed3_esm"
        result_dir = args.result_dir if args.result_dir else f"esm_{args.esm_model}_results"
        input_dim = esm_embedding_dims[args.esm_model]
    
    os.makedirs(result_dir, exist_ok=True)

    train_args = {
        'epochs': 500,
        'batch_size': 64,
        'device': args.device,
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-4,
        'lr': 3e-5,
        'es_patience': 20,
        'save': True
    }
    train_args = SimpleNamespace(**train_args)
    print(train_args)

    for seq_len in seq_lens:
        train_ls, val_ls, test_ls = torch.load(f'{data_dir}/{seq_len}.pt')
        train_data_loader = DataLoader(train_ls, batch_size=train_args.batch_size)
        val_data_loader = DataLoader(val_ls, batch_size=train_args.batch_size)
        test_data_loader = DataLoader(test_ls, batch_size=train_args.batch_size)

        for i in range(10):
            model = GNNTrans(input_dim=input_dim, hidden_dim=256, num_layers=2)
            model.to(train_args.device)
            print(model)

            train_val = TrainProcessor(
                model=model,
                loaders=[train_data_loader, val_data_loader, test_data_loader],
                args=train_args
            )
            best_model, test_metrics = train_val.train()
            print('test loss: {:5f}; test acc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}'.format(
                test_metrics.loss, test_metrics.acc, test_metrics.auroc, test_metrics.auprc))

            if train_args.save:
                save_dir = f'./{result_dir}/{seq_len}'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, '{}_acc{:.4f}_roc{:.4f}_prc{:.4f}_f1{:.4}_mcc{:.4f}_precision{:.4f}_recall{:.4f}.pt'.format(
                    i, test_metrics.acc, test_metrics.auroc, test_metrics.auprc, test_metrics.f1, test_metrics.mcc, test_metrics.precision, test_metrics.recall))
                torch.save(best_model.state_dict(), save_path)
                save_metrics_to_csv("./", test_metrics, args.embedding_type, seq_len)
