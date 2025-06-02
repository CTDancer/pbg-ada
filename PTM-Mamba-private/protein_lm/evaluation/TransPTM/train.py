import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

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
import protein_lm.evaluation.TransPTM.config as config
from protein_lm.evaluation.TransPTM.summary import make_all_tables_and_plots

# Set CUDA visible devices based on config file
if config.CUDA_VISIBLE_DEVICES not in ["", None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

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

def log_update(text):
    print(text)
    sys.stdout.flush()

def get_timestamp():
    # Get current datetime
    now = datetime.now()

    # Format the date part as 'jan1_2025' format
    month_name = now.strftime('%b').lower() 
    day = now.day
    year = now.year
    date_part = f"{month_name}{day}_{year}"

    # Calculate the total number of seconds since midnight
    midnight = datetime(year, now.month, day)
    seconds_since_midnight = int((now - midnight).total_seconds())

    # Combine the date part with the number of seconds
    custom_format = f"{date_part}_{seconds_since_midnight}"
    return custom_format

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def save_metrics_to_csv(result_dir, rep, metrics, model_type, seq_len):
    # Extract metrics as simple numeric values
    metrics_dict = {
        'loss': metrics.loss,
        'acc': metrics.acc.item() if isinstance(metrics.acc, torch.Tensor) else metrics.acc,
        'auroc': metrics.auroc,
        'auprc': metrics.auprc,
        'f1': metrics.f1,
        'mcc': metrics.mcc,
        'precision': metrics.precision,
        'recall': metrics.recall
    }
    
    df = pd.DataFrame([metrics_dict])
    df['model'] = model_type
    df['seq_len'] = seq_len
    df['replicate'] = rep
    csv_path = os.path.join(result_dir, f'{model_type}_test_results.csv')
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else: # if the file is already there, just add on the results instead of overwriting the file 
        df.to_csv(csv_path, mode='a', header=False, index=False)

def get_fresh_model(model_str, device, mamba_ckpt='../../ckpt/bi_mamba-esm-ptm_token_input/best.ckpt'):
    if 'esm2_' in model_str:
        esm_model, num_layers = esm_models[model_str]
        esm_model, alphabet = esm_model()
        batch_converter = alphabet.get_batch_converter()
        esm_model = esm_model.to(device).eval()
        return esm_model, batch_converter, num_layers
    if model_str.isin(["ptm_mamba","ptm_transformer","mamba_saprot"]):
        mamba = PTMMamba(mamba_ckpt, device=device)
        return mamba
    
def process_data(df, seq_lens, model_str, device, outdir, mamba_ckpt=None):
    if 'esm2_' in model_str:
        esm_model, batch_converter, num_layers = get_fresh_model(model_str, device)
        mamba = None
    if model_str.isin(["ptm_mamba","ptm_transformer","mamba_saprot"]):
        mamba = get_fresh_model(model_str, device, mamba_ckpt=mamba_ckpt)
        batch_converter, esm_model, num_layers = None, None, None
    
    embed_dir = f'embeddings/{model_str}'
                 
    # iterate through seq lengths, and embed each one
    for seq_len in tqdm(seq_lens):
        log_update(f"processing seq length {seq_len}...")
        
        save_path = f'processed_data/{model_str}/{seq_len}.pt'
        
        if os.path.exists(save_path): log_update(f"\tloading cached weights from {save_path}")
        else:
            # make directory to save processed data of this model type, if needed 
            os.makedirs(f'{outdir}/processed_data', exist_ok=True)
            os.makedirs(f'{outdir}/processed_data/{model_str}', exist_ok=True)
            os.makedirs(f'{outdir}/embeddings/{model_str}', exist_ok=True) 
        
        processor = Processor(df, length=seq_len, 
                              embed_dir=embed_dir, 
                              device=device, 
                              esm_model=esm_model, 
                              batch_converter=batch_converter, 
                              mamba=mamba, num_layers=num_layers)
        data_ls = processor.run()
        torch.save(data_ls, save_path)

def train_one_replicate(rep, seq_len, model_str, input_dim, data_dir, result_dir, train_args, train_ls, val_ls, test_ls):
    # make dataloaders 
    train_data_loader = DataLoader(train_ls, batch_size=train_args.batch_size)
    val_data_loader = DataLoader(val_ls, batch_size=train_args.batch_size)
    test_data_loader = DataLoader(test_ls, batch_size=train_args.batch_size)

    # get fresh model
    model = GNNTrans(input_dim=input_dim, hidden_dim=256, num_layers=2)
    model.to(train_args.device)
    #print(model)

    train_val = TrainProcessor(
        model=model,
        loaders=[train_data_loader, val_data_loader, test_data_loader],
        args=train_args
    )
    best_model, test_metrics = train_val.train()
    print('test loss: {:5f}; test acc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}'.format(
        test_metrics.loss, test_metrics.acc, test_metrics.auroc, test_metrics.auprc))

    # if we're saving, save one result / seq length
    if train_args.save:
        save_dir = f'./{result_dir}/{seq_len}'
        # if this directory doesn't exist, make it 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        # make save path
        model_save_path = os.path.join(save_dir, f'replicate{rep}_best_model.pt')
        
        # save the state dict to this path 
        torch.save(best_model.state_dict(), model_save_path)
        
        # save metrics to csv
        save_metrics_to_csv(result_dir, rep, test_metrics, model_str, seq_len) # COME BACK TO THIS

def train_n_replicates(df, seq_lens, model_str, n_replicates, train_args, device, outdir, mamba_ckpt=None):
    process_data(df, seq_lens, model_str, device, outdir, mamba_ckpt=mamba_ckpt)
    
    # Prepare directories
    data_dir = f'processed_data/{model_str}'
    
    result_dir = f'{outdir}/test_metrics/{model_str}'
    os.makedirs(result_dir, exist_ok=True)
    
    # Acquire input dimension
    if model_str.isin(["ptm_mamba","ptm_transformer","mamba_saprot"]):
        input_dim = 768
    elif 'esm2_' in model_str:
        input_dim = esm_embedding_dims[model_str]
            
    # loop through seq lens
    for seq_len in seq_lens:
        # load data
        train_ls, val_ls, test_ls = torch.load(f'{data_dir}/{seq_len}.pt')
        
        for rep in range(1, 1+ n_replicates):
            log_update(f"\ntraining replicate {rep}...")
            train_one_replicate(rep, seq_len, model_str, input_dim, data_dir, result_dir, train_args, train_ls, val_ls, test_ls)
    
def combine_results_csvs(result_dir, path_list):
    """
    Args:
        result_dir: outer directory for results (date and time)
        path_list: contains paths to the results files
    """
    final_csv_path = f'{result_dir}/test_results.csv'
    
    for p in path_list:
        df = pd.read_csv(p)
        
        if not os.path.isfile(final_csv_path):
            df.to_csv(final_csv_path, index=False)
        else: # if the file is already there, just add on the results instead of overwriting the file 
            df.to_csv(final_csv_path, mode='a', header=False, index=False)
        
def main():
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    # Check what the cuda visible devices are 
    log_update(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Set output directory
    outdir = 'results/' + get_timestamp()
    # Make out paths if needed
    if not os.path.exists("results"): os.mkdir("results")
    if not os.path.exists(outdir): 
        os.mkdir(outdir)
        os.mkdir(f'{outdir}/figures')

    # set some of the things that used to come in as command-line arguments
    INPUT_CSV = "dataset.csv" # Path to the input CSV file containing protein sequences
    N_REPLICATES = config.N_REPLICATES
    
    # read input data
    df = pd.read_csv(INPUT_CSV)
    
    # set sequence lengths
    seq_lens = [11,15,21,25,31,35,41,45,51,55,61] # list of seq lengths
    
    # Collect all selected models in model_strs
    model_strs = []
    if config.BENCHMARK_MAMBA==True: model_strs.append("ptm_mamba")
    if config.BENCHMARK_PTM_TRANSFORMER==True: model_strs.append("ptm_transformer")
    if config.BENCHMARK_MAMBA_SAPROT==True: model_strs.append("mamba_saprot")
    for k, v in config.BENCHMARK_ESM.items(): 
        if v==True: model_strs.append(k)
    
    # Define ckpt paths for each type of model
    CKPT_PATHS = {
        "ptm_mamba": "../../ckpt/bi_mamba-esm-ptm_token_input/best.ckpt",
        "ptm_transformer": "../../ckpt/ptm-transformer/best.ckpt",
        "mamba_saprot": "../../ckpt/ptm_mamba_saprot/best.ckpt"
    }
    
    # Write the model training settings to a file in outdir for user's convenience
    with open(f'{outdir}/train_settings.txt', 'w') as f:
        f.write(f"Models being trained: {','.join(model_strs)}")
        f.write(f'\nReplicates: {N_REPLICATES}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f'\nDevice: {device}')
        f.write("\nckpt paths for mamba models:")
        for model_name, ckpt_path in CKPT_PATHS.items():
            f.write(f"\n\t{model_name}: {ckpt_path}")
        
    train_args = {
        'epochs': 500,
        'batch_size': 64,
        'device': device,
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-4,
        'lr': 3e-5,
        'es_patience': 20,
        'save': True
    }
    log_update('Training Arguments:')
    for k, v in train_args.items():
        log_update(f'\t{k}: {v}')
    train_args = SimpleNamespace(**train_args)      # change datatype for easier argument access
    
    model_csvs = []
        
    for model_str in model_strs:
        # process data and train n_replicates models
        # Need to get fresh model, train for each of the replicates
        train_n_replicates(df, seq_lens, model_str, N_REPLICATES, train_args, device, outdir, mamba_ckpt=CKPT_PATHS.get(mamba_ckpt,None))
    
        model_csvs.append(f'{outdir}/test_metrics/{model_str}/{model_str}_test_results.csv')
     
     # combine the different results files   
    combine_results_csvs(outdir, model_csvs)
    
    make_all_tables_and_plots(outdir)

if __name__ == '__main__':    
    main()
