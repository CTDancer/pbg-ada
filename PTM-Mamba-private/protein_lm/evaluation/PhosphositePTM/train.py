#------------------------------------------------------------------------------------------------------------------------
# Train each competing model 5x and store results
# -----------------------------------------------------------------------------------------------------------------------

import sys
from datetime import datetime
import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.classification import AUROC, Precision, Recall, Accuracy, MatthewsCorrCoef, F1Score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from protein_lm.evaluation.PhosphositePTM.models import CustomDataset, ESM3BClassificationModel, ESM650MClassificationModel, MambaClassificationModel, OnehotClassificationModel, SequenceLengthSampler, collate_fn, compute_metrics
from protein_lm.evaluation.PhosphositePTM.plot import plot_barchart, plot_linegraph
from protein_lm.modeling.scripts.infer import PTMMamba
import protein_lm.evaluation.PhosphositePTM.config as config

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

# Create DataLoader and Split Data
def create_dataloader(file_path, batch_size):
    dataset = CustomDataset(file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        sampler=SequenceLengthSampler(dataset))
    return loader

def test(model,test_loader):
    device = next(model.parameters()).device
    model.eval()
    test_labels = []
    test_outputs = []
    dfs = []
    for batch in test_loader:
        sequences = batch['sequence']
        labels = batch['label']
        loss, labels, outputs, probabilities = model.common_step(batch, 0)
        metrics = compute_metrics(outputs, probabilities, labels)
        metrics = {
            "Length": torch.tensor([len(sequence) for sequence in sequences]).float().mean().item(),
            **metrics
        }
        dfs.append(pd.DataFrame([metrics]))
    df = pd.concat(dfs)
    
    return df

def get_fresh_model(model_str, ckpt_path):
    if model_str=='mamba':
        return MambaClassificationModel(ckpt_path=ckpt_path)
    if model_str=='mamba_saprot':
        return MambaClassificationModel(ckpt_path=ckpt_path)
    if model_str=='ptm_transformer':
        return MambaClassificationModel(ckpt_path=ckpt_path)
    if model_str=='esm2_3b':
        return ESM3BClassificationModel()
    if model_str=='esm2_650m':
        return ESM650MClassificationModel()
    if model_str=='onehot':
        return OnehotClassificationModel()

def test_replicates(model_str, train_file_path, val_file_path, test_file_path, ckpt_paths, n_replicates=5, bs=512):
    # this is loop 1 - initialize the DF with the first run
    log_update(f'preparing to run {n_replicates} replicates of {model_str}.')
    
    log_update(f'replicate 1, {model_str}: creating the train loader...') 
    train_loader = create_dataloader(train_file_path, batch_size=bs)
    val_loader = create_dataloader(val_file_path, batch_size=bs)
    test_loader = create_dataloader(test_file_path, batch_size=bs)

    # make a loop to do each of these five times 
    # default is max epochs = 2, but let's change that to 5 and see if it helps with reproducibility 
    log_update(f'replicate 1, {model_str}: initializing the trainer...') 
    trainer = pl.Trainer(max_epochs=2, devices=1)
    
    log_update(f'replicate 1, {model_str}: initializing the model...') 
    ckpt_path = None
    model = get_fresh_model(model_str, ckpt_paths.get(model_str,None))
    
    log_update(f'replicate 1, {model_str}: training...') 
    trainer.fit(model, train_loader, val_loader)
    
    log_update(f'replicate 1, {model_str}: testing...') 
    df = test(model, test_loader)
    n_results = len(df)     # count the number of results in each run
    df['run'] = [1]*n_results       # add a column for which run this was
    log_update(df)
    for i in range(2, n_replicates+1):
        log_update(f'replicate {i}, {model_str}: creating the train loader...') 
        train_loader = create_dataloader(train_file_path, batch_size=bs)
        val_loader = create_dataloader(val_file_path, batch_size=bs)
        test_loader = create_dataloader(test_file_path, batch_size=bs)

        # make a loop to do each of these five times 
        # default is max epochs = 2, but let's change that to 5 and see if it helps with reproducibility 
        log_update(f'replicate {i}, {model_str}: initializing the trainer...') 
        trainer = pl.Trainer(max_epochs=2, devices=1)
        
        log_update(f'replicate {i}, {model_str}: initializing the model...') 
        model = get_fresh_model(model_str)
        
        log_update(f'replicate {i}, {model_str}: training...') 
        trainer.fit(model, train_loader, val_loader)
        
        log_update(f'replicate {i}, {model_str}: testing....') 
        temp = test(model, test_loader)
        temp['run'] = [i]*n_results
        df = pd.concat([df, temp])
        log_update(df)

    return df

def main(outdir='',n_replicates=5):
    # Set CUDA visible devices based on config file
    if config.CUDA_VISIBLE_DEVICES not in ["", None]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        log_update(f"set cuda device to {config.CUDA_VISIBLE_DEVICES}")
    
    # Make out paths if needed
    if not os.path.exists("results"): os.mkdir("results")
    if not os.path.exists(outdir): os.mkdir(outdir)
    
    # Set seed
    set_seed(42)
    
    # Define file paths
    train_file_path = 'PhosphositePTM.train.txt'
    val_file_path = 'PhosphositePTM.valid.txt'
    test_file_path = 'PhosphositePTM.test.txt'

    # Define batch size
    bs = 256 # for memory
    
    # Define ckpt paths for mamba versions
    CKPT_PATHS = {
        "mamba": "../../ckpt/bi_mamba-esm-ptm_token_input/best.ckpt",
        "ptm_transformer": "../../ckpt/ptm-transformer/best.ckpt",
        "mamba_saprot": "../../ckpt/ptm_mamba_saprot/best.ckpt"
    }

    # Extract chosen models from config
    model_strs = f"{'mamba,' if config.BENCHMARK_MAMBA else ''}" + f"{'mamba_saprot,' if config.BENCHMARK_MAMBA_SAPROT else ''}" + f"{'ptm_transformer,' if config.BENCHMARK_PTM_TRANSFORMER else ''}" 
    model_strs = model_strs + f"{'esm2_3b,' if config.BENCHMARK_ESM3B else ''}" + f"{'esm2_650m,' if config.BENCHMARK_ESM650M else ''}" + f"{'onehot,' if config.BENCHMARK_ONEHOT else ''}"
    model_strs = [x for x in model_strs.rstrip(',').split(',') if x!='']
    
    # Write the model training settings to a file in outdir for user's convenience
    with open(f'{outdir}/train_settings.txt', 'w') as f:
        f.write(f"Models being trained: {','.join(model_strs)}")
        f.write(f'\nReplicates: {n_replicates}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f'\nDevice: {device}')

    # Loop through models and train them
    plot_kwargs = {
        'plot_save_dir': outdir,
        'onehot_path': None,
        'esm2_650m_path': None,
        'esm2_3b_path': None,
        'mamba_path': None,
        'ptm_transformer_path': None,
        'mamba_saprot_path': None
    }
    
    for i in range(len(model_strs)):
        model_str = model_strs[i]
        
        log_update(f'\nbenchmarking model: {model_str}...')
        
        # train and test
        test_df = test_replicates(model_str, train_file_path, val_file_path, test_file_path, CKPT_PATHS, n_replicates=n_replicates, bs=bs)
        
        # save
        test_results_path = f'{outdir}/{model_str}_test_metrics.csv'
        test_df.to_csv(test_results_path)
        
        # prepare to plot
        plot_kwargs[f'{model_str}_path'] = test_results_path
    
    # Make plots
    plot_kwargs = {key: value for key, value in plot_kwargs.items() if value not in [None, '']}
    plot_barchart(**plot_kwargs)
    plot_linegraph(**plot_kwargs)
    

if __name__ == "__main__":
    outdir = 'results/' + get_timestamp()
    main(outdir=outdir, n_replicates=config.N_REPLICATES)