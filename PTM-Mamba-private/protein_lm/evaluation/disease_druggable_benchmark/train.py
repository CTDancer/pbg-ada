import protein_lm.evaluation.disease_druggable_benchmark.config as config
import os

# Set CUDA visible devices based on config file
if config.CUDA_VISIBLE_DEVICES not in ["", None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    
import pandas as pd
import numpy as np
import random
from datetime import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import esm
import argparse
import sys
from protein_lm.evaluation.disease_druggable_benchmark.model import CustomDataset, ESMClassificationModel, MambaClassificationModel, OnehotClassificationModel, collate_fn, compute_metrics, SequenceLengthSampler
from protein_lm.evaluation.disease_druggable_benchmark.summary import make_table
from protein_lm.evaluation.disease_druggable_benchmark.plot import make_auroc_curve, make_pr_curve

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

def freeze_base_model(model):
    for name, param in model.named_parameters():
        if 'mlp' not in name:  # Assuming 'mlp' is the classification head
            param.requires_grad = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def create_dataloader(dataframe, batch_size, task, model_type, token_to_index):
    dataset = CustomDataset(dataframe, task, model_type, token_to_index)
    # create loader with shuffle=False so we can use SequenceLengthSampler instead
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, sampler=SequenceLengthSampler(dataset))
    return loader

def initialize_dataloaders(task, train_df, test_df, token_to_index, model_str, batch_size=8):
    train_loader = create_dataloader(train_df, batch_size=batch_size, task=task, model_type=model_str, token_to_index=token_to_index)
    test_loader = create_dataloader(test_df, batch_size=batch_size, task=task, model_type=model_str, token_to_index=token_to_index)
    return train_loader, test_loader

def get_fresh_model(model_str, token_to_index, ckpt_path):
    """ 
    Return a freshly initialized model, with everything frozen except the MLP head
    """
    log_update(f"\tInitializing fresh {model_str} model...")
    if model_str=='onehot': 
        model = OnehotClassificationModel(token_to_index, hidden_size=128)
    if model_str=='onehotptm': 
        model = OnehotClassificationModel(token_to_index, hidden_size=128)
    if model_str=='ptm_mamba':
        model = MambaClassificationModel(ckpt_path, hidden_size=768)

    # ESM2 models
    esm_models = {
        "esm2_t48_15B_UR50D": {"model_fn": esm.pretrained.esm2_t48_15B_UR50D, "hidden_size": 5120, "repr_layer": 48},
        "esm2_t36_3B_UR50D": {"model_fn": esm.pretrained.esm2_t36_3B_UR50D, "hidden_size": 2560, "repr_layer": 36},
        "esm2_t33_650M_UR50D": {"model_fn": esm.pretrained.esm2_t33_650M_UR50D, "hidden_size": 1280, "repr_layer": 33},
        "esm2_t30_150M_UR50D": {"model_fn": esm.pretrained.esm2_t30_150M_UR50D, "hidden_size": 640, "repr_layer": 30},
        "esm2_t12_35M_UR50D": {"model_fn": esm.pretrained.esm2_t12_35M_UR50D, "hidden_size": 480, "repr_layer": 12},
        "esm2_t6_8M_UR50D": {"model_fn": esm.pretrained.esm2_t6_8M_UR50D, "hidden_size": 320, "repr_layer": 6},
    }
    
    if 'esm2_' in model_str:
        esm_info = esm_models[model_str]
        model = ESMClassificationModel(
            esm_info["model_fn"]()[0], 
            esm_info["model_fn"]()[1].get_batch_converter(), 
            hidden_size=esm_info["hidden_size"], 
            repr_layer=esm_info["repr_layer"]
        )
    
    freeze_base_model(model)  # Freeze the base model layers
    return model

def train_and_test_replicates(model_str, task_name, task_train, task_test, token_to_index, outdir='', n_replicates=1, ckpt_path='/protein_lm/ckpt/bi_mamba-esm-ptm_token_input/best.ckpt',device=torch.device("cpu")):  
    # Train and test replicates
    # this is loop 1 - initialize the DF with the first run
    log_update(f'\nreplicate 1/{n_replicates}, {model_str}, {task_name} task:\n\tInitializing dataloaders...') 
    train_loader, test_loader = initialize_dataloaders(task_name, task_train, task_test, token_to_index, model_str, batch_size=8)

    # Initialize model - result has everything frozen except MLP head
    log_update(f'\tInitializing new model...')
    model = get_fresh_model(model_str, token_to_index, ckpt_path)
    model.to(device)
    
    # Train
    log_update('\nTraining...') 
    trainer = pl.Trainer(max_epochs=2,devices=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader)
    log_update('\nTesting...')
    test_df, prob_and_label_df = test(model, test_loader)
    
    n_results = len(test_df)     # count the number of results in each run
    test_df['run'] = [1]*n_results       # add a column for which run this was
    n_test_samples = len(prob_and_label_df)
    prob_and_label_df['run'] = [1]*n_test_samples
    
    # Loop thorugh the other replicates
    for i in range(2, n_replicates+1):
        log_update(f'\nreplicate {i}/{n_replicates}, {model_str}, {task_name} task:\n\tInitializing dataloaders...') 
        train_loader, test_loader = initialize_dataloaders(task_name, task_train, task_test, token_to_index, model_str, batch_size=8)

        log_update(f'\tInitializing new model...')
        model = get_fresh_model(model_str, token_to_index, ckpt_path)
        
        # Train
        log_update('\nTraining...') 
        trainer = pl.Trainer(max_epochs=2,devices=1 if torch.cuda.is_available() else 0)
        trainer.fit(model, train_loader)
        log_update('\nTesting...')
        temp_test_df, temp_prob_and_label_df = test(model, test_loader)
        
        temp_test_df['run'] = [i]*n_results       # add a column for which run this was
        temp_prob_and_label_df['run'] = [i]*n_test_samples
        
        # Append this run's results to all the others
        test_df = pd.concat([test_df, temp_test_df])
        prob_and_label_df = pd.concat([prob_and_label_df, temp_prob_and_label_df])

    # Write the results
    test_results_path = f'{outdir}/test_metrics/{task_name}_test_results_{model_str}.csv'
    test_df.to_csv(test_results_path,index=False)
    
    prob_and_label_results_path = f'{outdir}/test_probabilities/{task_name}_test_probs_and_labels_{model_str}.csv'
    prob_and_label_df.to_csv(prob_and_label_results_path,index=False)
    
    return test_results_path, prob_and_label_results_path

def test(model, test_loader, device=torch.device("cpu")):
    #device = next(model.parameters()).device
    model.eval()
    model.to(device)
    dfs = []
    all_probs = torch.empty((0, 2)) # track all probs for ROC curve at the end
    all_labels = torch.empty(0) # track all labels for ROC curve at the end
    
    with torch.no_grad():  # Ensure gradients are not tracked - we don't need them for test
        for batch in test_loader:
            sequences = batch['sequence']
            labels = batch['label']
            loss, labels, outputs, probabilities = model.common_step(batch, 0)
            
            all_probs = torch.cat((all_probs, probabilities), dim=0) 
            all_labels = torch.cat((all_labels, labels), dim=0) 
            
            metrics = compute_metrics(outputs, probabilities, labels)
            metrics = {"Length": torch.tensor([len(sequence) for sequence in sequences]).float().mean().item(), **metrics}
            dfs.append(pd.DataFrame([metrics]))
        
    test_df = pd.concat(dfs) # make test_df
    
    # make dataframe with probabilities and labels for ROC curve later
    probabilities_np = all_probs.numpy()
    all_labels_np = all_labels.numpy()

    # Create a pandas DataFrame
    prob_and_label_data = {
        'prob_0': probabilities_np[:, 0],  # Probability of class 0
        'prob_1': probabilities_np[:, 1],  # Probability of class 1
        'label': all_labels_np             # True labels
    }
    prob_and_label_df = pd.DataFrame(prob_and_label_data)
    return test_df, prob_and_label_df

def main():     
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
        os.mkdir(f'{outdir}/test_metrics')
        os.mkdir(f'{outdir}/test_probabilities')
    
    # Set seed for reproducibility
    set_seed(42)

    # Collect all selected models in model_strs
    model_strs = []
    if config.BENCHMARK_ONEHOT==True: model_strs.append("onehot")
    if config.BENCHMARK_ONEHOTPTM==True: model_strs.append("onehotptm")
    if config.BENCHMARK_MAMBA==True: model_strs.append("ptm_mamba")
    if config.BENCHMARK_PTM_TRANSFORMER==True: model_strs.append("ptm_transformer")
    if config.BENCHMARK_MAMBA_SAPROT==True: model_strs.append("mamba_saprot")
    for k, v in config.BENCHMARK_ESM.items(): 
        if v==True: model_strs.append(k)
    
    # Write the model training settings to a file in outdir for user's convenience
    with open(f'{outdir}/train_settings.txt', 'w') as f:
        f.write(f"Models being trained: {','.join(model_strs)}")
        f.write(f"\nDruggability benchmark: {'running' if config.DRUGGABILITY else 'not running'}")
        f.write(f"\nDisease benchmark: {'running' if config.DISEASE else 'not running'}")
        f.write(f'\nReplicates: {config.N_REPLICATES}')
        f.write(f'\nDevice: {device}')


    # Define important dirs
    default_ckpt_path = '../../ckpt/bi_mamba-esm-ptm_token_input/best.ckpt'
    input_data_path = 'seqs_df.csv'
    
    # Define ckpt paths for mamba versions
    CKPT_PATHS = {
        "ptm_mamba": "../../ckpt/bi_mamba-esm-ptm_token_input/best.ckpt",
        "ptm_transformer": "../../ckpt/ptm-transformer/best.ckpt",
        "mamba_saprot": "../../ckpt/ptm_mamba_saprot/best.ckpt"
    }
    
    # Shared configurations
    tokens = [
        "<cls>", "<pad>", "<eos>", "<unk>", ".", "-", "<null_1>", "<mask>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", "PTM",
        "<N-linked (GlcNAc...) asparagine>", "<Pyrrolidone carboxylic acid>", "<Phosphoserine>", "<Phosphothreonine>", "<N-acetylalanine>", "<N-acetylmethionine>", "<N6-acetyllysine>", "<Phosphotyrosine>", 
        "<S-diacylglycerol cysteine>", "<N6-(pyridoxal phosphate)lysine>", "<N-acetylserine>", "<N6-carboxylysine>", "<N6-succinyllysine>", "<S-palmitoyl cysteine>", "<O-(pantetheine 4'-phosphoryl)serine>", 
        "<Sulfotyrosine>", "<O-linked (GalNAc...) threonine>", "<Omega-N-methylarginine>", "<N-myristoyl glycine>", "<4-hydroxyproline>", "<Asymmetric dimethylarginine>", "<N5-methylglutamine>", 
        "<4-aspartylphosphate>", "<S-geranylgeranyl cysteine>", "<4-carboxyglutamate>"
    ]
    token_to_index = {token: i for i, token in enumerate(tokens)}
    
    # Dictionary to change model names for input to summary.py
    summary_kwarg_prefixes={
        "onehot": "onehot", 
        "onehotptm": "onehotptm", 
        "ptm_mamba": "mamba",
        "ptm_transformer": "ptm_transformer",
        "mamba_saprot": "mamba_saprot",
        "esm2_t48_15B_UR50D": "esm2_15b",
        "esm2_t36_3B_UR50D": "esm2_3b",
        "esm2_t33_650M_UR50D": "esm2_650m",
        "esm2_t30_150M_UR50D": "esm2_150m",
        "esm2_t12_35M_UR50D": "esm2_35m",
        "esm2_t6_8M_UR50D": "esm2_8m"
    }

    # Load the data
    log_update('\nReading the input data, seqs_df.csv...')
    seqs_df = pd.read_csv(input_data_path)

    # Split data into training and testing datasets for both tasks
    log_update('\nSplitting data...')
    if config.DRUGGABILITY:
        drugptm_train_df = seqs_df[(seqs_df['is_part_of_druggability_dataset'] == True) & (seqs_df['is_train'] == True)]
        drugptm_test_df = seqs_df[(seqs_df['is_part_of_druggability_dataset'] == True) & (seqs_df['is_train'] == False)]
    if config.DISEASE:
        disease_train_df = seqs_df[(seqs_df['is_part_of_disease_dataset'] == True) & (seqs_df['is_train'] == True)]
        disease_test_df = seqs_df[(seqs_df['is_part_of_disease_dataset'] == True) & (seqs_df['is_train'] == False)]

    # Initialize summary kwargs so we can summarize the results after training
    druggable_summary_kwargs = {
        'results_dir': outdir,
        'table_name': 'druggable_summary.csv',
    }
    druggable_plot_kwargs = {
        'results_dir': outdir,
        'task_name': 'druggability',
    }
    disease_summary_kwargs = {
        'results_dir': outdir,
        'table_name': 'disease_summary.csv',
    }
    disease_plot_kwargs = {
        'results_dir': outdir,
        'task_name': 'disease',
    }
    for i in range(len(model_strs)):
        model_str = model_strs[i]
        
        log_update(f'\nbenchmarking model: {model_str}...')
        
        # train and test
        if config.DRUGGABILITY:
            test_results_path, prob_and_label_path = train_and_test_replicates(model_str, 'druggability', drugptm_train_df, drugptm_test_df, token_to_index, outdir=outdir,
                                                     n_replicates=config.N_REPLICATES,
                                                     ckpt_path=CKPT_PATHS.get(model_str,default_ckpt_path),
                                                     device=device)
            druggable_summary_kwargs[f'{summary_kwarg_prefixes[model_str]}_path'] = test_results_path
            druggable_plot_kwargs[f'{summary_kwarg_prefixes[model_str]}_path'] = prob_and_label_path
        if config.DISEASE:
            test_results_path, prob_and_label_path = train_and_test_replicates(model_str, 'disease', disease_train_df, disease_test_df, token_to_index, outdir=outdir,
                                                     n_replicates=config.N_REPLICATES,
                                                     ckpt_path=CKPT_PATHS.get(model_str,default_ckpt_path),
                                                     device=device)
            disease_summary_kwargs[f'{summary_kwarg_prefixes[model_str]}_path'] = test_results_path
            disease_plot_kwargs[f'{summary_kwarg_prefixes[model_str]}_path'] = prob_and_label_path
        
    # make summary
    log_update('\nSummarizing results...')
    make_table(**druggable_summary_kwargs)
    make_table(**disease_summary_kwargs)
    
    # make auroc and prc curves
    log_update('\nMaking plots...')
    make_auroc_curve(**druggable_plot_kwargs)
    make_auroc_curve(**disease_plot_kwargs)
    make_pr_curve(**druggable_plot_kwargs)
    make_pr_curve(**disease_plot_kwargs)
        
if __name__ == "__main__":
    main()