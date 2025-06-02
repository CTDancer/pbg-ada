import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import esm
import argparse
from model import CustomDataset, ESMClassificationModel, MambaClassificationModel, OnehotClassificationModel, collate_fn, compute_metrics

def create_dataloader(dataframe, batch_size, task, model_type, token_to_index):
    dataset = CustomDataset(dataframe, task, model_type, token_to_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

def initialize_dataloaders(task, train_df, test_df, token_to_index):
    loaders = {}
    model_types = ['onehot', 'onehotptm', 'ptm_mamba'] + list(esm_models.keys())
    for model_type in model_types:
        train_loader = create_dataloader(train_df, batch_size=32, task=task, model_type=model_type, token_to_index=token_to_index)
        test_loader = create_dataloader(test_df, batch_size=32, task=task, model_type=model_type, token_to_index=token_to_index)
        loaders[model_type] = (train_loader, test_loader)
    return loaders

def initialize_models(selected_esm_versions, token_to_index, ckpt_path):
    models = {
        'onehot': OnehotClassificationModel(token_to_index, hidden_size=128),
        'onehotptm': OnehotClassificationModel(token_to_index, hidden_size=128),
        'ptm_mamba': MambaClassificationModel(ckpt_path, hidden_size=768),
    }
    for esm_name in selected_esm_versions:
        esm_info = esm_models[esm_name]
        models[esm_name] = lambda esm_info=esm_info: ESMClassificationModel(
            esm_info["model_fn"]()[0], 
            esm_info["model_fn"]()[1].get_batch_converter(), 
            hidden_size=esm_info["hidden_size"], 
            repr_layer=esm_info["repr_layer"]
        )
    return models

def train_and_test_models(task, loaders, models):
    results = {}
    for model_type, model_init in models.items():
        model = model_init() if callable(model_init) else model_init
        trainer = pl.Trainer(max_epochs=2, devices=1)
        train_loader, test_loader = loaders[model_type]
        trainer.fit(model, train_loader)
        test_df = test(model, test_loader)
        results[model_type] = test_df
        test_df.to_csv(f'{task}_test_results_{model_type}.csv')
    return results

def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    dfs = []
    for batch in test_loader:
        sequences = batch['sequence']
        labels = batch['label']
        loss, labels, outputs = model.common_step(batch, 0)
        metrics = compute_metrics(outputs, labels)
        metrics = {"Length": torch.tensor([len(sequence) for sequence in sequences]).float().mean().item(), **metrics}
        dfs.append(pd.DataFrame([metrics]))
    df = pd.concat(dfs)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on druggability and disease datasets")
    parser.add_argument("--esm_versions", nargs='+', default=["esm2_t36_3B_UR50D", "esm2_t33_650M_UR50D"], help="List of ESM versions to use")
    parser.add_argument("--ckpt_path", type=str, default="/workspace/ckpt/bi_mamba-esm-ptm_token_input/best.ckpt", help="Path to the Mamba checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
    
    args = parser.parse_args()
    
    # Shared configurations
    tokens = [
        "<cls>", "<pad>", "<eos>", "<unk>", ".", "-", "<null_1>", "<mask>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", "PTM",
        "<N-linked (GlcNAc...) asparagine>", "<Pyrrolidone carboxylic acid>", "<Phosphoserine>", "<Phosphothreonine>", "<N-acetylalanine>", "<N-acetylmethionine>", "<N6-acetyllysine>", "<Phosphotyrosine>", 
        "<S-diacylglycerol cysteine>", "<N6-(pyridoxal phosphate)lysine>", "<N-acetylserine>", "<N6-carboxylysine>", "<N6-succinyllysine>", "<S-palmitoyl cysteine>", "<O-(pantetheine 4'-phosphoryl)serine>", 
        "<Sulfotyrosine>", "<O-linked (GalNAc...) threonine>", "<Omega-N-methylarginine>", "<N-myristoyl glycine>", "<4-hydroxyproline>", "<Asymmetric dimethylarginine>", "<N5-methylglutamine>", 
        "<4-aspartylphosphate>", "<S-geranylgeranyl cysteine>", "<4-carboxyglutamate>"
    ]
    token_to_index = {token: i for i, token in enumerate(tokens)}

    esm_models = {
        "esm2_t48_15B_UR50D": {"model_fn": esm.pretrained.esm2_t48_15B_UR50D, "hidden_size": 5120, "repr_layer": 48},
        "esm2_t36_3B_UR50D": {"model_fn": esm.pretrained.esm2_t36_3B_UR50D, "hidden_size": 2560, "repr_layer": 36},
        "esm2_t33_650M_UR50D": {"model_fn": esm.pretrained.esm2_t33_650M_UR50D, "hidden_size": 1280, "repr_layer": 33},
        "esm2_t30_150M_UR50D": {"model_fn": esm.pretrained.esm2_t30_150M_UR50D, "hidden_size": 640, "repr_layer": 30},
        "esm2_t12_35M_UR50D": {"model_fn": esm.pretrained.esm2_t12_35M_UR50D, "hidden_size": 480, "repr_layer": 12},
        "esm2_t6_8M_UR50D": {"model_fn": esm.pretrained.esm2_t6_8M_UR50D, "hidden_size": 320, "repr_layer": 6},
    }

    # Load the data
    seqs_df = pd.read_csv('seqs_df.csv')

    # Split data into training and testing datasets for both tasks
    drugptm_train_df = seqs_df[(seqs_df['is_part_of_druggability_dataset'] == True) & (seqs_df['is_train'] == True)]
    drugptm_test_df = seqs_df[(seqs_df['is_part_of_druggability_dataset'] == True) & (seqs_df['is_train'] == False)]
    disease_train_df = seqs_df[(seqs_df['is_part_of_disease_dataset'] == True) & (seqs_df['is_train'] == True)]
    disease_test_df = seqs_df[(seqs_df['is_part_of_disease_dataset'] == True) & (seqs_df['is_train'] == False)]

    # Initialize DataLoaders for each task and model type
    drug_loaders = initialize_dataloaders('druggability', drugptm_train_df, drugptm_test_df, token_to_index)
    disease_loaders = initialize_dataloaders('disease', disease_train_df, disease_test_df, token_to_index)

    # Initialize models
    models = initialize_models(args.esm_versions, token_to_index, args.ckpt_path)

    # Train and test models
    drug_results = train_and_test_models('druggability', drug_loaders, models)
    disease_results = train_and_test_models('disease', disease_loaders, models)
