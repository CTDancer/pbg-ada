import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.classification import AUROC, Precision, Recall, Accuracy, MatthewsCorrCoef, F1Score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from protein_lm.evaluation.PhosphositePTM.models import CustomDataset, ESMClassificationModel, MambaClassificationModel, OnehotClassificationModel, SequenceLengthSampler, collate_fn, compute_metrics
from protein_lm.modeling.scripts.infer import PTMMamba


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
        loss, labels, outputs = model.common_step(batch, 0)
        metrics = compute_metrics(outputs, labels)
        metrics = {
            "Length": torch.tensor([len(sequence) for sequence in sequences]).float().mean().item(),
            **metrics
        }
        dfs.append(pd.DataFrame([metrics]))
    # overall_metrics = compute_metrics(test_outputs, test_labels)
    # overall_metrics = {
    #     "Length": "Overall",
    #     **overall_metrics
    # }
    # dfs.append(pd.DataFrame([overall_metrics]))
    df = pd.concat(dfs)
    
    return df



# File paths
train_file_path = 'protein_lm/evaluation/PhosphositePTM/PhosphositePTM.train.txt'
val_file_path = 'protein_lm/evaluation/PhosphositePTM/PhosphositePTM.valid.txt'
test_file_path = 'protein_lm/evaluation/PhosphositePTM/PhosphositePTM.test.txt'

bs = 512
train_loader = create_dataloader(train_file_path, batch_size=bs)
val_loader = create_dataloader(val_file_path, batch_size=bs)
test_loader = create_dataloader(test_file_path, batch_size=bs)

# Create PyTorch Lightning model
mamba_model = MambaClassificationModel()
esm_model = ESMClassificationModel()
onehot_model = OnehotClassificationModel()
# Train the model
trainer = pl.Trainer(max_epochs=2, devices=1)
trainer.fit(onehot_model, train_loader, val_loader)


test_df = test(mamba_model, test_loader)
test_df.to_csv('test_metrics.csv')

