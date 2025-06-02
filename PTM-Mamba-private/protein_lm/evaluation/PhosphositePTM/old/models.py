import esm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.classification import AUROC, Precision, Recall, Accuracy, MatthewsCorrCoef, F1Score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from protein_lm.modeling.scripts.infer import PTMMamba


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size=128,num_labels=2,dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features 
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []

        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                seq, label = line.strip().split(',')
                if len(seq) < 256 or len(seq) > 512:
                    continue
                if len(seq) <= 0:
                    continue
                self.data.append(seq)
                self.labels.append(torch.tensor(list(map(int, label))))
        print(f"Loaded {len(self.data)} sequences")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'sequence': self.data[idx], 'label': self.labels[idx]}


def collate_fn(batch):
    min_seq_len = min([len(x['sequence']) for x in batch])
    if min_seq_len > 1000:
        min_seq_len = 1000
    # crop sequences to the same length
    sequences = [x['sequence'][:min_seq_len] for x in batch]
    labels = torch.stack([x['label'][:min_seq_len] for x in batch])
    return {'sequence': sequences, 'label': labels}


class SequenceLengthSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.sorted_indices = sorted((range(len(data_source))), key=lambda x: len(data_source[x]['sequence']))

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.data_source)
    
def compute_metrics(outputs, labels):
    outputs = outputs.cpu()
    labels = labels.cpu()
    return {
            'accuracy': Accuracy("binary")(outputs, labels).item(),
            'precision': Precision("binary")(outputs, labels).item(),
            'recall': Recall("binary")(outputs, labels).item(),
            'f1': F1Score("binary")(outputs, labels).item(),
            'mcc': MatthewsCorrCoef("binary")(outputs, labels).item(),
            'auroc': AUROC("binary")(outputs, labels).item(),
            'auprc': average_precision_score(labels.cpu().numpy(), outputs.cpu().numpy())
        }
class MambaClassificationModel(pl.LightningModule):
    def __init__(self, ):
        super(MambaClassificationModel, self).__init__()
        ckpt_path = "/workspace/ckpt/bi_mamba-esm-ptm_token_input/best.ckpt"
        self.mamba = PTMMamba(ckpt_path, device='cuda:0')
        self.embedding = lambda sequences: self.mamba(sequences).hidden_states
        self.mlp = ClassificationHead(768, 2)  # Output a single value for each token

    def forward(self, sequences):
        x = self.embedding(sequences).to(self.device)
        x = self.mlp(x)
        return x

    def common_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label'].to(self.device)
        logits = self(sequences)
        loss = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.99]).to(labels.device))(
            logits.view(-1, 2), labels.view(-1))
        outputs = torch.softmax(logits, dim=-1).argmax(dim=-1)
        return loss, labels, outputs

    def training_step(self, batch, batch_idx):
        loss,labels, outputs = self.common_step(batch, batch_idx)
        acc = Accuracy("binary")(outputs.cpu(), labels.cpu())
        auroc = AUROC("binary")(outputs.cpu(), labels.cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_auroc', auroc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, outputs = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        metrics = compute_metrics(outputs, labels)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, prog_bar=True)
        self.print(metrics)
        return labels, outputs

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class OnehotClassificationModel(MambaClassificationModel):
    def __init__(self):
        super(OnehotClassificationModel, self).__init__()
        _, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embedding_layer = nn.Embedding(60,128)
        self.mlp = ClassificationHead(128, 2)
        self.embedding = self._compute_embedding
        
    def _compute_embedding(self,sequences):
        inputs = [
            (str(i),seq) for i,seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
        batch_tokens = batch_tokens.to(
            self.device
        )[..., 1:-1]
        return self.embedding_layer(batch_tokens)
    
class ESMClassificationModel(MambaClassificationModel):
    def __init__(self):
        super(ESMClassificationModel, self).__init__()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.mlp = ClassificationHead(2560, 2)
        self.embedding = self._compute_embedding
        
    def _compute_embedding(self,sequences):
        inputs = [
            (str(i),seq) for i,seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
        batch_tokens = batch_tokens[..., 1:-1].to(
            self.device
        )
        out = self.esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
        
        embedding = out["representations"][36]
        return embedding
