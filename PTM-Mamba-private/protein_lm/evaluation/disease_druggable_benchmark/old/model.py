import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, task, model_type, token_to_index):
        self.data = dataframe
        self.task = task
        self.model_type = model_type

        if 'ptm' in model_type:
            self.sequences = dataframe['ptm_seq'].tolist()
        else:
            self.sequences = dataframe['wt_seq'].tolist()

        if task == 'druggability':
            self.labels = dataframe['is_druggable'].astype(int).tolist()
        elif task == 'disease':
            self.labels = dataframe['is_disease'].astype(int).tolist()

        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        return {'sequence': sequence, 'label': torch.tensor(label)}

def collate_fn(batch):
    sequences = [x['sequence'] for x in batch]
    labels = torch.tensor([x['label'] for x in batch])
    return {'sequence': sequences, 'label': labels}

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=128, num_labels=2, dropout=0.1):
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
    def __init__(self, ckpt_path, hidden_size=128):
        super(MambaClassificationModel, self).__init__()
        self.mamba = PTMMamba(ckpt_path, device='cuda:0')
        self.embedding = lambda sequences: self.mamba(sequences).hidden_states
        self.mlp = ClassificationHead(hidden_size, 2)

    def forward(self, sequences):
        x = self.embedding(sequences).to(self.device)
        x = self.mlp(x)
        return x

    def common_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label'].to(self.device)
        logits = self(sequences)
        loss = nn.CrossEntropyLoss()(logits, labels)
        outputs = torch.softmax(logits, dim=-1).argmax(dim=-1)
        return loss, labels, outputs

    def training_step(self, batch, batch_idx):
        loss, labels, outputs = self.common_step(batch, batch_idx)
        acc = Accuracy()(outputs.cpu(), labels.cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, outputs = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        metrics = compute_metrics(outputs, labels)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, prog_bar=True)
        return labels, outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class OnehotClassificationModel(MambaClassificationModel):
    def __init__(self, token_to_index, hidden_size=128):
        super(OnehotClassificationModel, self).__init__(ckpt_path=None, hidden_size=hidden_size)
        self.token_to_index = token_to_index
        self.embedding_layer = nn.Embedding(len(token_to_index), hidden_size)
        self.mlp = ClassificationHead(hidden_size, 2)
        self.embedding = self._compute_embedding
        
    def _compute_embedding(self, sequences):
        max_len = 1024
        sequences = [[self.token_to_index.get(token, self.token_to_index["<unk>"]) for token in seq[:max_len]] for seq in sequences]
        batch_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=self.token_to_index["<pad>"])
        return self.embedding_layer(batch_tokens.to(self.device))

class ESMClassificationModel(MambaClassificationModel):
    def __init__(self, esm_model, batch_converter, hidden_size, repr_layer):
        super(ESMClassificationModel, self).__init__(ckpt_path=None, hidden_size=hidden_size)
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.repr_layer = repr_layer
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.mlp = ClassificationHead(hidden_size, 2)
        self.embedding = self._compute_embedding
        
    def _compute_embedding(self, sequences):
        inputs = [(str(i), seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
        batch_tokens = batch_tokens[..., 1:-1].to(self.device)
        out = self.esm_model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        embedding = out["representations"][self.repr_layer]
        return embedding
