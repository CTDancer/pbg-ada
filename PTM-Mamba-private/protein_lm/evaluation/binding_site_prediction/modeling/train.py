
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from icecream import ic
import pandas as pd
import pytorch_lightning as pl
import ast
import wandb

residue_seqs_csv_path = '/workspace/protein_lm/evaluation/binding_site_prediction/data/residue_seqs_processed.csv'
residue_seqs_df = pd.read_csv(residue_seqs_csv_path)
residue_seqs_df['aligned_labels_with_gaps'] = residue_seqs_df['aligned_labels_with_gaps'].apply(ast.literal_eval)
residue_seqs_df['aligned_labels'] = residue_seqs_df['aligned_labels'].apply(ast.literal_eval)
residue_seqs_df['labels'] = residue_seqs_df['labels'].apply(ast.literal_eval)
residue_seqs_df.head()

# %%
max_seq_len = max(residue_seqs_df['wt_seq'].apply(len))
ic(max_seq_len)

class ResiduePredictionDataset(Dataset):
	def __init__(self, df, embedding_column, padding_mask_column, max_seq_len, cutoff_seq_len=1000):
		self.df = df
		self.embedding_column = embedding_column
		self.padding_mask_column = padding_mask_column
		self.max_seq_len = max_seq_len
		self.cutoff_seq_len = cutoff_seq_len

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		embedding_path = row[self.embedding_column]
		embedding = torch.load(embedding_path)
		labels = torch.tensor(row['aligned_labels'], dtype=torch.float32)
		return {'embedding': embedding, 'labels': labels}
	

test_dataset = ResiduePredictionDataset(residue_seqs_df, 'esm_650m_embedding_path', 'esm_650m_embedding_padding_mask_path', max_seq_len)

embedding, labels = next(iter(test_dataset))

# %%
from torch.nn.utils.rnn import pad_sequence

def crop_seq(input_ids, max_seq_len):
    """
    randomly crop sequences to max_seq_len
    Args:
        input_ids: tensor of shape (seq_len)
        max_seq_len: int
    """
    seq_len = len(input_ids)
    if seq_len <= max_seq_len:
        return input_ids
    else:
        start_idx = torch.randint(0, seq_len - max_seq_len + 1, (1,)).item()
        return input_ids[start_idx : start_idx + max_seq_len]
def collate_fn(batch):
    labels = [item['labels'] for item in batch]
    min_seq_len = min([len(label) for label in labels])
    embeddings = [item['embedding'] for item in batch]
    embeddings = torch.stack([crop_seq(embedding, min_seq_len) for embedding in embeddings])

    labels = torch.stack([crop_seq(label, min_seq_len) for label in labels])

    return embeddings, labels

	


train_df = residue_seqs_df[residue_seqs_df['is_train'] == True]
val_df = residue_seqs_df[residue_seqs_df['is_val'] == True]
test_df = residue_seqs_df[residue_seqs_df['is_test'] == True]

ic(len(train_df), len(val_df), len(test_df))

# %%
esm_train_dataset = ResiduePredictionDataset(train_df, 'esm_650m_embedding_path', 'esm_650m_embedding_padding_mask_path', max_seq_len)
esm_val_dataset = ResiduePredictionDataset(val_df, 'esm_650m_embedding_path', 'esm_650m_embedding_padding_mask_path', max_seq_len)
esm_test_dataset = ResiduePredictionDataset(test_df, 'esm_650m_embedding_path', 'esm_650m_embedding_padding_mask_path', max_seq_len)

mamba_wt_train_dataset = ResiduePredictionDataset(train_df, 'mamba_without_ptms_embedding_path', 'mamba_without_ptms_embedding_padding_mask_path', max_seq_len)
mamba_wt_val_dataset = ResiduePredictionDataset(val_df, 'mamba_without_ptms_embedding_path', 'mamba_without_ptms_embedding_padding_mask_path', max_seq_len)
mamba_wt_test_dataset = ResiduePredictionDataset(test_df, 'mamba_without_ptms_embedding_path', 'mamba_without_ptms_embedding_padding_mask_path', max_seq_len)

mamba_ptm_train_dataset = ResiduePredictionDataset(train_df, 'mamba_with_ptms_embedding_path', 'mamba_with_ptms_embedding_padding_mask_path', max_seq_len)
mamba_ptm_val_dataset = ResiduePredictionDataset(val_df, 'mamba_with_ptms_embedding_path', 'mamba_with_ptms_embedding_padding_mask_path', max_seq_len)
mamba_ptm_test_dataset = ResiduePredictionDataset(test_df, 'mamba_with_ptms_embedding_path', 'mamba_with_ptms_embedding_padding_mask_path', max_seq_len)

# %%
batch_size = 64
esm_train_loader = DataLoader(esm_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
esm_val_loader = DataLoader(esm_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
esm_test_loader = DataLoader(esm_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

mamba_wt_train_loader = DataLoader(mamba_wt_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
mamba_wt_val_loader = DataLoader(mamba_wt_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
mamba_wt_test_loader = DataLoader(mamba_wt_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

mamba_ptm_train_loader = DataLoader(mamba_ptm_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
mamba_ptm_val_loader = DataLoader(mamba_ptm_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
mamba_ptm_test_loader = DataLoader(mamba_ptm_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %%
embeddings, labels = next(iter(esm_train_loader))
labels

# %%


# %%
embedding_path = '/workspace/protein_lm/evaluation/binding_site_prediction/data/embeddings/A0A0B4J1L0_mamba_with_ptms.pt'
embedding = torch.load(embedding_path)

# %% [markdown]
# ## Define model

# %%
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class Model(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers=3):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.net = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
		self.linear2 = nn.Linear(hidden_dim, 2)
			
	
	def forward(self, x):
		x = F.relu(self.linear1(x))
		for layer in self.net:
			x = F.relu(layer(x))
		x = self.linear2(x)
		return x  # Removed softmax here

class ResiduePredictionModel(pl.LightningModule):
	def __init__(self, input_dim, hidden_dim, test_loader, lr=1e-3):
		super().__init__()
		self.model = Model(input_dim, hidden_dim)
		self.lr = lr
		self.test_loader = test_loader

	def training_step(self, batch, batch_idx):

		embeddings, labels = batch
		labels = labels.to(torch.long)
		predictions = self.model(embeddings)
		loss = F.cross_entropy(predictions.reshape(-1, 2), labels.reshape(-1))
		f1 = f1_score(labels.cpu().numpy().reshape(-1), torch.argmax(predictions, dim=-1).cpu().numpy().reshape(-1))
		matthews = matthews_corrcoef(labels.cpu().numpy().reshape(-1), torch.argmax(predictions, dim=-1).cpu().numpy().reshape(-1))
		recall = recall_score(labels.cpu().numpy().reshape(-1), torch.argmax(predictions, dim=-1).cpu().numpy().reshape(-1))
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_matthews', matthews, on_step=True, on_epoch=True, 
		prog_bar=True, logger=True)
		self.log('train_recall', recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		
		return loss
	
	def configure_optimizers(self):
		return optim.Adam(self.parameters(), lr=self.lr)
	
	def validation_step(self, batch, batch_idx):
		embeddings, labels = batch
		labels = labels.to(torch.long)
		predictions = self.model(embeddings)
		loss = F.cross_entropy(predictions, labels)
		
		# loss = F.cross_entropy(predictions.reshape(-1, 2), labels.reshape(-1))
	
		return loss
	
	def test_step(self, batch, batch_idx):
		embeddings, labels = batch
		labels = labels.to(torch.long)
		predictions = self.model(embeddings)
		loss = F.cross_entropy(predictions.reshape(-1, 2), labels.reshape(-1))
		predictions = torch.argmax(predictions, dim=1)
		ic(predictions, labels)
		return {'loss': loss, 'predictions': predictions, 'labels': labels}
	
	
	def test_dataloader(self):
		return self.test_loader
		

# Assuming esm_650m_embedding_dim, input_dim, hidden_dim, and learning rate are defined elsewhere
esm_650m_embedding_dim = 768
input_dim = esm_650m_embedding_dim
hidden_dim = 512
model = ResiduePredictionModel(input_dim, hidden_dim, esm_test_loader, lr=3e-2)
trainer = pl.Trainer(max_epochs=20, devices=[5],overfit_batches=0)  # Ensure overfit_batches is set appropriately
# Assuming esm_train_loader and esm_val_loader are defined elsewhere
trainer.fit(model, mamba_ptm_train_loader, mamba_ptm_val_loader)
