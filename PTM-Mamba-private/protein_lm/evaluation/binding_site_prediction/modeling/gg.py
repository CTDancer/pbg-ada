# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from icecream import ic
import pandas as pd
import pytorch_lightning as pl
import ast

# %% [markdown]
# ## Load dataset

# %%
residue_seqs_csv_path = '/workspace/protein_lm/evaluation/binding_site_prediction/data/residue_seqs_processed.csv'
residue_seqs_df = pd.read_csv(residue_seqs_csv_path)
residue_seqs_df['aligned_labels_with_gaps'] = residue_seqs_df['aligned_labels_with_gaps'].apply(ast.literal_eval)
residue_seqs_df['aligned_labels'] = residue_seqs_df['aligned_labels'].apply(ast.literal_eval)
residue_seqs_df['labels'] = residue_seqs_df['labels'].apply(ast.literal_eval)
residue_seqs_df.head()

# %%
max_seq_len = max(residue_seqs_df['wt_seq'].apply(len))
ic(max_seq_len)

# %% [markdown]
# ## Create dataloaders

# %%
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
		padding_mask_path = row[self.padding_mask_column]
		embedding = torch.load(embedding_path)
		padding_mask = torch.load(padding_mask_path)
		padding_mask = padding_mask.to(torch.bool)
		labels = torch.tensor(row['aligned_labels'], dtype=torch.float32)
		padded_labels = F.pad(labels, (0, self.max_seq_len - labels.shape[0]), value=-1)
		cutoff_embedding = embedding[:self.cutoff_seq_len]
		cutoff_padding_mask = padding_mask[:self.cutoff_seq_len]
		cutoff_padded_labels = padded_labels[:self.cutoff_seq_len]
		return cutoff_embedding, cutoff_padding_mask, cutoff_padded_labels

# %%
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
batch_size = 32
esm_train_loader = DataLoader(esm_train_dataset, batch_size=batch_size, shuffle=True)
esm_val_loader = DataLoader(esm_val_dataset, batch_size=batch_size, shuffle=False)
esm_test_loader = DataLoader(esm_test_dataset, batch_size=batch_size, shuffle=False)

mamba_wt_train_loader = DataLoader(mamba_wt_train_dataset, batch_size=batch_size, shuffle=True)
mamba_wt_val_loader = DataLoader(mamba_wt_val_dataset, batch_size=batch_size, shuffle=False)
mamba_wt_test_loader = DataLoader(mamba_wt_test_dataset, batch_size=batch_size, shuffle=False)

mamba_ptm_train_loader = DataLoader(mamba_ptm_train_dataset, batch_size=batch_size, shuffle=True)
mamba_ptm_val_loader = DataLoader(mamba_ptm_val_dataset, batch_size=batch_size, shuffle=False)
mamba_ptm_test_loader = DataLoader(mamba_ptm_test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# ## Define model

# %%
class ResiduePredictionModel(pl.LightningModule):
	def __init__(self, input_dim, hidden_dim, lr=1e-3):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, 2)
		self.lr = lr
	
	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		x = F.softmax(x, dim=-1)
		return x
	
	def training_step(self, batch, batch_idx):
		embeddings, padding_mask, labels = batch
		ic(labels)
		padding_mask = padding_mask.to(torch.bool)
		labels = labels.to(torch.long)
		predictions = self(embeddings)
		predictions = predictions[padding_mask]
		labels = labels[padding_mask]
		loss = F.cross_entropy(predictions, labels)
		ic(predictions)
		ic(labels)
		ic(loss.item())
		accuracy = (predictions.argmax(dim=-1) == labels).float().mean()
		ic(accuracy.item())
		self.log('train_loss', loss)
		return loss
	
	def configure_optimizers(self):
		return optim.Adam(self.parameters(), lr=self.lr)
	
	def validation_step(self, batch, batch_idx):
		embeddings, padding_mask, labels = batch
		padding_mask = padding_mask.to(torch.bool) # 32 x 1000
		labels = labels.to(torch.long)
		predictions = self(embeddings) # 32 x 1000 x 2
		predictions = predictions[padding_mask]
		labels = labels[padding_mask]
		loss = F.cross_entropy(predictions, labels)
		ic(predictions)
		ic(labels)
		accuracy = (predictions.argmax(dim=-1) == labels).float().mean()
		self.log('val_loss', loss)
		self.log('val_accuracy', accuracy)
		ic(loss.item(), accuracy.item())
		return loss

esm_650m_embedding_dim = 1280
input_dim = esm_650m_embedding_dim
hidden_dim = 256
model = ResiduePredictionModel(input_dim, hidden_dim)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
ic(num_params)

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
now = pd.Timestamp.now()
run_name = f'binding_prediction_{now}'
ic(run_name)
# wandb.init(project='binding_prediction', name=run_name, id=run_name)
# wandb_logger = pl.loggers.WnadbLogger(log_model='all')
# torch.cuda.empty_cache()
trainer = pl.Trainer(max_steps=2, devices=[5])
trainer.fit(model, esm_train_loader, esm_val_loader)

# %%


# %%


# %%



