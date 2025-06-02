import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from peft import BOFTConfig, get_peft_model
from datasets import load_from_disk

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10,
    'margin': 1.0
}


# Siamese NN with Model Parallelism
class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.embedding_dim = encoder.config.hidden_size
        self.projection = nn.Linear(self.embedding_dim * 2, self.embedding_dim).to('cuda:7')

    def forward(self, target_tokens, binder_tokens, decoy_tokens):
        target_embedding = self.encoder(target_tokens).last_hidden_state[:, 0, :].to('cuda:0')
        binder_embedding = self.encoder(binder_tokens).last_hidden_state[:, 0, :].to('cuda:1')
        decoy_embedding = self.encoder(decoy_tokens).last_hidden_state[:, 0, :].to('cuda:2')

        # Compute joint embeddings
        anchor_embedding = torch.cat((target_embedding, binder_embedding), dim=-1).to('cuda:3')
        positive_embedding = torch.cat((binder_embedding, target_embedding), dim=-1).to('cuda:4')
        negative_embedding = torch.cat((decoy_embedding, binder_embedding), dim=-1).to('cuda:5')

        # Project joint embeddings back to original dimensions
        anchor_embedding = self.projection(anchor_embedding.to('cuda:7')).to('cuda:6')
        positive_embedding = self.projection(positive_embedding.to('cuda:7')).to('cuda:6')
        negative_embedding = self.projection(negative_embedding.to('cuda:7')).to('cuda:6')

        return anchor_embedding, positive_embedding, negative_embedding


# Generate scores for candidate binders
def generate_scores(siamese_net, tokenizer, target_seq, candidate_binders, decoy_seq):
    siamese_net.eval()
    scores = []

    with torch.no_grad():
        target_tokens = tokenizer(target_seq, return_tensors="pt", padding=True, truncation=True).to(device)
        decoy_tokens = tokenizer(decoy_seq, return_tensors="pt", padding=True, truncation=True).to(device)

        for binder_seq in candidate_binders:
            binder_tokens = tokenizer(binder_seq, return_tensors="pt", padding=True, truncation=True).to(device)
            target_embedding, binder_embedding, decoy_embedding = siamese_net(target_tokens, binder_tokens, decoy_tokens)
            target_binder_similarity = torch.cosine_similarity(target_embedding, binder_embedding)
            target_decoy_similarity = torch.cosine_similarity(target_embedding, decoy_embedding)
            score = target_binder_similarity - target_decoy_similarity
            scores.append(score.item())

    return scores


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained ESM-2-650M model and tokenizer
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name).to('cuda:0')

siamese_ppi_net = SiameseNetwork(model).to('cuda:0')

# Define the triplet loss function
criterion = nn.TripletMarginLoss(margin=HYPERPARAMS['margin']).to('cuda:6')

# Define the optimizer
optimizer = optim.Adam(siamese_ppi_net.parameters(), lr=HYPERPARAMS['learning_rate'])

# Load dataset
train_dataset = load_from_disk('train_dataset')
val_dataset = load_from_disk('val_dataset')
test_dataset = load_from_disk('test_dataset')

# Training loop
for epoch in range(HYPERPARAMS['num_epochs']):
    # Training
    siamese_ppi_net.train()
    train_loss = 0.0
    for batch in train_dataset:
        target_tokens = torch.tensor(batch['anchor_input_ids']).to('cuda:0')
        binder_tokens = torch.tensor(batch['positive_input_ids']).to('cuda:1')
        decoy_tokens  = torch.tensor(batch['negative_input_ids']).to('cuda:2')

        # Forward pass
        target_embedding, binder_embedding, decoy_embedding = siamese_ppi_net(target_tokens, binder_tokens, decoy_tokens)

        # Compute the triplet loss
        loss = criterion(target_embedding, binder_embedding, decoy_embedding).to('cuda:6')

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("greats")
    train_loss /= len(train_dataset)

    # Validation
    siamese_ppi_net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataset:
            target_tokens = torch.tensor(batch['anchor_input_ids']).to('cuda:0')
            binder_tokens = torch.tensor(batch['positive_input_ids']).to('cuda:1')
            decoy_tokens = torch.tensor(batch['negative_input_ids']).to('cuda:2')
            target_embedding, binder_embedding, decoy_embedding = siamese_ppi_net(target_tokens, binder_tokens, decoy_tokens)
            loss = criterion(target_embedding, binder_embedding, decoy_embedding).to('cuda:6')
            val_loss += loss.item()
    val_loss /= len(val_dataset)

    print(f"Epoch [{epoch+1}/{HYPERPARAMS['num_epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Testing
siamese_ppi_net.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_dataset:
        target_tokens = torch.tensor(batch['anchor_input_ids']).to('cuda:0')
        binder_tokens = torch.tensor(batch['positive_input_ids']).to('cuda:1')
        decoy_tokens = torch.tensor(batch['negative_input_ids']).to('cuda:2')
        target_embedding, binder_embedding, decoy_embedding = siamese_ppi_net(target_tokens, binder_tokens, decoy_tokens)
        loss = criterion(target_embedding, binder_embedding, decoy_embedding).to('cuda:6')
        test_loss += loss.item()
test_loss /= len(test_dataset)

print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
torch.save(siamese_ppi_net.state_dict(), "siamese_ppi_model.pth")
