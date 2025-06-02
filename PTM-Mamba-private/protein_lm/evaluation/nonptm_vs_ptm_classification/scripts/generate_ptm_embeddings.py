from protein_lm.modeling.scripts.infer import PTMMamba
from protein_lm.evaluation.nonptm_vs_ptm_classification.scripts.populate_ptm_sequences import AAToken, Protein
from icecream import ic
import os
import pickle
import torch
from tqdm import tqdm


def get_ptm_sequences(sequences_dir):
  sequences = []
  for file in os.listdir(sequences_dir):
    with open(f'{sequences_dir}/{file}', 'rb') as f:
      protein_id = file.split('.')[0]
      sequence = pickle.load(f)
      sequences.append((protein_id, sequence))
  return sequences

def generate_embeddings(sequences, mamba, embeddings_dir):
    batch_size = 1
    for i in tqdm(range(0, len(sequences), batch_size)):
      try:
        batch = sequences[i:i + batch_size]
        sequences_batch = [seq for _, seq in batch]
        protein_ids_batch = [protein_id for protein_id, _ in batch]
        max_sequence_length = max([len(seq) for seq in sequences_batch])
        padded_sequences_batch = []
        for seq in sequences_batch:
          padded_sequences_batch.append(''.join([tok.ptm_token for tok in seq] + ['<pad>'] * (max_sequence_length - len(seq))))
          output = mamba(padded_sequences_batch)
        embeddings = output.hidden_states
        for j, protein_id in enumerate(protein_ids_batch):
          embedding_file_path = f'{embeddings_dir}/{protein_id}_mamba_embedding.pt'
          with open(embedding_file_path, 'wb') as f:
            torch.save(embeddings[j], f)
      except Exception as e:
        ic(e)
	# batch_size = 32

	# for i in range(0, batch_size, batch_size):
	# 	batch = sequences[i:i+batch_size]
	# 	sequences = [seq for _, seq in batch]
	# 	output = mamba(sequences)
	# 	embeddings = output.hidden_states
	# 	ic(embeddings.shape)
  
		# for j, (protein_id, _) in enumerate(batch):
		#     with open(f'{embeddings_dir}/{protein_id}_ptmmamba_embedding.pt', 'wb') as f:
		#         torch.save(embeddings[j], f)
		# # break


if __name__ == "__main__":
	ic('working')
	sequences_dir = 'protein_lm/evaluation/nonptm_vs_ptm_classification/data/sequences'
	ckpt_path = "ckpt/bi_mamba-esm-ptm_token_input/best.ckpt"
	embeddings_dir = 'protein_lm/evaluation/nonptm_vs_ptm_classification/data/embeddings'

	sequences = get_ptm_sequences(sequences_dir)
	sequences = [(protein_id, sequence) for protein_id, sequence in sequences if len(sequence) <= 8000]
	mamba = PTMMamba(ckpt_path,device='cuda')
	generate_embeddings(sequences, mamba, embeddings_dir)



	# seq = "M<N-acetylalanine>K"
	# output = mamba(seq)
	# ic(output.logits.shape)
	# ic(output.hidden_states.shape)
