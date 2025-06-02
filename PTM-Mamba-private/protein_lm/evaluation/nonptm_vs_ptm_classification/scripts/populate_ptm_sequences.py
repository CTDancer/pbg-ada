from typing import List, Optional, Set, Tuple, Dict
import os
import pickle
from Bio import SeqIO
from icecream import ic
import pandas as pd
from tqdm import tqdm

sequence_pickle_dir = 'protein_lm/evaluation/nonptm_vs_ptm_classification/data/sequences'


class AAToken:
  def __init__(self, non_ptm_token: str, ptm_token: str):
    self.non_ptm_token = non_ptm_token
    self.ptm_token = ptm_token

  def __repr__(self):
    return self.ptm_token


class Protein:
  def __init__(self, sequence: str, protein_id: str, reload: bool = True):
    self.protein_id: str = protein_id
    self.one_hot_embedding_file: Optional[str] = None
    self.esm650_embedding_file: Optional[str] = None
    self.esm3b_embedding_file: Optional[str] = None
    self.mamba_embedding_file: Optional[str] = None
    self.ptm_modifications: Set[Tuple[int, str]] = set()
    self.sequence_file: str = self._tokenize_sequence(sequence, reload=reload)


  def get_token_sequence(self):
    with open(self.sequence_file, 'rb') as f:
      return pickle.load(f)


  def _tokenize_sequence(self, sequence: str, reload: bool = True) -> List[AAToken]:
    token_sequence_file = f'{sequence_pickle_dir}/{self.protein_id}.pkl'

    if not reload:
      if os.path.exists(token_sequence_file):
        return token_sequence_file

    token_sequence = [AAToken(non_ptm_token=c, ptm_token=c) for c in sequence]

    with open(token_sequence_file, 'wb') as f:
      pickle.dump(token_sequence, f)

    return token_sequence_file

  def __repr__(self):
    return f'{self.protein_id}[{"".join([str(token) for token in self.get_token_sequence()])}]'
  
  
def extract_uniprot_id(description):
  parts = description.split('|')
  if len(parts) > 1 and parts[0] in ['sp', 'tr']:
    return parts[1]
  return None

def populate_protein_objects(fasta_file):
  for record in SeqIO.parse(fasta_file, "fasta"):
    uniprot_id = extract_uniprot_id(record.description)
    if uniprot_id:
      protein = Protein(sequence=str(record.seq), protein_id=uniprot_id, reload=False)
      proteins[uniprot_id] = protein
    else:
      print('ERROR PARSING FASTA')
      

def get_dbptm_df(combined_dbptm_file_path):
  dbptm_df = pd.read_csv(combined_dbptm_file_path,
                  sep='\t',
                  header=None,
                  names=['protein_name',
                        'protein_id',
                        'modified_position',
                        'modification_type',
                        'pubmed_id',
                        'sequence_context'])
  return dbptm_df


def populate_ptm_modifications(dbptm_df):
  dbptm_df.apply(add_ptm_modification, axis=1)


def add_ptm_modification(row):
    protein_id = row['protein_id']
    modified_position = row['modified_position']
    modification_type = row['modification_type']
    if protein_id in proteins:
        ptm_modifications = proteins[protein_id].ptm_modifications
        ptm_modifications.add((modified_position, modification_type))


def modify_all_aa_token_sequences():
  [modify_protein_aa_token_sequence(protein_id)
  for protein_id in proteins.keys()]


def modify_protein_aa_token_sequence(protein_id):
  protein = proteins[protein_id]
  modifications = protein.ptm_modifications
  token_sequence_file = protein.sequence_file
  token_sequence = protein.get_token_sequence()

  [modify_aa_token(token_sequence, mod) for mod in modifications]

  with open(token_sequence_file, 'wb') as f:
    pickle.dump(token_sequence, f)


def modify_aa_token(token_sequence, modification):
  position, modification_type = modification
  if position == 0 or position > len(token_sequence): return
  affected_aa = token_sequence[position-1].non_ptm_token
  specific_modification_map = {
    ('Phosphorylation', 'S'): '<Phosphoserine>',
    ('Phosphorylation', 'T'): '<Phosphothreonine>',
    ('Phosphorylation', 'Y'): '<Phosphotyrosine>',
    ('Acetylation', 'A'): '<N-acetylalanine>',
    ('Acetylation', 'M'): '<N-acetylmethionine>',
    ('Acetylation', 'K'): '<N6-acetyllysine>',
    ('Acetylation', 'S'): '<N-acetylserine>',
    ('Myristoylation', 'G'): '<N-myristoyl glycine>',
    ('Hydroxylation', 'P'): '<4-hydroxyproline>',
    ('S-palmitoylation', 'C'): '<S-palmitoyl cysteine>',
    ('Geranylgeranylation', 'C'): '<S-geranylgeranyl cysteine>',
    ('S-diacylglycerol', 'C'): '<S-diacylglycerol cysteine>'
  }
  general_modification_map = {
    'N-linked Glycosylation': '<N-linked (GlcNAc...) asparagine>',
    'Gamma-carboxyglutamic acid': '<4-carboxyglutamate>',
    'Pyrrolidone carboxylic acid': '<Pyrrolidone carboxylic acid>'
  }
  modified_token = specific_modification_map.get((modification_type, affected_aa))
  if modified_token is None:
    modified_token = general_modification_map.get(modification_type)
  if modified_token is None:
    return
  token_sequence[position-1].ptm_token = modified_token


if __name__ == "__main__":
  proteins: Dict[str, Protein] = {}
  all_sequences_file_path = 'protein_lm/evaluation/nonptm_vs_ptm_classification/data/all_sequences.fasta'
  combined_dbptm_file_path = 'protein_lm/evaluation/nonptm_vs_ptm_classification/data/dbptm.txt'
  
  populate_protein_objects(all_sequences_file_path)
  ic('populating protein objects')
  dbptm_df = get_dbptm_df(combined_dbptm_file_path)
  ic('populating dbptm dataframe')
  populate_ptm_modifications(dbptm_df)
  ic(proteins['A0AV96'].ptm_modifications)
  modify_all_aa_token_sequences()
  ic(''.join(proteins['A0AV96'].get_token_sequence()))
  