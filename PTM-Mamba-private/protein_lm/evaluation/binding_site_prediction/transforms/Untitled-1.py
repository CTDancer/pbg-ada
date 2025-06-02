# %%
import pandas as pd
import numpy as np
import os
from icecream import ic

# %%
ptm_data_csv_path = '/workspace/protein_lm/evaluation/binding_site_prediction/data/PPBS/labels_test_70_filtered.csv'
ppbs_df = pd.read_csv(ptm_data_csv_path)

ic(len(ppbs_df))
ppbs_df.head()

# %%
ptm_csv_path = '/workspace/protein_lm/evaluation/binding_site_prediction/data/ptm_data.csv'

ptm_df = pd.read_csv(ptm_csv_path)
ptm_df = ptm_df[['AC_ID', 'wt_seq', 'ptm_seq']]

ic(len(ptm_df))
ptm_df.head()

# %%
ac_ids = list(ptm_df['AC_ID'])
ic(len(ac_ids))
ic(len(set(ac_ids)))

# %%
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
from tqdm import tqdm
import pickle

def get_pdb_ids(uniprot_accession):
    url = f"https://www.uniprot.org/uniprot/{uniprot_accession}.txt"
    response = requests.get(url)
    pdb_ids = []

    if response.status_code == 200:
        for line in response.text.split('\n'):
            if line.startswith('DR   PDB;'):
                pdb_id = line.split(';')[1].strip()
                pdb_ids.append(pdb_id)
    else:
        print(f"Failed to retrieve data for accession ID {uniprot_accession}. Status code: {response.status_code}")

    return uniprot_accession, pdb_ids

def get_pdb_ids_for_accession_list(accession_list):
    results_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks and create a list of futures
        futures = [executor.submit(get_pdb_ids, accession) for accession in accession_list]

        # Iterate over futures as they complete (i.e., when the HTTP requests finish)
        for future in tqdm(as_completed(futures), total=len(accession_list), desc="Fetching PDB IDs"):
            try:
                accession, pdb_ids = future.result()
                results_dict[accession] = pdb_ids
            except Exception as exc:
                print(f"Task generated an exception: {exc}")
                
            # periodically save results_dict to a pickle file
            if len(results_dict) % 100 == 0:
                with open('/workspace/protein_lm/evaluation/binding_site_prediction/data/nested_pdb_ids.pkl', 'wb') as f:
                    pickle.dump(results_dict, f)

    # Construct the ordered nested list based on the original accession list
    return [results_dict[accession] for accession in accession_list]

accession_list = ac_ids
nested_pdb_ids = get_pdb_ids_for_accession_list(accession_list)

print("Nested list of PDB IDs for each UniProt accession ID:")
for accession, pdb_list in zip(accession_list, nested_pdb_ids):
    print(f"{accession}: {pdb_list}")

# save the nested_pdb_ids
with open('/workspace/protein_lm/evaluation/binding_site_prediction/data/nested_pdb_ids.pkl', 'wb') as f:
    pickle.dump(nested_pdb_ids, f)


