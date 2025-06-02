"""Pull full sequences from PDB files for error sequences"""
import json
import os
import logging
from Bio import PDB
import warnings
import requests
import pickle
import pandas as pd
import argparse

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)
logging.basicConfig(filename='pdb.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AA_CODE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# Missing residues are recoreded in REMARK 465 fields in pdb files
def extract_remark_465(pdb_file_path):
    remark_465_lines = []
    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK 465     "):
                remark_465_lines.append(line.strip())
    return remark_465_lines[2:]


def parse_remark_465(remark_465_lines):
    missing_residues = {}
    for line in remark_465_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        chain_id = parts[3]
        resseq = int(parts[4])
        resname = parts[2]
        # print(resname, chain_id, resseq)
        if chain_id not in missing_residues:
            missing_residues[chain_id] = []
        missing_residues[chain_id].append((resseq, resname))
    return missing_residues


def extract_sequences(structure, target_chain_id, missing_residues):
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        residues = list(chain.get_residues())
        if chain_id == target_chain_id:
            seq_list = []
            resseq_set = set(res.get_id()[1] for res in residues)
            min_resseq_struct = min(resseq_set, default=1)
            max_resseq_struct = max(resseq_set, default=0)
            max_resseq_missing = max((x[0] for x in missing_residues.get(chain_id, [])), default=0)
            resseq_max = max(max_resseq_struct, max_resseq_missing)

            for i in range(min_resseq_struct, resseq_max + 1):
                if i in resseq_set:
                    resname = next(res.get_resname() for res in residues if res.get_id()[1] == i)
                    seq_list.append(AA_CODE_MAP.get(resname, 'X'))
                elif chain_id in missing_residues and i in [x[0] for x in missing_residues[chain_id]]:
                    resname = next(x[1] for x in missing_residues[chain_id] if x[0] == i)
                    seq_list.append(AA_CODE_MAP.get(resname, 'X'))

            chain_seq = ''.join(seq_list).strip('X')

    return chain_seq


def download_pdb(pdb_id, id):
    # proxies = {
    #     "http": "http://127.0.0.1:1080",
    #     "https": "http://127.0.0.1:1080",
    # }

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    file_path = f"pdb{id}/{pdb_id}.pdb"

    while (True):
        try:
            # Download the PDB file
            # response = requests.get(url, proxies=proxies)
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as file:
                file.write(response.content)
            # print(f"Downloaded {pdb_id}.pdb")
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {pdb_id}.pdb: {e}")
            continue


def delete_pdb(pdb_id, chain_id, id):
    file_path = f"pdb{id}/{pdb_id}.pdb"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted {pdb_id}.pdb for {chain_id}")
    else:
        print(f"File {pdb_id}.pdb does not exist")


def process_entry(entry, chain_id, results, id):
    pdb_id = entry[0:4]

    pdb_file_path = download_pdb(pdb_id, id)

    if os.path.exists(pdb_file_path):
        try:
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb_id, pdb_file_path)

            # Extract and parse REMARK 465
            remark_465 = extract_remark_465(pdb_file_path)
            missing_residues = parse_remark_465(remark_465)

            # Get the full sequences for target chain
            chain_seq = extract_sequences(structure, chain_id, missing_residues)

            for index, row in results.iterrows():
                if row['PDB_ID'] == pdb_id:
                    if row['Chain1'] == chain_id:
                        results.at[index, 'Sequence1'] = chain_seq
                    elif row['Chain2'] == chain_id:
                        results.at[index, 'Sequence2'] = chain_seq
                    else:
                        NotImplementedError

            delete_pdb(pdb_id, chain_id, id)

        except Exception as e:
            logging.error(f'Failed to process {pdb_id}: {str(e)}')
    else:
        logging.error(f'PDB file {pdb_id}.pdb not found')


def main(id):
    # Load the PDB_ID list and corresponding chain ID
    df = pd.read_csv(f'contaminated_data/error_6A_results_batch_{id}.csv')
    # print(df)
    pdb_id_list = df['PDB_ID'].tolist()
    chain_id_list = df['Chain'].tolist()
    processed = []

    rs = pd.read_csv(f'raw_data/processed_6A_results_batch_{id}.csv')

    for i in range(len(pdb_id_list)):
        entry, chain_id = pdb_id_list[i], chain_id_list[i].upper()  # 6x85_D_F, D
        if {entry: chain_id} not in processed:
            processed.append({entry: chain_id})
            process_entry(entry, chain_id, rs, id)

        if i % 100 == 0:
            rs.to_csv(f'raw_data/corrected_processed_6A_results_batch_{id}.csv', index=False)
            print(f"Saving for i={i}")

    rs.to_csv(f'raw_data/corrected_processed_6A_results_batch_{id}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-id')

    args = parser.parse_args()

    print(int(args.id))

    if not os.path.exists(f"pdb{int(args.id)}"):
        os.makedirs(f"pdb{int(args.id)}")
    main(int(args.id))
