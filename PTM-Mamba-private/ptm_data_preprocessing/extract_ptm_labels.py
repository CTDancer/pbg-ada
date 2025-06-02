import re
import pandas as pd
from tqdm import tqdm
import requests

def extract_AC_ID(lines, i):
    """
    extract the AC ID from the line
    """
    if lines[i].startswith('AC'):
        pattern = re.compile(r'AC\s+([A-Z0-9]+);')
        match = pattern.match(lines[i])
        if match:
            return {'AC_ID':match.group(1)}
    else:
        return {}

def extract_ptm_labels(lines, i,):
    """
    extract the PTM labels from the line
    """
    ret_dict = {}
    pos = None
    label = None
    label_pattern = re.compile(r'FT\s+/note="(.+)"')
    if lines[i].startswith('FT') and ".." not in lines[i]:
        pseudo_pos = lines[i].split()[-1]
        if re.match(r'^\d+$', pseudo_pos):
            pos = int(pseudo_pos)
            
        if lines[i+1].startswith('FT') and "/note=" in lines[i+1]:
            match = label_pattern.match(lines[i+1])
            if match:
                label = match.group(1)
        if pos and label:
            ret_dict.update({pos:label})
    return ret_dict



def split_db(lines,label_extct_fns,csv_path):
    """
    split the uniprot_sprot.dat database into individual entries.
    """
    pattern = re.compile(r'AC\s+([A-Z0-9]+);')
    entry_dict = {}
    df = pd.DataFrame()
    num_lines = len(lines)
    for i in tqdm(range(num_lines)):
        ac_id = None
        if lines[i].startswith('ID'):
            label_dict = {}
            for j in range(i+1, num_lines):
                if lines[j].startswith('AC'):
                    ac_id = lines[j].split(";")[0][5:]
                for f in label_extct_fns:
                    l_dict = f(lines, j)
                    label_dict.update(l_dict)
                if lines[j] == '//':
                    i = j
                    break
        if ac_id and label_dict:
            entry_dict.update({ac_id:label_dict})
            df = pd.concat([df,pd.DataFrame.from_dict({
                "AC_ID":[ac_id]*len(label_dict),
                "pos":list(label_dict.keys()),
                "label":list(label_dict.values())
            })], ignore_index=True, axis=0)
            df.to_csv(csv_path)
    return df, entry_dict



def main(sprot_dat_path, csv_path):
    """
    extract the PTM labels from the uniprot_sprot.dat database
    """
    label_extct_fns = [extract_ptm_labels]
    with open(sprot_dat_path, 'r') as f:
        lines = f.read().splitlines()
    entry_dict = split_db(lines, label_extct_fns, csv_path)
    return entry_dict

def download():
    dat_url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
    fasta_url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz'
    
    dat_destination_path = 'uniprot_sprot.dat.gz'
    fasta_destination_path = 'uniprot_sprot.fasta.gz'

    # Download DAT file
    print(f"Downloading {dat_url} to {dat_destination_path}")
    with requests.get(dat_url, stream=True) as response:
        with open(dat_destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print("Download completed.")

    # Download FASTA file
    print(f"Downloading {fasta_url} to {fasta_destination_path}")
    with requests.get(fasta_url, stream=True) as response:
        with open(fasta_destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print("Download completed.")


if __name__ == '__main__':
    sprot_dat_path = 'uniprot_sprot.dat'
    csv_path = 'sprot_labels.csv'
    df, entry_dict = main(sprot_dat_path, csv_path)
    df.to_csv(csv_path)
    print(entry_dict)