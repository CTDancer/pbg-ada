import requests
import pandas as pd
from tqdm import tqdm
import pdb

df = pd.read_csv('/home/tc415/muPPIt/dataset/biolip/BioLiP.csv')
pdb_list = df.iloc[:, 0].tolist()
peptide_chain_list = df.iloc[:, 5].tolist()
peptide_list = [None] * len(pdb_list)

for i in tqdm(range(len(pdb_list))):
    successful = False
    pdb_id = pdb_list[i]
    response = requests.get(f'https://www.rcsb.org/fasta/entry/{pdb_id}/display')

    # if i < 2:
    #     pdb.set_trace()

    if response.status_code == 200:
        fasta_content = response.text
        lines = fasta_content.split('\n')
        for j in range(len(lines)):
            line = lines[j]
            if line.startswith('>'):
                chain = line.split('|')[1]
                assert 'Chain' in chain
                if 'Chains' in chain:
                    candidates = chain.split('Chains')[1].split(',')
                else:
                    candidates = chain.split('Chain')[1].split(',')
                candidates = [candidate.strip() for candidate in candidates]
                for k in range(len(candidates)):
                    if '[' in candidates[k]:
                        candidates[k] = candidates[k].split(']')[0][-1]

                if peptide_chain_list[i] in candidates:
                    peptide_list[i] = lines[j+1]
                    successful = True
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")
        continue

    if not successful:
        print(f"No matching chain found for {pdb_id} in BioLiP dataset. Line {i}")

new_col = pd.DataFrame(peptide_list, columns=['peptide'])

df['peptide'] = new_col

df.to_csv('/home/tc415/muPPIt/dataset/biolip/BioLiP_with_peptides.csv', index=False)
