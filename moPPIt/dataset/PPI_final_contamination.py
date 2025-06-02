"""Final motif contamination after pulling full sequences"""
import pandas as pd
import blosum as bl
import ast
import pickle
import pandas as pd
from Bio import SeqIO
from math import ceil
from sklearn.model_selection import train_test_split
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse

def main(i):
    random.seed(42)

    blosum = bl.BLOSUM(62)
    def get_least_likely_substitution(residue):
        if residue not in blosum:
            return residue  # If residue is not in Blosum matrix, return it as is
        matrix_keys = list(blosum.keys())
        min_score = min(blosum[residue][r] for r in matrix_keys if r != '*' and r != 'J')
        least_likely_residues = [r for r in matrix_keys if r != '*' and r != 'J' and blosum[residue][r] == min_score]
        least_likely_residue = random.choice(least_likely_residues)
        return least_likely_residue


    df = pd.read_csv(f"raw_data/corrected_processed_6A_results_batch_{i}.csv")

    output_csv = f"contaminated_data/processed_6A_results_batch_{i}.csv"
    error_csv = f"contaminated_data/error_6A_results_batch_{i}.csv"

    new_rows = []
    error_rows = []

    for idx, row in df.iterrows():
        flag1 = False   # check whether there are errors when mutating Sequence1
        flag2 = False   # check whether there are errors when mutation Sequence2

        chain1 = row['Chain1'].upper()
        chain2 = row['Chain2'].upper()
        sequence1 = row['Sequence1']
        sequence2 = row['Sequence2']
        chain_1_motifs = ast.literal_eval(row['Chain_1_motifs'])
        chain_2_motifs = ast.literal_eval(row['Chain_2_motifs'])
        chain_1_offset = row['Chain_1_offset']
        chain_2_offset = row['Chain_2_offset']

        # Create a new entry by mutating sequence1
        sequence1_list = list(sequence1)
        modified_chain_1_motifs = []
        if len(chain_1_motifs) > 0:

            # Ignore entries where motif length equals sequence length cuz it'll be too hard for models to learn
            if len(chain_1_motifs) == len(sequence1):
                flag1 = True

            for motif in chain_1_motifs:
                res, pos = motif.split('_')
                # Errors for motifs or there are unalignments between sequence and motif
                if int(pos) >= len(sequence1) or int(pos) < 0 or res != sequence1[int(pos)]:
                    error_rows.append({
                        'PDB_ID': row['PDB_ID'] + '_' + chain1 + '_' + chain2,
                        'Chain': chain1,
                        'Sequence': sequence1,
                        'Error_motif': motif,
                        'Chain_offset': row['Chain_1_offset']
                    })
                    flag1 = True
                    break

                least_likely_residue = get_least_likely_substitution(res)
                sequence1_list[int(pos)] = least_likely_residue
                modified_chain_1_motifs.append(res + '_' + pos + '_' + least_likely_residue)

            # only save the entries without errors or do not need to be ignored
            if flag1 is False:
                modified_sequence1 = ''.join(sequence1_list)
                new_rows.append({
                    'PDB_ID': row['PDB_ID'] + '_' + chain1 + '_' + chain2,
                    'Chain1': chain1,
                    'Sequence1': modified_sequence1,
                    'Chain2': chain2,
                    'Sequence2': sequence2,
                    'Chain_1_motifs': str(modified_chain_1_motifs),
                    'Chain_2_motifs': row['Chain_2_motifs'],
                    'Chain_1_offset': row['Chain_1_offset'],
                    'Chain_2_offset': row['Chain_2_offset'],
                    'Modified_chain': chain1,
                    'Original_sequence': sequence1,
                })

        # if sequence2 is the same as sequence1 and so as the motifs, do not need to mutate sequence2
        if sequence1 == sequence2 and chain_1_motifs == chain_2_motifs:
            continue

        # Create a new entry by mutating sequence2, using the same logic as sequenc1
        if len(chain_2_motifs) > 0:
            if len(chain_2_motifs) == len(sequence2):
                flag2 == True
            sequence2_list = list(sequence2)
            modified_chain_2_motifs = []
            for motif in chain_2_motifs:
                res, pos = motif.split('_')
                if int(pos) >= len(sequence2) or int(pos) < 0 or res != sequence2[int(pos)]:
                    error_rows.append({
                        'PDB_ID': row['PDB_ID'] + '_' + chain1 + '_' + chain2,
                        'Chain': chain2,
                        'Sequence': sequence2,
                        'Error_motif': motif,
                        'Chain_offset': row['Chain_2_offset']
                    })
                    flag2 = True
                    break

                least_likely_residue = get_least_likely_substitution(res)
                sequence2_list[int(pos)] = least_likely_residue
                modified_chain_2_motifs.append(res + '_' + pos + '_' + least_likely_residue)

            if flag2 is False:
                modified_sequence2 = ''.join(sequence2_list)
                new_rows.append({
                    'PDB_ID': row['PDB_ID'] + '_' + chain2 + '_' + chain1,
                    'Chain1': chain1,
                    'Sequence1': sequence1,
                    'Chain2': chain2,
                    'Sequence2': modified_sequence2,
                    'Chain_1_motifs': row['Chain_1_motifs'],
                    'Chain_2_motifs': str(modified_chain_2_motifs),
                    'Chain_1_offset': row['Chain_1_offset'],
                    'Chain_2_offset': row['Chain_2_offset'],
                    'Modified_chain': chain2,
                    'Original_sequence': sequence2,
                })



    # Finished mutation
    new_df = pd.DataFrame(new_rows)

    # Deduplicate
    columns_to_check = ['Sequence1', 'Sequence2', 'Chain_1_motifs', 'Chain_2_motifs', 'Chain_1_offset', 'Chain_2_offset']
    deduplicated_new_df = new_df.drop_duplicates(subset=columns_to_check)
    print(f"Number of rows before deduplication: {len(new_df)}")
    print(f"Number of rows after deduplication: {len(deduplicated_new_df)}")

    deduplicated_new_df.to_csv(output_csv, index=False)

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(error_csv, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i')

    args = parser.parse_args()

    i_s = args.i    # 2,3,4,5,6,7,8,9,10

    for i in i_s.split(','):
        print(int(i))
        main(int(i))