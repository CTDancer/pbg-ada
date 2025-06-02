from Bio.PDB import PDBParser, NeighborSearch, PDBList
from Bio.PDB.Polypeptide import is_aa
from argparse import ArgumentParser
import numpy as np


def main(pdb_file_path):
    pdb_list = PDBList()

    # Dictionary to store residue information
    residue_info = {}

    # Load the PDB file using a PDBParser
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('multimer', pdb_file_path)

    for chain in structure.get_chains():
        chain_id = chain.get_id()
        for residue in chain:
            if is_aa(residue):
                residue_id = residue.get_id()[1]
                residue_key = (chain_id, residue_id)
                residue_info[residue_key] = {
                    "chain_id": chain_id,
                    "residue_id": residue_id,
                    "residue_name": residue.get_resname(),
                    "coordinates": residue["CA"].get_coord()
                }

    # Calculate distances between residues
    distance_results = {}
    for residue_key1, info1 in residue_info.items():
        for residue_key2, info2 in residue_info.items():
            if residue_key1 != residue_key2:
                distance = np.linalg.norm(info1["coordinates"] - info2["coordinates"])
                if info1["chain_id"] != info2["chain_id"]:
                    chain_dist = "Different chains"
                else:
                    chain_dist = "Same chain"
                distance_results[(residue_key1, residue_key2)] = (distance, chain_dist)

    # Set the distance threshold for interaction (e.g., 5 Å)
    interaction_distance_threshold = 10.0

    # List to store pairs of interacting residues
    interacting_residue_pairs = []
    chain_B = []

    # Iterate through distance_results to find interacting pairs
    for (residue_key1, residue_key2), (distance, chain_dist) in distance_results.items():
        if chain_dist == "Different chains" and distance <= interaction_distance_threshold:
            interacting_residue_pairs.append((residue_key1, residue_key2, distance))
            if residue_key1[0] == 'B':
                chain_B.append(residue_key1[1])

    # Print or process the interacting_residue_pairs list as needed
    # for pair in interacting_residue_pairs:
    #     print("Interacting Pair:", pair)

    print(sorted(list(set(chain_B))))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-path', type=str, required=True)

    args = parser.parse_args()
    main(args.path)
