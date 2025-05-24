from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

def calculate_min_distance(residue_atoms, peptide_atoms):
    """
    Calculate the minimum distance between any atom in the residue and any atom in the peptide.
    """
    min_distance = np.inf
    for atom1 in residue_atoms:
        for atom2 in peptide_atoms:
            distance = atom1 - atom2  # Biopython allows direct distance calculation
            if distance < min_distance:
                min_distance = distance
    return min_distance

def calculate_centroid_distance(residue_atoms, peptide_atoms):
    """
    Calculate the distance between the centroids of the residue and the peptide.
    """
    residue_coords = np.array([atom.get_coord() for atom in residue_atoms])
    peptide_coords = np.array([atom.get_coord() for atom in peptide_atoms])
    
    residue_centroid = residue_coords.mean(axis=0)
    peptide_centroid = peptide_coords.mean(axis=0)
    
    centroid_distance = np.linalg.norm(residue_centroid - peptide_centroid)
    return centroid_distance

def get_peptide_atoms(model, peptide_chain_id):
    """
    Extract all atoms from the peptide chain.
    """
    peptide_chain = model[peptide_chain_id]
    peptide_atoms = list(peptide_chain.get_atoms())
    return peptide_atoms

def get_residue_atoms(model, protein_chain_id, residue_indices):
    """
    Extract atoms from specified residues in the protein chain.
    """
    protein_chain = model[protein_chain_id]
    residues = []
    for res in protein_chain:
        res_id = res.get_id()[1]
        if res_id in residue_indices and is_aa(res, standard=True):
            residues.append(res)
    return residues

def compute_average_distance(pdb_file, protein_chain_id, peptide_chain_id, residue_indices, distance_type='min'):
    """
    Compute the average distance of specified residues to the peptide.
    
    Parameters:
    - pdb_file: Path to the PDB file.
    - protein_chain_id: Chain ID of the protein.
    - peptide_chain_id: Chain ID of the peptide.
    - residue_indices: List of residue numbers on the protein.
    - distance_type: 'min' for minimum distance, 'centroid' for centroid distance.
    
    Returns:
    - average_distance: The average distance across specified residues.
    - detailed_distances: Dictionary with residue indices as keys and distances as values.
    """
    # Initialize the parser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Assume single model; adjust if multiple models exist
    model = structure[0]
    
    # Extract peptide atoms
    peptide_atoms = get_peptide_atoms(model, peptide_chain_id)
    
    # Extract specified residues
    residues = get_residue_atoms(model, protein_chain_id, residue_indices)
    
    if not residues:
        raise ValueError("No valid residues found with the provided indices.")
    
    detailed_distances = {}
    
    for res in residues:
        res_id = res.get_id()[1]
        res_atoms = list(res.get_atoms())
        
        if distance_type == 'min':
            distance = calculate_min_distance(res_atoms, peptide_atoms)
        elif distance_type == 'centroid':
            distance = calculate_centroid_distance(res_atoms, peptide_atoms)
        else:
            raise ValueError("Invalid distance_type. Choose 'min' or 'centroid'.")
        
        detailed_distances[res_id] = distance
    
    average_distance = np.mean(list(detailed_distances.values()))
    
    return average_distance, detailed_distances

def parse_motif(motif: str) -> list:
    parts = motif.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    result = [i+1 for i in result]
    return result

if __name__ == "__main__":
    # Parameters (Modify these as needed)
    pdb_file = "/home/tc415/discrete-diffusion-guidance/pdbs/UBC9_docked.pdb"       # Path to your PDB file
    protein_chain_id = "A"                 # Protein chain ID
    peptide_chain_id = "B"                 # Peptide chain ID
    residue_indices = parse_motif('123-127')   # Replace with your residue numbers
    distance_type = 'min'                  # 'min' or 'centroid'
    
    try:
        avg_dist, dist_details = compute_average_distance(
            pdb_file,
            protein_chain_id,
            peptide_chain_id,
            residue_indices,
            distance_type=distance_type
        )
        
        print(f"Distance Type: {distance_type.capitalize()} Distance")
        print("Individual Residue Distances:")
        for res_id in sorted(dist_details):
            print(f"Residue {res_id}: {dist_details[res_id]:.2f} Å")
        
        print(f"\nAverage Distance for specified residues: {avg_dist:.2f} Å")
    
    except Exception as e:
        print(f"Error: {e}")
