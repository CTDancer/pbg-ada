from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

def get_close_residues(pdb_file, protein_chain_id, peptide_chain_id, distance_threshold=3.5):
    # Initialize the parser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Assume single model; adjust if multiple models exist
    model = structure[0]
    
    # Extract protein and peptide chains
    protein_chain = model[protein_chain_id]
    peptide_chain = model[peptide_chain_id]
    
    # Extract all atoms from peptide
    peptide_atoms = [atom for atom in peptide_chain.get_atoms()]
    
    # Create a NeighborSearch object with all atoms in the structure
    all_atoms = list(model.get_atoms())
    neighbor_search = NeighborSearch(all_atoms)
    
    # Set to store unique residues
    close_residues = set()
    
    # Iterate through each atom in the peptide and find nearby atoms in the protein
    for atom in peptide_atoms:
        nearby_atoms = neighbor_search.search(atom.coord, distance_threshold, level='A')  # 'A' for atoms
        for nearby_atom in nearby_atoms:
            # Check if the nearby atom is in the protein chain
            if nearby_atom.get_parent().get_parent().id == protein_chain_id:
                residue = nearby_atom.get_parent()
                if is_aa(residue, standard=True):
                    close_residues.add(residue)
    
    # Sort residues by their sequence position
    sorted_residues = sorted(close_residues, key=lambda r: r.get_id()[1])
    
    # Prepare a list of tuples (residue index, residue name)
    result = [(res.get_id()[1], res.get_resname()) for res in sorted_residues]
    
    return result

if __name__ == "__main__":
    pdb_file = "/home/tc415/discrete-diffusion-guidance/pdbs/UCLH5_docked.pdb"  # Replace with your PDB file path
    protein_chain_id = "A"  # Replace with your protein chain ID
    peptide_chain_id = "B"  # Replace with your peptide chain ID
    
    close_residues = get_close_residues(pdb_file, protein_chain_id, peptide_chain_id, distance_threshold=3.5)
    
    print("Amino acids in the protein within 3.5 Å of the peptide:")
    for res_id, res_name in close_residues:
        print(f"Residue {res_id}: {res_name}")
    print([res_id for res_id, _ in close_residues])
