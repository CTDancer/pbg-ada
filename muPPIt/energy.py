from pyrosetta import *
init()

from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.protocols import *
from pyrosetta.rosetta.core.select import *
import os

def pack(pose, chain, posi, amino, scorefxn):

    # Select Mutate Position
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(pose.pdb_info().pdb2pose(chain, posi))
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(mut_posi.apply(pose)))

    # Select Neighbor Position
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose)))

    # Select No Design Area
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(not_design.apply(pose)))

    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    # Disable design
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    # Enable design
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))

    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    #Perform The Move
    if not os.getenv("DEBUG"):
      packer.apply(pose)


PDB = '/home/tc415/muPPIt/pdbs/fold_mutant_mfpecmairvrnqcpwcydgrery_model_0'

cleanATOM(f'{PDB}.pdb')
pose = pose_from_pdb(f'{PDB}.clean.pdb')

sf = create_score_function("ref2015")
sf.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
sf.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
sf.set_weight(rosetta.core.scoring.angle_constraint, 1)

mmap = MoveMap()
mmap.set_bb(True)   
mmap.set_chi(True)   
mmap.set_jump(True) 

relax = rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(sf)
relax.max_iter(200)
relax.dualspace(True)
relax.set_movemap(mmap)

relax.apply(pose)

# Save
pose.dump_pdb(f'{PDB}.relax.pdb')

#Load the previously-relaxed pose.
relaxPose = pose_from_pdb(f"{PDB}.relax.pdb")

#Clone it
original = relaxPose.clone()
scorefxn = get_score_function()
print("\nEnergy:", scorefxn(original),"\n")

# pack(relaxPose,'B', 15+16, 'S', scorefxn)
# relax.apply(relaxPose)
# print("\nNew Energy:", scorefxn(relaxPose),"\n")
# print("\nOld Energy:", scorefxn(original),"\n")

os.remove(f"{PDB}.relax.pdb")
os.remove(f"{PDB}.clean.pdb")
