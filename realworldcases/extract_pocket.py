from rdkit import Chem
from scipy.spatial import cKDTree
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO, Select


class PocketResiduesSelect(Select):
    def __init__(self, ligand_coords, cutoff=10.0, structure=None):
        self.cutoff = cutoff
        self.ligand_coords = ligand_coords
        self.hit_residues = set()
        if structure is not None:
            self.precompute_hit_residues(structure)

    def precompute_hit_residues(self, structure):

        atom_coords = []
        atom_to_residue = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0].strip() != '':
                        continue
                    residue_id = residue.full_id
                    for atom in residue:
                        coord = atom.coord
                        atom_coords.append(coord)
                        atom_to_residue[len(atom_coords) - 1] = residue_id
        
        if not atom_coords:
            return
        
        ligand_tree = cKDTree(self.ligand_coords)
        atom_coords_arr = np.array(atom_coords)
        
        indices = ligand_tree.query_ball_point(atom_coords_arr, self.cutoff)
        
        for i, res_indices in enumerate(indices):
            if res_indices:
                residue_id = atom_to_residue[i]
                self.hit_residues.add(residue_id)

    def accept_residue(self, residue):
        
        if residue.id[0].strip() != '':
            return False
        
        return residue.full_id in self.hit_residues
    
def extract_ligand_coords(sdf_file):
    """Extracts all atom coordinates from a ligand in an SDF file."""
    mol = Chem.MolFromMol2File(sdf_file, removeHs=True)
    if mol is None:
        raise ValueError(f"Could not read ligand SDF file: {sdf_file}")
    conf = mol.GetConformer()
    return [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

def extract_pocket(pdb_file, sdf_file, output_file="pocket.pdb"):
    try:
        ligand_coords = extract_ligand_coords(sdf_file)
        
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)
        
        selector = PocketResiduesSelect(ligand_coords, cutoff=10.0, structure=structure)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_file, selector)
        
    except Exception as e:
        raise e


if __name__ == '__main__':

    clean_protein = "realworldcases/GSK3b_JNK3/JNK3_4WHZ_clean.pdb"
    dst_ligand = "realworldcases/GSK3b_JNK3/4WHZ_3NL.mol2"
    pocket_file = "realworldcases/GSK3b_JNK3/JNK3_pocket_10A.pdb"

    extract_pocket(
        clean_protein,
        dst_ligand,
        pocket_file
    )
