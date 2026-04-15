import os
from rdkit import Chem
from tqdm import tqdm
import itertools
import pickle
import argparse
from scipy.spatial import cKDTree
import numpy as np
import os
from rdkit import Chem
from tqdm import tqdm
import itertools
import pickle
from Bio.PDB import PDBParser, PDBIO, Select
import shutil


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
    mol = Chem.MolFromMolFile(sdf_file)
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

    parser = argparse.ArgumentParser(description='Process BindingNetV2 dataset.')
    parser.add_argument('--level', type=str, choices=['high', 'moderate', 'low'], 
                       default='high', help='Quality level of the dataset')
    parser.add_argument('--raw_dataset_path', type=str, required=True,
                       help='Root path of the dataset')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path of the dataset')
    args = parser.parse_args()
    
    level = args.level  # high, moderate, low

    root_path = os.path.join(args.raw_dataset_path, level)
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if os.path.exists(f"{output_path}/{level}_molecule_dict.pkl"):
        with open(f"{output_path}/{level}_molecule_dict.pkl", "rb") as f:
            molecule_dict = pickle.load(f)
    else:
        molecule_dict = {}
        for sub_target_path in tqdm(os.listdir(root_path)):
            target_path = os.path.join(root_path, sub_target_path)
            target_id = int(sub_target_path[13:])
            for sub_pair_path in os.listdir(target_path):
                pair_path = os.path.join(target_path, sub_pair_path)
                pair_id = int(sub_pair_path[6:])
                sdf_path = os.path.join(pair_path, 'ligand.sdf')
                # pdb_path = os.path.join(pair_path, 'protein.pdb')
                rdmol = Chem.MolFromMolFile(sdf_path)
                inchi_key = Chem.MolToInchiKey(rdmol)


                molecule_dict.setdefault(inchi_key, {})
                if target_id not in molecule_dict[inchi_key]:
                    molecule_dict[inchi_key][target_id] = f'{sub_target_path}/{sub_pair_path}'

        with open(f"{output_path}/{level}_molecule_dict.pkl", "wb") as f:
            pickle.dump(molecule_dict, f)

    print('Total ligands:', len(molecule_dict))

    cnt1 = sum(1 for mol_dict in molecule_dict.values() if len(mol_dict) > 1)
    print('Dual ligands:', cnt1)
    cnt2 = sum(len(mol_dict.values()) for mol_dict in molecule_dict.values() if len(mol_dict) > 1)
    print('Dual pockets:', cnt2)

    if os.path.exists(f"{output_path}/{level}_dual_pairs.pkl"):
        with open(f"{output_path}/{level}_dual_pairs.pkl", "rb") as f:
            dual_pairs = pickle.load(f)
    else:
        dual_pairs = []
        for inchi_key, mol_dict in tqdm(molecule_dict.items()):
            if len(mol_dict) > 1:
                mol_list = list(mol_dict.values())
                # print(inchi_key)
                # print(mol_list)
                combinations = list(itertools.combinations(range(len(mol_list)), 2))
                # print(combinations)
                for comb in combinations:
                    # print(comb)
                    # print(mol_list[comb[0]], mol_list[comb[1]])
                    # input()
                    mol_1_path = mol_list[comb[0]]
                    mol_2_path = mol_list[comb[1]]
                    mol_1 = Chem.MolFromMolFile(os.path.join(root_path, mol_1_path, 'ligand.sdf'))
                    mol_2 = Chem.MolFromMolFile(os.path.join(root_path, mol_2_path, 'ligand.sdf'))
                    if mol_1.HasSubstructMatch(mol_2) and mol_2.HasSubstructMatch(mol_1):
                        dual_pairs.append((mol_list[comb[0]], mol_list[comb[1]]))

        with open(f"{output_path}/{level}_dual_pairs.pkl", "wb") as f:
            pickle.dump(dual_pairs, f)

    print('Dual pairs:', len(dual_pairs))
    print('Dual pairs example:', dual_pairs[:10])


    for pair in tqdm(dual_pairs):

        for rel_dir in pair:
            src_dir = os.path.join(root_path, rel_dir)
            dst_dir = os.path.join(output_path, rel_dir)

            pocket_file = os.path.join(dst_dir, 'pocket_10A_clean.pdb')

            if os.path.exists(pocket_file):
                continue

            os.makedirs(dst_dir, exist_ok=True)

            src_ligand = os.path.join(src_dir, 'ligand.sdf')
            src_protein = os.path.join(src_dir, 'protein.pdb')
            dst_ligand = os.path.join(dst_dir, 'ligand.sdf')
            dst_protein = os.path.join(dst_dir, 'protein.pdb')

            shutil.copyfile(src_ligand, dst_ligand)
            shutil.copyfile(src_protein, dst_protein)

            clean_protein = os.path.join(dst_dir, 'protein_clean.pdb')
            cmd = '(echo "HEADER    POCKET"; echo "COMPND    POCKET"; ' \
                'grep ATOM "%s" | grep -v "H  $"; echo "END   ") > "%s"' % (
                    dst_protein,
                    clean_protein
                )
            os.system(cmd)

            extract_pocket(
                clean_protein,
                dst_ligand,
                pocket_file
            )
