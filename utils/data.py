import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig


atom_families = [
    "Acceptor", "Donor", "Aromatic", "Hydrophobe", 
    "LumpedHydrophobe", "NegIonizable", "PosIonizable", "ZnBinder"
]
atom_families_id = {f: i for i, f in enumerate(atom_families)}
bond_types = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

class PDBProtein(object):
    aa_name_sym = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
        "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
        "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    aa_name_number = {k: i for i, (k, _) in enumerate(aa_name_sym.items())}
    backbone_names = ["CA", "C", "N", "O"]

    def __init__(self, data, mode="auto"):
        super().__init__()
        if (data[-4:].lower() == ".pdb" and mode == "auto") or mode == "path":
            with open(data, "r") as f:
                self.block = f.read()
        else:
            self.block = data
        self.periodtable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == "ATOM":
                element_sym = line[76:78].strip().capitalize()
                if len(element_sym) == 0:
                    element_sym = line[13:14]
                if element_sym != 'H':
                    yield {
                        "type": "ATOM",
                        "line": line,
                        "element_sym": element_sym, 
                        "atom_id": int(line[6:11]),
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "chain": line[21:22].strip(),
                        "res_id": int(line[22:26]),
                        "res_insert_id": line[26:27].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occupacy": float(line[54:60]),
                        "segment": line[72:76].strip(),
                        "charge": line[78:80].strip(),
                    }
            elif line[0:6].strip() == "HETATM" and line[17:20].strip() != "HOH":
                element_sym = line[76:78].strip().capitalize()
                if len(element_sym) == 0:
                    element_sym = line[12:14]
                if element_sym != 'H':
                    yield {
                        "type": "HETATM",
                        "line": line,
                        "element_sym": element_sym, 
                        "atom_id": int(line[6:11]),
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "chain": line[21:22].strip(),
                        "res_id": int(line[22:26]),
                        "res_insert_id": line[26:27].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occupacy": float(line[54:60]),
                        "segment": line[72:76].strip(),
                        "charge": line[78:80].strip(),
                    }
            elif line[0:6].strip() == "HEADER":
                yield {
                    "type": "HEADER",
                    "value": line[10:].strip()
                }
            elif line[0:6].strip() == "ENDMDL":
                break

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom["type"] == "HEADER":
                self.title = atom["value"].lower()
                continue
            elif atom["type"] == "ATOM":
                self.atoms.append(atom)
                atomic_number = self.periodtable.GetAtomicNumber(atom["element_sym"])
                num_atoms = len(self.element)
                self.element.append(atomic_number)
                self.atomic_weight.append(self.periodtable.GetAtomicWeight(atomic_number))
                self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
                self.atom_name.append(atom["atom_name"])
                self.is_backbone.append(atom["atom_name"] in self.backbone_names)
                self.atom_to_aa_type.append(self.aa_name_number[atom["res_name"]])
            elif atom["type"] == "HETATM":
                self.atoms.append(atom)
                atomic_number = self.periodtable.GetAtomicNumber(atom["element_sym"])
                num_atoms = len(self.element)
                self.element.append(atomic_number)
                self.atomic_weight.append(self.periodtable.GetAtomicWeight(atomic_number))
                self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
                self.atom_name.append(atom["atom_name"])
                self.is_backbone.append(atom["atom_name"] in self.backbone_names)
                self.atom_to_aa_type.append(int(len(self.aa_name_sym)))

            chain_res_id = "%s_%s_%d_%s" % (atom["chain"], atom["segment"], atom["res_id"], atom["res_insert_id"])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    "name": atom["res_name"],
                    "atoms": [num_atoms],
                    "chain": atom["chain"],
                    "segment": atom["segment"],
                }
            else:
                assert residues_tmp[chain_res_id]["name"] == atom["res_name"]
                assert residues_tmp[chain_res_id]["chain"] == atom["chain"]
                residues_tmp[chain_res_id]["atoms"].append(num_atoms)
        
        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue["atoms"]:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.backbone_names:
                    residue[f"pos_{self.atom_name[atom_idx]}"] = self.pos[atom_idx]
            residue["center_of_mass"] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            if residue["name"] in self.aa_name_sym:
                self.amino_acid.append(self.aa_name_number[residue["name"]])
            else:
                self.amino_acid.append(int(len(self.aa_name_sym)))
            self.center_of_mass.append(residue["center_of_mass"])
            for name in self.backbone_names:
                pos_key = f"pos_{name}"
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue["center_of_mass"])

    def to_dict_atom(self):
        return {
            "element": np.array(self.element, dtype=np.int64),
            "molecule_name": self.title,
            "pos": np.array(self.pos, dtype=np.float32),
            "is_backbone": np.array(self.is_backbone, dtype=bool),
            "atom_name": self.atom_name,
            "atom_to_aa_type": np.array(self.atom_to_aa_type, dtype=np.int64),
        }

    def to_dict_residue(self):
        return {
            "amino_acid": np.array(self.amino_acid, dtype=np.int64),
            "center_of_mass": np.array(self.center_of_mass, dtype=np.float32),
            "pos_CA": np.array(np.pos_CA, dtype=np.float32),
            "pos_C": np.array(np.pos_C, dtype=np.float32),
            "pos_N": np.array(np.pos_N, dtype=np.float32),
            "pos_O": np.array(np.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion="center_of_mass"):
        center = np.array(center).reshape(3)
        select = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            if distance < radius:
                select.append(residue)
        return select

    def query_residues_ligand(self, ligand, radius, criterion="center_of_mass"):
        select = []
        sel_idx = set()
        for center in ligand["pos"]:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    select.append(residue)
                    sel_idx.add(i)
        return select

    def residues_to_pdb_block(self, residues, name="pocket"):
        block = f"HEADER    {name}\n"
        block += f"COMPND    {name}\n"
        for residue in residues:
            for atom_idx in residue["atoms"]:
                block += self.atoms[atom_idx]["line"] + "\n"
        block += "END\n"
        return block

def parse_lig_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        mol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError(f"Unknown ligand file, it has to be sdf or mol2 file!")

    # sort atoms
    mol_for_rank = Chem.Mol(mol)
    Chem.SanitizeMol(mol_for_rank)
    ranks = list(Chem.CanonicalRankAtoms(mol_for_rank, breakTies=True, includeChirality=True))
    atom_order = sorted(range(mol_for_rank.GetNumAtoms()), key=lambda i: ranks[i])
    mol = Chem.RenumberAtoms(mol, atom_order)

    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    feat_mat = np.zeros([num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx
    hybrid = []
    for atom in mol.GetAtoms():
        hyb = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybrid.append((idx, hyb))
    hybrid = sorted(hybrid)
    hybrid = [h[1] for h in hybrid]

    # Get element and center of mass
    periodtable = Chem.GetPeriodicTable()
    pos = np.array(mol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    sum_pos = 0
    sum_mass = 0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = periodtable.GetAtomicWeight(atom_num)
        sum_pos += pos[atom_idx] * atom_weight
        sum_mass += atom_weight
    center_of_mass = sum_pos / sum_mass
    element = np.array(element, dtype=int)

    # Get edge type
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()]]
    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    
    data = {
        "smiles": Chem.MolToSmiles(mol),
        "element": element,
        "pos": pos,
        "bond_index": edge_index,
        "bond_type": edge_type,
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "center_of_mass": center_of_mass,
        "atom_feature": feat_mat,
        "hybridization": hybrid
    }
    return data

def parse_drug3d_mol(path):
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        mol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError(f"Unknown ligand file, it has to be sdf or mol2 file!")
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    feat_mat = np.zeros([num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx
    hybrid = []
    for atom in mol.GetAtoms():
        hyb = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybrid.append((idx, hyb))
    hybrid = sorted(hybrid)
    hybrid = [h[1] for h in hybrid]

    conf = mol.GetConformer()
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        ele = atom.GetAtomicNum()
        pos_list.append(list(pos))
        ele_list.append(ele)
    
    row, col = [], []
    bond_type = []
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        b_index = [
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ]
        bond_type += 2*[b_type]
        row += [b_index[0], b_index[1]]
        col += [b_index[1], b_index[0]]
    
    bond_type = np.array(bond_type, dtype=np.int64)
    bond_index = np.array([row, col],dtype=np.int64)

    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]

    data = {
        'element': np.array(ele_list, dtype=np.int64),
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        "atom_feature": feat_mat,
        "hybridization": hybrid
    }
    return data
