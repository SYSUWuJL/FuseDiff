import os
import pickle
import lmdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from utils.data import PDBProtein, parse_lig_file, parse_drug3d_mol
from typing import Optional
from rdkit import Chem
from utils.evaluation import scoring_func


def to_torch_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

class ProteinLigandData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def protein_ligand_dicts(protein_dict=None, ligand_dict=None, frag_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)
        if protein_dict is not None:
            for k, v in protein_dict.items():
                instance["protein_" + k] = v
        if ligand_dict is not None:
            for k, v in ligand_dict.items():
                instance["ligand_" + k] = v
        if frag_dict is not None:
            for k, v in frag_dict.items():
                instance["frag_" + k] = v

        instance["ligand_nbh_list"] = {
            i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) 
            if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]
        }
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == "ligand_bond_index":
            return self["ligand_element"].size(0)
        else:
            return super().__inc__(key, value)


class DualPLData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def from_protein_ligand_dicts_dualdata(ligand_dict=None, protein_dict1=None, protein_dict2=None, 
                                           ligand_dict1=None, ligand_dict2=None, 
                                           merged_protein=None, virtual_ligand=None, 
                                           **kwargs):
        instance = ProteinLigandData(**kwargs)

        # if ligand_dict is not None:
        #     for key, item in ligand_dict.items():
        #         instance['ligand_' + key] = item
                
        if protein_dict1 is not None:
            for key, item in protein_dict1.items():
                instance['protein1_' + key] = item

        if protein_dict2 is not None:
            for key, item in protein_dict2.items():
                instance['protein2_' + key] = item

        if ligand_dict1 is not None:
            for key, item in ligand_dict1.items():
                instance['ligand1_' + key] = item

        if ligand_dict2 is not None:
            for key, item in ligand_dict2.items():
                instance['ligand2_' + key] = item

        if virtual_ligand is not None:
            for key, item in virtual_ligand.items():
                instance['ligand_' + key] = item

        if merged_protein is not None:
            for key, item in merged_protein.items():
                instance['merged_protein_' + key] = item


        if getattr(instance, 'ligand1_element', None) is not None:
            assert torch.equal(instance.ligand1_element, instance.ligand2_element), (
                f'ligand_element mismatch\n'
                f'ligand1_element: {instance.ligand1_element.tolist()}\n'
                f'ligand2_element: {instance.ligand2_element.tolist()}\n'
                f'shape1: {instance.ligand1_element.shape}, shape2: {instance.ligand2_element.shape}'
            )
            instance['ligand_element'] = instance.ligand1_element
            del instance.ligand1_element
            del instance.ligand2_element

        if getattr(instance, 'ligand1_bond_index', None) is not None:
            assert torch.equal(instance.ligand1_bond_index, instance.ligand2_bond_index), (
                f'ligand_bond_index mismatch\n'
                f'ligand1_bond_index: {instance.ligand1_bond_index.tolist()}\n'
                f'ligand2_bond_index: {instance.ligand2_bond_index.tolist()}\n'
                f'shape1: {instance.ligand1_bond_index.shape}, shape2: {instance.ligand2_bond_index.shape}'
            )
            instance['ligand_bond_index'] = instance.ligand1_bond_index
            del instance.ligand1_bond_index
            del instance.ligand2_bond_index

        if getattr(instance, 'ligand1_bond_type', None) is not None:
            assert torch.equal(instance.ligand1_bond_type, instance.ligand2_bond_type), (
                f'ligand_bond_type mismatch\n'
                f'ligand1_bond_type: {instance.ligand1_bond_type.tolist()}\n'
                f'ligand2_bond_type: {instance.ligand2_bond_type.tolist()}\n'
                f'shape1: {instance.ligand1_bond_type.shape}, shape2: {instance.ligand2_bond_type.shape}'
            )
            instance['ligand_bond_type'] = instance.ligand1_bond_type
            del instance.ligand1_bond_type
            del instance.ligand2_bond_type

        if getattr(instance, 'ligand1_atom_feature', None) is not None:
            assert torch.equal(instance.ligand1_atom_feature, instance.ligand2_atom_feature), (
                f'ligand_atom_feature mismatch\n'
                f'ligand1_atom_feature: {instance.ligand1_atom_feature.tolist()}\n'
                f'ligand2_atom_feature: {instance.ligand2_atom_feature.tolist()}\n'
                f'shape1: {instance.ligand1_atom_feature.shape}, shape2: {instance.ligand2_atom_feature.shape}'
            )
            instance['ligand_atom_feature'] = instance.ligand1_atom_feature
            del instance.ligand1_atom_feature
            del instance.ligand2_atom_feature
            
        if getattr(instance, 'ligand1_hybridization', None) is not None:
            assert torch.equal(instance.ligand1_hybridization, instance.ligand2_hybridization), (
                f'ligand_hybridization mismatch\n'
                f'ligand1_hybridization: {instance.ligand1_hybridization.tolist()}\n'
                f'ligand2_hybridization: {instance.ligand2_hybridization.tolist()}\n'
                f'len1: {len(instance.ligand1_hybridization)}, len2: {len(instance.ligand2_hybridization)}'
            )
            instance['ligand_hybridization'] = instance.ligand1_hybridization
            del instance.ligand1_hybridization
            del instance.ligand2_hybridization

        if getattr(instance, 'ligand1_smiles', None) is not None:
            instance['ligand_smiles'] = instance.ligand1_smiles
            del instance.ligand1_smiles
            del instance.ligand2_smiles
            
        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                if instance.ligand_bond_index[0, k].item() == i]
                                    for i in instance.ligand_bond_index[0]}

        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == "ligand_bond_index":
            return self["ligand_element"].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataset(Dataset):
    def __init__(self, path, transform=None, version="final"):
        super().__init__()
        self.path = path.rstrip('/')
        self.index_path = os.path.join(self.path, "index.pkl")
        self.processed_path = os.path.join(
            os.path.dirname(self.path), os.path.basename(self.path) + f"_processed_{version}.lmdb"
        )
        self.transform = transform
        self.database = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f"{self.processed_path} does not exist, start to process data!")
            self._process()

    def _process(self):
        database = lmdb.open(
            self.processed_path, map_size=10*(1024*1024*1024),
            create=True, subdir=False, readonly=False,
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)
        
        num_skip = 0
        with database.begin(write=True, buffers=True) as db:
            for i, (pocket_fn, _, ligand_fn, _, logp, tpsa, sa, qed, aff, _) in enumerate(tqdm(index)):
                if pocket_fn is not None:
                    try:
                        pocket_dict = PDBProtein(os.path.join(self.path, pocket_fn)).to_dict_atom()
                        ligand_dict = parse_drug3d_mol(os.path.join(self.path, ligand_fn))
                        data = ProteinLigandData.protein_ligand_dicts(
                            protein_dict=to_torch_dict(pocket_dict),
                            ligand_dict=to_torch_dict(ligand_dict)
                        )
                        data.protein_filename = pocket_fn
                        data.ligand_filename = ligand_fn
                        data.logp = logp
                        data.tpsa = tpsa
                        data.sa = sa
                        data.qed = qed
                        data.aff = aff
                        data = data.to_dict()
                        db.put(key=str(i).encode(), value=pickle.dumps(data))
                    except:
                        num_skip += 1
                        print(f"Skipping {num_skip} {pocket_fn} {ligand_fn}!")
                        continue
                else:
                    continue
        database.close()

    def _build_db(self):
        assert self.database is None
        self.database = lmdb.open(
            self.processed_path, map_size=10*(1024*1024*1024), create=False, 
            subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
        )
        with self.database.begin() as db:
            self.keys = list(db.cursor().iternext(values=False))

    def __len__(self):
        if self.database is None:
            self._build_db()
        return len(self.keys)

    def get_ori_data(self, idx):
        if self.database is None:
            self._build_db()
        key = self.keys[idx]
        data = pickle.loads(self.database.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data


class DualDataset(Dataset):

    def __init__(self, 
                 raw_path: str, 
                 transform: Optional[callable] = None,
                 version: str = 'final'
                 ):
        """
        Args:
            raw_path: Path to the raw data directory
            transform: Data transformation function applied to a single ProteinLigandData
            version: Dataset version identifier (used for the processed file name)
        """
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'h_dual_pairs_final.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.db = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        # with self.db.begin() as txn:
        #     self.keys = list(txn.cursor().iternext(values=False))
        self.txn = self.db.begin(buffers=True)
        self.keys = list(self.txn.cursor().iternext(values=False))

    def _close_db(self):
        if self.db is not None:
            self.txn = None
            self.db.close()
            self.db = None
            self.keys = None
        
    def _process(self):
        """
            Process raw data and store paired samples into LMDB
        """
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:

            # BindingNetV2_dualdata_high
            for i, pair in enumerate(tqdm(index)):
                pair1, pair2 = pair

                p1_path = os.path.join(self.raw_path, pair1)
                p2_path = os.path.join(self.raw_path, pair2)

                ligand_fn1 = os.path.join(p1_path, 'ligand.sdf')
                ligand_fn2 = os.path.join(p2_path, 'ligand.sdf')
                pocket_fn1 = os.path.join(p1_path, 'pocket_10A_clean.pdb')
                pocket_fn2 = os.path.join(p2_path, 'pocket_10A_clean.pdb')

                pocket_dict_1 = PDBProtein(pocket_fn1).to_dict_atom()
                pocket_dict_2 = PDBProtein(pocket_fn2).to_dict_atom()
                ligand_dict_1 = parse_lig_file(ligand_fn1)
                ligand_dict_2 = parse_lig_file(ligand_fn2)
 
                try:
                    data = ProteinLigandData.from_protein_ligand_dicts_dualdata(
                        protein_dict1=to_torch_dict(pocket_dict_1),
                        protein_dict2=to_torch_dict(pocket_dict_2),
                        ligand_dict1=to_torch_dict(ligand_dict_1),
                        ligand_dict2=to_torch_dict(ligand_dict_2),
                    )
                except AssertionError as ae:
                    error_msg = str(ae)

                    if any(keyword in error_msg for keyword in ['ligand_element', 'ligand_bond_index', 'ligand_bond_type', 'ligand_atom_feature']):
                        print(f'i: {i}, pair: {pair}, {error_msg}')
                    else:
                        raise ae

                    num_skipped += 1
                    continue

                data.BindingNetv2_id_1 = pair1
                data.BindingNetv2_id_2 = pair2

                mol = Chem.MolFromSmiles(data.ligand_smiles)
                chem_results = scoring_func.get_chem(mol)
                data.logp = chem_results['logp']
                data.tpsa = chem_results['tpsa']
                data.sa = chem_results['sa']
                data.qed = chem_results['qed']

                data = data.to_dict()
                txn.put(
                    key=str(i).encode(),
                    value=pickle.dumps(data)
                )
                # if i > 10:
                #     break
        print(f'num_skipped: {num_skipped}')

        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        
        buf = self.txn.get(key)
        data = pickle.loads(bytes(buf))
        
        
        # data = pickle.loads(self.db.begin().get(key))
        data = DualPLData(**data)
        data.id = idx
        return data
    
    def __del__(self):
        self._close_db()


def get_dataset(config, *args, **kwargs):
    if config.name == "protein_ligand":
        dataset = ProteinLigandDataset(config.path, *args, **kwargs)
    elif config.name == 'dual':
        dataset = DualDataset(config.path, *args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset name: {config.name}")
    
    if "split" in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset, None
