import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from utils.dataset import ProteinLigandData
import utils.data as utils_data


aromatic_idx = utils_data.atom_families_id["Aromatic"]

# H, B, C, N, O, F, P, S, Cl, Br, I
map_atom_type_only_to_index = {
    1: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    15: 6,
    16: 7,
    17: 8,
    35: 9,
    53: 10,
}

map_atom_type_aromatic_to_index = {
    (1, False): 0,
    (5, False): 1,
    (6, False): 2,
    (6, True): 3,
    (7, False): 4,
    (7, True): 5,
    (8, False): 6,
    (8, True): 7,
    (9, False): 8,
    (15, False): 9,
    (15, True): 10,
    (16, False): 11,
    (16, True): 12,
    (17, False): 13,
    (35, False): 14,
    (53, False): 15,
}

map_atom_type_full_to_index = {
    (1, 'S', False): 0,
    (5, 'SP2', False): 1,
    (6, 'SP', False): 2,
    (6, 'SP2', False): 3,
    (6, 'SP2', True): 4,
    (6, 'SP3', False): 5,
    (7, 'SP', False): 6,
    (7, 'SP2', False): 7,
    (7, 'SP2', True): 8,
    (7, 'SP3', False): 9,
    (8, 'SP2', False): 10,
    (8, 'SP2', True): 11,
    (8, 'SP3', False): 12,
    (9, 'SP3', False): 13,
    (15, 'SP2', False): 14,
    (15, 'SP2', True): 15,
    (15, 'SP3', False): 16,
    (15, 'SP3D', False): 17,
    (16, 'SP2', False): 18,
    (16, 'SP2', True): 19,
    (16, 'SP3', False): 20,
    (16, 'SP3D', False): 21,
    (16, 'SP3D2', False): 22,
    (17, 'SP3', False): 23,
    (35, 'SP3', False): 24,
    (53, 'SP3', False): 25,
}

map_index_to_atom_type_only = {v: k for k, v in map_atom_type_only_to_index.items()}
map_index_to_atom_type_aromatic = {v: k for k, v in map_atom_type_aromatic_to_index.items()}
map_index_to_atom_type_full = {v: k for k, v in map_atom_type_full_to_index.items()}

def get_atomic_number_from_index(index, mode):
    if mode == "basic":
        atomic_number = [map_index_to_atom_type_only[i] for i in index.tolist()]
    elif mode == "aromatic":
        atomic_number = [map_index_to_atom_type_aromatic[i][0] for i in index.tolist()]
    elif mode == "full":
        atomic_number = [map_index_to_atom_type_full[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number

def is_aromatic_from_index(index, mode):
    if mode == 'basic':
        is_aromatic = None
    elif mode == 'aromatic':
        is_aromatic = [map_index_to_atom_type_aromatic[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [map_index_to_atom_type_full[i][2] for i in index.tolist()]
    else:
        raise ValueError
    return is_aromatic

def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == "basic":
        index = map_atom_type_only_to_index[int(atom_num)]
    elif mode == "aromatic":
        if (int(atom_num), bool(is_aromatic)) in map_atom_type_aromatic_to_index:
            index = map_atom_type_aromatic_to_index[(int(atom_num), bool(is_aromatic))]
        else:
            print(int(atom_num), bool(is_aromatic))
            index = map_atom_type_aromatic_to_index[1, False]
    elif mode == "full":
        index = map_atom_type_full_to_index[(int(atom_num), str(hybridization), bool(is_aromatic))]
    else:
        raise ValueError
    return index


class FeatureComplex(object):
    def __init__(self, mode="basic", sample=False):
        super().__init__()
        # H, C, N, O, Mg, P, S, Ca, Mn, Fe, Co, Ni, Cu, Zn, Se, Mo
        # self.protein_atomic_numbers = torch.LongTensor([1, 6, 7, 8, 12, 15, 16, 20, 25, 26, 27, 28, 29, 30, 34, 42])
        # H, C, N, O, S, Se
        self.protein_atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])
        self.max_num_aa = 21
        assert mode in ["basic", "aromatic", "full"], "Mode has to be one of basic, aromatic or full!"
        self.mode = mode
        self.sample = sample
        self.ele_to_nodetype = {k[0]: v for k, v in map_atom_type_aromatic_to_index.items()}
        self.nodetype_to_ele = {v: k[0] for k, v in map_atom_type_aromatic_to_index.items()}
        self.follow_batch = ["protein1_element", "protein2_element", "ligand_element", "ligand_bond_type", "ligand_halfedge_type"]
        self.exclude_keys = ["ligand_nbh_list", "num_bonds", "num_atoms"]

    @property
    def protein_feat_dim(self):
        protein_feat_dim = self.protein_atomic_numbers.size(0) + self.max_num_aa + 1
        return protein_feat_dim

    @property
    def atom_feat_dim(self):
        if self.mode == "basic":
            ligand_atom_feat_dim = len(map_atom_type_only_to_index)
        elif self.mode == "aromatic":
            ligand_atom_feat_dim = len(map_atom_type_aromatic_to_index)
        elif self.mode == "full":
            ligand_atom_feat_dim = len(map_atom_type_full_to_index)
        return ligand_atom_feat_dim

    @property
    def bond_feat_dim(self):
        bond_feat_dim = len(utils_data.bond_types)
        return bond_feat_dim

    def __call__(self, data: ProteinLigandData):
        data.protein1_num_atoms = len(data.protein1_element)
        data.protein2_num_atoms = len(data.protein2_element)
        element_1 = data.protein1_element.view(-1, 1) == self.protein_atomic_numbers.view(1, -1)
        element_2 = data.protein2_element.view(-1, 1) == self.protein_atomic_numbers.view(1, -1)
        amino_acid_1 = F.one_hot(data.protein1_atom_to_aa_type, num_classes=self.max_num_aa)
        amino_acid_2 = F.one_hot(data.protein2_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone_1 = data.protein1_is_backbone.view(-1, 1).long()
        is_backbone_2 = data.protein2_is_backbone.view(-1, 1).long()
        x_1 = torch.cat([element_1, amino_acid_1, is_backbone_1], dim=-1)
        x_2 = torch.cat([element_2, amino_acid_2, is_backbone_2], dim=-1)
        data.protein1_atom_feat = x_1
        data.protein2_atom_feat = x_2

        element_list = data.ligand_element
        hybrid_list = data.ligand_hybridization
        aromatic_list = [v[aromatic_idx] for v in data.ligand_atom_feature]

        y = torch.tensor(
            [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybrid_list, aromatic_list)]
        )
        data.ligand_atom_feat_full = y
        data.ligand_bond_feat = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.bond_types))

        # build half edge
        if not self.sample:
            data.ligand_num_atoms = len(data.ligand_element)
            edge_type_mat = torch.zeros([data.ligand_num_atoms, data.ligand_num_atoms], dtype=torch.long)

            assert len(data.ligand_bond_type) % 2 == 0
            data.ligand_num_bonds = int(len(data.ligand_bond_type) / 2)
            for i in range(data.ligand_num_bonds * 2):
                edge_type_mat[data.ligand_bond_index[0, i], data.ligand_bond_index[1, i]] = data.ligand_bond_type[i]
            halfedge_index = torch.triu_indices(data.ligand_num_atoms, data.ligand_num_atoms, offset=1)
            halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
            data.ligand_halfedge_index = halfedge_index
            data.ligand_halfedge_type = halfedge_type
            assert (data.ligand_halfedge_type > 0).sum() == data.ligand_num_bonds
            data.num_nodes = data.ligand_num_atoms

        return data

    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < self.atom_feat_dim)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos[isnot_masked_atom]
        
        # get bond
        if self.bond_feat_dim == 1:
            return {
                'element': element,
                'atom_pos': atom_pos,
                'atom_prob': atom_prob,
            }
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        
        is_bond = (edge_type > 0) & (edge_type <= self.bond_feat_dim)  # larger is mask type
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            bond_prob = bond_prob[~bond_for_masked_atom]

        bond_type = np.concatenate([bond_type, bond_type])
        bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            'element': element,
            'atom_pos': atom_pos,
            'bond_type': bond_type,
            'bond_index': bond_index,
            
            'atom_prob': atom_prob,
            'bond_prob': bond_prob,
        }


class RandomRotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data

def make_data_placeholder(n_nodes_list=None, device=None):
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }
