import os
import sys
import shutil
import argparse
import math
import pickle
sys.path.append('.')

import torch
import numpy as np
from scipy import spatial
import torch.utils.tensorboard
from rdkit import Chem

from torch_geometric.data import Batch
from models.model import FuseDiff
from models.bond_predictor import BondPredictor
from utils.sample_utils import seperate_outputs
from torch_geometric.transforms import Compose
from utils.atom_num_config import CONFIG
from utils.data import PDBProtein
from utils.dataset import to_torch_dict, get_dataset, DualPLData
from utils.transforms import FeatureComplex, make_data_placeholder
from utils.misc import *
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])

def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)

def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    atom_num = np.random.choice(num_atom_list, p=prob_list)
    return atom_num


def pdb_to_pocket(pocket_pdb_path_1, pocket_pdb_path_2):
    pocket_dict_1 = PDBProtein(pocket_pdb_path_1).to_dict_atom()
    pocket_dict_2 = PDBProtein(pocket_pdb_path_2).to_dict_atom()
    ligand_dict={
        "element": torch.empty([0, ], dtype=torch.long),
        "hybridization": torch.empty([0, ], dtype=torch.long),
        "pos": torch.empty([0, 3], dtype=torch.float),
        "bond_index": torch.empty([2, 0], dtype=torch.long),
        "bond_type": torch.empty([0, ], dtype=torch.long),
        "atom_feature": torch.empty([0, 8], dtype=torch.float),
    }
    data = DualPLData.from_protein_ligand_dicts_dualdata(
        protein_dict1=to_torch_dict(pocket_dict_1),
        protein_dict2=to_torch_dict(pocket_dict_2),
        ligand_dict1=to_torch_dict(ligand_dict),
        ligand_dict2=to_torch_dict(ligand_dict),
    )

    return data

def main(args):
    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    #log_dir = args.logdir:5
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, 'sample.yml'))

    # # Transform
    logger.info('Loading data placeholder...')
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    if config.model.gen_mode == 'denovo':
        featurizer = FeatureComplex(
            config.data.transform.ligand_atom_mode, 
            sample=config.data.transform.sample
        )
    # else:
    #     featurizer = FeatureComplexWithFrag(ligand_atom_mode, sample=config.sample.sample)
    transform = Compose([
        featurizer,
    ])
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # # Model
    logger.info('Loading diffusion model...')
    # train_config.model.class_dim = 4
    # ckpt['model']['class_emb.0.weight'] = ckpt['model']['class_emb.0.weight'][:, :4]

    if train_config.model.name == 'FuseDiff':
        model = FuseDiff(
            config=train_config.model,
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % train_config.model.name)
    
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    
    # label
    logp = torch.tensor([float(config.model.logp)], device=args.device).unsqueeze(-1)
    tpsa = torch.tensor([float(config.model.tpsa)], device=args.device).unsqueeze(-1)
    sa = torch.tensor([float(config.model.sa)], device=args.device).unsqueeze(-1)
    qed = torch.tensor([float(config.model.qed)], device=args.device).unsqueeze(-1)
    # aff = torch.tensor([float(config.model.aff)], device=args.device).unsqueeze(-1)
    batch_lab_single = torch.cat((logp, tpsa, sa, qed), dim=1)

    # batch_size = config.sample.batch_size
    # batch_lab = torch.tensor([list(batch_lab_single[0]) for _ in range(batch_size)]).to(args.device)

    # # Bond predictor and guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(
            config=ckpt_bond['config']['model'],
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'], strict=True)
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None

    # Load pocket or test set data
    if config.sample.mode == 'given_pockets':
        logger.info('Loading given pockets...')
        pocket_1 = config.model.target_1
        pocket_2 = config.model.target_2
        logger.info(f"Given pockets: pocket_1 = {pocket_1}, pocket_2 = {pocket_2}")
        data = pdb_to_pocket(pocket_1, pocket_2)
        data = transform(data)
        data_list = [data]
    elif config.sample.mode == 'BindingNetv2_validset':
        logger.info('Loading BindingNetv2-dual valid set...')
        dataset, subsets = get_dataset(
            config = config.data,
            transform = transform,
        )
        data_list = subsets['test']
    elif config.sample.mode == 'dualdiff_testset':
        logger.info('Loading dualdiff test set...')
        with open('data/dualdiff_testset/synergy_idx_list.pkl', 'rb') as f:
            synergy_idx_list = pickle.load(f)
        data_list = []
        for idx1, idx2 in synergy_idx_list:
            p1_path = f'data/dualdiff_testset/{idx1}/{idx2}/pocket_10A_1.pdb'
            p2_path = f'data/dualdiff_testset/{idx1}/{idx2}/pocket_10A_2.pdb'
            data = pdb_to_pocket(p1_path, p2_path)
            data = transform(data)
            data_list.append(data)
    else:
        raise NotImplementedError('Sample mode invalid!')

    data_length = len(data_list)
    logger.info(f"Number of samples to generate: {data_length}")

    for i in tqdm(range(data_length), desc='Sample'):
        
        logger.info(f'{config.sample.mode}: sampling test set NO.{i}...')
        
        data = data_list[i]

        num_samples = config.sample.num_mols
        batch_size = config.sample.batch_size

        if config.sample.mode == 'given_pockets':
            result_path = os.path.join(log_dir, 'given_pockets_output')
        elif config.sample.mode == 'BindingNetv2_validset':
            result_path = os.path.join(log_dir, f'BindingNetv2_dual_validset_{i}')
        elif config.sample.mode == 'dualdiff_testset':
            result_path = os.path.join(log_dir, str(synergy_idx_list[i][0]), str(synergy_idx_list[i][1]))
        else:
            raise NotImplementedError('Sample mode invalid!')

        if not os.path.exists(os.path.join(result_path, 'sample.pt')):
            try:
                # generating molecules
                mol_list_1 = []
                mol_list_2 = []

                num_batch = int(np.ceil(num_samples / batch_size))
                n_recon_success, n_complete = 0, 0

                for j in tqdm(range(num_batch)):

                    n_graphs = batch_size if j < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
                    batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch=featurizer.follow_batch).to(args.device)

                    if config.sample.sample_method == "priori":
                        pocket_size_1 = get_pocket_size(batch.protein1_pos.detach().cpu().numpy())
                        pocket_size_2 = get_pocket_size(batch.protein2_pos.detach().cpu().numpy())

                        ligand_num_atoms_1 = [sample_atom_num(pocket_size_1).astype(int) for _ in range(n_graphs)]
                        ligand_num_atoms_2 = [sample_atom_num(pocket_size_2).astype(int) for _ in range(n_graphs)]

                        ligand_num_atoms = [math.ceil((num_1 + num_2) / 2) for num_1, num_2 in zip(ligand_num_atoms_1, ligand_num_atoms_2)]
                    else:
                        raise ValueError

                    logger.info(f'ligand_num_atoms: {ligand_num_atoms}')

                    batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=args.device)
                    batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
                    
                    # inference
                    if config.model.gen_mode == 'denovo':
                        outputs = model.sample(
                            n_graphs=n_graphs,

                            protein_node_1=batch.protein1_atom_feat.float(), 
                            protein_pos_1=batch.protein1_pos, 
                            protein_batch_1=batch.protein1_element_batch,

                            protein_node_2=batch.protein2_atom_feat.float(), 
                            protein_pos_2=batch.protein2_pos, 
                            protein_batch_2=batch.protein2_element_batch,

                            ligand_batch=batch_node,
                            halfedge_index=halfedge_index,
                            halfedge_batch=batch_halfedge,
                            batch_lab=torch.tensor([list(batch_lab_single[0]) for _ in range(n_graphs)]).to(args.device),
                            gui_strength=config.sample.gui_strength,
                            bond_predictor=bond_predictor,
                            guidance=guidance,
                        )
                    else:
                        raise ValueError

                    outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
                    
                    # decode outputs to molecules
                    batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
                    output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)

                    for i_mol, output_mol in enumerate(output_list):
                        mol_info_1 = featurizer.decode_output(
                            pred_node=output_mol['pred'][0],
                            pred_pos=output_mol['pred'][1],
                            pred_halfedge=output_mol['pred'][3],
                            halfedge_index=output_mol['halfedge_index'],
                        )  # note: traj is not used

                        mol_info_2 = featurizer.decode_output(
                            pred_node=output_mol['pred'][0],
                            pred_pos=output_mol['pred'][2],
                            pred_halfedge=output_mol['pred'][3],
                            halfedge_index=output_mol['halfedge_index'],
                        )  # note: traj is not used

                        try:
                            rdmol_1 = reconstruct_from_generated_with_edges(mol_info_1, add_edge=add_edge)
                            rdmol_2 = reconstruct_from_generated_with_edges(mol_info_2, add_edge=add_edge)
                        except MolReconsError:
                            logger.warning('Reconstruction error encountered.')
                            mol_list_1.append(None)
                            mol_list_2.append(None)
                            continue
                        n_recon_success += 1

                        mol_info_1['rdmol'] = rdmol_1
                        mol_info_2['rdmol'] = rdmol_2
                        smiles_1 = Chem.MolToSmiles(rdmol_1)
                        smiles_2 = Chem.MolToSmiles(rdmol_2)
                        mol_info_1['smiles'] = smiles_1
                        mol_info_2['smiles'] = smiles_2
                        # contain_B_1 = re.search(r'B(?![rR]\b)', smiles_1)
                        # contain_B_2 = re.search(r'B(?![rR]\b)', smiles_1)

                        if '.' in smiles_1 or '.' in smiles_2:
                            logger.warning('Incomplete molecule: %s' % smiles_1)
                            mol_list_1.append(None)
                            mol_list_2.append(None)
                            continue
                        # elif contain_B_1 or contain_B_2:
                        #     logger.warning('Element Boron in molecule: %s' % smiles_1)
                        #     continue
                        else:    # Pass checks!
                            logger.info('Success: %s' % smiles_1)
                            n_complete += 1
                            mol_list_1.append(rdmol_1)
                            mol_list_2.append(rdmol_2)

                logger.info('Reconstruction done!')
                logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

                results = {
                    'mols_1': mol_list_1,
                    'mols_2': mol_list_2,
                    'mol_info_1': mol_info_1,
                    'mol_info_2': mol_info_2,
                }
                os.makedirs(result_path, exist_ok=True)
                torch.save(results, os.path.join(result_path, f'sample.pt'))
                logger.info(f'{config.sample.mode}: test set NO.{i} results are saved in {result_path}')

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error(
                        f"ERROR: {config.sample.mode} test set NO.{i} ran out of memory "
                        f"for batch_size {batch_size}, skipping this sample."
                    )
                    continue
                else:
                    logger.error(f'Unseen ERROR: {e}!!!')
                    raise e
        else:
            logger.info(f'{config.sample.mode}: test set NO.{i} already sampled, skipping...')


if __name__ == '__main__':
    # Usage: python scripts/sample.py --config configs/sample.yml --outdir logs_sample/
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_dual.yml')
    parser.add_argument('--outdir', type=str, required=True, default='logs_sample/')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    main(args)
    