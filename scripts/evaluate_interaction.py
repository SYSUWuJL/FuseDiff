import argparse
import os

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy
from rdkit import Chem
from tqdm import tqdm

from utils import misc
from utils.evaluation.docking_vina import VinaDockingTask
from glob import glob
import pickle
import re
from utils.data import parse_lig_file


def eval_single_datapoint(ligand_rdmol_list, protein_path, args, center):

    results = []
    
    n_eval_success = 0

    for rdmol in tqdm(ligand_rdmol_list):
        if rdmol is None:
            results.append({
                'mol': None,
                'smiles': None,
                'protein_path': protein_path,
            })
            continue

        try:
            Chem.SanitizeMol(rdmol)
        except Chem.rdchem.AtomValenceException as e:
            err = e
            N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
            index = N4_valence.findall(err.args[0])
            if len(index) > 0:
                rdmol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
                Chem.SanitizeMol(rdmol)
        
        smiles = Chem.MolToSmiles(rdmol)
        
        if '.' in smiles:
            results.append({
                'mol': None,
                'smiles': smiles,
                'protein_path': protein_path,
            })
            continue
        
        mol = rdmol
        
        vina_task = VinaDockingTask(
            protein_path=protein_path,
            ligand_rdmol=deepcopy(mol),
            size_factor=None,
            center=center.tolist(),
        )
        vina_results = {}
        
        try:

            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
            
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
            
            vina_results.update({
               'score_only': score_only_results,
               'minimize': minimize_results
            })
            
            if args.docking_mode == 'vina_full':
                dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                vina_results.update({
                    'dock': dock_results,
                })
            elif args.docking_mode == 'vina_score':
                pass
            else:
                raise NotImplementedError
        except RuntimeError as e:
            if 'The ligand is outside the grid box' in str(e):
                print(f'{protein_path}: {e}')
                results.append({
                    'mol': None,
                    'smiles': smiles,
                    'protein_path': protein_path,
                })
                continue
            else:
                raise e
        
        n_eval_success += 1
            
        results.append({
            'mol': mol,
            'smiles': smiles,
            'protein_path': protein_path,
            'vina': vina_results,
        })
    # logger.info(f'Evaluate No {id} done! {len(ligand_rdmol_list)} samples in total. {n_eval_success} eval success!')
    return results


if __name__ == '__main__':
    # exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--synergy_idx', type=int, required=True, help="Index of the synergy to evaluate")
    parser.add_argument('--sample_path', type=str, required=True, default='logs_sampling/dualdiff/')
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')
    logger.info(args)
    
    idx = args.synergy_idx
    
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open('data_dualdiff/data/processed/drug_synergy/synergy_idx_list.pkl', 'rb') as f:
        synergy_idx_list = pickle.load(f)
    with open('data_dualdiff/data/processed/dock/index_dict.pkl', 'rb') as f:
        index_dict = pickle.load(f)
    idx_to_smiles = index_dict['idx_to_smiles']
    
    idx1, idx2 = synergy_idx_list[idx]
    smiles1 = idx_to_smiles[idx1]
    smiles2 = idx_to_smiles[idx2]

    for pocket_id in [1, 2]:

        logger.info(f"evaluating P{pocket_id} of synergy id: {args.synergy_idx}")


        smilesx = smiles1 if pocket_id == 1 else smiles2
        
        sample_path = os.path.join(args.sample_path, f'{idx1}', f'{idx2}')

        sample_result = torch.load(os.path.join(sample_path, 'sample.pt'))
        
        if pocket_id == 1:
            ligand_rdmol_list = sample_result['mols_1']
        elif pocket_id == 2:
            ligand_rdmol_list = sample_result['mols_2']

        protein_path = glob(f"data_dualdiff/data/processed/dock/ligand_protein_dataset_v2/{smilesx}/*/protein_clean.pdb")[0]
        anchor_ligand_dict = parse_lig_file(glob(f"data_dualdiff/data/processed/dock/ligand_protein_dataset_v2/{smilesx}/*/ligand.sdf")[0])

        testset_results = eval_single_datapoint(ligand_rdmol_list, protein_path, args, center=anchor_ligand_dict['center_of_mass'])


        logger.info(f'Evaluate No {idx} done! {len(ligand_rdmol_list)} samples in total. {len([x for x in testset_results if x["mol"] is not None])} eval success!')
        if args.docking_mode in ['vina', 'qvina']:
            vina = [x['vina'][0]['affinity'] for x in testset_results if x['mol'] is not None]
            logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
        elif args.docking_mode in ['vina_full', 'vina_score']:
            vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in testset_results if x['mol'] is not None]
            vina_min = [x['vina']['minimize'][0]['affinity'] for x in testset_results if x['mol'] is not None]
            logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
            logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
            if args.docking_mode == 'vina_full':
                vina_dock = [x['vina']['dock'][0]['affinity'] for x in testset_results if x['mol'] is not None]
                logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

        logger.info(f'Evaluation for index {idx} completed successfully.')
