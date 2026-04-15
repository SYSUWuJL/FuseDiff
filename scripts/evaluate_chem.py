import argparse
import os
import torch
from rdkit import Chem, DataStructs
import numpy as np
from utils import misc
from utils.evaluation import scoring_func
import re


def clean_mols(ligand_rdmol_list):
    clean_rdmol_list = []

    for rdmol in ligand_rdmol_list:
        if rdmol is None:
            clean_rdmol_list.append(None)
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
            clean_rdmol_list.append(None)
            continue
        
        clean_rdmol_list.append(rdmol)

    return clean_rdmol_list


def eval_diversity(ligand_rdmol_list):
    diversity = []
    for i in range(0, len(ligand_rdmol_list), 10):
        mols = ligand_rdmol_list[i:i+10]
        mols = [m for m in mols if m is not None]
        n = len(mols)
        if n < 2:
            # diversity.append(0)
            continue
        fps = [Chem.RDKFingerprint(m) for m in mols]
        s = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dist = 1.0 - sim
                s += dist
                cnt += 1
        diversity.append(s / cnt)

    logger.info('diversity:  Mean: %.2f Median: %.2f' % (np.mean(diversity), np.median(diversity)))
    return diversity


def eval_chem(ligand_rdmol_list):
    results = []
    for rdmol in ligand_rdmol_list:
        if rdmol is None:
            results.append(None)
            continue
        chem_scores = scoring_func.get_chem(rdmol)
        results.append(chem_scores)
    
    qed = [r['qed'] for r in results if r is not None]
    sa = [r['sa'] for r in results if r is not None]
    logp = [r['logp'] for r in results if r is not None]
    lipinski = [r['lipinski'] for r in results if r is not None]
    tpsa = [r['tpsa'] for r in results if r is not None]

    logger.info('QED:        Mean: %.2f Median: %.2f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:         Mean: %.2f Median: %.2f' % (np.mean(sa), np.median(sa)))
    logger.info('logP:       Mean: %.2f Median: %.2f' % (np.mean(logp), np.median(logp)))
    logger.info('Lipinski:   Mean: %.2f Median: %.2f' % (np.mean(lipinski), np.median(lipinski)))
    logger.info('TPSA:       Mean: %.2f Median: %.2f' % (np.mean(tpsa), np.median(tpsa)))

    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_path', type=str, required=True)
    
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')
    logger.info(args)

    # collect all the mols
    ligand_rdmol_list = []
    pocket_path_list = []


    for dir1 in os.listdir(args.sample_path):
        dir1_path = os.path.join(args.sample_path, dir1)
        if not os.path.isdir(dir1_path):
            continue

        for dir2 in os.listdir(dir1_path):
            d1 = int(dir1)
            d2 = int(dir2)
            
            sample_path = os.path.join(dir1_path, dir2)
            if not os.path.isdir(sample_path):
                continue

            pt_path = os.path.join(sample_path, 'sample.pt')
            sample_result = torch.load(pt_path)
            ligand_rdmol_list.extend(sample_result['mols_1'])


    logger.info('Number of ligands: %d', len(ligand_rdmol_list))
    clean_rdmol_list = clean_mols(ligand_rdmol_list)

    eval_diversity(clean_rdmol_list)

    chem_results = eval_chem(clean_rdmol_list)