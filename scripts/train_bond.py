import sys
import os
import shutil
import argparse
import numpy as np
sys.path.append('.')

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from models.bond_predictor import BondPredictor
from utils.dataset import get_dataset
import utils.transforms as transforms
from utils.misc import *
from utils.train_utils import *


def get_auroc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum_auroc = 0
    classes = set(y_true)
    for c in classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        sum_auroc += auroc * np.sum(y_true == c)
    avg_auroc = sum_auroc / len(y_true)
    return avg_auroc

def train(args, config, model, train_iterator, optimizer, scaler, logger, writer, it):
    optimizer.zero_grad(set_to_none=True)
    batch = next(train_iterator).to(args.device)

    protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
    pos_noise = torch.randn_like(batch.ligand_pos) * config.train.pos_noise_std
    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=config.train.use_amp):
        loss_dict, _ = model.get_loss(
            protein_node = batch.protein_atom_feat.float(), 
            protein_pos = batch.protein_pos + protein_noise, 
            protein_batch = batch.protein_element_batch,
            ligand_node = batch.ligand_atom_feat_full,
            ligand_pos = batch.ligand_pos + pos_noise,
            ligand_batch = batch.ligand_element_batch,
            halfedge_type = batch.ligand_halfedge_type,
            halfedge_index = batch.ligand_halfedge_index,
            halfedge_batch = batch.ligand_halfedge_type_batch,
            num_mol = batch.num_graphs,
        )
    loss = loss_dict['loss']
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    if it % config.train.train_report_iter == 0:
        log_info = '[Train] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v.item()) for k, v in loss_dict.items()
        ])
        logger.info(log_info)
        for k, v in loss_dict.items():
            writer.add_scalar('train/%s' % k, v.item(), it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

def validate(args, config, model, val_loader, scheduler, logger, writer, it):
    sum_n =  0   # num of loss
    sum_loss_dict = {} 
    all_pred_half_e, all_true_half_e = [], []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validate'):
            batch = batch.to(args.device)
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=config.train.use_amp):
                loss_dict, pred_dict = model.get_loss(
                    # compose
                    protein_node = batch.protein_atom_feat.float(), 
                    protein_pos = batch.protein_pos, 
                    protein_batch = batch.protein_element_batch,
                    ligand_node = batch.ligand_atom_feat_full,
                    ligand_pos = batch.ligand_pos,
                    ligand_batch = batch.ligand_element_batch,
                    halfedge_type = batch.ligand_halfedge_type,
                    halfedge_index = batch.ligand_halfedge_index,
                    halfedge_batch = batch.ligand_halfedge_type_batch,
                    num_mol = batch.num_graphs,
                )
            if len(sum_loss_dict) == 0:
                sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
            else:
                for key in sum_loss_dict.keys():
                    sum_loss_dict[key] += loss_dict[key].item()
            sum_n += 1
            all_pred_half_e.append(pred_dict["pred_ligand_halfedge"].cpu().numpy())
            all_true_half_e.append(batch.ligand_halfedge_type.cpu().numpy())
    
    # finish all batches
    avg_loss_dict = {k: v / sum_n for k, v in sum_loss_dict.items()}
    avg_loss = avg_loss_dict['loss']
    # update lr scheduler
    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    elif config.train.scheduler.type == 'warmup_plateau':
        scheduler.step_ReduceLROnPlateau(avg_loss)
    else:
        scheduler.step()

    bond_auroc = get_auroc(np.concatenate(all_true_half_e), np.concatenate(all_pred_half_e, axis=0))
    avg_loss_dict['bond_auroc'] = bond_auroc

    log_info = '[Validate] Iter %d | ' % it + ' | '.join([
        '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items()
    ])
    logger.info(log_info)
    for k, v in avg_loss_dict.items():
        writer.add_scalar('val/%s' % k, v, it)
    writer.flush()
    return avg_loss_dict

def main(args):
    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    # log_dir = args.logdir
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    success_ckpt_dir = os.path.join(ckpt_dir, 'success')
    os.makedirs(success_ckpt_dir, exist_ok=True)
    fail_ckpt_dir = os.path.join(ckpt_dir, 'fail')
    os.makedirs(fail_ckpt_dir, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Transforms
    featurizer = transforms.FeatureComplex(
        config.data.transform.ligand_atom_mode, 
        sample=config.data.transform.sample
    )
    transform = Compose([
        featurizer,
    ])

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.data,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['val']
    logger.info(f'Train dataset: {len(train_set)}, Validation dataset: {len(val_set)}.')
    train_loader = DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
    )
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(
        val_set, 
        config.train.batch_size, 
        shuffle=False,
        follow_batch=featurizer.follow_batch, 
        exclude_keys=featurizer.exclude_keys
    )

    # Model
    logger.info('Building model...')
    model = BondPredictor(
        config=config.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim
    ).to(args.device)
    num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'Num of trainable parameters: {num_parameters / 1e6:.4f} M.')
    logger.info(f'protein feature dim: {featurizer.protein_feat_dim}, ligand atom feature dim: {featurizer.atom_feat_dim}, ligand bond feature dim: {featurizer.bond_feat_dim}.')

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)

    # Resume or not
    resume_step = 0
    if config.train.resume:
        resume_step = config.train.resume_ckpt.split('.')[0]
        logger.info(
            f"Loading model from checkpoint: {config.train.resume_ckpt}... at {resume_step} step."
        )
        checkpoint = torch.load(os.path.join(fail_ckpt_dir, config.train.resume_ckpt), map_location=args.device)
        #checkpoint = torch.load(os.path.join(success_ckpt_dir, config.train.resume_ckpt), map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.to(args.device)

    # Load previous model parameters or not
    if config.train.prev_model:
        logger.info(
            f"Loading model from checkpoint: {config.train.prev_ckpt}."
        )
        checkpoint = torch.load(config.train.prev_ckpt, map_location=args.device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.to(args.device)

    try:
        model.train()
        early_stop = 0
        best_loss, best_bond_auroc, best_it = 1e9, 0, 0
        # best_loss, best_bond_auroc, best_it = 0.438841, 0.941529, 4650000
        for it in range(int(resume_step)+1, config.train.max_iters+1):
            try:
                train(args, config, model, train_iterator, optimizer, scaler, logger, writer, it)
            except RuntimeError as e:
                logger.error('Runtime Error ' + str(e))
                logger.error('Skipping Iteration %d' % it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                loss_dict = validate(args, config, model, val_loader, scheduler, logger, writer, it)
                if loss_dict['loss'] < best_loss:
                    best_loss, best_bond_auroc, best_it = loss_dict['loss'], loss_dict['bond_auroc'], it
                    ckpt_path = os.path.join(success_ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    early_stop = 0
                    model.train()
                else:
                    ckpt_path = os.path.join(fail_ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    early_stop += 1
                    logger.info(
                                f"Val loss is not improved at Iter {it}. Best val loss: {best_loss:.6f} and Best bond auroc: {best_bond_auroc:.6f} are achieved at Iter {best_it}."
                            )
                    logger.info('Early stop is {} for Iter {}'.format(early_stop, it))

                # if early_stop >= 20:
                #     logger.info('Early Stopping!!! No Improvement on Loss for 20 validation Iters.')
                #     break

    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    # Usage: python scripts/train_bond.py --config ./configs/train/train_bond.yml --device cuda:0 --logdir ./bond_logs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/train_bondpred.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./bond_logs')
    args = parser.parse_args()

    main(args)    
