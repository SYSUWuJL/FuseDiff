import sys
import os
import shutil
import argparse
import numpy as np
import gc
sys.path.append('.')

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.utils.tensorboard
from models.model import FuseDiff
import utils.transforms as transforms
from utils.dataset import get_dataset
from utils.misc import *
from utils.train_utils import inf_iterator, get_optimizer, get_scheduler


def get_auroc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    sum_auroc = 0
    for c in classes:
        y_pred_class = np.nan_to_num(y_pred[:, c], nan=0, posinf=1e10, neginf=-1e10)
        auroc = roc_auc_score((y_true == c).astype(int), y_pred_class, multi_class='ovr', labels=[c])
        sum_auroc += auroc * np.sum(y_true == c)
    avg_auroc = sum_auroc / len(y_true)
    return avg_auroc

def train(args, config, model, train_iterator, optimizer, scaler, logger, writer, it):
    optimizer.zero_grad(set_to_none=True)
    batch = next(train_iterator).to(args.device)
    batch_size = batch.num_graphs

    batch_logp = torch.tensor([float(item) for item in batch.logp], device=args.device).unsqueeze(-1)
    batch_tpsa = torch.tensor([float(item) for item in batch.tpsa], device=args.device).unsqueeze(-1)
    batch_sa = torch.tensor([float(item) for item in batch.sa], device=args.device).unsqueeze(-1)
    batch_qed = torch.tensor([float(item) for item in batch.qed], device=args.device).unsqueeze(-1)
    # batch_aff = torch.tensor([float(item) for item in batch.aff], device=args.device).unsqueeze(-1)
    batch_lab = torch.cat((batch_logp, batch_tpsa, batch_sa, batch_qed), dim=1)
    batch_lab[np.where(np.random.rand(batch_size)<config.train.threshold)] = 0
    
    protein_noise_1 = torch.randn_like(batch.protein1_pos) * config.train.pos_noise_std
    protein_noise_2 = torch.randn_like(batch.protein2_pos) * config.train.pos_noise_std
    pos_noise_1 = torch.randn_like(batch.ligand1_pos) * config.train.pos_noise_std
    pos_noise_2 = torch.randn_like(batch.ligand2_pos) * config.train.pos_noise_std
    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=config.train.use_amp):
        loss_dict, _ = model.get_loss(
            protein_node_1 = batch.protein1_atom_feat.float(), 
            protein_node_2 = batch.protein2_atom_feat.float(), 
            protein_pos_1 = batch.protein1_pos + protein_noise_1, 
            protein_pos_2 = batch.protein2_pos + protein_noise_2, 
            protein_batch_1 = batch.protein1_element_batch,
            protein_batch_2 = batch.protein2_element_batch,

            ligand_node = batch.ligand_atom_feat_full,
            ligand_pos_1 = batch.ligand1_pos + pos_noise_1,
            ligand_pos_2 = batch.ligand2_pos + pos_noise_2,
            ligand_batch = batch.ligand_element_batch,

            halfedge_type = batch.ligand_halfedge_type,
            halfedge_index = batch.ligand_halfedge_index,
            halfedge_batch = batch.ligand_halfedge_type_batch,

            num_mol = batch.num_graphs,
            batch_lab = batch_lab
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
    all_pred_v, all_true_v = [], []
    all_pred_half_e, all_true_half_e = [], []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validate'):
            batch = batch.to(args.device)
            batch_size = batch.num_graphs

            batch_logp = torch.tensor([float(item) for item in batch.logp], device=args.device).unsqueeze(-1)
            batch_tpsa = torch.tensor([float(item) for item in batch.tpsa], device=args.device).unsqueeze(-1)
            batch_sa = torch.tensor([float(item) for item in batch.sa], device=args.device).unsqueeze(-1)
            batch_qed = torch.tensor([float(item) for item in batch.qed], device=args.device).unsqueeze(-1)
            # batch_aff = torch.tensor([float(item) for item in batch.aff], device=args.device).unsqueeze(-1)
            batch_lab = torch.cat((batch_logp, batch_tpsa, batch_sa, batch_qed), dim=1)
            batch_lab[np.where(np.random.rand(batch_size)<config.train.threshold)] = 0

            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=config.train.use_amp):
                loss_dict, pred_dict = model.get_loss(
                    protein_node_1 = batch.protein1_atom_feat.float(), 
                    protein_node_2 = batch.protein2_atom_feat.float(), 
                    protein_pos_1 = batch.protein1_pos, 
                    protein_pos_2 = batch.protein2_pos, 
                    protein_batch_1 = batch.protein1_element_batch,
                    protein_batch_2 = batch.protein2_element_batch,

                    ligand_node = batch.ligand_atom_feat_full,
                    ligand_pos_1 = batch.ligand1_pos,
                    ligand_pos_2 = batch.ligand2_pos,
                    ligand_batch = batch.ligand_element_batch,

                    halfedge_type = batch.ligand_halfedge_type,
                    halfedge_index = batch.ligand_halfedge_index,
                    halfedge_batch = batch.ligand_halfedge_type_batch,

                    num_mol = batch.num_graphs,
                    batch_lab = batch_lab, 
                )
            if len(sum_loss_dict) == 0:
                sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
            else:
                for key in sum_loss_dict.keys():
                    sum_loss_dict[key] += loss_dict[key].item()
            sum_n += 1
            all_pred_v.append(pred_dict["pred_ligand_node"].detach().cpu().to(torch.float32).numpy())
            all_true_v.append(batch.ligand_atom_feat_full.detach().cpu().numpy())

            all_pred_half_e.append(pred_dict["pred_ligand_halfedge"].detach().cpu().to(torch.float32).numpy())
            all_true_half_e.append(batch.ligand_halfedge_type.detach().cpu().numpy())
    
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

    true_v = np.concatenate(all_true_v, axis=0)
    pred_v = np.concatenate(all_pred_v, axis=0)
    true_e = np.concatenate(all_true_half_e, axis=0)
    pred_e = np.concatenate(all_pred_half_e, axis=0)

    atom_auroc = get_auroc(true_v, pred_v)
    bond_auroc = get_auroc(true_e, pred_e)
    avg_loss_dict['atom_auroc'] = atom_auroc
    avg_loss_dict['bond_auroc'] = bond_auroc
    
    del all_true_v, all_pred_v, all_true_half_e, all_pred_half_e
    del true_v, pred_v, true_e, pred_e
    gc.collect()
    torch.cuda.empty_cache()

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
    #log_dir = args.logdir
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
    transform_list = [featurizer]
    if config.data.transform.random_rot:
        transform_list.append(transforms.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.data,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Train dataset: {len(train_set)}, Validation dataset: {len(val_set)}.')
    print(train_set[0])

    train_loader = DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
        # prefetch_factor=1,
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
    if config.model.name == 'FuseDiff':
        model = FuseDiff(
            config=config.model,
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % config.model.name)
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
        checkpoint['model']['class_emb.0.weight'] = checkpoint['model']['class_emb.0.weight'][:, :4]
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(args.device)

    try:
        model.train()
        early_stop = 0
        best_loss, best_atom_auroc, best_bond_auroc, best_it = 1e9, 0, 0, 0
        # best_loss, best_atom_auroc, best_bond_auroc, best_it = 1.123276, 0.958042, 0.968242, 416000
        for it in range(int(resume_step)+1, config.train.max_iters+1):
            try:
                train(args, config, model, train_iterator, optimizer, scaler, logger, writer, it)
            except RuntimeError as re:
                torch.cuda.empty_cache()
                logger.error('Runtime Error ' + str(re))
                logger.error('Skipping Iteration %d' % it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                try:
                    loss_dict = validate(args, config, model, val_loader, scheduler, logger, writer, it)
                except RuntimeError as re:
                    torch.cuda.empty_cache()
                    logger.error('Runtime Error ' + str(re))
                    logger.error('Skipping Validation %d' % it)
                if loss_dict['loss'] < best_loss:
                    best_loss, best_atom_auroc, best_bond_auroc, best_it = loss_dict['loss'], loss_dict['atom_auroc'], loss_dict['bond_auroc'], it
                    ckpt_path = os.path.join(success_ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    early_stop = 0
                    logger.info(
                                f"Best val loss achieved at Iter {it}."
                            )
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
                                f"Val loss is not improved at Iter {it}. Best val loss: {best_loss:.6f} and Best atom auroc: {best_atom_auroc:.6f} and Best bond auroc: {best_bond_auroc:.6f} are achieved at Iter {best_it}."
                            )
                    logger.info('Early stop is {} for Iter {}'.format(early_stop, it))

                # if early_stop >= 20:
                #     logger.info('Early Stopping!!! No Improvement on Loss for 20 validation Iters.')
                #     break

    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    # Usage: python scripts/train.py --config configs/train.yml --device cuda:0 --logdir logs_train/
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='logs_train/')
    args = parser.parse_args()

    main(args)
