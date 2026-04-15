import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean


class TimeEmbedder(nn.Module):
    def __init__(self, dim, total_time):
        super().__init__()
        self.dim = dim
        self.total_time = total_time

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        rel_t = x / (self.total_time - 1)
        emb_sin  = [torch.sin((rel_t + bias) * 0.5 * np.pi) for bias in torch.linspace(0, 1, half_dim+1, device=device)[:-1]]
        emb_cos = [torch.cos((rel_t + bias) * 0.5 * np.pi) for bias in torch.linspace(0, 1, self.dim - half_dim+1, device=device)[:-1]]
        emb = torch.stack(emb_sin + emb_cos, dim=-1)
        return emb

class SineTimeEmbedder(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=5000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

## --- torch utils ---
def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

## --- probabily ---

# categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)

# ----- beta  schedule -----

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod

def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def get_beta_schedule(beta_schedule, num_timesteps, **kwargs):
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    kwargs['beta_start'] ** 0.5,
                    kwargs['beta_end'] ** 0.5,
                    num_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = kwargs['beta_end'] * np.ones(num_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_timesteps, 1, num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        s = dict.get(kwargs, 's', 6)
        betas = np.linspace(-s, s, num_timesteps)
        betas = sigmoid(betas) * (kwargs['beta_end'] - kwargs['beta_start']) + kwargs['beta_start']
    elif beta_schedule == "cosine":
        s = dict.get(kwargs, 's', 0.008)
        betas = cosine_beta_schedule(num_timesteps, s=s)
    elif beta_schedule == "advance":
        scale_start = dict.get(kwargs, 'scale_start', 0.999)
        scale_end = dict.get(kwargs, 'scale_end', 0.001)
        width = dict.get(kwargs, 'width', 2)
        betas = advance_schedule(num_timesteps, scale_start, scale_end, width)
    elif beta_schedule == "segment":
        betas = segment_schedule(num_timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_timesteps,)
    return betas

def compose(protein_h, protein_pos, protein_batch, ligand_h, ligand_pos, ligand_batch):
    all_batch = torch.cat([ligand_batch, protein_batch], dim=0)
    sort_idx = torch.sort(all_batch, stable=True).indices
    all_batch = all_batch[sort_idx]

    ligand_mask = torch.cat([
        torch.ones(ligand_batch.size(0), device=ligand_batch.device).bool(),
        torch.zeros(protein_batch.size(0), device=protein_batch.device).bool()
    ], dim=0)[sort_idx]
    all_h = torch.cat([ligand_h, protein_h], dim=0)[sort_idx]
    all_pos = torch.cat([ligand_pos, protein_pos], dim=0)[sort_idx]

    return all_h, all_pos, all_batch, ligand_mask

def edge_compose(sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h, ligand_edge_index, ligand_edge_batch):
    all_batch = torch.cat([ligand_edge_batch, sub_edge_batch], dim=0)

    ligand_edge_mask = torch.cat([
        torch.ones(ligand_edge_batch.size(0), device=ligand_edge_batch.device).bool(),
        torch.zeros(sub_edge_batch.size(0), device=sub_edge_batch.device).bool()
    ], dim=0)
    all_edge_h = torch.cat([ligand_edge_h, sub_edge_h], dim=0)
    all_edge_index = torch.cat([ligand_edge_index, sub_edge_index], dim=1)

    return all_edge_h, all_edge_index, all_batch, ligand_edge_mask

def frag_ligand_compose(frag_h, frag_pos, frag_batch, ligand_h, ligand_pos, ligand_batch):
    all_batch = torch.cat([ligand_batch, frag_batch], dim=0)

    ligand_mask = torch.cat([
        torch.ones(ligand_batch.size(0), device=ligand_batch.device).bool(),
        torch.zeros(frag_batch.size(0), device=frag_batch.device).bool()
    ], dim=0)
    all_h = torch.cat([ligand_h, frag_h], dim=0)
    all_pos = torch.cat([ligand_pos, frag_pos], dim=0)

    return all_h, all_pos, all_batch, ligand_mask

def center_pos(protein_pos, ligand_pos, protein_batch, ligand_batch, mode="protein"):
    if mode == "none":
        offset = 0.
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, protein_batch, dim=0)
        protein_pos = protein_pos - offset[protein_batch]
        ligand_pos = ligand_pos - offset[ligand_batch]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def get_fragment_mask(ligand_batch, frag_batch):
    unique_molecule_indices = torch.unique(ligand_batch)
    fragment_masks = []
    for molecule_index in unique_molecule_indices:
        fragment_indices = torch.where(frag_batch == molecule_index)[0]
        ligand_indices = torch.where(ligand_batch == molecule_index)[0]
        fragment_mask = torch.zeros_like(ligand_batch[ligand_indices], dtype=torch.bool)
        fragment_mask[:len(fragment_indices)] = True
        fragment_masks.append(fragment_mask)
    frag_mask = torch.cat(fragment_masks)
    return frag_mask

