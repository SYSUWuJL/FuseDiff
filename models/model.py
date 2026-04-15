from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_scatter import scatter_mean
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.egnn import EgnnNet
from .common import *
from .diffusion import *


class FuseDiff(Module):
    def __init__(self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.protein_node_types = protein_node_types
        self.ligand_node_types = ligand_node_types
        self.num_edge_types = num_edge_types
        self.k = config.knn
        self.cutoff_mode = config.cutoff_mode
        self.center_pos_mode = config.center_pos_mode
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        if self.config.node_indicator:
            node_dim = config.node_dim - 1
        else:
            node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        class_dim = config.class_dim
        class_emb_dim = config.class_emb_dim
        self.protein_node_embedder = nn.Linear(protein_node_types, node_dim, bias=False) # protein element type
        # self.protein_edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # protein bond type
        self.protein_edge_embedder = nn.Linear(3, edge_dim, bias=False) # protein bond type
        if self.config.train_mode in ('ori', 'no_bond'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim - class_emb_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim - class_emb_dim, bias=False) # ligand bond type
        elif self.config.train_mode in ('no_lab', 'no_both'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim, bias=False) # ligand bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )
        self.class_emb = nn.Sequential(
            nn.Linear(class_dim, class_emb_dim * 4),
            nn.LayerNorm(class_emb_dim * 4),
            nn.GELU(),
            nn.Linear(class_emb_dim * 4, class_emb_dim)
        )
        
        # # denoiser
        if config.denoiser.backbone == 'EGNN':
            self.denoiser = EgnnNet(config.node_dim, config.edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.ligand_node_decoder = MLP(config.node_dim, ligand_node_types, config.node_dim)
        self.ligand_edge_decoder = MLP(config.edge_dim, num_edge_types, config.edge_dim)


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.ligand_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.ligand_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    
    def fix_zero_time(self, num_graphs, device, **kwargs): 
        time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)   
        pt = torch.ones_like(time_step).float() / self.num_timesteps  
        return time_step, pt

    def _get_edge_index(self, x, batch, ligand_mask):
        if self.cutoff_mode == "knn":
            edge_index = knn_graph(x, k=self.k, batch=batch, flow="target_to_source")
        else:
            raise ValueError(
                f"Unsupported cutoff mode: {self.cutoff_mode}! Please select cutoff mode among knn, hybrid."
            )
        return edge_index

    def _get_edge_type(self, edge_index, ligand_mask):
        src, dst = edge_index
        edge_type = torch.zeros(len(src), dtype=torch.int64).to(edge_index.device)
        n_src = ligand_mask[src] == 1
        n_dst = ligand_mask[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3

        nonzero_indices = torch.nonzero(edge_type).flatten()
        edge_type = torch.index_select(edge_type, dim=0, index=nonzero_indices)
        # edge_type = torch.zeros_like(edge_type)
        edge_index = torch.index_select(edge_index, dim=1, index=nonzero_indices)
        # edge_type = F.one_hot(edge_type, num_classes=self.num_edge_types)
        edge_type = F.one_hot(edge_type - 1, num_classes=3)
        return edge_type, edge_index


    def forward(
        self, 
        protein_node_1, protein_pos_1, protein_batch_1, 
        protein_node_2, protein_pos_2, protein_batch_2, 
        ligand_node_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch,
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        t, lab
    ):
        """
        Predict Ligand at step `0` given perturbed Ligand at step `t` with hidden dims and time step
        """
        # 1 node, edge and time embedding
        time_embed_node = self.time_emb(t.index_select(0, ligand_batch))
        class_embed_node = self.class_emb(lab.index_select(0, ligand_batch))
        time_embed_edge = self.time_emb(t.index_select(0, ligand_edge_batch))
        class_embed_edge = self.class_emb(lab.index_select(0, ligand_edge_batch))
        if self.config.train_mode in ('ori', 'no_bond'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node, class_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge, class_embed_edge], dim=-1)
        elif self.config.train_mode in ('no_lab', 'no_both'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge], dim=-1)
        protein_h_1 = self.protein_node_embedder(protein_node_1)
        protein_h_2 = self.protein_node_embedder(protein_node_2)

        if self.config.node_indicator:
            protein_h_1 = torch.cat([protein_h_1, torch.zeros(len(protein_h_1), 1).to(protein_h_1)], -1)
            protein_h_2 = torch.cat([protein_h_2, torch.zeros(len(protein_h_2), 1).to(protein_h_2)], -1)
            ligand_node_h_pert = torch.cat([ligand_node_h_pert, torch.ones(len(ligand_node_h_pert), 1).to(ligand_node_h_pert)], -1)

        # 2 combine protein and ligand input
        all_node_h_1, all_node_pos_1, all_node_batch_1, ligand_mask_1 = compose(
            protein_h_1, protein_pos_1, protein_batch_1, ligand_node_h_pert, ligand_pos_pert_1, ligand_batch
        )
        all_node_h_2, all_node_pos_2, all_node_batch_2, ligand_mask_2 = compose(
            protein_h_2, protein_pos_2, protein_batch_2, ligand_node_h_pert, ligand_pos_pert_2, ligand_batch
        )
        
        sub_edge_index_1 = self._get_edge_index(all_node_pos_1, all_node_batch_1, ligand_mask_1)
        sub_edge_index_2 = self._get_edge_index(all_node_pos_2, all_node_batch_2, ligand_mask_2)
        sub_edge_type_1, sub_edge_index_1 = self._get_edge_type(sub_edge_index_1, ligand_mask_1)
        sub_edge_type_2, sub_edge_index_2 = self._get_edge_type(sub_edge_index_2, ligand_mask_2)
        sub_edge_batch_1 = all_node_batch_1[sub_edge_index_1[0]]
        sub_edge_batch_2 = all_node_batch_2[sub_edge_index_2[0]]
        sub_edge_h_1 = self.protein_edge_embedder(sub_edge_type_1.to(torch.float32))
        sub_edge_h_2 = self.protein_edge_embedder(sub_edge_type_2.to(torch.float32))
        node_batch_counts_1 = torch.bincount(all_node_batch_1)
        node_batch_counts_2 = torch.bincount(all_node_batch_2)
        ligand_node_batch_counts = torch.bincount(ligand_batch)
        cumulative_nodes_1 = torch.cat([torch.tensor([0]).to(all_node_batch_1.device), torch.cumsum(node_batch_counts_1, dim=0)[:-1]])
        cumulative_nodes_2 = torch.cat([torch.tensor([0]).to(all_node_batch_2.device), torch.cumsum(node_batch_counts_2, dim=0)[:-1]])
        cumulative_ligand_nodes = torch.cat([torch.tensor([0]).to(ligand_batch.device), torch.cumsum(ligand_node_batch_counts, dim=0)[:-1]])
        new_ligand_edge_index_1 = ligand_edge_index + cumulative_nodes_1[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        new_ligand_edge_index_2 = ligand_edge_index + cumulative_nodes_2[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        # all_edge_h, all_edge_index, all_edge_batch, ligand_edge_mask = edge_compose(
        #     sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h_pert, new_ligand_edge_index, ligand_edge_batch
        # )
        all_edge_h_1, all_edge_index_1, all_edge_batch_1, ligand_edge_mask_1 = edge_compose(
            sub_edge_h_1, sub_edge_index_1, sub_edge_batch_1, ligand_edge_h_pert, new_ligand_edge_index_1, ligand_edge_batch
        )
        all_edge_h_2, all_edge_index_2, all_edge_batch_2, ligand_edge_mask_2 = edge_compose(
            sub_edge_h_2, sub_edge_index_2, sub_edge_batch_2, ligand_edge_h_pert, new_ligand_edge_index_2, ligand_edge_batch
        )


        node_h_1, node_pos_1, edge_h_1, node_h_2, node_pos_2, edge_h_2 = self.denoiser(
            node_h_1=all_node_h_1,
            node_pos_1=all_node_pos_1, 
            edge_h_1=all_edge_h_1, 
            edge_index_1=all_edge_index_1,
            node_time_1=t.index_select(0, all_node_batch_1).unsqueeze(-1) / self.num_timesteps,
            edge_time_1=t.index_select(0, all_edge_batch_1).unsqueeze(-1) / self.num_timesteps,
            ligand_mask_1=ligand_mask_1, 
            ligand_edge_mask_1 = ligand_edge_mask_1,
            
            node_h_2=all_node_h_2,
            node_pos_2=all_node_pos_2, 
            edge_h_2=all_edge_h_2, 
            edge_index_2=all_edge_index_2,
            node_time_2=t.index_select(0, all_node_batch_2).unsqueeze(-1) / self.num_timesteps,
            edge_time_2=t.index_select(0, all_edge_batch_2).unsqueeze(-1) / self.num_timesteps,
            ligand_mask_2=ligand_mask_2, 
            ligand_edge_mask_2 = ligand_edge_mask_2,
        )

        # assert torch.allclose(node_h_1[ligand_mask_1], node_h_2[ligand_mask_2], rtol=1e-5, atol=1e-5)
        # assert torch.allclose(edge_h_1[ligand_edge_mask_1], edge_h_2[ligand_edge_mask_2], rtol=1e-5, atol=1e-5)
    
        ligand_node_h = (node_h_1[ligand_mask_1] + node_h_2[ligand_mask_2]) / 2.
        ligand_node_pos_1 = node_pos_1[ligand_mask_1]
        ligand_node_pos_2 = node_pos_2[ligand_mask_2]
        ligand_edge_h = (edge_h_1[ligand_edge_mask_1] + edge_h_2[ligand_edge_mask_2]) / 2.
        n_halfedges = ligand_edge_h.shape[0] // 2
        pred_ligand_node = self.ligand_node_decoder(ligand_node_h)
        pred_ligand_halfedge = self.ligand_edge_decoder(ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:])
        pred_ligand_pos_1 = ligand_node_pos_1
        pred_ligand_pos_2 = ligand_node_pos_2
        
        return {
            'pred_ligand_node': pred_ligand_node,
            'pred_ligand_pos_1': pred_ligand_pos_1,
            'pred_ligand_pos_2': pred_ligand_pos_2,
            'pred_ligand_halfedge': pred_ligand_halfedge
        }  # ligand at step 0

    def get_loss(
        self, 
        protein_node_1, protein_pos_1, protein_batch_1, 
        protein_node_2, protein_pos_2, protein_batch_2, 
        ligand_node, ligand_pos_1, ligand_pos_2, ligand_batch,
        halfedge_type, halfedge_index, halfedge_batch,
        num_mol, batch_lab
    ):
        num_graphs = num_mol
        device = ligand_node.device
        protein_pos_1, ligand_pos_1, _ = center_pos(
            protein_pos_1, ligand_pos_1, protein_batch_1, ligand_batch, mode=self.center_pos_mode
        )
        protein_pos_2, ligand_pos_2, _ = center_pos(
            protein_pos_2, ligand_pos_2, protein_batch_2, ligand_batch, mode=self.center_pos_mode
        )

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2. perturb pos, node, edge
        pos_pert_1 = self.pos_transition.add_noise(ligand_pos_1, time_step, ligand_batch)
        pos_pert_2 = self.pos_transition.add_noise(ligand_pos_2, time_step, ligand_batch)
        node_pert = self.node_transition.add_noise(ligand_node, time_step, ligand_batch)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, halfedge_batch)
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        if self.categorical_space == 'discrete':
            ligand_node_pert, log_node_t, log_node_0 = node_pert
            ligand_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            ligand_node_pert, ligand_node_0 = node_pert
            ligand_halfedge_pert, ligand_halfedge_0 = halfedge_pert
        
        ligand_edge_pert = torch.cat([ligand_halfedge_pert, ligand_halfedge_pert], dim=0)
        ligand_pos_pert_1 = pos_pert_1
        ligand_pos_pert_2 = pos_pert_2

        # 3. forward to denoise
        preds = self.forward(
            protein_node_1, protein_pos_1, protein_batch_1,
            protein_node_2, protein_pos_2, protein_batch_2,
            ligand_node_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )
        pred_ligand_node = preds['pred_ligand_node']
        pred_ligand_pos_1 = preds['pred_ligand_pos_1']
        pred_ligand_pos_2 = preds['pred_ligand_pos_2']
        pred_ligand_halfedge = preds['pred_ligand_halfedge']

        # 4. loss
        # 4.1 pos loss
        loss_pos_1 = F.mse_loss(pred_ligand_pos_1, ligand_pos_1) * 0.5
        loss_pos_2 = F.mse_loss(pred_ligand_pos_2, ligand_pos_2) * 0.5
        loss_pos = loss_pos_1 + loss_pos_2
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length_1 = torch.norm(ligand_pos_1[bond_index[0]] - ligand_pos_1[bond_index[1]], dim=-1)
            true_length_2 = torch.norm(ligand_pos_2[bond_index[0]] - ligand_pos_2[bond_index[1]], dim=-1)
            pred_length_1 = torch.norm(pred_ligand_pos_1[bond_index[0]] - pred_ligand_pos_1[bond_index[1]], dim=-1)
            pred_length_2 = torch.norm(pred_ligand_pos_2[bond_index[0]] - pred_ligand_pos_2[bond_index[1]], dim=-1)
            loss_len_1 = F.mse_loss(pred_length_1, true_length_1) * 0.5
            loss_len_2 = F.mse_loss(pred_length_2, true_length_2) * 0.5
            loss_len = loss_len_1 + loss_len_2
    
        if self.categorical_space == 'discrete':
            # 4.2 node type loss
            log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, ligand_batch, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, ligand_batch, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=ligand_batch)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type loss
            log_halfedge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, log_halfedge_0, t=time_step, batch=halfedge_batch)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_ligand_node, ligand_node_0)  * 30
            loss_edge = F.mse_loss(pred_ligand_halfedge, ligand_halfedge_0) * 30

        # total loss
        if self.config.train_mode in ('ori', 'no_lab'):
            loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_pos_1': loss_pos_1,
            'loss_pos_2': loss_pos_2,
            'loss_node': loss_node,
            'loss_edge': loss_edge
        }
        elif self.config.train_mode in ('no_bond', 'no_both'):
            loss_total = loss_pos + loss_node + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
                'loss': loss_total,
                'loss_pos': loss_pos,
                'loss_node': loss_node
            }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
            loss_dict['loss_len_1'] = loss_len_1
            loss_dict['loss_len_2'] = loss_len_2

        pred_dict = {
            'pred_ligand_node': F.softmax(pred_ligand_node, dim=-1),
            'pred_ligand_pos_1': pred_ligand_pos_1,
            'pred_ligand_pos_2': pred_ligand_pos_2,
            'pred_ligand_halfedge': F.softmax(pred_ligand_halfedge, dim=-1)
        }
        return loss_dict, pred_dict

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_eps = extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - \
                      extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch) * eps
        return pos0_from_eps

    def _predict_eps_from_x0(self, xt, t, pred_x0, batch):
        return (
            (extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - pred_x0) /
            extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch)
        )


    def classifier_free(
        self, 
        protein_node_1, protein_pos_1, protein_batch_1,
        protein_node_2, protein_pos_2, protein_batch_2,
        ligand_node_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch, 
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        gui_strength, time_step, batch_lab
    ):
        """
        Compute new results for the start step in classifier free diffusion sampling.
        """
        preds_cond = self.forward(
            protein_node_1, protein_pos_1, protein_batch_1,
            protein_node_2, protein_pos_2, protein_batch_2,
            ligand_node_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )

        batch_lab_zero = torch.zeros(batch_lab.shape, device=ligand_batch.device)

        preds_uncond = self.forward(
            protein_node_1, protein_pos_1, protein_batch_1,
            protein_node_2, protein_pos_2, protein_batch_2,
            ligand_node_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab_zero
        )

        pred_eps_cond_1 = self._predict_eps_from_x0(
            xt=ligand_pos_pert_1, t=time_step, pred_x0=preds_cond['pred_ligand_pos_1'], batch=ligand_batch
        )
        pred_eps_cond_2 = self._predict_eps_from_x0(
            xt=ligand_pos_pert_2, t=time_step, pred_x0=preds_cond['pred_ligand_pos_2'], batch=ligand_batch
        )

        pred_eps_uncond_1 = self._predict_eps_from_x0(
            xt=ligand_pos_pert_1, t=time_step, pred_x0=preds_uncond['pred_ligand_pos_1'], batch=ligand_batch
        )
        pred_eps_uncond_2 = self._predict_eps_from_x0(
            xt=ligand_pos_pert_2, t=time_step, pred_x0=preds_uncond['pred_ligand_pos_2'], batch=ligand_batch
        )

        pred_eps_1 = (1 + gui_strength) * pred_eps_cond_1 - gui_strength * pred_eps_uncond_1
        pred_eps_2 = (1 + gui_strength) * pred_eps_cond_2 - gui_strength * pred_eps_uncond_2

        pred_ligand_pos_1 = self._predict_x0_from_eps(xt=ligand_pos_pert_1, t=time_step, eps=pred_eps_1, batch=ligand_batch)
        pred_ligand_pos_2 = self._predict_x0_from_eps(xt=ligand_pos_pert_2, t=time_step, eps=pred_eps_2, batch=ligand_batch)

        pred_ligand_node = preds_cond['pred_ligand_node'] + preds_uncond['pred_ligand_node']
        pred_ligand_halfedge = preds_cond['pred_ligand_halfedge'] + preds_uncond['pred_ligand_halfedge']

        return pred_ligand_pos_1, pred_ligand_pos_2, pred_ligand_node, pred_ligand_halfedge


    @torch.no_grad()
    def sample(
        self, n_graphs, 
        protein_node_1, protein_pos_1, protein_batch_1,
        protein_node_2, protein_pos_2, protein_batch_2,
        ligand_batch, halfedge_index, halfedge_batch, 
        batch_lab=None, gui_strength=None, 
        bond_predictor=None, guidance=None
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_all = len(ligand_batch)
        n_halfedges_all = len(halfedge_batch)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init
            
        pocket_center_pos_1 = scatter_mean(protein_pos_1, protein_batch_1, dim=0)
        pocket_center_pos_2 = scatter_mean(protein_pos_2, protein_batch_2, dim=0)
        ligand_center_pos_1 = pocket_center_pos_1[ligand_batch]
        ligand_center_pos_2 = pocket_center_pos_2[ligand_batch]
        assert torch.equal(torch.tensor(ligand_center_pos_1.shape), torch.tensor(ligand_center_pos_2.shape))
        init_noise_1 = torch.randn_like(ligand_center_pos_1)
        init_noise_2 = torch.randn_like(ligand_center_pos_2)
        ligand_pos_init_1 = ligand_center_pos_1 + init_noise_1
        ligand_pos_init_2 = ligand_center_pos_2 + init_noise_2
        protein_pos_1, ligand_pos_init_1, offset_1 = center_pos(protein_pos_1, ligand_pos_init_1, protein_batch_1, ligand_batch, self.center_pos_mode)
        protein_pos_2, ligand_pos_init_2, offset_2 = center_pos(protein_pos_2, ligand_pos_init_2, protein_batch_2, ligand_batch, self.center_pos_mode)

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert_1 = ligand_pos_init_1
        ligand_pos_pert_2 = ligand_pos_init_2
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)

        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            # # 2.1 inference
            if self.config.train_mode in ('ori', 'no_bond'):
                pred_ligand_pos_1, pred_ligand_pos_2, pred_ligand_node, pred_ligand_halfedge = self.classifier_free(
                    protein_node_1, protein_pos_1, protein_batch_1,
                    protein_node_2, protein_pos_2, protein_batch_2,
                    ligand_node_h_pert, ligand_pos_pert_1, ligand_pos_pert_2, ligand_batch, 
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    gui_strength, time_step, batch_lab
                )
            else:
                raise ValueError

            # 2.2 get the t - 1 state
            # pos 
            ligand_pos_prev_1 = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert_1, x_recon=pred_ligand_pos_1, t=time_step, batch=ligand_batch
            )
            ligand_pos_prev_2 = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert_2, x_recon=pred_ligand_pos_2, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)

                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
            
            # # 2.3 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):    # train_mode: ori
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in_1 = ligand_pos_pert_1.detach().requires_grad_(True)
                            ligand_pos_in_2 = ligand_pos_pert_2.detach().requires_grad_(True)
                            pred_bondpredictor_1 = bond_predictor(
                                protein_node_1, protein_pos_1, protein_batch_1,
                                ligand_node_h_in, ligand_pos_in_1, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            pred_bondpredictor_2 = bond_predictor(
                                protein_node_2, protein_pos_2, protein_batch_2,
                                ligand_node_h_in, ligand_pos_in_2, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta_1 = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor_1, ligand_pos_in_1, ligand_halfedge_type_prev, log_halfedge_type)
                            delta_2 = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor_2, ligand_pos_in_2, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev_1 = ligand_pos_prev_1 + delta_1
                        ligand_pos_prev_2 = ligand_pos_prev_2 + delta_2

            # # 2.5 update t-1
            ligand_pos_pert_1 = ligand_pos_prev_1
            ligand_pos_pert_2 = ligand_pos_prev_2
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev
        

        pred_ligand_pos_1 = pred_ligand_pos_1 + offset_1[ligand_batch]
        pred_ligand_pos_2 = pred_ligand_pos_2 + offset_2[ligand_batch]
        # # 3. get the final positions

        return {
            'pred': [pred_ligand_node, pred_ligand_pos_1, pred_ligand_pos_2, pred_ligand_halfedge],
        }


    def bond_guidance(self, gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, halfedge_type_prev, log_halfedge_type):
        if gui_type == 'entropy':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty':
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log().sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
            prob = torch.softmax(pred_bondpredictor, dim=-1)
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log()
            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'entropy_bond':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log()
            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit_bond':
            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit':
            ind_bond_notmask = (halfedge_type_prev <= 4)
            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent':
            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent_bond':
            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        else:
            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
        
        return delta

