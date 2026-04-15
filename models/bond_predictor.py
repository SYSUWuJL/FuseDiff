# writen by hqy
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_geometric.nn import knn_graph
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.egnn import EgnnNet_BondPredictor
from .common import *
from .diffusion import *


class BondPredictor(Module):
    def __init__(self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,  # explicite bond type: 0, 1, 2, 3, 4
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
        self.protein_node_embedder = nn.Linear(protein_node_types, node_dim, bias=False) # protein element type
        self.protein_edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # protein bond type
        self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim, bias=False)  # ligand element type
        self.ligand_edge_embedder = nn.Linear(ligand_node_types * 2, edge_dim - time_dim, bias=False) # the init edge features
        if self.num_timesteps != 0:
            self.time_emb = GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear')
        # # predictor
        self.encoder = EgnnNet_BondPredictor(config.node_dim, config.edge_dim, **config.encoder)
        self.edge_decoder = MLP(config.edge_dim + config.node_dim, num_edge_types, config.edge_dim, num_layer=3)
        
        self.edge_weight = torch.tensor([0.1]+[1.]*(self.num_edge_types-1), dtype=torch.float32)
        self.ce_loss = torch.nn.CrossEntropyLoss(self.edge_weight)


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        if self.num_timesteps == 0:
            return
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (define for compatibility)

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
        
    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
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
        edge_type = torch.zeros_like(edge_type)
        edge_index = torch.index_select(edge_index, dim=1, index=nonzero_indices)
        edge_type = F.one_hot(edge_type, num_classes=self.num_edge_types)
        return edge_type, edge_index

    def forward(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node_pert, ligand_pos_pert, ligand_batch,
        ligand_edge_index, ligand_edge_batch, t
    ):
        """
        Predict the edge type of edges defined by halfedge_index
        """
        
        # embedding 
        protein_h = self.protein_node_embedder(protein_node)
        ligand_edge_pert = torch.cat([ligand_node_pert[ligand_edge_index[0]], ligand_node_pert[ligand_edge_index[1]]], dim=-1)
        if self.num_timesteps != 0:
            time_embed_node = self.time_emb(t.index_select(0, ligand_batch))
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node], dim=-1)
            time_embed_node = self.time_emb(t.index_select(0, ligand_edge_batch))
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_node], dim=-1)
        else:
            ligand_node_h_pert = self.ligand_node_embedder(ligand_node_pert)
            ligand_edge_h_pert = self.ligand_edge_embedder(ligand_edge_pert)
            t = torch.zeros(ligand_batch.max() + 1, device=ligand_pos_pert.device, dtype=torch.long)

        if self.config.node_indicator:
            protein_h = torch.cat([protein_h, torch.zeros(len(protein_h), 1).to(protein_h)], -1)
            ligand_node_h_pert = torch.cat([ligand_node_h_pert, torch.ones(len(ligand_node_h_pert), 1).to(ligand_node_h_pert)], -1)

        # combine protein and ligand input
        all_node_h, all_node_pos, all_node_batch, ligand_mask = compose(
            protein_h, protein_pos, protein_batch, ligand_node_h_pert, ligand_pos_pert, ligand_batch
        )

        sub_edge_index = self._get_edge_index(all_node_pos, all_node_batch, ligand_mask)
        sub_edge_type, sub_edge_index = self._get_edge_type(sub_edge_index, ligand_mask)
        sub_edge_batch = all_node_batch[sub_edge_index[0]]
        sub_edge_h = self.protein_edge_embedder(sub_edge_type.to(torch.float32))
        node_batch_counts = torch.bincount(all_node_batch)
        ligand_node_batch_counts = torch.bincount(ligand_batch)
        cumulative_nodes = torch.cat([torch.tensor([0]).to(all_node_batch.device), torch.cumsum(node_batch_counts, dim=0)[:-1]])
        cumulative_ligand_nodes = torch.cat([torch.tensor([0]).to(ligand_batch.device), torch.cumsum(ligand_node_batch_counts, dim=0)[:-1]])
        new_ligand_edge_index = ligand_edge_index + cumulative_nodes[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        all_edge_h, all_edge_index, all_edge_batch, ligand_edge_mask = edge_compose(
            sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h_pert, new_ligand_edge_index, ligand_edge_batch
        )

        node_h, _, edge_h = self.encoder(
            node_h=all_node_h,
            node_pos=all_node_pos, 
            edge_h=all_edge_h, 
            edge_index=all_edge_index,
            node_time=t.index_select(0, all_node_batch).unsqueeze(-1) / max(self.num_timesteps, 1),
            edge_time=t.index_select(0, all_edge_batch).unsqueeze(-1) / max(self.num_timesteps, 1),
            ligand_mask=ligand_mask
        )
        
        ligand_node_h = node_h[ligand_mask]
        ligand_edge_h = edge_h[ligand_edge_mask]
        n_halfedges = ligand_edge_h.shape[0] // 2
        edge_h_extend = torch.cat([
            ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:],
            ligand_node_h[ligand_edge_index[0, :n_halfedges]] + ligand_node_h[ligand_edge_index[1, :n_halfedges]],
        ], dim=-1)
        pred_ligand_halfedge = self.edge_decoder(edge_h_extend)
        
        return pred_ligand_halfedge

    def get_loss(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node, ligand_pos, ligand_batch,
        halfedge_type, halfedge_index, halfedge_batch,
        num_mol
    ):
        num_graphs = num_mol
        device = ligand_pos.device
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, protein_batch, ligand_batch, mode=self.center_pos_mode
        )
        if self.num_timesteps != 0:
            time_step, _ = self.sample_time(num_graphs, device)
        else:
            time_step = None 

        # 1. prepare node hidden  (can be compatible for discrete and continuous)
        if self.num_timesteps != 0:
            ligand_pos_pert = self.pos_transition.add_noise(ligand_pos, time_step, ligand_batch)
            node_pert = self.node_transition.add_noise(ligand_node, time_step, ligand_batch)
            ligand_node_pert = node_pert[0]  # compatible for both discrete and continuous catergorical_space
        else:
            ligand_node_pert = F.one_hot(ligand_node, self.ligand_node_types).float()
            ligand_pos_pert = ligand_pos
        
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)

        # 2. forward to denoise
        pred_ligand_halfedge = self(
            protein_node, protein_pos, protein_batch, 
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_index, ligand_edge_batch, time_step
        )

        # 3. loss
        # 3.1 edge type
        loss_edge = self.ce_loss(pred_ligand_halfedge, halfedge_type)

        # 3.2 total
        loss_total = loss_edge
        
        loss_dict = {
            'loss': loss_total,
            'loss_edge': loss_edge,
        }
        pred_dict = {
            'pred_ligand_halfedge': F.softmax(pred_ligand_halfedge, dim=-1)
        }
        return loss_dict, pred_dict
    
