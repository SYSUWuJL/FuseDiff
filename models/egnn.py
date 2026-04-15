import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import knn_graph
from models.common import GaussianSmearing, MLP


class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim+node_dim+1, hidden_dim, hidden_dim) # add 1 for time

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(
        self, 
        x_1, edge_index_1, edge_attr_1, node_time_1, ligand_mask_1, ligand_edge_mask_1, 
        x_2, edge_index_2, edge_attr_2, node_time_2, ligand_mask_2, ligand_edge_mask_2, 
        ):

        N_1 = x_1.size(0)
        N_2 = x_2.size(0)
        # ligand_N_1 = x_1[ligand_mask_1].size(0)
        # ligand_N_2 = x_2[ligand_mask_2].size(0)
        # assert ligand_N_1 == ligand_N_2
        row_1, col_1 = edge_index_1   # (E,) , (E,)
        row_2, col_2 = edge_index_2   # (E,) , (E,)

        node_h_1 = self.node_net(x_1)  # (N, H)
        node_h_2 = self.node_net(x_2)  # (N, H)

        # Compose messages
        edge_h_1 = self.edge_net(edge_attr_1)  # (E, H_per_head)
        edge_h_2 = self.edge_net(edge_attr_2)  # (E, H_per_head)
        msg_j_1 = self.msg_net(edge_h_1 * node_h_1[col_1])
        msg_j_2 = self.msg_net(edge_h_2 * node_h_2[col_2])

        if self.use_gate:
            gate_1 = self.gate(torch.cat([edge_attr_1, x_1[col_1], node_time_1[col_1]], dim=-1))
            gate_2 = self.gate(torch.cat([edge_attr_2, x_2[col_2], node_time_2[col_2]], dim=-1))
            msg_j_1 = msg_j_1 * torch.sigmoid(gate_1)
            msg_j_2 = msg_j_2 * torch.sigmoid(gate_2)

        # assert torch.allclose(msg_j_1[ligand_edge_mask_1], msg_j_2[ligand_edge_mask_2], rtol=1e-5, atol=1e-5)
        msg_j_1[ligand_edge_mask_1] = msg_j_2[ligand_edge_mask_2] = (msg_j_1[ligand_edge_mask_1] + msg_j_2[ligand_edge_mask_2]) / 2.

        # Aggregate messages
        aggr_msg_1 = scatter_sum(msg_j_1, row_1, dim=0, dim_size=N_1)
        aggr_msg_2 = scatter_sum(msg_j_2, row_2, dim=0, dim_size=N_2)

        # aggr_msg_ligand_1 = scatter_sum(msg_j_1[ligand_edge_mask_1], row_1[ligand_edge_mask_1], dim=0, dim_size=ligand_N_1)
        # aggr_msg_ligand_2 = scatter_sum(msg_j_2[ligand_edge_mask_2], row_2[ligand_edge_mask_2], dim=0, dim_size=ligand_N_2)
        aggr_msg_protein_1 = scatter_sum(msg_j_1[~ligand_edge_mask_1], row_1[~ligand_edge_mask_1], dim=0, dim_size=N_1)
        aggr_msg_protein_2 = scatter_sum(msg_j_2[~ligand_edge_mask_2], row_2[~ligand_edge_mask_2], dim=0, dim_size=N_2)

        # aggr_msg_1_new = aggr_msg_protein_1.clone()
        # aggr_msg_1_new[ligand_mask_1] = aggr_msg_1_new[ligand_mask_1] + aggr_msg_ligand_1 + aggr_msg_protein_2[ligand_mask_2]
        # aggr_msg_2_new = aggr_msg_protein_2.clone()
        # aggr_msg_2_new[ligand_mask_2] = aggr_msg_2_new[ligand_mask_2] + aggr_msg_ligand_2 + aggr_msg_protein_1[ligand_mask_1]

        aggr_msg_1[ligand_mask_1] = aggr_msg_1[ligand_mask_1] + aggr_msg_protein_2[ligand_mask_2]
        aggr_msg_2[ligand_mask_2] = aggr_msg_2[ligand_mask_2] + aggr_msg_protein_1[ligand_mask_1]

        # assert torch.allclose(aggr_msg_1[ligand_mask_1], aggr_msg_2[ligand_mask_2], rtol=1e-5, atol=1e-5)
        aggr_msg_1[ligand_mask_1] = aggr_msg_2[ligand_mask_2] = (aggr_msg_1[ligand_mask_1] + aggr_msg_2[ligand_mask_2]) / 2.

        out_1 = self.centroid_lin(x_1) + aggr_msg_1
        out_2 = self.centroid_lin(x_2) + aggr_msg_2

        out_1 = self.layer_norm(out_1)
        out_2 = self.layer_norm(out_2)
        out_1 = self.out_transform(self.act(out_1))
        out_2 = self.out_transform(self.act(out_2))

        # assert torch.allclose(out_1[ligand_mask_1], out_2[ligand_mask_2], rtol=1e-5, atol=1e-5)
        out_1[ligand_mask_1] = out_2[ligand_mask_2] = (out_1[ligand_mask_1] + out_2[ligand_mask_2]) / 2.

        return out_1, out_2


    def forward_single(self, x, edge_index, edge_attr, node_time):
        # (node_h, edge_index, edge_h, node_time)
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
            H = 256
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        node_h = self.node_net(x)  # (N, H)

        # Compose messages
        edge_h = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(edge_h * node_h[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col], node_time[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class NodeEncoder(Module):
    
    def __init__(self, node_dim=256, edge_dim=64, key_dim=128, num_heads=4, 
                    num_blocks=6, k=48, cutoff=10.0, use_atten=True, use_gate=True,
                    dist_version='new'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
            self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim-self.additional_edge_feat)
            self.edge_emb = Linear(edge_dim, edge_dim)
        else:
            raise NotImplementedError('dist_version notimplemented')
        self.node_blocks = ModuleList()
        for _ in range(num_blocks):
            block = NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                use_atten=use_atten,
                use_gate=use_gate,
            )
            self.node_blocks.append(block)

    @property
    def out_channels(self):
        return self.node_dim

    def forward(self, h, pos, edge_index, is_mol):
        #NOTE in the encoder, the edge dose not change since the position of mol and protein is fixed
        edge_attr = self._add_edge_features(pos, edge_index, is_mol)
        for interaction in self.node_blocks:
            h = h + interaction(h, edge_index, edge_attr)
        return h

    @property
    def additional_edge_feat(self,):
        return 2

    def _add_edge_features(self, pos, edge_index, is_mol):
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        # 2-vector represent the two node types (atoms of protein or mol)
        edge_src_feat = is_mol[edge_index[0]].float().view(-1, 1)
        edge_dst_feat = is_mol[edge_index[1]].float().view(-1, 1)
        edge_attr = torch.cat([edge_attr, edge_src_feat, edge_dst_feat], dim=1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim+node_dim+1, out_dim, 32)  # +1 for time

    def forward(self, bond_feat_input, node_feat_input, time):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, time], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class QKVLin(Module):
    def __init__(self, h_dim, key_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_lin = Linear(h_dim, key_dim)
        self.k_lin = Linear(h_dim, key_dim)
        self.v_lin = Linear(h_dim, h_dim)

    def forward(self, inputs):
        n = inputs.size(0)
        return [
            self.q_lin(inputs).view(n, self.num_heads, -1),
            self.k_lin(inputs).view(n, self.num_heads, -1),
            self.v_lin(inputs).view(n, self.num_heads, -1),
        ]


class BondBlock(Module):
    def __init__(self, bond_dim, node_dim, use_gate=True, use_atten=False, num_heads=2, key_dim=128):
        super().__init__()
        self.use_atten = use_atten
        self.use_gate = use_gate
        inter_dim = bond_dim * 2

        self.bond_ffn_left = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        if self.use_atten:
            assert bond_dim % num_heads == 0
            assert key_dim % num_heads == 0
            # linear transformation for attention 
            self.qkv_left = QKVLin(bond_dim, key_dim, num_heads)
            self.qkv_right = QKVLin(bond_dim, key_dim, num_heads)
            self.layer_norm_atten1 = nn.LayerNorm(bond_dim)
            self.layer_norm_atten2 = nn.LayerNorm(bond_dim)
        
        self.node_ffn_left = Linear(node_dim, bond_dim)
        self.node_ffn_right = Linear(node_dim, bond_dim)

        self.self_ffn = Linear(bond_dim, bond_dim)
        self.layer_norm = nn.LayerNorm(bond_dim)
        self.out_transform = Linear(bond_dim, bond_dim)
        self.act = nn.ReLU()

    def forward(self, bond_h, bond_index, node_h, atten_index=None):
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        node_pos: (n, 3)
        """
        N = node_h.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(bond_h, node_h[left_node])
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(bond_h, node_h[right_node])
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        bond_h = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(node_h[left_node])
            + self.node_ffn_right(node_h[right_node])
            + self.self_ffn(bond_h)
        )
        bond_h = self.layer_norm(bond_h)

        if self.use_atten:
            index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right = atten_index

            # left node
            h_queries, h_keys, h_values = self.qkv_left(bond_h)
            queries_i = h_queries[index_query_bond_left]
            keys_j = h_keys[index_key_bond_left]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_left, dim=0)
            values_j = h_values[index_key_bond_left]
            num_attns = len(index_key_bond_left)
            bond_h = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_left, dim=0, dim_size=bond_h.size(0))
            bond_h = self.layer_norm_atten1(bond_h)

            # right node
            h_queries, h_keys, h_values = self.qkv_right(bond_h)
            queries_i = h_queries[index_query_bond_right]
            keys_j = h_keys[index_key_bond_right]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_right, dim=0)
            values_j = h_values[index_key_bond_right]
            num_attns = len(index_key_bond_right)
            bond_h = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_right, dim=0, dim_size=bond_h.size(0))
            bond_h = self.layer_norm_atten2(bond_h)

        bond_h = self.out_transform(self.act(bond_h))
        return bond_h


class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(
        self, 
        bond_h_1, bond_index_1, node_h_1, bond_time_1, ligand_mask_1, ligand_edge_mask_1, 
        bond_h_2, bond_index_2, node_h_2, bond_time_2, ligand_mask_2, ligand_edge_mask_2, 
        ):
        # (edge_h, edge_index, node_h, edge_time)
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        """
        N_1 = node_h_1.size(0)
        N_2 = node_h_2.size(0)
        left_node_1, right_node_1 = bond_index_1
        left_node_2, right_node_2 = bond_index_2

        msg_bond_left_1_all = self.bond_ffn_left(
            bond_h_1,
            node_h_1[left_node_1],
            bond_time_1,
        )                               # (b, hidden)

        msg_bond_left_1_agg = scatter_sum(
            msg_bond_left_1_all,
            right_node_1,
            dim=0,
            dim_size=N_1,
        )                               # (N_1, hidden)

        msg_bond_left_1 = msg_bond_left_1_agg[left_node_1]  # (b, hidden)


        msg_bond_left_2_all = self.bond_ffn_left(
            bond_h_2,
            node_h_2[left_node_2],
            bond_time_2,
        )                               # (b, hidden)

        msg_bond_left_2_agg = scatter_sum(
            msg_bond_left_2_all,
            right_node_2,
            dim=0,
            dim_size=N_2,
        )                               # (N_1, hidden)

        msg_bond_left_2 = msg_bond_left_2_agg[left_node_2]  # (b, hidden)



        msg_non_lig_src_1 = self.bond_ffn_left(
            bond_h_1[~ligand_edge_mask_1],
            node_h_1[left_node_1[~ligand_edge_mask_1]],
            bond_time_1[~ligand_edge_mask_1],
        )  # (b_non_lig, hidden)

        msg_non_lig_agg_1 = scatter_sum(
            msg_non_lig_src_1,
            right_node_1[~ligand_edge_mask_1],
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_left_1_non_ligand = msg_non_lig_agg_1[left_node_1]  # (b, hidden)


        msg_non_lig_src_2 = self.bond_ffn_left(
            bond_h_2[~ligand_edge_mask_2],
            node_h_2[left_node_2[~ligand_edge_mask_2]],
            bond_time_2[~ligand_edge_mask_2],
        )  # (b_non_lig, hidden)

        msg_non_lig_agg_2 = scatter_sum(
            msg_non_lig_src_2,
            right_node_2[~ligand_edge_mask_2],
            dim=0,
            dim_size=N_2,
        )  # (N_1, hidden)

        msg_bond_left_2_non_ligand = msg_non_lig_agg_2[left_node_2]  # (b, hidden)


        msg_bond_left_1[ligand_edge_mask_1] = msg_bond_left_1[ligand_edge_mask_1] + msg_bond_left_2_non_ligand[ligand_edge_mask_2]
        msg_bond_left_2[ligand_edge_mask_2] = msg_bond_left_2[ligand_edge_mask_2] + msg_bond_left_1_non_ligand[ligand_edge_mask_1]
        # assert torch.allclose(msg_bond_left_1[ligand_edge_mask_1], msg_bond_left_2[ligand_edge_mask_2], rtol=1e-5, atol=1e-5)
        msg_bond_left_1[ligand_edge_mask_1] = msg_bond_left_2[ligand_edge_mask_2] = (msg_bond_left_1[ligand_edge_mask_1] + msg_bond_left_2[ligand_edge_mask_2]) / 2.
        # msg_bond_left_2[ligand_edge_mask_2] = msg_bond_left_1[ligand_edge_mask_1]

        msg_bond_right_1_all = self.bond_ffn_right(
            bond_h_1,
            node_h_1[right_node_1],
            bond_time_1,
        )  # (b1, hidden)

        msg_bond_right_1_agg = scatter_sum(
            msg_bond_right_1_all,
            left_node_1,
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_right_1 = msg_bond_right_1_agg[right_node_1]  # (b1, hidden)


        msg_bond_right_2_all = self.bond_ffn_right(
            bond_h_2,
            node_h_2[right_node_2],
            bond_time_2,
        )  # (b2, hidden)

        msg_bond_right_2_agg = scatter_sum(
            msg_bond_right_2_all,
            left_node_2,
            dim=0,
            dim_size=N_2,
        )  # (N_2, hidden)

        msg_bond_right_2 = msg_bond_right_2_agg[right_node_2]  # (b2, hidden)


        msg_non_lig_right_src_1 = self.bond_ffn_right(
            bond_h_1[~ligand_edge_mask_1],
            node_h_1[right_node_1[~ligand_edge_mask_1]],
            bond_time_1[~ligand_edge_mask_1],
        )  # (b1_non_lig, hidden)

        msg_non_lig_right_agg_1 = scatter_sum(
            msg_non_lig_right_src_1,
            left_node_1[~ligand_edge_mask_1],
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_right_1_non_ligand = msg_non_lig_right_agg_1[right_node_1]  # (b1, hidden)


        msg_non_lig_right_src_2 = self.bond_ffn_right(
            bond_h_2[~ligand_edge_mask_2],
            node_h_2[right_node_2[~ligand_edge_mask_2]],
            bond_time_2[~ligand_edge_mask_2],
        )  # (b2_non_lig, hidden)

        msg_non_lig_right_agg_2 = scatter_sum(
            msg_non_lig_right_src_2,
            left_node_2[~ligand_edge_mask_2],
            dim=0,
            dim_size=N_2,
        )  # (N_2, hidden)

        msg_bond_right_2_non_ligand = msg_non_lig_right_agg_2[right_node_2]  # (b2, hidden)


        msg_bond_right_1[ligand_edge_mask_1] = msg_bond_right_1[ligand_edge_mask_1] + msg_bond_right_2_non_ligand[ligand_edge_mask_2]
        msg_bond_right_2[ligand_edge_mask_2] = msg_bond_right_2[ligand_edge_mask_2] + msg_bond_right_1_non_ligand[ligand_edge_mask_1]
        # assert torch.allclose(msg_bond_right_1[ligand_edge_mask_1], msg_bond_right_2[ligand_edge_mask_2], rtol=1e-5, atol=1e-5)
        msg_bond_right_1[ligand_edge_mask_1] = msg_bond_right_2[ligand_edge_mask_2] = (msg_bond_right_1[ligand_edge_mask_1] + msg_bond_right_2[ligand_edge_mask_2]) / 2.
        # msg_bond_right_2[ligand_edge_mask_2] = msg_bond_right_1[ligand_edge_mask_1]


        bond_h_1 = (
            msg_bond_left_1 + msg_bond_right_1
            + self.node_ffn_left(node_h_1[left_node_1])
            + self.node_ffn_right(node_h_1[right_node_1])
            + self.self_ffn(bond_h_1)
        )
        bond_h_2 = (
            msg_bond_left_2 + msg_bond_right_2
            + self.node_ffn_left(node_h_2[left_node_2])
            + self.node_ffn_right(node_h_2[right_node_2])
            + self.self_ffn(bond_h_2)
        )

        # assert torch.allclose(bond_h_1[ligand_edge_mask_1], bond_h_2[ligand_edge_mask_2], rtol=1e-4, atol=1e-4)
        # bond_h_1[ligand_edge_mask_1] = bond_h_2[ligand_edge_mask_2] = (bond_h_1[ligand_edge_mask_1] + bond_h_2[ligand_edge_mask_2]) / 2.
    

        bond_h_1 = self.layer_norm(bond_h_1)
        bond_h_2 = self.layer_norm(bond_h_2)

        bond_h_1 = self.out_transform(self.act(bond_h_1))
        bond_h_2 = self.out_transform(self.act(bond_h_2))

        # assert torch.allclose(bond_h_1[ligand_edge_mask_1], bond_h_2[ligand_edge_mask_2], rtol=1e-6, atol=1e-6)
        bond_h_1[ligand_edge_mask_1] = bond_h_2[ligand_edge_mask_2] = (bond_h_1[ligand_edge_mask_1] + bond_h_2[ligand_edge_mask_2]) / 2.

        return bond_h_1, bond_h_2


    def forward_single(self, bond_h, bond_index, node_h, bond_time):
        # (edge_h, edge_index, node_h, edge_time)
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        """
        N = node_h.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(bond_h, node_h[left_node], bond_time)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(bond_h, node_h[right_node], bond_time)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        bond_h = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(node_h[left_node])
            + self.node_ffn_right(node_h[right_node])
            + self.self_ffn(bond_h)
        )
        bond_h = self.layer_norm(bond_h)

        bond_h = self.out_transform(self.act(bond_h))
        return bond_h


class EgnnNet(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, attention_dim=128, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        # self.edge_embs = ModuleList()
        self.edge_embs_protein = ModuleList()
        self.edge_embs_ligand = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            # self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            self.edge_embs_protein.append(Linear(input_edge_dim, edge_dim))
            self.edge_embs_ligand.append(Linear(input_edge_dim + num_gaussians, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))

    def forward(
        self, 
        node_h_1, node_pos_1, edge_h_1, edge_index_1, node_time_1, edge_time_1, ligand_mask_1, ligand_edge_mask_1,
        node_h_2, node_pos_2, edge_h_2, edge_index_2, node_time_2, edge_time_2, ligand_mask_2, ligand_edge_mask_2, ):

        for i in range(self.num_blocks):

            # assert torch.equal(node_h_1[ligand_mask_1], node_h_2[ligand_mask_2])
            # assert torch.equal(edge_h_1[ligand_edge_mask_1], edge_h_2[ligand_edge_mask_2])
    
            # edge fetures before each block
            if self.update_pos or (i==0):
                edge_h_dist_1, relative_vec_1, distance_1 = self._build_edges_dist(node_pos_1, edge_index_1)
                edge_h_dist_2, relative_vec_2, distance_2 = self._build_edges_dist(node_pos_2, edge_index_2)

            if self.update_edge:
                # edge_h_1 = torch.cat([edge_h_1, edge_h_dist_1], dim=-1)
                # edge_h_2 = torch.cat([edge_h_2, edge_h_dist_2], dim=-1)

                edge_h_protein_1 = torch.cat([
                    edge_h_1[~ligand_edge_mask_1], edge_h_dist_1[~ligand_edge_mask_1]], dim=-1)
                edge_h_protein_2 = torch.cat([
                    edge_h_2[~ligand_edge_mask_2], edge_h_dist_2[~ligand_edge_mask_2]], dim=-1)

                edge_h_ligand_1 = torch.cat([
                    edge_h_1[ligand_edge_mask_1], edge_h_dist_1[ligand_edge_mask_1], edge_h_dist_2[ligand_edge_mask_2]], dim=-1)
                edge_h_ligand_2 = torch.cat([
                    edge_h_2[ligand_edge_mask_2], edge_h_dist_1[ligand_edge_mask_1], edge_h_dist_2[ligand_edge_mask_2]], dim=-1)
                
                                
                # sum_feat = edge_h_dist_1[ligand_edge_mask_1] + edge_h_dist_2[ligand_edge_mask_2]
                # diff_feat = (edge_h_dist_1[ligand_edge_mask_1] - edge_h_dist_2[ligand_edge_mask_2]).abs()
                # edge_h_ligand_1 = torch.cat([
                #     edge_h_1[ligand_edge_mask_1], sum_feat, diff_feat], dim=-1)
                # edge_h_ligand_2 = torch.cat([
                #     edge_h_2[ligand_edge_mask_2], sum_feat, diff_feat], dim=-1)
                
                # edge_h_ligand_1 = torch.cat([
                #     edge_h_1[ligand_edge_mask_1], (edge_h_dist_1[ligand_edge_mask_1] + edge_h_dist_2[ligand_edge_mask_2]) / 2.], dim=-1)
                # edge_h_ligand_2 = torch.cat([
                #     edge_h_2[ligand_edge_mask_2], (edge_h_dist_1[ligand_edge_mask_1] + edge_h_dist_2[ligand_edge_mask_2]) / 2.], dim=-1)
            else:
                edge_h_1 = edge_h_dist_1
                edge_h_2 = edge_h_dist_2

            edge_h_protein_1 = self.edge_embs_protein[i](edge_h_protein_1)
            edge_h_protein_2 = self.edge_embs_protein[i](edge_h_protein_2)
            edge_h_ligand_1 = self.edge_embs_ligand[i](edge_h_ligand_1)
            edge_h_ligand_2 = self.edge_embs_ligand[i](edge_h_ligand_2)
            # assert torch.allclose(edge_h_ligand_1, edge_h_ligand_2, rtol=1e-5, atol=1e-5)
            edge_ligand = (edge_h_ligand_1 + edge_h_ligand_2) / 2.
            # edge_h_protein_1 = self.edge_embs[i](edge_h_protein_1)
            # edge_h_protein_2 = self.edge_embs[i](edge_h_protein_2)
            # edge_h_ligand_1 = self.edge_embs[i](edge_h_ligand_1)
            # edge_h_ligand_2 = self.edge_embs[i](edge_h_ligand_2)

            edge_h_1 = edge_h_1.new_empty(ligand_edge_mask_1.size(0), self.edge_dim)
            edge_h_2 = edge_h_2.new_empty(ligand_edge_mask_2.size(0), self.edge_dim)

            edge_h_1[~ligand_edge_mask_1] = edge_h_protein_1
            edge_h_1[ligand_edge_mask_1]  = edge_ligand

            edge_h_2[~ligand_edge_mask_2] = edge_h_protein_2
            edge_h_2[ligand_edge_mask_2]  = edge_ligand

            # node and edge feature updates
            # node_h_with_edge_1, test_1 = self.node_blocks_with_edge[i].forward_dual(node_h_1, edge_index_1, edge_h_1, node_time_1, ligand_mask_1, ligand_edge_mask_1)
            # node_h_with_edge_2, test_2 = self.node_blocks_with_edge[i].forward_dual(node_h_2, edge_index_2, edge_h_2, node_time_2, ligand_mask_2, ligand_edge_mask_2)
            # assert torch.equal(node_h_1[ligand_mask_1], node_h_2[ligand_mask_2])
            # assert torch.equal(edge_h_1[ligand_edge_mask_1], edge_h_2[ligand_edge_mask_2])
            node_h_with_edge_1, node_h_with_edge_2 = self.node_blocks_with_edge[i].forward(
                node_h_1, edge_index_1, edge_h_1, node_time_1, ligand_mask_1, ligand_edge_mask_1,
                node_h_2, edge_index_2, edge_h_2, node_time_2, ligand_mask_2, ligand_edge_mask_2)
                
            if self.update_edge:
                # edge_h_1 = edge_h_1 + self.edge_blocks[i].forward_dual(edge_h_1, edge_index_1, node_h_1, edge_time_1)
                # edge_h_2 = edge_h_2 + self.edge_blocks[i].forward_dual(edge_h_2, edge_index_2, node_h_2, edge_time_2)
                # assert torch.equal(node_h_1[ligand_mask_1], node_h_2[ligand_mask_2])
                # assert torch.equal(edge_h_1[ligand_edge_mask_1], edge_h_2[ligand_edge_mask_2])
                delta_edge_h_1, delta_edge_h_2 = self.edge_blocks[i].forward(
                    edge_h_1, edge_index_1, node_h_1, edge_time_1, ligand_mask_1, ligand_edge_mask_1, 
                    edge_h_2, edge_index_2, node_h_2, edge_time_2, ligand_mask_2, ligand_edge_mask_2, )
                edge_h_1 = edge_h_1 + delta_edge_h_1
                edge_h_2 = edge_h_2 + delta_edge_h_2
            node_h_1 = node_h_1 + node_h_with_edge_1
            node_h_2 = node_h_2 + node_h_with_edge_2


            # pos updates
            if self.update_pos:
                # assert torch.equal(node_h_1[ligand_mask_1], node_h_2[ligand_mask_2])
                # assert torch.equal(edge_h_1[ligand_edge_mask_1], edge_h_2[ligand_edge_mask_2])
                delta_pos_1 = self.pos_blocks[i](node_h_1, edge_h_1, edge_index_1, relative_vec_1, distance_1, edge_time_1)
                delta_pos_2 = self.pos_blocks[i](node_h_2, edge_h_2, edge_index_2, relative_vec_2, distance_2, edge_time_2)
                node_pos_1 = node_pos_1 + delta_pos_1 * ligand_mask_1[:, None]
                node_pos_2 = node_pos_2 + delta_pos_2 * ligand_mask_2[:, None]

        return node_h_1, node_pos_1, edge_h_1, node_h_2, node_pos_2, edge_h_2
    
    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

    def forward(self, node_h, edge_h, edge_index, relative_vec, distance, edge_time):
        # (node_h, edge_h, edge_index, relative_vec, distance, edge_time)
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(node_h[edge_index_left])
        right_feat = self.right_lin_edge(node_h[edge_index_right])
        weight_edge = self.edge_lin(edge_h, left_feat * right_feat, edge_time)
        
        force_edge = weight_edge * relative_vec / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=node_h.shape[0])

        return delta_pos

class NodeBondNet(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, key_dim, num_heads, num_blocks, k, cutoff, use_atten, use_gate):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.bond_dim = bond_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
        self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        # node network
        self.lin_node = Linear(node_dim, node_dim)
        self.node_blocks_with_edge = ModuleList()
        self.node_blocks_with_bond = ModuleList()
        self.bond_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,  # never use atten for edges message becused too many edges
            ))
            self.node_blocks_with_bond.append(NodeBlock(
                node_dim=node_dim, edge_dim=bond_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,
            ))
            if bond_dim > 0:
                self.bond_blocks.append(BondBlock(
                    bond_dim=bond_dim, node_dim=node_dim, use_gate=use_gate,
                    use_atten=use_atten, key_dim=key_dim, num_heads=num_heads,
                ))

    def forward(self, node_h, node_pos, h_bond, bond_index, batch, is_mol, is_frag, return_edge=False):

        edge_attr, edge_index = self._build_edges(node_pos, batch, is_mol, is_frag)
        for i in range(self.num_blocks):
            # node updates with edges
            node_h_with_edge = self.node_blocks_with_edge[i](node_h, edge_index, edge_attr)
            if self.bond_dim > 0:
                # node updates with bonds
                node_h_with_bond = self.node_blocks_with_bond[i](node_h, bond_index, h_bond)
                # bond updates
                h_bond = h_bond + self.bond_blocks[i](h_bond, bond_index, node_h)
            else:
                node_h_with_bond = 0
            node_h = node_h + self.lin_node(node_h_with_edge + node_h_with_bond)
        if return_edge:
            return {
                'node_h': node_h,
                'h_bond': h_bond,
                'edge_attr': edge_attr,
                'edge_index': edge_index,
            }
        else:
            return {
                'node_h': node_h,
                'h_bond': h_bond,
            }

    @property
    def additional_edge_feat(self):
        return 6

    def _build_edges(self, pos, batch, is_mol, is_frag):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source') 
        # distance
        distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
        edge_attr = self.distance_expansion(distance)

        # 6-vector represent the two node types (atoms of protein or mol or frag)
        edge_src_feat = is_mol[edge_index[0]].long()
        edge_src_feat = edge_src_feat + 2 * is_frag[edge_index[0]].long()
        edge_dst_feat = is_mol[edge_index[1]].long()
        edge_dst_feat = edge_dst_feat + 2 * is_frag[edge_index[1]].long()
        edge_type_feat = torch.cat([
            F.one_hot(edge_src_feat, num_classes=3),
            F.one_hot(edge_dst_feat, num_classes=3),
        ], axis=-1)

        edge_attr = torch.cat([edge_attr, edge_type_feat], axis=-1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr, edge_index

    def _build_bond_atten(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        for node in torch.unique(left_node):
            ind_connect_left = (left_node == node)
            idx_connect_left = torch.nonzero(ind_connect_left)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            ind_connect_right = (right_node == node)
            idx_connect_right = torch.nonzero(ind_connect_right)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right



    def _build_bond_atten2(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        left_node_unique = torch.unique(left_node).cpu().numpy()
        right_node_unique = torch.unique(right_node).cpu().numpy()
        group2node_dict_left = {l:[] for l in left_node_unique}
        group2node_dict_right = {l:[] for l in right_node_unique}
        for i, node in enumerate(left_node.cpu().numpy()):
            group2node_dict_left[node] += [i]
        for i, node in enumerate(right_node.cpu().numpy()):
            group2node_dict_right[node] += [i]
        for node in left_node_unique:
            idx_connect_left = torch.LongTensor(group2node_dict_left[node]).to(bond_index.device)
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            idx_connect_right = torch.LongTensor(group2node_dict_right[node]).to(bond_index.device)
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right

class PosPredictor(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

        self.bond_dim = bond_dim
        if bond_dim > 0:
            self.left_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.right_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.bond_lin = BondFFN(bond_dim, bond_dim, node_dim, use_gate, out_dim=1)

    def forward(self, node_h, node_pos, h_bond, bond_index, edge_h, edge_index, is_frag):
        # 1 pos update through edges
        is_left_frag = is_frag[edge_index[0]]
        edge_index_left, edge_index_right = edge_index[:, is_left_frag]
        
        left_feat = self.left_lin_edge(node_h[edge_index_left])
        right_feat = self.right_lin_edge(node_h[edge_index_right])
        weight_edge = self.edge_lin(edge_h[is_left_frag], left_feat * right_feat)
        force_edge = weight_edge * (node_pos[edge_index_left] - node_pos[edge_index_right])
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=node_h.shape[0])

        # 2 pos update through bonds
        if self.bond_dim > 0:
            is_left_frag = is_frag[bond_index[0]]
            bond_index_left, bond_index_right = bond_index[:, is_left_frag]

            left_feat = self.left_lin_bond(node_h[bond_index_left])
            right_feat = self.right_lin_bond(node_h[bond_index_right])
            weight_bond = self.bond_lin(h_bond[is_left_frag], left_feat * right_feat)
            force_bond = weight_bond * (node_pos[bond_index_left] - node_pos[bond_index_right])
            delta_pos = delta_pos + scatter_sum(force_bond, bond_index_left, dim=0, dim_size=node_h.shape[0])
        
        pos_update = node_pos + delta_pos / 10.
        return pos_update #TODO: use only frag pos instead of all pos to save memory


class EgnnNet_BondPredictor(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))

    def forward(self, node_h, node_pos, edge_h, edge_index, node_time, edge_time, ligand_mask):
        for i in range(self.num_blocks):
            # edge fetures before each block
            if self.update_pos or (i==0):
                edge_h_dist, relative_vec, distance = self._build_edges_dist(node_pos, edge_index)
            if self.update_edge:
                edge_h = torch.cat([edge_h, edge_h_dist], dim=-1)
            else:
                edge_h = edge_h_dist
            edge_h = self.edge_embs[i](edge_h)
                
            # node and edge feature updates
            node_h_with_edge = self.node_blocks_with_edge[i].forward_single(node_h, edge_index, edge_h, node_time)
            if self.update_edge:
                edge_h = edge_h + self.edge_blocks[i].forward_single(edge_h, edge_index, node_h, edge_time)
            node_h = node_h + node_h_with_edge
            # pos updates
            if self.update_pos:
                delta_pos = self.pos_blocks[i](node_h, edge_h, edge_index, relative_vec, distance, edge_time)
                node_pos = node_pos + delta_pos * ligand_mask[:, None]
        return node_h, node_pos, edge_h

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance