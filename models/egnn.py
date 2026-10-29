import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
from torch_scatter import scatter_sum
from models.common import GaussianSmearing, MLP


def split_first_linear(mlp, dims):
    """Column blocks of an MLP's first Linear, matching a cat() of its inputs.

    An MLP fed cat([a, b, c], -1) computes W @ cat + bias in its first layer, and
    that equals W_a @ a + W_b @ b + W_c @ c + bias. Handing back the blocks lets a
    caller evaluate whichever term is constant per node once per node and gather
    the result, instead of gathering the node features and evaluating once per
    edge -- E/N is around 30 here. The returned tail (norm, activation, remaining
    layers) is row-wise, so it still runs per edge unchanged.

    Splitting the sum changes its grouping, so results agree to within rounding
    rather than bit-for-bit; see tests/README.md.
    """
    first = mlp.net[0]
    blocks, offset = [], 0
    for d in dims:
        blocks.append(first.weight[:, offset:offset + d])
        offset += d
    assert offset == first.weight.size(1), \
        f'dims {tuple(dims)} sum to {offset}, but the first Linear takes ' \
        f'{first.weight.size(1)} input columns'
    return blocks, first.bias, mlp.net[1:]


class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        self.edge_dim = edge_dim  # needed to split the gate's first layer

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
        x_1, edge_index_1, edge_attr_1, node_time_1, lig_idx_1,
        x_2, edge_index_2, edge_attr_2, node_time_2, lig_idx_2,
        n_lig_edges,
        ):
        """
        lig_idx_k:    (n_lig,) int64 rows of the ligand atoms in branch k.
        n_lig_edges:  ligand edges occupy rows [:n_lig_edges] in both branches,
                      see EgnnNet.forward for why this layout holds.
        """
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
            # The gate sees cat([edge_attr, x[col], time[col]]); its node term is the
            # same for every edge leaving a node, so project per node and gather.
            (w_edge, w_node, w_time), bias, tail = split_first_linear(
                self.gate, (self.edge_dim, self.node_dim, 1))
            gate_node_1 = F.linear(x_1, w_node)
            gate_node_2 = F.linear(x_2, w_node)
            # time can arrive in a narrower dtype than the weights; the cat this
            # replaces promoted it implicitly, so do it explicitly here
            t_1 = node_time_1[col_1].to(w_time.dtype)
            t_2 = node_time_2[col_2].to(w_time.dtype)
            gate_1 = tail(F.linear(edge_attr_1, w_edge, bias) + gate_node_1[col_1]
                          + F.linear(t_1, w_time))
            gate_2 = tail(F.linear(edge_attr_2, w_edge, bias) + gate_node_2[col_2]
                          + F.linear(t_2, w_time))
            msg_j_1 = msg_j_1 * torch.sigmoid(gate_1)
            msg_j_2 = msg_j_2 * torch.sigmoid(gate_2)

        # assert torch.allclose(msg_j_1[:n_lig_edges], msg_j_2[:n_lig_edges], rtol=1e-5, atol=1e-5)
        msg_j_1[:n_lig_edges] = msg_j_2[:n_lig_edges] = (msg_j_1[:n_lig_edges] + msg_j_2[:n_lig_edges]) / 2.

        # Aggregate messages
        aggr_msg_1 = scatter_sum(msg_j_1, row_1, dim=0, dim_size=N_1)
        aggr_msg_2 = scatter_sum(msg_j_2, row_2, dim=0, dim_size=N_2)

        # aggr_msg_ligand_1 = scatter_sum(msg_j_1[:n_lig_edges], row_1[:n_lig_edges], dim=0, dim_size=ligand_N_1)
        # aggr_msg_ligand_2 = scatter_sum(msg_j_2[:n_lig_edges], row_2[:n_lig_edges], dim=0, dim_size=ligand_N_2)
        aggr_msg_protein_1 = scatter_sum(msg_j_1[n_lig_edges:], row_1[n_lig_edges:], dim=0, dim_size=N_1)
        aggr_msg_protein_2 = scatter_sum(msg_j_2[n_lig_edges:], row_2[n_lig_edges:], dim=0, dim_size=N_2)

        # aggr_msg_1_new = aggr_msg_protein_1.clone()
        # aggr_msg_1_new[lig_idx_1] = aggr_msg_1_new[lig_idx_1] + aggr_msg_ligand_1 + aggr_msg_protein_2[lig_idx_2]
        # aggr_msg_2_new = aggr_msg_protein_2.clone()
        # aggr_msg_2_new[lig_idx_2] = aggr_msg_2_new[lig_idx_2] + aggr_msg_ligand_2 + aggr_msg_protein_1[lig_idx_1]

        aggr_msg_1[lig_idx_1] = aggr_msg_1[lig_idx_1] + aggr_msg_protein_2[lig_idx_2]
        aggr_msg_2[lig_idx_2] = aggr_msg_2[lig_idx_2] + aggr_msg_protein_1[lig_idx_1]

        # assert torch.allclose(aggr_msg_1[lig_idx_1], aggr_msg_2[lig_idx_2], rtol=1e-5, atol=1e-5)
        aggr_msg_1[lig_idx_1] = aggr_msg_2[lig_idx_2] = (aggr_msg_1[lig_idx_1] + aggr_msg_2[lig_idx_2]) / 2.

        out_1 = self.centroid_lin(x_1) + aggr_msg_1
        out_2 = self.centroid_lin(x_2) + aggr_msg_2

        out_1 = self.layer_norm(out_1)
        out_2 = self.layer_norm(out_2)
        out_1 = self.out_transform(self.act(out_1))
        out_2 = self.out_transform(self.act(out_2))

        # assert torch.allclose(out_1[lig_idx_1], out_2[lig_idx_2], rtol=1e-5, atol=1e-5)
        out_1[lig_idx_1] = out_2[lig_idx_2] = (out_1[lig_idx_1] + out_2[lig_idx_2]) / 2.

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


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_dim = bond_dim
        self.node_dim = node_dim
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

    def node_side(self, node_h):
        """Per-node half of forward(), for the caller to gather per edge.

        forward() touches the node features in exactly two places, node_linear and
        the node columns of the gate's first layer, and both are linear in them, so
        the work runs once per node instead of once per edge. The projections are
        also narrower than node_dim, so the gather that follows moves less memory
        than gathering the raw features would.

        Use with forward_projected(). forward() itself is unchanged, for callers
        whose node argument is not a plain gather of node features -- PosUpdate
        passes a per-edge product, which cannot be hoisted.
        """
        if not self.use_gate:
            return self.node_linear(node_h), None
        (_, w_node, _), _, _ = split_first_linear(
            self.gate, (self.bond_dim, self.node_dim, 1))
        return self.node_linear(node_h), F.linear(node_h, w_node)

    def forward_projected(self, bond_feat_input, node_proj, node_gate, time):
        """forward() with the node-side terms supplied by node_side()."""
        inter_feat = self.bond_linear(bond_feat_input) * node_proj
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            (w_bond, _, w_time), bias, tail = split_first_linear(
                self.gate, (self.bond_dim, self.node_dim, 1))
            # see the matching cast in NodeBlock.forward
            gate = tail(F.linear(bond_feat_input, w_bond, bias) + node_gate
                        + F.linear(time.to(w_time.dtype), w_time))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


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
        bond_h_1, bond_index_1, node_h_1, bond_time_1,
        bond_h_2, bond_index_2, node_h_2, bond_time_2,
        n_lig_edges,
        ):
        # (edge_h, edge_index, node_h, edge_time)
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        n_lig_edges: ligand edges occupy rows [:n_lig_edges] in both branches,
                     see EgnnNet.forward for why this layout holds.
        """
        N_1 = node_h_1.size(0)
        N_2 = node_h_2.size(0)
        left_node_1, right_node_1 = bond_index_1
        left_node_2, right_node_2 = bond_index_2

        # Node-side projections for both BondFFNs, once per node rather than once per
        # edge; the per-edge work then gathers these. See BondFFN.node_side.
        lproj_1, lgate_1 = self.bond_ffn_left.node_side(node_h_1)
        lproj_2, lgate_2 = self.bond_ffn_left.node_side(node_h_2)
        rproj_1, rgate_1 = self.bond_ffn_right.node_side(node_h_1)
        rproj_2, rgate_2 = self.bond_ffn_right.node_side(node_h_2)

        def gathered(t, idx):
            return None if t is None else t[idx]

        msg_bond_left_1_all = self.bond_ffn_left.forward_projected(
            bond_h_1,
            lproj_1[left_node_1],
            gathered(lgate_1, left_node_1),
            bond_time_1,
        )                               # (b, hidden)

        msg_bond_left_1_agg = scatter_sum(
            msg_bond_left_1_all,
            right_node_1,
            dim=0,
            dim_size=N_1,
        )                               # (N_1, hidden)

        msg_bond_left_1 = msg_bond_left_1_agg[left_node_1]  # (b, hidden)


        msg_bond_left_2_all = self.bond_ffn_left.forward_projected(
            bond_h_2,
            lproj_2[left_node_2],
            gathered(lgate_2, left_node_2),
            bond_time_2,
        )                               # (b, hidden)

        msg_bond_left_2_agg = scatter_sum(
            msg_bond_left_2_all,
            right_node_2,
            dim=0,
            dim_size=N_2,
        )                               # (N_1, hidden)

        msg_bond_left_2 = msg_bond_left_2_agg[left_node_2]  # (b, hidden)


        # The non-ligand messages used to be a second bond_ffn_left() call over the
        # non-ligand rows. BondFFN is row-wise -- two Linears, an MLP whose LayerNorm
        # normalizes the feature axis, and an elementwise gate -- so that call returned
        # exactly these rows of the full result, and slicing replaces it. Only the leading
        # n_lig_edges rows of the gather are ever read, so gather just those.
        msg_non_lig_agg_1 = scatter_sum(
            msg_bond_left_1_all[n_lig_edges:],
            right_node_1[n_lig_edges:],
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_left_1_non_ligand = msg_non_lig_agg_1[left_node_1[:n_lig_edges]]  # (n_lig, hidden)


        msg_non_lig_agg_2 = scatter_sum(
            msg_bond_left_2_all[n_lig_edges:],
            right_node_2[n_lig_edges:],
            dim=0,
            dim_size=N_2,
        )  # (N_2, hidden)

        msg_bond_left_2_non_ligand = msg_non_lig_agg_2[left_node_2[:n_lig_edges]]  # (n_lig, hidden)


        msg_bond_left_1[:n_lig_edges] = msg_bond_left_1[:n_lig_edges] + msg_bond_left_2_non_ligand
        msg_bond_left_2[:n_lig_edges] = msg_bond_left_2[:n_lig_edges] + msg_bond_left_1_non_ligand
        # assert torch.allclose(msg_bond_left_1[:n_lig_edges], msg_bond_left_2[:n_lig_edges], rtol=1e-5, atol=1e-5)
        msg_bond_left_1[:n_lig_edges] = msg_bond_left_2[:n_lig_edges] = (msg_bond_left_1[:n_lig_edges] + msg_bond_left_2[:n_lig_edges]) / 2.
        # msg_bond_left_2[:n_lig_edges] = msg_bond_left_1[:n_lig_edges]

        msg_bond_right_1_all = self.bond_ffn_right.forward_projected(
            bond_h_1,
            rproj_1[right_node_1],
            gathered(rgate_1, right_node_1),
            bond_time_1,
        )  # (b1, hidden)

        msg_bond_right_1_agg = scatter_sum(
            msg_bond_right_1_all,
            left_node_1,
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_right_1 = msg_bond_right_1_agg[right_node_1]  # (b1, hidden)


        msg_bond_right_2_all = self.bond_ffn_right.forward_projected(
            bond_h_2,
            rproj_2[right_node_2],
            gathered(rgate_2, right_node_2),
            bond_time_2,
        )  # (b2, hidden)

        msg_bond_right_2_agg = scatter_sum(
            msg_bond_right_2_all,
            left_node_2,
            dim=0,
            dim_size=N_2,
        )  # (N_2, hidden)

        msg_bond_right_2 = msg_bond_right_2_agg[right_node_2]  # (b2, hidden)


        # same slicing as the left half above
        msg_non_lig_right_agg_1 = scatter_sum(
            msg_bond_right_1_all[n_lig_edges:],
            left_node_1[n_lig_edges:],
            dim=0,
            dim_size=N_1,
        )  # (N_1, hidden)

        msg_bond_right_1_non_ligand = msg_non_lig_right_agg_1[right_node_1[:n_lig_edges]]  # (n_lig, hidden)


        msg_non_lig_right_agg_2 = scatter_sum(
            msg_bond_right_2_all[n_lig_edges:],
            left_node_2[n_lig_edges:],
            dim=0,
            dim_size=N_2,
        )  # (N_2, hidden)

        msg_bond_right_2_non_ligand = msg_non_lig_right_agg_2[right_node_2[:n_lig_edges]]  # (n_lig, hidden)


        msg_bond_right_1[:n_lig_edges] = msg_bond_right_1[:n_lig_edges] + msg_bond_right_2_non_ligand
        msg_bond_right_2[:n_lig_edges] = msg_bond_right_2[:n_lig_edges] + msg_bond_right_1_non_ligand
        # assert torch.allclose(msg_bond_right_1[:n_lig_edges], msg_bond_right_2[:n_lig_edges], rtol=1e-5, atol=1e-5)
        msg_bond_right_1[:n_lig_edges] = msg_bond_right_2[:n_lig_edges] = (msg_bond_right_1[:n_lig_edges] + msg_bond_right_2[:n_lig_edges]) / 2.
        # msg_bond_right_2[:n_lig_edges] = msg_bond_right_1[:n_lig_edges]


        # node_ffn_* are plain Linears over gathered node features, so projecting per
        # node and gathering afterwards is the same value for ~1/30 of the matmul --
        # and the gather moves edge_dim columns instead of node_dim.
        nffn_left_1, nffn_right_1 = self.node_ffn_left(node_h_1), self.node_ffn_right(node_h_1)
        nffn_left_2, nffn_right_2 = self.node_ffn_left(node_h_2), self.node_ffn_right(node_h_2)

        bond_h_1 = (
            msg_bond_left_1 + msg_bond_right_1
            + nffn_left_1[left_node_1]
            + nffn_right_1[right_node_1]
            + self.self_ffn(bond_h_1)
        )
        bond_h_2 = (
            msg_bond_left_2 + msg_bond_right_2
            + nffn_left_2[left_node_2]
            + nffn_right_2[right_node_2]
            + self.self_ffn(bond_h_2)
        )

        # assert torch.allclose(bond_h_1[:n_lig_edges], bond_h_2[:n_lig_edges], rtol=1e-4, atol=1e-4)
        # bond_h_1[:n_lig_edges] = bond_h_2[:n_lig_edges] = (bond_h_1[:n_lig_edges] + bond_h_2[:n_lig_edges]) / 2.


        bond_h_1 = self.layer_norm(bond_h_1)
        bond_h_2 = self.layer_norm(bond_h_2)

        bond_h_1 = self.out_transform(self.act(bond_h_1))
        bond_h_2 = self.out_transform(self.act(bond_h_2))

        # assert torch.allclose(bond_h_1[:n_lig_edges], bond_h_2[:n_lig_edges], rtol=1e-6, atol=1e-6)
        bond_h_1[:n_lig_edges] = bond_h_2[:n_lig_edges] = (bond_h_1[:n_lig_edges] + bond_h_2[:n_lig_edges]) / 2.

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
        # plain attribute (not a buffer) so it stays out of state_dict
        self._layout_checked = False

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

        # edge_compose() (models/diffusion.py) concatenates the ligand edges ahead of the
        # protein ones, so both ligand_edge_masks are True on a leading contiguous block and
        # every boolean index over edges reduces to a slice. Resolving the boundary and the
        # ligand node rows once here keeps the per-layer nonzero() calls -- and the implicit
        # device syncs they carry -- out of the block loop.
        n_lig_edges = int(ligand_edge_mask_1.sum())
        lig_idx_1 = ligand_mask_1.nonzero(as_tuple=True)[0]
        lig_idx_2 = ligand_mask_2.nonzero(as_tuple=True)[0]
        # The per-layer edge buffer used to come from new_empty(), which pinned it to the
        # caller's dtype; torch.cat would instead inherit it from the embedding Linears.
        # Pin it explicitly so the residual stream keeps its precision if autocast is ever
        # switched to fp16/bf16. No-op while everything is fp32.
        edge_dtype = edge_h_1.dtype

        if self.update_pos:
            # PosUpdate scatters each edge's force into edge_index[0] and the result is
            # masked down to ligand rows, so an edge whose source is a pocket atom can only
            # ever land in a row that gets discarded. Restrict the block to ligand-source
            # edges: a ligand row still receives from exactly the same edges, because every
            # edge reaching it has a ligand source by construction. The selection depends
            # only on edge_index, which is fixed for all num_blocks layers, so it is
            # resolved once here rather than per layer.
            pos_edge_1 = ligand_mask_1[edge_index_1[0]].nonzero(as_tuple=True)[0]
            pos_edge_2 = ligand_mask_2[edge_index_2[0]].nonzero(as_tuple=True)[0]
            pos_index_1, pos_index_2 = edge_index_1[:, pos_edge_1], edge_index_2[:, pos_edge_2]
            pos_time_1, pos_time_2 = edge_time_1[pos_edge_1], edge_time_2[pos_edge_2]
        if not self._layout_checked:
            self._check_layout(ligand_edge_mask_1, ligand_edge_mask_2, n_lig_edges,
                               lig_idx_1, lig_idx_2)

        for i in range(self.num_blocks):

            # assert torch.equal(node_h_1[lig_idx_1], node_h_2[lig_idx_2])
            # assert torch.equal(edge_h_1[:n_lig_edges], edge_h_2[:n_lig_edges])

            # edge fetures before each block
            if self.update_pos or (i==0):
                edge_h_dist_1, relative_vec_1, distance_1 = self._build_edges_dist(node_pos_1, edge_index_1)
                edge_h_dist_2, relative_vec_2, distance_2 = self._build_edges_dist(node_pos_2, edge_index_2)

            if self.update_edge:
                # edge_h_1 = torch.cat([edge_h_1, edge_h_dist_1], dim=-1)
                # edge_h_2 = torch.cat([edge_h_2, edge_h_dist_2], dim=-1)

                edge_h_protein_1 = torch.cat([
                    edge_h_1[n_lig_edges:], edge_h_dist_1[n_lig_edges:]], dim=-1)
                edge_h_protein_2 = torch.cat([
                    edge_h_2[n_lig_edges:], edge_h_dist_2[n_lig_edges:]], dim=-1)

                edge_h_ligand_1 = torch.cat([
                    edge_h_1[:n_lig_edges], edge_h_dist_1[:n_lig_edges], edge_h_dist_2[:n_lig_edges]], dim=-1)
                edge_h_ligand_2 = torch.cat([
                    edge_h_2[:n_lig_edges], edge_h_dist_1[:n_lig_edges], edge_h_dist_2[:n_lig_edges]], dim=-1)


                # sum_feat = edge_h_dist_1[:n_lig_edges] + edge_h_dist_2[:n_lig_edges]
                # diff_feat = (edge_h_dist_1[:n_lig_edges] - edge_h_dist_2[:n_lig_edges]).abs()
                # edge_h_ligand_1 = torch.cat([
                #     edge_h_1[:n_lig_edges], sum_feat, diff_feat], dim=-1)
                # edge_h_ligand_2 = torch.cat([
                #     edge_h_2[:n_lig_edges], sum_feat, diff_feat], dim=-1)

                # edge_h_ligand_1 = torch.cat([
                #     edge_h_1[:n_lig_edges], (edge_h_dist_1[:n_lig_edges] + edge_h_dist_2[:n_lig_edges]) / 2.], dim=-1)
                # edge_h_ligand_2 = torch.cat([
                #     edge_h_2[:n_lig_edges], (edge_h_dist_1[:n_lig_edges] + edge_h_dist_2[:n_lig_edges]) / 2.], dim=-1)
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

            # ligand edges first, protein edges after -- the layout asserted above
            edge_h_1 = torch.cat([edge_ligand, edge_h_protein_1], dim=0).to(edge_dtype)
            edge_h_2 = torch.cat([edge_ligand, edge_h_protein_2], dim=0).to(edge_dtype)

            # node and edge feature updates
            # node_h_with_edge_1, test_1 = self.node_blocks_with_edge[i].forward_dual(node_h_1, edge_index_1, edge_h_1, node_time_1, ligand_mask_1, ligand_edge_mask_1)
            # node_h_with_edge_2, test_2 = self.node_blocks_with_edge[i].forward_dual(node_h_2, edge_index_2, edge_h_2, node_time_2, ligand_mask_2, ligand_edge_mask_2)
            # assert torch.equal(node_h_1[lig_idx_1], node_h_2[lig_idx_2])
            # assert torch.equal(edge_h_1[:n_lig_edges], edge_h_2[:n_lig_edges])
            node_h_with_edge_1, node_h_with_edge_2 = self.node_blocks_with_edge[i].forward(
                node_h_1, edge_index_1, edge_h_1, node_time_1, lig_idx_1,
                node_h_2, edge_index_2, edge_h_2, node_time_2, lig_idx_2,
                n_lig_edges)

            if self.update_edge:
                # edge_h_1 = edge_h_1 + self.edge_blocks[i].forward_dual(edge_h_1, edge_index_1, node_h_1, edge_time_1)
                # edge_h_2 = edge_h_2 + self.edge_blocks[i].forward_dual(edge_h_2, edge_index_2, node_h_2, edge_time_2)
                # assert torch.equal(node_h_1[lig_idx_1], node_h_2[lig_idx_2])
                # assert torch.equal(edge_h_1[:n_lig_edges], edge_h_2[:n_lig_edges])
                delta_edge_h_1, delta_edge_h_2 = self.edge_blocks[i].forward(
                    edge_h_1, edge_index_1, node_h_1, edge_time_1,
                    edge_h_2, edge_index_2, node_h_2, edge_time_2,
                    n_lig_edges)
                edge_h_1 = edge_h_1 + delta_edge_h_1
                edge_h_2 = edge_h_2 + delta_edge_h_2
            node_h_1 = node_h_1 + node_h_with_edge_1
            node_h_2 = node_h_2 + node_h_with_edge_2


            # pos updates
            if self.update_pos:
                # assert torch.equal(node_h_1[lig_idx_1], node_h_2[lig_idx_2])
                # assert torch.equal(edge_h_1[:n_lig_edges], edge_h_2[:n_lig_edges])
                delta_pos_1 = self.pos_blocks[i](
                    node_h_1, edge_h_1[pos_edge_1], pos_index_1,
                    relative_vec_1[pos_edge_1], distance_1[pos_edge_1], pos_time_1)
                delta_pos_2 = self.pos_blocks[i](
                    node_h_2, edge_h_2[pos_edge_2], pos_index_2,
                    relative_vec_2[pos_edge_2], distance_2[pos_edge_2], pos_time_2)
                # delta_pos is already zero outside the ligand rows; the mask is kept so
                # that stays a stated property of the update rather than an implicit one.
                node_pos_1 = node_pos_1 + delta_pos_1 * ligand_mask_1[:, None]
                node_pos_2 = node_pos_2 + delta_pos_2 * ligand_mask_2[:, None]

        return node_h_1, node_pos_1, edge_h_1, node_h_2, node_pos_2, edge_h_2

    def _check_layout(self, ligand_edge_mask_1, ligand_edge_mask_2, n_lig_edges,
                      lig_idx_1, lig_idx_2):
        """Validate the input layout forward() slices against, once per process.

        Both the edge slicing and the row-wise pairing of the two branches are
        caller-side invariants: a violation would silently produce wrong numbers
        instead of raising, so it is worth one check. The layout is fixed by the
        compose helpers rather than by the data, hence checking the first batch
        is enough and keeps the syncs out of the training loop.
        """
        for mask in (ligand_edge_mask_1, ligand_edge_mask_2):
            assert bool(mask[:n_lig_edges].all()) and not bool(mask[n_lig_edges:].any()), \
                'ligand edges must form a leading contiguous block; see edge_compose()'
        assert lig_idx_1.numel() == lig_idx_2.numel(), \
            'the two branches must share the same ligand atoms'
        self._layout_checked = True

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

        # clamp to avoid division by zero for coincident atoms
        dist = distance.unsqueeze(-1).clamp_min(1e-6)
        force_edge = weight_edge * relative_vec / dist / (dist + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=node_h.shape[0])

        return delta_pos

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