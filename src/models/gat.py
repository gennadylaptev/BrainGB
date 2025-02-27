import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn import functional as F
from typing import Union, Tuple, Optional
from torch_geometric.typing import Size, OptTensor
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing, GATConv
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch import nn
import torch_geometric


# custom GAT layer
class MPGATConv(GATConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        gat_mp_type: str = "attention_weighted",
        abs_weights=True,
    ):
        super().__init__(in_channels, out_channels, heads)

        # gat_mp_type choices:
        # attention_weighted, attention_edge_weighted, sum_attention_edge, edge_node_concate, node_concate
        self.gat_mp_type = gat_mp_type
        input_dim = out_channels

        self.abs_weights = abs_weights

        if gat_mp_type == "edge_node_concate":
            input_dim = out_channels * 2 + 1
        elif gat_mp_type == "node_concate":
            input_dim = out_channels * 2

        # edge_lin is mandatory
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

    # define our message passage function
    def message(
        self,
        x_i, x_j,  # node embeddgins
        alpha_j, alpha_i,  # attention weights per node
        edge_attr,
        index,
        ptr,
        size_i,
    ):
        # x_j: [num edges, num heads, num channels (out dim)]

        # copied from PyG
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # add extra dim, so we get [m, 1, 1] shape where m = num of edges
        attention_score = alpha.unsqueeze(-1)

        # apply `abs` to edge_attr (weights)
        edge_weights = edge_attr.view(-1, 1).unsqueeze(-1)
        if self.abs_weights:
            edge_weights = torch.abs(edge_weights)

        # transform attention weights
        if self.gat_mp_type == "attention_weighted":
            # (1) att: s^(l+1) = s^l * alpha
            msg = x_j * attention_score
            return msg
        elif self.gat_mp_type == "attention_edge_weighted":
            # (2) e-att: s^(l+1) = s^l * alpha * e
            msg = x_j * attention_score * edge_weights
            return msg
        elif self.gat_mp_type == "sum_attention_edge":
            # (3) m-att-1: s^(l+1) = s^l * (alpha + e),
            # this one may not make sense cause it doesn't used attention score to control all
            msg = x_j * (attention_score + edge_weights)
            return msg
        elif self.gat_mp_type == "edge_node_concate":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat(
                [
                    x_i,
                    x_j * attention_score,
                    # reshape and expand to the given num of heads
                    # Note that we do not absolute values of weights here!
                    edge_weights.expand(-1, self.heads, -1),
                ],
                dim=-1,
            )
            msg = self.edge_lin(msg)
            return msg
        elif self.gat_mp_type == "node_concate":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat([x_i, x_j * attention_score], dim=-1)
            msg = self.edge_lin(msg)
            return msg
        # elif self.gat_mp_type == "sum_node_edge_weighted":
        #     # (5) m-att-3: s^(l+1) = (s^l + e) * alpha
        #     node_emb_dim = x_j.shape[-1]
        #     extended_edge = torch.cat([edge_weights] * node_emb_dim, dim=-1)
        #     sum_node_edge = x_j + extended_edge
        #     msg = sum_node_edge * attention_score
        #     return msg
        else:
            raise ValueError(f'Invalid message passing variant {self.gat_mp_type}')


class GAT(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        args,
        num_nodes,
        num_classes,
    ):
        super().__init__()

        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes

        # pass arguments from ArgumentsParser
        self.pooling = args.pooling
        hidden_dim = args.hidden_dim
        num_heads = args.num_heads
        num_layers = args.n_GNN_layers
        edge_emb_dim = args.edge_emb_dim
        gat_mp_type = args.gat_mp_type
        dropout = args.dropout

        gat_input_dim = input_dim

        # pack a bunch of convs into a list
        for i in range(num_layers - 1):
            conv = torch_geometric.nn.Sequential(
                'x, edge_index, edge_attr',
                [
                    (MPGATConv(
                        gat_input_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        gat_mp_type=gat_mp_type
                    ),'x, edge_index, edge_attr -> x'),
                    nn.Linear(hidden_dim * num_heads, hidden_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            # update current input_dim
            gat_input_dim = hidden_dim
            self.convs.append(conv)

        input_dim = 0

        # last conv is different for each type of pooling
        if self.pooling == "concat":
            node_dim = 8
            conv = torch_geometric.nn.Sequential(
                'x, edge_index, edge_attr',
                [
                    (
                        MPGATConv(
                            hidden_dim,
                            hidden_dim,
                            heads=num_heads,
                            dropout=dropout,
                            gat_mp_type=gat_mp_type,
                        ),
                        'x, edge_index, edge_attr -> x'
                    ),
                    nn.Linear(hidden_dim * num_heads, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, node_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(node_dim),
                ]
            )
            input_dim = node_dim * num_nodes

        elif self.pooling == 'sum' or self.pooling == 'mean':
            node_dim = 256
            input_dim = node_dim
            conv = torch_geometric.nn.Sequential(
                'x, edge_index, edge_attr',
                [
                    (
                        MPGATConv(
                            hidden_dim,
                            hidden_dim,
                            heads=num_heads,
                            dropout=dropout,
                            gat_mp_type=gat_mp_type,
                        ),
                        'x, edge_index, edge_attr -> x',
                    ),
                    nn.Linear(hidden_dim * num_heads, hidden_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(node_dim),
                ]
            )
        self.convs.append(conv)

        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        z = x

        # use abs edge weights! (however they made abs in MPGATConv.forward)
        edge_attr = torch.abs(edge_attr)

        # apply conv layers
        for i, conv in enumerate(self.convs):
            # bz * nodes, hidden
            z = conv(z, edge_index, edge_attr)

        # pooling
        if self.pooling == "concat":
            z = z.reshape((z.shape[0] // self.num_nodes, -1))
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)
        return out
