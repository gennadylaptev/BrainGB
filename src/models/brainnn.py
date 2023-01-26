import torch
from collections import defaultdict
import numpy as np
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F


# end-to-end classifier
# initialized in examples/build_model.py
class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn

        # these are not used :)
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data):
        # unpack DataBatch instance
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # run GNN
        g = self.gnn(x, edge_index, edge_attr, batch)

        # and just logsoftmax
        log_logits = F.log_softmax(g, dim=-1)

        return log_logits
