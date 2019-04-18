
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (GraphConv, NNConv, global_mean_pool, MessagePassing, RGCNConv)
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

class MyLayer(torch.nn.Module):
    def __init__(self, in_channels, edge_attrs, mid_channels, out_channels):
        super(MyLayer, self).__init__()

        # self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_1 = Seq(Lin(in_channels + edge_attrs, mid_channels), ReLU(), Lin(mid_channels, mid_channels))
        self.node_mlp_2 = Seq(Lin(mid_channels, mid_channels), ReLU(), Lin(mid_channels, out_channels))
        # self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

        def edge_model(src, dest, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            # out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            return edge_attr

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, col = edge_index

            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            # out = torch.cat([out, u[batch]], dim=1)
            return self.node_mlp_2(out)

        # def global_model(x, edge_index, edge_attr, u, batch):
        #     # x: [N, F_x], where N is the number of nodes.
        #     # edge_index: [2, E] with max entry N - 1.
        #     # edge_attr: [E, F_e]
        #     # u: [B, F_u]
        #     # batch: [N] with max entry B - 1.
        #     out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        #     return self.global_mlp(out)
        self.op = MetaLayer(edge_model, node_model, None)


    def forward(self, x, edge_index, edge_attr, batch):

        return self.op(x, edge_index, edge_attr, None, batch)[0]
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        # self.nn = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
        self.channels = 16
        self.conv1 = MyLayer(2, 1, 16, 16)
        self.conv2 = MyLayer(16, 1, 16, 16)
        self.conv3 = MyLayer(16, 1, 8, 1)

    def forward(self, data):
        out = self.conv1(data.x, data.edge_index, data.edge_attr, data.batch)
        out = self.conv2(out, data.edge_index, data.edge_attr, data.batch)
        out = self.conv3(out, data.edge_index, data.edge_attr, data.batch)
        return torch.squeeze(out)
