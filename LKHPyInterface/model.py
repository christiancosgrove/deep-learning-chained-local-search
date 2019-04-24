
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
        # self.node_mlp_1 = Seq(Lin(in_channels + edge_attrs, mid_channels), ReLU(), nn.BatchNorm1d(mid_channels), Lin(mid_channels, mid_channels))
        self.node_mlp_2 = Seq(Lin(in_channels*2, mid_channels), ReLU(), nn.BatchNorm1d(mid_channels), Lin(mid_channels, out_channels))
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

            # print('x ', x.size())
            # print('edge ', edge_attr.size())
            e = edge_attr.expand(edge_attr.size(0), x.size(1))


            # ones = torch.ones(e.size())
            # mask = torch.cat([ones, e], dim=1)

            # print('e ', e.size())
            # print('mask ', mask.size())
            # print('out ', out.size())
            # print('x[col] ', x[col].size())


            # out = torch.cat([x[col], edge_attr], dim=1)
            out = torch.cat([x[col], x[col] * e], dim=1)
            # print('oshape ', out.size())
            # out = self.node_mlp_1(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            # print('oshape ', out.size())

            # out = torch.cat([out, u[batch]], dim=1)
            # print('out ', out.size())

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
        self.conv1 = MyLayer(3, 1, 8, 64)

        self.convs = nn.ModuleList()

        for i in range(4):
            self.convs.append(MyLayer(64 + 3 + 64, 1, 64, 64))

        self.conv_last = MyLayer(64, 1, 16, 1)


    def forward(self, data):
        mb_size = 32

        out = self.conv1(data.x, data.edge_index, data.edge_attr, data.batch)

        for l in self.convs:
            # print(out.size())
            glob = torch.mean(out.reshape(mb_size, -1, 64), dim=1).unsqueeze(1).expand(mb_size, 500, 64).reshape(-1, 64)
            # print(glob.size())
            # print(data.batch)
            # print(data.x.size())
            # print(out.size())
            # print(glob.size())
            out = torch.cat([data.x, out, glob], dim=1)
            # print(out.size())
            out = l(out, data.edge_index, data.edge_attr, data.batch)
        
        out = self.conv_last(out, data.edge_index, data.edge_attr, data.batch)
        # print('osize ', out.size())
        out = torch.mean(out.reshape(mb_size, -1), dim=1)

        return torch.squeeze(out)
