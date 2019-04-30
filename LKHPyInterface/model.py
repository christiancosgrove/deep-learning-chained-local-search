
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (GraphConv, NNConv, global_mean_pool, MessagePassing, RGCNConv, GATConv)
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.nn import MetaLayer

class MyLayer(torch.nn.Module):
    def __init__(self, in_channels, edge_attrs, out_channels, global_features):
        super(MyLayer, self).__init__()

        # self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_1 = Seq(
            (Lin(in_channels+edge_attrs, out_channels)))#, nn.LeakyReLU()) #, nn.BatchNorm1d(out_channels), nn.LeakyReLU())
            # Lin(out_channels, out_channels), nn.LeakyReLU(), nn.BatchNorm1d(out_channels))
        self.node_mlp_2 = Seq(
            (Lin(out_channels + global_features, out_channels)))#, nn.LeakyReLU())#, nn.BatchNorm1d(out_channels), nn.LeakyReLU()
            # Lin(out_channels, out_channels), nn.LeakyReLU(), nn.BatchNorm1d(out_channels)
            # )
        self.global_mlp = Seq(
            (Lin(global_features + out_channels, global_features)))#, nn.LeakyReLU())#, nn.BatchNorm1d(global_features), nn.LeakyReLU()
            # Lin(global_features, global_features), nn.LeakyReLU(), nn.BatchNorm1d(global_features)
            # )

        # for m in self.node_mlp_1:
        #     if type(m) == nn.Linear:
        #         nn.init.kaiming_uniform_(m.weight)

        # for m in self.node_mlp_2:
        #     if type(m) == nn.Linear:
        #         nn.init.kaiming_uniform_(m.weight)
        # for m in self.global_mlp:
        #     if type(m) == nn.Linear:
        #         nn.init.kaiming_uniform_(m.weight)

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

            # print('eattr ',edge_attr.size())
            # print('oshape ', out.size())
            out = self.node_mlp_1(out)
            # out *= edge_attr

            # out *= edge_attr
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            # print('oshape ', out.size())

            # out = torch.cat([out, u[batch]], dim=1)
            # print('out ', out.size())
            # print('x ', x.size())

            out = torch.cat([out, u[batch]], dim=1)
            return self.node_mlp_2(out)

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.


            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)
        self.op = MetaLayer(edge_model, node_model, global_model)


    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.nn = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
        self.channels = 64
        self.conv1 = MyLayer(2, 4, self.channels, self.channels)

        self.convs = nn.ModuleList()
        self.conv_atts = nn.ModuleList()

        for i in range(4):
            self.convs.append(MyLayer(self.channels, 4, self.channels, self.channels))
            self.conv_atts.append(GATConv(self.channels + 2, self.channels))

        # self.conv_last = MyLayer(32, 1, 16)

        self.linear_out1 = (nn.Linear(self.channels, 64, bias=True))
        self.linear_out2 = (nn.Linear(64, 32, bias=True))
        self.linear_out3 = nn.Linear(32, 1)

        # self.edge_embed = nn.Linear(4, 32)

        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(32)


    def forward(self, data):

        mb_size = data.y.size(0)
        # print('batc ', )
        # e_embed = self.edge_embed(data.edge_attr)

        out, e, glob = self.conv1(data.x, data.edge_index, data.edge_attr, torch.zeros(mb_size, self.channels, device=data.x.device), data.batch)


        for i, l in enumerate(self.convs):
            cat = torch.cat([out, data.x], dim=1)
            cat = self.conv_atts[i](cat, data.edge_index)
            vs, e, g = l(cat, data.edge_index, data.edge_attr, glob, data.batch)
            out += vs
            glob += g

        

        # out = self.conv_last(out, data.edge_index, data.edge_attr, data.batch)

        # # print(str(data.batch.cpu().detach().numpy()))
        # # print('osize ', out[data.batch].size())
        # out = torch.mean(out.reshape(self.mb_size, 500, -1), dim=1)
        # prediction = nn.LeakyReLU()(self.bn1(self.linear_out1(glob)))
        # prediction = nn.LeakyReLU()(self.bn2(self.linear_out2(prediction)))
        prediction = nn.LeakyReLU()(self.linear_out1(glob))
        prediction = nn.LeakyReLU()(self.linear_out2(prediction))
        prediction = self.linear_out3(prediction)

        # print('glob ', glob.size())

        return torch.squeeze(prediction)
