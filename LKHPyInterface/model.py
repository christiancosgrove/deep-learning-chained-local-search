
import torch
from torch import nn
from torch.nn import functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channels = 16
        self.conv1 = GraphConv(n, self.channels)
        self.conv2 = GraphConv(self.channels, self.channels)
        self.conv3 = GraphConv(self.channels, self.channels)


        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(self.channels)

        self.lin1 = nn.Linear(self.channels, self.channels)
        self.lin2 = nn.Linear(self.channels, 1)


    def forward(self, data):
        bsize = data.x.size(0) // n
        x = data.x.view(-1, n)
        x = self.bn1(F.relu(self.conv1(x, data.edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.conv2(x, data.edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn3(F.relu(self.conv3(x, data.edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view((bsize, -1, self.channels))
        x = torch.mean(x, dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.squeeze()