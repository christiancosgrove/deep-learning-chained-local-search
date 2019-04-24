import torch
from torch_geometric.data import Data
import os
import numpy as np
import argparse
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
parser.add_argument("--train_examples", type=int)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

def load_data(file):
    dat = np.genfromtxt(file, skip_header=7, dtype='str')

    g = dat.shape[0]

    e = dat.shape[1] - 1
    n = int((1 + np.sqrt(1 + 8 * e)) / 2)
    x = np.zeros((g, n, n))


    edges = []

    for k in range(dat.shape[0]):
        c = 0
        l = []
        for i in range(n):
            for j in range(i+1, n):
                if dat[k, c] == '1':
                    l.append((i, j))
                    l.append((j, i))
                c+=1
        edges.append(np.array(l).transpose())
    y = np.zeros(g)
    y[dat[:, e] == 'Y'] = 1
    return n, edges, y


from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader


class MyOwnDataset(Dataset):
    def __init__(self, data_objects):
        self.data_objects = data_objects
        self.transform = None
        self.num_classes = 2

    def __len__(self):
        return len(self.data_objects)


    def get(self, idx):
        return self.data_objects[idx]


def make_graphs(es, ys, n):
    return [Data(x=torch.eye(n).view(n, n), edge_index=torch.tensor(es[i], dtype=torch.long), y=torch.tensor([ys[i]], dtype=torch.float)).to(device) for i in range(len(es))]

n, edges, y = load_data(args.data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_indices = np.random.choice(len(edges), args.train_examples, replace=False)

test_indices = np.ones(len(edges), dtype=np.long)
test_indices[train_indices] = 0


graphs_train = make_graphs([edges[i] for i in train_indices], y[train_indices], n)
graphs_test = make_graphs([edges[i] for i in range(test_indices.shape[0]) if test_indices[i] == 1], y[test_indices > 0], n)

dataset = MyOwnDataset(graphs_train)
test_dataset = MyOwnDataset(graphs_test)


mb_size = 64
train_loader = DataLoader(dataset, batch_size=mb_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=mb_size)

from torch_geometric.nn import (GraphConv, NNConv, global_mean_pool)


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

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-3)


def train(i):

    losses = []
    for data in train_loader:
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = nn.BCEWithLogitsLoss()(out, data.y)
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
    return np.mean(losses)



def test(loader):
    model.eval()
    accs = []
    for data in loader:
        out = (model(data) > torch.tensor([0.5]).to(device)).float()
        acc = out.eq(data.y).sum().item() / data.y.size(0)
        accs.append(acc)
    return np.mean(accs)
for i in range(args.epochs):
    train(i)

print(args.train_examples, test(train_loader), test(test_loader))