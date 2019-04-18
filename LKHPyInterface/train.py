import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
import gen_data
from model import Net
import argparse
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
parser.add_argument("--train_examples", type=int)
parser.add_argument("--mb_size", type=int, default=1)
parser.add_argument("epochs", type=int)
args = parser.parse_args()


train_dataset = gen_data.make_dataset(5, 10, 10)
test_dataset = gen_data.make_dataset(10, 100, 10)

train_loader = DataLoader(train_dataset, batch_size=args.mb_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.mb_size)

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)


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
        out = (torch.sigmoid(model(data)) * data.y).sum().item() / 4
        # acc = out.eq().sum().item() / data.y.size(0)
        accs.append(out)

    return np.mean(accs)


for i in range(args.epochs):
    train(i)
    print('Train accuracy: ', test(train_loader), '; test accuracy: ', test(test_loader))

print(args.train_examples, test(train_loader), test(test_loader))