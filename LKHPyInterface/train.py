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
import pickle
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
parser.add_argument("--train_examples", type=int)
parser.add_argument("--mb_size", type=int, default=32)
parser.add_argument("epochs", type=int)
parser.add_argument('--train_dataset')
parser.add_argument('--save_train')
parser.add_argument('--test_dataset')
parser.add_argument('--save_test')

args = parser.parse_args()

if args.train_dataset is not None:
    with open(args.train_dataset, 'rb') as f:
        train_dataset = pickle.load(f)
else:
    train_dataset = gen_data.make_dataset(256, 500, 100, 4)
    if args.save_train:
        with open(args.save_train, 'wb') as outfile:
            pickle.dump(train_dataset, outfile)

if args.test_dataset is not None:
    with open(args.test_dataset, 'rb') as f:
        test_dataset = pickle.load(f)
else:
    test_dataset = gen_data.make_dataset(128, 500, 100, 4)
    if args.save_test:
        with open(args.save_test, 'wb') as outfile:
            pickle.dump(test_dataset, outfile)

train_loader = DataLoader(train_dataset, batch_size=args.mb_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.mb_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)


def train(i):
    losses = []
    k = 0
    for data in tqdm(train_loader):
        data = data.to(device)

        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = nn.MSELoss()(out, data.y)
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        k += 1
        # if k % 100 == 0:
        # print('iloss ', loss)
        if k == 50:
            break

    return np.mean(losses)

def test(loader):
    model.eval()
    accs = []
    k = 0
    for data in tqdm(loader):
        out = nn.MSELoss()(model(data), data.y).item()
        accs.append(out)
        k += 1
        if k == 50:
            break

    return np.mean(accs)


for i in range(args.epochs):
    l = train(i)
    print('loss: ', l, ' Train accuracy: ', test(train_loader),'; test accuracy: ', test(test_loader))

print(args.train_examples, test(train_loader), test(test_loader))