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
import os




parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
parser.add_argument("--train_examples", type=int)
parser.add_argument("--mb_size", type=int, default=32)
parser.add_argument("epochs", type=int)
parser.add_argument('train_dataset')
# parser.add_argument('--save_train')
parser.add_argument('test_dataset')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--save_model')
parser.add_argument('--save_results')
# parser.add_argument('--save_test')

args = parser.parse_args()

chunk_size = 1024

if not os.path.exists(args.train_dataset) or args.overwrite:
    gen_data.make_dataset(args.train_dataset, 256*4, 150, 16, chunk_size, num_workers=16)
train_dataset = gen_data.MyOwnDataset(args.train_dataset, chunk_size)

if not os.path.exists(args.test_dataset) or args.overwrite:
    gen_data.make_dataset(args.test_dataset, 256, 150, 16, chunk_size, num_workers=16)
test_dataset = gen_data.MyOwnDataset(args.test_dataset, chunk_size)



train_loader = DataLoader(train_dataset, batch_size=args.mb_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.mb_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))#, weight_decay=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.5)#, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

def train(i):
    losses = []
    k = 0
    model.train()
    scheduler.step()
    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = nn.MSELoss()(out, data.y)

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        k += 1
        # if k % 100 == 0:
        # print('iloss ', loss)
        # if k == 200:
        #     break
    return np.mean(losses)

def test(loader):
    model.eval()
    accs = []
    k = 0
    ys = []


    miny = 0
    maxy = 0

    for data in loader:
        data = data.to(device)
        out = nn.MSELoss()(model(data), data.y).item()
        accs.append(out)
        k += 1

        y = data.y.cpu().detach().numpy()
        ys.append(y)

        # print('y ', y)

        if np.min(y) < miny:
            miny = np.min(y)
        if np.max(y) > maxy:
            maxy = np.max(y)


        # if k == 50:
        #     break
        # means.append(np.std(y)**2)

    # import matplotlib.pyplot as plt

    # plt.hist(np.reshape(ys, (-1)), bins=20)
    # plt.show()

    # print('min y', miny, ' maxy ', maxy)

    return np.mean(accs) / np.var(ys)#np.mean(np.abs(ys - np.mean(ys)))

results = []
for i in range(args.epochs):
    l = train(i)
    if i % 10 == 0:
        torch.save(model.state_dict(), args.save_model)
        print('epoch ', i, ' loss: ', l, ' train error: ', test(train_loader),'; test error: ', test(test_loader))
    results.append([i, l, test(train_loader), test(test_loader)])
    f = open(args.save_results, 'wb')
    pickle.dump(results, f)
    f.close()
    
print(args.train_examples, test(train_loader), test(test_loader))
