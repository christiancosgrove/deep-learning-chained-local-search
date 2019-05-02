import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from model import Net
import gen_data
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("train_dataset", type=str)
parser.add_argument("model", type=str)
parser.add_argument("--mb_size", type=int, default=64)
args = parser.parse_args()


chunk_size = 1024

train_dataset = gen_data.MyOwnDataset(args.train_dataset, chunk_size)

train_loader = DataLoader(train_dataset, batch_size=args.mb_size, shuffle=False)

device = torch.device('cpu')

model = Net().to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

# print(len(train_dataset))


preds = []
true = []
for data in tqdm(train_loader):
	ys = model(data).cpu().detach().numpy()
	preds += list(ys)
	true += list(data.y.cpu().detach().numpy())
model_selected_kick_values = []
for i in range(256*4):
	index = np.random.randint(16)
	while preds[16*i + index] > 0:
		index = np.random.randint(16)

	# best_model_prediction = np.argmin(preds[16*i:16*(i+1)])
	model_selected_kick_values.append(true[16*i + index])

# true = np.array(true)
# neg = true < 0


# print('err', np.mean(np.array(preds)[neg] - true[neg])**2 / np.var(true[neg]))

def dist(data):

	kick_edges = []
	for j in range(data.edge_attr.size(0)):
		if data.edge_attr[j, 3] == 1:
			kick_edges.append(data.edge_index[:, j])

	dx = data.x.cpu().detach().numpy()
	mids = [(dx[i] + dx[j]) / 2 for i,j in kick_edges]

	total_dist = 0
	for i in mids:
		for j in mids:
			total_dist += np.linalg.norm(i - j)

	return total_dist

nearest_selected_kick_values = []


for i in range(256*4):
	index = np.argmin([dist(ex) for ex in train_dataset[16*i:16*(i+1)]])
	nearest_selected_kick_values.append(true[16*i + index])

random_selected_kick_values = []

for i in range(256*4):
	index = np.random.randint(16)
	random_selected_kick_values.append(true[16*i + index])

oracle_selected_kick_values = []
for i in range(256*4):
	index = np.argmin(true[16*i:16*(i+1)])
	oracle_selected_kick_values.append(true[16*i + index])


# plt.hist(nearest_selected_kick_values, alpha=0.9, bins=100)
# plt.hist(random_selected_kick_values, alpha=0.9, bins=100)
# plt.hist(model_selected_kick_values, alpha=0.9, bins=100)

# plt.xlim(-0.8,0.15)
# plt.yscale('log')
# plt.xlabel('change in tour length')
# plt.ylabel('number of kicks')
# plt.legend(['close', 'random', 'ours'])
# plt.savefig('./comparison.eps')
# plt.show()

print('mean change in tour lengths: ')
print('close: ', np.mean(nearest_selected_kick_values), ' pm ', np.std(nearest_selected_kick_values))
print('random: ', np.mean(random_selected_kick_values), ' pm ', np.std(random_selected_kick_values))

print('oracle: ', np.mean(oracle_selected_kick_values), ' pm ', np.std(oracle_selected_kick_values))
print('ours: ', np.mean(model_selected_kick_values), ' pm ', np.std(model_selected_kick_values))
