import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import tqdm
from PIL import Image, ImageFile
import torchvision
import importlib
import os
import sys

# %load_ext cython
# %matplotlib inline
plt.style.use("ggplot")
# %config InlineBackend.figure_format = 'svg'
# %load_ext cython
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nodes = np.random.rand(50, 2) * 100

# Initial Plan
# generate starting tours for multiple problems
# generate a bunch of random kicks; find the one that works best on LKH
# train network to output that kick
# repeat training for the possible-not-optimal tours & problems that LKH outputs ("traps")

# LKH on 100 nodes takes ~2 seconds (3000 nodes: 13.4 seconds) but not using a lot of cpu => best bet is to run it in paralell

# %cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface
import LKH #global variables ARE saved between runs. Even if rerun this statement. Or use importlib reload.
import time
start = time.time()
params = {
	"PROBLEM_FILE":"placeholder",
	"RUNS":1,
	"MOVE_TYPE" : 2,
	"TRACE_LEVEL":0
}

with open("LKHProgram/TSP100.tsp", "r") as f:
	problemFileLines = f.readlines()
problemString = '\n'.join(problemFileLines)
with open("LKHProgram/TSP100.tsp", "r") as f:
	problemFileLines2 = f.readlines()
problemString2 = '\n'.join(problemFileLines2)

from multiprocessing import Process, Pool

# Dummy initial tour
# Simply a numpy array of len(nodes)
null1 = np.zeros(100, dtype=np.int32)
null2 = np.zeros(100, dtype=np.int32)


problems = [(problemString, params, null1), (problemString2, params, null2)]


# Initialize a process pool of one process - this will run problems sequentially
# To run multiple LKH processes in parallel, just use pass in 2 or more
pool = Pool(1)

# Run LKH on an array of problems
o = pool.starmap(LKH.run, problems)

# Print the output of LKH on those problems
print(o)
# p2.join()

time.time() - start

#TODO: Current problems: 1) global variables not rest 2) 5x slower for 3000pt prob (same time ~0.05s for 100node L2 prob), not sure why, seems to be a delay at some point
