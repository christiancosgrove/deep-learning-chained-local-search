import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import tqdm
from PIL import Image, ImageFile
import torchvision

%load_ext cython
%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'
%load_ext cython
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nodes = np.random.rand(50, 2) * 100

# Initial Plan
# generate starting tours for multiple problems
# generate a bunch of random kicks; find the one that works best on LKH
# train network to output that kick
# repeat training for the possible-not-optimal tours & problems that LKH outputs ("traps")

# LKH on 100 nodes takes ~2 seconds (3000 nodes: 13.4 seconds) but not using a lot of cpu => best bet is to run it in paralell

%cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface
import LKH #global variables ARE saved between runs. Even if rerun this statement. Or use importlib reload.
import time
start = time.time()
LKH.run("")
time.time() - start

#TODO: Current problems: 1) global variables not rest 2) 5x slower for 3000pt prob (same time ~0.05s for 100node L2 prob), not sure why, seems to be a delay at some point
