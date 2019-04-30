import pickle
import matplotlib.pyplot as plt
import numpy as np

f = open("./data/results.pkl", 'rb')
obj_file = pickle.load(f)
f.close()
indices = np.asarray([row[0] for row in obj_file])
losses = np.asarray([row[1] for row in obj_file])
train_errors = np.asarray([row[2] for row in obj_file])
test_errors = np.asarray([row[3] for row in obj_file])

