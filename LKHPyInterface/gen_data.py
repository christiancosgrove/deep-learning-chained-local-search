# %cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/deep-learning-chained-local-search/LKHPyInterface
import numpy as np
import LKH
from multiprocessing import Pool
from tqdm import tqdm
from collections import OrderedDict

import pickle 
import os


def gen_src_problems_clustered(num_problems, num_cities, dim=2, num_clusters=5):
    probs = []
    for i in range(num_problems):
        clusters = np.random.rand(num_clusters, dim)
        coords = []
        for j in range(num_cities):
            c = np.random.randint(len(clusters))
            coords.append(np.random.normal(clusters[c], 0.1))
        probs.append(np.array(coords))
    return np.array(probs)


def gen_src_problems(num_problems, num_cities, dim=2):
    return np.random.rand(num_problems, num_cities, dim)


# Generates a toy problem where the points lie in a circle
def gen_src_problems_circle(num_problems, num_cities, dim=2):
    space = np.linspace(0, 2 * np.pi, num_cities, endpoint=False)
    arr = np.array([np.cos(space), np.sin(space)]).T

    return arr.reshape(1, num_cities, dim)


def convert_euclidean_to_lkh(problem):
    pstring = ''

    pstring += 'NAME : SAMPLE_PROBLEM\n'
    pstring += 'COMMENT : NONE\n'
    pstring += 'TYPE : TSP\n'
    pstring += 'DIMENSION : {}\n'.format(problem.shape[0])
    pstring += 'EDGE_WEIGHT_TYPE : EUC_2D\n'
    pstring += 'NODE_COORD_SECTION\n'

    for i in range(problem.shape[0]):
        pstring += '{} {:.8e} {:.8e}\n'.format(i + 1, problem[i, 0], problem[i, 1])

    return pstring


def convert_euclideans_to_lkh(problems):
    return [convert_euclidean_to_lkh(problems[i]) for i in range(problems.shape[0])]


def convert_lkh_to_input(problems, problem_size, initial_tours=None, constrain=True):
    use_initial = 1
    if constrain:
        params = {
            "PROBLEM_FILE": "placeholder",
            "RUNS": 1,
            "MOVE_TYPE": 2,
            # "NONSEQUENTIAL_MOVE_TYPE": 4,
            # "SUBSEQUENT_PATCHING": "NO",
            "PATCHING_A": 0,
            "PATCHING_C": 0,
            "TRACE_LEVEL": 0,
            "SUBGRADIENT" : "NO",
            "MAX_TRIALS" : 2,
            # "TIME_LIMIT" : 1,
            # "MAX_CANDIDATES" : 1
            # "CANDIDATE_SET_TYPE" : "DELAUNAY"
        }
    else:
        params = {
            "PROBLEM_FILE": "placeholder",
            "RUNS": 1,
            "MOVE_TYPE": 2,
            # "NONSEQUENTIAL_MOVE_TYPE": 4,
            # "SUBSEQUENT_PATCHING": "NO",
            "PATCHING_A": 0,
            "PATCHING_C": 0,
            "TRACE_LEVEL": 0,
            "SUBGRADIENT" : "NO",
            "MAX_TRIALS" : 2,
            # "TIME_LIMIT" : 1,
            # "MAX_CANDIDATES" : 100
            # "CANDIDATE_SET_TYPE" : "DELAUNAY"
        }
    if initial_tours is None:
        initial_tours = np.zeros((len(problems), problem_size), dtype=np.int32)
        use_initial = 0

    return [(p, params, initial_tours[i] + 1, use_initial) for i, p in enumerate(problems)]


def run_lkh(converted_problems, num_workers, printDebug=False, progress=False):
    outs = []
    r = range(0, len(converted_problems), num_workers)
    if progress:
        r = tqdm(r)
    for i in r:
        pool = Pool(num_workers)
        outs += pool.starmap(LKH.run, [p + (int(printDebug),) for p in converted_problems[i: i + num_workers]])
        # pool.close()
        pool.terminate()
        # pool.join()

    return [o - 1 for o in outs]


def tour_to_pointer(tour):
    p = np.zeros_like(tour)
    for i in range(len(tour)):
        p[tour[i]] = tour[(i + 1) % len(tour)]

    return p


def pointer_to_tour(pointer):
    t = np.zeros_like(pointer)
    n = 0
    for i in range(len(pointer)):
        t[i] = n
        n = pointer[n]
    if n != 0:
        print(t)
        raise RuntimeError("NOT CONNECTED!")
    return t


def double_bridge(tour, i, j, k, l):
    tlen = len(tour)
    pointer = tour_to_pointer(tour)
    # print('dub bridge pointer before ', pointer)

    # double bridge 1
    o1 = pointer[tour[i]]
    pointer[tour[i]] = pointer[tour[j]]
    pointer[tour[j]] = o1

    # double bridge 2
    o2 = pointer[tour[k]]
    pointer[tour[k]] = pointer[tour[l]]
    pointer[tour[l]] = o2

    # print('dub bridge pointer ', pointer)

    return pointer_to_tour(pointer)


# TODO : IMPLEMENT RANDOM KICKS
def rand_kick(tour):
    ltour = len(tour)
    if ltour < 4:
        raise RuntimeError("Can't do a double bridge with less than 4 nodes!")

    # l = np.random.randint(ltour//4)
    # k = np.random.randint(l+2, 3*ltour//4)
    # i = np.random.randint(l+1, k)
    # j = np.random.randint(k + 1, l + ltour) % ltour

    l = np.random.randint(ltour)
    k = np.random.randint(l + 2, l + ltour - 2)
    i = np.random.randint(l + 1, k)
    j = np.random.randint(k + 1, l + ltour)

    l %= ltour
    k %= ltour
    i %= ltour
    j %= ltour

    return double_bridge(tour, i, j, k, l), (i, j, k, l)


# TODO: implement eval tour
def tour_length(tour, node_coords):
    return np.sum([np.linalg.norm(node_coords[tour[(i + 1) % len(tour)]] - node_coords[tour[i]]) for i in range(len(node_coords))])


LKH_SCALE = 10000


def gen_data(num_problems, problem_size, num_kicks=10, num_workers=4):
    src_problems = gen_src_problems_clustered(num_problems, problem_size)
    src_problems_converted = convert_euclideans_to_lkh(src_problems * LKH_SCALE)

    print('Generating stuck tours')
    stuck_tours = run_lkh(convert_lkh_to_input(src_problems_converted, problem_size, constrain=False), num_workers, False, True)
    
    best_nodes = []
    print ('Running LKH on kicks')

    temperature = 100

    ktours = []

    for i, stuck in enumerate(tqdm(stuck_tours)):

        kicks = []
        score_dist = []
        selected_nodes = []

        bn = None
        best_length = 1e100
        worst_length = -1

        if num_kicks % num_workers != 0:
            raise RuntimeError("num_kicks must be a multiple of num_workers")

        stuck_length = tour_length(stuck, src_problems[i])



        for k in tqdm(range(num_kicks // num_workers)):
            kicked_tours = []
            kicked_nodes = []
            for j in range(num_workers):
                tour, nodes = None, None

                num_rejected = 0
                # add acceptance criterion
                while tour is None or np.random.uniform() > np.max([0.1,np.exp(-(tour_length(tour, src_problems[i]) - stuck_length)/temperature)]):
                    tour, nodes = rand_kick(stuck)
                    num_rejected += 1
                # print('num rejected ', num_rejected)

                kicked_tours.append(tour)
                kicked_nodes.append(nodes)


            tours = run_lkh(convert_lkh_to_input([src_problems_converted[i]] * num_workers, problem_size, kicked_tours),
                            num_workers)

            scores = [tour_length(t, src_problems[i]) for t in tours]
            score_dist += scores

            kicks += kicked_tours
            selected_nodes += kicked_nodes

        mean = np.mean(score_dist)
        std = np.std(score_dist) + 0.01
        zs = [(s - mean)/std for s in score_dist]

        # zs = np.random.normal(size=len(score_dist))
        # rands = np.random.normal(size=len(score_dist))
        # zs = rands
        # min = np.min(score_dist)
        # zs = [(s - min) for s in score_dist]
        ktours.append((kicks, selected_nodes, np.array(score_dist) - stuck_length))

        # import matplotlib.pyplot as plt
        # plt.hist(np.array(score_dist) - stuck_length, bins=50)
        # plt.show()


    return (src_problems, ktours)


import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from scipy.spatial import Delaunay


def make_geometric_data(points, edges, best_nodes, zscore):
    n = points.shape[0]

    # get kick edge indices
    inds = []
    length = points.shape[0]
    for i in range(len(edges)):
        for e in [
            (best_nodes[0], (best_nodes[1] + 1)%length),
            ((best_nodes[0] + 1)%length, best_nodes[1]),
            (best_nodes[2], (best_nodes[3] + 1)%length),
            ((best_nodes[2] + 1)%length, best_nodes[3])]:
            if edges[i] == e:
                inds.append(i)


    # Assume that the first n edges correspond to tour
    features = np.zeros((n, 2))
    # features[:n, 0] = 1
    features[:, :2] = points

    # TODO : store length of edge
    edge_features = np.zeros((len(edges), 4))
    # edge_features = np.zeros((len(edges), 2))
    for i in range(len(points)):
        edge_features[i, 0] = 1

    for j in best_nodes:
        edge_features[j, 1] = 1

    for i in range(edge_features.shape[0]):
        edge_features[i, 2] = np.linalg.norm(points[edges[i][0]] - points[edges[i][1]])

    for j in inds:
        edge_features[j, 3] = 1

    # y = np.zeros(n)
    # if best_nodes[0] == -1:
    #     y = np.ones(n) / n
    # else:
    #     for i in best_nodes:
    #         y[i] = 1

    edges = np.array(edges).T

    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float), y=torch.tensor([zscore], dtype=torch.float))


def tour_edges(tour):
    return [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]

# TODO : implement other methods of "compressing" graph
def make_data_delaunay(points, tour, best_nodes, zscore):
    edges = OrderedDict.fromkeys(tour_edges(tour))
    tri = Delaunay(points)
    for triangle in tri.simplices:
        edges[(triangle[0], triangle[1])] = None
        edges[(triangle[0], triangle[2])] = None
        edges[(triangle[1], triangle[2])] = None

    length = points.shape[0]

    edges[(best_nodes[0], (best_nodes[1] + 1)%length)] = None
    edges[((best_nodes[0] + 1)%length, best_nodes[1])] = None
    edges[(best_nodes[2], (best_nodes[3] + 1)%length)] = None
    edges[((best_nodes[2] + 1)%length, best_nodes[3])] = None

    return make_geometric_data(points, list(edges.keys()), best_nodes, zscore)


def make_data_nearest(points, tour, best_nodes, zscore, k=5):
    edges = OrderedDict.fromkeys(tour_edges(tour))
    degrees = []
    for i in range(points.shape[0]):
        s = np.argsort(np.linalg.norm(points[i] - points, axis=1))
        d = 0
        for j in range(k):
            edges[(i, s[j+1])] = None
            d += 1
        for j in range(k + 1, len(s)):
            if np.random.uniform() < k / (len(s) - k+1):
                edges[(i, s[j])] = None
                d += 1

        degrees.append(d)

    print('average degree ', np.mean(degrees))

    return make_geometric_data(points, list(edges.keys()), best_nodes, zscore)


class MyOwnDataset(Dataset):
    def __init__(self, path, chunk_size):

        self.path = path
        self.chunk_size = chunk_size
        self.transform = None

        self.len = len(os.listdir(path)) * chunk_size
        self.curr_chunk_idx = -1

        self.data_objects = []
        for i in range(len(os.listdir(path))):
            with open(os.path.join(self.path, '{}.pkl'.format(i)), 'rb') as infile:
                self.data_objects += (pickle.load(infile))


    def __len__(self):
        return self.len

    def get_chunk(self, idx):
        if self.curr_chunk_idx == -1 or self.curr_chunk_idx != idx // self.chunk_size:
            self.curr_chunk_idx = idx // self.chunk_size
            with open(os.path.join(self.path, '{}.pkl'.format(idx // self.chunk_size)), 'rb') as infile:
                self.curr_chunk = pickle.load(infile)
        return self.curr_chunk

    def get(self, idx):
        # chunk = self.get_chunk(idx)
        # return chunk[idx % self.chunk_size]
        return self.data_objects[idx]


def make_dataset_chunk(num_problems, num_cities, num_kicks, num_workers=4):
    points, kick_info = gen_data(num_problems, num_cities, num_kicks, num_workers)

    data = []
    for i in range(num_problems):
        for j in range(num_kicks):
            data.append(make_data_delaunay(points[i], kick_info[i][0][j], kick_info[i][1][j], kick_info[i][2][j]))

    return data

def make_dataset(path, num_problems, num_cities, num_kicks, chunk_size, num_workers=4):

    os.makedirs(path, exist_ok=True)

    if num_problems*num_kicks % chunk_size != 0:
        raise RuntimeError("num_problems*num_kicks must be a multiple of chunk_size")
    if chunk_size % num_kicks != 0:
        raise RuntimeError("chunk_size must be a multiple of num_kicks")

    print('Generating dataset chunks ')
    for i in tqdm(range(num_problems*num_kicks // chunk_size)):
        with open(os.path.join(path, '{}.pkl'.format(i)), 'wb') as outfile:
            pickle.dump(make_dataset_chunk(chunk_size // num_kicks, num_cities, num_kicks, num_workers), outfile)




# src_problems = gen_src_problems_clustered(1, 100)
# src_problems_converted = convert_euclideans_to_lkh(src_problems * LKH_SCALE)

# stuck_tours = run_lkh(convert_lkh_to_input(src_problems_converted, 100), num_workers=1)

# # kicked = rand_kick(stuck_tours[0] - 1)[0]

# visualize_tour(src_problems[0], stuck_tours[0], arrows=False)

# print(make_data_delaunay(points, np.arange(10), (1, 2, 3, 4)))

# import matplotlib.pyplot as plt
# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())

# plt.show()
