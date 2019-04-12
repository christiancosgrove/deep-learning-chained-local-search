import numpy as np
import LKH
from multiprocessing import Pool


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

def convert_lkh_to_input(problems, problem_size, initial_tours=None):
	use_initial = 1
	params = {
		"PROBLEM_FILE":"placeholder",
		"RUNS":1,
		"MOVE_TYPE" : 5,
		"TRACE_LEVEL":0,
		"MAX_TRIALS" : 20
		# "CANDIDATE_SET_TYPE" : "DELAUNAY"
	}
	if initial_tours is None:
		initial_tours = np.zeros((len(problems), problem_size), dtype=np.int32)
		use_initial = 0

	return [(p, params, initial_tours[i], use_initial) for i, p in enumerate(problems)]

def run_lkh(converted_problems, num_workers):
	outs = []
	for i in range(0, len(converted_problems), num_workers):
		pool = Pool(num_workers)
		outs += pool.starmap(LKH.run, converted_problems[i : i+num_workers])
		pool.terminate()

	return outs

# TODO : IMPLEMENT RANDOM KICKS
def rand_kick(tour):
	return tour, (0, 1, 2, 3)

# TODO: implement eval tour
def tour_length(tour, node_coords):
	return np.sum([np.linalg.norm(node_coords[(i + 1) % len(tour)] - node_coords[i]) for i in range(len(node_coords))])


LKH_SCALE = 100000

def gen_data(num_problems, problem_size, num_kicks=10, num_workers=4):
	src_problems = gen_src_problems(num_problems, problem_size)
	src_problems_converted = convert_euclideans_to_lkh(src_problems * LKH_SCALE)

	stuck_tours = run_lkh(convert_lkh_to_input(src_problems_converted, problem_size), num_workers)
	best_nodes = []
	for i, stuck in enumerate(stuck_tours):
		kicked_tours = []
		kicked_nodes = []
		for j in range(num_kicks):
			tour, nodes = rand_kick(stuck)
			kicked_tours.append(tour)
			kicked_nodes.append(nodes)

		tours = run_lkh(convert_lkh_to_input([src_problems_converted[i]] * num_kicks, problem_size, kicked_tours), num_workers)
		scores = [tour_length(t, src_problems[i]) for t in tours]
		best = np.argmin(scores)
		best_nodes.append(best)

	return (src_problems, stuck_tours, best_nodes)

# gen_data(10, 10)

src_problems = gen_src_problems_clustered(1, 4000)
src_problems_converted = convert_euclideans_to_lkh(src_problems*LKH_SCALE)

stuck_tours = run_lkh(convert_lkh_to_input(src_problems_converted, 4000), 1)

from Visualize import visualize_tour
print(src_problems.shape)
visualize_tour(src_problems[0], stuck_tours[0])