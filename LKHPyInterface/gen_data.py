import numpy as np
import LKH
from multiprocessing import Pool

def gen_src_problems(num_problems, problem_size, dim=2):
	return np.random.rand(num_problems, problem_size, dim)


def convert_euclidean_to_lkh(problem):
	pstring = ''

	pstring += 'NAME : SAMPLE_PROBLEM\n'
	pstring += 'TYPE : TSP\n'
	pstring += 'DIMENSION : {}\n'.format(problem.shape[0])
	pstring += 'EDGE_WEIGHT_TYPE : EUC_2D\n'
	pstring += 'NODE_COORD_SECTION'

	for i in range(problem.shape[0]):
		pstring += '{} {} {}\n'.format(i + 1, problem[i, 0], problem[i, 1])
	return pstring


def convert_euclideans_to_lkh(problems):
	return [convert_euclidean_to_lkh(problems[i]) for i in range(problems.shape[0])]

def gen_data(num_problems, problem_size):
	src_problems = gen_src_problems(num_problems, problem_size)



