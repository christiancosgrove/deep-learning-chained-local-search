import matplotlib.pyplot as plt
import numpy as np


def visualize_tour(nodes, tour, out_path=None, connect_ends=True):
    """Save a picture of the tour for manual inspection.

    Arguments:
    :param nodes: numpy matrix of nodes, stored as ordered triples (index, x-coord, y-coord)
    :param tour: numpy array, representing order of the nodes, a permutation of the array [1, 2, ..., n]
    :param out_path: desired path to store the image
    :param connect_ends: boolean indicating whether to connect the last point in the tour to the first
    :return: None
    """

    n = nodes.shape[0]
    x = np.asarray(nodes[:, 1])
    y = np.asarray(nodes[:, 2])

    scale_x = x / np.max(x)
    scale_x -= np.mean(scale_x)

    scale_y = y / np.max(y)
    scale_y -= np.mean(scale_y)

    plt.plot(scale_x, scale_y, 'ro')

    for i in range(n - 1):
        curr_index = tour[i]
        next_index = tour[i + 1]

        curr_x = scale_x[curr_index - 1]
        curr_y = scale_y[curr_index - 1]

        next_x = scale_x[next_index - 1]
        next_y = scale_y[next_index - 1]

        plt.plot([curr_x, next_x], [curr_y, next_y], 'k-')

    if connect_ends:
        plt.plot([scale_x[tour[n - 1] - 1], scale_x[tour[0] - 1]], [scale_y[tour[n - 1] - 1], scale_y[tour[0] - 1]], 'k-')

    plt.show()

    if out_path is not None:
        plt.savefig(out_path)


if __name__ == '__main__':
    a = np.matrix('1 32 23; 2 -12 42; 3 21 -12; 4 0 5; 5 0 10; 6 10 0')
    t = np.array([1, 2, 6, 5, 4, 3])
    visualize_tour(a, t, out_path='check_tour.png', connect_ends=True)
