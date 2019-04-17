import matplotlib.pyplot as plt
import numpy as np


def visualize_tour(nodes, tour, out_path=None, connect_ends=True, plot_points=False, arrows=False):
    """Save a picture of the tour for manual inspection.

    Arguments:
    :param nodes: numpy matrix of nodes, stored as ordered triples (index, x-coord, y-coord)
    :param tour: numpy array, representing order of the nodes, a permutation of the array [1, 2, ..., n]
    :param out_path: desired path to store the image
    :param connect_ends: boolean indicating whether to connect the last point in the tour to the first
    :param plot_points: boolean to plot the nodes of the graph. Set to False especially on large graphs
    :param arrows: boolean to include arrows to indicate the direction of the tour
    :return: None
    """

    n = nodes.shape[0]
    x = np.asarray(nodes[:, 0])
    y = np.asarray(nodes[:, 1])

    scale_x = x / np.max(x)
    scale_x -= np.mean(scale_x)

    scale_y = y / np.max(y)
    scale_y -= np.mean(scale_y)

    if plot_points:
        plt.plot(scale_x, scale_y, 'ro')

    def draw_line(x1, y1, x2, y2):
        plt.plot([x1, x2], [y1, y2], 'k-')
        if arrows:
            midx = (x1 + x2)/2
            midy = (y1 + y2)/2
            dx = x2 - x1
            dy = y2 - y1
            l = np.sqrt(dx**2 + dy**2)
            dx /= l
            dy /= l
            plt.arrow(midx, midy, dx*0.1, dy*0.1, shape='full', length_includes_head=True, head_width=.05)

    for i in range(n - 1):
        curr_index = tour[i]
        next_index = tour[i + 1]

        curr_x = scale_x[curr_index]
        curr_y = scale_y[curr_index]

        next_x = scale_x[next_index]
        next_y = scale_y[next_index]

        draw_line(curr_x, curr_y, next_x, next_y)
        # plt.arrow(curr_x, curr_y, next_x - curr_x, next_y - curr_y)

    if connect_ends:
        draw_line(scale_x[tour[-1]], scale_y[tour[-1]], scale_x[tour[0]], scale_y[tour[0]])

    if out_path is not None:
        plt.savefig(out_path)

    plt.show()


if __name__ == '__main__':
    a = np.matrix('10 10; 10 0; 10 -10; 0 10; 0 0; 0 -10; -10 10; -10 0; -10 -10')
    t = np.array([0, 1, 2, 5, 4, 8, 7, 6, 3])
    visualize_tour(a, t, out_path='check_tour.png', connect_ends=True)
