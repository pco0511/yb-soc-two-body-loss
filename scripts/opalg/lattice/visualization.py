import itertools


import numpy as np
import matplotlib.pyplot as plt


def visualize_lattice_1d(lengths, d):
    dim = len(lengths)
    assert dim == 1
    
    n = lengths[0]
    plt.figure(figsize=(6, 2.5))
    points_x = d * np.arange(n)
    points_y = np.zeros_like(points_x)
    plt.title("Lattice")
    plt.xlim((-d, d * n))
    plt.scatter(points_x, points_y)
    plt.gca().axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.show()


def visualize_lattice_2d(lengths, d, figsize=None):
    dim = len(lengths)
    assert dim == 2
    
    nx, ny = lengths
    
    
    figsize = figsize or (6 * np.sqrt(nx / ny), 6 * np.sqrt(ny / nx))
    
    plt.figure(figsize=figsize)
    
    points_x, points_y = d * np.array(list(itertools.product(range(nx), range(ny)))).T
    
    plt.title("Lattice")
    plt.xlim((-d, d * nx))
    plt.ylim((-d, d * ny))
    plt.scatter(points_x, points_y)
    plt.tight_layout()
    plt.show()
