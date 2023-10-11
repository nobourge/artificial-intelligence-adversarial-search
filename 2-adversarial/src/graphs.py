# this file does the following:

# • report and compare the size of the paths found for the three search algorithms on the
# level 3.
# • compare the number of nodes extended during the search for BFS, DSF and A∗ when searching
# level 3 resolution and discuss the reasons for these differences.



import numpy as np
from typing import List, Tuple
from lle import World
from problem import SimpleSearchProblem, GemSearchProblem, CornerSearchProblem
from search import bfs, dfs, astar, Solution
from utils import print_items
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

dfs_path_size=  8096
bfs_path_size=  10
astar_path_size=  10
dfs_nodes_expanded=  10259
bfs_nodes_expanded=  11016
astar_nodes_expanded=  10   

# create a bar graph to show the size of the paths found for the three search algorithms on the level 3
# create a bar graph to show the number of nodes expanded for the three search algorithms on the level 3

# data to plot
n_groups = 3
path_sizes = (dfs_path_size, bfs_path_size, astar_path_size)
nodes_expanded = (dfs_nodes_expanded, bfs_nodes_expanded, astar_nodes_expanded)

# # create plot
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 1

# rects1 = plt.bar(index, path_sizes, bar_width,
# alpha=opacity,
# color='b',
# label='Path Size')

# rects2 = plt.bar(index + bar_width, nodes_expanded, bar_width,
# alpha=opacity, 
# color='g',
# label='Nodes Expanded')

# plt.xlabel('Search Algorithm')
# plt.ylabel('Number of Nodes')
# plt.title('Comparison of Search Algorithms')
# plt.xticks(index + bar_width, ('DFS', 'BFS', 'A*'))
# plt.legend()

# plt.tight_layout()
# plt.show()




# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Data to plot
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.8

# Plot Path Sizes
axs[0].bar(index, path_sizes, bar_width, alpha=opacity, color='b', label='Path Size')
axs[0].set_xlabel('Search Algorithm')
axs[0].set_ylabel('Path Size')
axs[0].set_title('Path Sizes of Different Search Algorithms')
axs[0].set_xticks(index)
axs[0].set_xticklabels(('DFS', 'BFS', 'A*'))

# Annotate each bar with the actual value
for i, value in enumerate(path_sizes):
    axs[0].text(i, value, str(value), ha='center', va='bottom')

# Plot Nodes Expanded
axs[1].bar(index, nodes_expanded, bar_width, alpha=opacity, color='g', label='Nodes Expanded')
axs[1].set_xlabel('Search Algorithm')
axs[1].set_ylabel('Nodes Expanded')
axs[1].set_title('Nodes Expanded in Different Search Algorithms')
axs[1].set_xticks(index)
axs[1].set_xticklabels(('DFS', 'BFS', 'A*'))

# Annotate each bar with the actual value
for i, value in enumerate(nodes_expanded):
    axs[1].text(i, value, str(value), ha='center', va='bottom')

# Show plots
plt.tight_layout()
plt.show()
