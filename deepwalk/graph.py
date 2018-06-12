#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import sys
import scipy.sparse as sp
import networkx as nx

import random
from random import shuffle

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   GRAPH UTILITY                                                                               #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
class Graph:
  def __init__(self, graph_file):
    g_npz = sp.load_npz(graph_file)
    self.g = nx.from_scipy_sparse_matrix(g_npz)
    self.num_of_nodes = self.g.number_of_nodes()
    self.num_of_edges = self.g.number_of_edges()
    self.edges = self.g.edges(data=True)
    self.nodes = self.g.nodes(data=True)


  # Returns a truncated random walk
  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self.g
    if start: # Start of random walk
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes(data=False)))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha: # Alpha : probability of restart
          path.append(rand.choice(list(G[cur].keys())))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]


  #   Creates list of random walks
  def build_deep_walks(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
    G = self.g
    walks = []

    nodes = list(G.nodes)
    
    for cnt in range(num_paths):
      rand.shuffle(nodes)
      for node in nodes:
        walks.append(self.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    
    return walks
