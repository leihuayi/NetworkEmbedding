#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import logging
import sys
import scipy.sparse as sp
import networkx as nx
from io import open
from os import path
from time import time
from glob import glob
from six import iterkeys
from collections import Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations

from concurrent.futures import ProcessPoolExecutor

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   GRAPH UTILITY                                                                               #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

logger = logging.getLogger("deepwalk")

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph:
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self,graph_file):
    g_npz = sp.load_npz(graph_file)
    self.g = nx.from_scipy_sparse_matrix(g_npz)
    self.num_of_nodes = self.g.number_of_nodes()
    self.num_of_edges = self.g.number_of_edges()
    self.edges = self.g.edges(data=True)
    self.nodes = self.g.nodes(data=True)

  #-----------------------------------------------------------------------------------------------#
  #                                                                                               #
  #   RANDOM_WALK                                                                                 #
  #   Returns a truncated random walk.                                                            #
  #   path_length: Length of the random walk.                                                     #
  #   alpha: probability of restarts.                                                             #
  #   start: the start node of the random walk.                                                   #
  #                                                                                               #
  #-----------------------------------------------------------------------------------------------#
  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self.g
    if start:
      path = [start[0]]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes(data=False)))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(list(G[cur].keys())))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   BUILD_DEEPWALK_CORPUS                                                                       #
#   Creates a lits of random walks                                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes)
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes)

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
