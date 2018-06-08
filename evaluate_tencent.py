#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import numpy as np
import sys
import scipy.sparse as sp

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Create dictionary (graph) our of sparse matrix                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   HELPER FUNCTIONS FOR SPLITTING DATA TRAIN, VALIDATION, TEST                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def format_csr(y_):
  y = [[] for x in range(y_.shape[0])]

  cy =  y_.tocoo()
  for i, j in zip(cy.row, cy.col):
      y[i].append(j)
  return y



#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   EVALUATE                                                                                    #
#   Perform Logistic Regression of embedding                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
  parser = ArgumentParser("evaluate",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
  parser.add_argument("--emb", required=True)
  parser.add_argument("--network", required=True)
  parser.add_argument('--dic-network-name', default='network')
  parser.add_argument('--dic-label-name', default='label')

  args = parser.parse_args()


  ## Load Embeddings
  embeddings_file = args.emb
  #matfile = args.network
  #model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
  
  ## Load labels
  #mat = loadmat(matfile)
  #A = mat[args.dic_network_name]
  #graph = sparse2graph(A)

  ## Labels train
  edges_train = np.load("tencent/train_edges.npy")
  row = np.array(edges_train[:,0])
  col = np.array(edges_train[:,1])
  size = np.amax([np.amax(row), np.amax(col)])+1 # TODO : CALCULATE WITH GRAPH
  data = np.ones(edges_train.shape[0])

  y_train = sp.csr_matrix((data, (row, col)), shape=(size, size))
  print(y)

  ## Labels test
  edges_test = np.load("tencent/test_edges.npy")
  row = np.array(edges_test[:,0])
  col = np.array(edges_test[:,1])
  size = np.amax([np.amax(row), np.amax(col)])+1 # TODO : CALCULATE WITH GRAPH
  data = np.ones(edges_test.shape[0])

  y_test = sp.csr_matrix((data, (row, col)), shape=(size, size))
  
  # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
  #features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

  ## Split in training, validation, test set


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Start                                                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
if __name__ == "__main__":
  sys.exit(main())
