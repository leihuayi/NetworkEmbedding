#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import numpy
import sys

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
def get_splits():
    idx_train = range(200)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    return idx_train, idx_val, idx_test

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
  parser.add_argument("--net", required=True)
  parser.add_argument('--dic-network-name', default='network')
  parser.add_argument('--dic-label-name', default='label')

  args = parser.parse_args()


  ## Load Embeddings
  embeddings_file = args.emb
  model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
  
  ## Load labels
  mat = loadmat(args.net)
  A = mat[args.dic_network_name]
  graph = sparse2graph(A)
  labels_matrix = mat[args.dic_label_name]
  labels_count = labels_matrix.shape[1]
  
  # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
  features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

  ## Split in training, validation, test set
  X, y = features_matrix, labels_matrix

  idx_train, idx_val, idx_test = get_splits()
  y_train_ = y[idx_train]
  y_val_ = y[idx_val]
  y_test_ = y[idx_test]
  X_train = X[idx_train]
  X_test = X[idx_test]

  y_train = format_csr(y_train_)
  y_test = format_csr(y_test_)

  ## Logistic Regression

  # Train on data
  logisticRegr = LogisticRegression()
  logisticRegr.fit(X_train, y_train)

  # Measure accuracy
  score = logisticRegr.score(X_test, y_test)

  # Output results
  print ('-------------------')
  print ('Score :   ', score)
  print ('-------------------')



#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Start                                                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
if __name__ == "__main__":
  sys.exit(main())
