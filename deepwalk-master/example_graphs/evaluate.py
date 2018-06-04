import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}

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


def main():
  parser = ArgumentParser("scoring",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
  parser.add_argument("--emb", required=True, help='Embeddings file')
  parser.add_argument("--network", required=True,
                      help='A .mat file containing the adjacency matrix and node labels of the input network.')
  parser.add_argument("--adj-matrix-name", default='network',
                      help='Variable name of the adjacency matrix inside the .mat file.')
  parser.add_argument("--label-matrix-name", default='group',
                      help='Variable name of the labels matrix inside the .mat file.')
  parser.add_argument("--num-shuffles", default=2, type=int, help='Number of shuffles.')
  parser.add_argument("--all", default=False, action='store_true',
                      help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
                      'By default, only training percents of 10, 50 and 90 are used.')

  args = parser.parse_args()


  ## Load Embeddings
  embeddings_file = args.emb
  matfile = args.network
  model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
  
  ## Load labels
  mat = loadmat(matfile)
  A = mat[args.adj_matrix_name]
  graph = sparse2graph(A)
  labels_matrix = mat[args.label_matrix_name]
  labels_count = labels_matrix.shape[1]
  mlb = MultiLabelBinarizer(range(labels_count))
  
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
  print ('Results, using embeddings of dimensionality', X.shape[1])
  print ('-------------------')
  print ('Score :   ', score)
  print ('-------------------')

if __name__ == "__main__":
  sys.exit(main())
