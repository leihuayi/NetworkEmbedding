import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

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
  # 0. Files
  embeddings_file = args.emb
  matfile = args.network
  
  # 1. Load Embeddings
  model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
  
  # 2. Load labels
  mat = loadmat(matfile)
  A = mat[args.adj_matrix_name]
  graph = sparse2graph(A)
  labels_matrix = mat[args.label_matrix_name]
  labels_count = labels_matrix.shape[1]
  mlb = MultiLabelBinarizer(range(labels_count))
  
  # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
  features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

  # 2. to score each train/test group
  X, y = features_matrix, labels_matrix

  idx_train, idx_val, idx_test = get_splits()
  y_train_ = y[idx_train]
  y_val_ = y[idx_val]
  y_test_ = y[idx_test]
  X_train = X[idx_train]
  X_test = X[idx_test]


  y_train = [[] for x in range(y_train_.shape[0])]

  cy =  y_train_.tocoo()
  for i, j in zip(cy.row, cy.col):
      y_train[i].append(j)

  assert sum(len(l) for l in y_train) == y_train_.nnz


  y_test = [[] for _ in range(y_test_.shape[0])]

  cy =  y_test_.tocoo()
  for i, j in zip(cy.row, cy.col):
      y_test[i].append(j)

  clf = TopKRanker(LogisticRegression())
  clf.fit(X_train, y_train_)

  # find out how many labels should be predicted
  top_k_list = [len(l) for l in y_test]
  preds = clf.predict(X_test, top_k_list)

  results = {}
  averages = ["micro", "macro"]
  for average in averages:
      results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)


  print ('Results, using embeddings of dimensionality', X.shape[1])
  print ('-------------------')

  print ('Score :   ', results)

  print ('-------------------')

if __name__ == "__main__":
  sys.exit(main())
