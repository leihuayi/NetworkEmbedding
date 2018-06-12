#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import numpy as np
import sys
import scipy.sparse as sp
import pickle
import networkx as nx

import argparse
from sklearn.metrics import roc_auc_score

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   EVALUATE                                                                                    #
#   Perform Logistic Regression of embedding                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--emb", required=True)
  parser.add_argument("--net", required=True)
  parser.add_argument("--testdir", required=True)
  parser.add_argument('--dic-network-name', default='network')
  parser.add_argument('--dic-label-name', default='label')

  args = parser.parse_args()


  ## Load graph
  G=sp.load_npz(args.net)
  graph = nx.from_scipy_sparse_matrix(G)

  ## Load Embeddings
  emb_mappings = pickle.load(open(args.emb, 'rb'))
  emb_list = []
  for node_index in emb_mappings.keys():
      node_emb = emb_mappings[node_index]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)
  
  print(emb_matrix.shape)
  

  # Load test edges
  edges_pos = np.load(args.testdir+"/test_edges.npy")
  edges_neg = np.load(args.testdir+"/test_edges_false.npy")

  ## Compute ROC score
  # Edge case
  if len(edges_pos) == 0 or len(edges_neg) == 0:
      return (None, None, None)

  # Store positive edge predictions, actual values
  preds_pos = []
  pos = []
  for edge in edges_pos:
      preds_pos.append(np.dot(emb_matrix[edge[0],:],emb_matrix[edge[1],:])) # Inner product for node similarity
      pos.append(1) # actual value (1 for positive)
      
  # Store negative edge predictions, actual values
  preds_neg = []
  neg = []
  for edge in edges_neg:
      preds_neg.append(np.dot(emb_matrix[edge[0],:],emb_matrix[edge[1],:])) # Inner product for node similarity
      neg.append(0) # actual value (0 for negative)
      
  # Calculate scores
  preds_all = np.hstack([preds_pos, preds_neg])
  labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
  roc_score = roc_auc_score(labels_all, preds_all)
  
  # return roc_score, roc_curve_tuple, ap_score
  print ('--------------------------------')
  print ('AUC ROC Score :   ', round(roc_score,3))
  print ('--------------------------------')



#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Start                                                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
if __name__ == "__main__":
  sys.exit(main())
