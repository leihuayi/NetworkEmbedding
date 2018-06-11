'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import scipy.sparse as sp


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimension, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   MAIN                                                                                        #
#   Parse input args                                                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--input', nargs='?', required=True)
	parser.add_argument('--output', nargs='?', required=True)
	parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
	parser.add_argument('--walk-length', type=int, default=40)
	parser.add_argument('--num-walks', type=int, default=10)
	parser.add_argument('--window-size', type=int, default=10,help='Context size for optimization. Default is 10.')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	args = parser.parse_args()

	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	grap = sp.load_npz(args.input)
	nx_G = nx.from_scipy_sparse_matrix(grap)
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	main()
