#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import argparse
import time
from gensim.models import Word2Vec
from multiprocessing import cpu_count

import graph

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   NODE2VEC                                                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def node2vec(args):
	start_time = time.time()

	# Init graph
	G = graph.Graph(args.input, args.p, args.q)

	total_walks = G.G.number_of_nodes() * args.num_walks
	data_size = total_walks * args.walk_length

	print("\nNumber of nodes: {}".format(G.G.number_of_nodes()))
	print("Total number of walks: {}".format(total_walks)) # Number of walks starting from each node
	print("Data size (walks*length): {}\n".format(data_size))

	# Create the random walks and store them in walks list
	print("Generate walks ...")

	G.preprocess_transition_probs()
	walks = G.build_node2vec_walks(args.num_walks, args.walk_length)

	# Feed to walks to Word2Vec model
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimension, window=5, min_count=0, sg=1, workers=cpu_count(), iter=args.iter)

  	# Save to output file
	print("----- Total time {:.2f}s -----".format(time.time() - start_time))
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
	parser.add_argument('--walk-length', type=int, default=40)
	parser.add_argument('--num-walks', type=int, default=10)
	parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter')
	parser.add_argument('--q', type=float, default=1, help='Input hyperparameter')
	args = parser.parse_args()

	node2vec(args)

if __name__ == "__main__":
	main()
