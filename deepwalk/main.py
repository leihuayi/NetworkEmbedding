#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import os
import sys
import random
import argparse
import time
from collections import Counter
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab

import graph
from model import Skipgram


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   COUNT_WORDS                                                                                 #
#    Helper function for Skipgram. Returns dictionary of the times each vertex appear in walks  #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def count_words(walks):
  c = Counter()
  for words in walks:
    c.update(words)
  return c


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   DEEPWALK                                                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def deepwalk(args):
  start_time = time.time()

  # Init graph
  G = graph.Graph(args.input)

  # Info about the walks
  total_walks = G.num_of_nodes * args.num_walks
  data_size = total_walks * args.walk_length

  print("\nNumber of nodes: {}".format(G.num_of_nodes))
  print("Total number of walks: {}".format(total_walks)) # Number of walks starting from each node
  print("Data size (walks*length): {}\n".format(data_size))

  # Create the random walks and store them in walks list
  print("Generate walks ...")
  walks = G.build_deep_walks(num_paths=args.num_walks, path_length=args.walk_length)

  # Apply model to each walk = sentence
  print("Applying %s on walks ..."% args.model)
  if args.model == 'skipgram' :
    vertex_counts = count_words(walks) # dictionary of the times each vertex appear in walks
    model = Skipgram(sentences=walks, vocabulary_counts=vertex_counts,size=args.dimension,window=5, min_count=0, trim_rule=None, workers=cpu_count(), iter=args.iter)
  else :
    if args.model == 'word2vec':
      model = Word2Vec(walks, size=args.dimension, window=5, min_count=0, sg=1, hs=1, workers=cpu_count())
    else:
      raise Exception("Unknown model: '%s'.  Valid models: 'word2vec', 'skipgram'" % args.model)

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
  parser = argparse.ArgumentParser()
  parser.add_argument('--format', default='mat')
  parser.add_argument('--input', nargs='?', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--num-walks', default=20, type=int)
  parser.add_argument('--walk-length', default=20, type=int)
  parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
  parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
  parser.add_argument('--model', default='word2vec', help='Type of model to apply on walks (word2vec/skipgram)')

  args = parser.parse_args()

  deepwalk(args)

if __name__ == "__main__":
  main()