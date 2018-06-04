#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

import graph
from gensim.models import Word2Vec

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   DEEPWALK                                                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def main():
  parser = ArgumentParser("deepwalk",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')

  parser.add_argument('--format', default='adjlist')

  parser.add_argument('--input', nargs='?', required=True)

  parser.add_argument('--output', required=True)

  parser.add_argument("-l", "--log", dest="log", default="INFO")

  parser.add_argument('--dic-network-name', default='network')

  parser.add_argument('--number-walks', default=10, type=int)

  parser.add_argument('--walk-length', default=40, type=int)

  parser.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')

  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)


  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.dic_network_name)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)


  ## Info about the walks
  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  ## Do walks
  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks, path_length=args.walk_length, alpha=0)

  ## Train on these walks
  print("Training...")
  model = Word2Vec(walks, size=args.representation_size, window=5, min_count=0, sg=1, hs=1, workers=1)

  model.wv.save_word2vec_format(args.output)

if __name__ == "__main__":
  main()