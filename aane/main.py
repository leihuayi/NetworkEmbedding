#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
import time
import argparse

from model import AANE

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   AANE                                                                                        #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def aane(args):
	start_time = time.time()

	# Init graph
	g_npz = sp.load_npz(args.input)
	G = sp.csc_matrix(g_npz) # column slicing is operated on csc matrices
	
	n = G.shape[0]
	print("\nNumber of nodes: {}".format(n))

	indexList = np.random.randint(25, size=n)  # 5-fold cross-validation indices

	Group1 = []
	Group2 = []
	[Group1.append(x) for x in range(0, n) if indexList[x] <= 20]  # 2 for 10%, 5 for 25%, 20 for 100% of training group
	[Group2.append(x) for x in range(0, n) if indexList[x] >= 21]  # test group
	n1 = len(Group1)  # num of nodes in training group
	n2 = len(Group2)  # num of nodes in test group
	G_matrix = G[Group1+Group2, :][:, Group1+Group2]

	# Launch AANE
	print("\nInitialization ...")
	H_Net = AANE(G_matrix, G_matrix, args).run()

	# Save to output file
	print("----- Total time {:.2f}s -----".format(time.time() - start_time))
	pickle.dump(H_Net, open(args.output, 'wb'))
	return

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   MAIN                                                                                        #
#   Parse input args                                                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', nargs='?', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--iter', default=5)
  parser.add_argument('--dimension', default=128)
  parser.add_argument('--lambd', default=0.05)
  parser.add_argument('--rho', default=5)

  args = parser.parse_args()

  aane(args)

if __name__ == "__main__":
  main()