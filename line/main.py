#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import tensorflow as tf
import numpy as np
import argparse
from model import Line
from graph import Graph
import pickle
import time

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   LINE                                                                                        #
#   Use Line with Tensorflow                                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def line(args):
    start_time = time.time()

    # Init graph
    graph = Graph(graph_file=args.input)

    args.num_of_nodes = graph.num_of_nodes
    args.iter = int(args.iter)
    print("\nNumber of nodes: {}".format(graph.num_of_nodes))

    model = Line(args)
    with tf.Session() as sess:
        print(args)
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0

        print('Tensorflow iterations :')
        for i in range(args.iter):
            t1 = time.time()
            u_i, u_j, label = graph.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1

            if i % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - i / args.iter)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('Iteration : %d/%d\t loss : %f\t sampling_time : %0.2f\t training_time : %0.2f\t' % (i,args.iter, loss, sampling_time, training_time))
                sampling_time, training_time = 0, 0

            if i == (args.iter - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                
                # Save to output file
                print("----- Total time {:.2f}s -----".format(time.time() - start_time))
                pickle.dump(graph.embedding_mapping(normalized_embedding), open(args.output, 'wb'))

    return


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   MAIN                                                                                        #
#   Parse input args                                                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dimension', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--iter', default=200)

    args = parser.parse_args()
    line(args)

if __name__ == '__main__':
    main()