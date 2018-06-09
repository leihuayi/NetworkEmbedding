#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import tensorflow as tf

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   LINE MODEL                                                                                   #
#   A subclass to allow more customization of the Word2Vec internals                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
class Line:
    def __init__(self, args):
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])

        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.embedding = tf.get_variable('target_embedding', [args.num_of_nodes, args.embedding_dim],initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)

        # First order priority
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        # Second order priority :  use context
        elif args.proximity == 'second-order':
            self.context_embedding = tf.get_variable('context_embedding', [args.num_of_nodes, args.embedding_dim], initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

        # Compute u_j^T * u_i
        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        # Compute loss = log sig (u_j^T * u_i)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))

        # Minimize loss
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


