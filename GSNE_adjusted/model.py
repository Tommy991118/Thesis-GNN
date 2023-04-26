#FIRST ORDER PROXIMITY

import tensorflow as tf 
import scipy.sparse as sp 
import numpy as np 
#import cyclic_learning_rate.clr as clr
#from keras.callbacks import *
#from clr_callback import *
from torch.optim.lr_scheduler import CyclicLR
#from keras.optimizers import Adam

seed = 42

def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype = np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape

class GSNE:
    def __init__(self, args):
        #tf.set_random_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.X1 = tf.SparseTensor(*sparse_feeder(args.X1))
        self.X2 = tf.SparseTensor(*sparse_feeder(args.X2))

        self.N1, self.D1 = args.X1.shape
        self.N2, self.D2 = args.X2.shape

        self.L = args.embedding_dim

        #PLEASE ENSURE THE LAST LAYER DIMENSION IS SAME FOR EVERYONE

        self.n_hidden1 = [6, 12, 28]
        self.n_hidden2 = [10, 16, 28]
        self.n_hidden3 = [10, 16, 28]
        self.n_hidden4 = [42, 36, 28]

        '''self.n_hidden1 = [14]
        self.n_hidden2 = [14]
        self.n_hidden3 = [14]
        self.n_hidden4 = [14]'''

        self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.node_type1 = tf.compat.v1.placeholder(name='node_type1', dtype=tf.int32, shape = ())
        self.node_type2 = tf.compat.v1.placeholder(name='node_type2', dtype=tf.int32, shape = ())

        self.__create_model(args.proximity)
        self.val_set = False
        
        tf.compat.v1.train.create_global_step()

        # softmax loss
        
        self.energy = -self.energy_kl(self.u_i, self.u_j, args.proximity, self.node_type1, self.node_type2)
        self.loss = -tf.reduce_mean(tf.math.log_sigmoid(self.label * self.energy))
        tf.compat.v1.summary.scalar('loss', self.loss)
        print(args.learning_rate)

        '''for cyclic learning rate'''
        global_step = tf.compat.v1.train.get_global_step() #Is a tensor that keeps track of the taken steps which can be used for tensorboard.
        learning_rate = tf.compat.v1.train.exponential_decay((1e-9), global_step=global_step,decay_steps=10, decay_rate=1.04)

        #learning_rate = clr.cyclic_learning_rate(global_step=global_step, learning_rate=1e-4,
                         #max_lr=19e-5,
                         #step_size=100, mode='exp_range', gamma = 0.99999)

        original_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)
        tf.compat.v1.summary.scalar("current_step",global_step)


        ###########################################################


        #original_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)

        

        ''' tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = self.optimizer.compute_gradients(self.loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        #After getting all the gradients in the five steps, we calculate the train step
        self.train_step = self.optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
 '''
        self.train_op = self.optimizer.minimize(self.loss, global_step = global_step)
        self.merged_summary = tf.compat.v1.summary.merge_all()
        

    def __create_model(self, proximity):
        w_init = tf.contrib.layers.xavier_initializer
        #w_init = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
        #w_init = tf.keras.initializers.random_normal
        sizes1 = [self.D1] + self.n_hidden1
        sizes2 = [self.D2] + self.n_hidden2
        # sizes3 = [self.D3] + self.n_hidden3
        # sizes4 = [self.D4] + self.n_hidden4

        #feature 1
        TRAINABLE = True
        with tf.name_scope("School"):
          for i in range(1, len(sizes1)):
             with tf.name_scope("enc{}".format(i)):
              W = tf.compat.v1.get_variable(name='W1{}'.format(i), shape=[sizes1[i - 1], sizes1[i]], dtype=tf.float32,
                                  initializer=w_init(), trainable = TRAINABLE)
              b = tf.compat.v1.get_variable(name='b1{}'.format(i), shape=[sizes1[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)

              if i == 1:
                  encoded1 = tf.sparse.sparse_dense_matmul(self.X1, W) + b
              else:
                  encoded1 = tf.matmul(encoded1, W) + b

              encoded1 = tf.nn.relu(encoded1)

              tf.compat.v1.summary.histogram('Weight', W)
              tf.compat.v1.summary.histogram('bias', b)
              tf.compat.v1.summary.histogram('activations', encoded1)


        #encoded1 = tf.Print(encoded1, [encoded1], message = "feature 1 encoder triggered")

        #feature 2
        TRAINABLE = True
        with tf.name_scope("House"):
          for i in range(1, len(sizes2)):
            with tf.name_scope("enc{}".format(i)):
              W = tf.compat.v1.get_variable(name='W2{}'.format(i), shape=[sizes2[i - 1], sizes2[i]], dtype=tf.float32,
                                  initializer=w_init(), trainable = TRAINABLE)
              b = tf.compat.v1.get_variable(name='b2{}'.format(i), shape=[sizes2[i]], dtype=tf.float32, initializer=w_init(), trainable = TRAINABLE)

              if i == 1:
                  encoded2 = tf.sparse.sparse_dense_matmul(self.X2, W) + b
              else:
                  encoded2 = tf.matmul(encoded2, W) + b

              encoded2 = tf.nn.relu(encoded2)

              tf.compat.v1.summary.histogram('Weight', W)
              tf.compat.v1.summary.histogram('bias', b)
              tf.compat.v1.summary.histogram('activations', encoded2)

        #encoded2 = tf.Print(encoded2, [encoded2], message = "feature 2 encoder triggered")


        W_mu1 = tf.compat.v1.get_variable(name='W_mu1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
        b_mu1 = tf.compat.v1.get_variable(name='b_mu1', shape=[40], dtype=tf.float32, initializer=w_init())

        W_mu2 = tf.compat.v1.get_variable(name='W_mu2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
        b_mu2 = tf.compat.v1.get_variable(name='b_mu2', shape=[self.L], dtype=tf.float32, initializer=w_init())
        
        embedding1_t = tf.nn.relu(tf.matmul(encoded1, W_mu1) + b_mu1)
        self.embedding1 = tf.matmul(embedding1_t, W_mu2) + b_mu2

        embedding2_t = tf.nn.relu(tf.matmul(encoded2, W_mu1) + b_mu1)
        self.embedding2 = tf.matmul(embedding2_t, W_mu2) + b_mu2

        W_sigma1 = tf.compat.v1.get_variable(name='W_sigma1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
        W_sigma2 = tf.compat.v1.get_variable(name='W_sigma2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())

        b_sigma1 = tf.compat.v1.get_variable(name='b_sigma1', shape=[40], dtype=tf.float32, initializer=w_init())
        b_sigma2 = tf.compat.v1.get_variable(name='b_sigma2', shape=[self.L], dtype=tf.float32, initializer=w_init())

        log_sigma1t = tf.nn.relu(tf.matmul(encoded1, W_sigma1) + b_sigma1)
        log_sigma1 = tf.matmul(log_sigma1t, W_sigma2) + b_sigma2
        self.sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
        #self.sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14

        log_sigma2t = tf.nn.relu(tf.matmul(encoded2, W_sigma1) + b_sigma1)
        log_sigma2 = tf.matmul(log_sigma2t, W_sigma2) + b_sigma2
        self.sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
        #self.sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

        if proximity == 'second-order':
            #feature 1

            for i in range(1, len(sizes1)):
                W = tf.compat.v1.get_variable(name='W_ctx1{}'.format(i), shape=[sizes1[i - 1], sizes1[i]], dtype=tf.float32,
                                    initializer=w_init())
                b = tf.compat.v1.get_variable(name='b_ctx1{}'.format(i), shape=[sizes1[i]], dtype=tf.float32, initializer=w_init())

                if i == 1:
                    encoded1 = tf.sparse.sparse_dense_matmul(self.X1, W) + b
                else:
                    encoded1 = tf.matmul(encoded1, W) + b

                encoded1 = tf.nn.relu(encoded1)

            #feature 2

            for i in range(1, len(sizes2)):
                W = tf.compat.v1.get_variable(name='W_ctx2{}'.format(i), shape=[sizes2[i - 1], sizes2[i]], dtype=tf.float32,
                                    initializer=w_init())
                b = tf.compat.v1.get_variable(name='b_ctx2{}'.format(i), shape=[sizes2[i]], dtype=tf.float32, initializer=w_init())

                if i == 1:
                    encoded2 = tf.sparse.sparse_dense_matmul(self.X2, W) + b
                else:
                    encoded2 = tf.matmul(encoded2, W) + b

                encoded2 = tf.nn.relu(encoded2)


            W_mu1 = tf.compat.v1.get_variable(name='W_mu_ctx1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
            b_mu1 = tf.compat.v1.get_variable(name='b_mu_ctx1', shape=[40], dtype=tf.float32, initializer=w_init())
            
            W_mu2 = tf.compat.v1.get_variable(name='W_mu_ctx2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
            b_mu2 = tf.compat.v1.get_variable(name='b_mu_ctx2', shape=[self.L], dtype=tf.float32, initializer=w_init())
            
            ctx_mu1_t = tf.nn.relu(tf.matmul(encoded1, W_mu1) + b_mu1)
            self.ctx_mu1 = tf.matmul(ctx_mu1_t, W_mu2) + b_mu2
            
            ctx_mu2_t = tf.nn.relu(tf.matmul(encoded2, W_mu1) + b_mu1)
            self.ctx_mu2 = tf.matmul(ctx_mu2_t, W_mu2) + b_mu2

            W_sigma1 = tf.compat.v1.get_variable(name='W_sigma_ctx1', shape=[sizes1[-1], 40], dtype=tf.float32, initializer=w_init())
            W_sigma2 = tf.compat.v1.get_variable(name='W_sigma_ctx2', shape=[40, self.L], dtype=tf.float32, initializer=w_init())
            
            b_sigma1 = tf.compat.v1.get_variable(name='b_sigma_ctx1', shape=[40], dtype=tf.float32, initializer=w_init())
            b_sigma2 = tf.compat.v1.get_variable(name='b_sigma_ctx2', shape=[self.L], dtype=tf.float32, initializer=w_init())
            
            log_sigma1t = tf.nn.relu(tf.matmul(encoded1, W_sigma1) + b_sigma1)
            log_sigma1 = tf.matmul(log_sigma1t, W_sigma2) + b_sigma2
            self.ctx_sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
            #self.ctx_sigma1 = tf.nn.sigmoid(log_sigma1) + 1 + 1e-14
			
            log_sigma2t = tf.nn.relu(tf.matmul(encoded2, W_sigma1) + b_sigma1)
            log_sigma2 = tf.matmul(log_sigma2t, W_sigma2) + b_sigma2
            self.ctx_sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14
            #self.ctx_sigma2 = tf.nn.sigmoid(log_sigma2) + 1 + 1e-14

            #########################################DEEPER MODEL END##################################


    def energy_kl(self, u_i, u_j, proximity, node_type1, node_type2):
        def f1():
          print("f1") 
          return tf.gather(self.embedding1, u_i), tf.gather(self.sigma1, u_i)
        def f2(): 
          print("f2")
          return tf.gather(self.embedding2, u_i), tf.gather(self.sigma2, u_i)
        def f3(): 
          print("f3")
          return tf.gather(self.embedding3, u_i), tf.gather(self.sigma3, u_i)
        def f4(): 
          print("f4")
          return tf.gather(self.embedding4, u_i), tf.gather(self.sigma4, u_i)

        # The helper functions below are never called.
        def f5(): 
          print("f5")
          return tf.gather(self.ctx_mu1, u_j), tf.gather(self.ctx_sigma1, u_j)
        def f6(): 
          print("f6")
          return tf.gather(self.ctx_mu2, u_j), tf.gather(self.ctx_sigma2, u_j)
        def f7(): 
          print("f7")
          return tf.gather(self.ctx_mu3, u_j), tf.gather(self.ctx_sigma3, u_j)
        def f8(): 
          print("f8")
          return tf.gather(self.ctx_mu4, u_j), tf.gather(self.ctx_sigma4, u_j)

        def f9():
          print("f9") 
          return tf.gather(self.embedding1, u_j), tf.gather(self.sigma1, u_j)
        def f10(): 
          print("f10")
          return tf.gather(self.embedding2, u_j), tf.gather(self.sigma2, u_j)
        def f11(): 
          print("f11")
          return tf.gather(self.embedding3, u_j), tf.gather(self.sigma3, u_j)
        def f12(): 
          print("f12")
          return tf.gather(self.embedding4, u_j), tf.gather(self.sigma4, u_j)

        mu_i, sigma_i = tf.case([(tf.equal(node_type1, 0), f1), (tf.equal(node_type1, 1), f2),
                              (tf.equal(node_type1, 2), f3), (tf.equal(node_type1, 3), f4)],
         default=None, exclusive=True)
        
        mu_j, sigma_j = tf.case([(tf.equal(node_type2, 0), f9), (tf.equal(node_type2, 1), f10),
                              (tf.equal(node_type2, 2), f11), (tf.equal(node_type2, 3), f12)],
         default=None, exclusive=True)

        sigma_ratio = sigma_j / sigma_i
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.math.log(sigma_ratio + 1e-11), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)

        ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        sigma_ratio = sigma_i / sigma_j
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.math.log(sigma_ratio + 1e-11), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

        ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        kl_distance = 0.5 * (ij_kl + ji_kl)

        return kl_distance
