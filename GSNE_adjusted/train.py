import tensorflow as tf
import argparse
from model import GSNE
from utils import DataUtils, score_link_prediction
import pickle
import time
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random
import copy
from IPython.display import clear_output
import random
from utils import DataUtils

#Below shows the amount of nodes of a POI.
node1_start, node1_end = (0, 50)
node2_start, node2_end = (50, 80)
node3_start, node3_end = (80, 120)
node4_start, node4_end = (120, 153)

seed = 6521
MODEL_ID = 20
tensorboard_path = 'House Price Content List/Tensorboard/' + str(MODEL_ID)
model_path = 'House Price Content List/Model Params/' + str(MODEL_ID)+'/'
embedding_path = '' + str(MODEL_ID)+'/'

def train_test_split_sampled(arrName, pivotColumn, num_of_intervals):
    #arrName = arrName[~np.isnan(arrName).any(axis=1)] This line of code removes all NaN values? But doesn't seem to work for some reason. 
    #print(arrName["price"]) #test that I put here myself.
    col = np.array(arrName[pivotColumn])
    #print(col)
    bins = np.linspace(min(col), max(col), num_of_intervals)
    left = 0
    right = 1
    train = np.array([arrName.values[0]])
    test_set = np.array([arrName.values[0]])
    train = np.delete(train,(0), axis = 0)
    #np.delete(val,(0), axis = 0)
    test_set = np.delete(test_set,(0), axis = 0)
    
    portion_arr = []
    while(right < len(bins)):
      left_val = bins[left]
      right_val = bins[right]
      portion_arr = arrName[arrName[pivotColumn] >= left_val]
      portion_arr = portion_arr[portion_arr[pivotColumn] < right_val]
      if len(portion_arr) < 10 and right !=len(bins) - 1:
        right = right + 1
        continue
      train_temp, test_temp = train_test_split(portion_arr, test_size = 0.2, random_state = seed)
      #print(train_temp.values.shape)
      #print(train.shape)
      train = np.concatenate((train, np.array(train_temp.values)))
      #val_temp, test_temp = train_test_split(val_test_temp, test_size = 0.5)
      test_set = np.concatenate((test_set, np.array(test_temp.values)))

      left = right
      right = right + 1
      portion_arr = []

    return train, test_set

def score_node_classification(features, z, features_test, labels_test, p_labeled=0.8, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.
    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm
    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    '''if p_labeled == 0.8:
        p_labeled = 1 - random.uniform(0.2, 0.8)'''

    p_labeled = 0.5

    # normalizes the input features, is default false.
    if norm:
        features = normalize(features)

    trace = []
    split_train1, split_train2 = None, None
    for seed in range(n_repeat):
        sss = ShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        rfr = RandomForestRegressor(n_jobs=-1)
        rfr.fit(features[split_train], z[split_train])
        predicted = rfr.predict(features[split_test])

        mae = mean_absolute_error(z[split_test], predicted)
        mse = mean_squared_error(z[split_test], predicted)

        rfr = RandomForestRegressor(n_jobs=-1)
        rfr.fit(features, z)
        predicted = rfr.predict(features_test)

        mae2 = mean_absolute_error(labels_test, predicted)
        mse2 = mean_squared_error(labels_test, predicted)

        trace.append((mae, mse, mae2, mse2))

    return np.array(trace).mean(0)


def check_performance(number_of_iter, test_indices):
    ps = pd.read_pickle('C:/Users/nino/Desktop/Python/GSNE/Embeddings/gsne_cora_ml_embedding_graduate_second-order.pkl')
    #pf = pd.read_csv('features_concatenated.csv', sep=',', header=None)
    #raw_features = pf.values[14268:, 1:]
    features_train = np.array([np.array(ps['mu'][k]) for k in range(0, len(ps['mu'])) if k not in test_indices])
    features_test = np.array([np.array(ps['mu'][k]) for k in range(14266, len(ps['mu'])) if k in test_indices])
    labels_df = pd.read_csv('Property_price.csv', delimiter=";")
    #print(labels_df.head(5))
    labels_train = labels_df[~np.isin(labels_df['ID'], test_indices)]['price'].values
    labels_test = labels_df[np.isin(labels_df['ID'], test_indices)]['price'].values
    ''' for k in range(len(features)):
      if np.random.rand() < 0.00:
        print(str(k) + " feats: " +str(features[k]))'''

    mae, mse, mae2, mse2 = score_node_classification(features_train, labels_train, features_test, labels_test, n_repeat = 1) 
    if number_of_iter % 300 == 0 or number_of_iter < 1000:
      print("Embedding Features Results - MAE: "+ str(mae)+" RMSE: "+ str(mse**0.5)+" TMAE: "+ str(mae2)+" TRMSE: "+ str(mse2**0.5))
    return mse**0.5, mae2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='cora_ml')
    parser.add_argument('--model', default='gsne', help='gsne')
    parser.add_argument('--suf', default='')
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--learning_rate', default=0.0002)
    parser.add_argument('--num_batches', type=int, default=600000)
    parser.add_argument('--is_all', default=True)  # train with all edges; no validation or test set
    args = parser.parse_args()
    args.is_all = True if args.is_all == 'True' else False
    train(args)

def train(args):
    """ graph_file = 'Data Tables Step 7 - Processed npzs_cleaned/graph_0_general_attributes.npz'
    graph_file1 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_1_house_region.npz'
    graph_file2 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_2_house_train.npz'
    graph_file3 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_3_property_school.npz'
    graph_file4 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_4_school_train.npz'
    graph_file5 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_5_train_train.npz' """
    import os
    os.chdir("C:/Users/nino/Desktop/Python/ThesisFinal/GSNE")

    graph_file = 'Data.npz' #General graph file
    graph_file1 = 'FINALEDUMMYDATASET1.npz' #graph file house-house
    graph_file2 = 'FINALEDUMMYDATASET2.npz' #graph file house-school

    #Dataset splitting for ensuring inductivity 
    price_file = 'Property_price.csv'
    #The price file contains an id variable and price variable
    df_price = pd.read_csv(price_file, delimiter=";")
    train, tesst = train_test_split_sampled(df_price, 'price', 2)

    train_indices = train[:, 0]
    test_indices = tesst[:, 0]

    # saves the indices that were used to train the embedding and to test.
    np.savetxt("train.txt", train_indices)
    np.savetxt("test.txt", test_indices)
 
    # Normal attribute and graph loaders, added with test indices for ensuring they don't get trained
    # Overall data loader 
    data_loader = DataUtils(graph_file, args.is_all, test_indices=test_indices) #Contains only attribute info 
    data_loader1 = DataUtils(graph_file1, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)
    data_loader2 = DataUtils(graph_file2, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)

    # If attribute data isn't given then the code below won't work, is it then okay to comment it out? 
    suffix = args.proximity
    args.X1 = data_loader.X1 if args.suf != 'oh' else sp.identity(data_loader1.X1.shape[0])
    args.X2 = data_loader.X2 if args.suf != 'oh' else sp.identity(data_loader2.X2.shape[0])
    
    m = args.model
    name = m + '_' + args.name
    
    if 'gsne' == m:
        model = GSNE(args)
    else:
        raise Exception("Only gsne available")

    writer = tf.compat.v1.summary.FileWriter(tensorboard_path)
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        #saver.restore(sess, model_path+"model_graduate_best.ckpt")
        writer.add_graph(sess.graph)
        print('-------------------------- ' + m + ' --------------------------')
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
    
        tf.compat.v1.global_variables_initializer().run()
        sampling_time, training_time = 0, 0

        previous_best = 9999999999

        for b in range(args.num_batches):
            if b%35000 == 0:
                clear_output()
            t1 = time.time()
            # Change the sampler to use only two graph files
            # The code below of the original das et al. graph is unclear, seems to have inconsistencies.
            # In essence it should fetch the next batch from the different data loaders.
            if b % 2 == 0:
                u_i, u_j, label, node_type1, node_type2 = data_loader1.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            else:
                u_i, u_j, label, node_type1, node_type2 = data_loader2.fetch_next_batch(batch_size=args.batch_size, K=args.K)
                
            #u_i, u_j, label, w = data_loader.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.node_type1 : node_type1, model.node_type2 : node_type2}

            t2 = time.time()
            sampling_time += t2 - t1

            #loss, _ = sess.run([model.loss, model.accum_ops], feed_dict=feed_dict)
            loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
            ''' if (b%5 == 4 or True):
              sess.run(model.train_step) '''
            training_time += time.time() - t2

            if b%5 < 5:
              s = sess.run(model.merged_summary, feed_dict = feed_dict)
              writer.add_summary(s, b)
              writer.flush()

            if b % 5000 < 5:
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

                sampling_time, training_time = 0, 0

            if b!=0 and ( b % 50 == 0 or b == (args.num_batches - 1)):
                if m == 'gsne':
                    
                    mu1, sigma1 = sess.run([model.embedding1, model.sigma1])
                    mu2, sigma2 = sess.run([model.embedding2, model.sigma2])

                    mu = copy.deepcopy(mu1)
                    mu[node1_start: node1_end] = mu1[node1_start: node1_end]
                    mu[node2_start: node2_end] = mu2[node2_start: node2_end]


                    sigma = copy.deepcopy(sigma1)
                    sigma[node1_start: node1_end] = sigma1[node1_start: node1_end]
                    sigma[node2_start: node2_end] = sigma2[node2_start: node2_end]

                    
                    #The code below stores the file object as a pickle.
                    #It is then opened in a wb (write-binary) mode.
                    #In essence, we're saving the mu and sigma here of the model, which represents its performance.
                    pickle.dump({'mu': data_loader.embedding_mapping(mu),
                                 'sigma': data_loader.embedding_mapping(sigma)},
                                open('C:/Users/nino/Desktop/Python/GSNE/Embeddings/' + '%s%s_embedding_graduate_%s.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                    
                    save_path = saver.save(sess, "C:/Users/nino/Desktop/Python/GSNE" + "model_graduate.ckpt")
                    
                    curr_mae, val_mae = check_performance(number_of_iter = b, test_indices = test_indices)

                    if curr_mae < previous_best:
                      previous_best = curr_mae
                      print("new best result train rmse: "+str(curr_mae) + " test mae:"+str(val_mae))
                      pickle.dump({'mu': data_loader.embedding_mapping(mu),
                                 'sigma': data_loader.embedding_mapping(sigma)},
                                open('C:/Users/nino/Desktop/Python/GSNE/Embeddings/' + '%s%s_embedding_graduate_%s_best.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                      save_path = saver.save(sess, model_path + "model_graduate_best.ckpt")
                else:
                    raise Exception("only GSNE supported")

if __name__ == '__main__':
    main()

                