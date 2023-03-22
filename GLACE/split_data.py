import numpy as np
import argparse
from utils import train_val_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='cora_ml')
    parser.add_argument('--p_val', type=float, default=0.05)
    parser.add_argument('--p_test', type=float, default=0.15)
    args = parser.parse_args()
    split(args)
# Can't use this function if the dataset is too small or too dense.
# Make sure that the dataset is sparse to use code below, otherwise can use the regular stratified train test split I think.
def split(args):
    graph_file = 'C:/Users/nino/Desktop/Python/ThesisFinal/GLACE/data/cora_ml/FINALEDUMMYDATASET.npz'
    #graph_file = 'data/%s/%s.npz' % (args.name, args.name)

    A, X, labels, val_edges, val_ground_truth, test_edges, test_ground_truth = train_val_test_split(graph_file, p_val=args.p_val, p_test=args.p_test)
    
    # print("INPUT LABELS:", labels)
    # print("val_ground_truth LABELS:", val_ground_truth)
    
    np.savez(("GLACE/data/cora_ml/FINALEDUMMYDATASET_train.npz"), adj_data=A.data, adj_indices=A.indices,
             adj_indptr=A.indptr, adj_shape=A.shape, attr_data=X.data, attr_indices=X.indices,
             attr_indptr=X.indptr, attr_shape=X.shape, labels=labels, val_edges=val_edges,
             val_ground_truth=val_ground_truth, test_edges=test_edges,
             test_ground_truth=test_ground_truth)
    # test_edges = np.vstack((data_loader.test_edges.T, data_loader.test_ground_truth)).T.astype(np.int32)
    # np.savez('data/%s/%s_train_test.npz' % (args.name, args.name), train_edges=test_edges, test_edges=test_edges)
    print('%s train data saved')


if __name__ == '__main__':
    main()
