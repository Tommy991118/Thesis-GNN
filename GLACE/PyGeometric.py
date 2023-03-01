import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OrdinalEncoder
# import torch_geometric
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

# Load CSV file"
df = pd.read_csv('data/cora_ml/dummydata.csv', delimiter=";")
print(df.head())

# Create nodes based on unique combinations of latitude and longitude
node_ids = df['id'].values
lat_lon = df[['latitude', 'longitude']].values #combination of latitude and longitude coordinates. 
unique_lat_lon, node_indices = np.unique(lat_lon, axis=0, return_inverse=True)
num_nodes = len(unique_lat_lon)

########################## PREPROCESSING COLUMN "some features"
# Create dictionary to map node ids to node indices
node_id_to_index = dict(zip(node_ids, range(num_nodes))) # Potentiall change it such that node_ids are zipped with coordinates.

# extract the "some features" column
some_features = df['some features'].values.reshape(-1, 1)

# create an instance of the ordinal encoder
encoder = OrdinalEncoder()

# fit and transform the "some features" column
encoded_features = encoder.fit_transform(some_features)

# replace the original "some features" column with the encoded values
df['some features'] = encoded_features
###############################################
 
print(df)
# Create sparse feature matrix. Extracts the values of the features from the dataframe.
features = sp.csr_matrix(df['some features'].values)

# Create sparse adjacency matrix
# Can't create it directly from a graph, because here we're converting it from a csv file.
row = node_indices[:-1]
col = node_indices[1:]
adj_matrix = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))

edges = np.array(list(node_id_to_index.items()))
print(edges)

# Below documentation shows that the attribute features like data/indices/indptr can be extracted from a compressed Sparse row matrix

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

# data = Data(x=features, edge_index=)

# Save as npz file
np.savez('data/cora_ml/your_graph.npz', edges=edges, adj_data=adj_matrix.data, adj_indices=adj_matrix.indices, adj_indptr=adj_matrix.indptr,
         adj_shape=adj_matrix.shape, attr_data=features.data, attr_indices=features.indices, attr_indptr=features.indptr,
         attr_shape=features.shape, labels=np.asarray(df["labels"]))

# file = torch_geometric.parse_npz("your_graph.npz")

# print(file.files)