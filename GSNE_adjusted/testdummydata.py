import networkx as nx
import scipy.sparse as sp
import numpy as np

# Create a graph with five nodes representing the latitude and longitude coordinates of POIs (schools, trains, etc)
# When using real data take into consideration the different POIs between which we want to create edges.


G = nx.Graph()
G.add_node((40.712776, -74.005974), attr_data1 = 1)
G.add_node((37.7749, -122.4194), attr_data1 = 2)
G.add_node((51.5074, -0.1278), attr_data1 = 3)
G.add_node((48.8566, 2.3522), population = 100)
G.add_node((52.5200, 13.4050), population = 300)
G.add_node((41.8781, -87.6298), ranking = 4)
G.add_node((34.0522, -118.2437), ranking = 5)
G.add_node((39.9526, -75.1652), ranking = 6)
G.add_node((45.5231, -122.6765), time = 3)
G.add_node((37.3362, -121.8906), time = 2)

# Add edges between the nodes
# Here we don't take the threshold delta into consideration when constructing the graph.
# When using an existing dataset we should look into how to choose a threshold.
G.add_edges_from([((40.712776, -74.005974), (37.7749, -122.4194)),
                  ((37.7749, -122.4194), (51.5074, -0.1278)),
                  ((51.5074, -0.1278), (48.8566, 2.3522)),
                  ((48.8566, 2.3522), (52.5200, 13.4050)),
                  ((52.5200, 13.4050), (40.712776, -74.005974)),
                  ((41.8781, -87.6298), (39.9526, -75.1652)),
                  ((34.0522, -118.2437), (45.5231, -122.6765)),
                  ((45.5231, -122.6765), (37.3362, -121.8906)),
                  ((37.3362, -121.8906), (41.8781, -87.6298)),
                  ((41.8781, -87.6298), (34.0522, -118.2437))])

# Convert the graph to a CSR matrix
adj_matrix = nx.adjacency_matrix(G)
csr_matrix = sp.csr_matrix(adj_matrix)

# Extract the CSR data arrays
adj_data = csr_matrix.data
adj_indices = csr_matrix.indices
adj_indptr = csr_matrix.indptr
adj_shape = csr_matrix.shape

# Define the edges as a list of tuples
edges = list(G.edges())

# Save the graph as an .npz file
np.savez('graph1.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)
np.savez('graph2.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)
np.savez('graph3.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)
np.savez('graph4.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)
np.savez('graph5.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)
np.savez('Data.npz', adj_data=adj_data, adj_indices=adj_indices,
         adj_indptr=adj_indptr, adj_shape=adj_shape, edges=edges)

# Print the CSR data arrays
print(adj_data)
print(adj_indices)
print(adj_indptr)
print(adj_shape)
