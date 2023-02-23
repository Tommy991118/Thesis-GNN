import numpy as np

data=np.load("data/cora_ml/cora_ml.npz")

print(data.files)
print(data["attr_indptr"])

import pickle

# Open the pickle file in read-binary mode
with open('C:/Users/nino/Desktop/Python/GLACE/emb/glace_cora_ml_embedding_first-order.pkl', 'rb') as f:
    # Load the contents of the pickle file into a variable
    data = pickle.load(f)

# Print the contents of the pickle file
print(data)
