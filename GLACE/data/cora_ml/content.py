import numpy as np
import pandas as pd

data = np.load("data/cora_ml/cora_ml_train.npz", allow_pickle=True)
mydata = np.load("data/cora_ml/your_graph_train.npz", allow_pickle=True)

# print(data.files)
# print("attr_indptr:", data["attr_indptr"])
# print("attr_data:", data["attr_data"])
# print("attr_indices:",data["attr_indices"])
# print("attr_shape:",data["attr_shape"])

# print(200*"*")

# print(mydata.files)
# print("attr_indptr:", mydata["attr_indptr"])
# print("attr_data:", mydata["attr_data"])
# print("attr_indices:",mydata["attr_indices"])
# print("attr_shape:",mydata["attr_shape"])
 
df = pd.DataFrame.from_dict({item: [data[item]] for item in data.files}, orient='index')
print(df)
print(100*"*")
df1 = pd.DataFrame.from_dict({item: [mydata[item]] for item in mydata.files}, orient='index')
print(df1)
#import pickle

# Open the pickle file in read-binary mode
#with open('C:/Users/nino/Desktop/Python/GLACE/emb/glace_cora_ml_embedding_first-order.pkl', 'rb') as f:
    # Load the contents of the pickle file into a variable
 #   data = pickle.load(f)

# Print the contents of the pickle file
#print(data)