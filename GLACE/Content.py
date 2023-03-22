import numpy as np

data=np.load("C:/Users/nino/Desktop/Python/ThesisFinal/GLACE/data/cora_ml/FINALEDUMMYDATASET2.npz")
data1=np.load("C:/Users/nino/Desktop/Python/ThesisFinal/GLACE/data/cora_ml/FINALEDUMMYDATASET3.npz")
print(data.files)
print(data1.files)
print(data)
print(data1)
print("OWN GENERATE")
for item in data.files:
    print(item + ' shape: ', data[item])
print("YARNE")

for item in data.files:
    print(item + ' shape: ', data1[item])
# Open the pickle file in read-binary mode
# with open('C:/Users/nino/Desktop/Python/GLACE/emb/glace_cora_ml_embedding_first-order.pkl', 'rb') as f:
#     # Load the contents of the pickle file into a variable
#     data = pickle.load(f)

# # Print the contents of the pickle file
# print(data)
