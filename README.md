# ThesisFinal
## Explanation Dummy Data
Dummy data: "dummydata.csv" in the Data folder of GLACE.

1) Inside of the GLACE model the python file: "PyGeometric.py" transforms a csv file of house price data into a graph npz file. The saved graph npz file is located in the Data folder, named "your_graph.npz".
2) Call the split function to obtain "your_graph_train.npz". This is necessary to run the train.py file.
3) The code generates an error when trying to calculate the energy, more specifically when trying to use the method tf.gather().


Note: the dummy data currently has "labels" as a variable, this is temporary and should be changed to house prices. It was to check whether the generated error was caused by this.
The error doesn't manifest when running "cora_ml_train.npz".
