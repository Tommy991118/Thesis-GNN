# Predicting house prices using geospatial network embeddings incorporating POIs
This Github repository outlines our conducted research on improving the predictive performance of house price predictions by utilizing Gaussian network embeddings. The repository contains code ranging from data preparation to model evaluation. The [GSNE](https://arxiv.org/pdf/2009.00254.pdf) and [GLACE]([https://arxiv.org/pdf/2009.00254.pdf](https://arxiv.org/pdf/1912.00536.pdf))
models were utilized to create the Gaussian network embeddings. Some adjustments have been made to these models to make them compatible with the King County datasets. We also decided to separate the Jupyter notebook encompassing the data preprocessing pipeline and the house price prediction models. This choice was made to facilitate future researchers in a user-friendly way to make adjustments to the different processes. 

# Requirements
```
Python 3.6.13

Networkx 2.5.1

Scikit-learn 0.24.2

Tensorflow 1.15.0

Scipy 1.5.4

Matplotlib 3.1.3

NumPy 1.21.5

Pandas 1.3.5
```

# Process of using network embeddings for house price predictions 

![Visual representation of using network embeddings for house price predictions](Figures/Overview_process.png)

## **1. Data Preprocessing Pipeline** 
The Jupyter notebook "DataPipeline.ipynb" represents the data pipeline of transforming tabular data to graph files. The current pipeline is compatible with GLACE and GSNE_adjusted models. The notebook utilized open-source datasets of King county: "kc_school_data.csv" and "kc_house_data.csv", respresenting respectively data on schools and houses in King County. The input data only has as requirement that it must contain variables concerning its geographical coordinates (latitude and longitude) for the house or point of interest (POI). 

<p align="center">
  <img src="Figures/Overview_GraphFiles.png" alt="Visual representation of the data pipeline" />
</p>
2) After generating the graph .npz files by executing the "DataPipeline.ipynb" notebook, a network embedding model must be executed. The GLACE folder contains all the necessary components to seamlessly run the model after the execution of the aforementioned notebook. For the data used in this repository the folder of GSNE_adjusted should be utilized, since that code is adjusted to be compatible for using two points of interest.


3) Ultimately, the "Modelling_embedding.ipynb" notebook should be executed to conduct a benchmark analysis of the performance of the machine learning models on the generated Gaussian network embedding. 








