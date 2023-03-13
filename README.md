# ThesisFinal

The jupyter notebook "DataPipeline.ipynb" currently shows all the steps to transforming a csv file to a graph npz file.

Data requirements:
1) A csv file with latitude and longitude coordinates to create nodes
2) A csv file with id and price
Currently performed on the King's Count dataset.


The output is a graph npz file compatible with the GLACE and Graph2Gauss models.


Ideas:
- Create a Python file with a function that when given a csv file it automatically transforms it to a graph npz file.

