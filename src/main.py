from src.helper import helper
import numpy as np
import networkx as nx
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
obj = helper()
obj.create_env()
print("environment created")
#in case of julia path issue, uncomment jl.install()

dataset="MUTAG"
#load dataset online
graphs,y= obj.return_dataset(dataset)
#load dataset from file
# graphs, y = obj.load_data(dataset)
# graphs, labels = obj.load_data_from_numpy()
bins = 100
print("total graphs in the dataset:",len(graphs))
scores,max_val = obj.GenerateAllApproxEmb(graphs)
scaled_data, max_val = obj.remove_outliers(scores)
emb = obj.generateHistogram(scaled_data, max_val,bins)
print("embedding shape:", emb.shape)
