import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# https://github.com/nathasha-naranpanawa/COMP4702_2024/blob/main/PracW3.ipynb

# @title Classification
# load the dataset
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv", names=["X1", "X2", "Y"])


# KNN evaulation

# Decision tree evaluation

# Neural Network evaluation