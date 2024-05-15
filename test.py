import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape",	"wing_vein_length",	"wing_area"
])

DATA_ITEM_1 = "Longitude"
DATA_ITEM_2 = "Latitude"

# Select features and target
X = data[[DATA_ITEM_1, DATA_ITEM_2]]  # Using only Thorax_length and wing_loading
y = data["Species"].astype(int)  # Convert target labels to integers

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Evaluate the classifier
y_train_pred = knn.predict(X_train)
train_misclassification_rate = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy Rate: {train_misclassification_rate:.2f}")

y_test_pred = knn.predict(X_test)
test_misclassification_rate = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy Rate: {test_misclassification_rate:.2f}")

# Plotting the decision boundary
h = 0.01  # Step size in the mesh
x_min, x_max = X_train[DATA_ITEM_1].min() - 1, X_train[DATA_ITEM_1].max() + 1
y_min, y_max = X_train[DATA_ITEM_2].min() - 1, X_train[DATA_ITEM_2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh using the model
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.astype(int)  # Ensure numerical dtype
Z = Z.reshape(xx.shape)

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Plot the decision boundary
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot the training points
plt.scatter(X_train[DATA_ITEM_1], X_train[DATA_ITEM_2], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# plt.xlim(0.8, 1.5)
# plt.ylim(1.5, 2.2)
plt.title(f"2-Class classification (k = {k}, weights = 'uniform')")
plt.xlabel(DATA_ITEM_1)
plt.ylabel(DATA_ITEM_2)

plt.show()

# w2 wing_area:
# Training Accuracy Rate: 0.53
# Test Accuracy Rate: 0.23

# l3p  wing_veign_length:
# Training Accuracy Rate: 0.54
# Test Accuracy Rate: 0.18

"""
import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", "Year_start", "Year_end",
                          "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"])

# Define all possible features
feature_list = ["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", 'w1', 'w2', 'w3', "wing_loading", "wing_shape",
                "wing_vein_length", "wing_area"]

# Create all combinations of two features
feature_combinations = list(combinations(feature_list, 2))

# Define target variable
y = data["Species"].astype(int)  # Convert target labels to integers

best_accuracy = 0
best_features = None

# Evaluate KNN classifier for each combination using cross-validation
for features in feature_combinations:
    X = data[list(features)]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    y_train_pred = knn.predict(X)
    score = accuracy_score(y_train_pred, y)  # 5-fold cross-validation
    # average_accuracy = scores.mean()

    if score > best_accuracy:
        best_accuracy = score
        best_features = features

print(f"Best Feature Combination: {best_features}")
print(f"Best Cross-Validated Accuracy: {best_accuracy:.2f}")

# highest: Training Misclassification Rate: 0.46    Test Misclassification Rate: 0.82

# Use the best features for further analysis or model training if needed

"""
