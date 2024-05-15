# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
#
# # Load the dataset
# data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
#                    names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
#                           "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
#                           'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"
# ])
#
# DATA_ITEM_1 = "wing_vein_length"
# DATA_ITEM_2 = "wing_shape"
#
# # Select features and target
# X = data[[DATA_ITEM_1, DATA_ITEM_2]]
# y = data["Species"].astype(int)
#
# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # PCA transformation
# pca = PCA(n_components=2)  # Use 2 principal components
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
#
# # Create and train the Decision Tree classifier on the PCA-transformed data
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(X_train_pca, y_train)
#
# # Evaluate the classifier
# y_train_pred = dt.predict(X_train_pca)
# train_misclassification_rate = accuracy_score(y_train, y_train_pred)
# print(f"Training Misclassification Rate: {train_misclassification_rate:.2f}")
#
# y_test_pred = dt.predict(X_test_pca)
# test_misclassification_rate = accuracy_score(y_test, y_test_pred)
# print(f"Test Misclassification Rate: {test_misclassification_rate:.2f}")
#
# # Plotting the decision boundary
# h = 0.01  # Step size in the mesh
# x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
# y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh using the model
# Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# # Define color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
#
# # Plot the decision boundary
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
#
# # Plot the PCA-transformed training points
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Decision Tree Classification with PCA")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
#
# plt.show()
#
#
# ## With a dimension reduction PCA its 0.75 on testing data, without its 0.76


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape",	"wing_vein_length",	"wing_area"
])

# Specify the features and target variable
features = ["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]
target = 'Species'  # Update this if your target variable name is different

# Split data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)  # 30% for testing

# Create a Decision Tree classifier
tree_classifier = DecisionTreeClassifier(max_depth=3)  # You can adjust max_depth as needed

# Train the classifier
tree_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = tree_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Check unique values in the 'Species' column
print("Species values:", data['Species'].unique())
print("Population values:", data['Population'].unique())


# Optionally, print the decision tree rules
# tree_rules = export_text(tree_classifier, feature_names=features)
# print(tree_rules)

# first version accuracy (2 params): 0.25
# this version accuracy: 0.285 d=5 | 0.290 with depth = 3 |


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy: 67.63%