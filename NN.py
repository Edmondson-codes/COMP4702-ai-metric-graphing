# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # Load the dataset
# data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
#                    names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
#                           "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
#                           'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"])
#
# DATA_ITEM_1 = "wing_vein_length"
# DATA_ITEM_2 = "wing_shape"
#
# # Select features and target
# X = data[[DATA_ITEM_1, DATA_ITEM_2]]
# y = data["Species"].astype(int)
#
#
#
#
# from sklearn.preprocessing import LabelEncoder
#
# # Encode the labels to start from 0 and be consecutive integers
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)
#
# # Split into training and test sets using the encoded labels
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
# # PCA transformation
# pca = PCA(n_components=2)  # Use 2 principal components
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
# # Create a Neural Network model with the correct number of output neurons
# model = models.Sequential([
#     layers.Dense(10, activation='relu', input_shape=(2,)),
#     layers.Dense(10, activation='relu'),
#     layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # ensure correct number of output neurons
# ])
#
# # Compile, train, and evaluate as before
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train_pca, y_train, epochs=50, batch_size=10)
# train_loss, train_accuracy = model.evaluate(X_train_pca, y_train)
# test_loss, test_accuracy = model.evaluate(X_test_pca, y_test)
# print(f"Training Accuracy: {1 - train_accuracy:.2f}")
# print(f"Test Accuracy: {1 -   test_accuracy:.2f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical  # Updated import

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end", "Temperature",
                          "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", 'w1', 'w2', 'w3', "wing_loading",
                          "wing_shape",	"wing_vein_length",	"wing_area"
])


# Specify features and target
features = ["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]
target = 'Species'

# Prepare the feature data and target labels
X = data[features]
y = data[target]

# Encoding categorical data
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_dummy = to_categorical(y_encoded)  # Use to_categorical directly

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.1, random_state=42)

# Feature Scaling      orig:   79.77% 80.35%  80.35%   16&8: 81.50% 81.50% 78.03%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network architecture
model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(y_dummy.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Species values:", data['Species'].unique())
print("Population values:", data['Population'].unique())

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Generate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
#
# # Plotting the confusion matrix
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt="d")
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# Test Accuracy: 34.39%    Now 78.32%

# Optionally, save the model
# model.save('species_classification_model.h5')
