import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"
])

# Assuming 'Species' is the column to predict
X = data[["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]]
y = data['Species']  # Make sure 'Species' is the correct column name

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store accuracies for different values of k
accuracies = []
k_values = range(1, 1081)  # k values from 1 to 20

# Loop over various values of 'k' from 1 to 20
for k in k_values:
    # Create KNN classifier for each k
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier
    knn.fit(X_train, y_train)
    # Predict on the test data
    y_pred = knn.predict(X_test)
    # Calculate and store the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
# plt.title('K-NN varying number of neighbors')
# plt.xlabel('Number of neighbors (k)')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.xticks(k_values)  # Ensure all k values are marked
# plt.show()

# Print the best k value
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Best value of k: {best_k} with accuracy: {max(accuracies)}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming the rest of your code is unchanged and accuracies is already filled

# Convert accuracies list to a Pandas Series for easier manipulation
accuracies_series = pd.Series(accuracies)

# Calculate the moving average of the accuracies
window_size = 15
moving_avg = accuracies_series.rolling(window=window_size).mean()

# Correct the shift of the moving average by shifting it to the left
# Half of the window size is typically used to center the average correctly
moving_avg = moving_avg.shift(-window_size//2)

# Calculate standard deviation for the rolling window
rolling_std = accuracies_series.rolling(window=window_size).std()

# Plotting the accuracies
plt.figure(figsize=(12, 8))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(k_values, moving_avg, linestyle='-', color='r', label='Adjusted 15-pt Moving Average')

# Add a shaded area to represent the standard deviation around the moving average
plt.fill_between(k_values, (moving_avg-rolling_std), (moving_avg+rolling_std), color='r', alpha=0.2, label='Standard Deviation')

plt.title('K-NN Varying Number of Neighbors')
plt.xlabel('Number of neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values)  # Ensure all k values are marked
plt.legend()
plt.show()

# Print the best k value
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Best value of k: {best_k} with accuracy: {max(accuracies)}")


