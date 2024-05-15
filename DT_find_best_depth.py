import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"])

# Specify the features and target variable
features = ["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]
target = 'Species'

# Split data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  # 30% for testing

# Explore various depths
depths = range(1, 61)  # Depths from 1 to 10
accuracies = []

for depth in depths:
    # Create a Decision Tree classifier with varying depth
    tree_classifier = DecisionTreeClassifier(max_depth=depth)
    # Train the classifier
    tree_classifier.fit(X_train, y_train)
    # Predict on the test data
    y_pred = tree_classifier.predict(X_test)
    # Calculate and record the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Depth: {depth}, Accuracy: {accuracy}")

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'accuracies' and 'depths' are defined as shown previously

# Calculate the moving average and standard deviation for the valid range
window_size = 10
moving_avg = np.convolve(accuracies, np.ones(window_size) / window_size, mode='valid')
moving_std = np.array([np.std(accuracies[max(0, i - window_size + 1):i + 1]) for i in range(window_size - 1, len(accuracies))])

# Define depths for moving average and standard deviation calculations
depths_ma = depths[window_size - 1:]  # Adjust depths to match the length of moving_avg and moving_std

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(depths_ma, moving_avg, linestyle='-', color='r', label='10-pt Moving Average')
plt.fill_between(depths_ma, moving_avg - moving_std, moving_avg + moving_std, color='r', alpha=0.2, label='Standard Deviation')

plt.title('Decision Tree Accuracy by Tree Depth')
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(np.arange(min(depths), max(depths) + 1, 5))  # Adjust the x-ticks to show every 5th depth
plt.legend()
plt.show()
