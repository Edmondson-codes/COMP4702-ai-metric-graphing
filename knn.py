
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
                   names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape",	"wing_vein_length",	"wing_area"
])


# Assuming 'species' is the column to predict
X = data[["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]]
y = data['Species']  # Make sure 'species' is the correct column name

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors= 7)  # You can adjust the number of neighbors

# Train the classifier
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


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

# Accuracy: 69.65%

