import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data_path = '../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops (copy).csv'
data = pd.read_csv(data_path, names=["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", 'w1', 'w2', 'w3', "wing_loading",
                          "wing_shape",	"wing_vein_length",	"wing_area"])

# Selecting numerical data for PCA
numerical_data = data.select_dtypes(include=[np.number])

# Initialize PCA
pca = PCA(n_components=13)  # Only compute the first three components
pca.fit(numerical_data)

# Accessing the PCA loadings
loadings = pca.components_

# Plotting the loadings
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.bar(range(len(loadings[i])), loadings[i])
    ax.set_title(f'PCA Component {i+1}')
    ax.set_xticks(range(len(numerical_data.columns)))
    ax.set_xticklabels(numerical_data.columns, rotation=90)
    ax.set_ylabel('Loading Value')

plt.tight_layout()
plt.show()

# Print the features with highest loadings for each component
for i in range(10):
    print(f"Top contributing features for Component {i+1}:")
    component_loadings = pd.Series(loadings[i], index=numerical_data.columns)
    print(component_loadings.abs().nlargest(3))  # Display the top 3 features for each component
