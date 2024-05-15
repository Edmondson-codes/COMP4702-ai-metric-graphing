import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data_path = '../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops (copy).csv'
data = pd.read_csv(data_path,
                   names=["Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape",	"wing_vein_length",	"wing_area"])

# Selecting numerical data for PCA
numerical_data = data.select_dtypes(include=[np.number])

# Initialize PCA
pca = PCA()

# Fit PCA on the numerical data
pca.fit(numerical_data)

# Cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance Ratio')

# Set x-axis ticks to display whole numbers
plt.xticks(ticks=np.arange(len(cumulative_variance_ratio)), labels=np.arange(1, len(cumulative_variance_ratio) + 1))


plt.grid(True)
plt.show()
