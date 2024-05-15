import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv", names=["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"])

# Select the columns of interest
columns_of_interest = ["Species", "Population", "Latitude", "Longitude", 'Year_start', "Year_end",
                          "Temperature", "Replicate", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3",
                          'w1', 'w2', 'w3', "wing_loading", "wing_shape", "wing_vein_length", "wing_area"]
filtered_data = data[columns_of_interest]

# Compute the correlation matrix
correlation_matrix = filtered_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='GnBu')
plt.title('Correlation Matrix')
plt.show()

# Species	Population	Latitude	Longitude	Year_start	Year_end	Temperature	Replicate	Sex	Thorax_length	l2	l3p	l3d	lpd	l3	w1	w2	w3	wing_loading	wing_shape	wing_vein_length	wing_area