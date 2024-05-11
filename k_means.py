# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer purchase history data (replace this with your dataset)
# Each row represents a customer and each column represents a product category or purchase amount
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Grocery': [100, 80, 70, 120, 90],
    'Clothing': [50, 60, 40, 70, 30],
    'Electronics': [150, 100, 80, 200, 120]
}

# Creating a DataFrame from the data
df = pd.DataFrame(data)

# Extracting purchase history features (excluding CustomerID)
X = df.drop('CustomerID', axis=1)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating and fitting KMeans model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# Adding the cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Visualizing the clusters
plt.figure(figsize=(8, 6))
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster].drop('Cluster', axis=1)
    plt.scatter(cluster_data['Grocery'], cluster_data['Clothing'], label=f'Cluster {cluster}')

plt.title('Customer Segmentation based on Purchase History')
plt.xlabel('Grocery Spend')
plt.ylabel('Clothing Spend')
plt.legend()
plt.show()
