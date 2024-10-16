# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Generate synthetic data and standardize it.
2. Compute WCSS for different cluster counts to find the elbow point.
3. Apply K-Means clustering with the optimal number of clusters.
4. Plot the Elbow Method and cluster visualization with centroids.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117
*/
```
```
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate synthetic data for customer segmentation
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Apply K-Means Clustering for optimal clusters (e.g., 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the Elbow Method graph
ax[0].plot(range(1, 11), wcss, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('No. of Clusters')
ax[0].set_ylabel('WCSS')

# Plot the clustered data with cluster centers
ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
ax[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
ax[1].set_title("K-Means Clustering for Customer Segmentation")
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/8b515b7d-f9b4-4261-b1bb-eef99b91fce9)
![image](https://github.com/user-attachments/assets/2bc0ec0f-7e57-403d-be6d-69da8ce86b4e)





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
