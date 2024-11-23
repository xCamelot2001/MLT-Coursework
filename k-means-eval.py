import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate random data points
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 points in 2 dimensions

# Create a K-means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels for each point
labels = kmeans.labels_

# Evaluate the clustering using silhouette score
silhouette_avg = silhouette_score(X, labels)
print("Silhouette score:", silhouette_avg)

# Visualize the clusters
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("K-means Clustering")
plt.show()
