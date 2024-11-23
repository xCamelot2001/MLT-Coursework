import nltk
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Download necessary NLTK data
nltk.download('brown')

# Load the Brown Corpus
#corpus = brown.words() #Full dataset will take time.
corpus = brown.words()[:10000] #Partial data, fast.

# Tokenize and vectorize the corpus
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Experiment with different numbers of clusters
k_values = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_values:
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    #inertia (sum of squared distances to centroids)
    inertia = kmeans.inertia_
    silhouette_score_ = silhouette_score(X, labels)

    inertias.append(inertia)
    silhouette_scores.append(silhouette_score_)

# Plot the elbow method and silhouette score
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()
