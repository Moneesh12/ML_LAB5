import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])

# Range of k values
k_values = range(2, 11)

# Lists to store metrics
silhouette_scores = []
calinski_scores = []
davies_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    
    sil = silhouette_score(X, kmeans.labels_)
    ch = calinski_harabasz_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    
    silhouette_scores.append(sil)
    calinski_scores.append(ch)
    davies_scores.append(db)

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")

plt.subplot(1, 3, 2)
plt.plot(k_values, calinski_scores, marker='o', color='green')
plt.title("Calinski-Harabasz Score vs K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("CH Score")

plt.subplot(1, 3, 3)
plt.plot(k_values, davies_scores, marker='o', color='red')
plt.title("Davies-Bouldin Index vs K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
