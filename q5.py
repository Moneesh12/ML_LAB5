import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)

# Calculate clustering metrics
sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_score = davies_bouldin_score(X_train, kmeans.labels_)

# Print results
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
