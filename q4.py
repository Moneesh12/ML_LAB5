import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# K-Means Clustering (k=2)
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_train)

 # Cluster labels for each sample,Coordinates of cluster centers
labels = kmeans.labels_              
centers = kmeans.cluster_centers_      

print("Cluster Labels for Training Data:")
print(labels)

print("\nCluster Centers:")
print(centers)
