import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])

# Store inertia values
distortions = []

# Loop through k values
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    distortions.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(6,4))
plt.plot(range(2, 20), distortions, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Distortion)")
plt.grid(True)
plt.show()
