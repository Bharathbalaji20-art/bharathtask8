import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('Mall_Customers.csv')
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df.drop(['CustomerID'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(6, 5))
sns.scatterplot(x='PC1', y='PC2', data=X_pca_df)
plt.title('PCA Projection of Dataset')
plt.show()

sse = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(6, 5))
plt.plot(K_range, sse, 'bo-')
plt.xlabel('Number of Clusters K')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method For Optimal K')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
X_pca_df['Cluster'] = cluster_labels

plt.figure(figsize=(6, 5))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=X_pca_df)
plt.title('K-Means Clusters (K=3)')
plt.show()

score = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score for K=3: {score:.4f}')
