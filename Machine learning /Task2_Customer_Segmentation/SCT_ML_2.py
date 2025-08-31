# 📌 Customer Segmentation using K-Means with 3D Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Select Features (Age, Income, Spending Score)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method (to find optimal clusters)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 5: Train KMeans (choose k=5 for this example)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Add cluster labels back
df['Cluster'] = y_kmeans

# Step 7: 2D Visualization (Income vs Spending)
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set1', data=df, s=100)
plt.title("Customer Segments (2D View)")
plt.legend()
plt.show()

# Step 8: 3D Visualization (Age vs Income vs Spending)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=df['Cluster'], cmap='Set1', s=80)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.set_title("Customer Segments (3D View)")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Step 9: Cluster Profiles
print("\nCluster Profiles (Mean values):")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
