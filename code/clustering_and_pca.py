import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load cleaned dataset
lending = pd.read_csv("../cleaned_data/lending_club_cleaned.csv")


# ----- KMEANS WITHOUT PCA: LENDING CLUB ----

# Keep quantitative columns for KMeans
columns_to_keep = ["emp_length", "annual_income", "debt_to_income", "delinq_2y", "inquiries_last_12m",
                   "total_credit_lines", "open_credit_lines", "total_credit_limit", "total_credit_utilized",
                   "loan_amount", "term", "interest_rate", "installment"]

lending_kmeans = lending[columns_to_keep].copy()

# Make sure all selected columns are numeric
for col in columns_to_keep:
    lending_kmeans[col] = pd.to_numeric(lending_kmeans[col], errors="coerce")

# Replace missing values with median
lending_kmeans = lending_kmeans.replace([np.inf, -np.inf], np.nan)
lending_kmeans = lending_kmeans.fillna(lending_kmeans.median())

# Scale the data
scaler = StandardScaler()
lending_scaled = scaler.fit_transform(lending_kmeans)

# Convert scaled data back to dataframe
lending_scaled = pd.DataFrame(lending_scaled, columns=columns_to_keep)

# Final checks
print(lending_scaled.head())
print(lending_scaled.info())
print(lending_scaled.isnull().sum())
print("Duplicate rows:", lending_scaled.duplicated().sum())

# Save prepared KMeans dataset
lending_scaled.to_csv("../cleaned_data/lending_club_kmeans_no_pca_prepared.csv", index=False)

# Kmeans without PCA: k = 2
kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
lending["cluster_k2"] = kmeans2.fit_predict(lending_scaled)

print("KMeans without PCA, k = 2")
print("Inertia:", kmeans2.inertia_)
print(lending["cluster_k2"].value_counts().sort_index())

print(pd.crosstab(lending["cluster_k2"], lending["loan_status"], normalize="index") * 100)

cluster_profile_k2 = lending.groupby("cluster_k2")[columns_to_keep].mean().round(2)
print(cluster_profile_k2)

cluster_profile_k2.to_csv("../cleaned_data/lending_kmeans_no_pca_profile_k2.csv")

centers_k2 = pd.DataFrame(kmeans2.cluster_centers_, columns=columns_to_keep)

plt.figure(figsize=(12, 4))
plt.imshow(centers_k2, aspect="auto")
plt.colorbar(label="Standardized Value")
plt.title("KMeans Without PCA Cluster Centers, k = 2")
plt.xlabel("Variable")
plt.ylabel("Cluster")
plt.xticks(range(len(columns_to_keep)), columns_to_keep, rotation=45, ha="right")
plt.yticks(range(2), ["Cluster 0", "Cluster 1"])
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_no_pca_k2_heatmap.png", dpi=200)
plt.show()

# Kmeans without PCA: k = 3
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
lending["cluster_k3"] = kmeans3.fit_predict(lending_scaled)

print("KMeans without PCA, k = 3")
print("Inertia:", kmeans3.inertia_)
print(lending["cluster_k3"].value_counts().sort_index())

print(pd.crosstab(lending["cluster_k3"], lending["loan_status"], normalize="index") * 100)

cluster_profile_k3 = lending.groupby("cluster_k3")[columns_to_keep].mean().round(2)
print(cluster_profile_k3)

cluster_profile_k3.to_csv("../cleaned_data/lending_kmeans_no_pca_profile_k3.csv")

centers_k3 = pd.DataFrame(kmeans3.cluster_centers_, columns=columns_to_keep)

plt.figure(figsize=(12, 4))
plt.imshow(centers_k3, aspect="auto")
plt.colorbar(label="Standardized Value")
plt.title("KMeans Without PCA Cluster Centers, k = 3")
plt.xlabel("Variable")
plt.ylabel("Cluster")
plt.xticks(range(len(columns_to_keep)), columns_to_keep, rotation=45, ha="right")
plt.yticks(range(3), ["Cluster 0", "Cluster 1", "Cluster 2"])
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_no_pca_k3_heatmap.png", dpi=200)
plt.show()


# Kmeans without PCA: k = 4
kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
lending["cluster_k4"] = kmeans4.fit_predict(lending_scaled)

print("KMeans without PCA, k = 4")
print("Inertia:", kmeans4.inertia_)
print(lending["cluster_k4"].value_counts().sort_index())

print(pd.crosstab(lending["cluster_k4"], lending["loan_status"], normalize="index") * 100)

cluster_profile_k4 = lending.groupby("cluster_k4")[columns_to_keep].mean().round(2)
print(cluster_profile_k4)

cluster_profile_k4.to_csv("../cleaned_data/lending_kmeans_no_pca_profile_k4.csv")

centers_k4 = pd.DataFrame(kmeans4.cluster_centers_, columns=columns_to_keep)

plt.figure(figsize=(12, 4))
plt.imshow(centers_k4, aspect="auto")
plt.colorbar(label="Standardized Value")
plt.title("KMeans Without PCA Cluster Centers, k = 4")
plt.xlabel("Variable")
plt.ylabel("Cluster")
plt.xticks(range(len(columns_to_keep)), columns_to_keep, rotation=45, ha="right")
plt.yticks(range(4), ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_no_pca_k4_heatmap.png", dpi=200)
plt.show()

# Compare k values
inertia = pd.DataFrame({
    "k": [2, 3, 4],
    "inertia": [kmeans2.inertia_, kmeans3.inertia_, kmeans4.inertia_]
})

print(inertia)

plt.figure(figsize=(8, 5))
plt.plot(inertia["k"], inertia["inertia"], marker="o")
plt.title("KMeans Without PCA Inertia Comparison")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks([2, 3, 4])
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_no_pca_inertia.png", dpi=200)
plt.show()

# Save Lending Club dataset with cluster labels
lending.to_csv("../cleaned_data/lending_club_kmeans_no_pca_clusters.csv", index=False)


# Create inertia comparison table
comparison = pd.DataFrame({
    "k": [2, 3, 4],
    "inertia": [kmeans2.inertia_, kmeans3.inertia_, kmeans4.inertia_],
    "cluster_counts": [
        lending["cluster_k2"].value_counts().sort_index().to_dict(),
        lending["cluster_k3"].value_counts().sort_index().to_dict(),
        lending["cluster_k4"].value_counts().sort_index().to_dict()
    ]
})

comparison["inertia"] = comparison["inertia"].round(2)

print(comparison)

comparison.to_csv("../cleaned_data/lending_kmeans_no_pca_comparison.csv", index=False)

# Create image of comparison table
comparison_preview = comparison.copy()
comparison_preview["cluster_counts"] = comparison_preview["cluster_counts"].astype(str)

plt.figure(figsize=(10, 2))
plt.axis("off")
plt.table(cellText=comparison_preview.values,
          colLabels=comparison_preview.columns,
          loc="center")
plt.title("KMeans Without PCA Numeric Comparison")
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_no_pca_numeric_comparison.png", dpi=200)
plt.show()


# ----- KMEANS USING PCA: LENDING CLUB ----

# Reduce the same scaled KMeans data to 3 principal components
pca = PCA(n_components=3)
lending_pca = pca.fit_transform(lending_scaled)

# Convert PCA data back to dataframe
lending_pca = pd.DataFrame(lending_pca, columns=["pc1", "pc2", "pc3"])

# Final checks
print(lending_pca.head())
print(lending_pca.info())
print(lending_pca.isnull().sum())
print("Duplicate rows:", lending_pca.duplicated().sum())

# Explained variance
explained_variance = pd.DataFrame({
    "component": ["pc1", "pc2", "pc3"],
    "explained_variance_ratio": pca.explained_variance_ratio_
})

explained_variance["explained_variance_ratio"] = explained_variance["explained_variance_ratio"].round(4)

print(explained_variance)
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 4))

# Save PCA prepared dataset
lending_pca.to_csv("../cleaned_data/lending_club_pca_3d_prepared.csv", index=False)
explained_variance.to_csv("../cleaned_data/lending_club_pca_explained_variance.csv", index=False)

# Kmeans using PCA: k = 2
kmeans_pca2 = KMeans(n_clusters=2, random_state=42, n_init=10)
lending["pca_cluster_k2"] = kmeans_pca2.fit_predict(lending_pca)

print("KMeans using PCA, k = 2")
print("Inertia:", kmeans_pca2.inertia_)
print(lending["pca_cluster_k2"].value_counts().sort_index())

print(pd.crosstab(lending["pca_cluster_k2"], lending["loan_status"], normalize="index") * 100)

pca_profile_k2 = lending.groupby("pca_cluster_k2")[columns_to_keep].mean().round(2)
print(pca_profile_k2)

pca_profile_k2.to_csv("../cleaned_data/lending_kmeans_pca_profile_k2.csv")


# Kmeans using PCA: k = 3
kmeans_pca3 = KMeans(n_clusters=3, random_state=42, n_init=10)
lending["pca_cluster_k3"] = kmeans_pca3.fit_predict(lending_pca)

print("KMeans using PCA, k = 3")
print("Inertia:", kmeans_pca3.inertia_)
print(lending["pca_cluster_k3"].value_counts().sort_index())

print(pd.crosstab(lending["pca_cluster_k3"], lending["loan_status"], normalize="index") * 100)

pca_profile_k3 = lending.groupby("pca_cluster_k3")[columns_to_keep].mean().round(2)
print(pca_profile_k3)

pca_profile_k3.to_csv("../cleaned_data/lending_kmeans_pca_profile_k3.csv")


# Kmeans using PCA: k = 4
kmeans_pca4 = KMeans(n_clusters=4, random_state=42, n_init=10)
lending["pca_cluster_k4"] = kmeans_pca4.fit_predict(lending_pca)

print("KMeans using PCA, k = 4")
print("Inertia:", kmeans_pca4.inertia_)
print(lending["pca_cluster_k4"].value_counts().sort_index())

print(pd.crosstab(lending["pca_cluster_k4"], lending["loan_status"], normalize="index") * 100)

pca_profile_k4 = lending.groupby("pca_cluster_k4")[columns_to_keep].mean().round(2)
print(pca_profile_k4)

pca_profile_k4.to_csv("../cleaned_data/lending_kmeans_pca_profile_k4.csv")


# Compare PCA k values
pca_inertia = pd.DataFrame({
    "k": [2, 3, 4],
    "inertia": [kmeans_pca2.inertia_, kmeans_pca3.inertia_, kmeans_pca4.inertia_]
})

print(pca_inertia)

plt.figure(figsize=(8, 5))
plt.plot(pca_inertia["k"], pca_inertia["inertia"], marker="o")
plt.title("KMeans Using PCA Inertia Comparison")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks([2, 3, 4])
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_pca_inertia.png", dpi=200)
plt.show()


# Create inertia comparison table
pca_comparison = pd.DataFrame({
    "k": [2, 3, 4],
    "inertia": [kmeans_pca2.inertia_, kmeans_pca3.inertia_, kmeans_pca4.inertia_],
    "cluster_counts": [
        lending["pca_cluster_k2"].value_counts().sort_index().to_dict(),
        lending["pca_cluster_k3"].value_counts().sort_index().to_dict(),
        lending["pca_cluster_k4"].value_counts().sort_index().to_dict()
    ]
})

pca_comparison["inertia"] = pca_comparison["inertia"].round(2)

print(pca_comparison)

pca_comparison.to_csv("../cleaned_data/lending_kmeans_pca_comparison.csv", index=False)

# Create image of comparison table
pca_comparison_preview = pca_comparison.copy()
pca_comparison_preview["cluster_counts"] = pca_comparison_preview["cluster_counts"].astype(str)

plt.figure(figsize=(10, 2))
plt.axis("off")
plt.table(cellText=pca_comparison_preview.values,
          colLabels=pca_comparison_preview.columns,
          loc="center")
plt.title("KMeans Using PCA Numeric Comparison")
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_pca_numeric_comparison.png", dpi=200)
plt.show()


# 3D visualization for k = 3
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(lending_pca["pc1"],
                     lending_pca["pc2"],
                     lending_pca["pc3"],
                     c=lending["pca_cluster_k3"],
                     s=15,
                     alpha=0.6)

ax.set_title("KMeans Using PCA 3D Clustering, k = 3")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("../figures/lending_kmeans_pca_k3_3d.png", dpi=200)
plt.show()

# Save Lending Club dataset with PCA cluster labels
lending.to_csv("../cleaned_data/lending_club_kmeans_pca_clusters.csv", index=False)


# ----- AGGLOMERATIVE HIERARCHICAL CLUSTERING ----

# Use a sample of the prepared data
lending_sample_scaled = lending_scaled.sample(n=500, random_state=42)
lending_sample = lending.loc[lending_sample_scaled.index].copy()

# Perform Agglomerative Clustering using Sklearn
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
lending_sample["agg_cluster"] = agg.fit_predict(lending_sample_scaled)

# Final checks
print(lending_sample_scaled.head())
print(lending_sample_scaled.info())
print(lending_sample_scaled.isnull().sum())
print("Duplicate rows:", lending_sample_scaled.duplicated().sum())

print("Agglomerative Hierarchical Clustering, k = 3")
print(lending_sample["agg_cluster"].value_counts().sort_index())

print(pd.crosstab(lending_sample["agg_cluster"], lending_sample["loan_status"], normalize="index") * 100)

agg_profile = lending_sample.groupby("agg_cluster")[columns_to_keep].mean().round(2)
print(agg_profile)

# Save Agglomerative results
agg_profile.to_csv("../cleaned_data/lending_agglomerative_profile.csv")
lending_sample.to_csv("../cleaned_data/lending_agglomerative_sample_clusters.csv", index=False)

# Create dendrogram using Python
linked = linkage(lending_sample_scaled.to_numpy(), method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode="lastp", p=30)
plt.title("Agglomerative Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Group")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("../figures/lending_agglomerative_dendrogram.png", dpi=200)
plt.show()

# Create cluster profile heatmap
agg_centers = lending_sample.groupby("agg_cluster")[columns_to_keep].mean()

agg_centers_scaled = pd.DataFrame(
    scaler.transform(agg_centers),
    columns=columns_to_keep,
    index=agg_centers.index
)

plt.figure(figsize=(12, 4))
plt.imshow(agg_centers_scaled, aspect="auto")
plt.colorbar(label="Standardized Value")
plt.title("Agglomerative Clustering Cluster Profiles, k = 3")
plt.xlabel("Variable")
plt.ylabel("Cluster")
plt.xticks(range(len(columns_to_keep)), columns_to_keep, rotation=45, ha="right")
plt.yticks(range(3), ["Cluster 0", "Cluster 1", "Cluster 2"])
plt.tight_layout()
plt.savefig("../figures/lending_agglomerative_cluster_profiles.png", dpi=200)
plt.show()
