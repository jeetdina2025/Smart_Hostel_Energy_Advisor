import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("plots", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load unscaled and scaled room-level datasets from Phase 3
df_rooms = pd.read_csv("processed_data/kmeans_room_unscaled.csv")
df_scaled = pd.read_csv("processed_data/kmeans_room_scaled.csv")

room_cols = [
    "Occupancy", "AC_Hours", "Fan_Hours", "Light_Hours",
    "Appliance_Hours", "Room_Capacity", "Energy_kWh"
]
X_scaled = df_scaled[room_cols].values

# ==============================================================================
# 1. SLIDE 19 VERIFICATION: THE ELBOW METHOD (CHOOSING k)
# ==============================================================================
print("=== 1. ELBOW METHOD: WCSS & SILHOUETTE SCORE (SLIDE 19 STYLE) ===")
k_values = list(range(1, 8))
wcss_values = []
silhouette_values = []

for k in k_values:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    km.fit(X_scaled)
    wcss = km.inertia_
    wcss_values.append(wcss)
    sil = silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan
    silhouette_values.append(sil)
    drop = (wcss_values[-2] - wcss) if k > 1 else 0.0
    print(f"k = {k} | Best WCSS = {wcss:8.4f} | WCSS Drop = {drop:8.4f} | Silhouette = {sil:.4f}")

# ==============================================================================
# 2. FIT K-MEANS WITH OPTIMAL k = 3 & ORDER CLUSTERS BY CONSUMPTION
# ==============================================================================
km3 = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
raw_labels = km3.fit_predict(X_scaled)
df_rooms["Cluster_Raw"] = raw_labels

# Sort clusters so Cluster 0 = Low, Cluster 1 = Moderate, Cluster 2 = High Energy
sorted_idx = df_rooms.groupby("Cluster_Raw")["Energy_kWh"].mean().sort_values().index
cluster_map = {old_label: new_label for new_label, old_label in enumerate(sorted_idx)}
df_rooms["Cluster"] = df_rooms["Cluster_Raw"].map(cluster_map)

tier_labels = {
    0: "Cluster 0 (Low Usage)",
    1: "Cluster 1 (Moderate Usage)",
    2: "Cluster 2 (High Usage)"
}
df_rooms["Cluster_Name"] = df_rooms["Cluster"].map(tier_labels)
df_rooms.drop(columns=["Cluster_Raw"], inplace=True)

# Reorder scaled centroids to match 0, 1, 2
centroids_scaled = km3.cluster_centers_[sorted_idx]

print("\n=== 2. CLUSTER CENTROIDS (UNSCALED ROOM AVERAGES) ===")
cluster_summary = df_rooms.groupby(["Cluster", "Cluster_Name"])[room_cols].mean().round(2)
cluster_summary.insert(0, "Rooms_Count", df_rooms["Cluster"].value_counts().sort_index().values)
print(cluster_summary.to_string(), "\n")

# ==============================================================================
# 3. SLIDE 8 & SLIDE 16 VERIFICATION: EUCLIDEAN DISTANCE & PER-CLUSTER WCSS
# ==============================================================================
print("=== 3. EUCLIDEAN DISTANCE CHECK FOR ROOM R001 (SLIDE 8 STYLE) ===")
r001_vec = X_scaled[0]
for j in range(3):
    dist = np.sqrt(np.sum((r001_vec - centroids_scaled[j]) ** 2))
    print(f"d(R001, centroid_{j}) = {dist:.4f}  (squared distance d^2 = {dist**2:.4f})")
print(f"-> Shortest distance wins: Room R001 joins {df_rooms.loc[0, 'Cluster_Name']}\n")

print("=== 4. FINAL WCSS BREAKDOWN BY CLUSTER (SLIDE 16 STYLE) ===")
total_wcss = 0.0
for j in range(3):
    mask = (df_rooms["Cluster"] == j).values
    cluster_points = X_scaled[mask]
    wcss_j = np.sum((cluster_points - centroids_scaled[j]) ** 2)
    total_wcss += wcss_j
    rooms_list = ", ".join(df_rooms.loc[mask, "Room_ID"].head(5).tolist()) + ("..." if mask.sum() > 5 else "")
    print(f"{tier_labels[j]:26s} | n = {mask.sum():2d} rooms ({rooms_list}) | WCSS_{j} = {wcss_j:.4f}")
print(f"Total WCSS across all 50 rooms = {total_wcss:.4f}\n")

# Save clustered rooms dataset for Phase 8 (PCA overlay) and report
df_rooms.to_csv("processed_data/kmeans_rooms_clustered.csv", index=False)

# ==============================================================================
# 4. GENERATE HIGH-RESOLUTION PLOTS FOR REPORT
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left Subplot: Elbow Method Curve (Slide 19 style)
axes[0].plot(k_values, wcss_values, marker="o", lw=2.5, markersize=8, color="#1f77b4")
axes[0].scatter([3], [wcss_values[2]], s=220, facecolors="none", edgecolors="#ff7f0e", lw=2.5, zorder=5)
axes[0].annotate(
    f"Elbow at k = 3\n(WCSS = {wcss_values[2]:.2f})",
    xy=(3, wcss_values[2]), xytext=(3.6, wcss_values[2] + 45),
    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2),
    fontsize=11, fontweight="bold", color="#d95f02"
)
axes[0].set_title("Elbow Method for Optimal k (WCSS vs. k)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)")
axes[0].set_xticks(k_values)

# Right Subplot: Clustered Rooms in Physical Units (AC_Hours vs Energy_kWh)
palette_map = {
    "Cluster 0 (Low Usage)": "#2ca02c",
    "Cluster 1 (Moderate Usage)": "#1f77b4",
    "Cluster 2 (High Usage)": "#d62728"
}
sns.scatterplot(
    data=df_rooms, x="AC_Hours", y="Energy_kWh",
    hue="Cluster_Name", style="Cluster_Name", palette=palette_map,
    s=100, alpha=0.9, ax=axes[1]
)

# Plot unscaled cluster centroids as large squares (like Slide 16)
centroids_unscaled = df_rooms.groupby("Cluster")[["AC_Hours", "Energy_kWh"]].mean()
axes[1].scatter(
    centroids_unscaled["AC_Hours"], centroids_unscaled["Energy_kWh"],
    s=230, marker="s", c=["#2ca02c", "#1f77b4", "#d62728"],
    edgecolors="black", lw=2, label="Centroids (μ)"
)
axes[1].set_title("k-Means Room Clusters (k = 3): AC Usage vs. Daily Energy", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Average Daily AC Usage (Hours)")
axes[1].set_ylabel("Average Daily Energy Consumption (kWh)")
axes[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig("plots/6_kmeans_elbow_and_clusters.png", dpi=300)
plt.close()

print("Saved clustered dataset and plot:")
print(" - processed_data/kmeans_rooms_clustered.csv")
print(" - plots/6_kmeans_elbow_and_clusters.png")