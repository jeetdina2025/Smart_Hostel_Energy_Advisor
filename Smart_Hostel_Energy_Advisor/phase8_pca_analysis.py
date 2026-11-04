import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.makedirs("plots", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load datasets prepared in earlier phases
df_daily_scaled = pd.read_csv("processed_data/pca_scaled_data.csv")
df_raw = pd.read_csv("hostel_energy_dataset.csv")
df_rooms_clustered = pd.read_csv("processed_data/kmeans_rooms_clustered.csv")
df_rooms_scaled = pd.read_csv("processed_data/kmeans_room_scaled.csv")

pca_features = [
    "Occupancy", "Outdoor_Temp", "AC_Hours", "Fan_Hours",
    "Light_Hours", "Appliance_Hours", "Room_Capacity"
]

# ==============================================================================
# 1. DAILY OBSERVATIONS PCA (500 ROWS x 7 FEATURES)
# ==============================================================================
pca_daily = PCA()
X_daily_pca = pca_daily.fit_transform(df_daily_scaled[pca_features])

# Flip signs of PC1 and PC2 so positive loadings correspond to higher usage/temperature
for i in [0, 1]:
    if np.sum(pca_daily.components_[i]) < 0:
        pca_daily.components_[i] *= -1
        X_daily_pca[:, i] *= -1

eigenvalues = pca_daily.explained_variance_
var_ratio = pca_daily.explained_variance_ratio_ * 100
cum_var_ratio = np.cumsum(var_ratio)

print("=== 1. PCA EIGENVALUES & EXPLAINED VARIANCE (500 DAILY RECORDS) ===")
pca_summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(pca_features))],
    "Eigenvalue": np.round(eigenvalues, 4),
    "Variance_Explained (%)": np.round(var_ratio, 2),
    "Cumulative_Variance (%)": np.round(cum_var_ratio, 2)
})
print(pca_summary.to_string(index=False), "\n")

print("=== 2. PRINCIPAL COMPONENT LOADINGS (EIGENVECTORS: PC1 TO PC3) ===")
loadings_df = pd.DataFrame(
    pca_daily.components_[:3].T,
    index=pca_features,
    columns=["PC1 (Occupancy/Usage)", "PC2 (Weather/AC)", "PC3 (Room Capacity)"]
).round(4)
print(loadings_df.to_string(), "\n")

# Save 3-component reduced daily dataset
df_pca_out = pd.DataFrame(X_daily_pca[:, :3], columns=["PC1", "PC2", "PC3"]).round(4)
df_pca_out["Energy_kWh"] = df_raw["Energy_kWh"]
df_pca_out.to_csv("processed_data/pca_reduced_3d.csv", index=False)

# ==============================================================================
# 2. ROOM-LEVEL PCA (50 ROOMS) TO VISUALIZE K-MEANS CLUSTERS IN 2D
# ==============================================================================
room_cols = [
    "Occupancy", "AC_Hours", "Fan_Hours", "Light_Hours",
    "Appliance_Hours", "Room_Capacity", "Energy_kWh"
]
pca_room = PCA(n_components=2)
X_room_pca = pca_room.fit_transform(df_rooms_scaled[room_cols])

if np.sum(pca_room.components_[0]) < 0:
    pca_room.components_[0] *= -1
    X_room_pca[:, 0] *= -1

room_var = pca_room.explained_variance_ratio_ * 100
print("=== 3. ROOM-LEVEL 2D PCA FOR K-MEANS VISUALIZATION (50 ROOMS) ===")
print(f"Room PC1 Explained Variance : {room_var[0]:.2f}%")
print(f"Room PC2 Explained Variance : {room_var[1]:.2f}%")
print(f"Total 2D Variance Retained  : {room_var[0] + room_var[1]:.2f}%\n")

df_rooms_clustered["PC1"] = np.round(X_room_pca[:, 0], 4)
df_rooms_clustered["PC2"] = np.round(X_room_pca[:, 1], 4)

# ==============================================================================
# 3. GENERATE HIGH-RESOLUTION PCA PLOTS FOR REPORT
# ==============================================================================

# Plot 7: Scree Plot & Cumulative Explained Variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

pcs = [f"PC{i+1}" for i in range(len(pca_features))]
axes[0].bar(pcs, var_ratio, color="#1f77b4", alpha=0.8, edgecolor="black", label="Individual Variance (%)")
axes[0].plot(pcs, cum_var_ratio, color="#d62728", marker="o", lw=2.5, markersize=7, label="Cumulative Variance (%)")
axes[0].axhline(80, color="green", linestyle="--", lw=1.5, label="80% Retention Threshold")
for i, v in enumerate(var_ratio):
    axes[0].text(i, v + 1.8, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, 108)
axes[0].set_title("PCA Scree Plot: Individual & Cumulative Variance", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Principal Components")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].legend(loc="center right")

# Right Subplot: Daily Observations Projected onto PC1 & PC2 (Colored by Energy_kWh)
sc = axes[1].scatter(
    X_daily_pca[:, 0], X_daily_pca[:, 1],
    c=df_raw["Energy_kWh"], cmap="plasma", s=45, alpha=0.85, edgecolors="k", linewidth=0.3
)
cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label("Daily Energy Consumption (kWh)")

# Overlay Loading Vectors (Biplot Arrows)
for i, feature in enumerate(pca_features):
    lx = pca_daily.components_[0, i] * 3.2
    ly = pca_daily.components_[1, i] * 3.2
    axes[1].arrow(0, 0, lx, ly, color="#2b2b2b", width=0.02, head_width=0.14, alpha=0.85)
    axes[1].text(lx * 1.13, ly * 1.13, feature, fontsize=9, fontweight="bold", ha="center", va="center")

axes[1].axhline(0, color="gray", linestyle="--", lw=1)
axes[1].axvline(0, color="gray", linestyle="--", lw=1)
axes[1].set_title(f"PCA Biplot (PC1: {var_ratio[0]:.1f}%, PC2: {var_ratio[1]:.1f}% — Total: {var_ratio[0]+var_ratio[1]:.1f}%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("PC1: Occupancy & General Appliance Activity")
axes[1].set_ylabel("PC2: Outdoor Temperature & AC Cooling Demand")

plt.tight_layout()
plt.savefig("plots/7_pca_scree_and_biplot.png", dpi=300)
plt.close()

# Plot 8: 2D PCA Projection of the 50 Hostel Rooms Colored by K-Means Cluster
plt.figure(figsize=(9, 6))
palette_map = {
    "Cluster 0 (Low Usage)": "#2ca02c",
    "Cluster 1 (Moderate Usage)": "#1f77b4",
    "Cluster 2 (High Usage)": "#d62728"
}
sns.scatterplot(
    data=df_rooms_clustered, x="PC1", y="PC2",
    hue="Cluster_Name", style="Cluster_Name", palette=palette_map,
    s=120, alpha=0.9, edgecolor="black"
)
for _, row in df_rooms_clustered.iloc[::3].iterrows():
    plt.text(row["PC1"] + 0.08, row["PC2"] + 0.05, row["Room_ID"], fontsize=8.5, alpha=0.8)

plt.axhline(0, color="gray", linestyle="--", lw=1)
plt.axvline(0, color="gray", linestyle="--", lw=1)
plt.title(f"Hostel Rooms in 2D PCA Space Colored by k-Means Clusters ({room_var[0]+room_var[1]:.1f}% Variance Retained)", fontsize=13, fontweight="bold")
plt.xlabel(f"Room PC1 ({room_var[0]:.1f}% Variance — Overall Consumption & Occupancy Scale)")
plt.ylabel(f"Room PC2 ({room_var[1]:.1f}% Variance — AC vs. Non-AC Usage Contrast)")
plt.legend(title="k-Means Usage Tier", loc="best")
plt.tight_layout()
plt.savefig("plots/8_pca_kmeans_room_projection.png", dpi=300)
plt.close()

print("Saved PCA dataset and 2 high-resolution figures:")
print(" - processed_data/pca_reduced_3d.csv")
print(" - plots/7_pca_scree_and_biplot.png")
print(" - plots/8_pca_kmeans_room_projection.png")