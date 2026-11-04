import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a subfolder for processed data so the main folder stays clean
os.makedirs("processed_data", exist_ok=True)

# 1. Dataset loading (preserving original CSV)
df_original = pd.read_csv("hostel_energy_dataset.csv")
df = df_original.copy()

print("=== 1. DATASET LOADED ===")
print(f"Original shape: {df_original.shape} | Working copy shape: {df.shape}\n")

# 2. Check rows, columns, and data types
print("=== 2. DATA TYPES & STRUCTURE ===")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head(), "\n")

# 3. Check missing values and duplicate records
print("=== 3. MISSING VALUES & DUPLICATES ===")
print("Missing values per column:\n", df.isnull().sum())
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate (Room_ID, Date) records:", df.duplicated(subset=["Room_ID", "Date"]).sum(), "\n")

# 4. Check invalid values and logical constraints
print("=== 4. LOGICAL CONSTRAINTS & VALIDATION ===")
occ_gt_cap = (df["Occupancy"] > df["Room_Capacity"]).sum()
neg_values = (df.select_dtypes(include=[np.number]) < 0).sum().sum()
hours_gt_24 = (df[["AC_Hours", "Fan_Hours", "Light_Hours", "Appliance_Hours"]] > 24).sum().sum()
zero_occ_usage = ((df["Occupancy"] == 0) & (df[["AC_Hours", "Fan_Hours", "Light_Hours", "Appliance_Hours"]].sum(axis=1) > 0)).sum()
zero_occ_energy = df[(df["Occupancy"] == 0) & (df["Energy_kWh"] > 0)]

print(f"Occupancy > Room_Capacity violations : {occ_gt_cap}")
print(f"Negative numerical values            : {neg_values}")
print(f"Appliance usage > 24 hours           : {hours_gt_24}")
print(f"Occupancy == 0 with active hours > 0 : {zero_occ_usage}")
print(f"Occupancy == 0 with standby Energy>0 : {len(zero_occ_energy)} rows (Max = {zero_occ_energy['Energy_kWh'].max()} kWh)\n")

# 5. Summary of numerical and categorical variables
print("=== 5. SUMMARY STATISTICS (NUMERICAL) ===")
print(df.describe().round(2), "\n")

print("=== CATEGORICAL / DISCRETE COUNTS ===")
print("Unique Rooms :", df["Room_ID"].nunique())
print("Unique Dates :", df["Date"].nunique())
print("Weekend counts:", df["Weekend"].value_counts().to_dict())
print("Floor counts  :", df["Floor"].value_counts().sort_index().to_dict())
print("Capacity counts:", df["Room_Capacity"].value_counts().sort_index().to_dict(), "\n")

# 6. Prepare separate feature sets for Regression, K-means, and PCA
numerical_features = [
    "Occupancy", "Outdoor_Temp", "AC_Hours", "Fan_Hours",
    "Light_Hours", "Appliance_Hours", "Room_Capacity"
]
categorical_features = ["Weekend", "Floor"]
target = "Energy_kWh"

# A. Regression dataset (drop Room_ID and Date identifiers)
df_regression = df[numerical_features + categorical_features + [target]].copy()
df_regression.to_csv("processed_data/regression_data.csv", index=False)

# B. K-Means dataset (Aggregate by Room_ID across 10 days + Standardize)
room_cols = ["Occupancy", "AC_Hours", "Fan_Hours", "Light_Hours", "Appliance_Hours", "Room_Capacity", "Energy_kWh"]
df_rooms = df.groupby("Room_ID", as_index=False)[room_cols].mean().round(3)
df_rooms.to_csv("processed_data/kmeans_room_unscaled.csv", index=False)

scaler_kmeans = StandardScaler()
kmeans_scaled_array = scaler_kmeans.fit_transform(df_rooms[room_cols])
df_kmeans_scaled = pd.DataFrame(kmeans_scaled_array, columns=room_cols)
df_kmeans_scaled.insert(0, "Room_ID", df_rooms["Room_ID"])
df_kmeans_scaled.to_csv("processed_data/kmeans_room_scaled.csv", index=False)

# C. PCA dataset (Standardize continuous features across all 500 observations)
scaler_pca = StandardScaler()
pca_scaled_array = scaler_pca.fit_transform(df[numerical_features])
df_pca_scaled = pd.DataFrame(pca_scaled_array, columns=numerical_features)
df_pca_scaled.to_csv("processed_data/pca_scaled_data.csv", index=False)

print("=== 6. PREPROCESSED FILES SAVED IN 'processed_data/' ===")
print(" - processed_data/regression_data.csv     :", df_regression.shape)
print(" - processed_data/kmeans_room_unscaled.csv:", df_rooms.shape)
print(" - processed_data/kmeans_room_scaled.csv  :", df_kmeans_scaled.shape)
print(" - processed_data/pca_scaled_data.csv     :", df_pca_scaled.shape)