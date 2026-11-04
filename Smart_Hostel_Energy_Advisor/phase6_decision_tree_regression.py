import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load preprocessed regression dataset
df = pd.read_csv("processed_data/regression_data.csv")

features = [
    "Occupancy", "Outdoor_Temp", "AC_Hours", "Fan_Hours",
    "Light_Hours", "Appliance_Hours", "Room_Capacity", "Weekend", "Floor"
]
target = "Energy_kWh"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ==============================================================================
# 1. SLIDE 6 & 8 VERIFICATION: ROOT NODE VARIANCE & FIRST SPLIT VARIANCE REDUCTION
# ==============================================================================
dt_root_check = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=42)
dt_root_check.fit(X_train, y_train)

tree_ = dt_root_check.tree_
root_feat_name = features[tree_.feature[0]]
root_threshold = tree_.threshold[0]

n = len(y_train)
root_mean = y_train.mean()
root_var = np.var(y_train)  # Divisor n (matches Slide 4 & 6)

left_mask = X_train[root_feat_name] <= root_threshold
y_L = y_train[left_mask]
y_R = y_train[~left_mask]

n_L, n_R = len(y_L), len(y_R)
mean_L, mean_R = y_L.mean(), y_R.mean()
var_L, var_R = np.var(y_L), np.var(y_R)

weighted_child_var = (n_L / n) * var_L + (n_R / n) * var_R
variance_reduction = root_var - weighted_child_var

print("=== 1. CART VARIANCE REDUCTION CHECK (ROOT SPLIT) ===")
print(f"Root Node (n={n})       : Mean ȳ = {root_mean:.4f} kWh | Variance (MSE) = {root_var:.4f}")
print(f"Best Root Split Chosen   : {root_feat_name} <= {root_threshold:.2f} hours")
print(f"  -> Left Child  (n_L={n_L}): Mean ȳ_L = {mean_L:.4f} kWh | Var(S_L) = {var_L:.4f}")
print(f"  -> Right Child (n_R={n_R}): Mean ȳ_R = {mean_R:.4f} kWh | Var(S_R) = {var_R:.4f}")
print(f"Weighted Child Variance  : ({n_L}/{n})*{var_L:.4f} + ({n_R}/{n})*{var_R:.4f} = {weighted_child_var:.4f}")
print(f"Variance Reduction (VR)  : {root_var:.4f} - {weighted_child_var:.4f} = {variance_reduction:.4f}\n")

# ==============================================================================
# 2. SLIDE 17 VERIFICATION: DEPTH COMPARISON & OVERFITTING ANALYSIS
# ==============================================================================
print("=== 2. DEPTH VS. OVERFITTING TABLE (SLIDE 17 STYLE) ===")
depths = [1, 2, 3, 4, 5, 6, None]
depth_records = [{
    "Max_Depth": "0 (Root Only)",
    "Leaves": 1,
    "Train_MSE": round(root_var, 4),
    "Train_R2": 0.0000,
    "Test_MSE": round(mean_squared_error(y_test, np.full_like(y_test, root_mean)), 4),
    "Test_R2": round(r2_score(y_test, np.full_like(y_test, root_mean)), 4)
}]

for d in depths:
    dt = DecisionTreeRegressor(criterion="squared_error", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    tr_pred = dt.predict(X_train)
    te_pred = dt.predict(X_test)
    depth_records.append({
        "Max_Depth": str(d) if d is not None else "None (Full)",
        "Leaves": dt.get_n_leaves(),
        "Train_MSE": round(mean_squared_error(y_train, tr_pred), 4),
        "Train_R2": round(r2_score(y_train, tr_pred), 4),
        "Test_MSE": round(mean_squared_error(y_test, te_pred), 4),
        "Test_R2": round(r2_score(y_test, te_pred), 4)
    })

df_depths = pd.DataFrame(depth_records)
print(df_depths.to_string(index=False), "\n")

# ==============================================================================
# 3. INTERPRETABLE TREE (DEPTH = 3) RULES & FEATURE IMPORTANCE
# ==============================================================================
dt_interpretable = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=42)
dt_interpretable.fit(X_train, y_train)

print("=== 3. DECISION RULES FOR INTERPRETABLE TREE (MAX_DEPTH = 3) ===")
print(export_text(dt_interpretable, feature_names=features))

# Also fit Depth = 4 for richer feature importance comparison
dt_best = DecisionTreeRegressor(criterion="squared_error", max_depth=4, random_state=42)
dt_best.fit(X_train, y_train)

print("=== 4. FEATURE IMPORTANCES (VARIANCE REDUCTION SHARE AT DEPTH = 4) ===")
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": np.round(dt_best.feature_importances_, 4)
}).sort_values(by="Importance", ascending=False)
print(imp_df.to_string(index=False), "\n")

# ==============================================================================
# 4. GENERATE HIGH-RESOLUTION PLOTS FOR REPORT
# ==============================================================================

# Plot 4: Visual Flowchart of the Decision Tree (Depth = 3)
plt.figure(figsize=(18, 9))
plot_tree(
    dt_interpretable,
    feature_names=features,
    filled=True,
    rounded=True,
    fontsize=10,
    precision=2
)
plt.title("Decision Tree Regressor (Max Depth = 3, 8 Leaves) — Hostel Energy Rules", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/4_decision_tree_structure.png", dpi=300)
plt.close()

# Plot 5: Overfitting Curve (Train vs Test MSE across Depths) & Staircase Comparison (Slide 16)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left Subplot: Train vs Test MSE by Depth
plot_depths = ["1", "2", "3", "4", "5", "6", "Full"]
train_mses = [r["Train_MSE"] for r in depth_records[1:]]
test_mses = [r["Test_MSE"] for r in depth_records[1:]]

axes[0].plot(plot_depths, train_mses, marker="o", lw=2, label="Training MSE", color="#1f77b4")
axes[0].plot(plot_depths, test_mses, marker="s", lw=2, label="Test MSE", color="#d62728")
axes[0].set_title("Effect of Tree Depth on MSE (Overfitting Analysis)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Maximum Tree Depth")
axes[0].set_ylabel("Mean Squared Error (MSE)")
axes[0].legend()

# Right Subplot: Slide 16 Style 1D Staircase (Decision Tree Depth=2 vs Linear Line on AC_Hours)
X_ac = X_train[["AC_Hours"]]
dt_1d = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X_ac, y_train)
lr_1d = LinearRegression().fit(X_ac, y_train)

ac_grid = np.linspace(0, 11.5, 300).reshape(-1, 1)
ac_grid_df = pd.DataFrame(ac_grid, columns=["AC_Hours"])

axes[1].scatter(X_train["AC_Hours"], y_train, color="gray", alpha=0.35, s=30, label="Training Data")
axes[1].plot(ac_grid, lr_1d.predict(ac_grid_df), color="#1f77b4", lw=2.5, label="Linear Fit (Straight Line)")
axes[1].plot(ac_grid, dt_1d.predict(ac_grid_df), color="#ff7f0e", lw=2.8, label="Regression Tree (Depth=2 Staircase)")
axes[1].set_title("Linear Regression vs. Regression Tree Staircase (AC_Hours)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Daily AC Usage (Hours)")
axes[1].set_ylabel("Energy Consumption (kWh)")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/5_tree_overfitting_and_staircase.png", dpi=300)
plt.close()

print("Saved 2 high-resolution figures in 'plots/' folder:")
print(" - plots/4_decision_tree_structure.png")
print(" - plots/5_tree_overfitting_and_staircase.png")