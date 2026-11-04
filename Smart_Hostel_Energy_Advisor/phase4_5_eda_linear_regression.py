import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create folder for report plots
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load the regression dataset prepared in Phase 3
df = pd.read_csv("processed_data/regression_data.csv")

# ==========================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("=== PHASE 4: CORRELATIONS WITH Energy_kWh ===")
corr_matrix = df.corr()
print(corr_matrix["Energy_kWh"].sort_values(ascending=False).round(4), "\n")

# Plot 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Hostel Energy Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/1_correlation_heatmap.png", dpi=300)
plt.close()

# Plot 2: EDA Overview (AC Hours vs Energy & Weekday vs Weekend)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.scatterplot(
    data=df, x="AC_Hours", y="Energy_kWh", hue="Occupancy",
    palette="viridis", s=60, alpha=0.85, ax=axes[0]
)
axes[0].set_title("Daily AC Usage vs. Energy Consumption", fontsize=12, fontweight="bold")
axes[0].set_xlabel("AC Usage (Hours/Day)")
axes[0].set_ylabel("Energy Consumption (kWh)")

sns.boxplot(
    data=df, x="Weekend", y="Energy_kWh", hue="Weekend",
    palette="Set2", legend=False, ax=axes[1]
)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Weekday (0)", "Weekend (1)"])
axes[1].set_title("Weekday vs. Weekend Energy Consumption", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Day Type")
axes[1].set_ylabel("Energy Consumption (kWh)")

plt.tight_layout()
plt.savefig("plots/2_eda_patterns.png", dpi=300)
plt.close()

# ==========================================
# PHASE 5: LINEAR REGRESSION (OBJECTIVE 1)
# ==========================================
features = [
    "Occupancy", "Outdoor_Temp", "AC_Hours", "Fan_Hours",
    "Light_Hours", "Appliance_Hours", "Room_Capacity", "Weekend", "Floor"
]
target = "Energy_kWh"

X = df[features]
y = df[target]

# 80% Train (400 rows), 20% Test (100 rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

print("=== PHASE 5: LINEAR REGRESSION COEFFICIENTS ===")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient (Weight)": np.round(lr_model.coef_, 4)
}).sort_values(by="Coefficient (Weight)", ascending=False)
print(coef_df.to_string(index=False))
print(f"Intercept (b0): {lr_model.intercept_:.4f}\n")

print("=== LINEAR REGRESSION EVALUATION METRICS ===")
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Train Set -> MSE: {train_mse:.4f} | RMSE: {np.sqrt(train_mse):.4f} kWh | MAE: {mean_absolute_error(y_train, y_train_pred):.4f} kWh | R2: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test Set  -> MSE: {test_mse:.4f} | RMSE: {np.sqrt(test_mse):.4f} kWh | MAE: {mean_absolute_error(y_test, y_test_pred):.4f} kWh | R2: {r2_score(y_test, y_test_pred):.4f}\n")

# Plot 3: Actual vs Predicted & Residuals (Test Set)
residuals = y_test - y_test_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_test_pred, color="#1f77b4", alpha=0.75, edgecolors="k")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit (y = ŷ)")
axes[0].set_title(f"Linear Regression: Actual vs. Predicted (Test R² = {r2_score(y_test, y_test_pred):.4f})", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Actual Energy (kWh)")
axes[0].set_ylabel("Predicted Energy (kWh)")
axes[0].legend()

axes[1].scatter(y_test_pred, residuals, color="#ff7f0e", alpha=0.75, edgecolors="k")
axes[1].axhline(0, color="red", linestyle="--", lw=2)
axes[1].set_title("Residual Plot (Predicted vs. Error)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted Energy (kWh)")
axes[1].set_ylabel("Residual Error (y - ŷ) [kWh]")

plt.tight_layout()
plt.savefig("plots/3_linear_regression_diagnostics.png", dpi=300)
plt.close()

print("Saved 3 high-resolution figures in 'plots/' folder:")
print(" - plots/1_correlation_heatmap.png")
print(" - plots/2_eda_patterns.png")
print(" - plots/3_linear_regression_diagnostics.png")