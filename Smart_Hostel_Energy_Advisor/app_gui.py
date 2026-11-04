import os
import math
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from PIL import Image, ImageTk

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# 1. FILES, CONSTANTS & MODEL TRAINING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hostel_energy_dataset.csv")

# Project assumption, not a verified electricity billing rate.
TARIFF_RS = 7.50
MONTH_DAYS = 30

FEATURES = [
    "Occupancy",
    "Outdoor_Temp",
    "AC_Hours",
    "Fan_Hours",
    "Light_Hours",
    "Appliance_Hours",
    "Room_Capacity",
    "Weekend",
    "Floor",
]

TARGET = "Energy_kWh"

ROOM_COLS = [
    "Occupancy",
    "AC_Hours",
    "Fan_Hours",
    "Light_Hours",
    "Appliance_Hours",
    "Room_Capacity",
    "Energy_kWh",
]

# Equipment power assumptions used only for advisor estimates.
AC_KW = 1.20
FANS_KW = 0.15
LIGHTS_KW = 0.048
APPLIANCES_KW = 0.36
FAN_RETROFIT_SAVING_KW = 0.09

df = pd.read_csv(DATA_PATH)

required_columns = set(FEATURES + [TARGET, "Room_ID"])
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "Dataset is missing columns: "
        + ", ".join(sorted(missing_columns))
    )

X = df[FEATURES]
y = df[TARGET]

lr_model = LinearRegression().fit(X, y)

dt_model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
).fit(X, y)

df_rooms = df.groupby("Room_ID", as_index=False)[ROOM_COLS].mean()

if len(df_rooms) < 3:
    raise ValueError("At least three rooms are required for clustering.")

scaler_km = StandardScaler()
X_km_scaled = scaler_km.fit_transform(df_rooms[ROOM_COLS])

km_model = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=20,
    random_state=42
).fit(X_km_scaled)

df_rooms["Cluster_Raw"] = km_model.labels_

cluster_means = (
    df_rooms.groupby("Cluster_Raw")["Energy_kWh"]
    .mean()
    .sort_values()
)

cluster_map = {
    int(raw_cluster): tier
    for tier, raw_cluster in enumerate(cluster_means.index)
}

tier_means = {
    cluster_map[int(raw_cluster)]: float(mean_energy)
    for raw_cluster, mean_energy in cluster_means.items()
}

TIER_NAMES = {
    0: "Low usage",
    1: "Moderate usage",
    2: "High usage",
}


# ==========================================
# 2. WINDOW & THEME
# ==========================================
BG = "#f4f6f9"
CARD = "#ffffff"
PRIMARY = "#1f4e78"
PRIMARY_DARK = "#163a5c"
TEXT = "#253244"
MUTED = "#64748b"
BORDER = "#d6dee8"

root = tk.Tk()
root.title(
    "Smart Hostel Energy Advisor — "
    "ML Decision Support System (25BIT039)"
)
root.geometry("1120x850")
root.minsize(1000, 800)
root.configure(bg=BG)

style = ttk.Style(root)
style.theme_use("clam")

style.configure(
    "Card.TLabelframe",
    background=CARD,
    bordercolor=BORDER,
    relief="solid",
    borderwidth=1
)

style.configure(
    "Card.TLabelframe.Label",
    background=BG,
    foreground=PRIMARY,
    font=("Segoe UI", 10, "bold")
)

style.configure(
    "Card.TLabel",
    background=CARD,
    foreground=TEXT,
    font=("Segoe UI", 10)
)

style.configure(
    "TEntry",
    padding=3,
    fieldbackground=CARD
)

# Only the advisor row receives extra vertical space.
root.columnconfigure(0, weight=1)
root.rowconfigure(2, weight=1)


# ==========================================
# 3. HEADER
# ==========================================
header = tk.Frame(
    root,
    bg=PRIMARY,
    padx=18,
    pady=12
)
header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=16,
    pady=(12, 10)
)

tk.Label(
    header,
    text="SMART HOSTEL ENERGY ADVISOR",
    font=("Segoe UI", 15, "bold"),
    fg="white",
    bg=PRIMARY,
    anchor="w"
).pack(fill="x")

tk.Label(
    header,
    text=(
        "Jeet Dina  |  Roll No: 25BIT039  |  ICT(H2)  |  "
        "Problem A  |  Assumed tariff: ₹7.50/kWh"
    ),
    font=("Segoe UI", 9),
    fg="#e0eaf3",
    bg=PRIMARY,
    anchor="w"
).pack(fill="x", pady=(4, 0))


# ==========================================
# 4. INPUTS & PREDICTIONS
# ==========================================
top_frame = tk.Frame(root, bg=BG)
top_frame.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=16,
    pady=(0, 10)
)

top_frame.columnconfigure(0, weight=6, uniform="top")
top_frame.columnconfigure(1, weight=5, uniform="top")

form = ttk.LabelFrame(
    top_frame,
    text="  1. Room & Daily Usage Parameters  ",
    style="Card.TLabelframe",
    padding=(14, 10)
)
form.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 10)
)

results = ttk.LabelFrame(
    top_frame,
    text="  2. Predictions & Cost Analysis  ",
    style="Card.TLabelframe",
    padding=(16, 10)
)
results.grid(
    row=0,
    column=1,
    sticky="nsew"
)

field_layout = [
    ("Occupancy", "Occupancy (0–4)", "4", 0, 0),
    ("Outdoor_Temp", "Outdoor temp (°C)", "35.5", 1, 0),
    ("Room_Capacity", "Capacity (2/3/4)", "4", 2, 0),
    ("Weekend", "Weekend (0/1)", "0", 3, 0),
    ("Floor", "Floor (0–4)", "4", 4, 0),

    ("AC_Hours", "AC (hrs/day)", "8.0", 0, 2),
    ("Fan_Hours", "Fans (hrs/day)", "12.0", 1, 2),
    ("Light_Hours", "Lights (hrs/day)", "10.0", 2, 2),
    ("Appliance_Hours", "Appliances (hrs/day)", "7.0", 3, 2),
]

entries = {}

for feature, label_text, default_value, row, column in field_layout:
    ttk.Label(
        form,
        text=label_text,
        style="Card.TLabel",
        font=("Segoe UI", 9)
    ).grid(
        row=row,
        column=column,
        sticky="w",
        padx=(0, 8),
        pady=5
    )

    entry = ttk.Entry(
        form,
        width=8,
        font=("Segoe UI", 10)
    )
    entry.insert(0, default_value)
    entry.grid(
        row=row,
        column=column + 1,
        sticky="ew",
        padx=(0, 16 if column == 0 else 0),
        pady=5
    )

    entries[feature] = entry

form.columnconfigure(1, weight=1)
form.columnconfigure(3, weight=1)

ttk.Separator(form, orient="horizontal").grid(
    row=5,
    column=0,
    columnspan=4,
    sticky="ew",
    pady=(8, 8)
)

predict_btn = tk.Button(
    form,
    text="Predict Energy, Cost & Classify Room",
    font=("Segoe UI", 10, "bold"),
    bg=PRIMARY,
    fg="white",
    activebackground=PRIMARY_DARK,
    activeforeground="white",
    relief="flat",
    bd=0,
    padx=12,
    pady=8,
    cursor="hand2"
)
predict_btn.grid(
    row=6,
    column=0,
    columnspan=4,
    sticky="ew"
)

lr_var = tk.StringVar(value="--")
dt_var = tk.StringVar(value="--")
km_var = tk.StringVar(value="--")
status_var = tk.StringVar(value="Ready")

result_labels = []


def add_result_block(title, variable, color):
    tk.Label(
        results,
        text=title,
        font=("Segoe UI", 9, "bold"),
        fg=color,
        bg=CARD,
        anchor="w"
    ).pack(fill="x", pady=(5, 2))

    label = tk.Label(
        results,
        textvariable=variable,
        font=("Segoe UI", 10),
        fg=TEXT,
        bg=CARD,
        anchor="w",
        justify="left"
    )
    label.pack(fill="x", pady=(0, 7))
    result_labels.append(label)


add_result_block("LINEAR REGRESSION", lr_var, "#0b5394")
add_result_block("DECISION TREE · DEPTH 3", dt_var, "#b45309")
add_result_block("K-MEANS ROOM PROFILE", km_var, "#237442")

tk.Label(
    results,
    textvariable=status_var,
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=CARD,
    anchor="w",
    justify="left"
).pack(fill="x", pady=(3, 0))


def resize_result_labels(event):
    for label in result_labels:
        label.configure(wraplength=max(200, event.width - 40))


results.bind("<Configure>", resize_result_labels)


# ==========================================
# 5. ADVISOR — NO SCROLLING TEXT AREA
# ==========================================
tips_card = ttk.LabelFrame(
    root,
    text="  3. Advisor Recommendations & Cost-Saving Actions  ",
    style="Card.TLabelframe",
    padding=(14, 10)
)
tips_card.grid(
    row=2,
    column=0,
    sticky="nsew",
    padx=16,
    pady=(0, 10)
)

tips_header_var = tk.StringVar(value="Recommendations")

tips_header_lbl = tk.Label(
    tips_card,
    textvariable=tips_header_var,
    font=("Segoe UI", 10, "bold"),
    fg=PRIMARY,
    bg=CARD,
    anchor="w"
)
tips_header_lbl.pack(fill="x", pady=(0, 8))

tips_body = tk.Frame(
    tips_card,
    bg="#f8fafc",
    padx=12,
    pady=6
)
tips_body.pack(fill="both", expand=True)

tip_labels = []

# Five compact recommendation rows, displayed together.
for _ in range(5):
    label = tk.Label(
        tips_body,
        text="",
        font=("Segoe UI", 10),
        fg=TEXT,
        bg="#f8fafc",
        anchor="nw",
        justify="left",
        padx=0,
        pady=5
    )
    label.pack(fill="x")
    tip_labels.append(label)

tk.Label(
    tips_card,
    text=(
        "Equipment loads and savings are estimates using assumed power "
        "ratings. Monthly cost assumes the same usage for 30 days."
    ),
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=CARD,
    anchor="w",
    justify="left"
).pack(fill="x", pady=(7, 0))


def resize_advisor_labels(event):
    wrap_width = max(200, event.width - 28)
    for label in tip_labels:
        label.configure(wraplength=wrap_width)


tips_body.bind("<Configure>", resize_advisor_labels)


# ==========================================
# 6. PLOT VIEWER
# ==========================================
def open_image(relative_path, title, description):
    filepath = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(filepath):
        messagebox.showerror(
            "Plot File Not Found",
            f"Could not find:\n{filepath}\n\n"
            "Run your plot-generation scripts first.",
            parent=root
        )
        return

    try:
        with Image.open(filepath) as image:
            original_image = image.copy()
    except OSError as error:
        messagebox.showerror(
            "Unable to Open Plot",
            str(error),
            parent=root
        )
        return

    popup = tk.Toplevel(root)
    popup.title(title)
    popup.geometry("1200x700")
    popup.minsize(750, 450)
    popup.configure(bg=CARD)

    banner = tk.Frame(
        popup,
        bg=PRIMARY,
        padx=16,
        pady=12
    )
    banner.pack(fill="x")

    tk.Label(
        banner,
        text=title,
        font=("Segoe UI", 12, "bold"),
        fg="white",
        bg=PRIMARY,
        anchor="w"
    ).pack(fill="x")

    description_label = tk.Label(
        banner,
        text=description,
        font=("Segoe UI", 9),
        fg="#e0eaf3",
        bg=PRIMARY,
        anchor="w",
        justify="left"
    )
    description_label.pack(fill="x", pady=(4, 0))

    banner.bind(
        "<Configure>",
        lambda event: description_label.configure(
            wraplength=max(200, event.width - 32)
        )
    )

    image_label = tk.Label(
        popup,
        bg=CARD,
        bd=0
    )
    image_label.pack(
        fill="both",
        expand=True,
        padx=12,
        pady=12
    )

    def resize_image(event):
        available_width = max(1, event.width - 8)
        available_height = max(1, event.height - 8)

        width, height = original_image.size
        scale = min(
            available_width / width,
            available_height / height
        )

        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale))
        )

        resized = original_image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(resized, master=popup)

        image_label.configure(image=photo)
        image_label.image = photo

    image_label.bind("<Configure>", resize_image)


# ==========================================
# 7. FOUR COLORED PLOT BUTTONS
# ==========================================
plot_frame = ttk.LabelFrame(
    root,
    text="  4. Model Plots  ",
    style="Card.TLabelframe",
    padding=(10, 8)
)
plot_frame.grid(
    row=3,
    column=0,
    sticky="ew",
    padx=16,
    pady=(0, 12)
)

plot_buttons = [
    (
        "Linear Regression",
        "#175892",
        "plots/3_linear_regression_diagnostics.png",
        "Linear Regression Diagnostics",
        "Actual versus predicted energy and residual diagnostics."
    ),
    (
        "Decision Tree",
        "#b85b19",
        "plots/4_decision_tree_structure.png",
        "Decision Tree Structure",
        "View the decision rules and predicted energy at the leaf nodes."
    ),
    (
        "k-Means Clusters",
        "#287d44",
        "plots/6_kmeans_elbow_and_clusters.png",
        "k-Means Clusters & Elbow Plot",
        "View room-profile groups and the clustering elbow plot."
    ),
    (
        "PCA Scree & Biplot",
        "#70459a",
        "plots/7_pca_scree_and_biplot.png",
        "PCA Scree Plot & Biplot",
        "View explained variance and feature loadings from your PCA analysis."
    ),
]

for column, (
    button_text,
    color,
    filepath,
    title,
    description
) in enumerate(plot_buttons):
    plot_frame.columnconfigure(
        column,
        weight=1,
        uniform="plots"
    )

    tk.Button(
        plot_frame,
        text=button_text,
        font=("Segoe UI", 10, "bold"),
        bg=color,
        fg="white",
        activebackground=color,
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=10,
        pady=9,
        cursor="hand2",
        command=lambda p=filepath, t=title, d=description: open_image(
            p, t, d
        )
    ).grid(
        row=0,
        column=column,
        sticky="ew",
        padx=5
    )


# ==========================================
# 8. INPUT VALIDATION
# ==========================================
def read_inputs():
    values = {}

    for feature in FEATURES:
        raw_value = entries[feature].get().strip()

        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(
                f"Please enter a valid number for "
                f"{feature.replace('_', ' ')}."
            )

        if not math.isfinite(value):
            raise ValueError("All inputs must be finite numbers.")

        values[feature] = value

    integer_fields = [
        "Occupancy", "Room_Capacity", "Weekend", "Floor"
    ]

    for feature in integer_fields:
        if not values[feature].is_integer():
            raise ValueError(
                f"{feature.replace('_', ' ')} must be a whole number."
            )

    if not 0 <= values["Occupancy"] <= 4:
        raise ValueError("Occupancy must be between 0 and 4.")

    if values["Room_Capacity"] not in (2, 3, 4):
        raise ValueError("Room capacity must be 2, 3, or 4.")

    if values["Occupancy"] > values["Room_Capacity"]:
        raise ValueError("Occupancy cannot exceed room capacity.")

    if values["Weekend"] not in (0, 1):
        raise ValueError("Weekend must be 0 or 1.")

    if not 0 <= values["Floor"] <= 4:
        raise ValueError("Floor must be between 0 and 4.")

    if not 25 <= values["Outdoor_Temp"] <= 38:
        raise ValueError("Outdoor temperature must be 25–38°C.")

    for feature in [
        "AC_Hours",
        "Fan_Hours",
        "Light_Hours",
        "Appliance_Hours"
    ]:
        if not 0 <= values[feature] <= 24:
            raise ValueError("Usage hours must be between 0 and 24.")

    return values


# ==========================================
# 9. PREDICTION & COMPACT ADVISOR LOGIC
# ==========================================
def predict_energy():
    try:
        values = read_inputs()
    except ValueError as error:
        messagebox.showwarning(
            "Check Input",
            str(error),
            parent=root
        )
        return

    input_df = pd.DataFrame([values], columns=FEATURES)

    pred_lr = max(
        0.0,
        float(lr_model.predict(input_df)[0])
    )
    pred_dt = float(dt_model.predict(input_df)[0])

    lr_daily_cost = pred_lr * TARIFF_RS
    dt_daily_cost = pred_dt * TARIFF_RS

    km_input = pd.DataFrame(
        [{
            "Occupancy": values["Occupancy"],
            "AC_Hours": values["AC_Hours"],
            "Fan_Hours": values["Fan_Hours"],
            "Light_Hours": values["Light_Hours"],
            "Appliance_Hours": values["Appliance_Hours"],
            "Room_Capacity": values["Room_Capacity"],
            "Energy_kWh": pred_lr
        }],
        columns=ROOM_COLS
    )

    raw_cluster = int(
        km_model.predict(
            scaler_km.transform(km_input)
        )[0]
    )
    tier = cluster_map[raw_cluster]
    tier_mean = tier_means[tier]

    lr_var.set(
        f"{pred_lr:.2f} kWh/day  |  ₹{lr_daily_cost:.2f}/day\n"
        f"₹{lr_daily_cost * MONTH_DAYS:.2f}/month"
    )

    dt_var.set(
        f"{pred_dt:.2f} kWh/day  |  ₹{dt_daily_cost:.2f}/day\n"
        f"₹{dt_daily_cost * MONTH_DAYS:.2f}/month"
    )

    km_var.set(
        f"{TIER_NAMES[tier]} · Cluster {tier}\n"
        f"Group average: {tier_mean:.2f} kWh/day "
        f"(₹{tier_mean * TARIFF_RS:.2f}/day)"
    )

    tips = []

    if values["Occupancy"] == 0:
        tips.append(
            f"UNOCCUPIED ROOM — Predicted use: {pred_lr:.2f} kWh/day "
            f"(₹{lr_daily_cost:.2f}/day). Check for equipment left running."
        )
    else:
        per_student = pred_lr / values["Occupancy"]
        tips.append(
            f"PER STUDENT — {per_student:.2f} kWh/day "
            f"(₹{per_student * TARIFF_RS:.2f}/day), shared by "
            f"{int(values['Occupancy'])} occupants."
        )

    if values["AC_Hours"] > 5:
        ac_energy = values["AC_Hours"] * AC_KW
        daily_saving = 1.5 * AC_KW * TARIFF_RS

        tips.append(
            f"AC — {values['AC_Hours']:.1f} hrs/day ≈ "
            f"{ac_energy:.2f} kWh. Cutting runtime by 1.5 hours "
            f"could save ₹{daily_saving:.2f}/day "
            f"or ₹{daily_saving * MONTH_DAYS:.0f}/month."
        )

    if values["Fan_Hours"] > 10 or values["Light_Hours"] > 8:
        auxiliary_energy = (
            values["Fan_Hours"] * FANS_KW
            + values["Light_Hours"] * LIGHTS_KW
        )
        fan_saving = (
            values["Fan_Hours"]
            * FAN_RETROFIT_SAVING_KW
            * TARIFF_RS
        )

        tips.append(
            f"FANS & LIGHTS — Estimated use: "
            f"{auxiliary_energy:.2f} kWh/day. Replacing two 75 W fans "
            f"with two 30 W fans could save ₹{fan_saving:.2f}/day."
        )

    if values["Appliance_Hours"] > 4:
        appliance_energy = (
            values["Appliance_Hours"] * APPLIANCES_KW
        )
        tips.append(
            f"APPLIANCES — Estimated use: "
            f"{appliance_energy:.2f} kWh/day "
            f"(₹{appliance_energy * TARIFF_RS:.2f}/day). "
            "Reduce unnecessary runtime and switch off unused devices."
        )

    if tier == 2:
        heading = "HIGH USAGE — Prioritize load checks"
        foreground = "#9f2525"
        background = "#fff3f3"
        tips.append(
            "ROOM PROFILE — High-usage group. Review AC schedules "
            "and measure equipment loads before choosing upgrades."
        )
    elif tier == 1:
        heading = "MODERATE USAGE — Review daily runtime"
        foreground = "#926019"
        background = "#fff9ed"
        tips.append(
            "ROOM PROFILE — Moderate-usage group. Start with the "
            "largest loads and avoid unnecessary equipment runtime."
        )
    else:
        heading = "LOW USAGE — Continue monitoring"
        foreground = "#27633c"
        background = "#f0f8f2"
        tips.append(
            "ROOM PROFILE — Low-usage group. Maintain sensible "
            "equipment schedules and monitor changes in consumption."
        )

    tips_header_var.set(heading)
    tips_header_lbl.configure(fg=foreground)
    tips_body.configure(bg=background)

    for index, label in enumerate(tip_labels):
        label.configure(bg=background)

        if index < len(tips):
            label.configure(text=f"•  {tips[index]}")
            if not label.winfo_manager():
                label.pack(fill="x")
        else:
            label.configure(text="")
            label.pack_forget()

    status_var.set("Results match the current inputs.")


# ==========================================
# 10. TRACK EDITS & START APPLICATION
# ==========================================
def mark_inputs_changed(event=None):
    status_var.set("Inputs changed — click Predict to update results.")


for entry in entries.values():
    entry.bind("<KeyRelease>", mark_inputs_changed)

predict_btn.configure(command=predict_energy)

# Enter runs a prediction without moving away from the inputs.
root.bind("<Return>", lambda event: predict_energy())

predict_energy()
root.mainloop()