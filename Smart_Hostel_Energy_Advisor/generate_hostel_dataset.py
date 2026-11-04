import numpy as np
import pandas as pd

# Fixed seed makes the generated dataset reproducible.
rng = np.random.default_rng(42)

number_of_rooms = 50
number_of_days = 10

room_ids = [f"R{i:03d}" for i in range(1, number_of_rooms + 1)]
room_capacities = rng.choice([2, 3, 4], size=number_of_rooms, p=[0.25, 0.50, 0.25])
room_floors = rng.integers(0, 5, size=number_of_rooms)
dates = pd.date_range("2026-09-01", periods=number_of_days, freq="D")

records = []

for day_index, date in enumerate(dates):
    daily_temperature = 30 + 3 * np.sin(day_index / 2) + rng.normal(0, 0.8)
    weekend = int(date.dayofweek >= 5)

    for room_index, room_id in enumerate(room_ids):
        capacity = int(room_capacities[room_index])
        floor = int(room_floors[room_index])

        # Weekend occupancy is assumed to be lower.
        probability = 0.78 if weekend == 0 else 0.52
        occupancy = int(rng.binomial(capacity, probability))

        if occupancy == 0:
            ac_hours = fan_hours = light_hours = appliance_hours = 0.0
        else:
            # Usage hours are generated with realistic ranges and random variation.
            ac_base = 2.0 + max(0, daily_temperature - 25) * 0.48
            ac_hours = float(np.clip(rng.normal(ac_base + 0.35 * occupancy, 1.4), 0, 12))
            fan_hours = float(np.clip(rng.normal(8 + 0.8 * occupancy, 2), 0, 18))
            light_hours = float(np.clip(rng.normal(5 + occupancy, 1.5), 0, 14))
            appliance_hours = float(np.clip(rng.normal(1.5 + 0.9 * occupancy, 1.2), 0, 10))

        # Assumed power ratings: AC 1.2 kW; two fans 0.075 kW each;
        # four lights 0.012 kW each; two other appliances 0.18 kW each.
        energy = (
            ac_hours * 1.20
            + fan_hours * 2 * 0.075
            + light_hours * 4 * 0.012
            + appliance_hours * 2 * 0.18
            + rng.normal(0, 0.35)
        )

        records.append({
            "Room_ID": room_id,
            "Date": date.strftime("%Y-%m-%d"),
            "Occupancy": occupancy,
            "Outdoor_Temp": round(daily_temperature, 1),
            "AC_Hours": round(ac_hours, 2),
            "Fan_Hours": round(fan_hours, 2),
            "Light_Hours": round(light_hours, 2),
            "Appliance_Hours": round(appliance_hours, 2),
            "Weekend": weekend,
            "Room_Capacity": capacity,
            "Floor": floor,
            "Energy_kWh": round(max(0, energy), 2)
        })

df = pd.DataFrame(records)
df.to_csv("hostel_energy_dataset.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
