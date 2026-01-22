import pandas as pd
import numpy as np

# ---------------------------------------------------
# CONFIG (EV PARAMETERS)
# ---------------------------------------------------
MASS_KG = 1600          # vehicle mass
C_RR = 0.015            # rolling resistance
G = 9.81                # gravity (m/s^2)
SLOPE_ALPHA = 1.5       # slope inefficiency factor

INPUT_FILE = "output/edges.csv"
OUTPUT_FILE = "output/edges_with_energy.csv"

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# Safety checks
required_cols = [
    "length",
    "elevation_gain",
    "grade_abs"
]

for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"Missing required column: {col}")

# ---------------------------------------------------
# ENERGY CALCULATION (JOULES)
# ---------------------------------------------------
# Rolling resistance
df["E_roll_J"] = C_RR * MASS_KG * G * df["length"]

# Gravitational (uphill only)
df["E_grav_J"] = MASS_KG * G * df["elevation_gain"]

# Slope inefficiency
df["E_slope_J"] = SLOPE_ALPHA * df["grade_abs"] * df["length"]

# Total energy
df["energy_joules"] = (
    df["E_roll_J"] +
    df["E_grav_J"] +
    df["E_slope_J"]
)

# Convert to Wh
df["energy_wh"] = df["energy_joules"] / 3600.0

# ---------------------------------------------------
# CLEANUP (OPTIONAL BUT RECOMMENDED)
# ---------------------------------------------------
df.drop(columns=["E_roll_J", "E_grav_J", "E_slope_J"], inplace=True)

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Energy labels generated")
print(f"Saved → {OUTPUT_FILE}")
print("Sample:")
print(df[[
    "length",
    "elevation_gain",
    "grade_abs",
    "energy_wh"
]].head())
