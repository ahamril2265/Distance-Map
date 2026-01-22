import pandas as pd

# ---------------------------------------------------
# LOAD EDGE DATA
# ---------------------------------------------------
edges = pd.read_csv("output/edges.csv")

# ---------------------------------------------------
# CLEAN / ENCODE FEATURES
# ---------------------------------------------------

# Handle missing highway types
edges["highway"] = edges["highway"].fillna("unknown")

# One-hot encode road type
highway_dummies = pd.get_dummies(edges["highway"], prefix="road")
edges = pd.concat([edges, highway_dummies], axis=1)

# ---------------------------------------------------
# BASELINE ENERGY MODEL (synthetic labels)
# ---------------------------------------------------
A = 1.0    # distance weight
B = 10.0   # elevation gain penalty
C = 2.0    # elevation loss penalty

edges["energy_cost"] = (
    A * edges["length"] +
    B * edges["elevation_gain"] +
    C * edges["elevation_loss"]
)

# ---------------------------------------------------
# FINAL FEATURE SET
# ---------------------------------------------------
feature_cols = [
    "length",
    "elevation_gain",
    "elevation_loss",
    "slope_percent",
    "grade_abs"
] + list(highway_dummies.columns)

X = edges[feature_cols]
y = edges["energy_cost"]

# ---------------------------------------------------
# SAVE DATASETS
# ---------------------------------------------------
X.to_csv("output/X_features.csv", index=False)
y.to_csv("output/y_energy.csv", index=False)

print("✅ ML dataset prepared")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
