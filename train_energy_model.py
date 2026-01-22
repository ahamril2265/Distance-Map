import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DATA_FILE = "output/edges_with_energy.csv"
MODEL_FILE = "output/energy_model.joblib"

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv(DATA_FILE)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
NUMERIC_FEATURES = [
    "length",
    "elevation_gain",
    "elevation_loss",
    "grade_abs",
    "slope_percent"
]

CATEGORICAL_FEATURES = [
    "highway"
]

TARGET = "energy_wh"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", model),
    ]
)

# ---------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
pipeline.fit(X_train, y_train)

# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Training complete")
print(f"MAE (Wh): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
joblib.dump(pipeline, MODEL_FILE)
print(f"💾 Model saved → {MODEL_FILE}")
