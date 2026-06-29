"""
Create sliding window sequences for forecasting
MATCHES PROPOSAL: Predict single timestep 1 hour ahead
- Input: 12 timesteps (1 hour history)
- Output: 1 timestep at t+12 (1 hour future)
- Features: 8 parameters (CO, CO2, PM25, PM10, NO2, ozone, temp, humidity)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

# =====================================================
# PATH SETUP — dari config.py backend
# =====================================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
BASE_DIR   = Path(Config.BASE_DIR)
DATA_DIR   = Path(Config.DATA_FOLDER)
MODELS_DIR = Path(Config.MODEL_FOLDER)

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[MLP] 📂 Working directory: {Path.cwd()}")
print(f"[MLP] 📂 BASE_DIR  : {BASE_DIR}")
print(f"[MLP] 📂 DATA_DIR  : {DATA_DIR}")
print(f"[MLP] 📂 MODELS_DIR: {MODELS_DIR}")

# =====================================================
# CONFIG
# =====================================================
LOOKBACK = 12       # 12 timesteps × 5 min = 1 hour history
HORIZON  = 12       # Predict 12 steps ahead = 1 hour future
FEATURES = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'temp', 'humidity']

print(f"\n[MLP] ⚙️ Configuration:")
print(f"[MLP]    Lookback window   : {LOOKBACK} timesteps (1 hour)")
print(f"[MLP]    Prediction horizon: {HORIZON} timesteps ahead (1 hour)")
print(f"[MLP]    Features          : {FEATURES}")
print(f"[MLP]    Input size        : {LOOKBACK * len(FEATURES)} = 96 nodes (FLAT)")
print(f"[MLP]    Output size       : {len(FEATURES)} = 8 nodes (single timestep)")

# =====================================================
# LOAD DATA
# =====================================================
csv_path = DATA_DIR / "sensor_data_raw.csv"

if not csv_path.exists():
    print(f"\n[MLP] ❌ ERROR: File not found: {csv_path}")
    print(f"[MLP] ❌ Run export_data.py first!")
    exit(1)

df = pd.read_csv(csv_path)
print(f"\n[MLP] 📂 Loaded from {csv_path}")
print(f"[MLP] 📊 Total samples: {len(df)}")

# =====================================================
# DATA VALIDATION
# =====================================================
print("\n[MLP] 🔍 Data Validation:")
missing = df[FEATURES].isnull().sum().sum()
print(f"[MLP]    Missing values: {missing}")

if missing > 0:
    print(f"[MLP]    Removing {missing} missing values...")
    df = df.dropna(subset=FEATURES)

# Check negative values (temp boleh minus)
NON_NEGATIVE = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'humidity']
for col in NON_NEGATIVE:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"[MLP]    ⚠️ {col}: {neg_count} negative values found")

df = df[(df[NON_NEGATIVE] >= 0).all(axis=1)]
print(f"[MLP] ✅ After validation: {len(df)} samples")

min_required = LOOKBACK + HORIZON
if len(df) < min_required:
    print(f"\n[MLP] ❌ ERROR: Insufficient data!")
    print(f"[MLP]    Need at least {min_required} samples (lookback + horizon)")
    print(f"[MLP]    Got only {len(df)} samples")
    exit(1)

data = df[FEATURES].values
print(f"[MLP] 📐 Data shape: {data.shape}")

# =====================================================
# CREATE SEQUENCES (SINGLE-STEP OUTPUT)
# =====================================================
def create_sequences_single_step(data, lookback, horizon):
    """
    Create sequences for single-step prediction
    Returns:
        X: (n_sequences, lookback, n_features) - input sequences
        y: (n_sequences, n_features) - target at t+horizon
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback+horizon-1])
    return np.array(X), np.array(y)

print(f"\n[MLP] 🔄 Creating sequences...")
print(f"[MLP]    Strategy: Predict value at t+{HORIZON} using t-{LOOKBACK-1} to t")
X, y = create_sequences_single_step(data, LOOKBACK, HORIZON)

print(f"\n[MLP] ✅ Sequences created:")
print(f"[MLP]    X shape: {X.shape} (n_sequences, lookback, n_features)")
print(f"[MLP]    y shape: {y.shape} (n_sequences, n_features)")

if len(X) == 0:
    print(f"\n[MLP] ❌ ERROR: No sequences created!")
    exit(1)

assert X.shape[1] == LOOKBACK,      f"Lookback mismatch: {X.shape[1]} != {LOOKBACK}"
assert X.shape[2] == len(FEATURES), f"Features mismatch: {X.shape[2]} != {len(FEATURES)}"
assert y.shape[1] == len(FEATURES), f"Output features mismatch: {y.shape[1]} != {len(FEATURES)}"
print(f"[MLP] ✅ Shape verification passed")

# =====================================================
# FLATTEN INPUT FOR MLP
# =====================================================
X_flat = X.reshape(X.shape[0], -1)  # (n_samples, 96)

print(f"\n[MLP] 📏 Flattened for MLP input:")
print(f"[MLP]    X_flat shape: {X_flat.shape} → 96 input nodes (12 timesteps × 8 features)")
print(f"[MLP]    y shape     : {y.shape} → 8 output nodes")

# =====================================================
# TRAIN TEST SPLIT (TIME SERIES SAFE - NO SHUFFLE)
# =====================================================
split_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=split_ratio, shuffle=False
)

print(f"\n[MLP] 🔀 Train/Test Split (80/20, no shuffle):")
print(f"[MLP]    Train samples: {len(X_train)}")
print(f"[MLP]    Test samples : {len(X_test)}")
print(f"[MLP]    Total        : {len(X_train) + len(X_test)}")

# =====================================================
# NORMALIZATION
# MLP scaler_X: fit on (n, 96) → 96 means (one per timestep-feature combo)
# =====================================================
print(f"\n[MLP] 📊 Normalizing data (StandardScaler — fit on flattened 96 features)...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

print(f"[MLP] ✅ Normalization complete")
print(f"[MLP]    scaler_X: mean shape = {scaler_X.mean_.shape} (96 timestep-feature positions)")
print(f"[MLP]    scaler_y: mean shape = {scaler_y.mean_.shape} (8 features)")

# =====================================================
# SCALER VERIFICATION
# =====================================================
test_sample_inverse = scaler_y.inverse_transform(y_train_scaled[0:1])
reconstruction_error = np.abs(y_train[0:1] - test_sample_inverse).mean()

print(f"\n[MLP] 🧪 Inverse Transform Test:")
print(f"[MLP]    Reconstruction error: {reconstruction_error:.6f}")
if reconstruction_error > 0.01:
    print(f"[MLP]    ⚠️ Warning: High reconstruction error!")
else:
    print(f"[MLP]    ✅ Inverse transform working correctly")

# =====================================================
# SAVE DATASETS
# =====================================================
print(f"\n[MLP] 💾 Saving MLP datasets...")
np.save(DATA_DIR / "X_train_mlp.npy", X_train_scaled)
np.save(DATA_DIR / "X_test_mlp.npy",  X_test_scaled)
np.save(DATA_DIR / "y_train_mlp.npy", y_train_scaled)
np.save(DATA_DIR / "y_test_mlp.npy",  y_test_scaled)
print(f"[MLP] ✅ Datasets saved to {DATA_DIR}")

# =====================================================
# SAVE SCALERS
# =====================================================
joblib.dump(scaler_X, MODELS_DIR / "scaler_X_mlp.pkl")
joblib.dump(scaler_y, MODELS_DIR / "scaler_y_mlp.pkl")
print(f"[MLP] ✅ Scalers saved: scaler_X_mlp.pkl, scaler_y_mlp.pkl")

# =====================================================
# SAVE CONFIGURATION
# =====================================================
config = {
    'model_type':    'MLP',
    'lookback':      LOOKBACK,
    'horizon':       HORIZON,
    'features':      FEATURES,
    'input_size':    LOOKBACK * len(FEATURES),
    'output_size':   len(FEATURES),
    'input_format':  'flat_2D',
    'train_samples': len(X_train),
    'test_samples':  len(X_test),
    'total_samples': len(X_train) + len(X_test)
}

config_path = DATA_DIR / "preprocessing_config_mlp.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"[MLP] ✅ Config saved to {config_path}")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*60)
print("[MLP] ✅ MLP PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n[MLP] 📊 Dataset Summary:")
print(f"[MLP]    Total sequences : {len(X_train) + len(X_test)}")
print(f"[MLP]    Train/Test split: {len(X_train)}/{len(X_test)}")
print(f"[MLP]    Input shape     : (batch, 96) — FLAT 12 timesteps × 8 features")
print(f"[MLP]    Output shape    : (batch, 8)  — single timestep prediction")
print(f"[MLP]    Prediction target: Values at t+{HORIZON} (1 hour ahead)")

print(f"\n[MLP] 📁 Files created:")
print(f"[MLP]    ✅ {DATA_DIR / 'X_train_mlp.npy'}")
print(f"[MLP]    ✅ {DATA_DIR / 'X_test_mlp.npy'}")
print(f"[MLP]    ✅ {DATA_DIR / 'y_train_mlp.npy'}")
print(f"[MLP]    ✅ {DATA_DIR / 'y_test_mlp.npy'}")
print(f"[MLP]    ✅ {MODELS_DIR / 'scaler_X_mlp.pkl'}")
print(f"[MLP]    ✅ {MODELS_DIR / 'scaler_y_mlp.pkl'}")
print(f"[MLP]    ✅ {config_path}")

print(f"\n[MLP] 🚀 Next step: python train_MLP.py")
print("="*60)