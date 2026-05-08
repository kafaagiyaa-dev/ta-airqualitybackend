"""
Create sliding window sequences for LSTM forecasting
MATCHES PROPOSAL (Tabel 3.5):
- Input: (12, 4) shape — 12 timesteps × 4 features (3D, NOT flattened)
- Output: 1 timestep at t+12 (1 hour future)
- Scaler: fit per-feature (not per-timestep-position like MLP)
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

print(f"[LSTM] 📂 Working directory: {Path.cwd()}")
print(f"[LSTM] 📂 BASE_DIR  : {BASE_DIR}")
print(f"[LSTM] 📂 DATA_DIR  : {DATA_DIR}")
print(f"[LSTM] 📂 MODELS_DIR: {MODELS_DIR}")

# =====================================================
# CONFIG (MATCHES PROPOSAL - SAME AS MLP)
# =====================================================
LOOKBACK  = 12       # 12 timesteps × 5 min = 1 hour history
HORIZON   = 12       # Predict 12 steps ahead = 1 hour future
FEATURES  = ['CO', 'CO2', 'PM25', 'PM10']

print(f"\n[LSTM] ⚙️ Configuration:")
print(f"[LSTM]    Lookback window  : {LOOKBACK} timesteps (1 hour)")
print(f"[LSTM]    Prediction horizon: {HORIZON} timesteps ahead (1 hour)")
print(f"[LSTM]    Features         : {FEATURES}")
print(f"[LSTM]    Input shape      : ({LOOKBACK}, {len(FEATURES)}) = 3D tensor (NOT flat)")
print(f"[LSTM]    Output size      : {len(FEATURES)} nodes (single timestep)")
print(f"\n[LSTM]    NOTE: LSTM uses 3D input  (n, 12, 4) — berbeda dari MLP")
print(f"[LSTM]          MLP  uses 2D input  (n, 48)   — flat")
print(f"[LSTM]          Scaler LSTM: per-feature (4 means)")
print(f"[LSTM]          Scaler MLP : per-timestep-feature (48 means)")

# =====================================================
# LOAD DATA (SAME CSV AS MLP)
# =====================================================
csv_path = DATA_DIR / "sensor_data_raw.csv"

if not csv_path.exists():
    print(f"\n[LSTM] ❌ ERROR: File not found: {csv_path}")
    print(f"[LSTM] ❌ Run export_data.py first!")
    exit(1)

df = pd.read_csv(csv_path)
print(f"\n[LSTM] 📂 Loaded from {csv_path}")
print(f"[LSTM] 📊 Total samples: {len(df)}")

# =====================================================
# DATA VALIDATION
# =====================================================
print("\n[LSTM] 🔍 Data Validation:")
missing = df[FEATURES].isnull().sum().sum()
print(f"[LSTM]    Missing values: {missing}")

if missing > 0:
    print(f"[LSTM]    Removing {missing} missing values...")
    df = df.dropna(subset=FEATURES)

df = df[(df[FEATURES] >= 0).all(axis=1)]
print(f"[LSTM] ✅ After validation: {len(df)} samples")

min_required = LOOKBACK + HORIZON
if len(df) < min_required:
    print(f"\n[LSTM] ❌ ERROR: Insufficient data!")
    print(f"[LSTM]    Need at least {min_required} samples")
    print(f"[LSTM]    Got only {len(df)} samples")
    exit(1)

data = df[FEATURES].values
print(f"[LSTM] 📐 Data shape: {data.shape}")

# =====================================================
# CREATE SEQUENCES
# =====================================================
def create_sequences_single_step(data, lookback, horizon):
    """
    Create sequences for single-step prediction.
    Returns:
        X: (n_sequences, lookback, n_features) — 3D for LSTM
        y: (n_sequences, n_features)            — target at t+horizon
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback+horizon-1])
    return np.array(X), np.array(y)

print(f"\n[LSTM] 🔄 Creating sequences...")
X, y = create_sequences_single_step(data, LOOKBACK, HORIZON)

print(f"\n[LSTM] ✅ Sequences created:")
print(f"[LSTM]    X shape: {X.shape} (n_sequences, lookback, n_features) ← 3D untuk LSTM")
print(f"[LSTM]    y shape: {y.shape} (n_sequences, n_features)")

assert X.shape[1] == LOOKBACK,       f"Lookback mismatch: {X.shape[1]} != {LOOKBACK}"
assert X.shape[2] == len(FEATURES),  f"Features mismatch: {X.shape[2]} != {len(FEATURES)}"
assert y.shape[1] == len(FEATURES),  f"Output features mismatch: {y.shape[1]} != {len(FEATURES)}"
print(f"[LSTM] ✅ Shape verification passed")

# =====================================================
# TRAIN/TEST SPLIT (NO SHUFFLE)
# =====================================================
split_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, shuffle=False
)

print(f"\n[LSTM] 🔀 Train/Test Split (80/20, no shuffle):")
print(f"[LSTM]    Train: {len(X_train)} | Test: {len(X_test)}")

# =====================================================
# NORMALIZATION — PER-FEATURE (CRITICAL DIFFERENCE VS MLP)
# =====================================================
# MLP  scaler_X: fit on (n, 48) → 48 means (one per timestep-feature combo)
# LSTM scaler_X: fit on (n*12, 4) → 4 means (one per feature)
# Ini memastikan scaling konsisten terlepas dari posisi timestep.
# =====================================================
print(f"\n[LSTM] 📊 Normalizing data (per-feature StandardScaler — 4 means, bukan 48)...")

n_train, T, F = X_train.shape

X_train_2d = X_train.reshape(-1, F)
X_test_2d  = X_test.reshape(-1, F)

scaler_X_lstm = StandardScaler()
X_train_2d_scaled = scaler_X_lstm.fit_transform(X_train_2d)
X_test_2d_scaled  = scaler_X_lstm.transform(X_test_2d)

X_train_scaled = X_train_2d_scaled.reshape(X_train.shape)
X_test_scaled  = X_test_2d_scaled.reshape(X_test.shape)

scaler_y_lstm = StandardScaler()
y_train_scaled = scaler_y_lstm.fit_transform(y_train)
y_test_scaled  = scaler_y_lstm.transform(y_test)

print(f"[LSTM] ✅ Normalization complete")
print(f"[LSTM]    scaler_X_lstm: mean shape = {scaler_X_lstm.mean_.shape} (4 features, bukan 48)")
print(f"[LSTM]    scaler_y_lstm: mean shape = {scaler_y_lstm.mean_.shape}")

# =====================================================
# SCALER VERIFICATION
# =====================================================
test_inverse = scaler_y_lstm.inverse_transform(scaler_y_lstm.transform(y_train[0:1]))
reconstruction_error = np.abs(y_train[0:1] - test_inverse).mean()

print(f"\n[LSTM] 🧪 Inverse Transform Test:")
print(f"[LSTM]    Reconstruction error: {reconstruction_error:.6f}")
if reconstruction_error > 0.01:
    print(f"[LSTM]    ⚠️ Warning: High reconstruction error!")
else:
    print(f"[LSTM]    ✅ Inverse transform working correctly")

# =====================================================
# SAVE DATASETS
# =====================================================
print(f"\n[LSTM] 💾 Saving LSTM datasets (terpisah dari MLP)...")
np.save(DATA_DIR / "X_train_lstm.npy", X_train_scaled)
np.save(DATA_DIR / "X_test_lstm.npy",  X_test_scaled)
np.save(DATA_DIR / "y_train_lstm.npy", y_train_scaled)
np.save(DATA_DIR / "y_test_lstm.npy",  y_test_scaled)
print(f"[LSTM] ✅ Datasets saved")

# =====================================================
# SAVE SCALERS
# =====================================================
joblib.dump(scaler_X_lstm, MODELS_DIR / "scaler_X_lstm.pkl")
joblib.dump(scaler_y_lstm, MODELS_DIR / "scaler_y_lstm.pkl")
print(f"[LSTM] ✅ Scalers saved: scaler_X_lstm.pkl, scaler_y_lstm.pkl")

# =====================================================
# SAVE CONFIGURATION
# =====================================================
config_lstm = {
    'model_type':   'LSTM',
    'lookback':     LOOKBACK,
    'horizon':      HORIZON,
    'features':     FEATURES,
    'input_shape':  [LOOKBACK, len(FEATURES)],
    'input_format': '3D_sequence',
    'output_size':  len(FEATURES),
    'scaler_type':  'per_feature',
    'train_samples': len(X_train),
    'test_samples':  len(X_test),
    'total_samples': len(X_train) + len(X_test)
}

config_path = DATA_DIR / "preprocessing_config_lstm.json"
with open(config_path, 'w') as f:
    json.dump(config_lstm, f, indent=2)
print(f"[LSTM] ✅ Config saved to {config_path}")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*60)
print("[LSTM] ✅ LSTM PREPROCESSING COMPLETE!")
print("="*60)

print(f"\n[LSTM] 📊 Dataset Summary:")
print(f"[LSTM]    Total sequences : {len(X_train) + len(X_test)}")
print(f"[LSTM]    Train/Test split: {len(X_train)}/{len(X_test)}")
print(f"[LSTM]    Input shape     : (batch, 12, 4) — 3D tensor untuk LSTM")
print(f"[LSTM]    Output shape    : (batch, 4) — single timestep prediction")

print(f"\n[LSTM] 📁 Files created:")
print(f"[LSTM]    ✅ {DATA_DIR / 'X_train_lstm.npy'}")
print(f"[LSTM]    ✅ {DATA_DIR / 'X_test_lstm.npy'}")
print(f"[LSTM]    ✅ {DATA_DIR / 'y_train_lstm.npy'}")
print(f"[LSTM]    ✅ {DATA_DIR / 'y_test_lstm.npy'}")
print(f"[LSTM]    ✅ {MODELS_DIR / 'scaler_X_lstm.pkl'}")
print(f"[LSTM]    ✅ {MODELS_DIR / 'scaler_y_lstm.pkl'}")
print(f"[LSTM]    ✅ {config_path}")

print(f"\n[LSTM] ⚠️  MLP files (X_train_mlp.npy, scaler_X_mlp.pkl, dll.) TIDAK diubah.")
print(f"\n[LSTM] 🚀 Next step: python train_LSTM.py")
print("="*60)