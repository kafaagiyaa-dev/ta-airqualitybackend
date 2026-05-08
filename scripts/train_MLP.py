"""
Train MLP Forecaster for Air Quality
MATCHES PROPOSAL (Tabel 3.4):
- Input layer : 48 nodes (flat: 12 timesteps × 4 features)
- Hidden 1    : 64 neurons, ReLU
- Hidden 2    : 32 neurons, ReLU
- Output      : 4 neurons, Linear
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import joblib

# =====================================================
# PATH SETUP — dari config.py backend
# =====================================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
BASE_DIR   = Path(Config.BASE_DIR)
DATA_DIR   = Path(Config.DATA_FOLDER)
MODELS_DIR = Path(Config.MODEL_FOLDER)

print(f"[MLP] 📂 BASE_DIR  : {BASE_DIR}")
print(f"[MLP] 📂 DATA_DIR  : {DATA_DIR}")
print(f"[MLP] 📂 MODELS_DIR: {MODELS_DIR}")

# =====================================================
# CHECK REQUIRED FILES
# =====================================================
required_files = [
    DATA_DIR / 'X_train_mlp.npy',
    DATA_DIR / 'X_test_mlp.npy',
    DATA_DIR / 'y_train_mlp.npy',
    DATA_DIR / 'y_test_mlp.npy',
    DATA_DIR / 'preprocessing_config_mlp.json'
]

print("\n[MLP] 🔍 Checking required files:")
all_exist = True
for file_path in required_files:
    exists = file_path.exists()
    status = "✅" if exists else "❌"
    print(f"[MLP]    {status} {file_path.name}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n[MLP] ❌ ERROR: Missing required files!")
    print("[MLP] ❌ Run prepare_sequences_MLP.py first!")
    exit(1)

print("[MLP] ✅ All required files found!")

# =====================================================
# LOAD CONFIGURATION
# =====================================================
with open(DATA_DIR / 'preprocessing_config_mlp.json', 'r') as f:
    config = json.load(f)

print("\n[MLP] ⚙️ Configuration loaded:")
print(f"[MLP]    Model type  : {config['model_type']}")
print(f"[MLP]    Input size  : {config['input_size']} (flat 48 nodes)")
print(f"[MLP]    Output size : {config['output_size']}")
print(f"[MLP]    Lookback    : {config['lookback']} timesteps")
print(f"[MLP]    Horizon     : {config['horizon']} timesteps ahead")

# =====================================================
# LOAD DATA
# =====================================================
print("\n[MLP] 📂 Loading datasets...")
X_train = np.load(DATA_DIR / 'X_train_mlp.npy')
X_test  = np.load(DATA_DIR / 'X_test_mlp.npy')
y_train = np.load(DATA_DIR / 'y_train_mlp.npy')
y_test  = np.load(DATA_DIR / 'y_test_mlp.npy')

print(f"[MLP] ✅ Data loaded:")
print(f"[MLP]    Train: X={X_train.shape}, y={y_train.shape}")
print(f"[MLP]    Test : X={X_test.shape},  y={y_test.shape}")

assert X_train.shape[1] == config['input_size'],  "Input size mismatch!"
assert y_train.shape[1] == config['output_size'], "Output size mismatch!"
print(f"[MLP] ✅ Shape verification passed")

# =====================================================
# BUILD MLP MODEL (EXACT PROPOSAL ARCHITECTURE - TABEL 3.4)
# =====================================================
print("\n[MLP] 🏗️ Building MLP model (Tabel 3.4)...")
print("[MLP]    Architecture:")
print("[MLP]    - Input layer  : 48 nodes (flat)")
print("[MLP]    - Hidden layer 1: 64 neurons (ReLU)")
print("[MLP]    - Hidden layer 2: 32 neurons (ReLU)")
print("[MLP]    - Output layer : 4 neurons (Linear)")
print("[MLP]    - NO dropout (not in proposal)")

model = Sequential([
    Dense(64, activation='relu', input_shape=(48,), name='hidden1'),
    Dense(32, activation='relu', name='hidden2'),
    Dense(4,  activation='linear', name='output')
], name='AirQualityForecaster_MLP')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n[MLP] 📋 Model Summary:")
model.summary()

total_params = model.count_params()
print(f"\n[MLP] 📊 Total parameters: {total_params:,}")

# =====================================================
# CALLBACKS
# =====================================================
MODELS_DIR.mkdir(parents=True, exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss', patience=20,
    restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    str(MODELS_DIR / 'mlp_best.h5'),
    monitor='val_loss', save_best_only=True, verbose=1
)

# =====================================================
# TRAIN MODEL
# =====================================================
print("\n" + "="*60)
print("[MLP] 🚀 STARTING MLP TRAINING")
print("="*60)

EPOCHS     = 200
BATCH_SIZE = 32

training_start = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

training_time = time.time() - training_start

print("\n" + "="*60)
print("[MLP] ✅ MLP TRAINING COMPLETED!")
print("="*60)
print(f"[MLP]    Training time: {training_time:.1f}s")

# =====================================================
# SAVE FINAL MODEL
# =====================================================
final_model_path = MODELS_DIR / 'air_quality_mlp.h5'
model.save(str(final_model_path))
print(f"\n[MLP] 💾 Final model saved: {final_model_path}")

# =====================================================
# EVALUATE MODEL
# =====================================================
print("\n[MLP] 📊 Evaluating on test set...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\n[MLP] 📈 Test Metrics (scaled):")
print(f"[MLP]    Loss (MSE): {test_loss:.4f}")
print(f"[MLP]    MAE       : {test_mae:.4f}")

# =====================================================
# INFERENCE TIME MEASUREMENT (1000 ITERATIONS)
# =====================================================
print("\n[MLP] ⏱️ Measuring inference time (1000 iterations, single sample)...")

single_sample = X_test[0:1]
n_iters = 1000

for _ in range(10):
    _ = model.predict(single_sample, verbose=0)

infer_start = time.time()
for _ in range(n_iters):
    _ = model.predict(single_sample, verbose=0)
infer_time_ms = (time.time() - infer_start) / n_iters * 1000

print(f"[MLP]    Inference time: {infer_time_ms:.4f} ms/sample (avg of {n_iters} iterations)")

# =====================================================
# PREDICTIONS & INVERSE TRANSFORM
# =====================================================
print("\n[MLP] 🧪 Testing predictions...")
y_pred_scaled = model.predict(X_test, verbose=0)

scaler_y = joblib.load(MODELS_DIR / 'scaler_y_mlp.pkl')
y_pred          = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

neg_count = (y_pred < 0).sum()
print(f"\n[MLP] 🔍 Negative Prediction Check: {neg_count} found")

if neg_count > 0:
    features = ['CO', 'CO2', 'PM25', 'PM10']
    for i, feat in enumerate(features):
        n = (y_pred[:, i] < 0).sum()
        if n > 0:
            print(f"[MLP]    {feat}: {n} negative predictions")
    y_pred = np.maximum(y_pred, 0)
    print(f"[MLP]    ✅ Clipped to >= 0")
else:
    print(f"[MLP]    ✅ No negative predictions!")

mae_original  = np.abs(y_test_original - y_pred).mean()
rmse_original = np.sqrt(((y_test_original - y_pred) ** 2).mean())

features = ['CO', 'CO2', 'PM25', 'PM10']
print(f"\n[MLP] 📊 Per-Feature MAE (original scale):")
per_feature_mae = {}
for i, feat in enumerate(features):
    mae_f = np.abs(y_test_original[:, i] - y_pred[:, i]).mean()
    per_feature_mae[feat] = float(mae_f)
    print(f"[MLP]    {feat}: {mae_f:.4f}")

print(f"\n[MLP] 📊 Overall Metrics (original scale):")
print(f"[MLP]    MAE : {mae_original:.4f}")
print(f"[MLP]    RMSE: {rmse_original:.4f}")

mask = y_test_original != 0
mape_original = np.abs((y_test_original[mask] - y_pred.reshape(y_test_original.shape)[mask]) /
                        y_test_original[mask]).mean() * 100
print(f"[MLP]    MAPE: {mape_original:.2f}%")

print(f"\n[MLP] 📋 Sample Predictions (first 5):")
for i in range(min(5, len(y_pred))):
    print(f"\n[MLP]    Sample {i+1}:")
    print(f"[MLP]       Actual   : {dict(zip(features, y_test_original[i].round(2)))}")
    print(f"[MLP]       Predicted: {dict(zip(features, y_pred[i].round(2)))}")
    print(f"[MLP]       Error    : {dict(zip(features, np.abs(y_test_original[i] - y_pred[i]).round(2)))}")

# =====================================================
# SAVE METRICS
# =====================================================
metrics = {
    'model_architecture': {
        'type':              'MLP',
        'input_nodes':       48,
        'hidden1_neurons':   64,
        'hidden2_neurons':   32,
        'output_nodes':      4,
        'hidden_activation': 'relu',
        'output_activation': 'linear',
        'total_parameters':  int(total_params)
    },
    'training_config': {
        'epochs':                   EPOCHS,
        'batch_size':               BATCH_SIZE,
        'optimizer':                'adam',
        'loss':                     'mse',
        'early_stopping_patience':  20,
        'training_time_seconds':    round(training_time, 2)
    },
    'dataset': {
        'train_samples':  int(len(X_train)),
        'test_samples':   int(len(X_test)),
        'total_samples':  int(len(X_train) + len(X_test)),
        'lookback':       config['lookback'],
        'horizon':        config['horizon'],
        'features':       config['features']
    },
    'results': {
        'test_loss_mse':                    float(test_loss),
        'test_mae_scaled':                  float(test_mae),
        'test_mae_original':                float(mae_original),
        'test_rmse_original':               float(rmse_original),
        'test_mape_original_pct':           float(mape_original),
        'per_feature_mae':                  per_feature_mae,
        'epochs_trained':                   len(history.history['loss']),
        'final_train_loss':                 float(history.history['loss'][-1]),
        'final_val_loss':                   float(history.history['val_loss'][-1]),
        'final_train_mae':                  float(history.history['mae'][-1]),
        'final_val_mae':                    float(history.history['val_mae'][-1]),
        'negative_predictions_before_clip': int(neg_count),
        'inference_time_ms_per_sample':     round(infer_time_ms, 4)
    }
}

metrics_path = MODELS_DIR / 'training_metrics_mlp.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n[MLP] 💾 Metrics saved: {metrics_path}")

# =====================================================
# PLOT TRAINING HISTORY
# =====================================================
print("\n[MLP] 📊 Creating training plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
axes[0].set_title('MLP Loss (MSE)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Mean Squared Error', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE',   linewidth=2)
axes[1].set_title('MLP MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('MLP Training History — Air Quality Forecaster', fontsize=15, fontweight='bold')
plt.tight_layout()

plot_path = MODELS_DIR / 'training_history_mlp.png'
plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
print(f"[MLP] ✅ Plot saved: {plot_path}")
plt.close()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*60)
print("[MLP] ✅ MLP TRAINING PIPELINE COMPLETE!")
print("="*60)

print(f"\n[MLP] 📁 Generated Files:")
print(f"[MLP]    ✅ {final_model_path}")
print(f"[MLP]    ✅ {MODELS_DIR / 'mlp_best.h5'}")
print(f"[MLP]    ✅ {metrics_path}")
print(f"[MLP]    ✅ {plot_path}")

print(f"\n[MLP] 📊 MLP Performance:")
print(f"[MLP]    MAE  (original): {mae_original:.4f}")
print(f"[MLP]    RMSE (original): {rmse_original:.4f}")
print(f"[MLP]    MAPE           : {mape_original:.2f}%")
print(f"[MLP]    Epochs trained : {len(history.history['loss'])}")
print(f"[MLP]    Parameters     : {total_params:,}")
print(f"[MLP]    Inference time : {infer_time_ms:.4f} ms/sample")

print(f"\n[MLP] 🎯 Model Characteristics:")
print(f"[MLP]    ✅ Follows proposal Tabel 3.4 architecture exactly")
print(f"[MLP]    ✅ Input FLAT (48) → Dense(64) → Dense(32) → Output(4, Linear)")
print(f"[MLP]    ✅ No dropout (not in proposal)")

print("="*60)
print("[MLP] 🚀 Next Step: Run train_LSTM.py then compare metrics")
print("="*60)