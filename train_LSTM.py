"""
Train LSTM Forecaster for Air Quality - 8 Parameters
Architecture:
- Input       : (12, 8)  — 12 timesteps × 8 features (3D)
- LSTM layer 1: 64 units, tanh/sigmoid (return_sequences=True)
- LSTM layer 2: 32 units, tanh/sigmoid (return_sequences=False)
- Dense layer : 32 neurons, ReLU
- Output layer: 8 neurons, Linear
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import time
import joblib
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

print(f"[LSTM] 📂 BASE_DIR  : {BASE_DIR}")
print(f"[LSTM] 📂 DATA_DIR  : {DATA_DIR}")
print(f"[LSTM] 📂 MODELS_DIR: {MODELS_DIR}")

# =====================================================
# CHECK REQUIRED FILES
# =====================================================
required_files = [
    DATA_DIR   / 'X_train_lstm.npy',
    DATA_DIR   / 'X_test_lstm.npy',
    DATA_DIR   / 'y_train_lstm.npy',
    DATA_DIR   / 'y_test_lstm.npy',
    DATA_DIR   / 'preprocessing_config_lstm.json',
    MODELS_DIR / 'scaler_y_lstm.pkl',
]

print("\n[LSTM] 🔍 Checking required files:")
all_exist = True
for file_path in required_files:
    exists = file_path.exists()
    status = "✅" if exists else "❌"
    print(f"[LSTM]    {status} {file_path.name}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n[LSTM] ❌ ERROR: Missing required files!")
    print("[LSTM] ❌ Run prepare_sequences_LSTM.py first!")
    exit(1)

print("[LSTM] ✅ All required files found!")

# =====================================================
# LOAD CONFIGURATION
# =====================================================
with open(DATA_DIR / 'preprocessing_config_lstm.json', 'r') as f:
    config = json.load(f)

print("\n[LSTM] ⚙️ Configuration loaded:")
print(f"[LSTM]    Model type  : {config['model_type']}")
print(f"[LSTM]    Input shape : {config['input_shape']} (timesteps, features) — 3D")
print(f"[LSTM]    Output size : {config['output_size']}")
print(f"[LSTM]    Features    : {config['features']}")
print(f"[LSTM]    Scaler type : {config['scaler_type']}")

# =====================================================
# LOAD DATA
# =====================================================
print("\n[LSTM] 📂 Loading datasets...")
X_train = np.load(DATA_DIR / 'X_train_lstm.npy')
X_test  = np.load(DATA_DIR / 'X_test_lstm.npy')
y_train = np.load(DATA_DIR / 'y_train_lstm.npy')
y_test  = np.load(DATA_DIR / 'y_test_lstm.npy')

print(f"[LSTM] ✅ Data loaded:")
print(f"[LSTM]    Train: X={X_train.shape}, y={y_train.shape}")
print(f"[LSTM]    Test : X={X_test.shape},  y={y_test.shape}")

assert X_train.shape[1] == config['input_shape'][0], "Timesteps mismatch!"
assert X_train.shape[2] == config['input_shape'][1], "Features mismatch!"
assert y_train.shape[1] == config['output_size'],    "Output mismatch!"
print(f"[LSTM] ✅ Shape verification passed")

TIMESTEPS  = config['input_shape'][0]  # 12
N_FEATURES = config['input_shape'][1]  # 8
OUTPUT_SIZE = config['output_size']    # 8
FEATURES    = config['features']

# =====================================================
# BUILD LSTM MODEL
# =====================================================
print("\n[LSTM] 🏗️ Building LSTM model...")
print("[LSTM]    Architecture:")
print(f"[LSTM]    - Input        : ({TIMESTEPS}, {N_FEATURES}) 3D tensor")
print(f"[LSTM]    - LSTM layer 1 : 64 units (tanh, sigmoid gates, return_sequences=True)")
print(f"[LSTM]    - LSTM layer 2 : 32 units (tanh, sigmoid gates, return_sequences=False)")
print(f"[LSTM]    - Dense layer  : 32 neurons (ReLU)")
print(f"[LSTM]    - Output layer : {OUTPUT_SIZE} neurons (Linear)")
print(f"[LSTM]    - NO dropout (not in proposal)")

model = Sequential([
    LSTM(64, activation='tanh', recurrent_activation='sigmoid',
         return_sequences=True, input_shape=(TIMESTEPS, N_FEATURES), name='lstm_1'),
    LSTM(32, activation='tanh', recurrent_activation='sigmoid',
         return_sequences=False, name='lstm_2'),
    Dense(32,          activation='relu',   name='dense_hidden'),
    Dense(OUTPUT_SIZE, activation='linear', name='output')
], name='AirQualityForecaster_LSTM')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n[LSTM] 📋 Model Summary:")
model.summary()

total_params = model.count_params()
print(f"\n[LSTM] 📊 Total parameters: {total_params:,}")

# =====================================================
# COMPARE PARAMETER COUNT VS MLP
# =====================================================
mlp_metrics_path = MODELS_DIR / 'training_metrics_mlp.json'
mlp_params       = None
mlp_infer_time   = None

if mlp_metrics_path.exists():
    with open(mlp_metrics_path, 'r') as f:
        mlp_metrics = json.load(f)
    mlp_params     = mlp_metrics['model_architecture']['total_parameters']
    mlp_infer_time = mlp_metrics.get('results', {}).get('inference_time_ms_per_sample')

    ratio = total_params / mlp_params if mlp_params > 0 else 0
    print(f"\n[LSTM vs MLP] 📊 Parameter Comparison:")
    print(f"[LSTM vs MLP]    MLP  parameters: {mlp_params:,}")
    print(f"[LSTM vs MLP]    LSTM parameters: {total_params:,}")
    print(f"[LSTM vs MLP]    LSTM memiliki {ratio:.1f}× lebih banyak parameter dari MLP")

# =====================================================
# CALLBACKS
# =====================================================
MODELS_DIR.mkdir(parents=True, exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss', patience=20,
    restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    str(MODELS_DIR / 'lstm_best.h5'),
    monitor='val_loss', save_best_only=True, verbose=1
)

# =====================================================
# TRAIN MODEL
# =====================================================
print("\n" + "="*60)
print("[LSTM] 🚀 STARTING LSTM TRAINING")
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
print("[LSTM] ✅ LSTM TRAINING COMPLETED!")
print("="*60)
print(f"[LSTM]    Training time: {training_time:.1f}s")

# =====================================================
# SAVE FINAL MODEL
# =====================================================
final_model_path = MODELS_DIR / 'air_quality_lstm.h5'
model.save(str(final_model_path))
print(f"\n[LSTM] 💾 Final model saved: {final_model_path}")

# =====================================================
# EVALUATE MODEL
# =====================================================
print("\n[LSTM] 📊 Evaluating on test set...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\n[LSTM] 📈 Test Metrics (scaled):")
print(f"[LSTM]    Loss (MSE): {test_loss:.4f}")
print(f"[LSTM]    MAE       : {test_mae:.4f}")

# =====================================================
# INFERENCE TIME MEASUREMENT (1000 ITERATIONS)
# =====================================================
print("\n[LSTM] ⏱️ Measuring inference time (1000 iterations, single sample)...")

single_sample = X_test[0:1]
n_iters = 1000

for _ in range(10):
    _ = model.predict(single_sample, verbose=0)

infer_start = time.time()
for _ in range(n_iters):
    _ = model.predict(single_sample, verbose=0)
infer_time_ms = (time.time() - infer_start) / n_iters * 1000

print(f"[LSTM]    Inference time: {infer_time_ms:.4f} ms/sample (avg of {n_iters} iterations)")

if mlp_infer_time:
    ratio_infer = infer_time_ms / mlp_infer_time
    print(f"\n[LSTM vs MLP] ⏱️ Inference Time Comparison:")
    print(f"[LSTM vs MLP]    MLP  inference: {mlp_infer_time:.4f} ms/sample")
    print(f"[LSTM vs MLP]    LSTM inference: {infer_time_ms:.4f} ms/sample")
    print(f"[LSTM vs MLP]    LSTM {ratio_infer:.1f}× lebih lambat dari MLP")

# =====================================================
# PREDICTIONS & INVERSE TRANSFORM
# =====================================================
print("\n[LSTM] 🧪 Testing predictions...")
y_pred_scaled = model.predict(X_test, verbose=0)

scaler_y_lstm   = joblib.load(MODELS_DIR / 'scaler_y_lstm.pkl')
y_pred          = scaler_y_lstm.inverse_transform(y_pred_scaled)
y_test_original = scaler_y_lstm.inverse_transform(y_test)

neg_count = (y_pred < 0).sum()
print(f"\n[LSTM] 🔍 Negative Prediction Check: {neg_count} found")

if neg_count > 0:
    for i, feat in enumerate(FEATURES):
        n = (y_pred[:, i] < 0).sum()
        if n > 0:
            print(f"[LSTM]    {feat}: {n} negative predictions")
    # temp boleh minus, yang lain clip ke 0
    non_temp_idx = [i for i, f in enumerate(FEATURES) if f != 'temp']
    for idx in non_temp_idx:
        y_pred[:, idx] = np.maximum(y_pred[:, idx], 0)
    print(f"[LSTM]    ✅ Clipped non-temp features to >= 0")
else:
    print(f"[LSTM]    ✅ No negative predictions!")

mae_original  = np.abs(y_test_original - y_pred).mean()
rmse_original = np.sqrt(((y_test_original - y_pred) ** 2).mean())

print(f"\n[LSTM] 📊 Per-Feature MAE (original scale):")
per_feature_mae = {}
for i, feat in enumerate(FEATURES):
    mae_f = np.abs(y_test_original[:, i] - y_pred[:, i]).mean()
    per_feature_mae[feat] = float(mae_f)
    print(f"[LSTM]    {feat}: {mae_f:.4f}")

print(f"\n[LSTM] 📊 Overall Metrics (original scale):")
print(f"[LSTM]    MAE : {mae_original:.4f}")
print(f"[LSTM]    RMSE: {rmse_original:.4f}")

mask = y_test_original != 0
mape_original = np.abs((y_test_original[mask] - y_pred.reshape(y_test_original.shape)[mask]) /
                        y_test_original[mask]).mean() * 100
print(f"[LSTM]    MAPE: {mape_original:.2f}%")

print(f"\n[LSTM] 📋 Sample Predictions (first 5):")
for i in range(min(5, len(y_pred))):
    print(f"\n[LSTM]    Sample {i+1}:")
    print(f"[LSTM]       Actual   : {dict(zip(FEATURES, y_test_original[i].round(2)))}")
    print(f"[LSTM]       Predicted: {dict(zip(FEATURES, y_pred[i].round(2)))}")
    print(f"[LSTM]       Error    : {dict(zip(FEATURES, np.abs(y_test_original[i] - y_pred[i]).round(2)))}")

# =====================================================
# SAVE METRICS
# =====================================================
metrics = {
    'model_architecture': {
        'type':              'LSTM',
        'input_shape':       [TIMESTEPS, N_FEATURES],
        'input_format':      '3D_sequence',
        'lstm_layer1_units': 64,
        'lstm_layer2_units': 32,
        'dense_neurons':     32,
        'output_nodes':      OUTPUT_SIZE,
        'lstm_activation':   'tanh',
        'gate_activation':   'sigmoid',
        'dense_activation':  'relu',
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
        'features':       FEATURES
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

if mlp_params is not None:
    metrics['comparison_vs_mlp'] = {
        'mlp_parameters':          mlp_params,
        'lstm_parameters':         int(total_params),
        'param_ratio_lstm_vs_mlp': round(total_params / mlp_params, 2),
    }
    if mlp_infer_time:
        metrics['comparison_vs_mlp']['mlp_inference_ms']        = mlp_infer_time
        metrics['comparison_vs_mlp']['lstm_inference_ms']       = round(infer_time_ms, 4)
        metrics['comparison_vs_mlp']['infer_ratio_lstm_vs_mlp'] = round(infer_time_ms / mlp_infer_time, 2)

metrics_path = MODELS_DIR / 'training_metrics_lstm.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n[LSTM] 💾 Metrics saved: {metrics_path}")

# =====================================================
# PLOT TRAINING HISTORY
# =====================================================
print("\n[LSTM] 📊 Creating training plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
axes[0].set_title('LSTM Loss (MSE) — 8 Features', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Mean Squared Error', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE',   linewidth=2)
axes[1].set_title('LSTM MAE — 8 Features', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('LSTM Training History — Air Quality Forecaster (8 Parameters)', fontsize=13, fontweight='bold')
plt.tight_layout()

plot_path = MODELS_DIR / 'training_history_lstm.png'
plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
print(f"[LSTM] ✅ Plot saved: {plot_path}")
plt.close()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*60)
print("[LSTM] ✅ LSTM TRAINING PIPELINE COMPLETE!")
print("="*60)

print(f"\n[LSTM] 📁 Generated Files:")
print(f"[LSTM]    ✅ {final_model_path}")
print(f"[LSTM]    ✅ {MODELS_DIR / 'lstm_best.h5'}")
print(f"[LSTM]    ✅ {metrics_path}")
print(f"[LSTM]    ✅ {plot_path}")

print(f"\n[LSTM] 📊 LSTM Performance:")
print(f"[LSTM]    MAE  (original): {mae_original:.4f}")
print(f"[LSTM]    RMSE (original): {rmse_original:.4f}")
print(f"[LSTM]    MAPE           : {mape_original:.2f}%")
print(f"[LSTM]    Epochs trained : {len(history.history['loss'])}")
print(f"[LSTM]    Parameters     : {total_params:,}")
print(f"[LSTM]    Inference time : {infer_time_ms:.4f} ms/sample")

if mlp_params is not None:
    print(f"\n[LSTM vs MLP] ⚖️ Trade-off Summary:")
    print(f"[LSTM vs MLP]    Parameters : LSTM {total_params/mlp_params:.1f}× lebih banyak dari MLP")
    if mlp_infer_time:
        print(f"[LSTM vs MLP]    Inference  : LSTM {infer_time_ms/mlp_infer_time:.1f}× lebih lambat dari MLP")
    print(f"[LSTM vs MLP]    → Bandingkan MAE di training_metrics_mlp.json vs training_metrics_lstm.json")
    print(f"[LSTM vs MLP]    → Ini adalah core trade-off analysis untuk kesimpulan proposal.")

print(f"\n[LSTM] 🎯 Model Characteristics:")
print(f"[LSTM]    ✅ Input 3D (batch, 12, 8) — tidak di-flatten seperti MLP")
print(f"[LSTM]    ✅ LSTM(64) → LSTM(32) → Dense(32, ReLU) → Output({OUTPUT_SIZE}, Linear)")
print(f"[LSTM]    ✅ No dropout (not in proposal)")
print(f"[LSTM]    ✅ Inference time diukur 1000 iterasi (sesuai proposal)")

print("="*60)
print("[LSTM] 🚀 Done! Semua hasil tersimpan di backend/models/")
print("="*60)