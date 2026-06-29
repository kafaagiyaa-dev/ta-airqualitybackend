"""
Export Firebase data for ML training
WITH 5-MINUTE INTERVAL RESAMPLING & OUTLIER REMOVAL
8 FEATURES: CO, CO2, PM25, PM10, NO2, ozone, temp, humidity
"""
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from pathlib import Path

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 DATA_DIR: {DATA_DIR}")

# =====================================================
# INITIALIZE FIREBASE
# =====================================================
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ta-airquality-default-rtdb.firebaseio.com'
})

# =====================================================
# FETCH ALL READINGS
# =====================================================
print("\n🔄 Fetching data from Firebase...")
ref = db.reference('/devices/esp32_001/readings')
data = ref.get()

if not data:
    print("❌ No data found in Firebase")
    exit(1)

print(f"✅ Retrieved {len(data)} records from Firebase")

# =====================================================
# CONVERT TO DATAFRAME
# =====================================================
FEATURES = ['CO', 'CO2', 'PM25', 'PM10', 'NO2', 'ozone', 'temp', 'humidity']

records = []
for key, value in data.items():
    records.append({
        'timestamp': value.get('timestamp'),
        'CO':       value.get('CO'),
        'CO2':      value.get('CO2'),
        'PM25':     value.get('PM25'),
        'PM10':     value.get('PM10'),
        'NO2':      value.get('NO2'),
        'ozone':    value.get('ozone'),
        'temp':     value.get('temp'),
        'humidity': value.get('humidity'),
    })

df = pd.DataFrame(records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"\n📊 Raw data loaded: {len(df)} samples")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# =====================================================
# REMOVE MISSING VALUES
# =====================================================
df = df.dropna()
print(f"\n🧹 After removing NaN: {len(df)} samples")

if len(df) == 0:
    print("❌ No data after removing NaN — pastikan ESP32 mengirim semua 8 field")
    exit(1)

# =====================================================
# OUTLIER REMOVAL
# =====================================================
print("\n🔍 Removing outliers...")
print(f"   Before: {len(df)} samples")

df_clean = df[
    (df['CO']       >= 0)   & (df['CO']       < 50)   &
    (df['CO2']      >= 300) & (df['CO2']       < 5000) &
    (df['PM25']     >= 0)   & (df['PM25']      < 500)  &
    (df['PM10']     >= 0)   & (df['PM10']      < 600)  &
    (df['NO2']      >= 0)   & (df['NO2']       < 1000) &
    (df['ozone']    >= 0)   & (df['ozone']     < 500)  &
    (df['temp']     >= -10) & (df['temp']       < 60)   &
    (df['humidity'] >= 0)   & (df['humidity']   <= 100)
].copy()

removed = len(df) - len(df_clean)
print(f"   After: {len(df_clean)} samples")
print(f"   Removed: {removed} outliers ({(removed/len(df)*100):.1f}%)")

# =====================================================
# RESAMPLE TO 5-MINUTE INTERVALS
# =====================================================
print("\n⏱️ Resampling to 5-minute intervals...")

df_clean = df_clean.set_index('timestamp')
df_resampled = df_clean.resample('5min').mean()
df_resampled = df_resampled.dropna()
df_final = df_resampled.reset_index()

print(f"   Resampled: {len(df_final)} samples")
print(f"   Time span: {(df_resampled.index[-1] - df_resampled.index[0]).total_seconds() / 3600:.1f} hours")

# =====================================================
# INTERVAL CHECK (WARNING ONLY, NO EXIT)
# =====================================================
print("\n✅ Verifying intervals...")
time_diffs = df_final['timestamp'].diff().dropna()
mean_interval = time_diffs.mean().total_seconds() / 60
irregular = (time_diffs != pd.Timedelta(minutes=5)).sum()
irregular_pct = irregular / len(time_diffs) * 100

print(f"   Mean interval: {mean_interval:.2f} minutes")
print(f"   Irregular intervals: {irregular} ({irregular_pct:.1f}%)")
if irregular_pct > 5:
    print(f"   ⚠️ Warning: banyak gap di data, tapi training tetap dilanjutkan")
else:
    print(f"   ✅ Interval OK")

# =====================================================
# DATA STATISTICS
# =====================================================
print("\n📊 Final Data Statistics:")
print(df_final[FEATURES].describe())

print("\n🔍 Negative Value Check:")
for col in FEATURES:
    if col == 'temp':
        continue  # temp boleh negatif
    neg_count = (df_final[col] < 0).sum()
    status = "✅" if neg_count == 0 else "❌"
    print(f"   {status} {col}: {neg_count} negative values")

# =====================================================
# MINIMUM DATA CHECK
# =====================================================
MINIMUM_SAMPLES = 50
if len(df_final) < MINIMUM_SAMPLES:
    print(f"\n❌ ERROR: Insufficient data!")
    print(f"   Got {len(df_final)} samples, need at least {MINIMUM_SAMPLES}")
    exit(1)

# =====================================================
# SAVE CSV
# =====================================================
file_path = DATA_DIR / "sensor_data_raw.csv"
df_final.to_csv(file_path, index=False)

print(f"\n💾 Saved to: {file_path}")
print(f"✅ Final dataset: {len(df_final)} samples")

print("\n📋 Sample Data (first 3 rows):")
print(df_final.head(3))
print("\n📋 Sample Data (last 3 rows):")
print(df_final.tail(3))

print("\n" + "="*60)
print("✅ EXPORT COMPLETE!")
print("="*60)
print(f"\n📊 Summary:")
print(f"   Total samples : {len(df_final)}")
print(f"   Features      : {', '.join(FEATURES)}")
print(f"   Time span     : {(df_final['timestamp'].iloc[-1] - df_final['timestamp'].iloc[0]).total_seconds() / 3600:.1f} hours")
print(f"\n🚀 Next step: python scripts/prepare_sequences.py")
print("="*60)