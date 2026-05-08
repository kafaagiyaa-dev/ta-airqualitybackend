"""
Export Firebase data for ML training
WITH 5-MINUTE INTERVAL RESAMPLING & OUTLIER REMOVAL
Matches proposal: 5-minute intervals for sliding window
"""
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from datetime import datetime
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
records = []
for key, value in data.items():
    records.append({
        'timestamp': value.get('timestamp'),
        'CO': value.get('CO'),
        'CO2': value.get('CO2'),
        'PM25': value.get('PM25'),
        'PM10': value.get('PM10')
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

# =====================================================
# OUTLIER REMOVAL (PROPOSAL REQUIREMENT)
# =====================================================
print("\n🔍 Removing outliers...")
print(f"   Before: {len(df)} samples")

# Sensor specifications and realistic ranges
# CO: 0-50 ppm (indoor/outdoor safe levels)
# CO2: 400-5000 ppm (ambient to very poor ventilation)
# PM2.5: 0-500 µg/m³ (good to hazardous)
# PM10: 0-600 µg/m³ (good to hazardous)

df_clean = df[
    (df['CO'] >= 0) & (df['CO'] < 50) &
    (df['CO2'] >= 400) & (df['CO2'] < 5000) &
    (df['PM25'] >= 0) & (df['PM25'] < 500) &
    (df['PM10'] >= 0) & (df['PM10'] < 600)
].copy()

removed = len(df) - len(df_clean)
print(f"   After: {len(df_clean)} samples")
print(f"   Removed: {removed} outliers ({(removed/len(df)*100):.1f}%)")

# =====================================================
# CRITICAL: RESAMPLE TO 5-MINUTE INTERVALS
# =====================================================
print("\n⏱️ RESAMPLING TO 5-MINUTE INTERVALS (PROPOSAL REQUIREMENT)...")
print("   This ensures consistent timesteps for sliding window")

df_clean = df_clean.set_index('timestamp')

# Resample to 5-minute intervals using mean aggregation
df_resampled = df_clean.resample('5T').mean()

# Remove rows with NaN (gaps in data)
df_resampled = df_resampled.dropna()

print(f"   Resampled data: {len(df_resampled)} samples")
print(f"   Time span: {(df_resampled.index[-1] - df_resampled.index[0]).total_seconds() / 3600:.1f} hours")

# Reset index to have timestamp as column
df_final = df_resampled.reset_index()

# =====================================================
# STRICTLY VERIFY 5-MINUTE INTERVALS
# =====================================================
print("\n✅ Verifying 5-minute intervals...")
time_diffs = df_final['timestamp'].diff().dropna()
expected_interval = pd.Timedelta(minutes=5)

# Calculate interval statistics
mean_interval = time_diffs.mean().total_seconds() / 60
mode_interval = time_diffs.mode()[0].total_seconds() / 60 if len(time_diffs) > 0 else 0

print(f"   Mean interval: {mean_interval:.2f} minutes")
print(f"   Mode interval: {mode_interval:.2f} minutes")

# Check for deviations
irregular = (time_diffs != expected_interval).sum()
irregular_pct = (irregular / len(time_diffs) * 100) if len(time_diffs) > 0 else 0

if irregular > 0:
    print(f"   ⚠️ Warning: {irregular} irregular intervals ({irregular_pct:.1f}%)")
    print(f"   Expected: exactly 5 minutes between all timestamps")
    
    # Show examples of irregular intervals
    irregular_indices = df_final[1:][time_diffs != expected_interval].head(5)
    if len(irregular_indices) > 0:
        print(f"\n   First irregular intervals:")
        for idx in irregular_indices.index[:5]:
            prev_time = df_final.loc[idx-1, 'timestamp']
            curr_time = df_final.loc[idx, 'timestamp']
            actual_diff = (curr_time - prev_time).total_seconds() / 60
            print(f"      {prev_time} → {curr_time} = {actual_diff:.1f} min")
    
    # STRICT MODE: Reject if too many irregular intervals
    if irregular_pct > 5:  # More than 5% irregular
        print(f"\n   ❌ ERROR: Too many irregular intervals!")
        print(f"   ❌ This data is NOT suitable for ML training")
        print(f"   ❌ Check your ESP32 code - should send every 5 minutes")
        print(f"\n   SOLUTION:")
        print(f"   1. Upload corrected ESP32 code (esp32_air_quality_5min.ino)")
        print(f"   2. Clear Firebase /readings data")
        print(f"   3. Collect fresh data for 24-48 hours")
        exit(1)
else:
    print(f"   ✅ All {len(time_diffs)} intervals are exactly 5 minutes")

# =====================================================
# DATA STATISTICS
# =====================================================
print("\n📊 Final Data Statistics:")
print(df_final[['CO', 'CO2', 'PM25', 'PM10']].describe())

# Check for negative values (should be none)
print("\n🔍 Negative Value Check:")
for col in ['CO', 'CO2', 'PM25', 'PM10']:
    neg_count = (df_final[col] < 0).sum()
    status = "✅" if neg_count == 0 else "❌"
    print(f"   {status} {col}: {neg_count} negative values")

# =====================================================
# MINIMUM DATA CHECK
# =====================================================
MINIMUM_SAMPLES = 100  # Need at least this many for meaningful training
if len(df_final) < MINIMUM_SAMPLES:
    print(f"\n❌ ERROR: Insufficient data!")
    print(f"   Got {len(df_final)} samples, need at least {MINIMUM_SAMPLES}")
    print(f"   Collect more data before training")
    exit(1)

# =====================================================
# SAVE CSV
# =====================================================
file_path = DATA_DIR / "sensor_data_raw.csv"
df_final.to_csv(file_path, index=False)

print(f"\n💾 Saved to: {file_path}")
print(f"✅ Final dataset: {len(df_final)} samples at 5-minute intervals")

# =====================================================
# SHOW SAMPLE DATA
# =====================================================
print("\n📋 Sample Data (first 5 rows):")
print(df_final.head())
print("\n📋 Sample Data (last 5 rows):")
print(df_final.tail())

print("\n" + "="*60)
print("✅ EXPORT COMPLETE!")
print("="*60)
print(f"\n📊 Summary:")
print(f"   Total samples: {len(df_final)}")
print(f"   Interval: 5 minutes (as per proposal)")
print(f"   Time span: {(df_final['timestamp'].iloc[-1] - df_final['timestamp'].iloc[0]).total_seconds() / 3600:.1f} hours")
print(f"   Features: CO, CO2, PM25, PM10")
print(f"\n🚀 Next step: python prepare_sequences.py")
print("="*60)