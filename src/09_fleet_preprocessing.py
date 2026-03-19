"""
08_fleet_preprocessing.py
-------------------------
Expand preprocessing to ALL eligible meters (no hardcoded limit).
Reuses the same quality filters and aggregation logic as 02_preprocessing.py,
but retains every meter that passes the criteria.

Outputs:
  - data/processed/meters_daily_fleet.parquet
  - data/processed/meters_hourly_fleet.parquet
  - data/processed/meter_metadata_fleet.csv
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import PROJECT_DIR, DATA_RAW, DATA_PROCESSED

# ── Configuration ──
MIN_RECORDS = 1000       # ~42 days of hourly data
MIN_COVERAGE = 0.80      # >= 80% of expected hours present
MAX_ZERO_FRAC = 0.80     # < 80% zero-consumption hours
MIN_DAILY_DAYS = 50      # minimum days in the daily series for SARIMA (m=7)

# ── 1. Load raw data ──
print("Loading raw data...")
raw_path = os.path.join(DATA_RAW, "water_meters.xlsx")
df = pd.read_excel(raw_path, engine="openpyxl")
print(f"  Raw shape: {df.shape}")

df.columns = df.columns.str.strip()
df.rename(columns={
    "Sampling Date": "timestamp",
    "Value": "consumption",
    "Year of Meter Installation": "meter_install_year",
}, inplace=True)

# ── 2. Identify meters by timestamp resets ──
print("Identifying meters...")
dates = df["timestamp"].values
breaks = [0]
for i in range(1, len(dates)):
    if dates[i] <= dates[i - 1]:
        breaks.append(i)

meter_ids = np.zeros(len(df), dtype=int)
for idx in range(len(breaks)):
    start = breaks[idx]
    end = breaks[idx + 1] if idx + 1 < len(breaks) else len(df)
    meter_ids[start:end] = idx

df["meter_id"] = meter_ids
n_meters = len(breaks)
print(f"  Found {n_meters} meters total")

# ── 3. Compute meter-level statistics ──
print("Computing meter statistics...")
meter_stats = df.groupby("meter_id").agg(
    n_records=("consumption", "count"),
    date_min=("timestamp", "min"),
    date_max=("timestamp", "max"),
    mean_consumption=("consumption", "mean"),
    std_consumption=("consumption", "std"),
    zero_fraction=("consumption", lambda x: (x == 0).mean()),
    annual_consumption=("Annual Consumption", "first"),
    n_residents=("Number of Residents", "first"),
    household_type=("Type of Household", "first"),
    usage_type=("Type of Usage", "first"),
).reset_index()

meter_stats["span_days"] = (
    (meter_stats["date_max"] - meter_stats["date_min"]).dt.total_seconds() / 86400
)
meter_stats["expected_records"] = (meter_stats["span_days"] * 24).astype(int) + 1
meter_stats["coverage"] = meter_stats["n_records"] / meter_stats["expected_records"]

# ── 4. Filter: keep ALL eligible meters ──
mask = (
    (meter_stats["n_records"] >= MIN_RECORDS)
    & (meter_stats["coverage"] >= MIN_COVERAGE)
    & (meter_stats["zero_fraction"] <= MAX_ZERO_FRAC)
)
candidates = meter_stats[mask].copy()
print(f"\n  After n_records >= {MIN_RECORDS}: {(meter_stats['n_records'] >= MIN_RECORDS).sum()}")
print(f"  After coverage >= {MIN_COVERAGE}: {mask.sum()} (added coverage + zero_fraction filters)")
print(f"  Candidate meters: {len(candidates)}")

# Additional filter: enough days for SARIMA
selected_ids = candidates["meter_id"].tolist()

# ── 5. Build hourly time series ──
print(f"\nBuilding hourly time series for {len(selected_ids)} meters...")
hourly_frames = []
kept_ids = []

for i, mid in enumerate(selected_ids):
    meter_df = df[df["meter_id"] == mid][["timestamp", "consumption"]].copy()
    meter_df = meter_df.set_index("timestamp").sort_index()
    meter_df = meter_df[~meter_df.index.duplicated(keep="first")]

    full_idx = pd.date_range(
        start=meter_df.index.min().floor("h"),
        end=meter_df.index.max().ceil("h"),
        freq="h",
    )
    meter_df = meter_df.reindex(full_idx)
    meter_df.index.name = "timestamp"

    n_missing = meter_df["consumption"].isna().sum()
    meter_df["consumption"] = meter_df["consumption"].interpolate(method="linear")
    meter_df["consumption"] = meter_df["consumption"].bfill().ffill()
    meter_df["meter_id"] = mid
    hourly_frames.append(meter_df.reset_index())
    kept_ids.append(mid)

    if (i + 1) % 50 == 0 or i == 0:
        print(f"  Processed {i + 1}/{len(selected_ids)} meters")

print(f"  Total meters processed: {len(kept_ids)}")
hourly_df = pd.concat(hourly_frames, ignore_index=True)

# ── 6. Daily aggregation ──
print("Aggregating to daily...")
hourly_df["date"] = hourly_df["timestamp"].dt.date

daily_df = hourly_df.groupby(["meter_id", "date"]).agg(
    daily_consumption=("consumption", "sum"),
    mean_hourly=("consumption", "mean"),
    max_hourly=("consumption", "max"),
    min_hourly=("consumption", "min"),
    std_hourly=("consumption", "std"),
    n_hours=("consumption", "count"),
).reset_index()
daily_df["date"] = pd.to_datetime(daily_df["date"])

# Final filter: enough days for SARIMA
day_counts = daily_df.groupby("meter_id")["date"].count()
valid_meters = day_counts[day_counts >= MIN_DAILY_DAYS].index.tolist()
daily_df = daily_df[daily_df["meter_id"].isin(valid_meters)]
hourly_df = hourly_df[hourly_df["meter_id"].isin(valid_meters)]
selected = candidates[candidates["meter_id"].isin(valid_meters)]

print(f"  After min {MIN_DAILY_DAYS} days filter: {len(valid_meters)} meters")

# ── 7. Summary statistics ──
print(f"\n{'='*60}")
print(f"  FLEET SUMMARY")
print(f"{'='*60}")
print(f"  Total meters in raw data:  {n_meters}")
print(f"  Eligible (quality filters): {len(candidates)}")
print(f"  Final (>= {MIN_DAILY_DAYS} days):         {len(valid_meters)}")
day_counts_final = daily_df.groupby("meter_id")["date"].count()
print(f"  Days range: {day_counts_final.min()} - {day_counts_final.max()}")
print(f"  Median days: {day_counts_final.median():.0f}")
print(f"  Daily rows: {len(daily_df)}")
print(f"  Hourly rows: {len(hourly_df)}")

# ── 8. Save ──
os.makedirs(DATA_PROCESSED, exist_ok=True)
hourly_df.to_parquet(os.path.join(DATA_PROCESSED, "meters_hourly_fleet.parquet"), index=False)
daily_df.to_parquet(os.path.join(DATA_PROCESSED, "meters_daily_fleet.parquet"), index=False)
selected.to_csv(os.path.join(DATA_PROCESSED, "meter_metadata_fleet.csv"), index=False)

print(f"\nSaved to {DATA_PROCESSED}/")
print("Done.")
