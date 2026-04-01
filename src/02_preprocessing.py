"""
02_preprocessing.py
-------------------
Load raw Excel data, identify individual meters, select best candidates,
handle missing values, and produce daily-aggregated time series.

Outputs:
  - data/processed/meters_hourly.parquet   (hourly, selected meters, with NaN filled)
  - data/processed/meters_daily.parquet    (daily aggregated, selected meters)
  - data/processed/meter_metadata.csv      (metadata for selected meters)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import PROJECT_DIR, DATA_RAW, DATA_PROCESSED

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading raw data...")
raw_path = os.path.join(DATA_RAW, "water_meters.xlsx")
df = pd.read_excel(raw_path, engine="openpyxl")
print(f"  Raw shape: {df.shape}")
print(f"  Date range: {df['Sampling Date'].min()} to {df['Sampling Date'].max()}")

# Clean column names
df.columns = df.columns.str.strip()
df.rename(columns={
    "Sampling Date": "timestamp",
    "Value": "consumption",
    "Year of Meter Installation": "meter_install_year",
}, inplace=True)

# ---------------------------------------------------------------------------
# 2. Identify individual meters (by timestamp resets)
# ---------------------------------------------------------------------------
print("\nIdentifying meters...")
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
print(f"  Found {n_meters} meters")

# ---------------------------------------------------------------------------
# 3. Compute meter-level statistics for selection
# ---------------------------------------------------------------------------
print("\nComputing meter statistics...")
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

# Compute coverage span in days
meter_stats["span_days"] = (
    (meter_stats["date_max"] - meter_stats["date_min"]).dt.total_seconds() / 86400
)
# Expected hourly records given the span
meter_stats["expected_records"] = (meter_stats["span_days"] * 24).astype(int) + 1
meter_stats["coverage"] = meter_stats["n_records"] / meter_stats["expected_records"]

print(f"  Meters with coverage > 90%: {(meter_stats['coverage'] > 0.9).sum()}")
print(f"  Meters with > 1000 records: {(meter_stats['n_records'] > 1000).sum()}")

# ---------------------------------------------------------------------------
# 4. Select meters for analysis
# ---------------------------------------------------------------------------
# Selection criteria:
#   - High coverage (> 80%)
#   - Enough records (> 1000)
#   - Not too many zeros (< 80%) - some consumption activity needed
#   - Domestic usage
SELECTION_CRITERIA = {
    "min_records": 1000,
    "min_coverage": 0.80,
    "max_zero_fraction": 0.80,
}

mask = (
    (meter_stats["n_records"] >= SELECTION_CRITERIA["min_records"])
    & (meter_stats["coverage"] >= SELECTION_CRITERIA["min_coverage"])
    & (meter_stats["zero_fraction"] <= SELECTION_CRITERIA["max_zero_fraction"])
)
candidates = meter_stats[mask].copy()
print(f"\n  Candidate meters after filtering: {len(candidates)}")

# Prioritize meters with longest coverage (full Jan-Apr span if available)
candidates = candidates.sort_values(["n_records", "annual_consumption"], ascending=[False, True])

# First, include all meters with >2000 records (full coverage Jan-Apr)
full_coverage = candidates[candidates["n_records"] > 2000]
# Then fill up to 10 with diverse consumption levels from remaining
remaining = candidates[candidates["n_records"] <= 2000].sort_values("annual_consumption")
n_remaining = max(0, 10 - len(full_coverage))
if len(remaining) > n_remaining and n_remaining > 0:
    indices = np.linspace(0, len(remaining) - 1, n_remaining, dtype=int)
    extra = remaining.iloc[indices]
else:
    extra = remaining.head(n_remaining)
selected = pd.concat([full_coverage, extra])

selected_ids = selected["meter_id"].tolist()
print(f"  Selected {len(selected_ids)} meters: {selected_ids}")
print(f"  Annual consumption range: {selected['annual_consumption'].min()} - {selected['annual_consumption'].max()}")

# ---------------------------------------------------------------------------
# 5. Build hourly time series per meter (with regular DatetimeIndex)
# ---------------------------------------------------------------------------
print("\nBuilding hourly time series...")
hourly_frames = []

for mid in selected_ids:
    meter_df = df[df["meter_id"] == mid][["timestamp", "consumption"]].copy()
    meter_df = meter_df.set_index("timestamp").sort_index()

    # Remove duplicates (keep first)
    meter_df = meter_df[~meter_df.index.duplicated(keep="first")]

    # Create regular hourly index spanning the full range
    full_idx = pd.date_range(
        start=meter_df.index.min().floor("h"),
        end=meter_df.index.max().ceil("h"),
        freq="h",
    )
    meter_df = meter_df.reindex(full_idx)
    meter_df.index.name = "timestamp"

    # Count missing before interpolation
    n_missing = meter_df["consumption"].isna().sum()
    pct_missing = n_missing / len(meter_df) * 100

    # Interpolate missing values (linear, as in course script 2)
    meter_df["consumption"] = meter_df["consumption"].interpolate(method="linear")
    # Fill any remaining NaN at edges
    meter_df["consumption"] = meter_df["consumption"].bfill().ffill()

    meter_df["meter_id"] = mid
    hourly_frames.append(meter_df.reset_index())

    print(f"  Meter {mid}: {len(meter_df)} hours, {n_missing} missing ({pct_missing:.1f}%)")

hourly_df = pd.concat(hourly_frames, ignore_index=True)

# ---------------------------------------------------------------------------
# 6. Daily aggregation
# ---------------------------------------------------------------------------
print("\nAggregating to daily...")
hourly_df["date"] = hourly_df["timestamp"].dt.date

daily_df = hourly_df.groupby(["meter_id", "date"]).agg(
    daily_consumption=("consumption", "sum"),   # total m³ per day
    mean_hourly=("consumption", "mean"),
    max_hourly=("consumption", "max"),
    min_hourly=("consumption", "min"),
    std_hourly=("consumption", "std"),
    n_hours=("consumption", "count"),
).reset_index()

daily_df["date"] = pd.to_datetime(daily_df["date"])

for mid in selected_ids:
    m = daily_df[daily_df["meter_id"] == mid]
    print(f"  Meter {mid}: {len(m)} days, mean daily={m['daily_consumption'].mean():.4f} m³")

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
os.makedirs(DATA_PROCESSED, exist_ok=True)

hourly_df.to_parquet(os.path.join(DATA_PROCESSED, "meters_hourly.parquet"), index=False)
daily_df.to_parquet(os.path.join(DATA_PROCESSED, "meters_daily.parquet"), index=False)
selected.to_csv(os.path.join(DATA_PROCESSED, "meter_metadata.csv"), index=False)

print(f"\nSaved to {DATA_PROCESSED}/")
print(f"  meters_hourly.parquet: {len(hourly_df)} rows")
print(f"  meters_daily.parquet:  {len(daily_df)} rows")
print(f"  meter_metadata.csv:    {len(selected)} meters")
print("\nDone.")
