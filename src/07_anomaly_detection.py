"""
07_anomaly_detection.py
-----------------------
SARIMA-based anomaly detection for water meter leak detection.

Methodology (based on course techniques):
  A SARIMA model captures the "normal" consumption pattern.
  Observations that produce large standardized residuals
  deviate significantly from the model's expectations
  and are flagged as anomalies.

Detection rules:
  1. Point anomalies: |standardized residual| > threshold (e.g., 2 or 3 sigma)
  2. Persistent anomalies: runs of consecutive positive residuals > k days
     (sustained overconsumption → possible leak)
  3. Level shift detection: rolling mean of residuals deviating from 0

Outputs:
  - output/figures/anomalies/ (anomaly plots per meter)
  - output/tables/anomaly_summary.csv
  - output/tables/detected_anomalies.csv
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
from utils import DATA_PROCESSED, set_plot_style, save_fig, save_table

set_plot_style()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA_THRESHOLD = 2.5   # standard deviations for point anomaly
RUN_LENGTH = 5          # consecutive days of positive residuals for leak detection
ROLLING_WINDOW = 7      # rolling window for level shift detection
LEVEL_SHIFT_THRESHOLD = 1.5  # sigma for rolling mean anomaly

# ---------------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------------
daily = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_daily.parquet"))
with open(os.path.join(DATA_PROCESSED, "fitted_models.pkl"), "rb") as f:
    fitted_models = pickle.load(f)

print(f"Running anomaly detection for {len(fitted_models)} meters")
print(f"  Point anomaly threshold: {SIGMA_THRESHOLD} sigma")
print(f"  Persistent leak run length: {RUN_LENGTH} days")
print(f"  Level shift window: {ROLLING_WINDOW} days, threshold: {LEVEL_SHIFT_THRESHOLD} sigma")

# ---------------------------------------------------------------------------
# Anomaly detection loop
# ---------------------------------------------------------------------------
all_anomalies = []
all_summaries = []

for mid, model_data in fitted_models.items():
    print(f"\n{'='*60}")
    print(f"METER {mid}")
    print(f"{'='*60}")

    y_orig = model_data["y_original"]
    y_model = model_data["y_model"]
    residuals = model_data["residuals"]
    fitted = model_data["fitted"]
    transform = model_data["transform"]
    model = model_data["model"]

    order = model.order
    seasonal = model.seasonal_order
    dates = y_orig.index

    # Standardized residuals (ensure numpy arrays for clean integer indexing)
    residuals = np.asarray(residuals, dtype=float)
    res_std = residuals / np.nanstd(residuals)

    # -----------------------------------------------------------------------
    # 1. Point anomalies: |standardized residual| > threshold
    # -----------------------------------------------------------------------
    point_anomaly_mask = np.abs(res_std) > SIGMA_THRESHOLD
    point_high = res_std > SIGMA_THRESHOLD    # overconsumption
    point_low = res_std < -SIGMA_THRESHOLD    # underconsumption

    n_point = point_anomaly_mask.sum()
    print(f"  Point anomalies: {n_point} ({n_point/len(res_std)*100:.1f}%)")
    print(f"    High (overconsumption): {point_high.sum()}")
    print(f"    Low (underconsumption): {point_low.sum()}")

    # -----------------------------------------------------------------------
    # 2. Persistent anomalies: runs of positive residuals
    #    Continuous overconsumption may indicate a leak
    # -----------------------------------------------------------------------
    positive = res_std > 0
    runs = []
    current_run_start = None
    current_run_len = 0

    for i in range(len(positive)):
        if positive[i]:
            if current_run_start is None:
                current_run_start = i
            current_run_len += 1
        else:
            if current_run_len >= RUN_LENGTH:
                runs.append((current_run_start, current_run_start + current_run_len - 1))
            current_run_start = None
            current_run_len = 0
    if current_run_len >= RUN_LENGTH:
        runs.append((current_run_start, current_run_start + current_run_len - 1))

    persistent_mask = np.zeros(len(res_std), dtype=bool)
    for start, end in runs:
        persistent_mask[start:end + 1] = True

    n_persistent = len(runs)
    print(f"  Persistent anomaly runs (>={RUN_LENGTH} days positive): {n_persistent}")
    for start, end in runs:
        print(f"    Days {start}-{end} ({dates[start].strftime('%Y-%m-%d')} to {dates[end].strftime('%Y-%m-%d')}), "
              f"mean residual={res_std[start:end+1].mean():.2f} sigma")

    # -----------------------------------------------------------------------
    # 3. Level shift detection: rolling mean of residuals
    # -----------------------------------------------------------------------
    rolling_res = pd.Series(res_std).rolling(ROLLING_WINDOW, center=True).mean()
    level_shift_mask = np.abs(rolling_res.values) > LEVEL_SHIFT_THRESHOLD
    level_shift_mask = np.nan_to_num(level_shift_mask, nan=0).astype(bool)

    n_level_shift = level_shift_mask.sum()
    print(f"  Level shift days: {n_level_shift}")

    # -----------------------------------------------------------------------
    # Combined anomaly flag (any of the three)
    # -----------------------------------------------------------------------
    any_anomaly = point_anomaly_mask | persistent_mask | level_shift_mask

    # -----------------------------------------------------------------------
    # Record individual anomalies
    # -----------------------------------------------------------------------
    for i in range(len(dates)):
        if any_anomaly[i]:
            anomaly_types = []
            if point_anomaly_mask[i]:
                anomaly_types.append("point_high" if res_std[i] > 0 else "point_low")
            if persistent_mask[i]:
                anomaly_types.append("persistent")
            if level_shift_mask[i]:
                anomaly_types.append("level_shift")

            all_anomalies.append({
                "meter_id": mid,
                "date": dates[i],
                "consumption": y_orig.iloc[i],
                "standardized_residual": res_std[i],
                "anomaly_types": "|".join(anomaly_types),
            })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_summaries.append({
        "meter_id": mid,
        "order": f"SARIMA{order}x{seasonal}",
        "transform": transform,
        "n_days": len(y_orig),
        "n_point_anomalies": n_point,
        "n_persistent_runs": n_persistent,
        "n_level_shift_days": n_level_shift,
        "n_total_anomaly_days": any_anomaly.sum(),
        "pct_anomaly": any_anomaly.sum() / len(y_orig) * 100,
        "mean_consumption": y_orig.mean(),
    })

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Consumption with anomalies highlighted
    axes[0].plot(dates, y_orig.values, linewidth=0.8, label="Consumption", zorder=1)
    if point_high.sum() > 0:
        axes[0].scatter(dates[point_high], y_orig.values[point_high],
                        c="red", s=40, zorder=3, label=f"High anomaly (>{SIGMA_THRESHOLD}σ)")
    if point_low.sum() > 0:
        axes[0].scatter(dates[point_low], y_orig.values[point_low],
                        c="blue", s=40, zorder=3, label=f"Low anomaly (<-{SIGMA_THRESHOLD}σ)")
    # Shade persistent runs
    for start, end in runs:
        axes[0].axvspan(dates[start], dates[end], alpha=0.15, color="orange", zorder=0)
    if n_persistent > 0:
        axes[0].axvspan(dates[0], dates[0], alpha=0.15, color="orange", label="Persistent run")
    axes[0].set_title(f"Meter {mid} — Anomaly Detection")
    axes[0].set_ylabel("m³/day")
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Standardized residuals
    colors = np.where(point_anomaly_mask, "red", "steelblue")
    axes[1].bar(dates, res_std, width=0.8, color=colors, alpha=0.7)
    axes[1].axhline(SIGMA_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label=f"±{SIGMA_THRESHOLD}σ")
    axes[1].axhline(-SIGMA_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_ylabel("Standardized residual")
    axes[1].set_title("Standardized Residuals")
    axes[1].legend()

    # Panel 3: Rolling mean of residuals (level shift)
    axes[2].plot(dates, rolling_res.values, linewidth=1, color="purple")
    axes[2].fill_between(dates, rolling_res.values, 0,
                         where=np.abs(rolling_res.values) > LEVEL_SHIFT_THRESHOLD,
                         alpha=0.3, color="orange", label="Level shift")
    axes[2].axhline(LEVEL_SHIFT_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(-LEVEL_SHIFT_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="grey", linewidth=0.5)
    axes[2].set_ylabel(f"Rolling mean ({ROLLING_WINDOW}-day)")
    axes[2].set_title("Level Shift Detection")
    axes[2].legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    fig.tight_layout()
    save_fig(fig, f"meter_{mid}_anomalies", subdir="anomalies")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(all_summaries)
save_table(summary_df, "anomaly_summary")

if all_anomalies:
    anomalies_df = pd.DataFrame(all_anomalies)
    save_table(anomalies_df, "detected_anomalies")
    print(f"\nTotal anomaly records: {len(anomalies_df)}")
else:
    print("\nNo anomalies detected.")

print("\n" + "=" * 60)
print("ANOMALY DETECTION SUMMARY")
print("=" * 60)
print(summary_df[["meter_id", "n_days", "n_point_anomalies",
                   "n_persistent_runs", "n_level_shift_days",
                   "n_total_anomaly_days", "pct_anomaly"]].to_string(index=False))
print("\nDone.")
