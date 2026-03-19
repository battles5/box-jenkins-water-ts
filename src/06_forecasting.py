"""
06_forecasting.py
-----------------
Forecasting and cross-validation (course slides 120-142):
  1. Rolling window cross-validation (equivalent to R's tsCV)
  2. Forecast error metrics: RMSE, MAE, MAPE
  3. Scaled metrics (vs naive predictor)
  4. Forecast visualization with prediction intervals

Outputs:
  - output/figures/forecasting/ (forecast plots per meter)
  - output/tables/forecast_metrics.csv
"""

import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import DATA_PROCESSED, set_plot_style, save_fig, save_table

set_plot_style()

# ---------------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------------
daily = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_daily.parquet"))
with open(os.path.join(DATA_PROCESSED, "fitted_models.pkl"), "rb") as f:
    fitted_models = pickle.load(f)

print(f"Running forecasting for {len(fitted_models)} meters")

HORIZON = 7  # forecast up to 7 days ahead

# ---------------------------------------------------------------------------
# Cross-validation and forecasting
# ---------------------------------------------------------------------------
all_metrics = []

for mid, model_data in fitted_models.items():
    print(f"\n{'='*60}")
    print(f"METER {mid}")
    print(f"{'='*60}")

    y_model = model_data["y_model"]
    y_orig = model_data["y_original"]
    model = model_data["model"]
    transform = model_data["transform"]

    order = model.order
    seasonal = model.seasonal_order
    T = len(y_model)

    # --- 1. Rolling window cross-validation ---
    # Equivalent to forecast::tsCV() in R
    # Use numpy array to avoid pandas indexing issues
    y_arr = y_model.values.astype(float)
    initial = max(T // 2, 21)  # minimum training size
    errors = {h: [] for h in range(1, HORIZON + 1)}

    print(f"  Cross-validation: T={T}, initial={initial}, horizon={HORIZON}")
    n_success = 0
    for t in range(initial, T - HORIZON):
        train = y_arr[:t]
        try:
            cv_model = pm.ARIMA(order=order, seasonal_order=seasonal,
                                suppress_warnings=True)
            cv_model.fit(train)
            forecasts = cv_model.predict(n_periods=HORIZON)
            for h in range(1, HORIZON + 1):
                actual = y_arr[t + h - 1]
                errors[h].append(actual - forecasts[h - 1])
            n_success += 1
        except Exception as e:
            for h in range(1, HORIZON + 1):
                errors[h].append(np.nan)
    print(f"  CV folds successful: {n_success}/{T - HORIZON - initial}")

    # --- 2. Compute metrics ---
    metrics_h = {}
    for h in range(1, HORIZON + 1):
        e = np.array(errors[h])
        valid = ~np.isnan(e)
        e_valid = e[valid]

        rmse = np.sqrt(np.mean(e_valid ** 2)) if len(e_valid) > 0 else np.nan
        mae = np.mean(np.abs(e_valid)) if len(e_valid) > 0 else np.nan

        # Naive forecast errors (predict last observed value)
        naive_e = []
        for t in range(initial, T - HORIZON):
            naive_pred = y_arr[t - 1]  # last observed
            naive_e.append(y_arr[t + h - 1] - naive_pred)
        naive_e = np.array(naive_e)
        naive_valid = ~np.isnan(naive_e)
        naive_rmse = np.sqrt(np.mean(naive_e[naive_valid] ** 2)) if naive_valid.sum() > 0 else np.nan

        scaled_rmse = rmse / naive_rmse if naive_rmse > 0 else np.nan

        metrics_h[h] = {
            "RMSE": rmse,
            "MAE": mae,
            "Naive_RMSE": naive_rmse,
            "Scaled_RMSE": scaled_rmse,
        }

    metrics_df = pd.DataFrame(metrics_h).T
    metrics_df.index.name = "horizon"
    print(f"\n  Forecast metrics (transformed scale):")
    print(metrics_df.to_string())

    # --- 3. Forecast visualization ---
    # Produce h-step ahead forecasts from the full model
    forecasts_full = model.predict(n_periods=HORIZON, return_conf_int=True, alpha=0.05)
    pred_mean = forecasts_full[0]
    pred_ci = forecasts_full[1]

    # Create future dates
    last_date = y_orig.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Plot actual + forecast
    axes[0].plot(y_orig.index, y_orig.values, label="Actual", linewidth=0.8)

    # Back-transform if needed
    if transform == "log":
        shift = y_orig[y_orig > 0].min() / 2 if (y_orig > 0).any() else 0.001
        pred_display = np.exp(pred_mean) - shift
        ci_lo = np.exp(pred_ci[:, 0]) - shift
        ci_hi = np.exp(pred_ci[:, 1]) - shift
    elif "boxcox" in transform:
        lam_val = float(transform.split("=")[1].rstrip(")"))
        shift = y_orig[y_orig > 0].min() / 2 if (y_orig > 0).any() else 0.001
        from scipy.special import inv_boxcox
        pred_display = inv_boxcox(pred_mean, lam_val) - shift
        ci_lo = inv_boxcox(pred_ci[:, 0], lam_val) - shift
        ci_hi = inv_boxcox(pred_ci[:, 1], lam_val) - shift
    else:
        pred_display = pred_mean
        ci_lo = pred_ci[:, 0]
        ci_hi = pred_ci[:, 1]

    axes[0].plot(future_dates, pred_display, "r-", label="Forecast", linewidth=1.5)
    axes[0].fill_between(future_dates, ci_lo, ci_hi, alpha=0.2, color="red", label="95% CI")
    axes[0].set_title(f"Meter {mid} — SARIMA{order}x{seasonal} — 7-day forecast")
    axes[0].set_ylabel("m³/day")
    axes[0].legend()

    # Plot cross-validation metrics
    h_vals = list(range(1, HORIZON + 1))
    axes[1].plot(h_vals, [metrics_h[h]["RMSE"] for h in h_vals], "o-", label="RMSE")
    axes[1].plot(h_vals, [metrics_h[h]["MAE"] for h in h_vals], "s--", label="MAE")
    axes[1].plot(h_vals, [metrics_h[h]["Naive_RMSE"] for h in h_vals], "^:", label="Naive RMSE", color="grey")
    axes[1].set_xlabel("Forecast horizon h (days)")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Cross-validation forecast errors")
    axes[1].legend()
    axes[1].set_xticks(h_vals)

    fig.tight_layout()
    save_fig(fig, f"meter_{mid}_forecast", subdir="forecasting")

    # Store aggregate metrics
    all_metrics.append({
        "meter_id": mid,
        "order": f"SARIMA{order}x{seasonal}",
        "transform": transform,
        "RMSE_h1": metrics_h[1]["RMSE"],
        "RMSE_h7": metrics_h[HORIZON]["RMSE"],
        "MAE_h1": metrics_h[1]["MAE"],
        "MAE_h7": metrics_h[HORIZON]["MAE"],
        "Scaled_RMSE_h1": metrics_h[1]["Scaled_RMSE"],
        "Scaled_RMSE_h7": metrics_h[HORIZON]["Scaled_RMSE"],
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
all_metrics_df = pd.DataFrame(all_metrics)
save_table(all_metrics_df, "forecast_metrics")

print("\n" + "=" * 60)
print("FORECAST METRICS SUMMARY")
print("=" * 60)
print(all_metrics_df[["meter_id", "order", "RMSE_h1", "RMSE_h7",
                       "Scaled_RMSE_h1", "Scaled_RMSE_h7"]].to_string(index=False))
print("\nDone.")
