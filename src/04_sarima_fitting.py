"""
04_sarima_fitting.py
--------------------
SARIMA model identification and estimation following Box-Jenkins procedure.

For each meter:
  1. Apply transformation if needed (log/Box-Cox)
  2. Use auto_arima (equivalent to R's auto.arima) to find best SARIMA
  3. Also try manual candidates based on ACF/PACF inspection
  4. Compare models via AICc
  5. Save fitted models and results

Outputs:
  - output/tables/sarima_models.csv
  - output/figures/sarima/ (fitted vs actual plots)
  - data/processed/fitted_models.pkl
"""

import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DATA_PROCESSED, set_plot_style, save_fig, save_table,
    stationarity_tests,
)

set_plot_style()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
daily = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_daily.parquet"))
from utils import OUTPUT_TABLES
stationarity = pd.read_csv(os.path.join(OUTPUT_TABLES, "stationarity_tests.csv"))
meter_ids = sorted(daily["meter_id"].unique())

print(f"Fitting SARIMA models for {len(meter_ids)} meters")

# ---------------------------------------------------------------------------
# Fitting loop
# ---------------------------------------------------------------------------
all_results = []
fitted_models = {}

for mid in meter_ids:
    print(f"\n{'='*60}")
    print(f"METER {mid}")
    print(f"{'='*60}")

    m_daily = daily[daily["meter_id"] == mid].copy()
    m_daily = m_daily.set_index("date").sort_index()
    y = m_daily["daily_consumption"].astype(float)

    if len(y) < 21:
        print(f"  Skipping: only {len(y)} days")
        continue

    # Check if transformation needed (from stationarity analysis)
    stat_row = stationarity[stationarity["meter_id"] == mid]
    needs_transform = False
    if len(stat_row) > 0:
        needs_transform = bool(stat_row["needs_transform"].iloc[0])

    # Apply transformation if needed
    # For water data with zeros, use log(y + epsilon) to avoid log(0)
    transform = "none"
    y_model = y.copy()

    if needs_transform:
        # Shift to handle zeros, then log
        shift = y[y > 0].min() / 2 if (y > 0).any() else 0.001
        y_shifted = y + shift
        if (y_shifted > 0).all():
            # Try Box-Cox
            try:
                y_bc, lam = sp_stats.boxcox(y_shifted.values)
                if abs(lam) < 0.1:
                    # Close to log
                    y_model = np.log(y_shifted)
                    transform = "log"
                    print(f"  Applied log transformation (Box-Cox lambda={lam:.3f} ≈ 0)")
                else:
                    y_model = pd.Series(y_bc, index=y.index)
                    transform = f"boxcox(lambda={lam:.3f})"
                    print(f"  Applied Box-Cox transformation (lambda={lam:.3f})")
            except Exception:
                y_model = np.log(y_shifted)
                transform = "log"
                print(f"  Applied log transformation (Box-Cox failed)")
        else:
            print(f"  No transformation (data contains non-positive values after shift)")

    # --- Auto ARIMA (equivalent to R's auto.arima with stepwise=FALSE) ---
    print(f"  Running auto_arima (m=7, weekly seasonality)...")
    try:
        auto_model = pm.auto_arima(
            y_model,
            d=None,              # let it determine d
            D=None,              # let it determine D
            m=7,                 # weekly seasonality
            seasonal=True,
            stepwise=False,      # exhaustive search (like R: stepwise=F)
            suppress_warnings=True,
            error_action="ignore",
            max_p=4, max_q=4,
            max_P=2, max_Q=2,
            max_d=2, max_D=1,
            information_criterion="aicc",
            trace=False,
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        aicc = auto_model.aicc()
        bic = auto_model.bic()

        print(f"  Best model: SARIMA{order}x{seasonal_order}")
        print(f"  AICc={aicc:.2f}, BIC={bic:.2f}")

        # Coefficient significance
        summary = auto_model.summary()
        params = auto_model.params()
        try:
            cov = auto_model.arima_res_.cov_params()
            se = np.sqrt(np.diag(cov))
            z_stats = params / se
            param_names = auto_model.arima_res_.param_names
            print(f"  Parameters: {dict(zip(param_names, params))}")
            print(f"  |z| > 1.96: {dict(zip(param_names, np.abs(z_stats) > 1.96))}")
        except Exception:
            print(f"  Parameters: {params}")

    except Exception as e:
        print(f"  auto_arima failed: {e}")
        # Fallback: simple ARIMA(1,1,1)
        try:
            auto_model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0, 7))
            auto_model.fit(y_model)
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            aicc = auto_model.aicc()
            bic = auto_model.bic()
            print(f"  Fallback model: ARIMA{order}")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            continue

    # --- Also try a few manual candidates ---
    candidates = [
        (order, seasonal_order, "auto"),
    ]
    manual_specs = [
        ((1, 1, 0), (1, 0, 0, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
        ((1, 1, 1), (1, 1, 1, 7)),
    ]
    for m_order, m_sorder in manual_specs:
        try:
            m_model = pm.ARIMA(order=m_order, seasonal_order=m_sorder)
            m_model.fit(y_model)
            candidates.append((m_order, m_sorder, "manual"))
        except Exception:
            pass

    # --- Select best by AICc ---
    best_aicc = aicc
    best_model = auto_model
    best_label = "auto"
    for m_order, m_sorder, label in candidates[1:]:
        try:
            m = pm.ARIMA(order=m_order, seasonal_order=m_sorder)
            m.fit(y_model)
            m_aicc = m.aicc()
            if m_aicc < best_aicc:
                best_aicc = m_aicc
                best_model = m
                best_label = f"manual_{m_order}x{m_sorder}"
                print(f"  Manual SARIMA{m_order}x{m_sorder} AICc={m_aicc:.2f} (better!)")
        except Exception:
            pass

    final_order = best_model.order
    final_seasonal = best_model.seasonal_order

    # --- Plot fitted vs actual ---
    fitted_vals = best_model.predict_in_sample()
    residuals = y_model.values - fitted_vals

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(y.index, y.values, label="Actual", linewidth=0.8)
    # If transformed, back-transform fitted values for display
    if transform == "log":
        shift = y[y > 0].min() / 2 if (y > 0).any() else 0.001
        fitted_display = np.exp(fitted_vals) - shift
    elif "boxcox" in transform:
        lam_val = float(transform.split("=")[1].rstrip(")"))
        shift = y[y > 0].min() / 2 if (y > 0).any() else 0.001
        from scipy.special import inv_boxcox
        fitted_display = inv_boxcox(fitted_vals, lam_val) - shift
    else:
        fitted_display = fitted_vals
    axes[0].plot(y.index, fitted_display, label="Fitted", linewidth=0.8, alpha=0.8)
    axes[0].set_title(f"Meter {mid} - SARIMA{final_order}x{final_seasonal} (transform={transform})")
    axes[0].set_ylabel("m³/day")
    axes[0].legend()

    axes[1].plot(y.index, residuals, linewidth=0.5)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_title("Residuals (transformed scale)")
    axes[1].set_ylabel("Residual")
    fig.tight_layout()
    save_fig(fig, f"meter_{mid}_sarima_fit", subdir="sarima")

    # Store results
    all_results.append({
        "meter_id": mid,
        "n_days": len(y),
        "order_p": final_order[0],
        "order_d": final_order[1],
        "order_q": final_order[2],
        "seasonal_P": final_seasonal[0],
        "seasonal_D": final_seasonal[1],
        "seasonal_Q": final_seasonal[2],
        "seasonal_m": final_seasonal[3],
        "transform": transform,
        "AICc": best_aicc,
        "BIC": best_model.bic(),
        "selection": best_label,
        "n_params": len(best_model.params()),
    })

    fitted_models[mid] = {
        "model": best_model,
        "transform": transform,
        "y_original": y,
        "y_model": y_model,
        "residuals": residuals,
        "fitted": fitted_vals,
    }

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
save_table(results_df, "sarima_models")

with open(os.path.join(DATA_PROCESSED, "fitted_models.pkl"), "wb") as f:
    pickle.dump(fitted_models, f)

print("\n" + "=" * 60)
print("SARIMA MODEL SUMMARY")
print("=" * 60)
print(results_df[["meter_id", "order_p", "order_d", "order_q",
                   "seasonal_P", "seasonal_D", "seasonal_Q",
                   "transform", "AICc"]].to_string(index=False))
print(f"\nModels saved to {DATA_PROCESSED}/fitted_models.pkl")
print("Done.")
