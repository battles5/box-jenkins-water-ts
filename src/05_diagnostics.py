"""
05_diagnostics.py
-----------------
Model diagnostic checking (course slides 105-119):
  1. Residual time plot (no patterns)
  2. ACF of residuals (no autocorrelation)
  3. ACF of squared residuals (homoscedasticity)
  4. Fitted vs residuals scatter
  5. Fitted vs squared residuals scatter
  6. QQ-plot (normality)
  7. Ljung-Box test on residuals and squared residuals

Outputs:
  - output/figures/diagnostics/ (6-panel diagnostic plots per meter)
  - output/tables/diagnostics_tests.csv
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DATA_PROCESSED, set_plot_style, save_fig, save_table,
    plot_diagnostics, ljung_box_test,
)

set_plot_style()

# ---------------------------------------------------------------------------
# Load fitted models
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_PROCESSED, "fitted_models.pkl"), "rb") as f:
    fitted_models = pickle.load(f)

print(f"Running diagnostics for {len(fitted_models)} meters")

all_results = []

for mid, model_data in fitted_models.items():
    print(f"\n{'='*60}")
    print(f"METER {mid}")
    print(f"{'='*60}")

    residuals = pd.Series(model_data["residuals"])
    fitted = pd.Series(model_data["fitted"])
    model = model_data["model"]
    transform = model_data["transform"]

    order = model.order
    seasonal = model.seasonal_order

    # --- 1. Diagnostic plots (6-panel, as in course R scripts) ---
    fig = plot_diagnostics(
        residuals, fitted,
        title=f"Meter {mid} — SARIMA{order}x{seasonal} (transform={transform})",
        save_name=None,
    )
    save_fig(fig, f"meter_{mid}_diagnostics", subdir="diagnostics")

    # --- 2. Ljung-Box tests ---
    print(f"  SARIMA{order}x{seasonal}")
    lags = [5, 10, 15]
    lb_res, lb_sq = ljung_box_test(residuals, lags=lags, verbose=True)

    # --- 3. Summary statistics ---
    res_clean = residuals.dropna()
    from scipy import stats as sp_stats
    _, shapiro_p = sp_stats.shapiro(res_clean[:min(len(res_clean), 5000)])

    results = {
        "meter_id": mid,
        "order": f"SARIMA{order}x{seasonal}",
        "transform": transform,
        "residual_mean": res_clean.mean(),
        "residual_std": res_clean.std(),
        # Ljung-Box on residuals (lag=10)
        "LB_resid_stat_10": lb_res.loc[10, "lb_stat"] if 10 in lb_res.index else np.nan,
        "LB_resid_p_10": lb_res.loc[10, "lb_pvalue"] if 10 in lb_res.index else np.nan,
        "LB_resid_pass_10": (lb_res.loc[10, "lb_pvalue"] > 0.05) if 10 in lb_res.index else np.nan,
        # Ljung-Box on squared residuals (lag=10)
        "LB_sq_stat_10": lb_sq.loc[10, "lb_stat"] if 10 in lb_sq.index else np.nan,
        "LB_sq_p_10": lb_sq.loc[10, "lb_pvalue"] if 10 in lb_sq.index else np.nan,
        "LB_sq_pass_10": (lb_sq.loc[10, "lb_pvalue"] > 0.05) if 10 in lb_sq.index else np.nan,
        # Shapiro-Wilk normality
        "shapiro_p": shapiro_p,
        "normality_pass": shapiro_p > 0.05,
    }

    # Overall diagnostic assessment
    resid_ok = results.get("LB_resid_pass_10", False)
    sq_ok = results.get("LB_sq_pass_10", False)
    results["diagnostic_pass"] = bool(resid_ok and sq_ok)

    status = "PASS" if results["diagnostic_pass"] else "FAIL"
    details = []
    if not resid_ok:
        details.append("autocorrelated residuals")
    if not sq_ok:
        details.append("heteroscedastic")
    if not results["normality_pass"]:
        details.append("non-normal")

    print(f"\n  Diagnostic: {status}")
    if details:
        print(f"  Issues: {', '.join(details)}")

    all_results.append(results)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
save_table(results_df, "diagnostics_tests")

print("\n" + "=" * 60)
print("DIAGNOSTICS SUMMARY")
print("=" * 60)
print(results_df[["meter_id", "order", "LB_resid_pass_10",
                   "LB_sq_pass_10", "normality_pass", "diagnostic_pass"]].to_string(index=False))

n_pass = results_df["diagnostic_pass"].sum()
print(f"\n{n_pass}/{len(results_df)} meters pass full diagnostics")
print("Done.")
