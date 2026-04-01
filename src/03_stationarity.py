"""
03_stationarity.py
------------------
Stationarity analysis for selected meters.
Following the Box-Jenkins procedure (course slides 101-119):
  1. Visual inspection of time series
  2. ACF/PACF analysis
  3. ADF and KPSS unit root tests
  4. Differencing (regular and seasonal)
  5. Transformation assessment (log, Box-Cox)

Outputs:
  - output/figures/stationarity/ (plots per meter)
  - output/tables/stationarity_tests.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DATA_PROCESSED, set_plot_style, save_fig, save_table,
    stationarity_tests, plot_acf_pacf,
)

set_plot_style()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
hourly = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_hourly.parquet"))
daily = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_daily.parquet"))
metadata = pd.read_csv(os.path.join(DATA_PROCESSED, "meter_metadata.csv"))

meter_ids = sorted(hourly["meter_id"].unique())
print(f"Analyzing {len(meter_ids)} meters: {meter_ids}")

# We focus on daily data for SARIMA(p,d,q)(P,D,Q)_7
# and hourly data for SARIMA(p,d,q)(P,D,Q)_24
# Primary analysis: daily (cleaner, m=7 weekly seasonality)

all_results = []

for mid in meter_ids:
    print(f"\n{'='*60}")
    print(f"METER {mid}")
    print(f"{'='*60}")

    # Get daily series
    m_daily = daily[daily["meter_id"] == mid].copy()
    m_daily = m_daily.set_index("date").sort_index()
    y = m_daily["daily_consumption"]

    if len(y) < 20:
        print(f"  Skipping: only {len(y)} days")
        continue

    # ----- 1. Original series -----
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(y.index, y.values, linewidth=0.8)
    axes[0].set_title(f"Meter {mid} - Daily consumption (m³/day)")
    axes[0].set_ylabel("m³/day")
    axes[0].axhline(y.mean(), color="red", linestyle="--", alpha=0.5, label=f"mean={y.mean():.4f}")
    axes[0].legend()

    from statsmodels.graphics.tsaplots import plot_acf as _plot_acf
    _plot_acf(y.dropna(), lags=min(40, len(y) // 2 - 1), ax=axes[1], title=f"ACF - Meter {mid} (original)")
    fig.tight_layout()
    save_fig(fig, f"meter_{mid}_original", subdir="stationarity")

    # ----- 2. Stationarity tests on original -----
    res_orig = stationarity_tests(y, name=f"Meter {mid} (original)")

    # ----- 3. First difference (d=1) -----
    dy = y.diff().dropna()
    res_d1 = stationarity_tests(dy, name=f"Meter {mid} (d=1)")

    fig = plot_acf_pacf(dy, lags=min(35, len(dy) // 2 - 1),
                        title=f"Meter {mid} - diff(y)",
                        save_name=None)
    save_fig(fig, f"meter_{mid}_diff1_acf_pacf", subdir="stationarity")

    # ----- 4. Seasonal difference (D=1, m=7) if enough data -----
    res_D1 = {}
    if len(y) > 21:
        dy7 = y.diff(7).dropna()
        if len(dy7) > 10:
            res_D1 = stationarity_tests(dy7, name=f"Meter {mid} (D=1, m=7)")

            fig = plot_acf_pacf(dy7, lags=min(28, len(dy7) // 2 - 1),
                                title=f"Meter {mid} - seasonal diff(y, 7)",
                                save_name=None)
            save_fig(fig, f"meter_{mid}_sdiff7_acf_pacf", subdir="stationarity")

    # ----- 5. Both differences (d=1, D=1) -----
    res_dD = {}
    if len(y) > 21:
        ddy7 = y.diff(7).diff().dropna()
        if len(ddy7) > 10:
            res_dD = stationarity_tests(ddy7, name=f"Meter {mid} (d=1, D=1)")

            fig = plot_acf_pacf(ddy7, lags=min(28, len(ddy7) // 2 - 1),
                                title=f"Meter {mid} - diff(diff(y, 7))",
                                save_name=None)
            save_fig(fig, f"meter_{mid}_diff1_sdiff7_acf_pacf", subdir="stationarity")

    # ----- 6. Check for heteroscedasticity (need for transformation?) -----
    # Simple check: correlation between level and variance (rolling window)
    if len(y) >= 14:
        rolling_mean = y.rolling(7).mean().dropna()
        rolling_std = y.rolling(7).std().dropna()
        common_idx = rolling_mean.index.intersection(rolling_std.index)
        if len(common_idx) > 5:
            corr_level_var = np.corrcoef(
                rolling_mean.loc[common_idx].values,
                rolling_std.loc[common_idx].values
            )[0, 1]
            print(f"  Correlation(rolling mean, rolling std): {corr_level_var:.3f}")
            print(f"  {'-> Consider log/Box-Cox transformation' if abs(corr_level_var) > 0.5 else '-> No strong heteroscedasticity'}")
        else:
            corr_level_var = np.nan
    else:
        corr_level_var = np.nan

    # Store results
    all_results.append({
        "meter_id": mid,
        "n_days": len(y),
        "mean": y.mean(),
        "std": y.std(),
        "zero_frac": (y == 0).mean(),
        # Original series
        "ADF_p_orig": res_orig["ADF_p_value"],
        "KPSS_p_orig": res_orig["KPSS_p_value"],
        "stationary_orig": res_orig["ADF_p_value"] < 0.05 and res_orig["KPSS_p_value"] >= 0.05,
        # After d=1
        "ADF_p_d1": res_d1["ADF_p_value"],
        "KPSS_p_d1": res_d1["KPSS_p_value"],
        "stationary_d1": res_d1["ADF_p_value"] < 0.05 and res_d1["KPSS_p_value"] >= 0.05,
        # After D=1
        "ADF_p_D1": res_D1.get("ADF_p_value", np.nan),
        "KPSS_p_D1": res_D1.get("KPSS_p_value", np.nan),
        # After d=1, D=1
        "ADF_p_dD": res_dD.get("ADF_p_value", np.nan),
        "KPSS_p_dD": res_dD.get("KPSS_p_value", np.nan),
        # Heteroscedasticity
        "corr_level_var": corr_level_var,
        "needs_transform": abs(corr_level_var) > 0.5 if not np.isnan(corr_level_var) else False,
    })

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
save_table(results_df, "stationarity_tests")

print("\n" + "=" * 60)
print("STATIONARITY SUMMARY")
print("=" * 60)
print(results_df[["meter_id", "n_days", "stationary_orig", "stationary_d1",
                   "ADF_p_dD", "needs_transform"]].to_string(index=False))
print("\nDone.")
