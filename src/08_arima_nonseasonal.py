"""
08_arima_nonseasonal.py
-----------------------
Non-seasonal ARIMA analysis of USGS water quality data (specific conductance).

Site: S Wichita Rv at Low Flow Dam nr Guthrie, TX (USGS 07311782)
Parameter: specific conductance (µS/cm), daily values, 2015-01-01 to 2024-11-20

Steps:
  1. Load and preprocess (interpolate missing dates)
  2. Stationarity assessment (ADF + KPSS, original and differenced)
  3. ACF/PACF analysis
  4. Heteroscedasticity check
  5. Model identification via auto_arima (seasonal=False)
  6. Diagnostic checking (6-panel plot, Ljung-Box, Shapiro-Wilk)
  7. Rolling-window cross-validation with naive benchmark
  8. Forecast visualization

Outputs:
  - output/figures/nonseasonal/ (all plots)
  - output/tables/nonseasonal_*.csv and .tex
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pmdarima as pm
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DATA_RAW, set_plot_style, save_fig, save_table,
    stationarity_tests, plot_acf_pacf, plot_diagnostics, ljung_box_test,
)

set_plot_style()

SUBDIR = "nonseasonal"
HORIZON = 7

# USGS Water Services URL for automatic data retrieval
USGS_URL = (
    "https://waterservices.usgs.gov/nwis/dv/"
    "?sites=07311782&parameterCd=00095"
    "&startDT=2015-01-01&endDT=2024-11-20&format=rdb"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and preprocess
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

csv_path = os.path.join(DATA_RAW, "usgs_water_quality.csv")
if not os.path.exists(csv_path):
    print(f"Local file not found, downloading from USGS Water Services...")
    import urllib.request, io
    os.makedirs(DATA_RAW, exist_ok=True)
    response = urllib.request.urlopen(USGS_URL)
    lines = response.read().decode("utf-8").splitlines()
    # skip comment (#) and format lines; keep header + data
    data_lines = [l for l in lines if not l.startswith("#")]
    # first non-comment line is header, second is dtype descriptor
    header = data_lines[0].split("\t")
    rows = [l.split("\t") for l in data_lines[2:]]
    raw_df = pd.DataFrame(rows, columns=header)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw_df["datetime"]),
        "specific_conductance_us_cm": pd.to_numeric(
            raw_df.iloc[:, 4], errors="coerce"  # 5th column = value
        ),
    }).dropna()
    out.to_csv(csv_path, index=False)
    print(f"Saved {len(out)} rows to {csv_path}")

df = pd.read_csv(csv_path, parse_dates=["date"])
df = df.set_index("date").sort_index()
y_raw = df["specific_conductance_us_cm"].astype(float)

print(f"Raw observations: {len(y_raw)}")
print(f"Date range: {y_raw.index[0].date()} to {y_raw.index[-1].date()}")
print(f"Missing dates: {(pd.date_range(y_raw.index[0], y_raw.index[-1]) .difference(y_raw.index)).shape[0]}")

# Reindex to continuous daily and interpolate gaps
full_idx = pd.date_range(y_raw.index[0], y_raw.index[-1], freq="D")
y = y_raw.reindex(full_idx).interpolate(method="linear")
y.index.name = "date"

print(f"After interpolation: {len(y)} observations")
print(f"Mean={y.mean():.0f}, Std={y.std():.0f}, Min={y.min():.0f}, Max={y.max():.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Stationarity assessment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STATIONARITY ANALYSIS")
print("=" * 60)

# --- Original series ---
results_orig = stationarity_tests(y, name="original", verbose=True)

# --- Plot original series with ACF/PACF ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(y.index, y.values, linewidth=0.5, color="steelblue")
axes[0].set_title("S Wichita Rv: specific conductance (original)")
axes[0].set_ylabel("µS/cm")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(y.dropna(), lags=60, ax=axes[1], title="ACF (original)")
plot_pacf(y.dropna(), lags=60, ax=axes[2], title="PACF (original)", method="ywm")
fig.tight_layout()
save_fig(fig, "wq_original", subdir=SUBDIR)
print("  Saved: wq_original.pdf")

# --- First difference ---
y_diff = y.diff().dropna()
results_diff = stationarity_tests(y_diff, name="diff(1)", verbose=True)

# --- Plot differenced series with ACF/PACF ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(y_diff.index, y_diff.values, linewidth=0.5, color="steelblue")
axes[0].set_title("S Wichita Rv: specific conductance (differenced, $d=1$)")
axes[0].set_ylabel("$\\Delta$ µS/cm")

plot_acf(y_diff.dropna(), lags=60, ax=axes[1], title="ACF (differenced)")
plot_pacf(y_diff.dropna(), lags=60, ax=axes[2], title="PACF (differenced)", method="ywm")
fig.tight_layout()
save_fig(fig, "wq_diff1_acf_pacf", subdir=SUBDIR)
print("  Saved: wq_diff1_acf_pacf.pdf")

# --- Stationarity summary table ---
stat_rows = []
for label, res in [("Original", results_orig), ("Differenced ($d=1$)", results_diff)]:
    stat_rows.append({
        "Series": label,
        "ADF_stat": res["ADF_statistic"],
        "ADF_p": res["ADF_p_value"],
        "ADF_conclusion": "Stationary" if res["ADF_p_value"] < 0.05 else "Unit root",
        "KPSS_stat": res["KPSS_statistic"],
        "KPSS_p": res["KPSS_p_value"],
        "KPSS_conclusion": "Non-stationary" if res["KPSS_p_value"] < 0.05 else "Stationary",
    })
stat_df = pd.DataFrame(stat_rows)
save_table(stat_df, "nonseasonal_stationarity")
print("  Saved: nonseasonal_stationarity.csv")

# --- Heteroscedasticity check ---
roll_win = 30  # 30-day rolling window
roll_mean = y.rolling(roll_win).mean().dropna()
roll_var = y.rolling(roll_win).var().dropna()
common_idx = roll_mean.index.intersection(roll_var.index)
het_corr = np.corrcoef(roll_mean[common_idx], roll_var[common_idx])[0, 1]
print(f"\nHeteroscedasticity check: rolling mean-variance correlation = {het_corr:.3f}")
print(f"  |rho| {'<' if abs(het_corr) < 0.5 else '>='} 0.5 => {'no' if abs(het_corr) < 0.5 else ''} transformation needed")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Model identification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL IDENTIFICATION")
print("=" * 60)

# auto_arima with seasonal=False (non-seasonal ARIMA)
print("Running auto_arima (seasonal=False, exhaustive search)...")
auto_model = pm.auto_arima(
    y,
    d=None,              # let it determine d
    seasonal=False,      # non-seasonal
    stepwise=False,      # exhaustive search
    suppress_warnings=True,
    error_action="ignore",
    max_p=4, max_q=4,
    max_d=2,
    information_criterion="aicc",
    trace=True,
)

order = auto_model.order
aicc_auto = auto_model.aicc()
bic_auto = auto_model.bic()

print(f"\nBest model: ARIMA{order}")
print(f"AICc={aicc_auto:.2f}, BIC={bic_auto:.2f}")

# Parameter significance
params = auto_model.params()
try:
    param_names = auto_model.arima_res_.param_names
    cov = auto_model.arima_res_.cov_params()
    se = np.sqrt(np.diag(cov))
    z_stats = params / se
    p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stats)))
    print("\nParameter estimates:")
    for name, p, z, pv in zip(param_names, params, z_stats, p_vals):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"  {name:>12s}: {p:10.4f}  (z={z:6.2f}, p={pv:.4f}) {sig}")
except Exception:
    print(f"  Parameters: {params}")

# --- Also try a few manual candidates ---
print("\nManual candidate evaluation:")
candidates = [(order, "auto", aicc_auto, bic_auto, auto_model)]

manual_specs = [
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (0, 1, 2),
]

for m_order in manual_specs:
    if m_order == order:
        continue
    try:
        m = pm.ARIMA(order=m_order, suppress_warnings=True)
        m.fit(y)
        m_aicc = m.aicc()
        m_bic = m.bic()
        tag = " <-- better!" if m_aicc < aicc_auto else ""
        print(f"  ARIMA{m_order}: AICc={m_aicc:.2f}, BIC={m_bic:.2f}{tag}")
        candidates.append((m_order, "manual", m_aicc, m_bic, m))
    except Exception as e:
        print(f"  ARIMA{m_order}: failed ({e})")

# Select best by AICc
candidates.sort(key=lambda x: x[2])
best_order, best_label, best_aicc, best_bic, best_model = candidates[0]
print(f"\nSelected model: ARIMA{best_order} (source={best_label}, AICc={best_aicc:.2f})")

# --- Model comparison table ---
model_rows = []
for m_order, m_label, m_aicc, m_bic, _ in candidates:
    model_rows.append({
        "Model": f"ARIMA{m_order}",
        "Source": m_label,
        "AICc": m_aicc,
        "BIC": m_bic,
    })
model_df = pd.DataFrame(model_rows)
save_table(model_df, "nonseasonal_models")
print("  Saved: nonseasonal_models.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Diagnostic checking
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC CHECKING")
print("=" * 60)

fitted_vals = best_model.predict_in_sample()
residuals = pd.Series(y.values - fitted_vals, index=y.index)

# --- 6-panel diagnostic plot ---
fig = plot_diagnostics(
    residuals, pd.Series(fitted_vals, index=y.index),
    title=f"S Wichita Rv - ARIMA{best_order}",
)
save_fig(fig, "wq_diagnostics", subdir=SUBDIR)
print("  Saved: wq_diagnostics.pdf")

# --- Ljung-Box tests ---
lags = [5, 10, 15, 20]
lb_res, lb_sq = ljung_box_test(residuals, lags=lags, verbose=True)

# --- Shapiro-Wilk (on subsample if too large) ---
res_clean = residuals.dropna()
n_shapiro = min(len(res_clean), 5000)
_, shapiro_p = sp_stats.shapiro(res_clean[:n_shapiro])
print(f"\nShapiro-Wilk test: p={shapiro_p:.6f} ({'normal' if shapiro_p > 0.05 else 'non-normal'})")

# --- Diagnostics summary table ---
diag_results = {
    "Model": f"ARIMA{best_order}",
    "Residual_mean": res_clean.mean(),
    "Residual_std": res_clean.std(),
    "LB_resid_stat_10": lb_res.loc[10, "lb_stat"],
    "LB_resid_p_10": lb_res.loc[10, "lb_pvalue"],
    "LB_resid_pass_10": lb_res.loc[10, "lb_pvalue"] > 0.05,
    "LB_sq_stat_10": lb_sq.loc[10, "lb_stat"],
    "LB_sq_p_10": lb_sq.loc[10, "lb_pvalue"],
    "LB_sq_pass_10": lb_sq.loc[10, "lb_pvalue"] > 0.05,
    "Shapiro_p": shapiro_p,
    "Normality_pass": shapiro_p > 0.05,
}
diag_df = pd.DataFrame([diag_results])
save_table(diag_df, "nonseasonal_diagnostics")
print("  Saved: nonseasonal_diagnostics.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-validation and forecasting
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CROSS-VALIDATION AND FORECASTING")
print("=" * 60)

y_arr = y.values.astype(float)
T = len(y_arr)
initial = T - 365  # use last year for CV (keep ~2650 for training)
if initial < T // 2:
    initial = T // 2

errors_model = {h: [] for h in range(1, HORIZON + 1)}
errors_naive = {h: [] for h in range(1, HORIZON + 1)}

print(f"  CV: T={T}, initial={initial}, horizon={HORIZON}, folds={T - HORIZON - initial}")

n_success = 0
for t in range(initial, T - HORIZON):
    train = y_arr[:t]
    # Model forecast
    try:
        cv_model = pm.ARIMA(order=best_order, suppress_warnings=True)
        cv_model.fit(train)
        forecasts = cv_model.predict(n_periods=HORIZON)
        for h in range(1, HORIZON + 1):
            actual = y_arr[t + h - 1]
            errors_model[h].append(actual - forecasts[h - 1])
        n_success += 1
    except Exception:
        for h in range(1, HORIZON + 1):
            errors_model[h].append(np.nan)

    # Naive forecast: random walk (last observed value)
    naive_pred = y_arr[t - 1]
    for h in range(1, HORIZON + 1):
        actual = y_arr[t + h - 1]
        errors_naive[h].append(actual - naive_pred)

    if (t - initial) % 50 == 0:
        print(f"    fold {t - initial}/{T - HORIZON - initial} ...")

print(f"  CV folds successful: {n_success}/{T - HORIZON - initial}")

# --- Compute metrics ---
metrics_rows = []
for h in range(1, HORIZON + 1):
    e_m = np.array(errors_model[h])
    e_n = np.array(errors_naive[h])
    valid = ~np.isnan(e_m)
    e_m_v = e_m[valid]
    e_n_v = e_n[valid]

    rmse_model = np.sqrt(np.mean(e_m_v ** 2))
    mae_model = np.mean(np.abs(e_m_v))
    rmse_naive = np.sqrt(np.mean(e_n_v ** 2))
    mae_naive = np.mean(np.abs(e_n_v))
    scaled_rmse = rmse_model / rmse_naive if rmse_naive > 0 else np.nan

    metrics_rows.append({
        "Horizon": h,
        "RMSE": rmse_model,
        "MAE": mae_model,
        "Naive_RMSE": rmse_naive,
        "Naive_MAE": mae_naive,
        "Scaled_RMSE": scaled_rmse,
    })

metrics_df = pd.DataFrame(metrics_rows)
print(f"\nForecast metrics:")
print(metrics_df.to_string(index=False))

save_table(metrics_df, "nonseasonal_forecast_metrics")
print("  Saved: nonseasonal_forecast_metrics.csv")

# --- Forecast plot ---
forecasts_full = best_model.predict(n_periods=HORIZON, return_conf_int=True, alpha=0.05)
pred_mean = forecasts_full[0]
pred_ci = forecasts_full[1]

last_date = y.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Full series + forecast
axes[0].plot(y.index[-180:], y.values[-180:], label="Observed", linewidth=0.8)
axes[0].plot(future_dates, pred_mean, "r-", label="Forecast", linewidth=1.5)
axes[0].fill_between(future_dates, pred_ci[:, 0], pred_ci[:, 1],
                      alpha=0.2, color="red", label="95% CI")
axes[0].set_title(f"S Wichita Rv - ARIMA{best_order} - 7-day forecast")
axes[0].set_ylabel("µS/cm")
axes[0].legend()

# CV metrics plot
h_vals = list(range(1, HORIZON + 1))
axes[1].plot(h_vals, [metrics_rows[h-1]["RMSE"] for h in h_vals], "o-", label="ARIMA RMSE")
axes[1].plot(h_vals, [metrics_rows[h-1]["MAE"] for h in h_vals], "s--", label="ARIMA MAE")
axes[1].plot(h_vals, [metrics_rows[h-1]["Naive_RMSE"] for h in h_vals],
             "^:", label="Naive RMSE", color="grey")
axes[1].set_xlabel("Forecast horizon $h$ (days)")
axes[1].set_ylabel("Error (µS/cm)")
axes[1].set_title("Cross-validation forecast errors")
axes[1].legend()
axes[1].set_xticks(h_vals)

fig.tight_layout()
save_fig(fig, "wq_forecast", subdir=SUBDIR)
print("  Saved: wq_forecast.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Series: S Wichita Rv specific conductance (USGS 07311782)")
print(f"Observations: {T} (after interpolation)")
print(f"Selected model: ARIMA{best_order}")
print(f"AICc: {best_aicc:.2f}")
lb_pass = "PASS" if diag_results["LB_resid_pass_10"] else "FAIL"
sq_pass = "PASS" if diag_results["LB_sq_pass_10"] else "FAIL"
print(f"Ljung-Box residuals (lag=10): {lb_pass} (p={diag_results['LB_resid_p_10']:.4f})")
print(f"Ljung-Box squared (lag=10): {sq_pass} (p={diag_results['LB_sq_p_10']:.4f})")
print(f"Shapiro-Wilk: p={shapiro_p:.6f}")
print(f"Scaled RMSE h=1: {metrics_rows[0]['Scaled_RMSE']:.4f}")
print(f"Scaled RMSE h=7: {metrics_rows[6]['Scaled_RMSE']:.4f}")
print("\nDone.")
