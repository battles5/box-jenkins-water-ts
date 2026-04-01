"""
09_fleet_sarima.py
------------------
Fully automated Box-Jenkins SARIMA pipeline for all eligible meters.

For each meter:
  1. Heteroscedasticity check → Box-Cox if needed
  2. Stationarity: try d=0,1,2 × D=0,1 with dual ADF+KPSS test
  3. auto_arima (stepwise, m=7)
  4. Ljung-Box diagnostic (residuals + squared residuals)
  5. Rolling-window CV (Scaled RMSE vs seasonal naive)
  6. Standardized residuals → anomaly flags (point, persistent, level-shift)
  7. Map anomalies to Kimiya taxonomy (Types 1-7)

Outputs:
  - data/processed/fleet_results.parquet     (one row per meter)
  - data/processed/fleet_anomalies.parquet   (one row per anomaly-day)
  - data/processed/fleet_models.pkl          (serialized models)
  - output/figures/fleet/                    (per-meter diagnostic + forecast PNGs)
"""

import sys
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats as sp_stats
from scipy.special import inv_boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import PROJECT_DIR, DATA_PROCESSED, OUTPUT_FIGURES

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
SEASONAL_PERIOD = 7
HORIZON = 7
HETERO_THRESHOLD = 0.5       # rolling mean-var correlation for Box-Cox
SIGMA_THRESHOLD = 2.5        # point anomaly (standardized residual)
RUN_LENGTH = 5               # persistent anomaly (consecutive positive days)
LEVEL_SHIFT_WINDOW = 7       # rolling window for level shift
LEVEL_SHIFT_THRESHOLD = 1.5  # sigma for level shift

FLEET_FIG_DIR = os.path.join(OUTPUT_FIGURES, "fleet")
os.makedirs(FLEET_FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def dual_stationarity_test(series):
    """Return (adf_p, kpss_p). Series must be clean (no NaN)."""
    s = series.dropna()
    if len(s) < 15:
        return (1.0, 0.0)  # not enough data → assume non-stationary
    adf_p = adfuller(s, autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    return (adf_p, kpss_p)


def is_stationary(series):
    """Stationary if ADF rejects (p<0.05) AND KPSS does not reject (p>=0.05)."""
    adf_p, kpss_p = dual_stationarity_test(series)
    return adf_p < 0.05 and kpss_p >= 0.05, adf_p, kpss_p


def find_differencing(series, m=7):
    """
    Try d=0,1,2 and D=0,1 to find the minimum differencing that
    achieves stationarity. Returns (d, D, differenced_series).
    """
    for D in [0, 1]:
        s = series.copy()
        if D == 1:
            s = s.diff(m).dropna()
        for d in [0, 1, 2]:
            s_d = s.copy()
            for _ in range(d):
                s_d = s_d.diff().dropna()
            if len(s_d) < 15:
                continue
            ok, _, _ = is_stationary(s_d)
            if ok:
                return d, D, s_d
    # Fallback: d=1, D=0
    return 1, 0, series.diff().dropna()


def check_heteroscedasticity(series, window=7):
    """Return correlation between rolling mean and rolling variance."""
    rm = series.rolling(window).mean().dropna()
    rv = series.rolling(window).var().dropna()
    common = rm.index.intersection(rv.index)
    if len(common) < 10:
        return 0.0
    return abs(rm.loc[common].corr(rv.loc[common]))


def detect_anomalies(residuals, std_residuals):
    """
    Detect anomalies from standardized residuals. Returns dict with:
      - point_mask, persistent_mask, level_shift_mask
      - n_point, n_persistent_runs, n_level_shift_days
      - anomaly_any mask
    """
    n = len(std_residuals)

    # 1. Point anomalies
    point_mask = np.abs(std_residuals) > SIGMA_THRESHOLD
    point_high = std_residuals > SIGMA_THRESHOLD
    point_low = std_residuals < -SIGMA_THRESHOLD

    # 2. Persistent runs of positive residuals
    positive = std_residuals > 0
    runs = []
    run_start = None
    run_len = 0
    for i in range(n):
        if positive[i]:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= RUN_LENGTH:
                runs.append((run_start, run_start + run_len - 1))
            run_start = None
            run_len = 0
    if run_len >= RUN_LENGTH:
        runs.append((run_start, run_start + run_len - 1))

    persistent_mask = np.zeros(n, dtype=bool)
    for s, e in runs:
        persistent_mask[s:e + 1] = True

    # 3. Level shift
    rolling_res = pd.Series(std_residuals).rolling(LEVEL_SHIFT_WINDOW, center=True).mean()
    level_shift_mask = np.abs(rolling_res.values) > LEVEL_SHIFT_THRESHOLD
    level_shift_mask = np.nan_to_num(level_shift_mask, nan=0).astype(bool)

    any_anomaly = point_mask | persistent_mask | level_shift_mask

    return {
        "point_mask": point_mask,
        "point_high": point_high,
        "point_low": point_low,
        "persistent_mask": persistent_mask,
        "persistent_runs": runs,
        "level_shift_mask": level_shift_mask,
        "any_anomaly": any_anomaly,
        "n_point": int(point_mask.sum()),
        "n_point_high": int(point_high.sum()),
        "n_point_low": int(point_low.sum()),
        "n_persistent_runs": len(runs),
        "n_level_shift_days": int(level_shift_mask.sum()),
        "n_anomaly_days": int(any_anomaly.sum()),
    }


def map_kimiya_type(point_high, point_low, persistent, level_shift):
    """
    Map our detection flags to Kimiya anomaly taxonomy.
    Returns list of type strings.
    """
    types = []
    if point_high or point_low:
        types.append("Type1_Spike")
    if persistent and not level_shift:
        types.append("Type3_MultiDay")
    if level_shift:
        types.append("Type5_BaselineLeak")
    # Types 2, 4, 6, 7 require hourly data - cannot detect with daily SARIMA
    return types if types else ["none"]


def rolling_cv(y_arr, order, seasonal_order, lam=None, shift=0.0,
               horizon=HORIZON):
    """
    Rolling-window cross-validation. Returns dict with Scaled RMSE at h=1, h=7.
    """
    T = len(y_arr)
    initial = max(T // 2, 21)
    errors = {h: [] for h in range(1, horizon + 1)}
    naive_errors = {h: [] for h in range(1, horizon + 1)}
    n_folds = 0

    for t in range(initial, T - horizon):
        train = y_arr[:t]
        try:
            if lam is not None:
                train_bc = sp_stats.boxcox(train + shift, lmbda=lam)
            else:
                train_bc = train

            m = pm.ARIMA(order=order, seasonal_order=seasonal_order,
                         suppress_warnings=True)
            m.fit(train_bc)
            fc = m.predict(n_periods=horizon)

            if lam is not None:
                fc = inv_boxcox(fc, lam) - shift

            for h in range(1, horizon + 1):
                actual = y_arr[t + h - 1]
                errors[h].append(actual - fc[h - 1])
                if t + h - 1 - SEASONAL_PERIOD >= 0:
                    naive_errors[h].append(actual - y_arr[t + h - 1 - SEASONAL_PERIOD])
                else:
                    naive_errors[h].append(np.nan)
            n_folds += 1
        except Exception:
            for h in range(1, horizon + 1):
                errors[h].append(np.nan)
                naive_errors[h].append(np.nan)

    result = {"n_folds": n_folds, "n_total": T - horizon - initial}
    for h in [1, 7]:
        e = np.array(errors[h])
        ne = np.array(naive_errors[h])
        rmse = np.sqrt(np.nanmean(e**2))
        n_rmse = np.sqrt(np.nanmean(ne**2))
        result[f"rmse_h{h}"] = rmse
        result[f"naive_rmse_h{h}"] = n_rmse
        result[f"scaled_rmse_h{h}"] = rmse / n_rmse if n_rmse > 0 else np.nan
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_meter(meter_id, y_series):
    """
    Full automated Box-Jenkins pipeline for a single meter.
    Returns (result_dict, anomaly_records_list, model_data_dict).
    """
    t0 = time.time()
    y = y_series.copy().astype(float)
    n_days = len(y)
    result = {"meter_id": meter_id, "n_days": n_days}

    # ── Step 1: Heteroscedasticity check ──
    corr = check_heteroscedasticity(y)
    needs_transform = corr > HETERO_THRESHOLD
    result["hetero_corr"] = round(corr, 4)
    result["transform"] = "none"
    lam = None
    shift = 0.0

    if needs_transform:
        shift = y[y > 0].min() / 2 if (y <= 0).any() else 0.0
        y_shifted = y + shift
        try:
            y_bc, lam = sp_stats.boxcox(y_shifted.values)
            y_model = pd.Series(y_bc, index=y.index)
            result["transform"] = f"boxcox(lam={lam:.3f})"
            result["boxcox_lambda"] = round(lam, 4)
        except Exception:
            y_model = y
            lam = None
            result["transform"] = "boxcox_failed"
    else:
        y_model = y
        result["boxcox_lambda"] = np.nan

    # ── Step 2: Stationarity ──
    d, D, _ = find_differencing(y_model, m=SEASONAL_PERIOD)
    stat_ok, adf_p, kpss_p = is_stationary(
        y_model.diff().dropna() if d >= 1 else y_model
    )
    result["d"] = d
    result["D"] = D
    result["adf_p"] = round(adf_p, 4)
    result["kpss_p"] = round(kpss_p, 4)
    result["stationary"] = stat_ok

    # ── Step 3: auto_arima ──
    try:
        model = pm.auto_arima(
            y_model.values,
            d=d, D=D, m=SEASONAL_PERIOD,
            seasonal=True, stepwise=True,
            suppress_warnings=True, error_action="ignore",
            max_p=4, max_q=4, max_P=2, max_Q=2,
            max_d=2, max_D=1,
            information_criterion="aicc",
        )
        result["order"] = str(model.order)
        result["seasonal_order"] = str(model.seasonal_order)
        result["aicc"] = round(model.aicc(), 2)
        result["bic"] = round(model.bic(), 2)
        result["n_params"] = int(model.df_model())
        result["fit_success"] = True
    except Exception as e:
        result["order"] = "FAILED"
        result["seasonal_order"] = "FAILED"
        result["aicc"] = np.nan
        result["bic"] = np.nan
        result["n_params"] = 0
        result["fit_success"] = False
        result["error"] = str(e)[:100]
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result, [], None

    # ── Step 4: Diagnostics ──
    fitted_vals = model.predict_in_sample()
    residuals = y_model.values - fitted_vals
    res_std = residuals / np.nanstd(residuals)

    try:
        lb_res = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
        lb_sq = acorr_ljungbox(residuals**2, lags=[5, 10], return_df=True)
        lb_res_p10 = lb_res["lb_pvalue"].iloc[-1]
        lb_sq_p10 = lb_sq["lb_pvalue"].iloc[-1]
        diag_pass = lb_res_p10 > 0.05 and lb_sq_p10 > 0.05
    except Exception:
        lb_res_p10 = np.nan
        lb_sq_p10 = np.nan
        diag_pass = False

    result["lb_resid_p"] = round(lb_res_p10, 4) if not np.isnan(lb_res_p10) else np.nan
    result["lb_sq_p"] = round(lb_sq_p10, 4) if not np.isnan(lb_sq_p10) else np.nan
    result["diag_pass"] = diag_pass

    # Shapiro-Wilk
    try:
        sw_stat, sw_p = sp_stats.shapiro(residuals[:min(len(residuals), 5000)])
        result["shapiro_p"] = round(sw_p, 4)
    except Exception:
        result["shapiro_p"] = np.nan

    # ── Step 5: Cross-validation ──
    try:
        cv = rolling_cv(y.values, model.order, model.seasonal_order,
                         lam=lam, shift=shift)
        result.update(cv)
    except Exception:
        result["n_folds"] = 0
        result["scaled_rmse_h1"] = np.nan
        result["scaled_rmse_h7"] = np.nan

    # ── Step 6: Anomaly detection ──
    anom = detect_anomalies(residuals, res_std)
    result["n_point_anomalies"] = anom["n_point"]
    result["n_persistent_runs"] = anom["n_persistent_runs"]
    result["n_level_shift_days"] = anom["n_level_shift_days"]
    result["n_anomaly_days"] = anom["n_anomaly_days"]
    result["pct_anomaly"] = round(anom["n_anomaly_days"] / n_days * 100, 2)

    # ── Step 7: Build anomaly records with Kimiya mapping ──
    anomaly_records = []
    dates = y.index
    for i in range(n_days):
        if anom["any_anomaly"][i]:
            kimiya_types = map_kimiya_type(
                bool(anom["point_high"][i]),
                bool(anom["point_low"][i]),
                bool(anom["persistent_mask"][i]),
                bool(anom["level_shift_mask"][i]),
            )
            anomaly_records.append({
                "meter_id": meter_id,
                "date": dates[i],
                "consumption": y.iloc[i],
                "std_residual": round(float(res_std[i]), 4),
                "point": bool(anom["point_mask"][i]),
                "persistent": bool(anom["persistent_mask"][i]),
                "level_shift": bool(anom["level_shift_mask"][i]),
                "kimiya_types": "|".join(kimiya_types),
            })

    result["elapsed_s"] = round(time.time() - t0, 1)

    # Model data for serialization
    model_data = {
        "model": model,
        "y_original": y,
        "y_model": y_model,
        "fitted": fitted_vals,
        "residuals": residuals,
        "std_residuals": res_std,
        "transform": result["transform"],
        "lambda": lam,
        "shift": shift,
    }

    return result, anomaly_records, model_data


# ═══════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  FLEET-SCALE AUTOMATED SARIMA PIPELINE")
    print("=" * 70)

    # Load fleet data
    daily = pd.read_parquet(os.path.join(DATA_PROCESSED, "meters_daily_fleet.parquet"))
    meter_ids = sorted(daily["meter_id"].unique())
    print(f"  Meters to process: {len(meter_ids)}")
    print(f"  Total daily rows : {len(daily)}")

    all_results = []
    all_anomalies = []
    all_models = {}
    failed = []

    t_start = time.time()

    for idx, mid in enumerate(meter_ids):
        m_data = daily[daily["meter_id"] == mid].copy()
        m_data = m_data.set_index("date").sort_index()
        y = m_data["daily_consumption"]

        print(f"\n[{idx+1:3d}/{len(meter_ids)}] Meter {mid} ({len(y)} days) ... ", end="", flush=True)

        result, anomalies, model_data = process_meter(mid, y)
        all_results.append(result)
        all_anomalies.extend(anomalies)

        if model_data is not None:
            all_models[mid] = model_data

        status = "PASS" if result.get("diag_pass") else "FAIL"
        if not result.get("fit_success"):
            status = "FIT_ERROR"
            failed.append(mid)

        sr = result.get("scaled_rmse_h1", np.nan)
        sr_str = f"{sr:.3f}" if not np.isnan(sr) else "N/A"
        print(f"{result.get('order','?')}×{result.get('seasonal_order','?')} "
              f"| Diag: {status} | ScRMSE(h=1): {sr_str} "
              f"| Anom: {result.get('n_anomaly_days', 0)} days "
              f"| {result.get('elapsed_s', 0):.1f}s")

    elapsed_total = time.time() - t_start

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(DATA_PROCESSED, "fleet_results.parquet"), index=False)
    results_df.to_csv(os.path.join(DATA_PROCESSED, "fleet_results.csv"), index=False)

    if all_anomalies:
        anom_df = pd.DataFrame(all_anomalies)
        anom_df.to_parquet(os.path.join(DATA_PROCESSED, "fleet_anomalies.parquet"), index=False)
    else:
        anom_df = pd.DataFrame()

    with open(os.path.join(DATA_PROCESSED, "fleet_models.pkl"), "wb") as f:
        pickle.dump(all_models, f)

    # ── Summary ──
    n_total = len(meter_ids)
    n_success = results_df["fit_success"].sum()
    n_diag_pass = results_df["diag_pass"].sum()

    print(f"\n{'=' * 70}")
    print(f"  FLEET PIPELINE COMPLETE - {elapsed_total:.0f}s total")
    print(f"{'=' * 70}")
    print(f"  Meters processed:     {n_total}")
    print(f"  Fit success:          {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")
    print(f"  Diagnostic pass:      {n_diag_pass}/{n_success} ({n_diag_pass/max(n_success,1)*100:.1f}%)")
    if n_success > 0:
        sr1 = results_df.loc[results_df["fit_success"], "scaled_rmse_h1"]
        sr7 = results_df.loc[results_df["fit_success"], "scaled_rmse_h7"]
        print(f"  Scaled RMSE h=1:      median={sr1.median():.3f}, mean={sr1.mean():.3f}")
        print(f"  Scaled RMSE h=7:      median={sr7.median():.3f}, mean={sr7.mean():.3f}")
        print(f"  Meters with anomalies: {(results_df['n_anomaly_days'] > 0).sum()}")
        print(f"  Total anomaly days:   {results_df['n_anomaly_days'].sum()}")
    if failed:
        print(f"  Failed meters:        {failed}")

    # Kimiya type summary
    if len(anom_df) > 0:
        print(f"\n  Kimiya Anomaly Type Coverage (daily SARIMA baseline):")
        print(f"  {'Type':<25s} {'Detected':>10s} {'Note'}")
        print(f"  {'-'*60}")
        for kt in ["Type1_Spike", "Type3_MultiDay", "Type5_BaselineLeak"]:
            n = anom_df["kimiya_types"].str.contains(kt).sum()
            print(f"  {kt:<25s} {n:>10d}   ✓ detectable")
        for kt, note in [("Type2_NightTime", "requires hourly"),
                          ("Type4_TemporalAHC", "requires hourly"),
                          ("Type6_ACNZ", "requires hourly"),
                          ("Type7_ACS", "requires hourly")]:
            print(f"  {kt:<25s} {'-':>10s}   ✗ {note}")

    print(f"\n  Outputs saved to: {DATA_PROCESSED}/")
    print("  Done.")
