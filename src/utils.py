"""
Shared utilities for the Time Series Analysis exam project.
Water consumption anomaly detection via SARIMA-based methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_FIGURES = os.path.join(PROJECT_DIR, "output", "figures")
OUTPUT_TABLES = os.path.join(PROJECT_DIR, "output", "tables")

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    sns.set_palette("colorblind")


def save_fig(fig, name, subdir=""):
    path = os.path.join(OUTPUT_FIGURES, subdir) if subdir else OUTPUT_FIGURES
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{name}.pdf"))
    fig.savefig(os.path.join(path, f"{name}.png"))
    plt.close(fig)


def save_table(df, name):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_TABLES, f"{name}.csv"))
    with open(os.path.join(OUTPUT_TABLES, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(float_format="%.4f"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw_data():
    path = os.path.join(DATA_RAW, "water_meters.xlsx")
    df = pd.read_excel(path, engine="openpyxl")
    return df


def load_processed_data():
    path = os.path.join(DATA_PROCESSED, "meters_daily.parquet")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------
def stationarity_tests(series, name="series", verbose=True):
    """Run ADF and KPSS tests and return results as dict."""
    series_clean = series.dropna()

    # ADF test (H0: unit root exists)
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series_clean, autolag="AIC")

    # KPSS test (H0: series is stationary)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series_clean, regression="c", nlags="auto")

    results = {
        "ADF_statistic": adf_stat,
        "ADF_p_value": adf_p,
        "ADF_lags": adf_lags,
        "KPSS_statistic": kpss_stat,
        "KPSS_p_value": kpss_p,
        "KPSS_lags": kpss_lags,
    }

    if verbose:
        print(f"\n--- Stationarity tests for {name} ---")
        print(f"ADF:  stat={adf_stat:.4f}, p={adf_p:.4f}  "
              f"{'REJECT H0 (stationary)' if adf_p < 0.05 else 'FAIL TO REJECT H0 (unit root)'}")
        print(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f}  "
              f"{'REJECT H0 (non-stationary)' if kpss_p < 0.05 else 'FAIL TO REJECT H0 (stationary)'}")

    return results


# ---------------------------------------------------------------------------
# ACF / PACF plots
# ---------------------------------------------------------------------------
def plot_acf_pacf(series, lags=48, title="", save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f"ACF — {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title=f"PACF — {title}", method="ywm")
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots (residual analysis)
# ---------------------------------------------------------------------------
def plot_diagnostics(residuals, fitted, title="", save_name=None):
    """Full diagnostic panel: 6 plots as in the course R scripts."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Residuals over time
    axes[0, 0].plot(residuals, linewidth=0.7)
    axes[0, 0].axhline(0, color="grey", linewidth=0.8)
    axes[0, 0].set_title("Residuals")
    axes[0, 0].set_xlabel("Time")

    # 2. ACF of residuals
    plot_acf(residuals.dropna(), ax=axes[0, 1], title="ACF of residuals")

    # 3. ACF of squared residuals
    plot_acf(residuals.dropna() ** 2, ax=axes[1, 0], title="ACF of squared residuals")

    # 4. Fitted vs residuals
    axes[1, 1].scatter(fitted, residuals, s=8, alpha=0.5)
    axes[1, 1].axhline(0, color="grey", linewidth=0.8)
    axes[1, 1].set_xlabel("Fitted values")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Fitted vs Residuals")

    # 5. Fitted vs squared residuals
    axes[2, 0].scatter(fitted, residuals ** 2, s=8, alpha=0.5)
    axes[2, 0].set_xlabel("Fitted values")
    axes[2, 0].set_ylabel("Squared residuals")
    axes[2, 0].set_title("Fitted vs Squared Residuals")

    # 6. QQ plot
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[2, 1])
    axes[2, 1].set_title("Normal Q-Q Plot")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


def ljung_box_test(residuals, lags=None, verbose=True):
    """Ljung-Box test on residuals and squared residuals."""
    if lags is None:
        lags = [5, 10, 15]

    lb_res = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(residuals.dropna() ** 2, lags=lags, return_df=True)

    if verbose:
        print("\nLjung-Box test on residuals:")
        print(lb_res.to_string())
        print("\nLjung-Box test on squared residuals:")
        print(lb_sq.to_string())

    return lb_res, lb_sq


# ---------------------------------------------------------------------------
# Rolling window cross-validation (equivalent to forecast::tsCV in R)
# ---------------------------------------------------------------------------
def time_series_cv(series, model_func, h=1, initial=None):
    """
    Rolling-window cross-validation for time series forecasting.
    Equivalent to forecast::tsCV() in R.

    Parameters
    ----------
    series : pd.Series with DatetimeIndex
    model_func : callable(train_series, h) -> np.array of h forecasts
    h : int, forecast horizon
    initial : int, minimum training size (default: len(series) // 2)

    Returns
    -------
    errors : pd.DataFrame with columns h=1,...,h
    """
    T = len(series)
    if initial is None:
        initial = T // 2

    errors = {}
    for t in range(initial, T - h):
        train = series.iloc[:t]
        try:
            forecasts = model_func(train, h)
            actual = series.iloc[t:t + h].values
            errors[series.index[t]] = actual - forecasts
        except Exception:
            errors[series.index[t]] = np.full(h, np.nan)

    return pd.DataFrame(errors, index=[f"h={i+1}" for i in range(h)]).T


def forecast_metrics(errors, actuals=None):
    """Compute RMSE, MAE, MAPE from cross-validation errors."""
    results = {}
    for col in errors.columns:
        e = errors[col].dropna()
        results[col] = {
            "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAE": np.mean(np.abs(e)),
        }
    return pd.DataFrame(results).T
