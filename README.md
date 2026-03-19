# Box–Jenkins Time Series Analysis of Water-Domain Data

Application of the **Box–Jenkins SARIMA/ARIMA methodology** to two water-domain
datasets — one seasonal, one non-seasonal — following the iterative framework of
identification, estimation, diagnostic checking, and forecasting.

> Course project for *Time Series Analysis* (Prof. A. Magrini),
> Master in Statistical Learning and Data Science,
> Università degli Studi di Firenze — A.Y. 2024–2025.

---

## Methodology

The analysis follows the classical Box–Jenkins procedure as presented in
Box et al. (2015):

1. **Stationarity assessment** — visual inspection, ADF and KPSS dual-test
   strategy, differencing, heteroscedasticity check (rolling mean–variance
   correlation), variance-stabilising transformations (log / Box–Cox) when
   needed.
2. **Model identification** — ACF/PACF analysis combined with exhaustive
   `auto_arima` search (AICc selection) and manual candidate comparison.
3. **Diagnostic checking** — six-panel residual display (residuals over time,
   ACF of residuals, ACF of squared residuals, fitted vs. residuals,
   fitted vs. squared residuals, Q–Q plot), Ljung–Box test on residuals
   and squared residuals, Shapiro–Wilk normality test.
4. **Forecasting** — rolling-window cross-validation (equivalent to R's
   `tsCV`), RMSE, MAE, and Scaled RMSE against a naïve / seasonal-naïve
   benchmark.

## Datasets

| Dataset | Type | Observations | Period | Seasonality | Source |
|---------|------|-------------|--------|-------------|--------|
| Terranova smart water meters | Water consumption (m³/day) | 54–101 days × 90 meters | Jan–Apr 2025 | Weekly (*m* = 7) | Terranova network |
| S Wichita Rv specific conductance | Water quality (µS/cm) | 3 612 daily values | 2015–2024 | None (*Fs* = 0.11) | [USGS site 07311782](https://waterdata.usgs.gov/nwis) |

**Data files are not included in this repository.**
Place them in `data/raw/` before running the pipeline:

```
data/raw/
├── Final-Orso.xlsx              # Terranova hourly register readings
└── usgs_water_quality.csv       # USGS daily specific conductance
```

## Project structure

```
├── run_pipeline.py              # Orchestrates the full pipeline
├── requirements.txt             # Python dependencies
├── src/
│   ├── utils.py                 # Shared utilities (paths, plotting, tests)
│   ├── 01_eda.ipynb             # Exploratory data analysis (notebook)
│   ├── 02_preprocessing.py      # Data cleaning, selection, daily aggregation
│   ├── 03_stationarity.py       # ADF/KPSS tests, differencing, ACF/PACF
│   ├── 03_box_jenkins_analysis.ipynb  # Interactive ACF/PACF exploration
│   ├── 04_sarima_fitting.py     # SARIMA identification and estimation
│   ├── 05_diagnostics.py        # Residual diagnostics, Ljung-Box tests
│   ├── 06_forecasting.py        # Rolling-window CV, forecast metrics
│   ├── 07_anomaly_detection.py  # SARIMA-based anomaly flagging
│   ├── 08_arima_nonseasonal.py  # Non-seasonal ARIMA (USGS water quality)
│   ├── 09_fleet_preprocessing.py # Preprocessing for all 90 eligible meters
│   ├── 10_fleet_sarima.py       # Automated fleet-wide SARIMA pipeline
│   └── 11_fleet_analysis.ipynb  # Fleet results summary (notebook)
├── data/
│   ├── raw/                     # Input data (not tracked)
│   └── processed/               # Intermediate outputs (not tracked)
└── output/
    ├── figures/                  # Generated plots (not tracked)
    └── tables/                  # Generated CSV/LaTeX tables (not tracked)
```

## Quick start

```bash
# 1. Clone and set up environment
git clone https://github.com/battles5/time-series-analysis.git
cd time-series-analysis
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

# 2. Place data files in data/raw/ (see above)

# 3. Run the seasonal case studies (meters 503, 575, 345)
python run_pipeline.py

# 4. Run the non-seasonal ARIMA analysis
python run_pipeline.py --nonseasonal

# 5. Run the fleet-wide automated pipeline (90 meters)
python run_pipeline.py --fleet

# Or run everything at once
python run_pipeline.py --all
```

## Pipeline overview

### Seasonal analysis (SARIMA)

Three meters are analysed in detail to illustrate different scenarios:

| Meter | Days | Difficulty | Best model | Scaled RMSE (*h* = 1) |
|-------|------|-----------|-----------|----------------------|
| 503   | 101  | Moderate — rich seasonal structure | SARIMA(0,1,1)(0,1,1)₇ | 0.81 |
| 575   | 54   | Hard — extreme variance, Box-Cox needed | SARIMA(1,1,1)(0,1,1)₇ | 1.01 |
| 345   | 101  | Easy — regular weekly pattern | SARIMA(0,0,0)(1,0,0)₇ | 0.67 |

The methodology is then automated across **90 eligible meters**:
100% fit success, 95.6% diagnostic pass rate, median Scaled RMSE of 0.77.

### Non-seasonal analysis (ARIMA)

Daily specific conductance from the S Wichita River (a naturally hyper-saline
river in Texas) provides a long, clearly non-seasonal series:

| Model | AICc | Ljung-Box (lag 10) | Scaled RMSE (*h* = 1) | Scaled RMSE (*h* = 7) |
|-------|------|-------------------|----------------------|----------------------|
| ARIMA(1,1,3) | 66 510 | *p* = 0.56 (pass) | 0.97 | 0.91 |

## Requirements

- Python ≥ 3.10
- See [requirements.txt](requirements.txt) for packages

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J. & Khandakar, Y. (2008). Automatic Time Series Forecasting:
  The forecast Package for R. *J. Stat. Software*, 27(3).
- U.S. Geological Survey (2024). National Water Information System.
  https://waterdata.usgs.gov/nwis
