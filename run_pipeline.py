"""
Run the full analysis pipeline.

Usage:
    python run_pipeline.py                   # seasonal case studies only
    python run_pipeline.py --all             # seasonal + non-seasonal + fleet
    python run_pipeline.py --nonseasonal     # non-seasonal case study only
    python run_pipeline.py --fleet           # fleet analysis only

Prerequisites:
    - data/raw/Final-Orso.xlsx       (Terranova smart water meter data)
    - data/raw/usgs_water_quality.csv (USGS specific conductance, site 07311782)
    - pip install -r requirements.txt
"""
import subprocess
import sys
import os
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
SEASONAL_CASE_STUDIES = [
    "src/02_preprocessing.py",
    "src/03_stationarity.py",
    "src/04_sarima_fitting.py",
    "src/05_diagnostics.py",
    "src/06_forecasting.py",
    "src/07_anomaly_detection.py",
]

NONSEASONAL = [
    "src/08_arima_nonseasonal.py",
]

FLEET = [
    "src/09_fleet_preprocessing.py",
    "src/10_fleet_sarima.py",
]


def run_script(script_path):
    """Run a single pipeline script and display output summary."""
    print(f"\n{'#' * 70}")
    print(f"# Running {script_path}")
    print(f"{'#' * 70}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(line)
    if result.returncode != 0:
        err = result.stderr[-500:] if result.stderr else "none"
        print(f"STDERR: {err}")
    print(f"Exit code: {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run the analysis pipeline.")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages (seasonal + non-seasonal + fleet)")
    parser.add_argument("--nonseasonal", action="store_true",
                        help="Run non-seasonal ARIMA analysis only")
    parser.add_argument("--fleet", action="store_true",
                        help="Run fleet analysis only")
    args = parser.parse_args()

    scripts = []
    if args.nonseasonal:
        scripts = NONSEASONAL
    elif args.fleet:
        scripts = FLEET
    elif args.all:
        scripts = SEASONAL_CASE_STUDIES + NONSEASONAL + FLEET
    else:
        scripts = SEASONAL_CASE_STUDIES

    failed = []
    for s in scripts:
        rc = run_script(s)
        if rc != 0:
            failed.append(s)

    print(f"\n{'=' * 70}")
    print(f"Pipeline complete. {len(scripts) - len(failed)}/{len(scripts)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
