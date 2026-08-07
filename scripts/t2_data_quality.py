#!/usr/bin/env python3
"""
T2 — Data quality analysis, all four datasets.

Reads T1's standardized, year-partitioned output (repartitioned_data/<dataset>/
year=YYYY/*.parquet) and flags known data quality issues per row:
  - year_out_of_range: pickup year outside [dataset min_year, current_year]
    (excluding legitimate 12/31 -> 1/1 delayed-entry edge cases)
  - same_pickup_dropoff: pickup timestamp == dropoff timestamp
  - dropoff_before_pickup: dropoff earlier than pickup
  - zero_distance: trip distance == 0 (Yellow/Green/FHVHV only — FHV has no distance field)
  - zero_passengers: passenger count == 0 (Yellow/Green only — FHV/FHVHV have no passenger field)
  - negative_fare: fare amount < 0 (Yellow/Green/FHVHV only)

For each dataset: aggregates flag counts by year, saves a summary table (CSV)
and a stacked bar chart (PNG) to outputs/t2/.
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")

INPUT_BASE = os.path.join(WORK_DIR, "repartitioned_data")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t2")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Which quality checks apply to which dataset — based on T1's unified schemas
# (FHV has no fare/distance/passenger fields; FHVHV has no passenger field)
DATASET_CHECKS = {
    "yellow": {"min_year": 2012, "has_distance": True, "has_passengers": True, "has_fare": True},
    "green": {"min_year": 2014, "has_distance": True, "has_passengers": True, "has_fare": True},
    "fhv": {"min_year": 2015, "has_distance": False, "has_passengers": False, "has_fare": False},
    "fhvhv": {"min_year": 2019, "has_distance": True, "has_passengers": False, "has_fare": True},
}


def get_year_partitions(dataset: str):
    """Find all year=YYYY partition folders for this dataset."""
    pattern = os.path.join(INPUT_BASE, dataset, "year=*")
    return sorted(glob.glob(pattern))


def flag_quality_issues(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add boolean flag columns for each applicable data quality issue."""
    config = DATASET_CHECKS[dataset]
    flags = pd.DataFrame(index=df.index)

    pickup = df["Pickup_DateTime"]
    dropoff = df["Dropoff_DateTime"]

    # Year out of range — excluding legitimate 12/31 -> 1/1 delayed-entry edge case
    year = pickup.dt.year
    is_edge_case = (
        ((pickup.dt.month == 12) & (pickup.dt.day == 31)) |
        ((pickup.dt.month == 1) & (pickup.dt.day == 1))
    )
    flags["year_out_of_range"] = (
        ((year < config["min_year"]) | (year > CURRENT_YEAR)) & ~is_edge_case
    )

    flags["same_pickup_dropoff"] = (pickup == dropoff)

    # Dropoff before the dataset's minimum valid year is a missing-data
    # placeholder (e.g. FHV 2015-2017 uses 1989-01-01 as a sentinel for
    # "dropoff not recorded"), not a genuine chronological reversal.
    dropoff_is_placeholder = dropoff.dt.year < config["min_year"]
    flags["missing_dropoff_placeholder"] = dropoff_is_placeholder
    flags["dropoff_before_pickup"] = (dropoff < pickup) & ~dropoff_is_placeholder

    if config["has_distance"]:
        flags["zero_distance"] = (df["Trip_Distance"] == 0)

    if config["has_passengers"]:
        flags["zero_passengers"] = (df["Passenger_Count"] == 0)

    if config["has_fare"]:
        flags["negative_fare"] = (df["Fare_Amount"] < 0)

    flags["year"] = year
    return flags


def process_one_partition(f: str, dataset: str) -> pd.DataFrame:
    """Read one part-file, flag issues, return per-year issue counts."""
    df = pd.read_parquet(f)
    flags = flag_quality_issues(df, dataset)
    issue_cols = [c for c in flags.columns if c != "year"]
    year_summary = flags.groupby("year")[issue_cols].sum()
    year_summary["total_rows"] = flags.groupby("year").size()
    return year_summary


def process_dataset(dataset: str):
    logger.info("=" * 80)
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info("=" * 80)

    year_dirs = get_year_partitions(dataset)
    if not year_dirs:
        logger.error(f"[{dataset}] No partitions found at {INPUT_BASE}/{dataset}/ — did T1 run?")
        return

    files = []
    for year_dir in year_dirs:
        files.extend(glob.glob(os.path.join(year_dir, "*.parquet")))
    logger.info(f"[{dataset}] Found {len(files)} part-files across {len(year_dirs)} year partitions")

    all_summaries = []
    for i, f in enumerate(files, 1):
        try:
            all_summaries.append(process_one_partition(f, dataset))
        except Exception as e:
            logger.error(f"[{dataset}] Failed on {f}: {e}")
        if i % 20 == 0 or i == len(files):
            logger.info(f"[{dataset}] Processed {i}/{len(files)} files")

    combined = pd.concat(all_summaries).groupby(level=0).sum()
    combined = combined.sort_index()

    # Save table
    table_path = os.path.join(OUT_DIR, f"t2_{dataset}_quality_by_year.csv")
    combined.to_csv(table_path)
    logger.info(f"[{dataset}] Table saved to {table_path}")

    # Save chart — stacked bar of issue counts by year (excluding total_rows)
    issue_cols = [c for c in combined.columns if c != "total_rows"]
    fig, ax = plt.subplots(figsize=(14, 7))
    combined[issue_cols].plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"{dataset.capitalize()} Taxi — Data Quality Issues by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of flagged rows")
    ax.legend(title="Issue type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"t2_{dataset}_quality_by_year.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"[{dataset}] Chart saved to {fig_path}")

    logger.info(f"\n[{dataset}] Summary:\n{combined.to_string()}")


def main():
    parser = argparse.ArgumentParser(description="T2 — Data quality analysis")
    parser.add_argument("--dataset", choices=list(DATASET_CHECKS.keys()), default=None,
                         help="Process a single dataset only (default: all four)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CHECKS.keys())
    for dataset in datasets:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
