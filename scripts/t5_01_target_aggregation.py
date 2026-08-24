#!/usr/bin/env python3
"""
T5 step 1 — Aggregate trip demand to (PULocationID, date, hour), across all
four datasets. This is the base table every other T5 source (weather,
schools, attractions, events, MTA alerts) joins onto, and what T7 will
train on.

Per-file, low-memory pattern (same as T1/T2): read one part-file at a time,
group locally, accumulate partial results, then do a final re-aggregation
across all files since the same (zone, date, hour) can appear in multiple
source files.

Only rows within each dataset's valid year range are counted (matches T1's
min_year cutoffs, plus a sane upper bound) — excludes the tiny amount of
corrupted-timestamp rows T2 already found, without needing every T2 flag
column re-applied here.
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import pandas as pd

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")

INPUT_BASE = os.path.join(WORK_DIR, "repartitioned_data")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year

# Same min-year cutoffs as T1; max year is current+1 to allow for late-year data
DATASET_YEAR_RANGE = {
    "yellow": (2012, CURRENT_YEAR + 1),
    "green": (2014, CURRENT_YEAR + 1),
    "fhv": (2015, CURRENT_YEAR + 1),
    "fhvhv": (2019, CURRENT_YEAR + 1),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_valid_year_files(dataset: str):
    """Only glob year=YYYY folders within this dataset's valid range."""
    min_year, max_year = DATASET_YEAR_RANGE[dataset]
    files = []
    for year in range(min_year, max_year + 1):
        year_dir = os.path.join(INPUT_BASE, dataset, f"year={year}")
        files.extend(glob.glob(os.path.join(year_dir, "*.parquet")))
    return files


def aggregate_one_file(f: str) -> pd.DataFrame:
    """Read one part-file, return counts grouped by (PULocationID, date, hour)."""
    df = pd.read_parquet(f, columns=["PULocationID", "Pickup_DateTime"])
    df = df.dropna(subset=["PULocationID", "Pickup_DateTime"])
    df["date"] = df["Pickup_DateTime"].dt.date
    df["hour"] = df["Pickup_DateTime"].dt.hour
    df["PULocationID"] = df["PULocationID"].astype(int)

    counts = (
        df.groupby(["PULocationID", "date", "hour"])
        .size()
        .reset_index(name="trip_count")
    )
    return counts


def process_dataset(dataset: str):
    logger.info("=" * 80)
    logger.info(f"Aggregating dataset: {dataset.upper()}")
    logger.info("=" * 80)

    files = get_valid_year_files(dataset)
    logger.info(f"[{dataset}] Found {len(files)} part-files")

    if not files:
        logger.error(f"[{dataset}] No files found — did T1 run for this dataset?")
        return None

    partials = []
    for i, f in enumerate(files, 1):
        try:
            partials.append(aggregate_one_file(f))
        except Exception as e:
            logger.error(f"[{dataset}] Failed on {f}: {e}")
        if i % 20 == 0 or i == len(files):
            logger.info(f"[{dataset}] Processed {i}/{len(files)} files")

    logger.info(f"[{dataset}] Consolidating partial aggregates...")
    combined = pd.concat(partials, ignore_index=True)
    combined = (
        combined.groupby(["PULocationID", "date", "hour"])["trip_count"]
        .sum()
        .reset_index()
    )
    combined["dataset"] = dataset

    out_path = os.path.join(FEATURES_DIR, f"demand_{dataset}.parquet")
    combined.to_parquet(out_path, index=False)
    logger.info(f"[{dataset}] Saved {len(combined):,} (zone, date, hour) rows to {out_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="T5 step 1 — demand aggregation")
    parser.add_argument("--dataset", choices=list(DATASET_YEAR_RANGE.keys()), default=None,
                         help="Process a single dataset only (default: all four)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_YEAR_RANGE.keys())

    all_results = []
    for dataset in datasets:
        result = process_dataset(dataset)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 1:
        logger.info("=" * 80)
        logger.info("Building combined (all datasets) demand table...")
        combined_all = pd.concat(all_results, ignore_index=True)

        # Total across all 4 datasets, plus keep per-dataset breakdown too
        total = (
            combined_all.groupby(["PULocationID", "date", "hour"])["trip_count"]
            .sum()
            .reset_index()
            .rename(columns={"trip_count": "total_trip_count"})
        )
        per_dataset = combined_all.pivot_table(
            index=["PULocationID", "date", "hour"],
            columns="dataset",
            values="trip_count",
            fill_value=0,
        ).reset_index()
        per_dataset.columns = [
            f"{c}_trip_count" if c in DATASET_YEAR_RANGE else c
            for c in per_dataset.columns
        ]

        final = total.merge(per_dataset, on=["PULocationID", "date", "hour"], how="left")

        out_path = os.path.join(FEATURES_DIR, "demand_combined.parquet")
        final.to_parquet(out_path, index=False)
        logger.info(f"Combined table: {len(final):,} rows saved to {out_path}")

        # Small summary for the repo
        summary_path = os.path.join(OUT_DIR, "t5_target_summary.csv")
        summary = pd.DataFrame([{
            "total_rows": len(final),
            "unique_zones": final["PULocationID"].nunique(),
            "date_min": str(final["date"].min()),
            "date_max": str(final["date"].max()),
            "total_trips": int(final["total_trip_count"].sum()),
        }])
        summary.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
