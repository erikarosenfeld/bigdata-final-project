import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dask_ml.linear_model import PoissonRegression

import argparse
import glob
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd 

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")

INPUT_BASE = os.path.join(WORK_DIR, "repartitioned_data")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t7")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===============================================================================================
PICKUP_COL = "Pickup_DateTime"
TRIP_COUNT_COL = "trip_count"
FROM_TIME = {
    "yellow": pd.Timestamp("2012-01-01"),
    "green": pd.Timestamp("2014-01-01"),
    "fhv": pd.Timestamp("2015-01-01"),
    "fhvhv": pd.Timestamp("2019-02-01")}
TO_TIME = pd.Timestamp("2026-03-01")
FHVHV_OPERATOR_MAP = {
    "HV0002": "JUNO",
    "HV0003": "UBER",
    "HV0004": "VIA",
    "HV0005": "LYFT",}
RETAIN_COLS = ["hour", "operator", TRIP_COUNT_COL, "hour_of_day", "day_of_week", "month", "is_weekend", "year"]
FEATURE_COLS = ["hour", "hour_of_day", "day_of_week", "month", "is_weekend", "year"]
GROUP_COLS = ["hour", "operator"]

def get_year_partitions(dataset):
    pattern = os.path.join(INPUT_BASE, dataset, "year=*")
    return sorted(glob.glob(pattern))

def process_one_partition(f, dataset):
    df = dd.read_parquet(f)
    from_time_filter = FROM_TIME.get(dataset, pd.Timestamp("2019-02-01"))
    df = df[(df[PICKUP_COL]>=from_time_filter) & (df[PICKUP_COL]<TO_TIME)]
    dt = dd.to_datetime(df[PICKUP_COL])
    df["hour"] = dt.dt.floor("h")
    hourly_count = df.groupby("hour").size().rename(TRIP_COUNT_COL).reset_index().compute()
    hourly_count["hour"] = pd.to_datetime(hourly_count["hour"])
    hourly_count["operator"] = dataset
    return hourly_count

def process_one_partition_fhvhv(f, dataset):
    df = dd.read_parquet(f)
    from_time_filter = FROM_TIME.get(dataset, pd.Timestamp("2019-02-01"))
    df = df[(df[PICKUP_COL]>=from_time_filter) & (df[PICKUP_COL]<TO_TIME)]
    dt = dd.to_datetime(df[PICKUP_COL])
    df["hour"] = dt.dt.floor("h")
    
    all_operators_hourly_counts = []
    for operator_code, operator_name in FHVHV_OPERATOR_MAP.items():
        df_operator = df[df["HVFHS_License_Num"]==operator_code]
        hourly_count = df_operator.groupby("hour").size().rename(TRIP_COUNT_COL).reset_index().compute()
        hourly_count["hour"] = pd.to_datetime(hourly_count["hour"])
        hourly_count["operator"] = operator_name
        all_operators_hourly_counts.append(hourly_count)

    if not all_operators_hourly_counts:
        return pd.DataFrame(columns=["hour", "operator", TRIP_COUNT_COL])
    return pd.concat(all_operators_hourly_counts, ignore_index=True)

def process_dataset(dataset):
    logger.info("=" * 80)
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info("=" * 80)

    year_dirs = get_year_partitions(dataset)
    if not year_dirs:
        logger.error(f"[{dataset}] No partitions found at {INPUT_BASE}/{dataset}/")
        return

    files = []
    for year_dir in year_dirs:
        year_str = os.path.basename(year_dir).split("=")[-1]
        try:
            if int(year_str)>=int(FROM_TIME[dataset].year):
                files.extend(glob.glob(os.path.join(year_dir, "*.parquet")))
        except Exception as e:
            print(f"Error parsing year parquet file {year_str}")
    logger.info(f"[{dataset}] Found {len(files)} part-files across {len(year_dirs)} year partitions")

    process_partition_func = process_one_partition_fhvhv if dataset=="fhvhv" else process_one_partition
    part_counts = []
    for i, f in enumerate(files, 1):
        try:
            hourly_counts_per_partition = process_partition_func(f, dataset)
            part_counts.append(hourly_counts_per_partition)
        except Exception as e:
            logger.error(f"[{dataset}] Failed on {f}: {e}")
        if i % 20 == 0 or i == len(files):
            logger.info(f"[{dataset}] Processed {i}/{len(files)} files")
    if not part_counts:
        logger.error(f"[{dataset}] Missing counts!")

    # SUM COUNTS PER OPERATOR (IF OVERLAP IN FILES)
    combined_counts = (pd.concat(part_counts, ignore_index=True).groupby(GROUP_COLS, as_index=False)[TRIP_COUNT_COL].sum())
    combined_counts = combined_counts.sort_values(["hour", "operator"]).reset_index(drop=True)
    combined_counts["hour"] = pd.to_datetime(combined_counts["hour"])

    # EXTRA TIME FEATURES
    combined_counts["hour_of_day"] = combined_counts["hour"].dt.hour
    combined_counts["day_of_week"] = combined_counts["hour"].dt.dayofweek
    combined_counts["month"] = combined_counts["hour"].dt.month
    combined_counts["year"] = combined_counts["hour"].dt.year.astype(int)
    combined_counts["is_weekend"] = (combined_counts["day_of_week"] >= 5).astype(int)

    #  SAVE TO CSV 
    table_path = os.path.join(OUT_DIR, f"t7_{dataset}_trips_by_hour")
    combined_dd = dd.from_pandas(combined_counts, npartitions=16)
    combined_dd.to_parquet(table_path, write_index=False)
    logger.info(f"[{dataset}] Table saved to {table_path}")

def main():
    # CREATE COUNTS PER HOUR 
    datasets = ["yellow", "green", "fhv", "fhvhv"]
    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()