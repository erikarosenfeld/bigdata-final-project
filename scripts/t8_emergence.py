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
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t8")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PICKUP_COL = "Pickup_DateTime"
TRIP_COUNT_COL = "trip_count"
FROM_TIME = pd.Timestamp("2012-01-01")
TO_TIME = pd.Timestamp("2026-02-01")
FHVHV_OPERATOR_MAP = {
    "HV0002": "JUNO",
    "HV0003": "UBER",
    "HV0004": "VIA",
    "HV0005": "LYFT",
}

def get_year_partitions(dataset):
    pattern = os.path.join(INPUT_BASE, dataset, "year=*")
    return sorted(glob.glob(pattern))
    

def process_one_partition(f, dataset):
    # READ PARQUET FILE
    df = dd.read_parquet(f)
    df = df[(df[PICKUP_COL]>=FROM_TIME) & (df[PICKUP_COL]<=TO_TIME)]

    # INFER MONTH 
    dt = dd.to_datetime(df[PICKUP_COL])
    df["month"] = dt.dt.year.astype(str) + "-" + dt.dt.month.astype(str).str.zfill(2)

    # COUNTS 
    monthly_count = df.groupby("month").size().rename(TRIP_COUNT_COL).reset_index().compute()
    monthly_count["month"] = pd.to_datetime(monthly_count["month"])

    # EACH ROW HAS THE SAME OPERATOR (yellow, green)
    monthly_count["operator"] = dataset
    return monthly_count

def process_one_partition_fhvhv(f, dataset):
    df = dd.read_parquet(f)
    df = df[(df[PICKUP_COL]>=FROM_TIME) & (df[PICKUP_COL]<=TO_TIME)]
    dt = dd.to_datetime(df[PICKUP_COL])
    df["month"] = dt.dt.year.astype(str) + "-" + dt.dt.month.astype(str).str.zfill(2)

    all_operators_monthly_counts = []
    for operator_code, operator_name in FHVHV_OPERATOR_MAP.items():
        df_operator = df[df["HVFHS_License_Num"]==operator_code]
        monthly_count = df_operator.groupby("month").size().rename(TRIP_COUNT_COL).reset_index().compute()
        monthly_count["month"] = pd.to_datetime(monthly_count["month"])
        monthly_count["operator"] = operator_name
        all_operators_monthly_counts.append(monthly_count)

    if not all_operators_monthly_counts:
        return pd.DataFrame(columns=["month", TRIP_COUNT_COL, "operator"])
    return pd.concat(all_operators_monthly_counts, ignore_index=True)


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
            if int(year_str)>=2012:
                files.extend(glob.glob(os.path.join(year_dir, "*.parquet")))
        except Exception as e:
            print(f"Error parsing year parquet file {year_str}")
    logger.info(f"[{dataset}] Found {len(files)} part-files across {len(year_dirs)} year partitions")

    process_partition_func = process_one_partition_fhvhv if dataset=="fhvhv" else process_one_partition
    part_counts = []
    for i, f in enumerate(files, 1):
        try:
            monthly_counts_per_partition = process_partition_func(f, dataset)
            part_counts.append(monthly_counts_per_partition)
        except Exception as e:
            logger.error(f"[{dataset}] Failed on {f}: {e}")
    if not part_counts:
        logger.error(f"[{dataset}] Missing monthly counts!")

    combined_counts = (pd.concat(part_counts, ignore_index=True).groupby(["month", "operator"], as_index=False)[TRIP_COUNT_COL].sum())
    combined_counts = combined_counts.sort_values(["month", "operator"]).reset_index(drop=True)
    table_path = os.path.join(OUT_DIR, f"t8_{dataset}_trips_by_month.csv")
    combined_counts.to_csv(table_path)
    logger.info(f"[{dataset}] Table saved to {table_path}")


def main():
    datasets = ["yellow", "green", "fhv", "fhvhv"]
    for dataset in datasets:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
