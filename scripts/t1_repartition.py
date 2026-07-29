#!/usr/bin/env python3
"""
T1 — Repartition all four datasets by year with unified schemas.

For each dataset (Yellow, Green, FHV, FHVHV):
  1. Glob all parquet files (historical, read-only folder + newly downloaded T0 files)
  2. Standardize schema (rename columns to a unified name, add missing as NaN)
  3. Add a 'year' column derived from pickup datetime
  4. Repartition to ~200MB files with ~2M row row-groups, partitioned by year

Each dataset gets its own unified schema

A sanity check runs before processing: it inspects real column names found in
your files and flags any that aren't in COLUMN_MAP, so mapping gaps are
caught immediately.

Usage:
  python3 t1_repartition.py                  # process all 4 datasets
  python3 t1_repartition.py --dataset yellow  # process one dataset only
  python3 t1_repartition.py --check-only      # just run the schema sanity check, no processing
"""

import argparse
import glob
import logging
import os

import dask
import dask.dataframe as dd
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# Data lives in two places: historical (read-only) + newly downloaded (T0)
HISTORICAL_DIR = "/d/hpc/projects/FRI/bigdata/data/Taxi"
WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
NEW_DATA_DIR = os.path.join(WORK_DIR, "taxi_data_raw")
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")

# Repartitioned output — large, kept outside the repo
OUT_DATA_DIR = os.path.join(WORK_DIR, "repartitioned_data")
# Small logs/reports — inside the repo
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t1")

os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Per-dataset schema definitions
# ============================================================================

# --- YELLOW --- (mapping reused from assingment 3)
YELLOW_COLUMN_MAP = {
    'vendor_name': 'VendorID', 'vendor_id': 'VendorID', 'VendorID': 'VendorID',
    'Trip_Pickup_DateTime': 'Pickup_DateTime', 'pickup_datetime': 'Pickup_DateTime',
    'tpep_pickup_datetime': 'Pickup_DateTime',
    'Trip_Dropoff_DateTime': 'Dropoff_DateTime', 'dropoff_datetime': 'Dropoff_DateTime',
    'tpep_dropoff_datetime': 'Dropoff_DateTime',
    'Passenger_Count': 'Passenger_Count', 'passenger_count': 'Passenger_Count',
    'Trip_Distance': 'Trip_Distance', 'trip_distance': 'Trip_Distance',
    'Rate_Code': 'Rate_Code', 'rate_code': 'Rate_Code', 'RatecodeID': 'Rate_Code',
    'store_and_forward': 'Store_And_Fwd_Flag', 'store_and_fwd_flag': 'Store_And_Fwd_Flag',
    'PULocationID': 'PULocationID', 'DOLocationID': 'DOLocationID',
    'Start_Lon': 'Pickup_Lon', 'Start_Lat': 'Pickup_Lat',
    'End_Lon': 'Dropoff_Lon', 'End_Lat': 'Dropoff_Lat',
    'pickup_longitude': 'Pickup_Lon', 'pickup_latitude': 'Pickup_Lat',
    'dropoff_longitude': 'Dropoff_Lon', 'dropoff_latitude': 'Dropoff_Lat',
    'Payment_Type': 'Payment_Type', 'payment_type': 'Payment_Type',
    'Fare_Amt': 'Fare_Amount', 'fare_amount': 'Fare_Amount',
    'surcharge': 'Surcharge', 'extra': 'Surcharge',
    'mta_tax': 'MTA_Tax',
    'Tip_Amt': 'Tip_Amount', 'tip_amount': 'Tip_Amount',
    'Tolls_Amt': 'Tolls_Amount', 'tolls_amount': 'Tolls_Amount',
    'Total_Amt': 'Total_Amount', 'total_amount': 'Total_Amount',
    'improvement_surcharge': 'Improvement_Surcharge',
    'congestion_surcharge': 'Congestion_Surcharge',
    'airport_fee': 'Airport_Fee', 'Airport_fee': 'Airport_Fee',
    'cbd_congestion_fee': 'CBD_Congestion_Fee',
    # Pandas artifact from older files saved with a default index column — intentionally dropped
    '__index_level_0__': '__index_level_0__',
}
YELLOW_UNIFIED_COLUMNS = [
    'VendorID', 'Pickup_DateTime', 'Dropoff_DateTime', 'Passenger_Count',
    'Trip_Distance', 'Rate_Code', 'Store_And_Fwd_Flag', 'PULocationID',
    'DOLocationID', 'Pickup_Lon', 'Pickup_Lat', 'Dropoff_Lon', 'Dropoff_Lat',
    'Payment_Type', 'Fare_Amount', 'Surcharge', 'MTA_Tax', 'Tip_Amount',
    'Tolls_Amount', 'Total_Amount', 'Improvement_Surcharge',
    'Congestion_Surcharge', 'Airport_Fee', 'CBD_Congestion_Fee',
]

# --- GREEN --- (same fare structure as Yellow, plus trip_type/ehail_fee)
GREEN_COLUMN_MAP = {
    'VendorID': 'VendorID',
    'lpep_pickup_datetime': 'Pickup_DateTime',
    'lpep_dropoff_datetime': 'Dropoff_DateTime',
    'Store_and_fwd_flag': 'Store_And_Fwd_Flag', 'store_and_fwd_flag': 'Store_And_Fwd_Flag',
    'RatecodeID': 'Rate_Code',
    'PULocationID': 'PULocationID', 'DOLocationID': 'DOLocationID',
    'Pickup_longitude': 'Pickup_Lon', 'Pickup_latitude': 'Pickup_Lat',
    'Dropoff_longitude': 'Dropoff_Lon', 'Dropoff_latitude': 'Dropoff_Lat',
    'Passenger_count': 'Passenger_Count', 'passenger_count': 'Passenger_Count',
    'Trip_distance': 'Trip_Distance', 'trip_distance': 'Trip_Distance',
    'Fare_amount': 'Fare_Amount', 'fare_amount': 'Fare_Amount',
    'Extra': 'Surcharge', 'extra': 'Surcharge',
    'MTA_tax': 'MTA_Tax', 'mta_tax': 'MTA_Tax',
    'Tip_amount': 'Tip_Amount', 'tip_amount': 'Tip_Amount',
    'Tolls_amount': 'Tolls_Amount', 'tolls_amount': 'Tolls_Amount',
    'Ehail_fee': 'Ehail_Fee', 'ehail_fee': 'Ehail_Fee',
    'improvement_surcharge': 'Improvement_Surcharge',
    'Total_amount': 'Total_Amount', 'total_amount': 'Total_Amount',
    'Payment_type': 'Payment_Type', 'payment_type': 'Payment_Type',
    'Trip_type': 'Trip_Type', 'Trip_type ': 'Trip_Type', 'trip_type': 'Trip_Type',
    'congestion_surcharge': 'Congestion_Surcharge',
    'cbd_congestion_fee': 'CBD_Congestion_Fee',
}
GREEN_UNIFIED_COLUMNS = [
    'VendorID', 'Pickup_DateTime', 'Dropoff_DateTime', 'Store_And_Fwd_Flag',
    'Rate_Code', 'PULocationID', 'DOLocationID', 'Pickup_Lon', 'Pickup_Lat',
    'Dropoff_Lon', 'Dropoff_Lat', 'Passenger_Count', 'Trip_Distance',
    'Fare_Amount', 'Surcharge', 'MTA_Tax', 'Tip_Amount', 'Tolls_Amount',
    'Ehail_Fee', 'Improvement_Surcharge', 'Total_Amount', 'Payment_Type',
    'Trip_Type', 'Congestion_Surcharge', 'CBD_Congestion_Fee',
]

# --- FHV --- (minimal: no fare data at all)
FHV_COLUMN_MAP = {
    'Dispatching_base_num': 'Dispatching_Base_Num', 'dispatching_base_num': 'Dispatching_Base_Num',
    'Pickup_DateTime': 'Pickup_DateTime', 'pickup_datetime': 'Pickup_DateTime',
    'DropOff_datetime': 'Dropoff_DateTime', 'dropOff_datetime': 'Dropoff_DateTime',
    'dropoff_datetime': 'Dropoff_DateTime',
    'PUlocationID': 'PULocationID', 'PULocationID': 'PULocationID',
    'DOlocationID': 'DOLocationID', 'DOLocationID': 'DOLocationID',
    'SR_Flag': 'SR_Flag', 'sr_flag': 'SR_Flag',
    'Affiliated_base_number': 'Affiliated_Base_Num', 'affiliated_base_number': 'Affiliated_Base_Num',
}
FHV_UNIFIED_COLUMNS = [
    'Dispatching_Base_Num', 'Pickup_DateTime', 'Dropoff_DateTime',
    'PULocationID', 'DOLocationID', 'SR_Flag', 'Affiliated_Base_Num',
]

# --- FHVHV --- (largest schema: Uber/Lyft-style high-volume FHV)
FHVHV_COLUMN_MAP = {
    'hvfhs_license_num': 'HVFHS_License_Num',
    'dispatching_base_num': 'Dispatching_Base_Num',
    'originating_base_num': 'Originating_Base_Num',
    'request_datetime': 'Request_DateTime',
    'on_scene_datetime': 'On_Scene_DateTime',
    'pickup_datetime': 'Pickup_DateTime',
    'dropoff_datetime': 'Dropoff_DateTime',
    'PULocationID': 'PULocationID', 'DOLocationID': 'DOLocationID',
    'trip_miles': 'Trip_Distance', 'trip_time': 'Trip_Time',
    'base_passenger_fare': 'Fare_Amount',
    'tolls': 'Tolls_Amount', 'bcf': 'BCF', 'sales_tax': 'Sales_Tax',
    'congestion_surcharge': 'Congestion_Surcharge',
    'airport_fee': 'Airport_Fee',
    'tips': 'Tip_Amount', 'driver_pay': 'Driver_Pay',
    'shared_request_flag': 'Shared_Request_Flag',
    'shared_match_flag': 'Shared_Match_Flag',
    'access_a_ride_flag': 'Access_A_Ride_Flag',
    'wav_request_flag': 'WAV_Request_Flag',
    'wav_match_flag': 'WAV_Match_Flag',
    'cbd_congestion_fee': 'CBD_Congestion_Fee',
}
FHVHV_UNIFIED_COLUMNS = [
    'HVFHS_License_Num', 'Dispatching_Base_Num', 'Originating_Base_Num',
    'Request_DateTime', 'On_Scene_DateTime', 'Pickup_DateTime', 'Dropoff_DateTime',
    'PULocationID', 'DOLocationID', 'Trip_Distance', 'Trip_Time', 'Fare_Amount',
    'Tolls_Amount', 'BCF', 'Sales_Tax', 'Congestion_Surcharge', 'Airport_Fee',
    'Tip_Amount', 'Driver_Pay', 'Shared_Request_Flag', 'Shared_Match_Flag',
    'Access_A_Ride_Flag', 'WAV_Request_Flag', 'WAV_Match_Flag', 'CBD_Congestion_Fee',
]

DATASET_CONFIG = {
    "yellow": {"column_map": YELLOW_COLUMN_MAP, "unified_columns": YELLOW_UNIFIED_COLUMNS,
               "pickup_col": "Pickup_DateTime", "min_year_month": (2012, 1)},
    "green": {"column_map": GREEN_COLUMN_MAP, "unified_columns": GREEN_UNIFIED_COLUMNS,
              "pickup_col": "Pickup_DateTime", "min_year_month": (2014, 1)},
    "fhv": {"column_map": FHV_COLUMN_MAP, "unified_columns": FHV_UNIFIED_COLUMNS,
            "pickup_col": "Pickup_DateTime", "min_year_month": (2015, 1)},
    "fhvhv": {"column_map": FHVHV_COLUMN_MAP, "unified_columns": FHVHV_UNIFIED_COLUMNS,
              "pickup_col": "Pickup_DateTime", "min_year_month": (2019, 2)},
}

NUMERIC_COLUMNS_HINT = [
    "Passenger_Count", "Trip_Distance", "Pickup_Lon", "Pickup_Lat",
    "Dropoff_Lon", "Dropoff_Lat", "Fare_Amount", "Surcharge", "MTA_Tax",
    "Tip_Amount", "Tolls_Amount", "Total_Amount", "Improvement_Surcharge",
    "Congestion_Surcharge", "Airport_Fee", "CBD_Congestion_Fee", "Ehail_Fee",
    "PULocationID", "DOLocationID", "Trip_Time", "BCF", "Sales_Tax", "Driver_Pay",
]
DATETIME_COLUMNS_HINT = [
    "Pickup_DateTime", "Dropoff_DateTime", "Request_DateTime", "On_Scene_DateTime",
]


def get_files_for_dataset(dataset: str):
    """
    Glob files from both the historical (read-only) and new (T0) locations,
    then filter to the minimum year-month required by the instructions.
    """
    import re

    historical = sorted(glob.glob(os.path.join(HISTORICAL_DIR, f"{dataset}_tripdata_*.parquet")))
    new = sorted(glob.glob(os.path.join(NEW_DATA_DIR, f"{dataset}_tripdata_*.parquet")))
    # New files may overlap with historical if re-downloaded; de-dupe by basename
    seen = {os.path.basename(f) for f in historical}
    new_unique = [f for f in new if os.path.basename(f) not in seen]
    all_files = historical + new_unique

    min_year, min_month = DATASET_CONFIG[dataset]["min_year_month"]
    pattern = re.compile(rf"{dataset}_tripdata_(\d{{4}})-(\d{{2}})\.parquet$")

    filtered = []
    excluded_count = 0
    for f in all_files:
        m = pattern.search(os.path.basename(f))
        if not m:
            logger.warning(f"[{dataset}] Filename doesn't match expected pattern, skipping: {f}")
            continue
        year, month = int(m.group(1)), int(m.group(2))
        if (year, month) >= (min_year, min_month):
            filtered.append(f)
        else:
            excluded_count += 1

    if excluded_count:
        logger.info(f"[{dataset}] Excluded {excluded_count} file(s) before "
                     f"{min_year}-{min_month:02d} per project instructions")

    return filtered


def schema_sanity_check(dataset: str, files: list) -> bool:
    """
    Inspect real columns across all files for this dataset and flag any
    column not covered by COLUMN_MAP. Returns True if all columns are mapped.
    """
    column_map = DATASET_CONFIG[dataset]["column_map"]
    all_ok = True
    unmapped = {}

    for f in files:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(f).schema_arrow.names
        for col in cols:
            if col not in column_map:
                unmapped.setdefault(col, []).append(os.path.basename(f))
                all_ok = False

    if unmapped:
        logger.warning(f"[{dataset}] UNMAPPED COLUMNS FOUND — these will be silently dropped:")
        for col, fs in unmapped.items():
            logger.warning(f"  '{col}' seen in {len(fs)} file(s), e.g. {fs[0]}")
        logger.warning(f"[{dataset}] Add these to {dataset.upper()}_COLUMN_MAP before trusting the output.")
    else:
        logger.info(f"[{dataset}] Schema sanity check passed — all columns mapped.")

    return all_ok


def standardize_partition(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Standardize a single pandas dataframe to this dataset's unified schema."""
    config = DATASET_CONFIG[dataset]
    column_map = config["column_map"]
    unified_columns = config["unified_columns"]

    df = df.rename(columns=column_map)

    for col in unified_columns:
        if col not in df.columns:
            df[col] = None
    df = df[unified_columns]

    for col in unified_columns:
        if col in DATETIME_COLUMNS_HINT:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif col in NUMERIC_COLUMNS_HINT:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)

    return df


def process_dataset(dataset: str, check_only: bool = False):
    logger.info("=" * 80)
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info("=" * 80)

    files = get_files_for_dataset(dataset)
    min_year, min_month = DATASET_CONFIG[dataset]["min_year_month"]
    logger.info(f"[{dataset}] Found {len(files)} files "
                f"(filtered to {min_year}-{min_month:02d} onwards per instructions)")

    if not files:
        logger.error(f"[{dataset}] No files found, skipping")
        return

    schema_ok = schema_sanity_check(dataset, files)

    if check_only:
        return

    if not schema_ok:
        logger.warning(f"[{dataset}] Proceeding despite unmapped columns — "
                        f"those columns will be dropped. Fix COLUMN_MAP and rerun for correctness.")

    config = DATASET_CONFIG[dataset]
    pickup_col = config["pickup_col"]

    @dask.delayed
    def read_and_standardize(f):
        df = pd.read_parquet(f)
        return standardize_partition(df, dataset)

    delayed_dfs = [read_and_standardize(f) for f in files]
    meta = standardize_partition(pd.read_parquet(files[0]), dataset).iloc[0:0]
    ddf = dd.from_delayed(delayed_dfs, meta=meta)

    ddf["year"] = ddf[pickup_col].dt.year

    out_path = os.path.join(OUT_DATA_DIR, dataset)
    logger.info(f"[{dataset}] Repartitioning and saving to {out_path} ...")
    ddf.repartition(partition_size="200MB").to_parquet(
        out_path,
        partition_on=["year"],
        row_group_size=2_000_000,
        engine="pyarrow",
        write_index=False,
    )
    logger.info(f"[{dataset}] Done.")

    # Save a small per-year row count report to the repo's outputs/
    year_counts = ddf.groupby("year").size().compute()
    report_path = os.path.join(OUT_DIR, f"t1_{dataset}_year_counts.csv")
    year_counts.to_csv(report_path, header=["row_count"])
    logger.info(f"[{dataset}] Year counts saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="T1 — Repartition all datasets by year")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default=None,
                         help="Process a single dataset only (default: all four)")
    parser.add_argument("--check-only", action="store_true",
                         help="Only run the schema sanity check, no processing")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIG.keys())

    for dataset in datasets:
        process_dataset(dataset, check_only=args.check_only)


if __name__ == "__main__":
    main()