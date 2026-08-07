#!/usr/bin/env python3
"""
T4 — Format comparison on a moderate partition (2024 Green Taxi).

Uses the year=2024 Green Taxi partition already produced by T1
(repartitioned_data/green/year=2024/). Exports it to CSV, CSV.gz, HDF5,
and a DuckDB database file, then compares file size and pandas read speed
across all four formats.
"""

import glob
import logging
import os
import time

import duckdb
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")

INPUT_DIR = os.path.join(WORK_DIR, "repartitioned_data", "green", "year=2024")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t4")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "green_2024.csv")
CSV_GZ_PATH = os.path.join(OUT_DIR, "green_2024.csv.gz")
HDF5_PATH = os.path.join(OUT_DIR, "green_2024.h5")
DUCKDB_PATH = os.path.join(OUT_DIR, "green_2024.duckdb")
RESULTS_PATH = os.path.join(OUT_DIR, "t4_format_comparison.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_source_data() -> pd.DataFrame:
    """Load the 2024 Green Taxi partition (all part-files for that year)."""
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {INPUT_DIR} — did T1 finish for Green?"
        )
    logger.info(f"Loading {len(files)} part-file(s) from {INPUT_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def export_formats(df: pd.DataFrame):
    """Export the dataframe to each format, timing each export."""
    export_times = {}

    logger.info("Exporting to CSV...")
    t0 = time.time()
    df.to_csv(CSV_PATH, index=False)
    export_times["csv"] = time.time() - t0

    logger.info("Exporting to CSV.gz...")
    t0 = time.time()
    df.to_csv(CSV_GZ_PATH, index=False, compression="gzip")
    export_times["csv_gz"] = time.time() - t0

    logger.info("Exporting to HDF5...")
    t0 = time.time()
    df.to_hdf(HDF5_PATH, key="trips", mode="w", format="table")
    export_times["hdf5"] = time.time() - t0

    logger.info("Exporting to DuckDB...")
    t0 = time.time()
    if os.path.exists(DUCKDB_PATH):
        os.remove(DUCKDB_PATH)
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("CREATE TABLE trips AS SELECT * FROM df")
    con.close()
    export_times["duckdb"] = time.time() - t0

    return export_times


def time_read(label: str, read_fn, n_repeats: int = 3) -> float:
    """Time a read function, averaged over n_repeats runs."""
    times = []
    for _ in range(n_repeats):
        t0 = time.time()
        read_fn()
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    logger.info(f"  {label}: {avg:.3f}s avg over {n_repeats} runs")
    return avg


def benchmark_reads() -> dict:
    """Time reading each format back into a pandas dataframe."""
    logger.info("Benchmarking read speed (avg of 3 runs each)...")

    read_times = {}
    read_times["csv"] = time_read("CSV", lambda: pd.read_csv(CSV_PATH))
    read_times["csv_gz"] = time_read("CSV.gz", lambda: pd.read_csv(CSV_GZ_PATH, compression="gzip"))
    read_times["hdf5"] = time_read("HDF5", lambda: pd.read_hdf(HDF5_PATH, key="trips"))

    def read_duckdb():
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        result = con.execute("SELECT * FROM trips").fetchdf()
        con.close()
        return result

    read_times["duckdb"] = time_read("DuckDB", read_duckdb)

    return read_times


def get_file_sizes() -> dict:
    """Get file sizes in MB for each format."""
    return {
        "csv": os.path.getsize(CSV_PATH) / (1024 ** 2),
        "csv_gz": os.path.getsize(CSV_GZ_PATH) / (1024 ** 2),
        "hdf5": os.path.getsize(HDF5_PATH) / (1024 ** 2),
        "duckdb": os.path.getsize(DUCKDB_PATH) / (1024 ** 2),
    }


def main():
    logger.info("=" * 80)
    logger.info("T4 — Format comparison on 2024 Green Taxi partition")
    logger.info("=" * 80)

    df = load_source_data()

    # Also record the original parquet size for reference (not re-exported,
    # since it's already what T1 produced)
    parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    parquet_size_mb = sum(os.path.getsize(f) for f in parquet_files) / (1024 ** 2)

    export_times = export_formats(df)
    sizes = get_file_sizes()
    read_times = benchmark_reads()

    results = []
    for fmt in ["csv", "csv_gz", "hdf5", "duckdb"]:
        results.append({
            "format": fmt,
            "size_mb": round(sizes[fmt], 2),
            "export_time_s": round(export_times[fmt], 3),
            "read_time_s": round(read_times[fmt], 3),
        })
    results.append({
        "format": "parquet (original, from T1)",
        "size_mb": round(parquet_size_mb, 2),
        "export_time_s": None,
        "read_time_s": None,
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("\n" + results_df.to_string(index=False))
    logger.info(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
