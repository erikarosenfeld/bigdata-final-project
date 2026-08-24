#!/usr/bin/env python3
"""
T5 step 3 — Count schools per taxi zone via spatial join onto the demand target by PULocationID only.
"""

import logging
import os

import geopandas as gpd
import pandas as pd

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TAXI_ZONES_PATH = "/d/hpc/home/em51537/BIG_DATA/locations/taxi_zones.shp"
SCHOOLS_PATH = "/d/hpc/home/em51537/BIG_DATA/schools/SchoolPoints_APS_2024_08_28.shp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Loading taxi zones from {TAXI_ZONES_PATH}")
    taxi_zones = gpd.read_file(TAXI_ZONES_PATH)
    logger.info(f"Loading schools from {SCHOOLS_PATH}")
    schools = gpd.read_file(SCHOOLS_PATH)

    logger.info(f"Taxi zones CRS: {taxi_zones.crs}")
    logger.info(f"Schools CRS: {schools.crs}")
    schools = schools.to_crs(taxi_zones.crs)

    logger.info("Running spatial join (schools -> zones)...")
    joined = gpd.sjoin(schools, taxi_zones, how="left", predicate="intersects")
    school_counts = (
        joined.groupby("LocationID").size().reset_index(name="nr_schools")
    )

    # Ensure every zone appears, even ones with zero schools
    all_zones = taxi_zones[["LocationID"]].copy()
    all_zones["LocationID"] = all_zones["LocationID"].astype(int)
    school_counts["LocationID"] = school_counts["LocationID"].astype(int)
    result = all_zones.merge(school_counts, on="LocationID", how="left")
    result["nr_schools"] = result["nr_schools"].fillna(0).astype(int)
    result = result.rename(columns={"LocationID": "PULocationID"})

    out_path = os.path.join(FEATURES_DIR, "schools_per_zone.parquet")
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(result)} zones to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_schools_summary.csv")
    summary = pd.DataFrame([{
        "total_zones": len(result),
        "zones_with_schools": (result["nr_schools"] > 0).sum(),
        "total_schools_matched": result["nr_schools"].sum(),
        "max_schools_in_one_zone": result["nr_schools"].max(),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
