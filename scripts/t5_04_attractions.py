#!/usr/bin/env python3
"""
T5 step 4 — Fetch NYC tourist attractions (OpenStreetMap via Overpass API,
tourism=attraction), then count attractions per taxi zone via spatial join.
"""

import logging
import os

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TAXI_ZONES_PATH = "/d/hpc/home/em51537/BIG_DATA/locations/taxi_zones.shp"

# NYC bounding box (south, west, north, east)
NYC_BBOX = (40.4774, -74.2591, 40.9176, -73.7004)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_attractions() -> pd.DataFrame:
    """Query Overpass API for tourism=attraction nodes within NYC bbox.
    Uses GET (not POST) — every other API call that's worked in this
    project so far has been GET, and POST to Overpass was consistently
    rejected regardless of headers/library used."""
    south, west, north, east = NYC_BBOX
    query = f"""
    [out:json][timeout:60];
    node["tourism"="attraction"]({south},{west},{north},{east});
    out body;
    """
    response = requests.get(
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        params={"data": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-project/1.0)"},
        timeout=90,
    )
    response.raise_for_status()
    elements = response.json()["elements"]

    rows = []
    for el in elements:
        name = el.get("tags", {}).get("name", "Unnamed")
        rows.append({"name": name, "latitude": el["lat"], "longitude": el["lon"]})

    return pd.DataFrame(rows)


def main():
    logger.info("Fetching attractions from Overpass API...")
    attractions = fetch_attractions()
    logger.info(f"Fetched {len(attractions)} attractions")

    if len(attractions) == 0:
        logger.error("No attractions returned — check the Overpass query/bbox")
        return

    # Save raw attraction list too (small, useful on its own for T3/report maps)
    raw_path = os.path.join(FEATURES_DIR, "attractions_nyc.csv")
    attractions.to_csv(raw_path, index=False)
    logger.info(f"Raw attraction list saved to {raw_path}")

    logger.info(f"Loading taxi zones from {TAXI_ZONES_PATH}")
    taxi_zones = gpd.read_file(TAXI_ZONES_PATH)

    geometry = [Point(xy) for xy in zip(attractions["longitude"], attractions["latitude"])]
    attractions_gdf = gpd.GeoDataFrame(attractions, geometry=geometry, crs="EPSG:4326")
    attractions_gdf = attractions_gdf.to_crs(taxi_zones.crs)

    logger.info("Running spatial join (attractions -> zones)...")
    joined = gpd.sjoin(attractions_gdf, taxi_zones, how="left", predicate="intersects")
    attraction_counts = (
        joined.groupby("LocationID").size().reset_index(name="nr_attractions")
    )

    all_zones = taxi_zones[["LocationID"]].copy()
    all_zones["LocationID"] = all_zones["LocationID"].astype(int)
    attraction_counts["LocationID"] = attraction_counts["LocationID"].astype(int)
    result = all_zones.merge(attraction_counts, on="LocationID", how="left")
    result["nr_attractions"] = result["nr_attractions"].fillna(0).astype(int)
    result = result.rename(columns={"LocationID": "PULocationID"})

    out_path = os.path.join(FEATURES_DIR, "attractions_per_zone.parquet")
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(result)} zones to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_attractions_summary.csv")
    summary = pd.DataFrame([{
        "total_attractions_fetched": len(attractions),
        "total_zones": len(result),
        "zones_with_attractions": (result["nr_attractions"] > 0).sum(),
        "max_attractions_in_one_zone": result["nr_attractions"].max(),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
