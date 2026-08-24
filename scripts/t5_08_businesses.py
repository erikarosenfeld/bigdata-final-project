#!/usr/bin/env python3
"""
T5 — Fetch major business locations (OpenStreetMap via Overpass API) and
count them per taxi zone via spatial join. Covers the "vicinity/locations
of major businesses" augmentation category from the project instructions.

Uses office nodes (corporate/company offices) and large retail (shop=mall,
shop=department_store) as a proxy for major business presence — areas with
high business density are expected to generate commuter and business-travel
taxi demand.

Same Overpass mirror and GET-based request pattern as the attractions
script (t5_04_attractions.py).
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

OVERPASS_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_businesses() -> pd.DataFrame:
    """Query Overpass for office nodes and large retail within NYC bbox."""
    south, west, north, east = NYC_BBOX
    query = f"""
    [out:json][timeout:120];
    (
      node["office"]({south},{west},{north},{east});
      node["shop"="mall"]({south},{west},{north},{east});
      node["shop"="department_store"]({south},{west},{north},{east});
    );
    out body;
    """
    response = requests.get(
        OVERPASS_URL,
        params={"data": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-project/1.0)"},
        timeout=180,
    )
    response.raise_for_status()
    elements = response.json()["elements"]

    rows = []
    for el in elements:
        tags = el.get("tags", {})
        rows.append({
            "name": tags.get("name", "Unnamed"),
            "kind": tags.get("office") or tags.get("shop") or "unknown",
            "latitude": el["lat"],
            "longitude": el["lon"],
        })

    return pd.DataFrame(rows)


def main():
    logger.info("Fetching business locations from Overpass API...")
    businesses = fetch_businesses()
    logger.info(f"Fetched {len(businesses)} business locations")

    if len(businesses) == 0:
        logger.error("No businesses returned — check the Overpass query/bbox")
        return

    raw_path = os.path.join(FEATURES_DIR, "businesses_nyc.csv")
    businesses.to_csv(raw_path, index=False)
    logger.info(f"Raw business list saved to {raw_path}")

    logger.info(f"Loading taxi zones from {TAXI_ZONES_PATH}")
    taxi_zones = gpd.read_file(TAXI_ZONES_PATH)

    geometry = [Point(xy) for xy in zip(businesses["longitude"], businesses["latitude"])]
    businesses_gdf = gpd.GeoDataFrame(businesses, geometry=geometry, crs="EPSG:4326")
    businesses_gdf = businesses_gdf.to_crs(taxi_zones.crs)

    logger.info("Running spatial join (businesses -> zones)...")
    joined = gpd.sjoin(businesses_gdf, taxi_zones, how="left", predicate="intersects")
    business_counts = (
        joined.groupby("LocationID").size().reset_index(name="nr_businesses")
    )

    all_zones = taxi_zones[["LocationID"]].copy()
    all_zones["LocationID"] = all_zones["LocationID"].astype(int)
    business_counts["LocationID"] = business_counts["LocationID"].astype(int)
    result = all_zones.merge(business_counts, on="LocationID", how="left")
    result["nr_businesses"] = result["nr_businesses"].fillna(0).astype(int)
    result = result.rename(columns={"LocationID": "PULocationID"})

    out_path = os.path.join(FEATURES_DIR, "businesses_per_zone.parquet")
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(result)} zones to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_businesses_summary.csv")
    summary = pd.DataFrame([{
        "total_businesses_fetched": len(businesses),
        "total_zones": len(result),
        "zones_with_businesses": (result["nr_businesses"] > 0).sum(),
        "max_businesses_in_one_zone": result["nr_businesses"].max(),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")

    # Breakdown by business kind, useful for the report
    kind_counts = businesses["kind"].value_counts().head(15)
    logger.info(f"\nTop business types fetched:\n{kind_counts.to_string()}")


if __name__ == "__main__":
    main()