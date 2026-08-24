#!/usr/bin/env python3
"""
T5 — Merge all feature files into one final augmented dataset, ready for
T7 to train on directly.

Joins onto the demand target (zone, date, hour):
  + weather_nyc                      on (date, hour)
  + schools_per_zone                 on PULocationID
  + attractions_per_zone             on PULocationID
  + businesses_per_zone              on PULocationID
  + nyc_events_by_hour               on (date, hour)
  + mta_alerts_by_hour               on (date, hour)
  + flights_by_hour                  on (date, hour)
  + taxi zone lookup (Borough, Zone, service_zone)  on PULocationID

Zone-level attributes (schools, attractions, businesses, Borough, Zone,
service_zone) collapse to constants under T7's city-wide aggregation and
are therefore not used as model features; they are retained for
descriptive analysis and interpretability.
"""

import logging
import os

import pandas as pd

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(OUT_DIR, exist_ok=True)

LOOKUP_PATH = "/d/hpc/home/em51537/BIG_DATA/locations/taxi_zone_lookup.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_feature(filename: str) -> pd.DataFrame:
    path = os.path.join(FEATURES_DIR, filename)
    df = pd.read_parquet(path)
    logger.info(f"Loaded {filename}: {len(df):,} rows")
    return df


def main():
    logger.info("Loading demand_combined.parquet (base table)...")
    merged = load_feature("demand_combined.parquet")

    weather = load_feature("weather_nyc.parquet")
    schools = load_feature("schools_per_zone.parquet")
    attractions = load_feature("attractions_per_zone.parquet")
    businesses = load_feature("businesses_per_zone.parquet")
    events = load_feature("nyc_events_by_hour.parquet")
    mta_alerts = load_feature("mta_alerts_by_hour.parquet")
    flights = load_feature("flights_by_hour.parquet")

    logger.info("Joining weather on (date, hour)...")
    merged = merged.merge(weather, on=["date", "hour"], how="left")

    logger.info("Joining schools on PULocationID...")
    merged = merged.merge(schools, on="PULocationID", how="left")

    logger.info("Joining attractions on PULocationID...")
    merged = merged.merge(attractions, on="PULocationID", how="left")

    logger.info("Joining businesses on PULocationID...")
    merged = merged.merge(businesses, on="PULocationID", how="left")

    logger.info("Joining events on (date, hour) [city-wide]...")
    merged = merged.merge(events, on=["date", "hour"], how="left")

    logger.info("Joining MTA alerts on (date, hour) [city-wide]...")
    merged = merged.merge(mta_alerts, on=["date", "hour"], how="left")

    logger.info("Joining flights on (date, hour) [city-wide]...")
    merged = merged.merge(flights, on=["date", "hour"], how="left")

    logger.info(f"Joining taxi zone lookup from {LOOKUP_PATH}...")
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup = lookup.rename(columns={"LocationID": "PULocationID"})
    lookup["PULocationID"] = lookup["PULocationID"].astype(int)
    merged = merged.merge(
        lookup[["PULocationID", "Borough", "Zone", "service_zone"]],
        on="PULocationID", how="left"
    )

    # Count features: no match means zero occurrences (or, for
    # events/MTA/flights, outside that source's coverage window)
    count_cols = [
        "nr_schools", "nr_attractions", "nr_businesses",
        "big_venue_event_count", "street_closure_event_count",
        "subway_disruption_count", "arriving_flights_count",
        "departing_flights_pickup_proxy_count",
    ]
    for col in count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    out_path = os.path.join(FEATURES_DIR, "demand_augmented.parquet")
    merged.to_parquet(out_path, index=False)
    logger.info(f"Final augmented dataset saved: {len(merged):,} rows, "
                f"{len(merged.columns)} columns, to {out_path}")
    logger.info(f"Columns: {list(merged.columns)}")

    # Join coverage diagnostics — expected gaps for limited-range sources
    missing_weather = merged["temperature_c"].isna().sum()
    if missing_weather > 0:
        logger.warning(f"{missing_weather:,} rows ({100*missing_weather/len(merged):.2f}%) "
                        f"have no matching weather data")

    missing_zone_info = merged["Borough"].isna().sum()
    if missing_zone_info > 0:
        logger.warning(f"{missing_zone_info:,} rows ({100*missing_zone_info/len(merged):.2f}%) "
                        f"have no matching zone lookup entry")

    for col, label in [
        ("big_venue_event_count", "events"),
        ("subway_disruption_count", "MTA alerts"),
        ("arriving_flights_count", "flights"),
    ]:
        zero_pct = 100 * (merged[col] == 0).sum() / len(merged)
        logger.info(f"{label}: {zero_pct:.1f}% of rows have zero "
                     f"(expected outside that source's date coverage)")

    summary_path = os.path.join(OUT_DIR, "t5_merged_summary.csv")
    summary = pd.DataFrame([{
        "total_rows": len(merged),
        "total_columns": len(merged.columns),
        "rows_missing_weather": int(missing_weather),
        "rows_missing_zone_lookup": int(missing_zone_info),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()