#!/usr/bin/env python3
"""
T5 — Fetch MTA Service Alerts (subway disruptions: delays, suspensions,
etc.), aggregate to city-wide (date, hour) counts of NEW disruptions
starting in that hour.

Each disruption has multiple rows as it evolves (event_id + update_number,
e.g. delays -> part-suspended -> resolved) — groupped by event_id and take
the EARLIEST timestamp as when the disruption started, then count how many
distinct disruptions started per (date, hour) city-wide.

Data starts April 2020 (dataset's earliest coverage) — a pre-2020 archive
exists separately but isn't included here; this feature will simply be 0
for all pre-2020 rows when joined onto the demand table.
"""

import logging
import os

import pandas as pd
import requests

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ID = "7kct-peq7"  # MTA Service Alerts: Beginning April 2020
PAGE_LIMIT = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_all_alerts() -> pd.DataFrame:
    """Fetch all records from the Socrata dataset, paginated."""
    all_rows = []
    offset = 0

    while True:
        response = requests.get(
            f"https://data.ny.gov/resource/{DATASET_ID}.json",
            params={"$limit": PAGE_LIMIT, "$offset": offset},
            timeout=60,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        all_rows.extend(batch)
        logger.info(f"  Fetched {len(all_rows):,} rows so far...")
        offset += PAGE_LIMIT

    return pd.DataFrame(all_rows)


def main():
    logger.info("Fetching MTA Service Alerts...")
    alerts = fetch_all_alerts()
    logger.info(f"Total raw alert-update rows fetched: {len(alerts):,}")

    raw_path = os.path.join(FEATURES_DIR, "mta_alerts_raw.parquet")
    alerts.to_parquet(raw_path, index=False)
    logger.info(f"Raw alerts saved to {raw_path}")

    alerts["date_parsed"] = pd.to_datetime(alerts["date"], errors="coerce")
    alerts = alerts.dropna(subset=["date_parsed", "event_id"])

    # Group by event_id, take the EARLIEST timestamp as the disruption's start
    disruption_starts = (
        alerts.groupby("event_id")["date_parsed"].min().reset_index()
    )
    logger.info(f"Distinct disruptions (unique event_id): {len(disruption_starts):,}")

    disruption_starts["date"] = disruption_starts["date_parsed"].dt.date
    disruption_starts["hour"] = disruption_starts["date_parsed"].dt.hour

    agg = (
        disruption_starts.groupby(["date", "hour"])
        .size()
        .reset_index(name="subway_disruption_count")
    )

    out_path = os.path.join(FEATURES_DIR, "mta_alerts_by_hour.parquet")
    agg.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(agg):,} (date, hour) rows to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_mta_alerts_summary.csv")
    summary = pd.DataFrame([{
        "total_raw_rows": len(alerts),
        "distinct_disruptions": len(disruption_starts),
        "date_hour_rows": len(agg),
        "date_min": str(disruption_starts["date"].min()),
        "date_max": str(disruption_starts["date"].max()),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
