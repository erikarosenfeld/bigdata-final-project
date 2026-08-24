#!/usr/bin/env python3
"""
T5 — Process flight data (Kaggle: patrickzel/flight-delay-and-cancellation-
dataset-2019-2023, a 3M-row sample of BTS flight records, 2019-2023) into
two city-wide (date, hour) features for NYC-area airports (JFK, LGA, EWR):

  1. arriving_flights_count — actual arrival time + ~40 min buffer
     (deplaning/baggage claim) = when someone becomes a taxi PICKUP at the
     airport itself.

  2. departing_flights_pickup_proxy_count — scheduled (not actual) departure
     time minus 3 hours = proxy for when travelers are being picked up
     elsewhere in the city, heading to the airport. Uses scheduled time
     since that's what travelers actually plan their departure around, not
     a delay they don't yet know about.

Both are valid features now that the demand target is CITY-WIDE (not
per-zone) — a departure's pickup can happen anywhere in the city, which
was the original objection to using departures at all; that no longer
applies at city-wide grain.

NOTE: this is a random 3M-row SAMPLE of all US flights (2019-2023), not
the complete dataset — coverage for specific airports/hours may be
incomplete. Coverage is checked and logged below; if too sparse, this
should be treated as a rough proxy, not a precise count.

Usage:
  python3 t5_09_flights.py
"""

import logging
import os

import pandas as pd

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FLIGHTS_CSV = os.path.join(REPO_DIR, "flights_sample_3m.csv")
NYC_AIRPORTS = ["JFK", "LGA", "EWR"]

ARRIVAL_BUFFER_MIN = 40  # deplaning + baggage claim before becoming a pickup
DEPARTURE_LEAD_HOURS = 3  # how early travelers arrive at the airport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_hhmm(series: pd.Series) -> pd.Series:
    """Parse BTS's HHMM integer time format (e.g. 1155, 155, 2400) into
    minutes-since-midnight. 2400 (BTS's convention for midnight) maps to 0."""
    s = series.fillna(0).astype(float).astype(int).astype(str).str.zfill(4)
    s = s.replace("2400", "0000")
    hours = s.str[:2].astype(int)
    minutes = s.str[2:].astype(int)
    return hours * 60 + minutes


def main():
    logger.info(f"Loading {FLIGHTS_CSV}...")
    df = pd.read_csv(FLIGHTS_CSV, parse_dates=["FL_DATE"])
    logger.info(f"Loaded {len(df):,} total flight rows")

    # -------- Arrivals to NYC airports --------
    arrivals = df[(df["DEST"].isin(NYC_AIRPORTS)) & (df["CANCELLED"] == 0.0)].copy()
    logger.info(f"Arrivals to {NYC_AIRPORTS}: {len(arrivals):,} rows "
                f"({100*len(arrivals)/len(df):.3f}% of sample)")

    arrivals["arr_minutes"] = parse_hhmm(arrivals["ARR_TIME"])
    arrivals["pickup_minutes"] = arrivals["arr_minutes"] + ARRIVAL_BUFFER_MIN
    arrivals["pickup_dt"] = arrivals["FL_DATE"] + pd.to_timedelta(arrivals["pickup_minutes"], unit="m")
    arrivals["date"] = arrivals["pickup_dt"].dt.date
    arrivals["hour"] = arrivals["pickup_dt"].dt.hour

    arrivals_agg = (
        arrivals.groupby(["date", "hour"])
        .size()
        .reset_index(name="arriving_flights_count")
    )

    # -------- Departures from NYC airports --------
    departures = df[(df["ORIGIN"].isin(NYC_AIRPORTS)) & (df["CANCELLED"] == 0.0)].copy()
    logger.info(f"Departures from {NYC_AIRPORTS}: {len(departures):,} rows "
                f"({100*len(departures)/len(df):.3f}% of sample)")

    departures["dep_minutes"] = parse_hhmm(departures["CRS_DEP_TIME"])
    departures["pickup_minutes"] = departures["dep_minutes"] - (DEPARTURE_LEAD_HOURS * 60)
    departures["pickup_dt"] = departures["FL_DATE"] + pd.to_timedelta(departures["pickup_minutes"], unit="m")
    departures["date"] = departures["pickup_dt"].dt.date
    departures["hour"] = departures["pickup_dt"].dt.hour

    departures_agg = (
        departures.groupby(["date", "hour"])
        .size()
        .reset_index(name="departing_flights_pickup_proxy_count")
    )

    # -------- Merge and save --------
    agg = arrivals_agg.merge(departures_agg, on=["date", "hour"], how="outer")
    agg["arriving_flights_count"] = agg["arriving_flights_count"].fillna(0).astype(int)
    agg["departing_flights_pickup_proxy_count"] = (
        agg["departing_flights_pickup_proxy_count"].fillna(0).astype(int)
    )

    out_path = os.path.join(FEATURES_DIR, "flights_by_hour.parquet")
    agg.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(agg):,} (date, hour) rows to {out_path}")

    # Coverage check — since this is a SAMPLE, not the full dataset
    date_range = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    hours_with_data = agg["date"].nunique()
    logger.info(f"Distinct dates with any flight data: {hours_with_data:,} "
                f"out of {len(date_range):,} days in range "
                f"({100*hours_with_data/len(date_range):.1f}% coverage)")

    summary_path = os.path.join(OUT_DIR, "t5_flights_summary.csv")
    summary = pd.DataFrame([{
        "total_sample_rows": len(df),
        "arrivals_matched": len(arrivals),
        "departures_matched": len(departures),
        "date_hour_rows": len(agg),
        "date_min": str(agg["date"].min()),
        "date_max": str(agg["date"].max()),
        "distinct_dates_with_data": hours_with_data,
        "days_in_range": len(date_range),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
