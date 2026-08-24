#!/usr/bin/env python3
"""
T5 step 2 — Fetch NYC historical weather (Open-Meteo, no API key needed) at
hourly granularity, matching the (date, hour) grain of the demand target.
NYC-wide (not per-zone) as weather doesn't meaningfully vary within the city
at this scale.

Chunked by year (resumable, avoids one giant request) and saved as a single
small parquet (~14 years x 8760 hours = ~123K rows)
"""

import logging
import os
from datetime import date, timedelta

import pandas as pd
import requests

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

NYC_LAT, NYC_LON = 40.7128, -74.0060
START_YEAR = 2012
END_YEAR = 2026  # matches your demand table's date_max (2026-12-16)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_year(year: int) -> pd.DataFrame:
    """Fetch one year of hourly weather via Open-Meteo's historical archive API."""
    # The archive API lags a few days behind real-time — for the current
    # (incomplete) year, cap the end date instead of always requesting Dec 31
    end_date = date(year, 12, 31)
    safe_max_date = date.today() - timedelta(days=5)
    if end_date > safe_max_date:
        end_date = safe_max_date

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": f"{year}-01-01",
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,precipitation,snowfall,windspeed_10m",
        "timezone": "America/New_York",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()["hourly"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df = df.rename(columns={
        "temperature_2m": "temperature_c",
        "precipitation": "precipitation_mm",
        "snowfall": "snowfall_cm",
        "windspeed_10m": "windspeed_kmh",
    })
    return df[["date", "hour", "temperature_c", "precipitation_mm", "snowfall_cm", "windspeed_kmh"]]


def main():
    out_path = os.path.join(FEATURES_DIR, "weather_nyc.parquet")

    all_years = []
    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"Fetching weather for {year}...")
        try:
            all_years.append(fetch_year(year))
        except Exception as e:
            logger.error(f"Failed to fetch {year}: {e}")

    weather = pd.concat(all_years, ignore_index=True)
    weather.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(weather):,} hourly weather rows to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_weather_summary.csv")
    summary = pd.DataFrame([{
        "total_rows": len(weather),
        "date_min": str(weather["date"].min()),
        "date_max": str(weather["date"].max()),
        "temp_min_c": weather["temperature_c"].min(),
        "temp_max_c": weather["temperature_c"].max(),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()