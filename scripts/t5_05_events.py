#!/usr/bin/env python3
"""
T5 — Fetch NYC Permitted Event Information (Historical), build two separate
city-wide (date, hour) features:

  1. big_venue_event_count — events at named/keyword-matched large venues
     (Hall, Theater, Arena, Stadium, Coliseum, etc. + a whitelist of major
     venues like Madison Square Garden). 

  2. street_closure_event_count — any event with a real (non-N/A) Street
     Closure Type in the dataset, i.e. parades, marathons, street fairs,
     block parties, charity walks. Uses the dataset's own field as a
     reliable signal for events that could plausibly affect demand 
     via street closures, distinct from the "large indoor/outdoor venue" 
     signal above.

Kept as two separate features since they represent different phenomena so the models 
can learn and decide their relative importance in the demand prediction.
"""

import logging
import os
import re

import pandas as pd
import requests

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ID = "bkfu-528j"  # NYC Permitted Event Information - Historical
PAGE_LIMIT = 5000

# Generic keywords — chosen to be reasonably specific to large venues,
# avoiding words like "center"/"field"/"garden"/"park" alone which would
# also match small unrelated permits (Senior Center, community garden, etc.)
KEYWORDS = ["hall", "theater", "theatre", "arena", "stadium",
            "coliseum", "amphitheater", "amphitheatre", "pavilion", "bandshell"]

# Explicit whitelist for major venues whose names don't contain a safe
# generic keyword above (e.g. "Madison Square Garden" has none of the
# above unless "Theater at MSG" is used) — more precise than adding
# risky generic words like "center"/"field"/"garden" to the list above
NAMED_VENUES = [
    "madison square garden", "barclays center", "citi field",
    "lincoln center", "javits center", "prudential center",
    "ubs arena", "forest hills stadium", "yankee stadium",
    "citi field", "kings theatre", "radio city",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_all_events() -> pd.DataFrame:
    """Fetch all records from the Socrata dataset, paginated."""
    all_rows = []
    offset = 0

    while True:
        response = requests.get(
            f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json",
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
    logger.info("Fetching NYC Permitted Event Information (Historical)...")
    events = fetch_all_events()
    logger.info(f"Total raw events fetched: {len(events):,}")

    raw_path = os.path.join(FEATURES_DIR, "nyc_permitted_events_raw.parquet")
    events.to_parquet(raw_path, index=False)
    logger.info(f"Raw events saved to {raw_path}")

    # Filter to big-venue events: word-boundary keyword match OR named-venue
    # whitelist match. Uses regex \b (word boundary) instead of raw substring
    # search — "hall" as a substring matches inside "Halloween", inflating
    # false positives. Also explicitly excludes "City Hall" (Park), a very
    # common march/rally/walk endpoint that isn't itself a large event venue.
    name_col = events.get("event_name", pd.Series(dtype=str)).fillna("")
    loc_col = events.get("event_location", pd.Series(dtype=str)).fillna("")
    combined_text = (name_col + " " + loc_col).str.lower()

    keyword_pattern = r"\b(" + "|".join(KEYWORDS) + r")\b"
    keyword_mask = combined_text.str.contains(keyword_pattern, regex=True, na=False)

    named_venue_mask = combined_text.apply(lambda text: any(v in text for v in NAMED_VENUES))

    city_hall_mask = combined_text.str.contains(r"city hall", regex=True, na=False)

    big_venue_events = events[(keyword_mask | named_venue_mask) & ~city_hall_mask].copy()
    logger.info(f"Big-venue events (word-boundary keyword or named-venue match, "
                f"City Hall excluded): {len(big_venue_events):,} / {len(events):,}")

    # Diagnostic: which keyword matched each event, for sanity-checking
    def find_match_reason(text):
        for kw in KEYWORDS:
            if re.search(r"\b" + kw + r"\b", text):
                return kw
        for v in NAMED_VENUES:
            if v in text:
                return v
        return "unknown"

    matched_text = combined_text[(keyword_mask | named_venue_mask) & ~city_hall_mask]
    match_reasons = matched_text.apply(find_match_reason)
    logger.info(f"\nMatch reason breakdown:\n{match_reasons.value_counts().to_string()}")

    # Second feature: genuine street closures, using the dataset's own
    # "Street Closure Type" field rather than guessing from keywords —
    # a real signal for parades/marathons/street fairs/block parties etc,
    # separate from "big venue" events. Excludes null/N/A (no real closure).
    closure_col = events.get("street_closure_type", pd.Series(dtype=str)).fillna("")
    closure_col = closure_col.str.strip()
    has_closure = closure_col.ne("") & closure_col.str.lower().ne("n/a")
    street_closure_events = events[has_closure].copy()
    logger.info(f"Street closure events (has a real Street Closure Type): "
                f"{len(street_closure_events):,} / {len(events):,}")

    # Parse start datetime and aggregate — for both categories
    def parse_and_aggregate(df: pd.DataFrame, count_col_name: str) -> pd.DataFrame:
        df = df.copy()
        df["start_dt"] = pd.to_datetime(df["start_date_time"], errors="coerce")
        df = df.dropna(subset=["start_dt"])
        df["date"] = df["start_dt"].dt.date
        df["hour"] = df["start_dt"].dt.hour
        return (
            df.groupby(["date", "hour"])
            .size()
            .reset_index(name=count_col_name)
        )

    big_venue_agg = parse_and_aggregate(big_venue_events, "big_venue_event_count")
    street_closure_agg = parse_and_aggregate(street_closure_events, "street_closure_event_count")

    agg = big_venue_agg.merge(street_closure_agg, on=["date", "hour"], how="outer")
    agg["big_venue_event_count"] = agg["big_venue_event_count"].fillna(0).astype(int)
    agg["street_closure_event_count"] = agg["street_closure_event_count"].fillna(0).astype(int)

    out_path = os.path.join(FEATURES_DIR, "nyc_events_by_hour.parquet")
    agg.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(agg):,} (date, hour) rows to {out_path}")

    summary_path = os.path.join(OUT_DIR, "t5_nyc_events_summary.csv")
    summary = pd.DataFrame([{
        "total_raw_events": len(events),
        "big_venue_events_matched": len(big_venue_events),
        "street_closure_events_matched": len(street_closure_events),
        "date_hour_rows": len(agg),
        "date_min": str(agg["date"].min()),
        "date_max": str(agg["date"].max()),
    }])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary.to_string(index=False)}")

    # Show samples of both categories for sanity-checking
    big_venue_sample = big_venue_events["event_name"].dropna().unique()[:40]
    logger.info(f"\nSample BIG VENUE event names: {list(big_venue_sample)}")

    closure_sample = street_closure_events["event_name"].dropna().unique()[:40]
    logger.info(f"\nSample STREET CLOSURE event names: {list(closure_sample)}")


if __name__ == "__main__":
    main()