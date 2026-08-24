# TASK 7: AUGMENTED DATASET CITYWIDE PREPROCESSING FOR DISTRIBUTED ML 

import logging, os
import holidays 
import pandas as pd 
import dask.dataframe as dd 

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
INPUT_BASE = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t7")
FIG_DIR = os.path.join(OUT_DIR, "figs")
SAVE_DATASET_TO = os.path.join(OUT_DIR, "preprocessed_augmented_dataset")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# TRIP_COUNT_COLS: ["total_trip_count", "yellow_trip_count", "green_trip_count", "fhv_trip_count", "fhvhv_trip_count"] 
# EVENT_COLS: ["big_venue_event_count", "street_closure_event_count", "subway_disruption_event_count", "arriving_flights_count", "departing_flights_pickup_proxy_count"]
# WEATHER_COLS: ["temperature_c", "precipitation_mm", "snowfall_cm", "windspeed_khm"]

def process_augmented_dataset():
    logger.info("=" * 80)
    logger.info(f"Preprocessing augmented dataset.")
    logger.info("=" * 80)

    logger.info("Loading augmented dataset...")
    df = dd.read_parquet(os.path.join(INPUT_BASE, "demand_augmented.parquet"))
    df["date"] = dd.to_datetime(df["date"])
    df = df[df["date"]<pd.Timestamp("2026-03-01")]

    logger.info("Aggregating data city-wide...")
    agg_dict = {
        "total_trip_count": "sum", 
        "fhv_trip_count": "sum",
        "fhvhv_trip_count": "sum",
        "green_trip_count": "sum",
        "yellow_trip_count": "sum",
        "big_venue_event_count": "sum",
        "street_closure_event_count": "sum",
        "subway_disruption_count": "sum",
        "arriving_flights_count": "sum",
        "departing_flights_pickup_proxy_count": "sum",
        "temperature_c": "mean",
        "precipitation_mm": "mean",
        "snowfall_cm": "mean",
        "windspeed_kmh": "mean",}
    citywide = df.groupby(["date", "hour"]).agg(agg_dict).reset_index()

    # EXTRA TIME FEATURES
    logger.info("Adding extra time features...")
    us_holidays = holidays.US()
    citywide["date"] = dd.to_datetime(citywide["date"])
    citywide["month"] = citywide["date"].dt.month.astype(int)
    citywide["year"] = citywide["date"].dt.year.astype(int)
    citywide["day_of_week"] = citywide["date"].dt.dayofweek.astype(int)
    citywide["is_weekend"] = (citywide["day_of_week"]>=5).astype(int)
    citywide["is_holiday"] = citywide["date"].map_partitions(lambda x: x.dt.date.map(lambda d: d in us_holidays), meta=("is_holiday", "bool")).astype(int)

    # DEBUGGING
    citywide.head(100).to_csv(os.path.join(OUT_DIR, "sample_preprocessed.csv"))

    #  SAVE PARQUET
    repartitioned_citywide = citywide.repartition(npartitions=16)
    repartitioned_citywide.to_parquet(SAVE_DATASET_TO, engine="pyarrow", write_index=False)
    citywide.to_parquet(f"{SAVE_DATASET_TO}_whole.parquet", engine="pyarrow", write_index=False)
    logger.info(f"[CITYWIDE] Table saved to {SAVE_DATASET_TO}")
    logger.info(f"[CITYWIDE] Saved Parquet file size: {os.path.getsize(SAVE_DATASET_TO + '_whole.parquet')}")

def main():
    process_augmented_dataset()

if __name__ == "__main__":
    main()