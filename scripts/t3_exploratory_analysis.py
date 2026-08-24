import os
import logging
import duckdb 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
INPUT_BASE = os.path.join(WORK_DIR, "repartitioned_data")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t3")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATASET_CHECKS = {
    "yellow": {"min_year": 2012, "has_distance": True, "has_passengers": True, "has_fare": True},
    "green": {"min_year": 2014, "has_distance": True, "has_passengers": True, "has_fare": True},
    "fhv": {"min_year": 2015, "has_distance": False, "has_passengers": False, "has_fare": False},
    "fhvhv": {"min_year": 2019, "has_distance": True, "has_passengers": False, "has_fare": True},}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_heatmap(dataset):
    path_to_dataset = f"{INPUT_BASE}/{dataset}/year=*/*.parquet"
    df = duckdb.query(f"""
    SELECT HOUR(Pickup_DateTime) AS hour, 
    ISODOW(Pickup_DateTime) AS day_of_week, 
    COUNT(*) AS taxi_demand
    FROM read_parquet('{path_to_dataset}') 
    WHERE YEAR(Pickup_DateTime) > {DATASET_CHECKS[dataset]["min_year"]}
    AND Pickup_DateTime <= '2026-02-01'
    GROUP BY day_of_week, hour""").to_df()

    # HEATMAP PER HOUR PER DAY OF WEEK
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_data = df.pivot_table(index='day_of_week', columns='hour', values='taxi_demand')
    heatmap_data.reindex(range(1,8))
    heatmap_data.index = day_names

    # VIZUALIZATION
    plt.figure(figsize=(24, 7))
    sns.heatmap(heatmap_data, cmap="rainbow", annot=True, linewidth=0.5)
    plt.title(f"Taxi demand for {dataset} dataset per weekday and hour")
    plt.xlabel("Hour")
    plt.ylabel("Day of week")
    fig_path = os.path.join(FIG_DIR, f"heatmap_{dataset}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()


def get_monthly(dataset):
    path_to_dataset = f"{INPUT_BASE}/{dataset}/year=*/*.parquet"
    return duckdb.query(f"""
    SELECT DATE_TRUNC('month', Pickup_DateTime) AS month, COUNT(*) AS taxi_demand,
    FROM read_parquet('{path_to_dataset}') 
    WHERE Pickup_DateTime >= '2019-02-01' AND Pickup_DateTime <= '2026-02-01'
    GROUP BY month
    ORDER BY month""").to_df()

def get_duration_stats(dataset):
    if dataset == "fhvhv":
        duration = "Trip_Time/60.0"
    else:
        duration = "EXTRACT(EPOCH FROM (Dropoff_DateTime - Pickup_DateTime))/60.0"
    path_to_dataset = f"{INPUT_BASE}/{dataset}/year=*/*.parquet"
    con = duckdb.connect()
    return con.sql(f"""
        SELECT
            COUNT(*) AS trip_count,
            AVG({duration}) AS mean_duration,
            MEDIAN({duration}) AS median_duration,
            QUANTILE_CONT({duration}, 0.75) AS p75,
            QUANTILE_CONT({duration}, 0.90) AS p90,
            QUANTILE_CONT({duration}, 0.95) AS p95,
            QUANTILE_CONT({duration}, 0.99) AS p99
        FROM read_parquet('{path_to_dataset}') 
        WHERE YEAR(Pickup_DateTime) > {DATASET_CHECKS[dataset]["min_year"]}
        AND Pickup_DateTime <= '2026-02-01'
        AND {duration} BETWEEN 0 AND 1440""").df() 

def get_distance_stats(dataset):
    path_to_dataset = f"{INPUT_BASE}/{dataset}/year=*/*.parquet"
    if dataset == "fhv":
        return None

    distance = "Trip_Distance"
    con = duckdb.connect()
    return con.sql(f"""
        SELECT
            COUNT(*) AS trip_count,
            AVG({distance}) AS mean,
            MEDIAN({distance}) AS median,
            QUANTILE_CONT({distance}, 0.75) AS p75,
            QUANTILE_CONT({distance}, 0.90) AS p90,
            QUANTILE_CONT({distance}, 0.95) AS p95,
            QUANTILE_CONT({distance}, 0.99) AS p99
        FROM read_parquet('{path_to_dataset}')
        WHERE YEAR(Pickup_DateTime) > {DATASET_CHECKS[dataset]["min_year"]}
        AND Pickup_DateTime <= '2026-02-01'
        WHERE {distance} >= 0""").df() 

def main():
    monthly = pd.DataFrame({"month": pd.date_range(start="2019-02-01", end="2026-02-01", freq="MS")})
    datasets = ["yellow", "green", "fhv", "fhvhv"]
    for dataset in datasets:
        logger.info("=" * 80)
        logger.info(f"Processing dataset: {dataset.upper()}")
        logger.info("=" * 80)

        try:
            logger.info(f"Heatmap based on the temporal aggregation per hour per day of week.")
            get_heatmap(dataset)

            logger.info(f"Getting monthly taxi trips.")
            df_dataset = get_monthly(dataset)
            df_dataset = df_dataset.rename(columns={"taxi_demand":dataset})
            df_dataset["month"] = pd.to_datetime(df_dataset["month"])
            monthly = monthly.merge(df_dataset, on="month", how="left")

            logger.info(f"Calculating duration and distance stats (if information present in the dataset.)")
            duration_stats = get_duration_stats(dataset)
            distance_stats = get_distance_stats(dataset)

            if duration_stats is not None: 
                logger.info(f"Duration stats:")
                logger.info(f"{duration_stats}")
                
            if distance_stats is not None:
                logger.info(f"Distance stats:")
                logger.info(f"{distance_stats}")

        except Exception as e:
            print("[ERROR]", e)

    monthly = monthly.fillna(0)
    monthly_correlation = monthly[datasets].corr()
    monthly_correlation.to_csv(os.path.join(OUT_DIR, "monthly_corr.csv"), index=True)
    logger.info(f"Monthly trip demand correlation between the datasets: {monthly_correlation}")

if __name__ == "__main__":
    main()
