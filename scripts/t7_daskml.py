# DISTRIBUTED ML TRAINING: DASKML 

import dask 
import dask.dataframe as dd 
from dask.distributed import Client, LocalCluster
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import PoissonRegression
from dask_ml.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import argparse, logging, os, time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t7")
FIG_DIR = os.path.join(OUT_DIR, "figs")
PREPROCESSED_DATASET = os.path.join(OUT_DIR, "preprocessed_augmented_dataset")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TO_TIME = pd.Timestamp("2026-03-01")
TIME_SPLIT = pd.Timestamp("2025-01-01")
DATASET_YEAR_RANGE = {
    "yellow": pd.Timestamp("2012-01-01"),
    "green": pd.Timestamp("2014-01-01"),
    "fhv": pd.Timestamp("2019-02-01"),
    "fhvhv": pd.Timestamp("2019-02-01"),
    "total": pd.Timestamp("2012-01-01"),}

TRIP_COUNT_COLS = ["total_trip_count", "yellow_trip_count", "green_trip_count", "fhv_trip_count", "fhvhv_trip_count"] 
TIME_COLS = ["year", "is_weekend", "is_holiday"]
EVENT_COLS = ["big_venue_event_count", "street_closure_event_count", "subway_disruption_count", "arriving_flights_count", "departing_flights_pickup_proxy_count"]
WEATHER_COLS = ["temperature_c", "precipitation_mm", "snowfall_cm", "windspeed_kmh"]
FEATURE_COLS = TIME_COLS + EVENT_COLS + WEATHER_COLS
NUM_COLS = EVENT_COLS + WEATHER_COLS + ["year"]

def eval_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**0.5
    mae, rmse = dask.compute(mae, rmse)

    logger.info(f"METRICS:")
    logger.info(f"--> MAE: {mae}")
    logger.info(f"--> RMSE: {rmse}\n")    
    return mae, rmse

def process_dataset(dataset):
    logger.info("=" * 80)
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info("=" * 80)

    # LOADING DATA 
    logger.info("Loading data...")
    df = dd.read_parquet(PREPROCESSED_DATASET)
    
    # FILTER DATA BASED ON THE DATASET START
    logger.info("Filtering dataset based on the dataset time range...")
    df = df[df["date"]>=DATASET_YEAR_RANGE[dataset]]

    # ONE HOT ENCODING
    logger.info("Creating one hot encoding...")
    hour_categories = list(range(24))
    dow_categories = list(range(7))
    month_categories = list(range(1,13))

    df["hour"] = df["hour"].astype(pd.CategoricalDtype(categories=hour_categories))
    df["month"] = df["month"].astype(pd.CategoricalDtype(categories=month_categories))
    df["day_of_week"] = df["day_of_week"].astype(pd.CategoricalDtype(categories=dow_categories))
    df = dd.get_dummies(df, columns=["hour", "month", "day_of_week"], dtype=float)

    # FEATURES AND TARGET VARIABLE
    feature_col_names = FEATURE_COLS + [c for c in df.columns if c.startswith("hour_") or c.startswith("month_") or c.startswith("day_of_week_")]
    target_col_name = f"{dataset}_trip_count"

    logger.info(f"Splitting data into train and test set: before and after {TIME_SPLIT}")
    train = df[df["date"]<TIME_SPLIT] 
    test = df[df["date"]>=TIME_SPLIT]

    # SCALING THE NUMERIC COLUMNS 
    logger.info("Scaling numeric data using standard scaler...")
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train[NUM_COLS])
    test_num_scaled = scaler.transform(test[NUM_COLS])

    # NON NUMERICAL FEATURES
    other_feature_cols = [a for a in feature_col_names if a not in NUM_COLS]
    X_train = dd.concat([train_num_scaled, train[other_feature_cols]], axis=1).to_dask_array(lengths=True)
    X_test = dd.concat([test_num_scaled, test[other_feature_cols]], axis=1).to_dask_array(lengths=True)

    y_train = train[target_col_name].to_dask_array(lengths=True).ravel()
    y_test = test[target_col_name].to_dask_array(lengths=True).ravel()         
    return X_train, y_train, X_test, y_test

def client_setup(n_workers, mem_limit):
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=2, memory_limit=f"{mem_limit}MB")
    client = Client(cluster)
    logger.info(f"DASK Client link: {client.dashboard_link}")
    logger.info(f"Number of workers: {len(client.scheduler_info()['workers'])}")
    return client

def main():
    print("Distributed DaskML")
    parser = argparse.ArgumentParser(description="T7: Distributed ML")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--mem_limit", type=int, default=16)
    args = parser.parse_args()

    # DASK CLIENT
    client = client_setup(args.workers, args.mem_limit)

    # MODELS FOR EACH OF THE DATASETS AND COMBINED ===========
    datasets = ["yellow", "green", "fhv", "fhvhv", "total"]
    for dataset in datasets:       
        # PREPARE, SCALE AND SPLIT DATA ======================
        X_train, y_train, X_test, y_test = process_dataset(dataset)

        # TRAINING MODEL =====================================
        model = PoissonRegression(solver="lbfgs", max_iter=500)
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        t_train = time.perf_counter()-t_start
        logger.info(f"Finished DaskML training. Elapsed time: {t_train}.")

        # EVALUATION =========================================
        logger.info(f"Starting making predictions.")
        t_start = time.perf_counter()
        y_pred = model.predict(X_test) 
        t_pred = time.perf_counter()-t_start
        logger.info(f"Finished making predictions, elapsed time: {t_pred}.")
        
        mae, rmse = eval_model(y_test, y_pred)
        print(f"\nEvaluation of model for {dataset} dataset:")
        print("MAE:", mae)
        print("RMSE:", rmse)

    client.close()

if __name__ == "__main__":
    main()