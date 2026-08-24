# DISTRIBUTED ML TRAINING: XGBOOST 

import argparse, logging, os, time
import pandas as pd

import dask
import dask.dataframe as dd 
from dask.distributed import Client, LocalCluster

from xgboost import dask as dask_xgb
from dask_ml.preprocessing import StandardScaler
from dask_ml.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# DIRECTORIES 
WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t7")
FIG_DIR = os.path.join(OUT_DIR, "figs")
PREPROCESSED_DATASET = os.path.join(OUT_DIR, "preprocessed_augmented_dataset")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# TIME RANGES 
TO_TIME = pd.Timestamp("2026-03-01")
TIME_SPLIT = pd.Timestamp("2025-01-01")
DATASET_YEAR_RANGE = {"yellow": 2012, "green": 2014, "fhv": 2015, "fhvhv": 2019, "total": 2012,}

# COLUMN NAMES 
TRIP_COUNT_COLS = ["total_trip_count", "yellow_trip_count", "green_trip_count", "fhv_trip_count", "fhvhv_trip_count"] 
TIME_COLS = ["year", "is_weekend", "is_holiday"]
EVENT_COLS = ["big_venue_event_count", "street_closure_event_count", "subway_disruption_count", "arriving_flights_count", "departing_flights_pickup_proxy_count"]
WEATHER_COLS = ["temperature_c", "precipitation_mm", "snowfall_cm", "windspeed_kmh"]
FEATURE_COLS = TIME_COLS + EVENT_COLS + WEATHER_COLS
NUM_COLS = EVENT_COLS + WEATHER_COLS + ["year"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def eval_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**0.5

    logger.info(f"METRICS:")
    logger.info(f"--> MAE: {mae}")
    logger.info(f"--> RMSE: {rmse}\n")

    mae, rmse = dask.compute(mae, rmse)
    return mae, rmse 

def process_dataset(dataset):
    logger.info("=" * 80)
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info("=" * 80)

    # LOADING DATA 
    logger.info(f"[{dataset}] Loading data...")
    df = dd.read_parquet(PREPROCESSED_DATASET)
    logger.info("Filtering dataset based on the dataset time range...")
    df = df[df["year"]>=DATASET_YEAR_RANGE[dataset]]


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
    print("Number of workers:", len(client.scheduler_info()['workers']))
    return client

def main():
    print("Distributed XGBoost")
    parser = argparse.ArgumentParser(description="T7: Distributed ML")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--mem_limit", type=int, default=16)
    args = parser.parse_args()
    client = client_setup(args.workers, args.mem_limit)
   
    # MODELS FOR EACH OF THE DATASETS AND COMBINED
    datasets = ["yellow", "green", "fhv", "fhvhv", "total"]
    for dataset in datasets:
        X_train, y_train, X_test, y_test = process_dataset(dataset)
        
        # DISTRIBUTED DASK MATRIX
        dtrain = dask_xgb.DaskDMatrix(client, X_train, y_train)
        dtest = dask_xgb.DaskDMatrix(client, X_test, y_test)

        # XGB PARAMS - TRIPS COUNT DATA 
        xgb_parameters = {
            "objective": "count:poisson",
            "tree_method": "hist",
            "eval_metric": "poisson-nloglik",
            "max_depth": 5, 
            "subsample": 0.75,
            "learning_rate": 8e-1,
            "seed": 0,}
            #"verbosity":1,}

        # TRAINING 
        logger.info(f"Training XGB params: {xgb_parameters}")
        print(xgb_parameters)
        logger.info(f"Starting XGBoost distributed training.")
        t_start = time.perf_counter()
        output = dask_xgb.train(client, xgb_parameters, dtrain, num_boost_round=200, evals=[(dtrain, "train"), (dtest, "test"),])
        t_train = time.perf_counter() - t_start 
        logger.info(f"Finished XGBoost distributed training. Elapsed time: {t_train}.")

        # XGB OUTPUTS 
        booster = output["booster"]
        history = output["history"]

        # PREDICTION 
        logger.info(f"Starting XGBoost predictions.")
        t_start = time.perf_counter()
        y_pred = dask_xgb.predict(client, booster, dtest)
        t_pred = time.perf_counter() - t_start
        logger.info(f"Finished making predictions. Elapsed time: {t_pred}.")

        # EVALUATION 
        mae, rmse = eval_model(y_test, y_pred)
        print(f"\nEvaluation of model for {dataset} dataset:")
        print("MAE:", mae)
        print("RMSE:", rmse)

    client.close()

if __name__ == "__main__":
    main()