import pandas as pd

df = pd.read_parquet(
    "/d/hpc/projects/FRI/bigdata/students/em51537/repartitioned_data/fhv/year=2016/fhv_tripdata_2016-01.parquet"
)

print(df[["Pickup_DateTime", "Dropoff_DateTime"]].head(15))
print()
print("Rows where dropoff < pickup:", (df["Dropoff_DateTime"] < df["Pickup_DateTime"]).sum(), "/", len(df))
print()

diffs = (df["Pickup_DateTime"] - df["Dropoff_DateTime"]).dt.total_seconds()
print("Pickup - Dropoff gap (seconds), describe:")
print(diffs.describe())
