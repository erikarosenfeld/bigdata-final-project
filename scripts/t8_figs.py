import os, logging 
import pandas as pd
import matplotlib.pyplot as plt


WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
INPUT_DIR = os.path.join(REPO_DIR, "outputs", "t8")
FIG_DIR = os.path.join(INPUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

combined_df = pd.DataFrame()
for dataset in ["yellow", "green", "fhv", "fhvhv"]:
    table_path = os.path.join(INPUT_DIR, f"t8_{dataset}_trips_by_month.csv")
    df = pd.read_csv(table_path)[["month", "operator", "trip_count"]]
    df["month"] = pd.to_datetime(df["month"])
    df = df[df["month"]<pd.Timestamp("2026-02-01")]
    combined_df = pd.concat([combined_df, df], ignore_index=True)

pivot_df = combined_df.pivot(index='month', columns='operator', values='trip_count').fillna(0)
all_months = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='MS')
pivot_df = pivot_df.reindex(all_months, fill_value=0)

x = pivot_df.index
y = [pivot_df[col] for col in pivot_df.columns]

# GRAPH
plt.figure(figsize=(10, 6))
plt.stackplot(x, y, labels=pivot_df.columns, alpha=0.8)
plt.title("Number of taxi trips per operator per month")
plt.xlabel("Month")
plt.xticks(pivot_df.index[::3], pivot_df.index[::3].strftime("%Y-%m"), rotation=90)
plt.ylabel("Trip count")
plt.legend(title="Operator", loc='upper left')
plt.grid(alpha=0.07)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, f"t8_operators_per_month_stacked.pdf")
plt.savefig(fig_path, format="pdf", dpi=150)
logger.info(f"Plot saved to: {fig_path}.")

# PERCENTAGES
relative_df = 100*pivot_df.div(pivot_df.sum(axis=1), axis=0)
x = relative_df.index 
y = [relative_df[c] for c in relative_df.columns]

# RELATIVE TRIP VOLUME GRAPH 
plt.figure(figsize=(10,6))
plt.stackplot(x, y, labels=relative_df.columns, alpha=0.8)
plt.title("Percentage of taxi trips per operator per month")
plt.xticks(relative_df.index[::3], relative_df.index[::3].strftime("%Y-%m"), rotation=90)
plt.ylabel("Share of all trips per month [%]")
plt.xlabel("Month")
plt.legend()
plt.legend(title="Operator", loc='upper left')
plt.grid(alpha=0.07)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, f"t8_relative_shares.pdf")
plt.savefig(fig_path, format="pdf", dpi=150)
logger.info(f"Plot saved to: {fig_path}.")