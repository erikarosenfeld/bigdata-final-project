#!/usr/bin/env python3
"""
T5 — Generate augmentation figures into outputs/t5/figs/.
Read-only: reads demand_augmented.parquet (built by t5_09_merge_features.py).

Figures 1-2 are descriptive zone breakdowns (no causal claim).
Figures 3-5 examine time-varying augmentation features. Because temperature
and flight volume are both strongly correlated with hour-of-day (hot hours
are summer afternoons; flights land during daytime peaks), averaging across
all hours conflates the feature's effect with the diurnal demand cycle.
Each of these is therefore plotted BOTH ways:
  (a) pooled across all hours  — shows the confounded relationship
  (b) within fixed hour slots  — controls for time of day
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WORK_DIR = "/d/hpc/projects/FRI/bigdata/students/em51537"
REPO_DIR = os.path.join(WORK_DIR, "bigdata-final-project")
FEATURES_DIR = os.path.join(WORK_DIR, "t5_features")
OUT_DIR = os.path.join(REPO_DIR, "outputs", "t5")
FIG_DIR = os.path.join(OUT_DIR, "figs")

os.makedirs(FIG_DIR, exist_ok=True)

# Hour slots for the hour-controlled plots: morning peak, midday,
# evening peak, and a low-demand overnight hour
HOUR_SLOTS = [8, 13, 18, 3]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading demand_augmented.parquet...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "demand_augmented.parquet"))
    logger.info(f"  {len(df):,} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Figure 1: share of city-wide demand by service zone (descriptive)
    # ------------------------------------------------------------------
    by_service = df.groupby("service_zone")["total_trip_count"].sum().sort_values(ascending=False)
    share = 100 * by_service / by_service.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    share.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Share of city-wide taxi demand by service zone")
    ax.set_xlabel("Service zone")
    ax.set_ylabel("% of all trips")
    for i, v in enumerate(share.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_by_service_zone.png"), dpi=150)
    plt.close(fig)
    logger.info(f"\nDemand share by service zone (%):\n{share.round(2).to_string()}")
    share.round(2).to_csv(os.path.join(OUT_DIR, "t5_demand_share_by_service_zone.csv"),
                           header=["pct_of_all_trips"])

    # ------------------------------------------------------------------
    # Figure 2: demand by borough (descriptive)
    # ------------------------------------------------------------------
    by_borough = df.groupby("Borough")["total_trip_count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    by_borough.plot(kind="bar", ax=ax, color="darkseagreen", edgecolor="black")
    ax.set_title("Total taxi demand by borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Total trips")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_by_borough.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # City-wide hourly aggregation for time-varying features
    # ------------------------------------------------------------------
    hourly = df.groupby(["date", "hour"]).agg(
        demand=("total_trip_count", "sum"),
        temperature_c=("temperature_c", "first"),
        precipitation_mm=("precipitation_mm", "first"),
        arriving_flights_count=("arriving_flights_count", "first"),
    ).reset_index()

    # ------------------------------------------------------------------
    # Figure 3a/3b: demand vs temperature — pooled, and by fixed hour.
    # Restricted to -10..35 C; NYC rarely reaches the extremes, so outer
    # bins rest on very few observations and are unreliable.
    # ------------------------------------------------------------------
    temp = hourly.dropna(subset=["temperature_c"]).copy()
    temp = temp[(temp["temperature_c"] >= -10) & (temp["temperature_c"] <= 35)]
    temp["temp_bin"] = (temp["temperature_c"] // 5) * 5

    fig, ax = plt.subplots(figsize=(9, 5))
    temp.groupby("temp_bin")["demand"].mean().plot(marker="o", ax=ax, color="firebrick")
    ax.set_title("Mean hourly demand vs. temperature (all hours pooled)")
    ax.set_xlabel("Temperature [°C, 5° bins]")
    ax.set_ylabel("Mean trips per hour")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_temperature_pooled.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for h in HOUR_SLOTS:
        subset = temp[temp["hour"] == h]
        subset.groupby("temp_bin")["demand"].mean().plot(marker="o", ax=ax, label=f"{h:02d}:00")
    ax.set_title("Mean hourly demand vs. temperature, within fixed hours")
    ax.set_xlabel("Temperature [°C, 5° bins]")
    ax.set_ylabel("Mean trips per hour")
    ax.legend(title="Hour of day")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_temperature_by_hour.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4a/4b: demand vs precipitation — pooled, and by fixed hour
    # ------------------------------------------------------------------
    precip = hourly.dropna(subset=["precipitation_mm"]).copy()
    rain_bins = [-0.01, 0.01, 0.5, 2, 5, 1000]
    rain_labels = ["none", "light", "moderate", "heavy", "very heavy"]
    precip["rain_bin"] = pd.cut(precip["precipitation_mm"], bins=rain_bins, labels=rain_labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    precip.groupby("rain_bin", observed=True)["demand"].mean().plot(
        kind="bar", ax=ax, color="slateblue", edgecolor="black")
    ax.set_title("Mean hourly demand vs. precipitation (all hours pooled)")
    ax.set_xlabel("Precipitation intensity")
    ax.set_ylabel("Mean trips per hour")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_precipitation_pooled.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    for i, h in enumerate(HOUR_SLOTS):
        subset = precip[precip["hour"] == h]
        means = subset.groupby("rain_bin", observed=True)["demand"].mean()
        positions = [x + i * width for x in range(len(means))]
        ax.bar(positions, means.values, width=width, label=f"{h:02d}:00", edgecolor="black")
    ax.set_xticks([x + 1.5 * width for x in range(len(rain_labels))])
    ax.set_xticklabels(rain_labels)
    ax.set_title("Mean hourly demand vs. precipitation, within fixed hours")
    ax.set_xlabel("Precipitation intensity")
    ax.set_ylabel("Mean trips per hour")
    ax.legend(title="Hour of day")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_precipitation_by_hour.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5a/5b: demand vs arriving flights — pooled, and by fixed hour
    # ------------------------------------------------------------------
    flights = hourly[hourly["arriving_flights_count"] > 0].copy()
    if len(flights) == 0:
        logger.warning("No rows with arriving flights — skipping flight figures")
    else:
        flights["flight_bin"] = pd.qcut(flights["arriving_flights_count"], q=6, duplicates="drop")

        fig, ax = plt.subplots(figsize=(10, 5))
        flights.groupby("flight_bin", observed=True)["demand"].mean().plot(
            kind="bar", ax=ax, color="darkorange", edgecolor="black")
        ax.set_title("Mean hourly demand vs. arriving flights (all hours pooled, 2019-2023)")
        ax.set_xlabel("Arriving flights per hour (binned)")
        ax.set_ylabel("Mean trips per hour")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_flights_pooled.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for h in HOUR_SLOTS:
            subset = flights[flights["hour"] == h]
            if len(subset) < 20:
                continue
            subset = subset.copy()
            subset["fbin"] = pd.qcut(subset["arriving_flights_count"], q=4, duplicates="drop")
            means = subset.groupby("fbin", observed=True)["demand"].mean()
            ax.plot(range(len(means)), means.values, marker="o", label=f"{h:02d}:00")
        ax.set_title("Mean hourly demand vs. arriving flights, within fixed hours")
        ax.set_xlabel("Arriving flights per hour (quartile within that hour)")
        ax.set_ylabel("Mean trips per hour")
        ax.legend(title="Hour of day")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "t5_demand_vs_flights_by_hour.png"), dpi=150)
        plt.close(fig)

    logger.info(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()