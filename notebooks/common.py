# Databricks notebook source
# MAGIC %md
# MAGIC # Common config and helpers
# MAGIC Included by every notebook with `%run ./common`. Nothing in here is specific to one city:
# MAGIC adding a city means passing a new `city` parameter, never copying code.

# COMMAND ----------

import re
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOG = "workspace"          # default catalog in Databricks Free Edition
SCHEMA = "airbnb"
VOLUME = "raw"
RAW_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Training cities A and B. City C is deliberately not named anywhere in the code.
TRAIN_CITIES = ["new-orleans", "asheville"]

MODEL_NAME = f"{CATALOG}.{SCHEMA}.price_model"

EXPECTED_ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


def T(name: str) -> str:
    """Fully qualified table name."""
    return f"{CATALOG}.{SCHEMA}.{name}"


def validate_city(city: str) -> str:
    """City must be an Inside Airbnb slug such as 'new-orleans' (also keeps SQL predicates safe)."""
    city = city.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", city):
        raise ValueError(f"City must be an Inside Airbnb slug like 'new-orleans', got {city!r}")
    return city


def ensure_schema():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")


def write_partition(df, table: str, predicate: str, partition_by=("city",)):
    """Idempotent write: replaces only rows matching `predicate` (e.g. one city), so re-running
    a city never touches other cities and running a city twice gives the same result."""
    if spark.catalog.tableExists(table):
        (df.write.format("delta").mode("overwrite")
           .option("replaceWhere", predicate)
           .option("mergeSchema", "true")
           .saveAsTable(table))
    else:
        df.write.format("delta").partitionBy(*partition_by).saveAsTable(table)


# ---------- model features (shared by training and scoring) ----------

FEATURES_NUM = [
    # size
    "accommodates", "bedrooms", "beds", "bathrooms", "bath_shared",
    # location (city-agnostic: distance to the city's own median listing location)
    "dist_center_km",
    # stay rules
    "minimum_nights",
    # amenities
    "amenities_count", "has_pool", "has_hot_tub", "has_free_parking", "has_washer", "has_ac",
    # host_years (from host_since) and instant_bookable are empty in these Inside Airbnb snapshots
    "host_is_superhost", "calculated_host_listings_count",
    # reviews (review_scores_value is excluded on purpose: see leakage review)
    "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",
    "review_scores_location", "review_scores_cleanliness", "days_since_last_review",
]
FEATURES_CAT = ["room_type", "property_type"]


def prepare_X(pdf: pd.DataFrame) -> pd.DataFrame:
    X = pdf[FEATURES_NUM + FEATURES_CAT].copy()
    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in FEATURES_CAT:
        X[c] = X[c].fillna("unknown").astype(str)
    return X


# ---------- baseline: median price by room type x accommodates, fitted on training data ----------

def fit_baseline(train: pd.DataFrame) -> pd.DataFrame:
    a = (train.groupby(["room_type", "accommodates"]).price.agg(["median", "size"])
         .reset_index().rename(columns={"median": "median_price", "size": "n"}))
    a["level"] = "room_type_accommodates"
    b = (train.groupby("room_type").price.agg(["median", "size"])
         .reset_index().rename(columns={"median": "median_price", "size": "n"}))
    b["level"] = "room_type"
    b["accommodates"] = np.nan
    g = pd.DataFrame([{"level": "global", "room_type": None, "accommodates": np.nan,
                       "median_price": float(train.price.median()), "n": len(train)}])
    out = pd.concat([a, b, g], ignore_index=True)[["level", "room_type", "accommodates", "median_price", "n"]]
    out["accommodates"] = out["accommodates"].astype("float64")
    out["n"] = out["n"].astype("int64")
    return out


def apply_baseline(pdf: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
    """Fallback chain: room type x accommodates -> room type -> global median."""
    keys = pdf[["room_type", "accommodates"]].reset_index(drop=True).copy()
    keys["accommodates"] = pd.to_numeric(keys["accommodates"], errors="coerce").astype("float64")
    l1 = lookup[lookup.level == "room_type_accommodates"][["room_type", "accommodates", "median_price"]]
    l2 = lookup[lookup.level == "room_type"][["room_type", "median_price"]]
    g = float(lookup[lookup.level == "global"].median_price.iloc[0])
    out = (keys.merge(l1.rename(columns={"median_price": "b1"}), on=["room_type", "accommodates"], how="left")
               .merge(l2.rename(columns={"median_price": "b2"}), on="room_type", how="left"))
    return out.b1.fillna(out.b2).fillna(g).to_numpy()


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")
