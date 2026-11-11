# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory analysis — city A (New Orleans) and city B (Asheville)
# MAGIC
# MAGIC Runs after `01_bronze_ingest` and `02_silver_clean` for both training cities. City C is not loaded here.
# MAGIC
# MAGIC ## Summary
# MAGIC > **Fill in after running the notebook. Replace every TODO with a number or a finding.**
# MAGIC
# MAGIC **What we now know**
# MAGIC - Snapshots: New Orleans TODO, Asheville TODO. One row = one listing on the snapshot date; `id` is unique (section 1).
# MAGIC - TODO % of listings have no price. Missing price is (not) random: it goes together with TODO (section 2).
# MAGIC - Median price: New Orleans $TODO, Asheville $TODO; prices are right-skewed, so we model log price (section 3).
# MAGIC - TODO % of listings require 30+ nights; TODO listings above the 99.5th percentile (section 4).
# MAGIC - Leaks: `estimated_revenue_l365d` is price × estimated occupancy (verified: ratio ≈ TODO); `review_scores_value` is a judgement of price (section 6).
# MAGIC
# MAGIC **What we still do not know**
# MAGIC - The advertised price is one night on one date, not what guests actually pay (no fees, no seasonality, no booked price).
# MAGIC - Whether listings without a price are inactive, blocked, or simply not scraped.
# MAGIC - How regulation (permits / licence rules in both cities) shapes which listings exist; we only see survivors.
# MAGIC - Whether the price–feature relationships learned here hold in a third city.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = spark.table(T("silver_listings")).where(F.col("city").isin(TRAIN_CITIES)).toPandas()
assert set(pdf.city.unique()) <= set(TRAIN_CITIES)          # city C must stay unseen
pdf["log_price"] = np.log10(pdf.price.where(pdf.price > 0))

# COMMAND ----------

# MAGIC %md ## 1. Dataset: snapshot dates, grain, record counts

# COMMAND ----------

display(spark.table(T("bronze_ingest_log")).where(F.col("city").isin(TRAIN_CITIES)))

grain = pdf.groupby("city").agg(rows=("id", "size"), unique_ids=("id", "nunique"),
                                hosts=("host_id", "nunique"), neighbourhoods=("neighbourhood", "nunique"))
display(grain.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Column profile: types and missing values
# MAGIC Raw values are strings in bronze; this is the typed silver view. Missing rates per city, then the
# MAGIC **pattern** of missingness — which other column explains it.

# COMMAND ----------

profile_cols = ["price", "accommodates", "bedrooms", "beds", "bathrooms", "room_type", "property_type",
                "latitude", "minimum_nights", "availability_365", "has_availability", "number_of_reviews",
                "review_scores_rating", "review_scores_location", "host_is_superhost", "host_response_rate",
                "host_since", "amenities_count", "license", "estimated_revenue_l365d"]
prof = pd.DataFrame({
    "dtype": pdf[profile_cols].dtypes.astype(str),
    **{f"missing_{c}": pdf[pdf.city == c][profile_cols].isna().mean().round(3) for c in TRAIN_CITIES},
})
display(prof.reset_index().rename(columns={"index": "column"}))

# COMMAND ----------

# Pattern 1: is a missing price explained by the listing being unavailable / inactive?
pdf["price_missing"] = pdf.price.isna()
pdf["avail_zero"] = pdf.availability_365 == 0
pdf["no_recent_reviews"] = pdf.number_of_reviews_ltm.fillna(0) == 0
for col in ["avail_zero", "has_availability", "no_recent_reviews", "room_type"]:
    display(pd.crosstab([pdf.city, pdf[col]], pdf.price_missing, normalize="index").round(3).reset_index())

# COMMAND ----------

# Pattern 2: review scores are missing because there are no reviews (structural, not an error)
display(pd.crosstab([pdf.city, pdf.number_of_reviews.fillna(0) == 0], pdf.review_scores_rating.isna()).reset_index())

# Pattern 3: bedrooms / beds missing by room type (studios? hotel rooms?)
display(pdf.groupby(["city", "room_type"])[["bedrooms", "beds", "bathrooms"]].apply(lambda d: d.isna().mean()).round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## 3. Distributions of price and availability

# COMMAND ----------

q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995]
display(pdf.groupby(["city", "room_type"]).price.describe(percentiles=q).round(1).reset_index())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for c in TRAIN_CITIES:
    pdf[pdf.city == c].log_price.dropna().plot.hist(bins=60, alpha=0.5, ax=axes[0], label=c)
    pdf[pdf.city == c].availability_365.dropna().plot.hist(bins=50, alpha=0.5, ax=axes[1], label=c)
    pdf[pdf.city == c].accommodates.dropna().clip(upper=16).plot.hist(bins=16, alpha=0.5, ax=axes[2], label=c)
axes[0].set_title("log10(price)"); axes[1].set_title("availability_365"); axes[2].set_title("accommodates (16+ capped)")
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()

# COMMAND ----------

# Price vs. the obvious drivers
display(pdf.groupby(["city", "room_type", "accommodates"]).price.agg(["median", "size"]).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Outliers and how we classify them
# MAGIC | Class | Rule | Treatment |
# MAGIC |---|---|---|
# MAGIC | `missing` | no price | kept in silver/gold (counts, dashboard); not used for training or evaluation |
# MAGIC | `too_low` | price < $10 | data error / placeholder; excluded from model fitting, kept in evaluation |
# MAGIC | `extreme_high` | price > city p99.5 | possibly real (large houses, events) or a placeholder; excluded from fitting, kept in evaluation and also reported separately |
# MAGIC | `is_monthly` | minimum_nights ≥ 30 | a monthly rental, not a short stay; kept, `minimum_nights` is a feature |
# MAGIC
# MAGIC Below: the counts, and examples to judge whether high prices are real.

# COMMAND ----------

display(pd.crosstab([pdf.city, pdf.price_flag], pdf.is_monthly).reset_index())
display(pdf[pdf.price_flag == "extreme_high"][["city", "name", "room_type", "accommodates", "bedrooms",
                                                "price", "minimum_nights", "availability_365", "number_of_reviews"]]
        .sort_values("price", ascending=False).head(30))

# Within-group outliers: log price outside 1.5×IQR of its city × room type
g = pdf.dropna(subset=["log_price"]).groupby(["city", "room_type"]).log_price
iqr_lo, iqr_hi = g.transform(lambda s: s.quantile(.25) - 1.5 * (s.quantile(.75) - s.quantile(.25))), \
                 g.transform(lambda s: s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25)))
lp = pdf.dropna(subset=["log_price"])
print("share outside IQR fences per city:",
      ((lp.log_price < iqr_lo) | (lp.log_price > iqr_hi)).groupby(lp.city).mean().round(3).to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cleaning and transformations the pipeline applies (each with its reason)
# MAGIC | # | Step (layer) | Reason |
# MAGIC |---|---|---|
# MAGIC | 1 | Read CSV with `multiLine` + `escape='"'` (bronze) | descriptions contain line breaks and quotes; a naive read splits rows |
# MAGIC | 2 | Keep all columns as strings in bronze, add `city`, `snapshot_date`, `source_file` | bronze is a faithful copy; lineage for every row |
# MAGIC | 3 | `try_cast` for every number and date (silver) | serverless runs ANSI SQL: one bad value would fail the job |
# MAGIC | 4 | Price `"$1,234.00"` → 1234.0 | price is stored as text with currency symbol and thousands separator |
# MAGIC | 5 | `"95%"` → 0.95, `t/f` → 1/0 | response rates and booleans are text |
# MAGIC | 6 | Bathrooms from `bathrooms_text` when the numeric column is empty; `bath_shared` flag | the numeric column is often empty; "shared" matters for price |
# MAGIC | 7 | Amenities JSON → count + flags (pool, hot tub, free parking, washer, AC) | a raw JSON list is not usable as a feature |
# MAGIC | 8 | `accommodates ≤ 0` → null | impossible value |
# MAGIC | 9 | Deduplicate on `id` | grain is one listing |
# MAGIC | 10 | `price_flag`, `is_monthly`, `room_type_known` | classify outliers and unexpected values instead of silently dropping them |
# MAGIC | 11 | `host_years`, `days_since_last_review` relative to snapshot date | dates are only meaningful relative to the snapshot |
# MAGIC | 12 | `dist_center_km` to the city's median listing location (gold) | location feature that means the same in every city |
# MAGIC | 13 | Hash-based 80/20 split on `id` (gold) | fixed before any modelling; reproducible |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Leakage review — which columns must not be used to predict price
# MAGIC | Column | Verdict | Why |
# MAGIC |---|---|---|
# MAGIC | `price` (and anything parsed from it) | target | — |
# MAGIC | `estimated_revenue_l365d` | **leak** | computed by Inside Airbnb as estimated occupancy × price (checked below) |
# MAGIC | `review_scores_value` | **leak (soft)** | guests rate *value for money*, i.e. quality relative to the price they paid |
# MAGIC | `estimated_occupancy_l365d` | excluded | estimate built from reviews and minimum nights; booking volume is partly a *result* of price |
# MAGIC | `availability_30/60/90/365` | excluded | depends on bookings, which depend on price (reverse causality); also the strongest predictor of a *missing* price |
# MAGIC | calendar.csv prices | excluded | the same price, from another file |
# MAGIC | `name`, `description` | not used | free text can contain the price or "luxury" wording written after pricing; out of scope |
# MAGIC | `neighbourhood`, `city` | not used (design choice) | allowed, but category values differ in every city, so they cannot transfer; location enters as `dist_center_km` |

# COMMAND ----------

# Verify the revenue leak numerically
chk = pdf.dropna(subset=["price", "estimated_revenue_l365d", "estimated_occupancy_l365d"])
chk = chk[chk.estimated_occupancy_l365d > 0]
ratio = chk.estimated_revenue_l365d / (chk.price * chk.estimated_occupancy_l365d)
print("revenue / (price × occupancy): median", round(ratio.median(), 3), "| share within ±1%:", round(((ratio - 1).abs() < 0.01).mean(), 3))

# Rank all numeric columns by correlation with log price: anything suspiciously high gets a second look
num = pdf.select_dtypes("number").drop(columns=["price", "log_price"], errors="ignore")
corr = num.corrwith(pdf.log_price, method="spearman").dropna().sort_values(key=np.abs, ascending=False)
display(corr.round(3).rename("spearman_vs_log_price").reset_index().rename(columns={"index": "column"}))
