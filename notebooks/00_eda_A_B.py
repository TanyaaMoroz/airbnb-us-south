# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory analysis — city A (New Orleans) and city B (Asheville)
# MAGIC
# MAGIC Runs after `01_bronze_ingest` and `02_silver_clean` for both training cities. City C is not loaded here.
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC **What we now know**
# MAGIC - **Snapshots and grain.** New Orleans 2026-09-16 (7,263 listings, 3,208 hosts) and Asheville 2026-09-19 (2,865 listings, 1,730 hosts). One row = one listing on the snapshot date. `id` is unique: 0 duplicates, and no id appears in both cities (section 1).
# MAGIC - **Missing prices are not random.** 14.8% (New Orleans) and 10.5% (Asheville) of listings have no price (section 2).
# MAGIC   - Every listing without a price quote has no price (702 / 211), and almost all of these were carried over from a previous scrape.
# MAGIC   - With an empty calendar (`availability_365 = 0`) the price is missing for 83.1% / 96.0%, against 7.0% / 4.2% for the rest.
# MAGIC - **Review scores are missing exactly when a listing has 0 reviews** (1,209 / 232 listings, no exceptions). This is structural, not an error.
# MAGIC - **Prices are skewed.** The median is $197 (New Orleans) and $225 (Asheville). Price skew is 15 / 16, but only 0.4 / 0.3 on log10, so we model log price (section 3).
# MAGIC - **`price` is the per-night value of a quoted stay, not a list price.**
# MAGIC   - The quote is for a specific check-in, a median of 3 days after the scrape.
# MAGIC   - 48% / 23% of the quotes include a discount.
# MAGIC   - For listings with `minimum_nights ≥ 30` it is a monthly rate per night: median $111 vs $247 for the rest in New Orleans, $101 vs $234 in Asheville.
# MAGIC - **Monthly and extreme listings.** 40.0% (New Orleans) and 8.6% (Asheville) of listings require 30+ nights; for New Orleans that is 40.8% of the listings with a known minimum stay. 31 / 12 listings are above the city's p99.5 (about $2,000 in both cities; exact values in `silver_quality`), and about 12 / 5 of those look like real large group houses. No price is below $10 (section 4).
# MAGIC - **Leaks** (section 6):
# MAGIC   - `estimated_revenue_l365d` is exactly price × `estimated_occupancy_l365d`: median ratio 1.000, 100% of 6,464 listings within ±1%.
# MAGIC   - The undocumented `price_quote_price_per_night` is identical to `price`.
# MAGIC   - `review_scores_value` barely correlates with price (Spearman 0.03). We exclude it as a precaution, not because the data shows a leak.
# MAGIC - **Empty columns.** `host_since`, `instant_bookable`, `host_response_rate`, `host_acceptance_rate` and the numeric `bathrooms` are 100% empty in both snapshots. Bathrooms come from `bathrooms_text`. `host_years` and `instant_bookable` carried no information, so they were dropped from the model features.
# MAGIC
# MAGIC *The quote details (discounts, check-in lead time, `price_quote_*`), the licence split, the capacity cap and the 277 / 62 unexplained listings were checked on the raw listings file, outside this notebook.*
# MAGIC
# MAGIC **What we still do not know**
# MAGIC - What guests actually pay. The price is one quoted stay on one date, often discounted, with no fees, no seasonality and no booked price.
# MAGIC - Why 277 (New Orleans) and 62 (Asheville) listings have an open calendar and an "available" quote but still no price. For the rest we cannot tell inactive, fully booked or blocked apart.
# MAGIC - Whether unavailable days are booked or blocked by the host. The dictionary says it can be either, so availability and occupancy are not clean demand signals.
# MAGIC - How regulation (permits / licence rules in both cities) shapes which listings exist. We only see survivors. In New Orleans the licence is empty for 42.7% of monthly listings vs 1.8% of the rest.
# MAGIC - The real size of large group houses: `accommodates` is capped at 16.
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

# Pattern 1b: was a price quote attempted at all? `source` and `price_quote_*` are only in bronze.
bcols = [c for c in ["source", "price_quote_checkin_date"] if c in spark.table(T("bronze_listings")).columns]
qb = (spark.table(T("bronze_listings")).where(F.col("city").isin(TRAIN_CITIES))
      .select("city", F.trim("id").alias("id"), *bcols).toPandas())
q = pdf[["city", "id", "price_missing", "avail_zero"]].merge(qb, on=["city", "id"], how="left")
if "source" in q:
    q["source"] = q.source.fillna("<empty>")
    display(pd.crosstab([q.city, q.source], q.price_missing, normalize="index").round(3).reset_index())
if "price_quote_checkin_date" in q:
    q["quote_attempted"] = q.price_quote_checkin_date.notna()
    display(pd.crosstab([q.city, q.quote_attempted], q.price_missing).reset_index())

# Pattern 4: minimum_nights is missing only when the calendar is empty
display(pd.crosstab([pdf.city, pdf.minimum_nights.isna()], pdf.availability_365 == 0).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing values: what goes with them and why
# MAGIC | column | % missing NO / AVL | pattern | most likely explanation | evidence | what the pipeline does |
# MAGIC |---|---|---|---|---|---|
# MAGIC | `price` | 14.8 / 10.5 | 100% missing when no price quote was attempted (702 / 211), 699 / 211 of those are `source = previous scrape`; 83.1% / 96.0% missing when `availability_365 = 0` vs 7.0% / 4.2% otherwise; ~98–100% when the quote says not available | listings carried over from an earlier scrape, mostly with an empty calendar, get no new quote and so no price | strong for the no-quote group; 277 / 62 listings with an open calendar and an available quote are unexplained | `price_flag = 'missing'`: kept for counts, excluded from fit and evaluation |
# MAGIC | review scores, `first_review`, `last_review`, `reviews_per_month`, `days_since_last_review` | 16.6 / 8.1 | missing ⇔ `number_of_reviews = 0` (1,209 / 232, no exceptions) | nothing to score yet: structural | exact | kept null; imputed in the model |
# MAGIC | `beds` | 3.0 / 1.5 | 169 of 221 New Orleans cases are "Room in hotel"; 12.9% of New Orleans private rooms | hotel / room listings leave beds empty | moderate | imputed |
# MAGIC | `bathrooms` (after text fallback) | 1.4 / 0.0 | all 105 New Orleans cases are hotel rooms with empty `bathrooms_text`; the numeric `bathrooms` column is 100% empty | hotels leave it empty | strong | parsed from `bathrooms_text` |
# MAGIC | `minimum_nights`, `maximum_nights` | 1.9 / 0.3 | all 137 / 8 have `availability_365 = 0`, yet 126 / 8 have a price | no calendar → no calendar-based stay rules | strong | `is_monthly` becomes null; imputed |
# MAGIC | `has_availability` | 0.6 / 0.5 | all 47 / 14 have no price | same no-quote group | strong | null |
# MAGIC | `estimated_revenue_l365d` | 14.8 / 10.5 | missing exactly when price is missing | computed from price | exact | not a feature (leak) |
# MAGIC | `license` | 20.0 / 100 | New Orleans: 42.7% missing for monthly listings vs 1.8% for the rest; 220 "Exempt" | Asheville: not published; New Orleans: monthly rentals likely outside the short-term permit regime (hypothesis, only the correlation is shown) | moderate | not used |
# MAGIC | `host_since`, `instant_bookable`, `host_response_rate`, `host_acceptance_rate` | 100 / 100 | every row | not populated in this Inside Airbnb release | certain | `host_years` and `instant_bookable` dropped from `FEATURES_NUM` |

# COMMAND ----------

# MAGIC %md ## 3. Distributions of price and availability

# COMMAND ----------

q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995]
display(pdf.groupby("city").price.median().round(2).reset_index())
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
# The exact p99.5 threshold per city, as computed by silver
display(spark.table(T("silver_quality")).where(F.col("city").isin(TRAIN_CITIES))
        .select("city", "price_p995", "share_monthly", "share_price_missing"))
# 30 most expensive priced listings per city (not only extreme_high), with price per guest to judge them
top = pdf[pdf.price.notna()].assign(price_per_guest=lambda d: (d.price / d.accommodates).round(1))
display(top.sort_values("price", ascending=False).groupby("city").head(30)
        [["city", "id", "name", "room_type", "accommodates", "bedrooms", "price", "price_per_guest",
          "minimum_nights", "availability_365", "number_of_reviews", "price_flag"]]
        .sort_values(["city", "price"], ascending=[True, False]))

# Two price scales: listings with minimum_nights >= 30 are quoted at a monthly rate per night
display(pdf.groupby(["city", "is_monthly"]).price.agg(["size", "median"]).round(1).reset_index())

# Within-group outliers: log price outside 1.5×IQR of its city × room type
g = pdf.dropna(subset=["log_price"]).groupby(["city", "room_type"]).log_price
iqr_lo, iqr_hi = g.transform(lambda s: s.quantile(.25) - 1.5 * (s.quantile(.75) - s.quantile(.25))), \
                 g.transform(lambda s: s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25)))
lp = pdf.dropna(subset=["log_price"])
print("share outside IQR fences per city:",
      ((lp.log_price < iqr_lo) | (lp.log_price > iqr_hi)).groupby(lp.city).mean().round(3).to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Are the most expensive prices real? Verdict on the silver rules
# MAGIC **Top 30 per city.** I reviewed them by hand, using price per guest, reviews, minimum nights and capacity as the main signals.
# MAGIC - New Orleans: about 13 look like garbage or not a nightly price, 10 look real, 7 are unclear. All 30 are `extreme_high` (p99.5 ≈ $2,000, see `silver_quality` above).
# MAGIC - Asheville: 5 look like garbage, 15 look real, 10 are unclear. 12 are `extreme_high` (p99.5 ≈ $2,070).
# MAGIC
# MAGIC **Garbage or not a nightly price:**
# MAGIC - New Orleans "Sweet White Penthouse": 4 guests, $8,182/night, 30-night minimum, a monthly-priced quote.
# MAGIC - Two Uptown units: 6 guests, $4,883 each, 30-night minimum.
# MAGIC - New Orleans "Royal Street Hotel Room": 2 guests, $4,764, 0 reviews.
# MAGIC - Asheville "Massive 4 bedroom home": $11,680, 0 reviews.
# MAGIC
# MAGIC **Real:** whole-property group rentals with reviews and ≤ $250 per guest. Examples: "Burgundy Cottages" (18 bedrooms, $3,897) and Asheville "Secluded MtnTop Luxe" (16 guests, $2,251, 98 reviews).
# MAGIC
# MAGIC **Passes the rule but looks wrong:** Asheville "Edge of West" (2 guests, $2,000, `ok`).
# MAGIC
# MAGIC | rule | New Orleans | Asheville | verdict |
# MAGIC |---|---|---|---|
# MAGIC | `too_low` (< $10) | 0 rows (min $20) | 0 rows (min $22) | harmless, never fires; keep it as a guard for a new city |
# MAGIC | `extreme_high` (> p99.5) | 31 rows; ~12 look real (≥ 8 guests or ≥ 4 bedrooms, with reviews); 6 are monthly-priced | 12 rows; ~5 look real | reasonable as a fit exclusion (0.5% of rows). It mixes real group houses with junk, and it lets small listings at ~$2,000 through. Report `mae_*_ok` next to the full MAE |
# MAGIC | `is_monthly` (min nights ≥ 30) | 2,908 rows (40.0% of all, 40.8% of those with a known minimum stay), 2,024 priced; 32.8% of the fitting rows | 245 rows (8.6%), 165 priced; 6.5% of the fitting rows | the flag is right, but these prices are a monthly rate per night (median $111 vs $247). The baseline (room type × guests) cannot see this. Fit/evaluate nightly listings separately, or at least report MAE for each group |

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
# MAGIC | `review_scores_value` | excluded (precaution) | guests rate *value for money*, i.e. quality relative to the price they paid; but its correlation with price is only 0.03 (below) |
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leakage review: results
# MAGIC - **Revenue leak, confirmed exactly.** `estimated_revenue_l365d / (price × estimated_occupancy_l365d)` has median 1.000, and 100% of 6,464 listings with occupancy > 0 are within ±1%. Revenue is also missing exactly when price is missing.
# MAGIC   - Its Spearman correlation with log price is only 0.44, because occupancy varies. **A correlation table alone would not have caught this leak**: it is a leak by construction.
# MAGIC - **Undocumented copies of the price.** `price_quote_price_per_night` = `price` in 100% of rows (Spearman 1.000), and `price_quote_total_price` = price × quoted nights. Silver does not read them. They must never be selected as features.
# MAGIC - **No other column has |rho| > 0.8 with log price.** The strongest are the physical size columns: accommodates 0.63, bedrooms 0.61, bathrooms 0.60, beds 0.60. Next come `is_monthly` −0.45 and `minimum_nights` −0.33, which reflect the monthly price definition, not leakage.
# MAGIC - **`number_of_reviews_ltm` (a feature)** has Spearman 0.955 with `estimated_occupancy_l365d`, which we exclude because booking volume partly results from price. Keeping one and excluding the other is inconsistent; the model owner decides.
# MAGIC - **`review_scores_value`** has Spearman 0.03 with log price. We keep it excluded as a precaution.
# MAGIC - **`host_years` and `instant_bookable`** are 100% null in both snapshots and were dropped from the model features. They also cause the `ConstantInputWarning` above, which is harmless. The undocumented `hosts_time_as_host_years` (0% missing) is a possible replacement for `host_years`.
