# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — clean and typed listings for one city
# MAGIC
# MAGIC One row = one listing on the snapshot date. Every transformation below is listed with its reason
# MAGIC in the EDA notebook (section 5). All casts use `try_cast`: serverless runs in ANSI mode, where a
# MAGIC plain `cast` of an unexpected string fails the whole job instead of producing a null.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "new-orleans")
city = validate_city(dbutils.widgets.get("city"))

from functools import reduce

b = spark.table(T("bronze_listings")).where(F.col("city") == city)
n_bronze = b.count()
assert n_bronze > 0, f"No bronze rows for {city}: run 01_bronze_ingest first"
cols = set(b.columns)


def raw(name):
    return F.col(f"`{name}`") if name in cols else F.lit(None).cast("string")

def to_num(name, t="double"):
    return F.expr(f"try_cast(trim(`{name}`) as {t})") if name in cols else F.lit(None).cast(t)

def to_bool(name):
    v = F.lower(F.trim(raw(name)))
    return F.when(v == "t", 1).when(v == "f", 0).cast("int")

def to_pct(name):
    return (F.expr(f"try_cast(regexp_replace(`{name}`, '%', '') as double)") / 100) if name in cols else F.lit(None).cast("double")

def to_date(name):
    return F.expr(f"to_date(try_to_timestamp(trim(`{name}`), 'yyyy-MM-dd'))") if name in cols else F.lit(None).cast("date")

# price: "$1,234.00" -> 1234.0 ; anything unparseable -> null
price = (F.expr("try_cast(nullif(regexp_replace(price, '[^0-9.]', ''), '') as double)")
         if "price" in cols else F.lit(None).cast("double"))

# bathrooms: numeric column when present, otherwise parsed from text ("1.5 shared baths", "Half-bath")
bt = F.lower(raw("bathrooms_text"))
bt_num = F.regexp_extract(bt, r"(\d+(?:\.\d+)?)", 1)
bath_from_text = F.when(bt_num != "", bt_num.cast("double")).when(bt.contains("half"), F.lit(0.5))

# amenities: JSON array of strings
am = F.lower(raw("amenities"))
am_arr = F.from_json(raw("amenities"), "array<string>")

def has_amenity(*patterns):
    cond = reduce(lambda x, y: x | y, [am.rlike(p) for p in patterns])
    return F.when(am.isNull(), None).otherwise(cond.cast("int"))

snap = F.to_date(F.col("snapshot_date"))

s = (b.select(
        F.trim(raw("id")).alias("id"),
        "city", "snapshot_date",
        raw("name").alias("name"),
        F.trim(raw("host_id")).alias("host_id"),
        F.trim(raw("neighbourhood_cleansed")).alias("neighbourhood"),
        to_num("latitude").alias("latitude"),
        to_num("longitude").alias("longitude"),
        F.trim(raw("property_type")).alias("property_type"),
        F.trim(raw("room_type")).alias("room_type"),
        to_num("accommodates", "int").alias("accommodates"),
        F.coalesce(to_num("bathrooms"), bath_from_text).alias("bathrooms"),
        bt.contains("shared").cast("int").alias("bath_shared"),
        to_num("bedrooms").alias("bedrooms"),
        to_num("beds").alias("beds"),
        price.alias("price"),
        to_num("minimum_nights", "int").alias("minimum_nights"),
        to_num("maximum_nights", "int").alias("maximum_nights"),
        to_bool("has_availability").alias("has_availability"),
        to_num("availability_30", "int").alias("availability_30"),
        to_num("availability_90", "int").alias("availability_90"),
        to_num("availability_365", "int").alias("availability_365"),
        to_num("number_of_reviews", "int").alias("number_of_reviews"),
        to_num("number_of_reviews_ltm", "int").alias("number_of_reviews_ltm"),
        to_num("reviews_per_month").alias("reviews_per_month"),
        to_date("first_review").alias("first_review"),
        to_date("last_review").alias("last_review"),
        to_num("review_scores_rating").alias("review_scores_rating"),
        to_num("review_scores_accuracy").alias("review_scores_accuracy"),
        to_num("review_scores_cleanliness").alias("review_scores_cleanliness"),
        to_num("review_scores_checkin").alias("review_scores_checkin"),
        to_num("review_scores_communication").alias("review_scores_communication"),
        to_num("review_scores_location").alias("review_scores_location"),
        to_num("review_scores_value").alias("review_scores_value"),        # kept for EDA, not a model feature
        to_date("host_since").alias("host_since"),
        to_pct("host_response_rate").alias("host_response_rate"),
        to_pct("host_acceptance_rate").alias("host_acceptance_rate"),
        to_bool("host_is_superhost").alias("host_is_superhost"),
        to_bool("instant_bookable").alias("instant_bookable"),
        to_num("calculated_host_listings_count", "int").alias("calculated_host_listings_count"),
        raw("license").alias("license"),
        to_num("estimated_occupancy_l365d").alias("estimated_occupancy_l365d"),  # leakage review only
        to_num("estimated_revenue_l365d").alias("estimated_revenue_l365d"),      # LEAK: computed from price
        F.when(am_arr.isNull(), None).otherwise(F.size(am_arr)).alias("amenities_count"),
        has_amenity(r"\bpool\b(?! table)").alias("has_pool"),
        has_amenity(r"hot tub").alias("has_hot_tub"),
        has_amenity(r"free[^\"]*parking").alias("has_free_parking"),
        has_amenity(r"\bwasher\b").alias("has_washer"),
        has_amenity(r"air conditioning", r"central air").alias("has_ac"),
        snap.alias("snapshot_dt"),
     )
     .where(F.col("id").isNotNull() & (F.col("id") != ""))
     .dropDuplicates(["id"])
     .withColumn("host_years", F.datediff("snapshot_dt", "host_since") / 365.25)
     .withColumn("days_since_last_review", F.datediff("snapshot_dt", "last_review"))
     .withColumn("accommodates", F.when(F.col("accommodates") > 0, F.col("accommodates")))
     .withColumn("is_monthly", (F.col("minimum_nights") >= 30).cast("int"))
     .withColumn("room_type_known", F.col("room_type").isin(EXPECTED_ROOM_TYPES).cast("int"))
)

# Price outlier flag. The threshold is computed on this city's own data (a data-quality flag, not a model input).
p995 = s.where("price is not null").agg(F.percentile_approx("price", 0.995).alias("p")).first()["p"]
s = s.withColumn(
    "price_flag",
    F.when(F.col("price").isNull(), "missing")
     .when(F.col("price") < 10, "too_low")
     .when(F.lit(p995).isNotNull() & (F.col("price") > F.lit(p995)), "extreme_high")
     .otherwise("ok"))

write_partition(s, T("silver_listings"), f"city = '{city}'")

# COMMAND ----------

# Data-quality record per city: what the pipeline saw, so a new city's surprises are visible, not silent.
sv = spark.table(T("silver_listings")).where(F.col("city") == city)
n_silver = sv.count()
dq = sv.agg(
    F.lit(city).alias("city"),
    F.lit(n_bronze).cast("long").alias("n_bronze"),
    F.lit(n_silver).cast("long").alias("n_silver"),
    F.avg(F.col("price").isNull().cast("double")).alias("share_price_missing"),
    F.avg(F.col("accommodates").isNull().cast("double")).alias("share_accommodates_missing"),
    F.avg(F.col("latitude").isNull().cast("double")).alias("share_latlon_missing"),
    F.avg((F.col("room_type_known") == 0).cast("double")).alias("share_unknown_room_type"),
    F.concat_ws(", ", F.collect_set(F.when(F.col("room_type_known") == 0, F.col("room_type")))).alias("unknown_room_types"),
    F.avg(F.col("is_monthly").cast("double")).alias("share_monthly"),
    F.lit(p995).cast("double").alias("price_p995"),
).withColumn("checked_at", F.current_timestamp())
write_partition(dq, T("silver_quality"), f"city = '{city}'")
display(dq)
