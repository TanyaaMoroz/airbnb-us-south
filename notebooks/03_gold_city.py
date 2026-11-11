# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — metrics and model features for one city
# MAGIC
# MAGIC Writes (all partitioned by `city`):
# MAGIC - `gold_city_metrics` — one row per city (dashboard page 1 and 2, Genie)
# MAGIC - `gold_room_type_mix` — city × room type
# MAGIC - `gold_neighbourhood_metrics` — city × neighbourhood
# MAGIC - `gold_features` — one row per listing with model features and a fixed train/test split

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "new-orleans")
city = validate_city(dbutils.widgets.get("city"))
pred = f"city = '{city}'"

from pyspark.sql import Window

s = spark.table(T("silver_listings")).where(F.col("city") == city)
assert s.limit(1).count() == 1, f"No silver rows for {city}"

# COMMAND ----------

# --- metrics ---
city_metrics = s.groupBy("city", "snapshot_date").agg(
    F.count("*").alias("listings"),
    F.countDistinct("host_id").alias("hosts"),
    F.sum(F.col("price").isNotNull().cast("int")).alias("listings_with_price"),
    F.percentile_approx("price", 0.5).alias("median_price"),
    F.avg("availability_365").alias("avg_availability_365"),
    F.avg((F.col("availability_365") == 0).cast("double")).alias("share_unavailable_365"),
    F.percentile_approx("review_scores_rating", 0.5).alias("median_review_score"),
    F.avg((F.col("number_of_reviews") == 0).cast("double")).alias("share_no_reviews"),
    F.avg((F.col("room_type") == "Entire home/apt").cast("double")).alias("share_entire_home"),
    F.avg((F.col("calculated_host_listings_count") > 1).cast("double")).alias("share_multi_listing_hosts"),
    F.avg(F.col("is_monthly").cast("double")).alias("share_monthly"),
    F.percentile_approx("accommodates", 0.5).alias("median_accommodates"),
)
write_partition(city_metrics, T("gold_city_metrics"), pred)

w = Window.partitionBy("city")
room_mix = (s.groupBy("city", "room_type")
              .agg(F.count("*").alias("listings"), F.percentile_approx("price", 0.5).alias("median_price"))
              .withColumn("share", F.col("listings") / F.sum("listings").over(w)))
write_partition(room_mix, T("gold_room_type_mix"), pred)

neigh = s.groupBy("city", "neighbourhood").agg(
    F.count("*").alias("listings"),
    F.percentile_approx("price", 0.5).alias("median_price"),
    F.avg("review_scores_rating").alias("avg_review_score"),
    F.avg((F.col("room_type") == "Entire home/apt").cast("double")).alias("share_entire_home"),
    F.avg("availability_365").alias("avg_availability_365"),
    F.avg("latitude").alias("latitude"),
    F.avg("longitude").alias("longitude"),
)
write_partition(neigh, T("gold_neighbourhood_metrics"), pred)

# COMMAND ----------

# --- features ---
# Location as distance (km) to the city's own median listing location: means the same thing in any city,
# unlike neighbourhood names, which are different in every city.
c = s.agg(F.percentile_approx("latitude", 0.5).alias("lat"), F.percentile_approx("longitude", 0.5).alias("lon")).first()
lat1, lon1 = F.radians("latitude"), F.radians("longitude")
lat2, lon2 = F.radians(F.lit(c["lat"])), F.radians(F.lit(c["lon"]))
a = F.sin((lat2 - lat1) / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin((lon2 - lon1) / 2) ** 2
dist_km = 2 * 6371.0 * F.asin(F.sqrt(a))

# Deterministic 80/20 split per listing (hash of id): fixed before any model choice and reproducible on re-runs.
bucket = F.conv(F.substring(F.md5(F.col("id")), 1, 8), 16, 10).cast("long") % 100

features = (s.withColumn("dist_center_km", dist_km)
             .withColumn("split", F.when(bucket < 20, "test").otherwise("train"))
             .select("id", "city", "snapshot_date", "split", "price", "price_flag", "is_monthly",
                     "neighbourhood", "latitude", "longitude", "name",
                     *[f for f in FEATURES_NUM + FEATURES_CAT if f not in ("dist_center_km",)], "dist_center_km"))
write_partition(features, T("gold_features"), pred)

display(spark.table(T("gold_features")).where(pred).groupBy("split", "price_flag").count().orderBy("split", "price_flag"))
