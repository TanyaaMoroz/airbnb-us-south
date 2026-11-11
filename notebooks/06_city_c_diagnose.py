# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · City C — what happened on the first run, and why
# MAGIC
# MAGIC **When:** after Job 3 has run on the new city with the frozen `week1` model (`run_label = c_first_run`).
# MAGIC
# MAGIC **What it does (no decisions needed):**
# MAGIC 1. Checks how far the first run got (rows in every layer).
# MAGIC 2. Puts the three required numbers side by side.
# MAGIC 3. Compares the new city with the training cities (prices, sizes, rules, categories the model never saw).
# MAGIC 4. Breaks the error down by room type, size, price band, distance, monthly rentals, property type.
# MAGIC 5. Prints a **draft report** with the numbers filled in — copy it into the README and the presentation.
# MAGIC
# MAGIC It only writes one table, `gold_c_error_breakdown`, for the dashboard. It never changes the model.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "nashville")
dbutils.widgets.text("run_label", "c_first_run")
city = validate_city(dbutils.widgets.get("city"))
run_label = dbutils.widgets.get("run_label").strip()
assert city not in TRAIN_CITIES, "This notebook is for the new city, not a training city"


def exists(name):
    return spark.catalog.tableExists(T(name))

# COMMAND ----------

# MAGIC %md ## 1. Did the first run work? Rows in every layer

# COMMAND ----------

status = []
for t in ["bronze_listings", "silver_listings", "gold_features"]:
    status.append((t, spark.table(T(t)).where(F.col("city") == city).count() if exists(t) else 0))
n_pred = (spark.table(T("gold_predictions"))
          .where((F.col("city") == city) & (F.col("run_label") == run_label)).count()
          if exists("gold_predictions") else 0)
status.append((f"gold_predictions [{run_label}]", n_pred))
status_df = pd.DataFrame(status, columns=["layer", "rows"])
display(status_df)

if exists("silver_quality"):
    display(spark.table(T("silver_quality")).where(F.col("city").isin(TRAIN_CITIES + [city])))

assert n_pred > 0, (
    "There are no predictions for this run: the first run stopped before scoring. "
    "That is a valid first-run result. Open the Job 3 run, note which task failed and the error message, "
    "then fix the pipeline and rerun before running this notebook.")

# COMMAND ----------

# MAGIC %md ## 2. The three numbers

# COMMAND ----------

m = spark.table(T("gold_model_metrics")).toPandas()
ab_row = m[(m.city == "all_train") & (m.run_label == "week1")].sort_values("created_at").iloc[-1]
c_row = m[(m.city == city) & (m.run_label == run_label) & (m.eval_set == "all_priced")].sort_values("created_at").iloc[-1]

three = pd.DataFrame([
    {"number": f"Baseline on {city} (medians learned on A+B)", "mae_usd": round(c_row.mae_baseline)},
    {"number": "Model on A+B (test set)", "mae_usd": round(ab_row.mae_model)},
    {"number": f"Model on {city}, first run", "mae_usd": round(c_row.mae_model)},
])
display(three)

# COMMAND ----------

# MAGIC %md ## 3. How different is the new city from the training cities?

# COMMAND ----------

feat = spark.table(T("gold_features")).where(F.col("city").isin(TRAIN_CITIES + [city])).toPandas()
tr = feat[feat.city.isin(TRAIN_CITIES) & (feat.split == "train") & feat.price.notna()]
cf = feat[feat.city == city]

max_acc = tr.accommodates.max()
pt_counts = tr.property_type.value_counts()
frequent_pt = set(pt_counts[pt_counts >= 20].index)


def profile(d):
    pr = d[d.price.notna()]
    return {
        "listings": len(d),
        "listings_with_price": len(pr),
        "median_price": pr.price.median(),
        "p90_price": pr.price.quantile(0.9),
        "share_entire_home": (d.room_type == "Entire home/apt").mean(),
        "median_accommodates": d.accommodates.median(),
        "p95_accommodates": d.accommodates.quantile(0.95),
        "share_accommodates_above_training_max": (d.accommodates > max_acc).mean(),
        "share_min_nights_30_plus": (d.minimum_nights >= 30).mean(),
        "median_dist_center_km": d.dist_center_km.median(),
        "share_property_type_rare_or_unseen": (~d.property_type.isin(frequent_pt)).mean(),
        "share_bedrooms_missing": d.bedrooms.isna().mean(),
    }


shift = pd.DataFrame({**{c: profile(tr[tr.city == c]) for c in TRAIN_CITIES}, city: profile(cf)}).round(3)
display(shift.reset_index().rename(columns={"index": "measure"}))
print(f"Largest 'accommodates' seen in training: {max_acc}")

# COMMAND ----------

# MAGIC %md ## 4. Where does the model go wrong? (error = predicted − actual; negative = under-priced)

# COMMAND ----------

p = (spark.table(T("gold_predictions"))
     .where(((F.col("city") == city) & (F.col("run_label") == run_label)) |
            (F.col("city").isin(TRAIN_CITIES) & (F.col("run_label") == "week1")))
     .where("in_evaluation = 1")
     .toPandas())
p = p.merge(feat[["id", "city", "dist_center_km", "minimum_nights"]], on=["id", "city"], how="left")
p["group"] = np.where(p.city == city, city, "A+B test")

p["accommodates_bucket"] = pd.cut(p.accommodates, [0, 2, 4, 6, 8, 12, 1000],
                                  labels=["1-2", "3-4", "5-6", "7-8", "9-12", "13+"]).astype(str)
p["price_band"] = pd.cut(p.price, [0, 100, 200, 300, 500, 1000, np.inf],
                         labels=["<$100", "$100-199", "$200-299", "$300-499", "$500-999", "$1000+"],
                         right=False).astype(str)
p["distance_band"] = pd.cut(p.dist_center_km, [0, 2, 5, 10, 20, np.inf],
                            labels=["<2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"], right=False).astype(str)
p["monthly_rental"] = np.where(p.minimum_nights >= 30, "30+ nights", "short stay")
p["property_type_group"] = np.where(p.property_type.isin(frequent_pt), "common in training", "rare or unseen")

dims = ["room_type", "accommodates_bucket", "price_band", "distance_band", "monthly_rental", "property_type_group"]
parts = []
for dim in dims:
    g = (p.groupby(["group", dim], dropna=False)
          .agg(n=("price", "size"), mae=("abs_error", "mean"),
               mean_error=("error", "mean"), median_price=("price", "median"))
          .reset_index().rename(columns={dim: "bucket"}))
    g["dimension"] = dim
    g["bucket"] = g["bucket"].astype(str)
    parts.append(g)
bd = pd.concat(parts, ignore_index=True)[["dimension", "bucket", "group", "n", "mae", "mean_error", "median_price"]]
bd[["mae", "mean_error", "median_price"]] = bd[["mae", "mean_error", "median_price"]].round(1)

for dim in dims:
    print(dim)
    display(bd[bd.dimension == dim].sort_values(["group", "bucket"]))

(spark.createDataFrame(bd.assign(run_label=run_label))
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(T("gold_c_error_breakdown")))

# COMMAND ----------

# MAGIC %md ## 5. Draft report (copy into README and slides)

# COMMAND ----------

cm, abm = p[p.group == city], p[p.group == "A+B test"]
c_mae, ab_mae, c_base = float(c_row.mae_model), float(ab_row.mae_model), float(c_row.mae_baseline)
lvl_c = float(np.median(cm.price / cm.predicted_price))
lvl_ab = float(np.median(abm.price / abm.predicted_price))
bias_c = float(cm.error.mean())
scaled_mae = mae(cm.price, cm.predicted_price * lvl_c)
med_c = float(cm.price.median())
med_ab = float(abm.price.median())


def worst(dim):
    d = bd[(bd.dimension == dim) & (bd.group == city) & (bd.n >= 30)].sort_values("mae", ascending=False)
    return d.iloc[0] if len(d) else None


w_rt, w_acc = worst("room_type"), worst("accommodates_bucket")
sp = shift[city]

lines = [
    f"### {city}: first run with the unchanged week-1 pipeline and model",
    "",
    f"- **Did it run?** Bronze {status_df['rows'].iloc[0]:,} rows, silver {status_df['rows'].iloc[1]:,}, "
    f"gold {status_df['rows'].iloc[2]:,}, predictions {status_df['rows'].iloc[3]:,}.",
    f"- **Three numbers:** baseline on {city} ${c_base:.0f}; model on A+B ${ab_mae:.0f}; "
    f"model on {city} ${c_mae:.0f} ({c_mae / ab_mae:.1f}× the error on the training cities).",
    ("- The model is still **better** than the simple baseline on the new city."
     if c_mae < c_base else
     "- On the new city the model is **worse than the simple baseline** — it did not transfer."),
    f"- **Price level:** median price ${med_c:.0f} vs ${med_ab:.0f} in the training test sets. "
    f"On the new city, actual / predicted price has a median of {lvl_c:.2f} (training cities: {lvl_ab:.2f}), "
    f"so the model {'under' if lvl_c > 1 else 'over'}-prices a typical listing by about {abs(lvl_c - 1) * 100:.0f}%.",
    f"- **Direction:** average error ${bias_c:+.0f} per night "
    f"({'systematic under-pricing' if bias_c < 0 else 'systematic over-pricing'}).",
    f"- **Diagnostic, not a result:** scaling every prediction by {lvl_c:.2f} would cut MAE from ${c_mae:.0f} "
    f"to ${scaled_mae:.0f} on these same listings — the share of the gap that is just the price level.",
    f"- **Never seen in training:** {sp['share_accommodates_above_training_max'] * 100:.1f}% of listings sleep more "
    f"than {max_acc:.0f} guests (the training maximum); {sp['share_property_type_rare_or_unseen'] * 100:.1f}% have a "
    f"property type that was rare or absent in training.",
    f"- **Rules:** {sp['share_min_nights_30_plus'] * 100:.1f}% are 30+ night rentals in {city}, vs "
    + ", ".join(f"{shift[c]['share_min_nights_30_plus'] * 100:.1f}% in {c}" for c in TRAIN_CITIES)
    + " — `minimum_nights` is the second most important feature, so a different mix changes its meaning.",
]
if w_rt is not None:
    lines.append(f"- **Worst room type:** {w_rt.bucket} (MAE ${w_rt.mae:.0f}, average error ${w_rt.mean_error:+.0f}, n={w_rt['n']}).")
if w_acc is not None:
    lines.append(f"- **Worst size group:** {w_acc.bucket} guests (MAE ${w_acc.mae:.0f}, average error ${w_acc.mean_error:+.0f}, n={w_acc['n']}).")
print("\n".join(lines))
