# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 · Score one city — predictions for every listing → `gold_predictions`
# MAGIC
# MAGIC **Parameters**
# MAGIC - `city` — any city that has been through 01–03
# MAGIC - `model_alias` — `champion` (default) or `week1` (the frozen end-of-week-1 model, for the first city C run)
# MAGIC - `run_label` — e.g. `week1`, `c_first_run`, `c_after_fix`; predictions and metrics are kept per label
# MAGIC
# MAGIC For training cities the error is measured on the `test` split only. For any other city, every listing
# MAGIC with a price is unseen, so the error is measured on all of them (`eval_set = all_priced`).

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "new-orleans")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("run_label", "week1")
city = validate_city(dbutils.widgets.get("city"))
alias = dbutils.widgets.get("model_alias").strip()
run_label = dbutils.widgets.get("run_label").strip()

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, alias).version)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
lookup = spark.table(T("gold_baseline_lookup")).where(F.col("model_version") == version).toPandas()
print(f"{city}: model v{version} (@{alias}), run_label={run_label}")

pdf = spark.table(T("gold_features")).where(F.col("city") == city).toPandas()
assert len(pdf) > 0, f"No gold_features rows for {city}"

# COMMAND ----------

pdf["predicted_price"] = model.predict(prepare_X(pdf))
pdf["baseline_price"] = apply_baseline(pdf, lookup)
pdf["error"] = pdf.predicted_price - pdf.price            # > 0: model over-prices, < 0: under-prices
pdf["abs_error"] = pdf.error.abs()
pdf["pct_error"] = pdf.error / pdf.price

is_train_city = city in TRAIN_CITIES
eval_mask = pdf.price.notna() & ((pdf.split == "test") if is_train_city else True)
pdf["in_evaluation"] = eval_mask.astype(int)

out_cols = ["id", "city", "snapshot_date", "split", "in_evaluation", "name", "neighbourhood", "room_type",
            "property_type", "accommodates", "bedrooms", "latitude", "longitude", "price", "price_flag",
            "is_monthly", "predicted_price", "baseline_price", "error", "abs_error", "pct_error"]
preds = (spark.createDataFrame(pdf[out_cols].astype({"accommodates": "float64", "bedrooms": "float64"}))
         .withColumn("run_label", F.lit(run_label))
         .withColumn("model_version", F.lit(version))
         .withColumn("scored_at", F.current_timestamp()))
write_partition(preds, T("gold_predictions"), f"city = '{city}' AND run_label = '{run_label}'")

# COMMAND ----------

# New (non-training) city: record its error. Training-city metrics are written by 04_train_model.
d = pdf[eval_mask]
ok = d[d.price_flag == "ok"]
summary = pd.DataFrame([{
    "city": city, "eval_set": "test" if is_train_city else "all_priced", "n": len(d), "n_ok": len(ok),
    "mae_model": mae(d.price, d.predicted_price), "mae_baseline": mae(d.price, d.baseline_price),
    "mae_model_ok": mae(ok.price, ok.predicted_price), "mae_baseline_ok": mae(ok.price, ok.baseline_price),
    "median_ae_model": float(np.median(d.abs_error)) if len(d) else float("nan"),
    "median_price": float(d.price.median()) if len(d) else float("nan")}])
display(summary)

if not is_train_city:
    m_sdf = (spark.createDataFrame(summary)
             .withColumn("run_label", F.lit(run_label)).withColumn("model_version", F.lit(version))
             .withColumn("model_name", F.lit(alias)).withColumn("created_at", F.current_timestamp()))
    write_partition(m_sdf, T("gold_model_metrics"), f"run_label = '{run_label}' AND city = '{city}'")