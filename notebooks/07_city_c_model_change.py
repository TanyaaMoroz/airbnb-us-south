# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · City C — model changes, evaluated honestly
# MAGIC
# MAGIC **When:** after `06_city_c_diagnose`. Run by the **owner of `price_model`** (registering a version needs ownership).
# MAGIC
# MAGIC **The plan, fixed before looking at any result:**
# MAGIC - The new city is split with the same hash rule as A and B: **80% adaptation**, **20% held out**.
# MAGIC   The held-out 20% is never used to fit or choose anything; every "after" number comes from it.
# MAGIC - **Change 1 — recalibration** (cheap): keep the week-1 model, multiply its predictions by one factor,
# MAGIC   the median of actual / predicted price on the adaptation part. Fixes a price level the model never saw.
# MAGIC - **Change 2 — retrain with the new city** (the final model, decided in advance): same features and
# MAGIC   hyper-parameters as week 1, trained on A + B + the adaptation part of the new city.
# MAGIC - Both are compared on the held-out 20% with the week-1 model and with the baseline. We also check that
# MAGIC   retraining did not make A and B worse.
# MAGIC
# MAGIC The `week1` alias is never moved: the first-run model stays reproducible.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "nashville")
city = validate_city(dbutils.widgets.get("city"))
assert city not in TRAIN_CITIES
RUN_LABEL = "c_after_fix"

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
week1_version = str(client.get_model_version_by_alias(MODEL_NAME, "week1").version)
week1_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@week1")
model_name = (spark.table(T("gold_model_metrics"))
              .where("run_label = 'week1' AND city = 'all_train'")
              .orderBy(F.desc("created_at")).first()["model_name"])
print(f"week1 = v{week1_version} ({model_name})")


def make_model(name):
    """Rebuild the week-1 model type with the same hyper-parameters."""
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    if name.startswith("hgb"):
        lr = float(re.search(r"lr([0-9.]+)_", name).group(1))
        leaves = int(re.search(r"_l(\d+)$", name).group(1))
        pre = ColumnTransformer([("num", "passthrough", FEATURES_NUM), ("cat", onehot, FEATURES_CAT)])
        reg = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=600,
                                            l2_regularization=1.0, early_stopping=True, random_state=42)
    else:
        alpha = float(re.search(r"_a([0-9.]+)$", name).group(1))
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), FEATURES_NUM),
            ("cat", onehot, FEATURES_CAT)])
        reg = Ridge(alpha=alpha)
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("m", reg)]), func=np.log, inverse_func=np.exp)

# COMMAND ----------

feat_all = spark.table(T("gold_features")).where(F.col("city").isin(TRAIN_CITIES + [city])).toPandas()
feat = feat_all[feat_all.price.notna()]
ab_train = feat[feat.city.isin(TRAIN_CITIES) & (feat.split == "train")]
ab_test = feat[feat.city.isin(TRAIN_CITIES) & (feat.split == "test")]
c_train = feat[(feat.city == city) & (feat.split == "train")]
c_test = feat[(feat.city == city) & (feat.split == "test")]
ab_fit, c_fit = ab_train[ab_train.price_flag == "ok"], c_train[c_train.price_flag == "ok"]
print(f"{city}: adaptation {len(c_train)} listings, held-out {len(c_test)} listings")

# baselines: the week-1 one (A+B medians) and one refitted with the adaptation part
lookup_w1 = spark.table(T("gold_baseline_lookup")).where(F.col("model_version") == week1_version).toPandas()
lookup_new = fit_baseline(pd.concat([ab_train, c_train]))
base_w1_ctest = apply_baseline(c_test, lookup_w1)
base_new_ctest = apply_baseline(c_test, lookup_new)

# week-1 model, unchanged
pred_w1_ctest = week1_model.predict(prepare_X(c_test))

# change 1: recalibration factor learned on the adaptation part only
factor = float(np.median(c_fit.price.to_numpy() / week1_model.predict(prepare_X(c_fit))))
pred_recal_ctest = pred_w1_ctest * factor

# change 2: retrain on A + B + adaptation part
fit_all = pd.concat([ab_fit, c_fit])
new_model = make_model(model_name).fit(prepare_X(fit_all), fit_all.price.to_numpy())
pred_new_ctest = new_model.predict(prepare_X(c_test))
print(f"recalibration factor = {factor:.3f}")

# COMMAND ----------

def score(label, d, pred, base):
    ok = (d.price_flag == "ok").to_numpy()
    y = d.price.to_numpy()
    return {"model": label, "n": len(d), "mae": mae(y, pred), "mae_baseline": mae(y, base),
            "mae_without_outliers": mae(y[ok], pred[ok]), "mean_error": float(np.mean(pred - y))}


results = pd.DataFrame([
    score("Baseline, A+B medians", c_test, base_w1_ctest, base_w1_ctest),
    score("Baseline, A+B+new-city medians", c_test, base_new_ctest, base_new_ctest),
    score("Week-1 model, unchanged", c_test, pred_w1_ctest, base_w1_ctest),
    score("Change 1: week-1 model x recalibration", c_test, pred_recal_ctest, base_w1_ctest),
    score("Change 2: retrained with new city (final)", c_test, pred_new_ctest, base_new_ctest),
]).round(1)
print(f"All numbers on the held-out 20% of {city} ({len(c_test)} listings), USD per night")
display(results)

# did retraining hurt the training cities?
ab_check = []
for c in TRAIN_CITIES:
    d = ab_test[ab_test.city == c]
    ab_check.append({"city": c, "mae_week1": mae(d.price, week1_model.predict(prepare_X(d))),
                     "mae_retrained": mae(d.price, new_model.predict(prepare_X(d)))})
ab_check = pd.DataFrame(ab_check).round(1)
display(ab_check)

# COMMAND ----------

# MAGIC %md ## Register the retrained model (alias `c_after_fix` and `champion`; `week1` is not moved)

# COMMAND ----------

user = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/airbnb_price_model")
X_fit = prepare_X(fit_all)
with mlflow.start_run(run_name=f"{RUN_LABEL}_{model_name}"):
    mlflow.log_params({"model": model_name, "train_cities": ",".join(TRAIN_CITIES + [city + " (80%)"]),
                       "recalibration_factor": round(factor, 4)})
    mlflow.log_metrics({f"{city}_heldout_mae_week1": mae(c_test.price, pred_w1_ctest),
                        f"{city}_heldout_mae_recalibrated": mae(c_test.price, pred_recal_ctest),
                        f"{city}_heldout_mae_retrained": mae(c_test.price, pred_new_ctest)})
    info = mlflow.sklearn.log_model(new_model, "model", signature=infer_signature(X_fit, new_model.predict(X_fit)),
                                    registered_model_name=MODEL_NAME)
new_version = str(getattr(info, "registered_model_version", None) or max(
    int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")))
client.set_registered_model_alias(MODEL_NAME, RUN_LABEL, new_version)
client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
print(f"Registered v{new_version} as @{RUN_LABEL} and @champion; @week1 stays v{week1_version}")

# COMMAND ----------

# MAGIC %md ## Write predictions and metrics for the dashboard and Genie

# COMMAND ----------

write_partition(spark.createDataFrame(lookup_new.assign(model_version=new_version),
                                      "level string, room_type string, accommodates double, median_price double, n long, model_version string"),
                T("gold_baseline_lookup"), f"model_version = '{new_version}'", partition_by=("model_version",))

cf = feat_all[feat_all.city == city].copy()
cf["predicted_price"] = new_model.predict(prepare_X(cf))
cf["baseline_price"] = apply_baseline(cf, lookup_new)
cf["error"] = cf.predicted_price - cf.price
cf["abs_error"] = cf.error.abs()
cf["pct_error"] = cf.error / cf.price
cf["in_evaluation"] = ((cf.split == "test") & cf.price.notna()).astype("Int64")
cf["is_monthly"] = cf["is_monthly"].astype("Int64")
out_cols = ["id", "city", "snapshot_date", "split", "in_evaluation", "name", "neighbourhood", "room_type",
            "property_type", "accommodates", "bedrooms", "latitude", "longitude", "price", "price_flag",
            "is_monthly", "predicted_price", "baseline_price", "error", "abs_error", "pct_error"]
preds = (spark.createDataFrame(cf[out_cols].astype({"accommodates": "float64", "bedrooms": "float64"}))
         .withColumn("run_label", F.lit(RUN_LABEL)).withColumn("model_version", F.lit(new_version))
         .withColumn("scored_at", F.current_timestamp()))
write_partition(preds, T("gold_predictions"), f"city = '{city}' AND run_label = '{RUN_LABEL}'")


def metric_row(label, city_name, eval_set, d, pred, base, version, name):
    ok = (d.price_flag == "ok").to_numpy()
    y = d.price.to_numpy()
    return {"city": city_name, "eval_set": eval_set, "n": int(len(d)), "n_ok": int(ok.sum()),
            "mae_model": mae(y, pred), "mae_baseline": mae(y, base),
            "mae_model_ok": mae(y[ok], pred[ok]), "mae_baseline_ok": mae(y[ok], base[ok]),
            "median_ae_model": float(np.median(np.abs(y - pred))), "median_price": float(np.median(y)),
            "run_label": label, "model_version": version, "model_name": name}


rows_c = [
    metric_row("c_first_run", city, "c_heldout", c_test, pred_w1_ctest, base_w1_ctest, week1_version, model_name),
    metric_row("c_recalibrated", city, "c_heldout", c_test, pred_recal_ctest, base_w1_ctest, week1_version, f"{model_name} x {factor:.3f}"),
    metric_row(RUN_LABEL, city, "c_heldout", c_test, pred_new_ctest, base_new_ctest, new_version, model_name),
]
rows_ab = [metric_row(RUN_LABEL, c, "test", ab_test[ab_test.city == c],
                      new_model.predict(prepare_X(ab_test[ab_test.city == c])),
                      apply_baseline(ab_test[ab_test.city == c], lookup_new), new_version, model_name)
           for c in TRAIN_CITIES]


def to_sdf(rows):
    return spark.createDataFrame(pd.DataFrame(rows)).withColumn("created_at", F.current_timestamp())


write_partition(to_sdf(rows_c), T("gold_model_metrics"), f"city = '{city}' AND eval_set = 'c_heldout'")
write_partition(to_sdf(rows_ab), T("gold_model_metrics"), f"run_label = '{RUN_LABEL}' AND eval_set = 'test'")

# COMMAND ----------

# MAGIC %md ## Text for the README and slides

# COMMAND ----------

r = results.set_index("model")
w1 = r.loc["Week-1 model, unchanged", "mae"]
rc = r.loc["Change 1: week-1 model x recalibration", "mae"]
rt = r.loc["Change 2: retrained with new city (final)", "mae"]
ab_row_mae = float(spark.table(T("gold_model_metrics"))
                   .where("run_label = 'week1' AND city = 'all_train'")
                   .orderBy(F.desc("created_at")).first()["mae_model"])
print(f"""### Model changes for {city} (all numbers on the held-out 20%, {len(c_test)} listings)

| # | Change | Kind | Reason | MAE on held-out {city} |
|---|---|---|---|---|
| – | Week-1 model, unchanged | – | first run | ${w1:.0f} |
| 1 | Multiply predictions by {factor:.2f} (median actual/predicted on the 80% adaptation part) | model change | the new city's price level was never seen in training | ${rc:.0f} |
| 2 | Retrain with A + B + 80% of {city}, same features and hyper-parameters | model change | sizes, property types and rules the model had not seen | ${rt:.0f} |

Baseline on held-out {city}: ${r.loc['Baseline, A+B medians', 'mae']:.0f} (A+B medians), ${r.loc['Baseline, A+B+new-city medians', 'mae']:.0f} (with new-city medians).
Model on A+B test: ${ab_row_mae:.0f}.
Training cities after retraining: """ + ", ".join(f"{row.city} ${row.mae_week1:.0f} → ${row.mae_retrained:.0f}" for row in ab_check.itertuples()) + ".")
