# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Price model — train on A + B, evaluate once on the held-out test set
# MAGIC
# MAGIC 1. The 80/20 split was fixed in gold (`split` column) **before** any modelling.
# MAGIC 2. Baseline = median price by room type × accommodates on the training rows (fallback: room type, then global).
# MAGIC 3. Model choice uses 5-fold cross-validation **on the training rows only**.
# MAGIC 4. The chosen model is evaluated once on the test rows. Every reported number comes from the test set.
# MAGIC 5. The model is registered in Unity Catalog with aliases `week1` (frozen, used for the first city C run) and `champion`.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("run_label", "week1")
run_label = dbutils.widgets.get("run_label").strip()

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pdf = spark.table(T("gold_features")).where(F.col("city").isin(TRAIN_CITIES)).toPandas()
assert set(pdf.city.unique()) == set(TRAIN_CITIES), pdf.city.unique()

priced = pdf[pdf.price.notna()].copy()
train = priced[priced.split == "train"].copy()
test = priced[priced.split == "test"].copy()
fit_df = train[train.price_flag == "ok"]            # fit without missing / <$10 / >p99.5 prices
print(f"train={len(train)} (fit on {len(fit_df)}), test={len(test)}")
print(test.groupby("city").size())

# COMMAND ----------

# --- baseline (training rows only) ---
lookup = fit_baseline(train)
test["baseline"] = apply_baseline(test, lookup)

# --- candidate models, compared by CV on training rows only ---
def onehot():
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)

def make_ridge(alpha):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), FEATURES_NUM),
        ("cat", onehot(), FEATURES_CAT)])
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("m", Ridge(alpha=alpha))]), func=np.log, inverse_func=np.exp)

def make_hgb(lr, leaves):
    pre = ColumnTransformer([("num", "passthrough", FEATURES_NUM), ("cat", onehot(), FEATURES_CAT)])
    m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=600,
                                      l2_regularization=1.0, early_stopping=True, random_state=42)
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("m", m)]), func=np.log, inverse_func=np.exp)

candidates = {
    "ridge_a1": make_ridge(1.0),
    "ridge_a10": make_ridge(10.0),
    "hgb_lr0.05_l15": make_hgb(0.05, 15),
    "hgb_lr0.05_l31": make_hgb(0.05, 31),
    "hgb_lr0.1_l31": make_hgb(0.1, 31),
}

X_fit, y_fit = prepare_X(fit_df), fit_df.price.to_numpy()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_res = {name: -cross_val_score(m, X_fit, y_fit, cv=cv, scoring="neg_mean_absolute_error").mean()
          for name, m in candidates.items()}
cv_table = pd.DataFrame(sorted(cv_res.items(), key=lambda kv: kv[1]), columns=["model", "cv_mae"])
display(cv_table)
best_name = cv_table.model.iloc[0]
print("chosen:", best_name)

# COMMAND ----------

# --- fit the chosen model, evaluate ONCE on the test set ---
user = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/airbnb_price_model")
mlflow.set_registry_uri("databricks-uc")

model = candidates[best_name].fit(X_fit, y_fit)
X_test = prepare_X(test)
test["pred"] = model.predict(X_test)

def scope_rows(scope, d):
    ok = d[d.price_flag == "ok"]
    return {"city": scope, "eval_set": "test", "n": len(d), "n_ok": len(ok),
            "mae_model": mae(d.price, d.pred), "mae_baseline": mae(d.price, d.baseline),
            "mae_model_ok": mae(ok.price, ok.pred), "mae_baseline_ok": mae(ok.price, ok.baseline),
            "median_ae_model": float(np.median(np.abs(d.price - d.pred))),
            "median_price": float(d.price.median())}

rows = [scope_rows(c, test[test.city == c]) for c in TRAIN_CITIES] + [scope_rows("all_train", test)]
metrics = pd.DataFrame(rows)
display(metrics)

with mlflow.start_run(run_name=f"{run_label}_{best_name}") as run:
    mlflow.log_params({"model": best_name, "features_num": ",".join(FEATURES_NUM),
                       "features_cat": ",".join(FEATURES_CAT), "train_cities": ",".join(TRAIN_CITIES),
                       "n_fit": len(fit_df), "n_test": len(test)})
    for r in rows:
        mlflow.log_metrics({f"{r['city']}_mae_model": r["mae_model"], f"{r['city']}_mae_baseline": r["mae_baseline"]})
    info = mlflow.sklearn.log_model(model, "model", signature=infer_signature(X_fit, model.predict(X_fit)),
                                    registered_model_name=MODEL_NAME)

client = MlflowClient()
version = getattr(info, "registered_model_version", None) or max(
    int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
version = str(version)
client.set_registered_model_alias(MODEL_NAME, "champion", version)
if run_label == "week1":
    client.set_registered_model_alias(MODEL_NAME, "week1", version)
print(f"Registered {MODEL_NAME} v{version}")

# COMMAND ----------

# --- persist: metrics, baseline lookup, feature importance ---
m_sdf = (spark.createDataFrame(metrics)
         .withColumn("run_label", F.lit(run_label)).withColumn("model_version", F.lit(version))
         .withColumn("model_name", F.lit(best_name)).withColumn("created_at", F.current_timestamp()))
city_list = ", ".join(f"'{c}'" for c in TRAIN_CITIES + ["all_train"])
write_partition(m_sdf, T("gold_model_metrics"), f"run_label = '{run_label}' AND city IN ({city_list})")

lk = spark.createDataFrame(lookup.assign(model_version=version),
                           "level string, room_type string, accommodates double, median_price double, n long, model_version string")
write_partition(lk, T("gold_baseline_lookup"), f"model_version = '{version}'", partition_by=("model_version",))

sample = test.sample(min(len(test), 2000), random_state=42)
pi = permutation_importance(model, prepare_X(sample), sample.price.to_numpy(),
                            scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
imp = (pd.DataFrame({"feature": FEATURES_NUM + FEATURES_CAT, "mae_increase": pi.importances_mean})
         .sort_values("mae_increase", ascending=False))
display(imp)
write_partition(spark.createDataFrame(imp).withColumn("model_version", F.lit(version)),
                T("gold_feature_importance"), f"model_version = '{version}'", partition_by=("model_version",))
