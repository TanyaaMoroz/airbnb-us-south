# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — land the raw `listings.csv.gz` for one city
# MAGIC
# MAGIC **Parameters**
# MAGIC - `city` — Inside Airbnb slug (`new-orleans`, `asheville`, …)
# MAGIC - `snapshot_date` — `YYYY-MM-DD`; empty = latest snapshot listed on Inside Airbnb
# MAGIC - `download` — `true` downloads the file; `false` uses a file already uploaded to
# MAGIC   `/Volumes/workspace/airbnb/raw/<city>/<date>/listings.csv.gz` (fallback if the download is blocked)
# MAGIC
# MAGIC Bronze keeps every column as a **string**, exactly as published, plus lineage columns.
# MAGIC Typing and cleaning happen in silver.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("city", "new-orleans")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.dropdown("download", "true", ["true", "false"])

city = validate_city(dbutils.widgets.get("city"))
snapshot_date = dbutils.widgets.get("snapshot_date").strip()
download = dbutils.widgets.get("download") == "true"
ensure_schema()
print(f"city={city} snapshot={snapshot_date or '<latest>'} download={download}")

# COMMAND ----------

import os
import gzip
import shutil
import urllib.request

GET_THE_DATA = "https://insideairbnb.com/get-the-data/"
UA = {"User-Agent": "Mozilla/5.0 (university course project)"}


def find_listings_urls(city: str) -> dict:
    """Scrape the Inside Airbnb download page: {snapshot_date: url} for this city slug.
    No country/state path is hard-coded, so a new city needs no code change."""
    req = urllib.request.Request(GET_THE_DATA, headers=UA)
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    pat = re.compile(r"https?://data\.insideairbnb\.com/[a-z0-9\-/]+?/" + re.escape(city)
                     + r"/(\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz")
    return {m.group(1): m.group(0) for m in pat.finditer(html)}


def resolve_url(city: str, snapshot_date: str):
    found = find_listings_urls(city)
    if not found:
        raise RuntimeError(
            f"No listings.csv.gz link for '{city}' on {GET_THE_DATA}. Check the slug, or upload the "
            f"file to {RAW_ROOT}/{city}/<YYYY-MM-DD>/listings.csv.gz and rerun with download=false.")
    if not snapshot_date:
        snapshot_date = max(found)
    url = found.get(snapshot_date) or re.sub(r"/\d{4}-\d{2}-\d{2}/", f"/{snapshot_date}/", next(iter(found.values())))
    return url, snapshot_date


def latest_local_snapshot(city: str) -> str:
    root = f"{RAW_ROOT}/{city}"
    dates = sorted(d for d in os.listdir(root) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)) if os.path.isdir(root) else []
    if not dates:
        raise FileNotFoundError(f"No uploaded snapshot folders under {root}")
    return dates[-1]


def check_file(path: str):
    """Fail early and clearly on an HTML error page, an empty file or a truncated download."""
    with gzip.open(path, "rt", encoding="utf-8") as g:
        header = g.readline().lower()
    for col in ("id", "price", "room_type", "accommodates"):
        if f'{col}' not in header:
            raise ValueError(f"{path} does not look like an Inside Airbnb listings file (missing '{col}' in header)")

# COMMAND ----------

if download:
    url, snapshot_date = resolve_url(city, snapshot_date)
    dest_dir = f"{RAW_ROOT}/{city}/{snapshot_date}"
    dest = f"{dest_dir}/listings.csv.gz"
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
    else:
        tmp = f"/tmp/{city}_{snapshot_date}_listings.csv.gz"
        print(f"Downloading {url}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        check_file(tmp)
        shutil.copy(tmp, dest)
else:
    url = None
    snapshot_date = snapshot_date or latest_local_snapshot(city)
    dest = f"{RAW_ROOT}/{city}/{snapshot_date}/listings.csv.gz"
    if not os.path.exists(dest):
        raise FileNotFoundError(dest)

check_file(dest)
print(f"Using {dest}")

# COMMAND ----------

# Descriptions contain quotes and line breaks -> multiLine + escape='"'
raw = (spark.read
       .option("header", True)
       .option("multiLine", True)
       .option("quote", '"')
       .option("escape", '"')
       .option("mode", "PERMISSIVE")
       .csv(dest))

if "city" in raw.columns:                       # never let a source column clash with our partition key
    raw = raw.withColumnRenamed("city", "source_city")

bronze = (raw
          .withColumn("city", F.lit(city))
          .withColumn("snapshot_date", F.lit(snapshot_date))
          .withColumn("source_file", F.lit(dest))
          .withColumn("ingested_at", F.current_timestamp()))

write_partition(bronze, T("bronze_listings"), f"city = '{city}'")

n_rows = spark.table(T("bronze_listings")).where(F.col("city") == city).count()
log = (spark.createDataFrame(
            [(city, snapshot_date, url, dest, n_rows, len(raw.columns))],
            "city string, snapshot_date string, source_url string, source_file string, n_rows long, n_columns int")
       .withColumn("ingested_at", F.current_timestamp()))
write_partition(log, T("bronze_ingest_log"), f"city = '{city}'")

print(f"bronze_listings[{city}] = {n_rows} rows, {len(raw.columns)} source columns, snapshot {snapshot_date}")
