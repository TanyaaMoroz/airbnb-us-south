# AI/BI dashboard spec (week 2)

**All SQL below is untested.** The column names are taken from `03_gold_city.py`, `04_train_model.py` and `05_score_city.py`. Fill every `<fill after run>` from the dashboard itself after the gold tables exist.

## Design rules (Lecture 7)
- **Titles:** every page and chart title states the finding (a sentence), not the metric name.
- **Bars:** start at zero, sorted by value (not alphabetically). Horizontal bars for long category names.
- **Colour:**
  - Grey for context.
  - One accent colour for the thing the title talks about.
  - The same city always gets the same colour on every page: New Orleans and Asheville get their own colours, and city C gets a third one later.
- **Labels:** money in USD with no decimals, shares as %. Put the snapshot date in a subtitle.
- **Filters:** global filters only where they make sense. City on page 1; `run_label` on page 3.
- **City C:** must not appear anywhere until after its first run. Until then, pages 1–2 filter `city IN ('new-orleans', 'asheville')` and page 3 uses `run_label = 'week1'`.

## Datasets (one per query)

### Page 1: City overview
Title (placeholder): *"<City>: median entire home costs $<fill after run> a night, and <fill after run>% of listings are monthly rentals"*

| chart | dataset |
|---|---|
| KPI counters: listings, hosts, median price, share with no reviews, median review score, share monthly | `ds_city_metrics` |
| Room-type mix (bars, share, sorted) + median price per room type | `ds_room_mix` |
| Neighbourhoods by median price (top 15 bars, min 20 listings; grey for the rest) | `ds_neigh` |
| Map of neighbourhoods (point map, size = listings, colour = median price) | `ds_neigh` |
| Availability: avg `availability_365` and share unavailable | `ds_city_metrics` |

```sql
-- ds_city_metrics
SELECT city, snapshot_date, listings, hosts, listings_with_price, median_price,
       avg_availability_365, share_unavailable_365, median_review_score, share_no_reviews,
       share_entire_home, share_multi_listing_hosts, share_monthly, median_accommodates
FROM workspace.airbnb.gold_city_metrics
WHERE city IN ('new-orleans', 'asheville');

-- ds_room_mix
SELECT city, room_type, listings, share, median_price
FROM workspace.airbnb.gold_room_type_mix
WHERE city IN ('new-orleans', 'asheville')
ORDER BY city, listings DESC;

-- ds_neigh  (Asheville "neighbourhoods" are ZIP codes: say so in the subtitle)
SELECT city, neighbourhood, listings, median_price, avg_review_score,
       share_entire_home, avg_availability_365, latitude, longitude
FROM workspace.airbnb.gold_neighbourhood_metrics
WHERE city IN ('new-orleans', 'asheville') AND listings >= 20
ORDER BY median_price DESC;
```

### Page 2: Cross-city A vs B
Title (placeholder): *"Asheville is $<fill after run> pricier per night than New Orleans, but New Orleans is <fill after run>× more monthly"*

| chart | dataset |
|---|---|
| Side-by-side table / grouped bars of the key metrics (both cities) | `ds_city_long` |
| Median price by room type, grouped by city | `ds_room_mix` (reuse) |
| Room-type share by city (100% bars) | `ds_room_mix` |
| Difference column (Asheville − New Orleans) | `ds_city_diff` |

```sql
-- ds_city_long  (long format so one bar chart can show every metric side by side)
SELECT city, metric, value FROM (
  SELECT city,
         median_price, avg_availability_365, share_unavailable_365, median_review_score,
         share_no_reviews, share_entire_home, share_multi_listing_hosts, share_monthly
  FROM workspace.airbnb.gold_city_metrics
  WHERE city IN ('new-orleans', 'asheville')
) UNPIVOT (value FOR metric IN (median_price, avg_availability_365, share_unavailable_365,
           median_review_score, share_no_reviews, share_entire_home, share_multi_listing_hosts, share_monthly));

-- ds_city_diff
SELECT
  MAX(CASE WHEN city = 'asheville'   THEN median_price END) -
  MAX(CASE WHEN city = 'new-orleans' THEN median_price END) AS median_price_diff_usd,
  MAX(CASE WHEN city = 'asheville'   THEN share_monthly END) -
  MAX(CASE WHEN city = 'new-orleans' THEN share_monthly END) AS share_monthly_diff,
  MAX(CASE WHEN city = 'asheville'   THEN share_entire_home END) -
  MAX(CASE WHEN city = 'new-orleans' THEN share_entire_home END) AS share_entire_home_diff
FROM workspace.airbnb.gold_city_metrics
WHERE city IN ('new-orleans', 'asheville');
```

### Page 3: Model
Title (placeholder): *"The model beats the baseline by $<fill after run> MAE; it under-prices <fill after run> the most"*

**Query rules for this page:**
- Every prediction query filters `in_evaluation = 1`. That means the test split for training cities, and all priced listings for a new city.
- `error = predicted_price - price`. **Negative = under-priced** (the model suggests less than the listing asks); positive = over-priced.
- The most under-priced listings are `ORDER BY error ASC`; the most over-priced are `ORDER BY error DESC`.
- Add a dashboard filter on `run_label` (default `week1`). After the city C run, the same filter shows `c_first_run` / `c_after_fix` and city C appears automatically. No new charts are needed.

| chart | dataset |
|---|---|
| MAE model vs baseline per city (+ `all_train`), bars from zero | `ds_model_metrics` |
| Predicted vs actual scatter (log axes, 45° reference line), colour = city | `ds_pred` |
| MAE by room type (bars, sorted) | `ds_err_room` |
| MAE by neighbourhood (top 15, min 10 listings) | `ds_err_neigh` |
| Top 15 under-priced / top 15 over-priced listings (tables) | `ds_under`, `ds_over` |
| Optional: MAE nightly vs monthly (`is_monthly`) | `ds_err_monthly` |

```sql
-- ds_model_metrics
SELECT run_label, city, eval_set, n, n_ok, mae_model, mae_baseline, mae_model_ok, mae_baseline_ok,
       median_ae_model, median_price, model_version, model_name
FROM workspace.airbnb.gold_model_metrics
WHERE run_label = :run_label;

-- ds_pred
SELECT city, id, name, room_type, neighbourhood, price, predicted_price, baseline_price, error, price_flag, is_monthly
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label;

-- ds_err_room
SELECT city, room_type, COUNT(*) AS n, AVG(abs_error) AS mae_model,
       AVG(ABS(baseline_price - price)) AS mae_baseline, AVG(error) AS mean_error
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label
GROUP BY city, room_type
ORDER BY mae_model DESC;

-- ds_err_neigh
SELECT city, neighbourhood, COUNT(*) AS n, AVG(abs_error) AS mae_model, AVG(error) AS mean_error
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label
GROUP BY city, neighbourhood
HAVING COUNT(*) >= 10
ORDER BY mae_model DESC
LIMIT 15;

-- ds_under  (negative error = model suggests less than the listing asks)
SELECT city, id, name, room_type, accommodates, neighbourhood, price, predicted_price, error, pct_error, price_flag
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label
ORDER BY error ASC
LIMIT 15;

-- ds_over
SELECT city, id, name, room_type, accommodates, neighbourhood, price, predicted_price, error, pct_error, price_flag
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label
ORDER BY error DESC
LIMIT 15;

-- ds_err_monthly
SELECT city, is_monthly, COUNT(*) AS n, AVG(abs_error) AS mae_model, AVG(ABS(baseline_price - price)) AS mae_baseline
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND run_label = :run_label
GROUP BY city, is_monthly;
```

**Notes:**
- `:run_label` is a dashboard parameter (default `week1`). If parameters are awkward, use a field filter on `run_label` instead.
- `gold_predictions` partitions are written per `city` + `run_label`. Training-city predictions get the label passed to `05_score_city` (default `week1`).
- Expect the top under/over-priced tables to be dominated by `extreme_high` and monthly listings (see EDA §4). Consider a `price_flag = 'ok'` toggle so readers can see both views.
