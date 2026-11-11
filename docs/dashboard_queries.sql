-- AI/BI dashboard datasets. In the dashboard: Data tab -> "Create from SQL" -> paste one query per dataset.
-- Page titles must state a finding (Lecture 7). Templates are next to each page; fill the numbers from the charts.
-- Rules: bars start at zero, categories sorted by value, context in grey, one highlight colour, no pies, no dual axes.

-- =====================================================================
-- PAGE 1 — City overview
-- Title template (fill from the numbers): "<City> is the bigger market: X listings at $Y median vs Z at $W in <other city>"
-- =====================================================================

-- dataset: city_overview  -> counter widgets (one per number) with a city filter
SELECT city, listings, hosts, round(median_price) AS median_price,
       round(avg_availability_365) AS avg_days_available,
       round(median_review_score, 2) AS median_review_score,
       round(share_entire_home * 100) AS pct_entire_home,
       round(share_monthly * 100) AS pct_30plus_nights
FROM workspace.airbnb.gold_city_metrics;

-- dataset: room_mix  -> horizontal bar: room_type by pct, sorted, one bar group per city
SELECT city, room_type, listings, round(share * 100, 1) AS pct, round(median_price) AS median_price
FROM workspace.airbnb.gold_room_type_mix;

-- dataset: neighbourhoods  -> horizontal bar: top 15 by median price, filter on city
SELECT city, neighbourhood, listings, round(median_price) AS median_price,
       round(avg_review_score, 2) AS avg_review_score
FROM workspace.airbnb.gold_neighbourhood_metrics
WHERE listings >= 20
ORDER BY median_price DESC;

-- =====================================================================
-- PAGE 2 — Cross-city comparison
-- Title template (fill from the numbers): "<the one difference that matters most>, e.g. share of entire homes, availability, price per guest"
-- Use city_overview and room_mix side by side (A grey, B colour), plus:
-- =====================================================================

-- dataset: price_by_size  -> line or bar: median price by accommodates, one series per city
SELECT city, LEAST(accommodates, 12) AS guests_12_plus_capped, count(*) AS listings,
       round(percentile(price, 0.5)) AS median_price
FROM workspace.airbnb.silver_listings
WHERE price IS NOT NULL AND accommodates IS NOT NULL
GROUP BY city, LEAST(accommodates, 12);

-- =====================================================================
-- PAGE 3 — Model
-- Title template: "The model cuts the error by 30% in the training cities; on Nashville the first run was X times worse"
-- =====================================================================

-- dataset: model_metrics  -> table or bar (MAE model vs baseline per city and run), bars from zero
SELECT city, eval_set, run_label, n, round(mae_model) AS mae_model, round(mae_baseline) AS mae_baseline,
       round(median_ae_model) AS median_abs_error
FROM workspace.airbnb.gold_model_metrics
ORDER BY created_at;

-- dataset: predicted_vs_actual  -> scatter: price (x) vs predicted_price (y), colour = city, filter on run_label
SELECT city, run_label, room_type, price, round(predicted_price) AS predicted_price
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1 AND price < 1500;

-- dataset: error_by_room_type  -> horizontal bar sorted by mae; mean_error < 0 = under-priced
SELECT city, run_label, room_type, count(*) AS listings,
       round(avg(abs_error)) AS mae, round(avg(error)) AS mean_error
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1
GROUP BY city, run_label, room_type;

-- dataset: most_mispriced  -> table, filter on city and run_label
SELECT city, run_label, name, neighbourhood, room_type, accommodates, price,
       round(predicted_price) AS predicted_price, round(error) AS error,
       CASE WHEN error < 0 THEN 'under-priced' ELSE 'over-priced' END AS direction
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1
ORDER BY abs_error DESC
LIMIT 50;

-- dataset: city_c_breakdown  -> bars: mae by bucket, new city vs "A+B test", one chart per dimension
SELECT dimension, bucket, `group`, n, mae, mean_error, median_price
FROM workspace.airbnb.gold_c_error_breakdown
WHERE n >= 20;
