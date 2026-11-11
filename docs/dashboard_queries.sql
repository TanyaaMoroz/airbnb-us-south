-- AI/BI dashboard "Airbnb US South — prices and price model": SQL of every dataset.
-- Generated from docs/dashboard.lvdash.json (the dashboard export). To rebuild the dashboard, import that
-- JSON (Dashboards -> Create -> Import); this file is the readable version of the same queries.
-- Datasets that the export defines as a source + dimensions/measures are written here as one runnable
-- query: SELECT <dimensions>, <measures> FROM (<source>) GROUP BY <dimensions>.
-- Rules: every gold_predictions query filters in_evaluation = 1; error = predicted_price - price (negative = under-priced).

-- ====================================================================================================
-- dataset: predictions_eval  (Evaluated listings (predictions))
--   used by: Model: predicted vs actual price / filter-single-select
--   used by: Model: predicted vs actual price / filter-multi-select
--   used by: Model: predicted vs actual price / scatter "Predicted vs actual price"
--   used by: Model: predicted vs actual price / bar "MAE by room type"
--   used by: Model: predicted vs actual price / bar "MAE: nightly vs 30+ nights"
--   used by: Model: predicted vs actual price / table "Most under-predicted listings"
-- ====================================================================================================
SELECT
  name,
  city,
  city_name,
  split,
  neighbourhood,
  room_type,
  accommodates,
  price,
  price_flag,
  stay_type,
  predicted_price,
  error,
  abs_error,
  run_label,
  rn_under,
  rn_over
FROM
  (
    SELECT
      name,
      city,
      split,
      neighbourhood,
      room_type,
      accommodates,
      price,
      price_flag,
      CASE
        WHEN is_monthly = 1 THEN '30+ nights'
        ELSE 'nightly'
      END AS stay_type,
      predicted_price,
      error,
      abs_error,
      run_label,
      CASE
        WHEN city = 'new-orleans' THEN 'New Orleans'
        WHEN city = 'asheville' THEN 'Asheville'
        ELSE 'Nashville'
      END AS city_name,
      ROW_NUMBER() OVER (PARTITION BY city, run_label ORDER BY error ASC, id) AS rn_under,
      ROW_NUMBER() OVER (PARTITION BY city, run_label ORDER BY error DESC, id) AS rn_over
    FROM
      workspace.airbnb.gold_predictions
    WHERE
      in_evaluation = 1  )
  t
ORDER BY
  error ASC;

-- ====================================================================================================
-- dataset: price_by_guests  (Median price by guests (A+B))
--   used by: Cross-city: New Orleans vs Asheville / line "Median price by number of guests"
--   Median nightly price by number of guests (accommodates capped at 12), priced listings only
-- ====================================================================================================
SELECT
  `city` AS `city`,
  CASE WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS `city_name`,
  `guests` AS `guests`,
  MAX(median_price) AS `median_price`
FROM (
  SELECT city,
         LEAST(accommodates, 12) AS guests,
         percentile(price, 0.5) AS median_price
  FROM workspace.airbnb.silver_listings WHERE price IS NOT NULL AND city IN ('new-orleans', 'asheville') GROUP BY city, LEAST(accommodates, 12)
) AS src
GROUP BY `city`, 2, `guests`;

-- ====================================================================================================
-- dataset: model_metrics  (Model metrics (MAE by city and eval set))
--   used by: Model: predicted vs actual price / bar "Model vs baseline MAE (week 1)"
--   MAE of model vs baseline per city, eval set and run label
-- ====================================================================================================
SELECT
  `city` AS `city`,
  CASE WHEN city = 'all_train' THEN 'A+B (all train)' WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS `city_name`,
  `eval_set` AS `eval_set`,
  `run_label` AS `run_label`,
  `model_name` AS `model_name`,
  MAX(n) AS `n`,
  MAX(n_ok) AS `n_ok`,
  MAX(mae_model) AS `mae_model`,
  MAX(mae_baseline) AS `mae_baseline`,
  MAX(median_ae_model) AS `median_ae_model`,
  MAX(median_price) AS `median_price`
FROM workspace.airbnb.gold_model_metrics
GROUP BY `city`, 2, `eval_set`, `run_label`, `model_name`;

-- ====================================================================================================
-- dataset: city_metrics  (City metrics (New Orleans + Asheville))
--   used by: City overview / text "page title"
--   used by: City overview / counter "Listings"
--   used by: City overview / counter "Hosts"
--   used by: City overview / counter "Median price"
--   used by: City overview / counter "Avg days available per year"
--   used by: City overview / counter "Median review score"
--   used by: City overview / counter "Entire homes"
--   used by: City overview / counter "30+ night rentals"
--   used by: City overview / counter "Listings with no reviews"
--   used by: City overview / filter-single-select "City"
--   used by: Cross-city: New Orleans vs Asheville / bar "Median price (USD)"
--   used by: Cross-city: New Orleans vs Asheville / bar "Avg days available per year"
--   used by: Cross-city: New Orleans vs Asheville / bar "Median review score"
--   used by: Cross-city: New Orleans vs Asheville / bar "Key shares"
--   City-level Airbnb metrics for the two training cities, latest snapshot per city
-- ====================================================================================================
SELECT
  `city` AS `city`,
  CASE WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS `city_name`,
  MAX(listings) AS `listings`,
  MAX(hosts) AS `hosts`,
  MAX(listings_with_price) AS `listings_with_price`,
  MAX(median_price) AS `median_price`,
  MAX(avg_availability_365) AS `avg_availability_365`,
  MAX(median_review_score) AS `median_review_score`,
  MAX(share_entire_home) AS `share_entire_home`,
  MAX(share_multi_listing_hosts) AS `share_multi_listing_hosts`,
  MAX(share_monthly) AS `share_monthly`,
  MAX(share_no_reviews) AS `share_no_reviews`
FROM workspace.airbnb.gold_city_metrics
WHERE city IN ('new-orleans', 'asheville')
GROUP BY `city`, 2;

-- ====================================================================================================
-- dataset: room_type_mix  (Room type mix (New Orleans + Asheville))
--   used by: City overview / bar "Room-type mix"
--   used by: City overview / filter-single-select "City"
--   used by: Cross-city: New Orleans vs Asheville / bar "Median price by room type"
--   used by: Cross-city: New Orleans vs Asheville / bar "Room-type share per city"
--   Listings, median price and share per room type for the two training cities
-- ====================================================================================================
SELECT
  `city` AS `city`,
  CASE WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS `city_name`,
  `room_type` AS `room_type`,
  SUM(listings) AS `listings`,
  MAX(median_price) AS `median_price`,
  MAX(share) AS `share`
FROM workspace.airbnb.gold_room_type_mix
WHERE city IN ('new-orleans', 'asheville')
GROUP BY `city`, 2, `room_type`;

-- ====================================================================================================
-- dataset: neighbourhood_metrics  (Neighbourhood metrics (New Orleans + Asheville))
--   used by: City overview / bar "Top neighbourhoods by median price"
--   used by: City overview / symbol-map "Neighbourhood map"
--   used by: City overview / filter-single-select "City"
--   Neighbourhood-level metrics; Asheville neighbourhoods are ZIP codes
-- ====================================================================================================
SELECT
  `city` AS `city`,
  CASE WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS `city_name`,
  `neighbourhood` AS `neighbourhood`,
  `latitude` AS `latitude`,
  `longitude` AS `longitude`,
  `listings` AS `listings_n`,
  MAX(listings) AS `listings`,
  MAX(median_price) AS `median_price`,
  MAX(avg_review_score) AS `avg_review_score`,
  MAX(share_entire_home) AS `share_entire_home`,
  MAX(avg_availability_365) AS `avg_availability_365`
FROM workspace.airbnb.gold_neighbourhood_metrics
WHERE city IN ('new-orleans', 'asheville')
GROUP BY `city`, 2, `neighbourhood`, `latitude`, `longitude`, `listings`;

-- ====================================================================================================
-- dataset: city_diff  (Asheville minus New Orleans (diffs))
--   used by: Cross-city: New Orleans vs Asheville / table "Asheville minus New Orleans"
--   Asheville minus New Orleans for median price, entire homes share and 30+ night share, formatted for display
-- ====================================================================================================
SELECT
  `metric` AS `metric`,
  `sort_order` AS `sort_order`,
  MAX(new_orleans) AS `new_orleans`,
  MAX(asheville) AS `asheville`,
  MAX(difference) AS `difference`
FROM (
  WITH m AS (
    SELECT city, median_price, share_entire_home, share_monthly
    FROM workspace.airbnb.gold_city_metrics
    WHERE city IN ('new-orleans', 'asheville')
  ), v AS (
    SELECT 'Median price' AS metric, 1 AS sort_order,
      MAX(CASE WHEN city = 'new-orleans' THEN median_price END) AS no_val,
      MAX(CASE WHEN city = 'asheville' THEN median_price END) AS ash_val
    FROM m
    UNION ALL
    SELECT 'Share of entire homes', 2,
      MAX(CASE WHEN city = 'new-orleans' THEN 100 * share_entire_home END),
      MAX(CASE WHEN city = 'asheville' THEN 100 * share_entire_home END)
    FROM m
    UNION ALL
    SELECT 'Share of 30+ night listings', 3,
      MAX(CASE WHEN city = 'new-orleans' THEN 100 * share_monthly END),
      MAX(CASE WHEN city = 'asheville' THEN 100 * share_monthly END)
    FROM m
  ) SELECT metric, sort_order,
    CASE WHEN sort_order = 1
      THEN concat('$', cast(round(no_val, 0) AS string))
      ELSE concat(cast(round(no_val, 1) AS string), '%') END AS new_orleans,
    CASE WHEN sort_order = 1
      THEN concat('$', cast(round(ash_val, 0) AS string))
      ELSE concat(cast(round(ash_val, 1) AS string), '%') END AS asheville,
    CASE WHEN sort_order = 1
      THEN concat(iff(ash_val - no_val >= 0, '+', '-'), '$', cast(round(abs(ash_val - no_val), 0) AS string))
      ELSE concat(iff(ash_val - no_val >= 0, '+', '-'), cast(round(abs(ash_val - no_val), 1) AS string), ' pp') END AS difference
  FROM v
) AS src
GROUP BY `metric`, `sort_order`;

-- ====================================================================================================
-- dataset: pred_nbhd_mae  (MAE by neighbourhood (evaluated listings))
--   used by: Model: predicted vs actual price / filter-single-select
--   used by: Model: predicted vs actual price / filter-multi-select
--   used by: Model: predicted vs actual price / bar "Top neighbourhoods by MAE"
--   Per neighbourhood (per run and city) evaluated-listing counts and error sums; n_dim supports the at-least-10 filter
-- ====================================================================================================
SELECT
  `city` AS `city`,
  `city_name` AS `city_name`,
  `run_label` AS `run_label`,
  `neighbourhood` AS `neighbourhood`,
  `n` AS `n_dim`,
  SUM(abs_error_sum) / SUM(n) AS `mae`,
  SUM(error_sum) / SUM(n) AS `mean_error`,
  SUM(n) AS `evaluated_listings`
FROM (
  SELECT city, run_label, neighbourhood, city_name, n, abs_error_sum, error_sum FROM (
    SELECT city, run_label, neighbourhood,
      CASE WHEN city = 'new-orleans' THEN 'New Orleans' WHEN city = 'asheville' THEN 'Asheville' ELSE 'Nashville' END AS city_name,
      COUNT(*) AS n,
      SUM(abs_error) AS abs_error_sum,
      SUM(error) AS error_sum
    FROM workspace.airbnb.gold_predictions
    WHERE in_evaluation = 1
    GROUP BY city, run_label, neighbourhood
  )
) AS src
GROUP BY `city`, `city_name`, `run_label`, `neighbourhood`, `n`;

-- ====================================================================================================
-- dataset: c_error_breakdown  (Nashville vs A+B error breakdown)
--   used by: Model: predicted vs actual price / bar "Nashville vs A+B: MAE by segment"
--   Error of the new city (Nashville, first run) vs the A+B test set by dimension bucket
-- ====================================================================================================
SELECT
  `dimension` AS `dimension`,
  `bucket` AS `bucket`,
  `group` AS `group`,
  CASE WHEN `group` = 'nashville' THEN 'Nashville (first run)' ELSE 'A+B test set' END AS `group_name`,
  `run_label` AS `run_label`,
  `n` AS `n_dim`,
  MAX(mae) AS `mae`,
  MAX(mean_error) AS `mean_error`,
  MAX(median_price) AS `median_price`
FROM workspace.airbnb.gold_c_error_breakdown
GROUP BY `dimension`, `bucket`, `group`, 4, `run_label`, `n`;

-- ====================================================================================================
-- dataset: four_numbers  (The four numbers (headline MAEs))
--   used by: Model: predicted vs actual price / bar "The four numbers"
--   The four headline MAE numbers in fixed order; bars with no data yet are absent
-- ====================================================================================================
SELECT
  `metric` AS `metric`,
  `order_n` AS `order_n`,
  MAX(value) AS `value`
FROM (
  SELECT '1. Baseline MAE — Nashville, all priced (first run)' AS metric, 1 AS order_n, mae_baseline AS value FROM workspace.airbnb.gold_model_metrics WHERE city = 'nashville' AND run_label = 'c_first_run' AND eval_set = 'all_priced' UNION ALL SELECT '2. Model MAE — New Orleans + Asheville, week-1 test', 2, mae_model FROM workspace.airbnb.gold_model_metrics WHERE city = 'all_train' AND run_label = 'week1' AND eval_set = 'test' UNION ALL SELECT '3. Model MAE — Nashville, first run, held-out 20%', 3, mae_model FROM workspace.airbnb.gold_model_metrics WHERE city = 'nashville' AND run_label = 'c_first_run' AND eval_set = 'c_heldout' UNION ALL SELECT '4. Model MAE — Nashville after changes, held-out 20%', 4, mae_model FROM workspace.airbnb.gold_model_metrics WHERE city = 'nashville' AND run_label = 'c_after_fix' AND eval_set = 'c_heldout'
) AS src
GROUP BY `metric`, `order_n`;

-- ====================================================================================================
-- dataset: headline_mae  (Headline MAE numbers (one row))
--   used by: Model: predicted vs actual price / text "page title"
--   The four headline MAE numbers as four measures in a single row
-- ====================================================================================================
SELECT
  MAX(baseline_mae) AS `baseline_mae`,
  MAX(ab_test_mae) AS `ab_test_mae`,
  MAX(c_first_run_mae) AS `c_first_run_mae`,
  MAX(c_after_fix_mae) AS `c_after_fix_mae`
FROM (
  SELECT
    MAX(CASE WHEN city = 'nashville' AND run_label = 'c_first_run' AND eval_set = 'all_priced' THEN mae_baseline END) AS baseline_mae,
    MAX(CASE WHEN city = 'all_train' AND run_label = 'week1' AND eval_set = 'test' THEN mae_model END) AS ab_test_mae,
    MAX(CASE WHEN city = 'nashville' AND run_label = 'c_first_run' AND eval_set = 'c_heldout' THEN mae_model END) AS c_first_run_mae,
    MAX(CASE WHEN city = 'nashville' AND run_label = 'c_after_fix' AND eval_set = 'c_heldout' THEN mae_model END) AS c_after_fix_mae
  FROM workspace.airbnb.gold_model_metrics
) AS src;

-- ====================================================================================================
-- dataset: predictions_eval_over  (Evaluated listings (predictions, error descending))
--   used by: Model: predicted vs actual price / filter-single-select
--   used by: Model: predicted vs actual price / filter-multi-select
--   used by: Model: predicted vs actual price / table "Most over-predicted listings"
-- ====================================================================================================
SELECT
  name,
  city,
  city_name,
  split,
  neighbourhood,
  room_type,
  accommodates,
  price,
  price_flag,
  stay_type,
  predicted_price,
  error,
  abs_error,
  run_label,
  rn_under,
  rn_over
FROM
  (
    SELECT
      name,
      city,
      split,
      neighbourhood,
      room_type,
      accommodates,
      price,
      price_flag,
      CASE
        WHEN is_monthly = 1 THEN '30+ nights'
        ELSE 'nightly'
      END AS stay_type,
      predicted_price,
      error,
      abs_error,
      run_label,
      CASE
        WHEN city = 'new-orleans' THEN 'New Orleans'
        WHEN city = 'asheville' THEN 'Asheville'
        ELSE 'Nashville'
      END AS city_name,
      ROW_NUMBER() OVER (PARTITION BY city, run_label ORDER BY error ASC, id) AS rn_under,
      ROW_NUMBER() OVER (PARTITION BY city, run_label ORDER BY error DESC, id) AS rn_over
    FROM
      workspace.airbnb.gold_predictions
    WHERE
      in_evaluation = 1
  ) t
ORDER BY
  error DESC;
