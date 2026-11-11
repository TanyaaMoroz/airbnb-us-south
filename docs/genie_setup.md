# Genie Agent setup

**Create:** sidebar → Genie Agents → New. Add these tables (all in `workspace.airbnb`):
`gold_city_metrics`, `gold_room_type_mix`, `gold_neighbourhood_metrics`, `gold_predictions`, `gold_model_metrics`.

## General instructions (paste as is)

```
This space answers questions about Airbnb listings in three US South cities, from Inside Airbnb snapshots.
- City names: "City A" = 'new-orleans', "City B" = 'asheville', "City C" or "the new client" = 'nashville'.
  Always filter the city column with these exact lower-case values.
- Prices are USD per night as advertised on the snapshot date.
- "Typical price" means the median, never the average.
- Room types are exactly: 'Entire home/apt', 'Private room', 'Shared room', 'Hotel room'. "Entire home" = 'Entire home/apt'.
- gold_predictions: error = predicted_price - price. error < 0 means the model UNDER-prices; error > 0 means it OVER-prices.
  Use only rows with in_evaluation = 1 for any question about model accuracy or errors.
- Which model run to use, unless the user asks otherwise:
  'new-orleans' and 'asheville' -> run_label = 'week1'; 'nashville' -> run_label = 'c_after_fix'.
  "first run" on Nashville -> run_label = 'c_first_run'.
- MAE = mean of abs_error. When ranking groups by error, ignore groups with fewer than 10 listings.
- gold_model_metrics already holds the official accuracy numbers; prefer it for "how accurate is the model" questions.
- Round money to whole dollars.
```

## Example SQL queries (Instructions → SQL queries → Add)

**What is the median price of entire homes in City A?**
```sql
SELECT round(median_price) AS median_price_usd
FROM workspace.airbnb.gold_room_type_mix
WHERE city = 'new-orleans' AND room_type = 'Entire home/apt'
```

**Which neighbourhoods in City B have the most listings?**
```sql
SELECT neighbourhood, listings
FROM workspace.airbnb.gold_neighbourhood_metrics
WHERE city = 'asheville'
ORDER BY listings DESC
LIMIT 10
```

**Where does the model under-price listings the most?**
```sql
SELECT city, neighbourhood, count(*) AS listings, round(avg(error)) AS mean_error_usd
FROM workspace.airbnb.gold_predictions
WHERE in_evaluation = 1
  AND ((city IN ('new-orleans', 'asheville') AND run_label = 'week1')
       OR (city = 'nashville' AND run_label = 'c_after_fix'))
GROUP BY city, neighbourhood
HAVING count(*) >= 10
ORDER BY mean_error_usd ASC
LIMIT 10
```

**How accurate is the model in each city?**
```sql
SELECT city, eval_set, run_label, n, round(mae_model) AS mae_model_usd, round(mae_baseline) AS mae_baseline_usd
FROM workspace.airbnb.gold_model_metrics
ORDER BY created_at
```

## Test before the demo

Ask each example question in plain words and check the answer against the SQL above. If Genie gets one wrong,
add a sentence to the instructions that removes the ambiguity, and ask again. Screenshot each fix: these are
"the tricks you used to configure Genie" for the presentation.

Note: Asheville neighbourhoods may be ZIP codes rather than names; say so if asked in the demo.
