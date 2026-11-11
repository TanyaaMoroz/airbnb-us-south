# Column cheat-sheet: Inside Airbnb listings.csv detail v4.7 (first sheet)

- **Type and calc** come from the dictionary sheet "listings.csv detail v4.7" (dated 2026-03-01). A blank type means the dictionary leaves it blank.
- **% missing** figures come from the local pandas run (pandas check of the same snapshots), New Orleans / Asheville. Re-verify them on Databricks.
- **Silver** = read in `02_silver_clean.py`. **Feature** = in `FEATURES_NUM`/`FEATURES_CAT` in `common.py`, directly or through a derived column.
- **Leakage verdict:** ok / suspect / leak / not used.

## Columns used by silver/common

| column | type | calc | meaning (short) | trap / what to watch for | silver | feature | leakage |
|---|---|---|---|---|---|---|---|
| id | integer | | listing id | 19-digit ids: keep as string (silver does). Unique: 0 duplicates, 0 overlap between cities | y | n | not used |
| name | text | | listing title | free text. Mentions "Sleeps 26", "Superbowl" | y | n | not used (could contain price wording) |
| host_id | integer | | host id | 3,208 / 1,730 hosts | y | n | not used |
| neighbourhood_cleansed | text | y | geocoded neighbourhood | Asheville has **8 ZIP codes**, New Orleans has 70 names. Raw `neighbourhood` is 100% empty | y | n | not used (design) |
| latitude / longitude | numeric | | WGS84 | Inside Airbnb anonymises the location by a small offset | y | via `dist_center_km` | ok |
| property_type | text | | host-selected type | many rare values (handled by one-hot `min_frequency=20`) | y | y (cat) | ok |
| room_type | text | | Entire home/apt, Private room, Shared room, Hotel | **dictionary says `Hotel`, the data says `Hotel room`** (code uses `Hotel room`) | y | y (cat) | ok |
| accommodates | integer | | max guests | **capped at 16**: 185 / 111 rows sit at 16, and some names say "sleeps 19–30" | y | y | ok |
| bathrooms | numeric | | number of bathrooms | **100% empty**. The value comes from `bathrooms_text` | y | y | ok |
| bathrooms_text | string | | "1.5 shared baths" | "Half-bath" has no digit → 0.5. Empty for 105 New Orleans hotel rooms | y | via bathrooms, bath_shared | ok |
| bedrooms | integer | | bedrooms | 0.0% / 0.6% missing. Studios can be 0 | y | y | ok |
| beds | integer | | beds | 3.0% / 1.5% missing, mostly "Room in hotel" (169 of 221 in New Orleans) | y | y | ok |
| amenities | json | | JSON list | regex flags depend on wording ("pool table", "free street parking") | y | count + 5 flags | ok |
| price | currency | | "daily price"; `$` is an export artefact | **Actually the per-night value of a quote for a specific stay.** Discounted in 48.1% / 23.3%. Monthly-priced when min nights ≥ 30. Missing 14.8% / 10.5% | y | **target** | target |
| minimum_nights | integer | | min stay ("calendar rules may be different") | 40.0% / 8.6% are ≥ 30, and those prices are monthly-priced. Missing 1.9% / 0.3%, always together with `availability_365 = 0` | y | y | suspect (sets the price *definition*) |
| maximum_nights | integer | | max stay | same missing pattern as minimum_nights | y | n | not used |
| has_availability | boolean | | t/f | empty in 47 / 14 rows. All of those have no price | y | n | not used |
| availability_30 / 90 / 365 | integer | y | free days in the next x days | "not available" = **booked or blocked by the host** (cannot tell which) | y | n | suspect (reverse causality, excluded) |
| number_of_reviews | integer | | lifetime reviews | 0 reviews ⇔ all review_* columns missing | y | y | ok |
| number_of_reviews_ltm | integer | y | reviews in last 12 months | Spearman 0.955 with `estimated_occupancy_l365d` (an excluded column) | y | y | suspect |
| reviews_per_month | numeric | y | lifetime average | null when 0 reviews | y | n | not used |
| first_review / last_review | date | y | first / last review date | null when 0 reviews | y | via days_since_last_review | ok |
| review_scores_rating | (blank) | | overall score | missing ⇔ 0 reviews (16.6% / 8.1%). Scale 0–5 | y | y | ok |
| review_scores_accuracy, _checkin, _communication | (blank) | | sub-scores | same pattern | y | n | not used |
| review_scores_cleanliness, _location | (blank) | | sub-scores | same pattern | y | y | ok |
| review_scores_value | (blank) | | "value for money" score | Spearman with log price = 0.026, so no visible price leak | y | n | suspect (excluded as a precaution) |
| host_since | date | | host account date | **100% empty** → `host_years` is all null | y | via host_years | ok, but useless |
| host_response_rate / host_acceptance_rate | (blank) / – | | rates | **100% empty** | y | n | not used |
| host_is_superhost | boolean | | t/f | filled. 46.9% (New Orleans) vs 66.2% (Asheville) superhosts | y | y | ok |
| instant_bookable | boolean | | auto-booking | **100% empty** → feature all null | y | y | ok, but useless |
| calculated_host_listings_count | integer | y | host's listings in this scrape | commercial operators. Scrape-relative, so it depends on the city's size | y | y | ok |
| license | text | | permit number | Asheville 100% empty. New Orleans: "Exempt" 220, missing mostly for monthly listings | y | n | not used |
| estimated_occupancy_l365d | **not documented** | | — | capped at 255 nights (19.4% of rows at the cap). Derived from reviews | y | n | suspect (excluded) |
| estimated_revenue_l365d | **not documented** | | — | **= price × occupancy exactly** (100% within ±1%). Missing exactly when price is missing | y | n | **leak** |

## Other columns the dictionary marks as calculated

| column | type | calc | meaning | trap | silver | feature | leakage |
|---|---|---|---|---|---|---|---|
| listing_url, host_url | text | y | URLs | — | n | n | not used |
| scrape_id | bigint | y | scrape id | — | n | n | not used |
| last_scraped | datetime | y | scrape time | ≠ folder date (New Orleans 09-17 vs folder 09-16; Asheville spans 2 days) | n | n | not used |
| neighbourhood_group_cleansed | text | y | neighbourhood group | 100% empty | n | n | not used |
| minimum_minimum_nights … maximum_maximum_nights, minimum/maximum_nights_avg_ntm | integer/numeric | y | calendar-based min/max stays | `minimum_minimum_nights` has Spearman −0.333, the same as `minimum_nights` | n | n | not used |
| availability_60 | integer | y | free days in 60 | as other availability columns | n | n | not used |
| number_of_reviews_l30d | integer | y | reviews last 30 days | activity proxy | n | n | not used |
| calculated_host_listings_count_entire_homes / _private_rooms / _shared_rooms | integer | y | per-type counts | — | n | n | not used |

## Undocumented columns present in the files (watch-list)

| column | what it seems to be | verdict |
|---|---|---|
| price_quote_price_per_night | identical to `price` (Spearman 1.000; equal in 100% of rows) | **leak**: never use |
| price_quote_total_price | price × quoted nights | **leak** |
| price_quote_checkin_date / _checkout_date / _raw | the stay the price was quoted for; `_raw` JSON holds discount and `is_monthly_priced` | leak-adjacent: use only to explain price |
| hosts_time_as_host_years / _months, hosts_time_as_user_years / _months | host tenure. 0% missing, median 6 / 7 years | ok: candidate replacement for `host_years` |
| availability_eoy | availability to end of year | suspect (same as availability) |
| number_of_reviews_ly | reviews last year | suspect (activity) |
| host_profile_id / host_profile_url | ids | not used |

## Dictionary gaps
- **The code uses columns the dictionary does not document:** `estimated_occupancy_l365d`, `estimated_revenue_l365d`.
- **The files contain 15 columns the dictionary lacks:** host_profile_id, host_profile_url, hosts_time_as_user_years/months, hosts_time_as_host_years/months, price_quote_checkin_date, price_quote_checkout_date, price_quote_total_price, price_quote_price_per_night, price_quote_raw, availability_eoy, number_of_reviews_ly, estimated_occupancy_l365d, estimated_revenue_l365d.
- **Blank types in the dictionary:** host_response_time, host_response_rate, host_acceptance_rate, host_verifications, and all seven review_scores_*. Several also have no description (host_is_superhost, amenities, review_scores_*).
- **Types that disagree with the data or the code:**
  - `host_listings_count` / `host_total_listings_count` are typed "text" but are numbers.
  - `has_availability` and `instant_bookable` are "boolean" with t/f values.
  - `calendar_updated` is "date" but 100% empty.
- **Values that disagree with the data:**
  - `room_type` "Hotel" vs the data's "Hotel room".
  - `source` lists "neighbourhood search", but the files only contain empty or "previous scrape".
- **Documented but 100% empty in both files:** neighborhood_overview, host_since, host_response_time, host_response_rate, host_acceptance_rate, host_thumbnail_url, host_neighbourhood, host_total_listings_count, host_verifications, neighbourhood, neighbourhood_group_cleansed, bathrooms, calendar_updated, instant_bookable.

## What the dictionary tells us we still do not know
- **Availability ≠ demand.** A day is unavailable if it is *booked or blocked by the host*. We cannot separate the two, so occupancy and availability are not clean demand signals.
- **The price is advertised, not paid.** No booked price, no fees (cleaning, service, taxes are null in the quotes). It is one quoted stay at one date, sometimes with a last-minute or monthly discount. There is no seasonality.
- **`minimum_nights` may not reflect the calendar.** The dictionary itself says "calendar rules may be different". `minimum_minimum_nights` … `maximum_maximum_nights` show the calendar range.
- **Host listing counts:** the dictionary says Airbnb computes them by "unknown calculations".
- **`source`:** "previous scrape" listings were found in a scrape in the last 65 days and confirmed still on Airbnb. We do not know why they were not found by search this time.
- **`estimated_*`:** not documented in v4.7. The formula (revenue = price × occupancy) is shown only by our check. The occupancy model (review-based, capped at 255) is undocumented here.
- **Review scores:** no scale or definition in the dictionary.
