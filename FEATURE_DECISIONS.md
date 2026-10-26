# Feature decisions: King County house sales

Every decision Zeyad has made on a column, with the reason. Claude records the decisions; Zeyad makes them.

- **Data:** King County Assessor files (Seattle area): sales, residential buildings, parcels and lookup codes.
- **Rows:** clean single-house market sales from **2000 onwards**: 533,055 sales of 328,776 homes.
- **Split:** by home, 80% train and 20% test, seed 42: 426,259 training sales and 106,796 test sales.

## Rules for every column

- **Standardize every numeric input:** subtract the training average and divide by the training spread. The test sales use the same training numbers, so no test information leaks in.
- **Yes/no columns stay 0 or 1** and are not standardized (decided at feature 19, easier to read).
- **Each category group drops the column of its most common value (the baseline)** (D3, 25 October 2026, replacing the earlier "keep every column" rule from features 16–19). Zeyad wants zero redundancy: with every column kept, the columns of one group always add up to 1, so any one of them could be computed from the others. With the baseline gone, a sale of the most common value has 0 in every column of that group.
- **Rare values:** a value under 0.1% of sales joins "other"; if "other" itself stays under 0.1%, it joins the most common value (used for the 14 earlier keeps and street type, 25 October 2026). A column with only two common values becomes one yes/no column.
- **Log only where a column decision says so** (sale price, above-ground living area, lot area). The automatic log test for every area column was raised but not chosen.

### Professor's Phase 2 rules (shared by Zeyad on 25 October 2026)

1. All standardization and scaling is finished in Phase 2. None of it moves to Phase 3.
2. The report itself carries the feature analysis: distributions, the reason for each scaling choice, correlation with the label and cross-correlation between features, with the diagrams. Notebooks do not replace the report.
3. Every text feature is one-hot encoded or embedded (word2vec or doc2vec). A text feature dropped without a justification costs marks, so every dropped text column below has its own measured reason.
4. Any feature with a correlation of 0.85 or more with the label is dropped, because it leaks the answer.

## Decided one column at a time (24 October 2026)

| # | Column | Decision | Reason |
|---|---|---|---|
| 1 | `SalePrice`, sale price (the label) | **Predict the natural log of the price** for every sale. Keep all sales, including the expensive ones. | In raw dollars the most expensive 1% of sales carry 44.9% of the squared error; in log they carry 9.9%. The skew drops from 6.15 to 0.47. Features act as percentages of price. The base of the log does not change the results; the natural log makes small differences read as percentages. |
| 2 | `DocumentDate`, sale date | **Use sales from 2000 onwards.** Keep the true sale price as the label (no price adjustment to today). Add no market-level column: the model learns the market from the date. The date goes in as **years since 1 January 2000, with decimals** (for example 14 June 2010 = 10.45). | Only the 2000-onwards window keeps more than 500,000 rows. An adjusted price would be an estimate, not a fact. Whole years would lose the movement within a year. |
| 3 | `SqFtTotLiving`, total living area | **Replace with above-ground living area = total living area − finished basement, in natural log.** The basement parts get their own columns (Features 4 to 6). | Strongest size column (explains 50.0% within a month). Skew 1.3 becomes −0.07 after the log. 100 extra square feet add +3.4% above ground but +2.3% as finished basement (same appraisal area, same month), so one total would price both kinds of space the same. |
| 4 | `SqFtTotBasement`, total basement area | **Replace with other unfinished basement = total basement − finished basement − basement garage** (storage, laundry, furnace room). Plain square feet, then standardized (rule above). | Total living area already holds the finished basement, and the basement garage moves to the garage total, so only the leftover part is new. Present in 14.8% of training sales, median 360 sq ft; about +0.8% per 100 sq ft (same appraisal area and month). Checks: no house has a finished basement bigger than its basement, a garage bigger than its unfinished basement, or a negative leftover. |
| 5 | `SqFtFinBasement`, finished basement area | **Keep as its own column** (option Y). Together with decision 3, total living area is removed as a column, so the finished basement is counted exactly once. | Removing both the finished basement and total living area would lose the finished basement entirely. 41.1% of training sales have one, median 720 sq ft; about +2.3% per 100 sq ft against +3.4% above ground. |
| 6 | `SqFtGarageBasement`, basement garage area | **Add it to the attached garage area to make one total garage area.** | Keeps the garage out of the unfinished basement so no square foot is counted twice. |
| 7 | `SqFtGarageAttached`, attached garage area | **Combined with the basement garage into total garage area** (decision 6). Plain square feet, then standardized (rule above). | Total garage: 73.1% of training sales have one, median 480 sq ft; 23,103 sales have both kinds. |
| 8 | `SqFtOpenPorch`, open porch area | **Keep, plain square feet, standardized** (applied from the standardize rule). | 53.8% of training sales have an open porch; explains 7.9% within a month. The big price step is none → a small porch (−8.4% to +7.4%). Not yet measured with location held fixed. |
| 9 | `SqFtDeck`, deck area | **Keep, plain square feet, standardized.** | 50.7% of training sales have no deck; explains 4.1% within a month. Price rises steadily with size: none −5.3%, 350 sq ft +8.6%, 560 sq ft +28.9%. Part of the gap may be hillside lots and house type. Not yet measured with location held fixed. |
| 10 | `Bedrooms`, bedrooms | **Keep as a number, standardized.** **Remove the 95 sales with 0 bedrooms** (see decision 11–13 for the combined count). | Explains 15.0% within a month. Price climbs up to 5–6 bedrooms (+38.5%), then flattens. Moves with total living area (rank correlation 0.68), so its effect will shrink once size is in the model. |
| 11–13 | `BathFullCount`, `Bath3qtrCount`, `BathHalfCount`, the three bathroom counts | **Keep all three as separate counts, each standardized.** Not combined into one total. **Remove the 60 sales with no bathroom of any kind.** | Per extra bathroom (same appraisal area, month and living area): full +4.6%, three-quarter +2.8%, half +3.9%. The usual 1 / 0.75 / 0.5 total would price a half bathroom at half a full one; the sales say about 85%. 4,189 of the 4,285 sales with 0 full bathrooms have a three-quarter bathroom, so those are real houses. Zero-bedroom and no-bathroom sales together: 124 removed (31 are in both groups). Their median living area is 1,285 sq ft against 2,030 for all sales, so most look like ordinary small houses with the counts left blank rather than storage units. |
| 14 | `Stories`, number of stories | **Keep as a number (1 to 4, in halves), standardized.** Not split into categories. | Correlation with the label (log price): 0.297; with the month-adjusted label 0.321 (rank correlation 0.338). A straight line explains 10.3% of the month-adjusted label and 7 categories explain 11.8%, so categories gain only 1.5 points. The 2.5-story jump (+71.5%) covers 0.7% of sales. |
| 15 | `YrBuilt`, year built | **Keep as a number (the year), standardized.** Not turned into age at sale (the sale date is already an input, so age adds nothing). | U shape: pre-1930 houses +9% to +18%, 1940s–1960s −13% to −17%, 2010s +13.9%. Decades explain 6.5% of the month-adjusted label, a straight line only 1.7%. Correlation with the label 0.166, month-adjusted 0.131. |
| 16 | `YrRenovated`, year renovated | **Decade categories, one yes/no column each:** never renovated, 1960s or earlier, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s. | 96.1% never renovated (−0.7%). Renovated in the 1990s +41.3%, 2000s +56.5%, 2010s +23.0%. Explains 1.5%. The raw column writes 0 for never, so it cannot stay a plain number. |
| 17 | `BldgGrade`, building grade | **Categories, one yes/no column per grade 5 to 12.** Grades 1–4 (132 sales) merge into 5; grade 13 (215 sales) merges into 12. | The strongest column so far: explains 55.2% (a straight line 54.6%). Correlation with the label 0.598, month-adjusted 0.739. Each grade step adds roughly 30–40%: grade 7 −17.1%, grade 9 +55.9%, grade 12 +346.9%. |
| 18 | `FinBasementGrade`, finished basement grade | **Categories, one yes/no column each: 0 (no finished basement), 5, 6, 7, 8, 9, 10, 11.** Grade 4 (0.11%) merges into 5; anything above 11 merges into 11. | Explains 16.7%; correlation with the label 0.209, month-adjusted 0.231. Grade 0 appears exactly when finished basement area is 0. With a finished basement it equals building grade 51.4% of the time and is one step lower 43.1% (correlation 0.83). |
| 19 | `HeatSystem`, heating system | **One yes/no column (0 or 1) per heating type:** forced air, heat pump / air conditioning, electric baseboard, floor or wall heater, hot water, radiant, other (gravity and other merged). | Explains 6.0%; correlation ratio 0.24 (the codes are names, so a straight-line correlation does not apply). Electric baseboard −29.9%, floor or wall heater −27.3%, hot water +16.5%, heat pump / air conditioning +9.6%. |
| 20 | `HeatSource`, heating fuel | **One yes/no column (0 or 1) per fuel:** gas, electricity, oil, gas + solar, electricity + solar, other. | Explains 3.4%; correlation ratio 0.18. Gas +5.4% (73.2% of sales), electricity −14.0%, oil −9.8%, gas + solar +14.6%. May partly repeat heating system (not measured). |
| 21–24 | `FpSingleStory`, `FpMultiStory`, `FpFreestanding`, `FpAdditional`, the four fireplace counts | **Combine into one total fireplace count, standardized.** | Total: correlation with the label 0.284, month-adjusted 0.399, stronger than any single count (best single: multi-story 0.243). 14.1% of sales have no fireplace. Per extra fireplace (same area, month, living area): single-story +6.9%, multi-story +5.4%, freestanding +1.7%, additional −3.4%. |
| 25 | `ZipCode`, ZIP code | **Remove.** | 127 codes, 12.1% empty. Explains 44.4% on held-out homes, but on top of appraisal area it adds 4.0 points against 5.7 for the sub-area. |
| 26 | `DistrictName`, city | **Keep as categories, one yes/no column per city** (40 cities, unincorporated King County is its own city). | Explains 36.5% on held-out homes (correlation ratio 0.60); adds 1.4 points on top of appraisal area. Medina +314%, Clyde Hill +293%, Algona −44%. **Dropped on 25 October 2026 by the 0.85 rule** (see "Final selection"): 0.868 with area + sub-area, which is closer to the label (0.59 against 0.46). |
| 27 | `Area`, appraisal area | **Keep as categories, one yes/no column per area** (89 areas). | The best single location column: explains 55.0% on held-out homes (correlation ratio 0.74). **Dropped on 25 October 2026 by the 0.85 rule** (see "Final selection"): 1.000 with area + sub-area (every sub-area sits in one area), which is closer to the label (0.59 against 0.56). |
| 28 | `SubArea`, appraisal sub-area | **Join with its appraisal area into one "area + sub-area" code, one yes/no column per code** (299 codes), kept alongside decision 27. | The sub-area number repeats across areas, so it only means something joined with its area. Joined: explains 60.7% on held-out homes, +5.7 points on top of appraisal area, the largest gain of any location column. Keeping the area too lets a sub-area with few sales lean on its area. |
| 29 | `LevyCode`, tax district code | **Remove.** | 364 codes, median 130 sales per code. Explains 46.8% on held-out homes but mostly repeats city (the most expensive codes are Medina, Clyde Hill and Mercer Island). |
| 30 | `PlatName`, subdivision name | **One-hot only the subdivision names with enough sales; every other name goes into one "other" column.** **Threshold: at least 100 sales (800 yes/no columns on the full training set), chosen after the cross-correlation check (D4, 25 October 2026).** Empty (8.5% of sales) keeps its own column. Reopened on 25 October 2026 (first decided: remove). | 10,708 names, median 19 sales per name. All names: +12.4 points on top of area + sub-area, +5.5 once the quarter-mile cell is known. At least 100 sales: names cover 36.7% of sales, +3.1 and +1.5 points. At least 50 sales: 58.4% of sales, +6.0 and +2.9 points. One-hot keeps a text column the professor's rule 3 would not let us drop without a reason. Column counts on the full training set: at least 100 sales gives 800 columns (names cover 43.7% of sales), at least 50 gives 2,145 (65.4%), at least 200 gives 220 (25.5%). The gains above count names on 4 of every 5 training homes, as the held-out test needs, so they list fewer names. |
| 31–34 | `Range`, `Township`, `Section`, `QuarterSection`, the survey grid | **Convert into an approximate map position: miles east and miles north, two numbers, each standardized.** The four grid columns are then removed. | Township alone explains 31.1% (correlation with the month-adjusted label 0.450); range 2.2%, section 1.9%, quarter 0.1%, because a section number repeats in every township. The fixed numbering of the grid turns each grid code into a position. Miles north: correlation with the label 0.337, month-adjusted 0.448; miles east: 0.053 and 0.087. Approximate, because the grid has irregular squares near water and borders. Not yet measured inside a model. **Dropped on 25 October 2026 by the 0.85 rule** (see "Final selection"): miles north 0.998 and miles east 0.996 with area + sub-area, which is closer to the label (0.59 against 0.34 and 0.05). |
| 35 | `CurrentZoning`, zoning code | **Remove.** | 399 codes, median 30 sales per code. Explains 38.6% on held-out homes and +4.6 points on top of area + sub-area. The same code name means different rules in different cities (unincorporated King County "R-6" counts homes per acre, Mercer Island "R-15" is a minimum lot size), the column holds today's zoning rather than the zoning at the time of sale, and lot area (decision 50) may carry part of the gain (not measured). |
| 36–39 | `MtRainier`, `Olympics`, `Cascades`, `Territorial`, the first four views | **One yes/no column (0 or 1) per view: 1 when the house has that view at any rating.** The 0–4 rating is not kept. | Yes/no keeps almost everything the rating explains: Mount Rainier 0.2% (rating 0.3%), Olympic Mountains 3.0% (3.1%), Cascade Mountains 1.6% (1.8%), territorial (open land) 8.2% (9.4%). With the view: Rainier +23.6% (0.73% of sales), Olympics +77.8% (2.10%), Cascades +52.5% (2.40%), territorial +57.1% (8.67%). |
| 40–44 | `SeattleSkyline`, `PugetSound`, `LakeWashington`, `LakeSammamish`, `SmallLakeRiverCreek`, the next five views | **One yes/no column (0 or 1) per view, the same as 36–39.** | Yes/no explains: Seattle skyline 1.9% (rating 2.0%), Puget Sound 2.0% (2.1%), Lake Washington 6.4% (6.9%), Lake Sammamish 1.7% (1.9%), small lake, river or creek 0.2% (0.2%). With the view: skyline +116.6% (0.78% of sales), Puget Sound +55.9% (2.50%), Lake Washington +121.5% (2.36%), Lake Sammamish +128.3% (0.60%), small lake, river or creek +35.1% (0.72%). |
| 45 | `OtherView`, other view | **One yes/no column, the same as the other views.** | Yes/no explains 0.6% (rating 0.6%). With the view: +65.9% (0.57% of sales). |
| 46 | `WfntLocation`, waterfront: which water | **One-hot, 7 yes/no columns:** not on the water, Lake Washington, Lake Sammamish, Puget Sound, other lake, river (river or slough plus the Duwamish), other (Lake Union and the Ship Canal). | Explains 3.9% (correlation ratio 0.20). Lake Washington +389.1% (0.24% of sales), Lake Sammamish +326% (420 sales, kept as its own column although just under 0.1% of sales), Puget Sound +76.3%, other lake +30.7%, river or slough +0.6%. A fixed list of 8 place names, not free text, so one-hot rather than a word embedding (rule 3 allows either). |
| 47 | `WfntFootage`, waterfront length | **Keep as a number (feet), standardized.** 0 for houses not on the water. | 1.13% of sales have frontage: median 70 ft, top 1% over 395 ft, maximum 1,600 ft. Correlation with the label 0.079, month-adjusted 0.099. Among waterfront homes only, length shows no link with price (correlation −0.004). Having frontage matches decision 46 for all but 92 sales; checked again in the cross-correlation step. |
| 48 | `WfntBank`, waterfront bank height | **One-hot, 4 yes/no columns:** none (not waterfront), low (with "no bank" merged in, 32 sales), medium, high. | Explains 2.3% (correlation ratio 0.15). Low +115.7% (0.77% of sales), medium +31.2%, high +65.6%, no bank +288%. Only waterfront homes have a bank code, so it partly repeats decision 46 (cross-correlation step). |
| 49 | `TidelandShoreland`, tidelands or shorelands | **One yes/no column: 1 for any tidelands or shorelands code** (0.47% of sales). | Explains 2.4% (correlation ratio 0.15). House land plus the tidelands or shorelands +207.1% (0.44% of sales), house land only +223% (129 sales), tidelands or shorelands only −14% (12 sales). **Dropped on 25 October 2026 by the 0.85 rule** (see "Final selection"): 0.986 with which water (decision 46), which is closer to the label (0.15 against 0.12). |
| 50 | `SqFtLot`, lot area | **Natural log, then standardized.** | Skew 14.53 in plain square feet, 0.70 after the log; the largest standardized value drops from 65.3 to 6.5. Correlation with the label 0.102 (plain 0.100), month-adjusted 0.182 (plain 0.122); a straight line explains 3.3% against 1.5%. Across the county the price follows a U shape (small city lots and big lots both above typical), which is mostly location: inside the same appraisal area, month and living area, doubling the lot adds +4.4% (correlation 0.315). |
| 51 | `Topography`, hilly or uneven lot | **One yes/no column.** | Hilly or uneven: 8.38% of sales, +31.7%; flat −2.1%. Explains 2.8% (correlation ratio 0.17). |
| 52 | `TrafficNoise`, traffic noise | **Keep as a number 0 to 3 (none, moderate, high, extreme), standardized.** | The levels are ordered and the price falls steadily: none +0.2%, moderate −0.7%, high −3.6%, extreme −4.2%. Explains 0.0% across the county; busy roads are common in expensive city areas, so the effect may only show once location is known (not measured). One column. |
| 53 | `AirportNoise`, airport noise | **One yes/no column.** | Yes: 2.21% of sales, −34.2%. Explains 2.1%; correlation with the label −0.114, month-adjusted −0.144. Part of the gap is probably location (homes near the airport sit in cheaper south King County; not separated). |
| 54 | `AdjacentGreenbelt`, next to a green belt | **One yes/no column** (the text values Y and N become 1 and 0). | Yes: 3.50% of sales, +36.7%. Explains 1.1%; correlation with the label 0.068, month-adjusted 0.105. |
| R1 | `StreetName`, street name | **One-hot only the street names with enough sales, every other name in "other".** **Threshold: at least 200 sales (404 yes/no columns on the full training set), chosen after the cross-correlation check (D5, 25 October 2026).** Kept after the cross-correlation check. Reopened on 25 October 2026 from the earlier chat removals. | 2,404 names. At least 50 sales: 781 columns, cover 92.9% of sales, +7.0 points on top of area + sub-area, +2.7 once the quarter-mile cell is known. At least 100: 521 columns, 87.4%, +6.6 and +2.6. At least 200: 365 columns, 81.1%, +6.1 and +2.4. 85% of the names with at least 100 sales appear in more than one city, so one column can mix streets from different cities. Column counts on the full training set: at least 50 sales gives 879 columns (94.2% of sales), 100 gives 604 (89.6%), 200 gives 404 (83.1%); the counts above come from 4 of every 5 training homes. |
| R2 | `StreetType`, street type | **One-hot, 15 yes/no columns:** the 14 values with at least 0.1% of sales (avenue, street, place, court, way, drive, lane, road, boulevard, terrace, circle, parkway, loop, empty) plus "other" for the 9 rarer types. Kept after the cross-correlation check. Reopened on 25 October 2026 from the earlier chat removals. | +1.8 points on top of area + sub-area, +1.0 once the quarter-mile cell is known. Parkway +140.1% (0.14% of sales), terrace +33.2%, boulevard +20.6%, drive +19.9%, way +17.2%, avenue +0.2%, street −3.6%. |

### Duplicate columns removed (D1 and D2, 25 October 2026)

**Superseded the same evening by "Final selection" below:** once city and appraisal area left, only 2 column pairs still reached 0.85.

When two yes/no columns from different groups have a correlation of 0.95 or more, one of the two is dropped: the one from the more detailed group, in the order city, appraisal area, area + sub-area, street name, subdivision (least to most detailed). For the waterfront, "waterfront bank = none" is dropped, because it repeats "waterfront = not on the water" (0.982). With the thresholds of D4 and D5, 18 pairs reach 0.95 and 17 columns go:

- 11 area + sub-area columns: 29-2, 40-1, 41-3, 64-5, 84-2 and 90-1 (each the only sub-area of its area, so an exact copy of the area column), and 70-2, 75-10, 87-17, 87-19, 94-4 (near copies of Duvall, Snoqualmie, Algona, Pacific and Carnation).
- 3 appraisal area columns: 41, 56 and 86 (near copies of Enumclaw, Maple Valley and Covington).
- 2 subdivision columns: Broadmoor (an exact copy of sub-area 14-6) and Blue Ridge Addition (0.951 with sub-area 39-2).
- waterfront bank = none.

Result before D3: 1,736 columns from the 60 features. D3 then drops one baseline column per category group.

### Final selection (25 October 2026)

Zeyad's rules, applied in this order on the training sales:

1. **Features at 0.85 or more with each other:** drop the one weaker with the label, strongest pair first. Dropped 5 features: appraisal area (1.000 with area + sub-area), miles north (0.998), miles east (0.996), tidelands or shorelands (0.986 with which water) and city (0.868 with area + sub-area).
2. **Features at 0.85 or more with the label:** none. The highest is the sale date at 0.641.
3. **Bring back a dropped column** when it is under 0.85 with all of the 60 features and between 0.60 and 0.80 with the label. Only the parcel number (first half, 0.662) falls in the label window, and it is 0.998 with miles north, so nothing came back.
4. **One baseline column per category group** (D3): 15 columns.
5. **Columns at 0.85 or more with each other:** drop the one weaker with the label. Two pairs: subdivision Broadmoor (1.000 with sub-area 14-6) and subdivision Blue Ridge Addition (0.951 with sub-area 39-2).
6. **Check:** no column is an exact combination of other columns (smallest eigenvalue 0.0067); the most predictable column is 95.9% predicted by the others.

**Result: 55 features, 1,604 columns** (17 numbers, 23 yes/no, 1,564 one-hot) for 426,135 training sales and 106,755 test sales. Every list, threshold, baseline, average and spread is learned from the training sales only.

### First models (25 October 2026)

Fitted on 4 of every 5 training homes and scored on the 5th (85,226 sales of homes never seen in fitting). The test sales stay sealed until the final model is chosen.

| Model | R² (log price) | Typical error (RMSE, log price) | Median % off the sale price | Within 10% of the price |
|---|---|---|---|---|
| Median baseline | −0.005 | 0.646 | 41.7% | 11.3% |
| Linear regression | 0.900 | 0.204 | 13.0% | 39.7% |

Linear regression scores almost the same on the homes it was fitted on (R² 0.902), so it is not overfitting.

### Raised but not chosen

- Month of the year as an extra input (to learn seasons).
- Sale year as categories, for the straight-line baseline model only (lets that model follow the 2008 crash).

## Flagged for later checks

- **Cross-correlation check (P4):** thresholds settled (subdivision 100 sales, street name 200), duplicate columns removed (D1, D2). Mirror pairs inside one group: one baseline column per group (D3).
- **211 very cheap sales** (under $89,756) sit below the lower fence on the log scale. Check them in the "price does not fit the house" step. Nothing removed yet.
- **Basement split** (proposal, not decided): above-ground living area, finished basement, basement garage (goes into total garage area), and other unfinished basement. The basement garage sits inside the unfinished part in all 113,908 buildings that have one.

## Decided earlier in chat (24 October 2026)

These decisions were made on groups of columns and are also stored on the King County Feature Cards page.

### Transform (1)

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Land use and zoning | `PresentUse` | Present use (the property type) | use present use as the property type, grouping rare types into Other. **Encoding:** 6 yes/no columns: single family 89.06%, townhouse 8.99%, duplex 0.84%, single family in a commercial or industrial zone 0.56%, group home 0.31%, triplex 0.23%. The 14 rare types together stay under 0.1% of sales, so they join single family instead of forming a near-empty "other" column. |

### Keep (13)

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Lot and services | `WaterSystem` | Water supply | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column, private water supply (1.96% of sales); the rare codes (0.01%) count as water district. |
| Lot and services | `SewerSystem` | Sewer type | kept (only the no-effect group was removed); small price difference. **Encoding:** one yes/no column, private sewer (12.00% of sales); the rare codes count as public. |
| Lot and services | `Access` | Road access | kept; homes with this feature sold well above or below the typical home. **Encoding:** 3 yes/no columns: public 94.39%, private 5.49%, other 0.12% (restricted plus rare codes). |
| Lot and services | `StreetSurface` | Street surface | kept (only the no-effect group was removed); small price difference. **Encoding:** 3 yes/no columns: paved 97.60% (rare codes merged in), gravel 1.76%, code 0 (no name in the lookup file) 0.64%. |
| Lot and services | `InadequateParking` | Parking adequate or not | kept; homes with this feature sold well above or below the typical home. **Encoding:** 3 yes/no columns: adequate 71.21%, code 0 (no name in the lookup file) 28.34%, inadequate 0.45%. |
| Nuisances | `PowerLines` | Power lines nearby | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `AdjacentGolfFairway` | Next to a golf course | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `Easements` | Easement on the land | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `DeedRestrictions` | Deed restrictions | kept (only the no-effect group was removed); small price difference. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `NativeGrowthProtEsmt` | Protected native-plants strip | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `OtherDesignation` | Other legal designation | kept; homes with this feature sold well above or below the typical home. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `DNRLease` | Land leased from the state | kept (only the no-effect group was removed); small price difference. **Encoding:** one yes/no column (Y = 1). |
| Legal restrictions | `WaterProblems` | Water problems | kept (only the no-effect group was removed); small price difference. **Encoding:** one yes/no column (Y = 1). |

### Remove (81)

Street name and street type were reopened on 25 October 2026 and moved to rows R1 and R2 above.

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Sale record: filters and paperwork | `ExciseTaxNbr` | Sale form number | A form number unique to each sale (426,259 numbers for 426,259 training sales). It never repeats on a new sale, so it says nothing about the house. |
| Sale record: filters and paperwork | `Major` | Parcel number (first half) | removed as a model input, but kept in the data as the key that joins the files and splits train and test by home. |
| Sale record: filters and paperwork | `Minor` | Parcel number (second half) | removed as a model input, but kept in the data as the key that joins the files and splits train and test by home. |
| Sale record: filters and paperwork | `RecordingNbr` | Deed recording number | The deed's recording number: 414,971 different numbers, almost one per sale. It identifies the paperwork, not the house. |
| Sale record: filters and paperwork | `Volume` | Old record book volume | Empty for all 426,259 training sales, so the column holds no information. It points to the old paper record books. |
| Sale record: filters and paperwork | `Page` | Old record book page | Empty for all 426,259 training sales, so the column holds no information. It points to the old paper record books. |
| Sale record: filters and paperwork | `PlatNbr` | Subdivision map number (sale record) | Empty for all 426,259 training sales, so the column holds no information. Subdivision details come from the parcel file instead. |
| Sale record: filters and paperwork | `PlatType` | Subdivision map type (sale record) | Empty for all 426,259 training sales, so the column holds no information. Subdivision details come from the parcel file instead. |
| Sale record: filters and paperwork | `PlatLot` | Lot number (sale record) | Empty for all 426,259 training sales, so the column holds no information. The parcel file has its own lot number (removed below). |
| Sale record: filters and paperwork | `PlatBlock` | Block number (sale record) | Empty for all 426,259 training sales, so the column holds no information. The parcel file has its own block number (removed below). |
| Sale record: filters and paperwork | `SellerName` | Seller's name | Private people's and companies' names: 344,661 different names, median 1 sale per name, so a name almost never repeats on a new home. We keep personal names out of the model (the King County licence restricts the use of lists of individuals), and a name is not a property of the house. It explains 10.1% on held-out homes, probably because builders sell many new houses (not checked); year built already marks new houses. |
| Sale record: filters and paperwork | `BuyerName` | Buyer's name | Private people's and companies' names: 407,458 different names, median 1 sale per name; explains 0.7% on held-out homes. Personal names stay out of the model (the King County licence restricts the use of lists of individuals). |
| Sale record: filters and paperwork | `PropertyType` | Property type on the sale form | replaced by present use as the property type. |
| Sale record: filters and paperwork | `PrincipalUse` | Principal use (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleInstrument` | Deed type | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleReason` | Sale reason (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PropertyClass` | Property class (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleWarning` | Sale warnings (sale filter) | Empty for all 426,259 training sales, so the column holds no information. The clean-sale filter keeps only sales with no warning. |
| Sale record: filters and paperwork | `AFForestLand` | Sale form: forest land? | A yes/no box on the sale form (forest land): the same answer for 93.2% of sales, 6.8% empty. Explains 0.0% and +0.2 points on top of area + sub-area. |
| Sale record: filters and paperwork | `AFCurrentUseLand` | Sale form: farm or open-space programme? | A yes/no box on the sale form (farm or open-space programme): the same answer for 93.2% of sales, 6.8% empty. Explains 0.0% and +0.2 points on top of area + sub-area. |
| Sale record: filters and paperwork | `AFNonProfitUse` | Sale form: non-profit use? | A yes/no box on the sale form (non-profit use): the same answer for 93.2% of sales, 6.8% empty. Explains 0.0% and +0.2 points on top of area + sub-area. |
| Sale record: filters and paperwork | `AFHistoricProperty` | Sale form: historic property? | A yes/no box on the sale form (historic property): the same answer for 93.2% of sales, 6.8% empty. Explains 0.0% and +0.2 points on top of area + sub-area. |
| Size | `SqFt1stFloor` | First-floor area | the floors add up to the total living area. |
| Size | `SqFtHalfFloor` | Half-floor area | the floors add up to the total living area. |
| Size | `SqFt2ndFloor` | Second-floor area | the floors add up to the total living area. |
| Size | `SqFtUpperFloor` | Upper-floor area | the floors add up to the total living area. |
| Size | `SqFtUnfinFull` | Unfinished full-floor area | the county's own bookkeeping; almost every house has the same value. |
| Size | `SqFtUnfinHalf` | Unfinished half-floor area | the county's own bookkeeping; almost every house has the same value. |
| Size | `SqFtEnclosedPorch` | Enclosed porch area | weak extra that adds little once the neighbourhood is known. |
| Rooms and layout | `NbrLivingUnits` | Number of living units | the county's own bookkeeping; almost every house has the same value. |
| Rooms and layout | `BldgNbr` | Building number on the lot | no effect on price (same for every sale, empty, a record number or a name). |
| Rooms and layout | `DaylightBasement` | Basement with windows | weak extra that adds little once the neighbourhood is known. |
| Age, quality and condition | `BldgGradeVar` | Grade adjustment | the county's own bookkeeping; almost every house has the same value. |
| Age, quality and condition | `Condition` | House condition (poor to very good) | weak extra that adds little once the neighbourhood is known. |
| Age, quality and condition | `PcntComplete` | Percent complete | the county's own bookkeeping; almost every house has the same value. |
| Age, quality and condition | `Obsolescence` | Obsolescence | the county's own bookkeeping; almost every house has the same value. |
| Age, quality and condition | `PcntNetCondition` | Percent net condition | the county's own bookkeeping; almost every house has the same value. |
| Age, quality and condition | `AddnlCost` | Additional cost | the county's own bookkeeping; almost every house has the same value. |
| Systems and finishes | `BrickStone` | Percent brick or stone outside | weak extra that adds little once the neighbourhood is known. |
| Location | `SpecArea` | Special appraisal area | the county's own bookkeeping; almost every house has the same value. |
| Location | `SpecSubArea` | Special appraisal sub-area | the county's own bookkeeping; almost every house has the same value. |
| Location | `PlatLot (parcel file)` | Lot number (parcel record) | The lot number inside a subdivision map: 4,763 values, median 2 sales each. The same lot number repeats in every subdivision, so it means nothing without the subdivision name (decision 30). Explains 0.9% on held-out homes. |
| Location | `PlatBlock (parcel file)` | Block number (parcel record) | The block number inside a subdivision map: 599 values, 59.0% empty. The same block number repeats in every subdivision; explains 1.2% on held-out homes. |
| Location | `Address` | Full address | One address per home: 262,724 addresses, median 1 sale per address. A new home never shares its address with a training home, so the model cannot learn from it (explains 0.1% on held-out homes). Location comes from appraisal area, sub-area, city and the grid position. |
| Location | `BuildingNumber` | House number | The house number: 29,732 values, and the same number sits on unrelated streets across the county. Explains −5.8% on held-out homes, worse than guessing the average. |
| Location | `Fraction` | Fraction in the house number | A letter or fraction added to a house number: empty for 96.0% of sales. Adds +0.6 points on top of area + sub-area. |
| Location | `DirectionPrefix` | Direction before the street name | The direction before the street name (NE, SE and so on), which tells the quarter of a city: 9 values, 59.9% empty. Explains 6.2% alone but only +0.5 points on top of area + sub-area, which already holds that location. |
| Location | `DirectionSuffix` | Direction after the street name | The direction after the street name (NE, SE and so on): 10 values, 43.2% empty. Explains 8.9% alone but only +0.7 points on top of area + sub-area, which already holds that location. |
| Location | `PropName` | Property name | The name of a named property: empty for 93.9% of sales; 1,255 names with a median of 2 sales each. Adds +0.6 points on top of area + sub-area. |
| Land use and zoning | `HBUAsIfVacant` | Best use if the land were empty | the county's own bookkeeping; almost every house has the same value. |
| Land use and zoning | `HBUAsImproved` | Best use as it stands now | the county's own bookkeeping; almost every house has the same value. |
| Land use and zoning | `PropType` | Parcel property type | "R" (residential) for 100.0% of training sales, because the clean-sale filter keeps only houses. It carries nothing. |
| Land use and zoning | `CurrentUseDesignation` | Farm or open-space tax programme | the county's own bookkeeping; almost every house has the same value. |
| Land use and zoning | `HistoricSite` | Historic site | the county's own bookkeeping; almost every house has the same value. |
| Views | `ViewUtilization` | House designed for its view | weak extra that adds little once the neighbourhood is known. |
| Waterfront | `WfntPoorQuality` | Poor-quality waterfront | small waterfront detail that explains about 0%. |
| Waterfront | `WfntRestrictedAccess` | Restricted waterfront access | small waterfront detail that explains about 0%. |
| Waterfront | `WfntAccessRights` | Rights to someone else's waterfront | small waterfront detail that explains about 0%. |
| Waterfront | `WfntProximityInfluence` | Near water but not on it | small waterfront detail that explains about 0%. |
| Lot and services | `RestrictiveSzShape` | Lot too small or oddly shaped | the county's own bookkeeping; almost every house has the same value. |
| Lot and services | `PcntUnusable` | Percent of lot unusable | the county's own bookkeeping; almost every house has the same value. |
| Lot and services | `Unbuildable` | Lot can't be built on | the county's own bookkeeping; almost every house has the same value. |
| Lot and services | `LotDepthFactor` | Lot depth factor | the county's own bookkeeping; almost every house has the same value. |
| Lot and services | `NbrBldgSites` | Number of building sites | the county's own bookkeeping; almost every house has the same value. |
| Nuisances | `OtherNuisances` | Other nuisances | A yes/no flag: "no" for 97.1% of sales. Explains 0.0% and +0.1 points on top of area + sub-area. |
| Legal restrictions | `DevelopmentRightsPurch` | Development rights sold | no effect on price (same for every sale, empty, a record number or a name). |
| Legal restrictions | `TranspConcurrency` | Road capacity problems | no effect on price (same for every sale, empty, a record number or a name). |
| Legal restrictions | `OtherProblems` | Other problems | no effect on price (same for every sale, empty, a record number or a name). |
| Legal restrictions | `Contamination` | Contaminated land | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `HundredYrFloodPlain` | Flood zone | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `SeismicHazard` | Earthquake zone | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `LandslideHazard` | Landslide zone | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `SteepSlopeHazard` | Steep slope | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `ErosionHazard` | Erosion zone | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `CoalMineHazard` | Old coal mine area | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `CriticalDrainage` | Critical drainage area | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `LandfillBuffer` | Near a landfill | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `Stream` | Stream | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `Wetland` | Wetland | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `SpeciesOfConcern` | Protected species habitat | says 'no' for every property in the county, so it carries nothing. |
| Hazards and environment | `SensitiveAreaTract` | Sensitive area | says 'no' for every property in the county, so it carries nothing. |
