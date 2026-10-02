# Feature Selection and Pre-processing Summary

**Dataset:** Zillow Prize 2017 properties file (2,985,217 rows, 58 columns)
**Label:** `taxvaluedollarcnt` (assessed property value, regression task)

## 0. Row-level cleaning
- Rows with a missing label (1.15% of the data) are removed, since they cannot be used for training or evaluation and imputing a label would invent the answer.
- No rows are removed for outliers. Heavy-tailed numeric features are handled with a log transformation instead.

## 1. Columns we keep (12 features)

| Feature | Type | Why it is kept |
|---|---|---|
| `bathroomcnt` | Numeric | Only 0.1% missing, with a clear positive relationship with property value. |
| `bedroomcnt` | Numeric | Only 0.099% missing, with a meaningful positive relationship with property value. |
| `yearbuilt` | Numeric | 1.60% missing. Gives property age, and year order is meaningful. |
| `calculatedfinishedsquarefeet` | Numeric | 1.51% missing, with a strong relationship with property value. |
| `lotsizesquarefeet` | Numeric | Adds land-size information and has a measurable monotonic relationship with value. |
| `latitude` | Numeric | Precise location, finer than the county-level `fips`. |
| `longitude` | Numeric | Precise location, finer than the county-level `fips`. |
| `buildingqualitytypeid` | Categorical | Quality categories show substantially different value distributions. |
| `fips` | Categorical | The three counties show different property-value distributions. |
| `heatingorsystemtypeid` | Categorical | Heating-system categories show substantially different value distributions. |
| `propertylandusetypeid` | Categorical | 0.098% missing, with clear value differences across property types. |
| `propertycountylandusecode` | Categorical | Very low missingness, with meaningful value differences across its 234 categories. It gives more detailed land-use information than `propertylandusetypeid`. |

## 2. Transformations applied to the kept columns

| Column(s) | Transformation | Why |
|---|---|---|
| `bathroomcnt`, `bedroomcnt`, `yearbuilt` | Median imputation | Very few missing values, and the median is robust to extreme values. Fractional bathroom counts (1.5, 2.5) are preserved, and zero bedrooms is kept as a valid value. |
| `latitude`, `longitude` | Divide by 1,000,000, then median imputation | Zillow stores coordinates scaled by 1e6. Dividing gives standard degree values. |
| `calculatedfinishedsquarefeet` | Log transformation, then median imputation | Highly right-skewed, and the log makes the distribution more symmetric. |
| `lotsizesquarefeet` | Log transformation, then median imputation | Extremely right-skewed. No zero or negative values need special handling. |
| `buildingqualitytypeid` | One-hot encoding, with missing values as an "Unknown" category | The IDs are treated as categories because no lookup table defines their meaning, so we do not assume a higher ID means higher quality. |
| `fips` | One-hot encoding, with missing values as "Unknown" | Codes are identifiers, not quantities. |
| `heatingorsystemtypeid` | Map IDs to heating-system names, then one-hot encoding, with missing values as "Unknown" | Categories are nominal. |
| `propertylandusetypeid` | One-hot encoding, with missing values as "Unknown" | Only 16 observed categories, so one-hot is practical. |
| `propertycountylandusecode` | Frequency encoding, with missing values as "Unknown" | 234 categories would create too many one-hot columns. Frequencies are computed from the training data only, to prevent data leakage. |

## 3. Columns we drop and why

### Target leakage
| Column | Reason |
|---|---|
| `structuretaxvaluedollarcnt` | Structure value. Together with land value it sums exactly to the label. |
| `landtaxvaluedollarcnt` | Land value. Keeping it would directly reveal the label. |
| `taxamount` | 0.979 correlation with the label, and derived from the same tax assessment. |

### High missingness (more than 70% missing)
| Column(s) | Note |
|---|---|
| `airconditioningtypeid`, `architecturalstyletypeid`, `buildingclasstypeid` | Almost or entirely empty. |
| `basementsqft` | 99.95% missing, only 1,627 usable values. |
| `decktypeid` | 99.42% missing, and the only observed category is 66, so no variation. |
| `poolcnt`, `poolsizesum`, `pooltypeid2`, `pooltypeid7`, `pooltypeid10` | Approximately 99% missing. |
| `fireplacecnt` | 89.51% missing. |
| `fireplaceflag` | High missingness. |
| `garagecarcnt`, `garagetotalsqft`, `hashottuborspa` | More than 70% missing. |
| `storytypeid`, `threequarterbathnbr`, `typeconstructiontypeid` | High missingness. |
| `unitcnt`, `numberofstories` | High missingness. |
| `yardbuildingsqft17`, `yardbuildingsqft26` | High missingness. |
| `propertyzoningdesc` | High missingness. |
| `taxdelinquencyflag`, `taxdelinquencyyear` | 98.1% missing. |
| `censustractandblock` | High missingness. |

Keeping these would require assumptions about what the missing values mean (for example, whether a missing pool count means "no pool").

### Redundant with a kept feature
| Column(s) | Reason |
|---|---|
| `calculatedbathnbr` | Agrees with `bathroomcnt` in 100% of the 2,868,061 rows where both exist, and has more missing values. |
| `fullbathcnt` | Correlated with `bathroomcnt`, so it adds little new information. |
| `finishedfloor1squarefeet`, `finishedsquarefeet12`, `finishedsquarefeet13`, `finishedsquarefeet15`, `finishedsquarefeet50`, `finishedsquarefeet6` | Redundant square-footage measurements. `calculatedfinishedsquarefeet` is kept. |
| `rawcensustractandblock`, `regionidcity`, `regionidcounty`, `regionidneighborhood`, `regionidzip` | Location is already represented by `fips`, `latitude`, and `longitude`. These IDs would add several high-cardinality categorical features and need more complex encoding. |

### No useful variation or no predictive meaning
| Column | Reason |
|---|---|
| `parcelid` | Unique identifier, so it contains no trend to learn. |
| `assessmentyear` | About 99.9% of values are 2016, so there is almost no variation. |
| `roomcnt` | 77.6% of properties are recorded as having zero rooms, so it is unreliable as a room count. The related information is captured by `bedroomcnt` and `bathroomcnt`. |

## 4. Summary
- Total columns: 58 = 1 label + 12 kept features + 45 dropped columns