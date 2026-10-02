# import pandas as pd

# # Load data
# df = pd.read_csv("properties_2017.csv")

# col = "parcelid"
# label = "taxvaluedollarcnt"

# print("="*80)
# print(f"ANALYSIS OF {col}")
# print("="*80)

# # Basic stats
# print("\nBASIC INFORMATION")
# print(f"Rows: {len(df):,}")
# print(f"Missing: {df[col].isna().sum():,}")
# print(f"Unique values: {df[col].nunique():,}")

# unique_ratio = df[col].nunique() / len(df)

# print(f"Unique Ratio: {unique_ratio:.4f}")

# # Duplicate parcel IDs
# duplicate_count = df.duplicated(col).sum()

# print(f"Duplicate IDs: {duplicate_count:,}")

# # Frequency distribution
# freq = df[col].value_counts()

# print("\nFREQUENCY ANALYSIS")
# print(f"Max occurrences of same ID: {freq.max()}")
# print(f"IDs appearing once: {(freq == 1).sum():,}")
# print(f"IDs appearing >1 time: {(freq > 1).sum():,}")

# # Relationship with target
# temp = df[[col, label]].dropna()

# corr = temp[col].corr(temp[label])

# print("\nCORRELATION WITH LABEL")
# print(f"Pearson correlation: {corr:.6f}")

# # Check whether duplicated parcel IDs have different labels
# grouped = df.groupby(col)[label].nunique()

# conflicting = (grouped > 1).sum()

# print("\nMULTIPLE TARGET VALUES PER PARCEL")
# print(f"Parcel IDs with multiple target values: {conflicting:,}")

# # Heuristic decision
# print("\nPRELIMINARY DECISION")

# if unique_ratio > 0.95:
#     print("✓ Appears to be an identifier.")
#     print("✓ Very high cardinality.")
#     print("✓ Not suitable for one-hot encoding.")
#     print("✓ Usually dropped before modeling.")
# else:
#     print("Further investigation needed.")










import pandas as pd

df = pd.read_csv("properties_2017.csv")

# col = "airconditioningtypeid"
# label = "taxvaluedollarcnt"

# print("=" * 80)
# print(f"ANALYSIS OF {col}")
# print("=" * 80)

# # ============================================================
# # 1. BASIC INFORMATION
# # ============================================================

# print("\nBASIC INFORMATION")

# print(f"Rows: {len(df):,}")
# print(f"Data type: {df[col].dtype}")

# missing = df[col].isna().sum()
# missing_pct = missing / len(df) * 100

# print(f"Missing: {missing:,} ({missing_pct:.2f}%)")
# print(f"Non-missing: {len(df) - missing:,} ({100-missing_pct:.2f}%)")

# print(f"Unique categories: {df[col].nunique(dropna=True)}")

# # ============================================================
# # 2. CATEGORY FREQUENCIES
# # ============================================================

# print("\nCATEGORY FREQUENCIES")

# counts = df[col].value_counts(dropna=False)

# for category, count in counts.items():

#     if pd.isna(category):
#         name = "MISSING"
#     else:
#         name = str(category)

#     percentage = count / len(df) * 100

#     print(
#         f"{name:>10} : "
#         f"{count:>10,} "
#         f"({percentage:6.2f}%)"
#     )

# # ============================================================
# # 3. TARGET BY CATEGORY
# # ============================================================

# print("\nTARGET DISTRIBUTION BY AC TYPE")

# target_stats = (
#     df.groupby(col, dropna=False)[label]
#     .agg(
#         count="count",
#         mean="mean",
#         median="median",
#         std="std"
#     )
#     .sort_values("count", ascending=False)
# )

# print(target_stats)

# # ============================================================
# # 4. MISSING VS NON-MISSING
# # ============================================================

# print("\nMISSINGNESS VS TARGET")

# df["_ac_missing"] = df[col].isna()

# missing_stats = (
#     df.groupby("_ac_missing")[label]
#     .agg(
#         count="count",
#         mean="mean",
#         median="median",
#         std="std"
#     )
# )

# print(missing_stats)

# # ============================================================
# # 5. RARE CATEGORIES
# # ============================================================

# print("\nRARE CATEGORIES")

# non_missing_counts = df[col].value_counts()

# for category, count in non_missing_counts.items():

#     percentage = count / len(df) * 100

#     if percentage < 1:
#         print(
#             f"Category {category}: "
#             f"{count:,} rows ({percentage:.3f}%)"
#         )

# # ============================================================
# # 6. CATEGORY TARGET DIFFERENCES
# # ============================================================

# print("\nTARGET RANGE BY CATEGORY")

# for category in sorted(df[col].dropna().unique()):

#     values = df.loc[df[col] == category, label].dropna()

#     if len(values) > 0:

#         print(
#             f"Category {category}: "
#             f"min={values.min():,.0f}, "
#             f"median={values.median():,.0f}, "
#             f"mean={values.mean():,.0f}, "
#             f"max={values.max():,.0f}"
#         )

# # ============================================================
# # CLEANUP
# # ============================================================

# df.drop(columns="_ac_missing", inplace=True)



















# col = "basementsqft"
# label = "taxvaluedollarcnt"

# print("=" * 80)
# print(f"ANALYSIS OF {col}")
# print("=" * 80)

# # Basic information
# print("\nBASIC INFORMATION")
# print(f"Rows: {len(df):,}")
# print(f"Missing: {df[col].isna().sum():,} "
#       f"({df[col].isna().mean()*100:.2f}%)")
# print(f"Unique values: {df[col].nunique():,}")

# # Statistics
# print("\nSTATISTICS")
# print(df[col].describe())

# # Additional statistics
# print(f"Skewness: {df[col].skew():.3f}")

# # Zero values
# zero_count = (df[col] == 0).sum()

# print(f"Zero values: {zero_count:,} "
#       f"({zero_count/len(df)*100:.2f}%)")

# # Negative values
# negative_count = (df[col] < 0).sum()

# print(f"Negative values: {negative_count:,}")

# # Correlation
# temp = df[[col, label]].dropna()

# print("\nRELATIONSHIP WITH LABEL")

# print(f"Pearson correlation: "
#       f"{temp[col].corr(temp[label]):.4f}")

# print(f"Spearman correlation: "
#       f"{temp[col].corr(temp[label], method='spearman'):.4f}")

# # Target comparison: basement value vs no basement
# print("\nTARGET BY BASEMENT STATUS")

# df["_basement_status"] = df[col].isna()

# print(
#     df.groupby("_basement_status")[label]
#       .agg(["count", "mean", "median"])
# )

# # Outliers
# q1 = df[col].quantile(0.25)
# q3 = df[col].quantile(0.75)
# iqr = q3 - q1

# lower = q1 - 1.5 * iqr
# upper = q3 + 1.5 * iqr

# outliers = ((df[col] < lower) | (df[col] > upper)).sum()

# print("\nOUTLIERS")
# print(f"IQR lower bound: {lower:.2f}")
# print(f"IQR upper bound: {upper:.2f}")
# print(f"Outliers: {outliers:,}")

# # Cleanup
# df.drop(columns="_basement_status", inplace=True)









# col = "bathroomcnt"
# label = "taxvaluedollarcnt"

# s = df[col]

# print("=== BASIC INFO ===")
# print("Rows:", len(df))
# print("Missing:", s.isna().sum())
# print("Missing %:", s.isna().mean() * 100)
# print("Unique values:", s.nunique())

# print("\n=== VALUE DISTRIBUTION ===")
# print(s.describe())

# print("\n=== FREQUENCIES ===")
# print(s.value_counts(dropna=False).sort_index().head(30))

# print("\n=== INVALID / UNUSUAL VALUES ===")
# print("Zero bathrooms:", (s == 0).sum())
# print("Negative bathrooms:", (s < 0).sum())
# print("Bathrooms > 10:", (s > 10).sum())
# print("Maximum:", s.max())

# print("\n=== TARGET RELATIONSHIP ===")
# temp = df[[col, label]].dropna()

# print("Pearson correlation:", temp[col].corr(temp[label], method="pearson"))
# print("Spearman correlation:", temp[col].corr(temp[label], method="spearman"))

# print("\n=== TARGET BY BATHROOM COUNT ===")
# target_by_bath = (
#     temp.groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )

# print(target_by_bath.head(30))



# col = "bedroomcnt"
# label = "taxvaluedollarcnt"

# s = df[col]

# print("=== BASIC INFO ===")
# print("Rows:", len(df))
# print("Missing:", s.isna().sum())
# print("Missing %:", s.isna().mean() * 100)
# print("Unique values:", s.nunique())

# print("\n=== VALUE DISTRIBUTION ===")
# print(s.describe())

# print("\n=== FREQUENCIES ===")
# print(s.value_counts(dropna=False).sort_index())

# print("\n=== INVALID / UNUSUAL VALUES ===")
# print("Zero bedrooms:", (s == 0).sum())
# print("Negative bedrooms:", (s < 0).sum())
# print("Bedrooms > 10:", (s > 10).sum())
# print("Maximum:", s.max())

# print("\n=== TARGET RELATIONSHIP ===")
# temp = df[[col, label]].dropna()

# print("Pearson correlation:", temp[col].corr(temp[label], method="pearson"))
# print("Spearman correlation:", temp[col].corr(temp[label], method="spearman"))

# print("\n=== TARGET BY BEDROOM COUNT ===")
# print(
#     temp.groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )




# col = "buildingqualitytypeid"
# label = "taxvaluedollarcnt"

# s = df[col]

# print("=== BASIC INFO ===")
# print("Rows:", len(df))
# print("Missing:", s.isna().sum())
# print("Missing %:", s.isna().mean() * 100)
# print("Unique values:", s.nunique())

# print("\n=== VALUE FREQUENCIES ===")
# print(s.value_counts(dropna=False).sort_index())

# print("\n=== TARGET BY QUALITY TYPE ===")
# temp = df[[col, label]].dropna()

# print(
#     temp.groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )

# print("\n=== ORDINAL RELATIONSHIP ===")
# print("Spearman correlation:",
#       temp[col].corr(temp[label], method="spearman"))












# print("=== BASIC COMPARISON ===")

# both = df[["bathroomcnt", "calculatedbathnbr"]]

# print("Bathroomcnt missing:",
#       df["bathroomcnt"].isna().sum())

# print("Calculatedbathnbr missing:",
#       df["calculatedbathnbr"].isna().sum())

# print("\n=== AGREEMENT ===")

# compare = both.dropna()

# print("Rows where both exist:", len(compare))

# print("Exact agreement:",
#       (compare["bathroomcnt"] == compare["calculatedbathnbr"]).mean() * 100,
#       "%")

# print("Different values:",
#       (compare["bathroomcnt"] != compare["calculatedbathnbr"]).sum())

# print("\n=== DIFFERENCES ===")

# diff = compare[
#     compare["bathroomcnt"] != compare["calculatedbathnbr"]
# ].copy()

# diff["difference"] = (
#     diff["bathroomcnt"] - diff["calculatedbathnbr"]
# )

# print(diff["difference"].value_counts().sort_index())

# print("\n=== CORRELATION ===")
# print(
#     compare["bathroomcnt"]
#     .corr(compare["calculatedbathnbr"])
# )








# 












# col = "calculatedfinishedsquarefeet"

# s = df[col]

# print("Missing %:", s.isna().mean() * 100)
# print(s.describe())

# print("\nZero:", (s == 0).sum())
# print("Negative:", (s < 0).sum())

# print("\nSkewness:", s.skew())

# temp = df[[col, "taxvaluedollarcnt"]].dropna()

# print("\nPearson:", temp[col].corr(temp["taxvaluedollarcnt"]))
# print("Spearman:", temp[col].corr(temp["taxvaluedollarcnt"], method="spearman"))




# col = "fips"
# label = "taxvaluedollarcnt"

# print("=== BASIC INFO ===")
# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique values:", df[col].nunique())

# print("\n=== FREQUENCIES ===")
# print(df[col].value_counts(dropna=False).sort_index())

# print("\n=== TARGET BY FIPS ===")
# temp = df[[col, label]].dropna()

# print(
#     temp.groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )





# col = "fireplacecnt"
# label = "taxvaluedollarcnt"

# print("=== BASIC INFO ===")
# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique values:", df[col].nunique())

# print("\n=== FREQUENCIES ===")
# print(df[col].value_counts(dropna=False).sort_index())

# print("\n=== TARGET BY FIREPLACE COUNT ===")
# print(
#     df[[col, label]]
#     .dropna()
#     .groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )














# print("Missing bathroomcnt:", df["bathroomcnt"].isna().sum())
# print("Missing fullbathcnt:", df["fullbathcnt"].isna().sum())

# both = df[["bathroomcnt", "fullbathcnt"]].dropna()

# print("\nRows where both exist:", len(both))
# print("Unique bathroomcnt → fullbathcnt combinations:")
# print(both.drop_duplicates().sort_values(["bathroomcnt", "fullbathcnt"]).to_string(index=False))
# col = "fullbathcnt"
# label = "taxvaluedollarcnt"

# print("=== BASIC INFO ===")
# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique values:", df[col].nunique())

# print("\n=== STATISTICS ===")
# print(df[col].describe())

# print("\n=== SKEWNESS ===")
# print(df[col].skew())

# print("\n=== FREQUENCIES ===")
# print(df[col].value_counts(dropna=False).sort_index())

# print("\n=== TARGET RELATIONSHIP ===")
# print("Pearson:", df[[col, label]].corr().iloc[0, 1])
# print("Spearman:", df[[col, label]].corr(method="spearman").iloc[0, 1])

# print("\n=== TARGET BY FULL BATH COUNT ===")
# print(
#     df[[col, label]]
#     .dropna()
#     .groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )




# col = "heatingorsystemtypeid"
# label = "taxvaluedollarcnt"

# print("=== BASIC INFO ===")
# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique observed:", df[col].nunique())

# print("\n=== FREQUENCIES ===")
# print(df[col].value_counts(dropna=False).sort_index())

# print("\n=== TARGET BY HEATING TYPE ===")
# print(
#     df[[col, label]]
#     .dropna()
#     .groupby(col)[label]
#     .agg(["count", "mean", "median"])
# )




# cols = ["latitude", "longitude"]
# label = "taxvaluedollarcnt"

# for col in cols:
#     print(f"\n{'='*50}")
#     print(col)
#     print("="*50)

#     print("Missing:", df[col].isna().sum())
#     print("Missing %:", df[col].isna().mean() * 100)
#     print("Unique:", df[col].nunique())

#     print("\nStatistics:")
#     print(df[col].describe())

#     print("\nSkewness:", df[col].skew())

#     print("\nTarget correlation:")
#     print("Pearson:", df[[col, label]].corr().iloc[0, 1])
#     print("Spearman:", df[[col, label]].corr(method="spearman").iloc[0, 1])

# print("\n=== LATITUDE/LONGITUDE CORRELATION ===")
# print(df[cols].corr())





# col = "lotsizesquarefeet"
# label = "taxvaluedollarcnt"

# print("=== BASIC INFO ===")
# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique:", df[col].nunique())

# print("\n=== STATISTICS ===")
# print(df[col].describe())

# print("\n=== SKEWNESS ===")
# print(df[col].skew())

# print("\n=== INVALID VALUES ===")
# print("Zeros:", (df[col] == 0).sum())
# print("Negative:", (df[col] < 0).sum())

# print("\n=== TARGET RELATIONSHIP ===")
# print("Pearson:", df[[col, label]].corr().iloc[0, 1])
# print("Spearman:", df[[col, label]].corr(method="spearman").iloc[0, 1])







# col = "propertycountylandusecode"

# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique:", df[col].nunique())

# print("\nTop categories:")
# print(df[col].value_counts(dropna=False).head(20))

# print("\nTarget by category:")
# print(
#     df.groupby(col)["taxvaluedollarcnt"]
#       .agg(["count", "mean", "median"])
#       .sort_values("count", ascending=False)
#       .head(20)
# )









# col = "propertylandusetypeid"

# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique:", df[col].nunique())

# print("\nCategory counts:")
# print(df[col].value_counts(dropna=False))

# print("\nTarget by category:")
# print(
#     df.groupby(col)["taxvaluedollarcnt"]
#       .agg(["count", "mean", "median"])
#       .sort_values("count", ascending=False)
# )











# col = "roomcnt"

# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique:", df[col].nunique())

# print("\nStatistics:")
# print(df[col].describe())

# print("\nValue counts:")
# print(df[col].value_counts(dropna=False).sort_index())

# print("\nTarget by room count:")
# print(
#     df.groupby(col)["taxvaluedollarcnt"]
#       .agg(["count", "mean", "median"])
#       .sort_index()
# )














# col = "yearbuilt"

# print("Missing:", df[col].isna().sum())
# print("Missing %:", df[col].isna().mean() * 100)
# print("Unique:", df[col].nunique())

# print("\nStatistics:")
# print(df[col].describe())

# print("\nValue counts (oldest years):")
# print(df[col].value_counts().sort_index().head(20))

# print("\nValue counts (newest years):")
# print(df[col].value_counts().sort_index().tail(20))

# print("\nTarget by year:")
# print(
#     df.groupby(col)["taxvaluedollarcnt"]
#       .agg(["count", "mean", "median"])
#       .sort_index()
# )








