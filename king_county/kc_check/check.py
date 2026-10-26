"""Read-only feasibility check of the King County sales and building files."""
import sys
import pandas as pd

KING = sys.argv[1]
MIN_PRICE = 10_000
YEARS = [1990, 2000, 2010, 2015, 2020]

def key(frame):
    return frame["Major"].str.strip().str.zfill(6) + "-" + frame["Minor"].str.strip().str.zfill(4)

sales = pd.read_csv(f"{KING}/EXTR_RPSale.csv", dtype=str, encoding="latin-1",
                    usecols=["ExciseTaxNbr", "Major", "Minor", "DocumentDate", "SalePrice", "PrincipalUse",
                             "SaleInstrument", "SaleReason", "PropertyClass", "SaleWarning"])
for column in ["PrincipalUse", "SaleInstrument", "SaleReason", "PropertyClass"]:
    sales[column] = sales[column].str.strip()
print("all sale rows:", f"{len(sales):,}")
print("top SaleReason:", sales["SaleReason"].value_counts().head(4).to_dict())
print("top SaleInstrument:", sales["SaleInstrument"].value_counts().head(4).to_dict())
sales["price"] = pd.to_numeric(sales["SalePrice"], errors="coerce")
sales["date"] = pd.to_datetime(sales["DocumentDate"], format="%m/%d/%Y", errors="coerce")
sales["parcel"] = key(sales)
parcels_per_deed = sales.groupby("ExciseTaxNbr")["parcel"].transform("nunique")
clean = sales[(sales["SaleReason"] == "1") & sales["SaleInstrument"].isin(["2", "3"])
              & (sales["SaleWarning"].fillna("").str.strip() == "") & (parcels_per_deed == 1)
              & (sales["PrincipalUse"] == "6") & (sales["PropertyClass"] == "8")
              & (sales["price"] >= MIN_PRICE) & sales["date"].notna()]
print("clean residential sales:", f"{len(clean):,}", "| date range:", clean["date"].min().date(), "to", clean["date"].max().date())

buildings = pd.read_csv(f"{KING}/EXTR_ResBldg.csv", dtype=str, encoding="latin-1",
                        usecols=["Major", "Minor", "BldgNbr", "ZipCode", "YrBuilt", "YrRenovated"])
buildings["parcel"] = key(buildings)
per_parcel = buildings.groupby("parcel").size()
single = buildings[buildings["parcel"].map(per_parcel) == 1]
print("building rows:", f"{len(buildings):,}", "| parcels with exactly one house:", f"{len(single):,}",
      "| parcels with 2+ houses:", f"{(per_parcel > 1).sum():,}")

joined = clean.merge(single, on="parcel", how="inner")
joined["sale_year"] = joined["date"].dt.year
built = pd.to_numeric(joined["YrBuilt"], errors="coerce")
renovated = pd.to_numeric(joined["YrRenovated"], errors="coerce").fillna(0)
print("clean sales joined to a one-house parcel:", f"{len(joined):,}", "| distinct homes:", f"{joined['parcel'].nunique():,}")
for year in YEARS:
    recent = joined["sale_year"] >= year
    print(f"  sales from {year} on: {recent.sum():,}")
print("house built after its sale year (sale was of land or an older house):", f"{(built > joined['sale_year']).sum():,}")
print("house renovated after its sale year:", f"{(renovated > joined['sale_year']).sum():,}")
ok = (built <= joined["sale_year"]) & (renovated <= joined["sale_year"])
for year in YEARS:
    print(f"  usable (no later build or renovation) from {year} on: {(ok & (joined['sale_year'] >= year)).sum():,}")
zip_filled = joined["ZipCode"].fillna("").str.strip().str.match(r"^98\d{3}")
print("rows with a real 98xxx ZIP code:", f"{zip_filled.mean():.1%}")
