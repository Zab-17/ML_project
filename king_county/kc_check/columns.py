"""Read-only: per-column empty/zero shares on clean sales joined to one-house parcels."""
import sys
import pandas as pd

KING = sys.argv[1]

def read(name):
    frame = pd.read_csv(f"{KING}/{name}.csv", dtype=str, encoding="latin-1")
    frame = frame.apply(lambda column: column.str.strip())
    frame["parcel"] = frame["Major"].str.zfill(6) + "-" + frame["Minor"].str.zfill(4)
    return frame

sales = read("EXTR_RPSale")
price = pd.to_numeric(sales["SalePrice"], errors="coerce")
deeds = sales.groupby("ExciseTaxNbr")["parcel"].transform("nunique")
clean = sales[(sales["SaleReason"] == "1") & sales["SaleInstrument"].isin(["2", "3"])
              & (sales["SaleWarning"].fillna("") == "") & (deeds == 1) & (sales["PrincipalUse"] == "6")
              & (sales["PropertyClass"] == "8") & (price >= 10_000)]
buildings = read("EXTR_ResBldg")
single = buildings[buildings.groupby("parcel")["parcel"].transform("size") == 1]
parcels = read("EXTR_Parcel")
joined = (clean.merge(single, on="parcel", suffixes=("", "_bldg"))
               .merge(parcels, on="parcel", suffixes=("", "_parcel")))
print("rows:", f"{len(joined):,}")
source = {}
for file_name, frame in [("sales", sales), ("building", buildings), ("parcel", parcels)]:
    for column in frame.columns:
        source.setdefault(column, file_name)
for column in joined.columns:
    if column == "parcel":
        continue
    values = joined[column].fillna("")
    empty = (values == "").mean()
    zero = values.isin(["0", "0.0", "0.00"]).mean()
    distinct = values[values != ""].nunique()
    example = values[(values != "") & ~values.isin(["0"])].head(1).tolist()
    print(f"{source.get(column.replace('_bldg','').replace('_parcel',''), '?'):8s} | {column:24s} | empty {empty:6.1%} | zero {zero:6.1%} | distinct {distinct:>7,} | e.g. {example[0][:28] if example else '-'}")
