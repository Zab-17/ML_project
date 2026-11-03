"""The decision for every one of the 58 columns of the Zillow properties file, with its reason.

This is the Zillow version of the King County column cards. The measured numbers behind each reason (share
missing, number of values, strength with the label, strongest partner) are written next to the decision by
zillow_checks.py, in out/checks/column_check.csv.

"measured as" tells the association checks how to compare the raw column: a number (Pearson correlation) or a
category (correlation ratio against numbers, Cramer's V against categories).
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from dataclasses import dataclass

NUMBER, CATEGORY = "number", "category"
LABEL, KEEP, TRANSFORM, DROP, LEAK = "label", "kept", "kept after a transform", "dropped", "dropped: leaks the label"


@dataclass(frozen=True)
class RawColumn:
    name: str
    measured_as: str
    decision: str
    reason: str


NOT_IN_LOS_ANGELES = "Los Angeles never records it"

RAW_COLUMNS = [
    RawColumn("taxvaluedollarcnt", NUMBER, LABEL,
              "The label: the county's 2016 assessed value of the home, in dollars. The models predict its log."),
    # --- record numbers and columns that leak the label ---
    RawColumn("parcelid", NUMBER, DROP, "A record number; it never repeats on another home."),
    RawColumn("structuretaxvaluedollarcnt", NUMBER, LEAK,
              "Leaks the label: building value plus land value equals the label exactly."),
    RawColumn("landtaxvaluedollarcnt", NUMBER, LEAK,
              "Leaks the label: building value plus land value equals the label exactly."),
    RawColumn("taxamount", NUMBER, LEAK, "Leaks the label: the tax bill is about 1% of the assessed value."),
    RawColumn("taxdelinquencyflag", CATEGORY, DROP,
              "About the owner's unpaid tax, which is charged on the label; not a property of the home."),
    RawColumn("taxdelinquencyyear", NUMBER, DROP, "Same as the delinquency flag, and 98% empty."),
    RawColumn("assessmentyear", NUMBER, DROP, "Only 2016 assessments are kept (row rule 2), so it never changes."),
    # --- size ---
    RawColumn("calculatedfinishedsquarefeet", NUMBER, TRANSFORM, "Living area, as its log (skewed like the label)."),
    RawColumn("finishedsquarefeet12", NUMBER, DROP, "The same number as the living area wherever both exist."),
    RawColumn("finishedsquarefeet13", NUMBER, DROP, "Another version of the living area; 99.7% empty."),
    RawColumn("finishedsquarefeet15", NUMBER, DROP, "Another version of the living area; 93.6% empty."),
    RawColumn("finishedsquarefeet50", NUMBER, DROP, "First-floor area, part of the living area; 93.2% empty."),
    RawColumn("finishedsquarefeet6", NUMBER, DROP, "Another version of the living area; 99.3% empty."),
    RawColumn("finishedfloor1squarefeet", NUMBER, DROP, "First-floor area, part of the living area; 93.2% empty."),
    RawColumn("lotsizesquarefeet", NUMBER, TRANSFORM,
              "Lot area, as its log. Blank mostly for condominiums; filled with the training median."),
    RawColumn("basementsqft", NUMBER, DROP, "99.95% empty."),
    RawColumn("garagetotalsqft", NUMBER, KEEP,
              f"Garage area. {NOT_IN_LOS_ANGELES}; in Orange and Ventura only 4% and 1.5% are blank, so a blank "
              "there means no garage and becomes 0. The county and ZIP columns tell the two kinds of 0 apart."),
    RawColumn("garagecarcnt", NUMBER, DROP,
              "The same garages as the garage area, in the same homes; the area is the finer measure."),
    RawColumn("yardbuildingsqft17", NUMBER, DROP, "Patio area: recorded only in Ventura, 97% empty."),
    RawColumn("yardbuildingsqft26", NUMBER, DROP, "Storage shed area: 99.9% empty."),
    RawColumn("poolsizesum", NUMBER, DROP, "99% empty; the pool yes/no column carries the pools."),
    # --- rooms ---
    RawColumn("bathroomcnt", NUMBER, KEEP, "Bathrooms, halves included."),
    RawColumn("calculatedbathnbr", NUMBER, DROP, "The same number as bathrooms (correlation 1.000)."),
    RawColumn("fullbathcnt", NUMBER, DROP, "Part of bathrooms (correlation 0.987)."),
    RawColumn("threequarterbathnbr", NUMBER, DROP, f"Part of bathrooms; {NOT_IN_LOS_ANGELES}."),
    RawColumn("bedroomcnt", NUMBER, KEEP, "Bedrooms."),
    RawColumn("roomcnt", NUMBER, DROP,
              "Los Angeles writes 0 for every home; elsewhere it counts the rooms bedrooms and bathrooms count."),
    RawColumn("numberofstories", NUMBER, KEEP,
              f"Stories. {NOT_IN_LOS_ANGELES}; blanks get the training median (1 story)."),
    RawColumn("unitcnt", NUMBER, DROP,
              "Recorded only in Los Angeles; the property type (duplex, triplex, quadruplex) already gives it."),
    RawColumn("fireplacecnt", NUMBER, KEEP,
              f"Fireplaces. {NOT_IN_LOS_ANGELES}; in Orange and Ventura a blank means none and becomes 0."),
    RawColumn("fireplaceflag", CATEGORY, DROP, "99.8% empty; fireplaces carries the same homes."),
    # --- quality and building systems ---
    RawColumn("buildingqualitytypeid", CATEGORY, KEEP,
              "Building quality, 1 (best) to 12, one-hot. Recorded only in Los Angeles; elsewhere 'not recorded'."),
    RawColumn("heatingorsystemtypeid", CATEGORY, KEEP, "Heating system, one-hot; blank becomes 'not recorded'."),
    RawColumn("airconditioningtypeid", CATEGORY, KEEP, "Air conditioning, one-hot; blank becomes 'not recorded'."),
    RawColumn("architecturalstyletypeid", CATEGORY, DROP, "99.8% empty."),
    RawColumn("buildingclasstypeid", CATEGORY, DROP, "99.6% empty."),
    RawColumn("typeconstructiontypeid", CATEGORY, DROP, "99.8% empty."),
    RawColumn("storytypeid", CATEGORY, DROP, "99.95% empty."),
    RawColumn("decktypeid", CATEGORY, DROP, "Recorded only in Ventura; 99.4% empty."),
    RawColumn("hashottuborspa", CATEGORY, KEEP,
              "Hot tub or spa, yes/no. Recorded only in Orange, where 7% of homes have one."),
    RawColumn("poolcnt", CATEGORY, KEEP, "Pool, yes/no: 1 or blank in the file, so a blank means no pool."),
    RawColumn("pooltypeid2", CATEGORY, DROP, "Pool with spa: the pool and hot tub columns flag the same homes."),
    RawColumn("pooltypeid7", CATEGORY, DROP, "Pool without spa: the pool column flags the same homes."),
    RawColumn("pooltypeid10", CATEGORY, DROP, "Spa only: the hot tub column flags the same homes."),
    # --- property type ---
    RawColumn("propertylandusetypeid", CATEGORY, KEEP, "Property type, named from Zillow's dictionary, one-hot."),
    RawColumn("propertycountylandusecode", CATEGORY, KEEP,
              "The county's own land-use code, one-hot (Phase 2 used frequency encoding)."),
    RawColumn("propertyzoningdesc", CATEGORY, KEEP,
              "Zoning, one-hot; recorded only in Los Angeles, so 'not recorded' elsewhere. Phase 2 dropped it, but on "
              "held-out homes it is 0.41 with the label and only 0.51 with ZIP code and 0.68 with city, so it adds "
              "something the location columns do not (King County dropped zoning because there it repeated them)."),
    # --- location (Phase 2 dropped city, ZIP, neighbourhood and census tract, and used raw coordinates) ---
    RawColumn("latitude", NUMBER, TRANSFORM,
              "Becomes miles north of the centre of the training homes. Kept only if the correlation checks keep it."),
    RawColumn("longitude", NUMBER, TRANSFORM,
              "Becomes miles east of the centre of the training homes. Kept only if the correlation checks keep it."),
    RawColumn("fips", CATEGORY, KEEP, "County, named (Los Angeles, Orange, Ventura), one-hot."),
    RawColumn("regionidcounty", CATEGORY, DROP, "The same three counties as fips, under different numbers."),
    RawColumn("regionidcity", CATEGORY, KEEP, "City, one-hot."),
    RawColumn("regionidzip", CATEGORY, KEEP, "ZIP code, one-hot."),
    RawColumn("regionidneighborhood", CATEGORY, KEEP, "Neighbourhood, one-hot; blank becomes 'not recorded'."),
    RawColumn("rawcensustractandblock", CATEGORY, TRANSFORM,
              "Becomes the census tract (the block digits removed), one-hot."),
    RawColumn("censustractandblock", CATEGORY, DROP, "The same census block as rawcensustractandblock."),
    # --- age ---
    RawColumn("yearbuilt", NUMBER, KEEP, "Year built; blanks get the training median."),
]

COLUMN_BY_NAME = {column.name: column for column in RAW_COLUMNS}
