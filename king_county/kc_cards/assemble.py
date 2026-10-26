"""Writes the card answers and suggestions, then builds the page from the template."""
import json
import re
import sys

from labels import LABELS
from meanings import FAMILIES

STATS, TEMPLATE, OUT = sys.argv[1:4]
KEEP_FLAG_EXPLAINED = 0.005  # a yes/no flag is suggested for keeping when it explains at least 0.5% on its own

LOCATION_EXPECTED = ("Location is the strongest factor in every study we read. In Southern California, census tract alone "
                     "explained 30.6% of value differences and county only 0.4%: the finer the area, the more it explains.")
WATER_EXPECTED = ("No King County study has been researched yet. In San Diego, homes within 500 feet of the coast sold for about "
                  "102% more and homes within a mile for about 42% more, with almost no premium beyond 6 miles. A large "
                  "waterfront premium is expected here too.")
VIEW_EXPECTED = ("No King County study has been researched yet. In San Diego a view added about 4.9% (California study). "
                 "Water and mountain views are expected to add value here, more for better ratings.")
EXPECTED = {
    "SqFtTotLiving": "Positive with diminishing returns: within a neighbourhood a 1% larger home sells for about 0.43% to 0.56% more (San Diego and Sacramento studies). Size is worth more in expensive areas.",
    "SqFtLot": "Positive but small within a neighbourhood (1% more lot: +0.07% to +0.16%), flattening for big lots. Across a whole region it can even look negative, because big lots sit in cheaper outer areas. Land is most of a home's value in West Coast cities.",
    "Bedrooms": "Zero or negative once living area is held fixed: more bedrooms in the same space means smaller rooms (one small study found -6% per bedroom). On its own it looks positive because bigger homes have more bedrooms.",
    "BathFullCount": "Positive and small: about +1.8% to +2.4% per bathroom in California studies once location is known.",
    "Bath3qtrCount": "Positive and small: about +1.8% to +2.4% per bathroom in California studies once location is known.",
    "BathHalfCount": "Positive and small: about +1.8% to +2.4% per bathroom in California studies once location is known.",
    "YrBuilt": "U-shaped: value falls about 0.4% a year at first, then rises again for homes older than about 60 years (California and Dallas studies). A straight-line correlation understates a U-shape.",
    "YrRenovated": "No study researched. Expected positive when the renovation is recent, because a renovated house behaves like a younger one.",
    "BldgGrade": "No published estimate was found for quality grades. Expected strongly positive: the grade describes materials and workmanship.",
    "Condition": "No published estimate found. King County judges condition relative to the house's age, so its effect may only show once age is known.",
    "FinBasementGrade": "No published estimate found. Expected positive when a finished basement exists.",
    "SqFtGarageAttached": "Positive but small: garages add little once location is known (California studies).",
    "SqFtGarageBasement": "Positive but small: garages add little once location is known (California studies).",
    "FpSingleStory": "Positive in cold-climate studies (+12% in a Philadelphia industry brief, a weak source). No West Coast estimate found.",
    "FpMultiStory": "Positive in cold-climate studies (+12% in a Philadelphia industry brief, a weak source). No West Coast estimate found.",
    "FpFreestanding": "Positive in cold-climate studies (+12% in a Philadelphia industry brief, a weak source). No West Coast estimate found.",
    "FpAdditional": "Positive in cold-climate studies (+12% in a Philadelphia industry brief, a weak source). No West Coast estimate found.",
    "HeatSystem": "No West Coast estimate found. The value of heating and cooling features depends on climate.",
    "HeatSource": "No West Coast estimate found. Fuel type may partly stand in for neighbourhood and house age.",
    "TrafficNoise": "Negative: in Los Angeles a larger share of truck traffic on a nearby freeway lowered prices (a dissertation, not peer-reviewed). Noise effects usually show only once location is held fixed.",
    "AirportNoise": "Negative is expected. No airport-noise study was researched.",
    "Stories": "No published estimate found. Mostly a stand-in for size and layout.",
    "AdjacentGreenbelt": "Positive is expected: green space nearby carried a small premium in a Los Angeles study (+0.07% per 1% more green space).",
    "AdjacentGolfFairway": "Positive is expected; no golf-course study was researched.",
    "Topography": "No study researched. Hilly lots can cost more to build on but often come with views.",
}
for column in ["ZipCode", "DistrictName", "Area", "SubArea", "LevyCode", "PlatName", "CurrentZoning", "Range", "Township",
               "Section", "QuarterSection"]:
    EXPECTED[column] = LOCATION_EXPECTED
for column in ["MtRainier", "Olympics", "Cascades", "Territorial", "SeattleSkyline", "PugetSound", "LakeWashington",
               "LakeSammamish", "SmallLakeRiverCreek", "OtherView", "ViewUtilization"]:
    EXPECTED[column] = VIEW_EXPECTED
for column in ["WfntLocation", "WfntFootage", "WfntBank", "TidelandShoreland", "WfntPoorQuality", "WfntRestrictedAccess",
               "WfntAccessRights", "WfntProximityInfluence"]:
    EXPECTED[column] = WATER_EXPECTED

FILTER_COLUMNS = {"PrincipalUse", "SaleReason", "PropertyClass", "SaleWarning", "SaleInstrument"}
ADDRESS_COLUMNS = {"Address", "BuildingNumber", "Fraction", "DirectionPrefix", "StreetName", "StreetType", "DirectionSuffix"}
IDENTIFIERS = {"ExciseTaxNbr", "RecordingNbr", "Volume", "Page", "PlatNbr", "PlatType", "PlatLot", "PlatBlock",
               "PlatLot (parcel file)", "PlatBlock (parcel file)", "BldgNbr"}

OVERRIDES = {
    "SalePrice": ("label", "This is the label. Predict log(price): the skew of {skew} drops to {log_skew} after the log, and price errors are relative (a $50,000 miss matters more on a $300,000 home). Adjust old sales to one date with a house price index before training."),
    "DocumentDate": ("transform", "Not a house feature, but the price adjustment needs it. The typical sale rose from {first_price} in {first_year} to {last_price} in {last_year}. Turn it into sale year and month, and use it to bring every price to one date."),
    "Major": ("remove", "Half of the parcel number. Keep it in the data as the key that joins the files and decides the by-home train/test split, but never give it to the model."),
    "Minor": ("remove", "Half of the parcel number. Keep it in the data as the key that joins the files and decides the by-home train/test split, but never give it to the model."),
    "SellerName": ("remove", "A personal name: it carries no house information, and the county's terms restrict lists of people."),
    "BuyerName": ("remove", "A personal name: it carries no house information, and the county's terms restrict lists of people."),
    "PropertyType": ("remove", "Describes the category ticked on the sale form, not the house. It explains {explained} on its own, and its most common value ('land with previously used building') does not describe these homes in a useful way."),
    "SqFtTotLiving": ("transform", "The strongest size column: it explains {explained} on its own (rank correlation {spearman}). Keep it with a log: the skew of {skew} becomes {log_skew}."),
    "SqFt1stFloor": ("remove", "Part of SqFtTotLiving: the floors plus the finished basement add up to the total living area on {identity} of sales, so this repeats information the total already has."),
    "SqFt2ndFloor": ("remove", "Part of SqFtTotLiving (the parts add up to the total on {identity} of sales). Whether a house has a second floor is already in Stories."),
    "SqFtHalfFloor": ("remove", "Part of SqFtTotLiving, and {zero} of sales have no half floor. It explains about 0% on its own."),
    "SqFtUpperFloor": ("remove", "Part of SqFtTotLiving, and {zero} of sales have no upper floor. It explains about 0% on its own."),
    "SqFtUnfinFull": ("remove", "{zero} of sales are 0, and it explains about 0% on its own."),
    "SqFtUnfinHalf": ("remove", "{zero} of sales are 0, and it explains about 0% on its own."),
    "SqFtTotBasement": ("transform", "{zero} of sales have no basement. Turn it into a yes/no 'has a basement' column; the finished part is covered by SqFtFinBasement."),
    "SqFtFinBasement": ("transform", "Already counted inside SqFtTotLiving. Claude's idea, not from a study: split living area into above-ground area (SqFtTotLiving minus this) and finished basement, because basement space is usually priced lower. Otherwise remove it."),
    "SqFtGarageAttached": ("transform", "Explains {explained} on its own. {zero} have no attached garage. Combine with SqFtGarageBasement into one total garage area (0 = no garage)."),
    "SqFtGarageBasement": ("transform", "Explains only {explained} on its own. Combine with SqFtGarageAttached into one total garage area."),
    "SqFtOpenPorch": ("transform", "Explains {explained} on its own; {zero} have no open porch. Keep with log(1 + area): skew {skew} becomes {log_skew}."),
    "SqFtEnclosedPorch": ("remove", "{zero} of sales have none, and it explains about 0% on its own."),
    "SqFtDeck": ("transform", "Explains {explained} on its own; {zero} have no deck. Keep with log(1 + area): skew {skew} becomes {log_skew}."),
    "Bedrooms": ("keep", "Explains {explained} on its own. Keep it as a count. Expect its effect to shrink or turn negative once living area is in the model; that is real, not a bug."),
    "BathFullCount": ("transform", "Explains {explained} on its own. Combine the three bathroom columns into one total: full + 0.75 x three-quarter + 0.5 x half."),
    "Bath3qtrCount": ("transform", "Combine into total bathrooms (full + 0.75 x three-quarter + 0.5 x half). On its own it explains {explained}."),
    "BathHalfCount": ("transform", "Combine into total bathrooms (full + 0.75 x three-quarter + 0.5 x half). On its own it explains {explained}."),
    "Stories": ("keep", "Explains {explained} on its own, with only {distinct} values. Keep as a number."),
    "NbrLivingUnits": ("remove", "Almost every house has 1 unit, and it explains {explained} on its own."),
    "DaylightBasement": ("remove", "{empty} of sales are empty and it explains only {explained}. The basement columns already carry this."),
    "YrBuilt": ("transform", "Turn it into age at sale (sale year minus year built). It explains {explained} on its own with rank correlation {spearman}, but research expects a U-shape, which a straight-line number understates. Let the model see age as a curve (bands or a tree model)."),
    "YrRenovated": ("transform", "{zero} were never renovated. Turn it into 'renovated before the sale' (yes/no) or years since the renovation, measured at the sale date."),
    "BldgGrade": ("keep", "The strongest column in the whole file: it explains {explained} on its own (rank correlation {spearman}). Keep it as an ordered number from 1 to 13."),
    "BldgGradeVar": ("remove", "{zero} of sales are 0, so it cannot tell homes apart."),
    "Condition": ("remove", "A statewide California study found that condition adds little once the exact neighbourhood is known, and here it explains only {explained} on its own (rank correlation {spearman})."),
    "FinBasementGrade": ("keep", "Explains {explained} on its own; 0 means no finished basement ({zero} of sales). Keep as an ordered number."),
    "PcntComplete": ("remove", "{zero} of sales are 0, so it cannot tell homes apart."),
    "Obsolescence": ("remove", "{zero} of sales are 0, so it cannot tell homes apart."),
    "PcntNetCondition": ("remove", "{zero} of sales are 0, so it cannot tell homes apart."),
    "AddnlCost": ("remove", "{zero} of sales are 0, and it explains about 0% on its own."),
    "HeatSystem": ("transform", "Explains {explained} on its own across {distinct} codes. One-hot encode it, grouping the rare codes into Other."),
    "HeatSource": ("transform", "Explains {explained} on its own across {distinct} codes. One-hot encode it, grouping the rare codes into Other."),
    "BrickStone": ("remove", "{zero} of sales have no brick or stone, and it explains about 0% on its own."),
    "FpSingleStory": ("transform", "Explains {explained} on its own. Combine the four fireplace columns into one total fireplace count."),
    "FpMultiStory": ("transform", "Explains {explained} on its own. Combine the four fireplace columns into one total fireplace count."),
    "FpFreestanding": ("transform", "Combine into the total fireplace count; on its own it explains {explained}."),
    "FpAdditional": ("transform", "Combine into the total fireplace count; on its own it explains {explained}."),
    "ZipCode": ("transform", "Explains {explained} on its own across {distinct} ZIP codes. Encode it (target encoding learned on training homes only, or frequency encoding). {empty} are empty: fill those from Area or DistrictName, or mark them Unknown."),
    "DistrictName": ("transform", "Explains {explained} on its own with only {distinct} cities. One-hot encode it."),
    "Area": ("transform", "The strongest location column: {explained} on its own across {distinct} appraisal areas. These are the county's own valuation neighbourhoods. Encode it (target encoding on training homes only, or one-hot)."),
    "SubArea": ("transform", "Means nothing without Area (the numbers repeat across areas). Combine Area and SubArea into one code; on its own it explains {explained}."),
    "LevyCode": ("transform", "Explains {explained} on its own across {distinct} codes, but it describes tax districts and sits inside DistrictName. Encode it only if it adds something beyond Area and ZipCode; decide after comparing them together."),
    "PlatName": ("transform", "The most detailed neighbourhood column: {explained} on its own across {distinct} subdivisions. Too many values for one-hot: use target encoding learned on training homes only, with an Unknown group for the {empty} that are empty."),
    "Range": ("transform", "Alone it explains only {explained}. Combine Range + Township + Section into one square-mile grid cell code, which works like the grid tiles discussed for location."),
    "Township": ("transform", "Explains {explained} on its own because townships run north to south. Combine with Range and Section into one square-mile grid cell code."),
    "Section": ("transform", "Alone it explains {explained}. Combine with Range and Township into one square-mile grid cell code."),
    "QuarterSection": ("transform", "Alone it explains {explained}. Optionally add it to the grid cell code to make quarter-square-mile cells."),
    "Address": ("remove", "One value per house, so it carries no pattern as text. Its only use would be geocoding it into coordinates later."),
    "CurrentZoning": ("transform", "Explains {explained} on its own, but codes differ between cities ({distinct} codes). Encode it with target encoding on training homes only, or group it into broad zoning classes."),
    "PresentUse": ("transform", "{top_share} of sales are single family. Reduce it to single family / townhouse / other; on its own it explains {explained}."),
    "SqFtLot": ("transform", "Explains {explained} on its own. Keep with a log: skew {skew} becomes {log_skew}. Expect a small, flattening effect within a neighbourhood."),
    "Topography": ("keep", "{nonzero} of sales have a topography note, and it explains {explained} on its own (rank correlation {spearman}). Keep as a yes/no column."),
    "TrafficNoise": ("keep", "Weak on its own ({explained}), but noise usually shows only once location is held fixed. Ordered 0 to 3, cheap to keep."),
    "AirportNoise": ("keep", "Explains {explained} on its own, with a negative rank correlation of {spearman}. Keep as a yes/no or level column."),
    "WfntLocation": ("transform", "{zero} of sales are not waterfront, but waterfront homes sell for far more. Turn it into 'waterfront yes/no' plus the water body (Puget Sound, Lake Washington, other lake ...)."),
    "WfntFootage": ("transform", "Zero for non-waterfront homes. Keep with log(1 + feet) next to the waterfront yes/no column."),
    "WfntBank": ("transform", "Fold into the waterfront group: low-bank waterfront sold far above the typical home. On its own it explains {explained}."),
    "TidelandShoreland": ("transform", "Fold into the waterfront group: owning tidelands goes with very high prices. On its own it explains {explained}."),
    "PropName": ("remove", "{empty} of sales are empty; where filled it is a free-text name, not a measurable feature."),
    "ViewUtilization": ("remove", "{empty} of sales are empty and it explains only {explained}; the ten view ratings carry the information."),
    "PlatLot": ("remove", "Empty for every sale from 2000 on."),
    "AdjacentGolfFairway": ("keep", "Only 2,554 homes are next to a golf course, but they sold for +52.6% against the typical sale. The share explained is small because the feature is rare. This is a raw comparison that may partly reflect size or neighbourhood; the model will tell."),
    "NativeGrowthProtEsmt": ("keep", "Only 1,284 homes have a protected native-plants strip, but they sold for +51.6% against the typical sale (likely larger, newer lots). Rare, so the share explained is small; cheap yes/no column."),
    "OtherDesignation": ("keep", "1,575 homes carry another legal designation and sold for +42.0% against the typical sale. Rare, so the share explained is small; cheap yes/no column."),
    "Easements": ("keep", "11,611 homes have an easement and sold for +19.3% against the typical sale. Cheap yes/no column."),
    "Access": ("keep", "Private road access (23,399 homes) sold for +14.5% and walk-in-only access (209 homes) for -14.5% against the typical sale. Encode as categories, grouping the rare codes."),
    "WaterSystem": ("keep", "Homes on a private water supply (8,387) sold for +11.1% against the typical sale. Encode as water district / private / other."),
    "InadequateParking": ("keep", "Homes with parking marked inadequate (1,906) sold for -8.6% against the typical sale. Encode as adequate / inadequate / not recorded."),
    "PowerLines": ("keep", "4,560 homes have power lines nearby and sold for -7.7% against the typical sale. Cheap yes/no column."),
}
for view in ["Territorial", "LakeWashington", "Olympics", "PugetSound", "SeattleSkyline", "LakeSammamish", "Cascades"]:
    OVERRIDES[view] = ("keep", "Explains {explained} on its own; {zero} of sales have no such view. Keep as an ordered 0 to 4 rating. Also consider one extra column: the best view rating across all ten view columns.")
for view in ["MtRainier", "SmallLakeRiverCreek", "OtherView"]:
    OVERRIDES[view] = ("transform", "Rare ({zero} of sales have none) and explains only {explained}. Fold it into a 'best view rating' column built from all ten view columns instead of keeping it alone.")


PLAIN_NAMES = {"Area": "appraisal area", "SubArea": "appraisal sub-area", "ZipCode": "ZIP code", "DistrictName": "city",
               "PlatName": "subdivision name", "LevyCode": "tax district code", "Range": "range", "Township": "township",
               "Section": "section", "Stories": "number of stories"}


def plain(text):
    """Replaces column code names inside prose with plain-English names."""
    for code in sorted(LABELS, key=len, reverse=True):
        if code.startswith("SqFt") or code in PLAIN_NAMES or re.search(r"[a-z][A-Z]", code):
            name = PLAIN_NAMES.get(code) or LABELS[code][0].lower() + LABELS[code][1:]
            text = re.sub(rf"\b{re.escape(code)}\b", name, text)
    return text


def percent(value, digits=1):
    if value is None:
        return "no number"
    return f"{value:.2f}%" if 0 < abs(value) < 1 else f"{value:.{digits}f}%"


def money(value):
    return f"${value:,.0f}"


def number(value):
    if value is None:
        return "no number"
    return f"{value:,.0f}" if abs(value) >= 100 or float(value).is_integer() else f"{value:,.2f}"


def fill(text, card, summary):
    rel = card.get("relation") or {}
    num = card.get("numeric") or {}
    bars = (card.get("bars") or {}).get("items") or [[None, None, 0, 0, None]]
    years = card.get("years") or [[0, 0, 0]]
    return text.format(
        explained=percent((rel.get("explained") or 0) * 100), spearman=num.get("spearman"),
        skew=num.get("skew"), log_skew=num.get("log_skew"), empty=percent(card["empty_pct"]),
        zero=percent(card["zero_pct"]), nonzero=percent(100 - card["zero_pct"]), distinct=f"{card['distinct']:,}",
        identity=percent(summary["living_area_identity_pct"]), top_share=percent(bars[0][3]),
        first_year=years[0][0], first_price=money(years[0][2]), last_year=years[-2][0] if len(years) > 1 else years[-1][0],
        last_price=money(years[-2][2] if len(years) > 1 else years[-1][2]))


def suggestion(card, kind, summary):
    name = card["name"]
    explained = (card.get("relation") or {}).get("explained") or 0
    if name in OVERRIDES:
        verdict, reason = OVERRIDES[name]
        return verdict, fill(reason, card, summary)
    if card["empty_pct"] >= 99.99:
        return "remove", "Empty for every sale from 2000 on, so it carries nothing."
    if card["distinct"] <= 1:
        return "remove", "Has only one value in our sales, so it cannot tell homes apart. If hazards matter, they would have to come from outside hazard maps."
    if name in FILTER_COLUMNS:
        return "remove", "Already used to pick the clean sales, so only one or two values are left and it explains nothing."
    if name in IDENTIFIERS or kind == "identifier":
        return "remove", "An identifier or record number: it labels a document or lot and carries no pattern a model can learn."
    if name in ADDRESS_COLUMNS:
        return "remove", f"An address part. It explains {percent(explained * 100)} only because it stands in for location, and Area, ZipCode and PlatName carry location properly."
    if kind.startswith("flag") or card["distinct"] <= 3:
        if explained >= KEEP_FLAG_EXPLAINED:
            return "keep", f"Explains {percent(explained * 100)} on its own, above the 0.5% bar used for yes/no columns. Keep as 0/1."
        return "remove", f"Explains {percent(explained * 100)} on its own, under the 0.5% bar used for yes/no columns, and almost every sale has the same value."
    if card.get("nominal") or kind.startswith("category"):
        if explained >= KEEP_FLAG_EXPLAINED:
            return "transform", f"Explains {percent(explained * 100)} on its own, above the 0.5% bar. One-hot encode it, grouping rare codes into Other."
        return "remove", f"Explains {percent(explained * 100)} on its own, under the 0.5% bar, and one value covers most sales."
    if card["zero_pct"] >= 99 or explained < 0.002:
        return "remove", f"{percent(card['zero_pct'])} of sales are 0 and it explains {percent(explained * 100)} on its own."
    return "keep", f"Explains {percent(explained * 100)} on its own."


def answers(card, kind, family, summary):
    name = card["name"]
    num = card.get("numeric")
    rel = card.get("relation") or {}
    explained = (rel.get("explained") or 0) * 100
    bars = card.get("bars")

    a1 = card["meaning"] + (" The code meanings come from the county's lookup file." if card.get("decode") else "")

    if name == "SalePrice":
        a2 = "Money in US dollars, and the label. It is never an input."
    elif kind == "date":
        a2 = "A date. It becomes sale year and sale month, used to bring every price to one date."
    elif kind in ("identifier", "personal name") or name in IDENTIFIERS:
        a2 = "An identifier: a label for one record. Its numbers are not amounts and their order means nothing."
    elif kind.startswith("area") or kind.startswith("length") or kind.startswith("money") or kind == "percent":
        a2 = f"A measurement ({kind}). It stays a number."
    elif kind.startswith("count") or kind == "year" or kind.startswith("year"):
        a2 = f"A {kind}: a whole number where bigger means more, so it stays a number."
    elif kind.startswith("ordered"):
        a2 = f"An {kind}: the order means something (higher is better or stronger), so it can stay a number."
    elif kind.startswith("flag"):
        a2 = f"A yes/no {kind[5:] if kind.startswith('flag ') else 'flag'}: it becomes 0/1."
    elif "location" in kind or kind.startswith("category") or kind.startswith("address") or kind == "text":
        a2 = (f"A category ({kind}) with {card['distinct']:,} values: the codes are labels, not amounts, so it needs encoding "
              f"({'one-hot works' if card['distinct'] <= 40 else 'too many values for one-hot; target or frequency encoding'}).")
    else:
        a2 = f"{kind.capitalize()}."

    if card["empty_pct"] >= 99.99:
        a3 = "Empty for every sale from 2000 on: the county stopped filling it in for newer records."
    elif card["empty_pct"] > 0.5:
        a3 = f"{percent(card['empty_pct'])} of sales are empty."
        if name == "ZipCode":
            a3 += " The ZIP code is missing from the building record, not unknown: Area and DistrictName still place these homes."
        elif name == "PlatName":
            a3 += " Homes outside a recorded subdivision have no plat name; empty is a real group, not an error."
        elif name in ("DaylightBasement", "ViewUtilization"):
            a3 += " The county leaves this blank when the question does not apply (no basement, no view)."
    else:
        a3 = "Nothing is empty."
    if card["zero_pct"] >= 1 and name != "SalePrice":
        a3 += f" {percent(card['zero_pct'])} are 0, which here means the house has none (no deck, no view, no renovation), not that the value is unknown."
    if card.get("median_when_absent") is not None and card["distinct"] > 1 and kind not in ("identifier", "personal name"):
        a3 += f" Sales where it is empty or 0 went for {card['median_when_absent']:+.1f}% against the typical sale that month, and the rest for {card['median_when_present']:+.1f}%."

    if name == "SalePrice":
        a4 = (f"The clean-sale filter already removed prices under $10,000 and non-market sales. The highest training sale is "
              f"{money(num['max'])}; {num['above_fence']:,} sales sit above the upper fence. Those are expensive homes, not errors, "
              f"so keep them and use the log.")
    elif name == "ZipCode":
        a4 = (f"Some codes are malformed: the smallest is {number(num['min'])}, which is not a King County ZIP, and the largest is "
              f"{number(num['max'])}, which has 8 digits instead of 5. Treat anything that is not a five-digit code from 98001 to 98199 "
              f"as missing.")
    elif num and name not in IDENTIFIERS and kind not in ("identifier",) and card["distinct"] > 1 and not card.get("nominal"):
        a4 = f"Smallest {number(num['min'])}, largest {number(num['max'])}."
        if name == "SqFtTotLiving":
            a4 += " A house under 300 square feet or over 10,000 square feet deserves a look before training."
        elif name == "Bedrooms":
            a4 += f" {num['zeros']:,} sales show 0 bedrooms, which is impossible for a house: treat 0 as missing."
        elif name == "BathFullCount":
            a4 += f" {num['zeros']:,} sales show 0 full bathrooms; check them together with the other bathroom columns."
        elif name in ("YrBuilt",):
            a4 += " No house was built after its sale, because the filter removed those."
        elif num["above_fence"]:
            a4 += f" {num['above_fence']:,} sales sit above the upper fence; for a column where most homes have 0, that fence is simply 'more than none'."
        if num["negatives"]:
            a4 += f" {num['negatives']:,} negative values are impossible."
    elif card["distinct"] <= 1:
        a4 = "Only one value, so nothing can be impossible, and nothing can be learned either."
    else:
        a4 = "No impossible values found."
        if bars and bars.get("rare_share"):
            a4 += f" {percent(bars['rare_share'])} of sales fall in values with fewer than 30 sales; group those into Other."

    if card["distinct"] <= 1:
        a5 = "One value only."
    elif name == "DocumentDate":
        counts = sorted(card["years"], key=lambda y: y[1])
        a5 = (f"Sales run from {card['years'][0][0]} to {card['years'][-1][0]}. The quietest full year is {counts[0][0]} "
              f"({counts[0][1]:,} sales) and the busiest {counts[-1][0]} ({counts[-1][1]:,} sales). 2026 is a part year.")
    elif num and num.get("hist") and name not in IDENTIFIERS and kind not in ("identifier",) and not card.get("nominal"):
        skew, log_skew = num["skew"], num.get("log_skew")
        a5 = f"Median {number(num['p50'])}, middle half {number(num['p25'])} to {number(num['p75'])}. Skew {skew}."
        if log_skew is not None and abs(skew) > 1 and abs(log_skew) < abs(skew) / 2 and abs(log_skew) < 1:
            a5 += f" After log(1 + value) the skew becomes {log_skew}, close to symmetric, so a log is justified by the numbers."
        elif abs(skew) <= 1:
            a5 += " That is close to symmetric, so no log is needed."
        else:
            a5 += f" A log does not fix it (skew after log: {log_skew}), because most values are 0: a yes/no column plus the size works better."
    elif bars and bars["items"]:
        top = bars["items"][0]
        a5 = f"{card['distinct']:,} values. The most common, {top[0]}{' (' + top[1] + ')' if top[1] else ''}, covers {percent(top[3])} of sales."
    else:
        a5 = f"{card['distinct']:,} different values, almost one per sale."

    if name == "SalePrice":
        a6 = "This is the label itself. Every other card measures against it."
    elif name == "DocumentDate":
        a6 = "Prices rose over time, which is why every other card compares a sale with the typical sale in the same month."
    else:
        a6 = f"On its own it explains {percent(explained)} of the variation in time-adjusted price."
        if num and num.get("spearman") is not None and not card.get("nominal"):
            a6 += f" Rank correlation {num['spearman']}: {'higher values go with higher prices' if num['spearman'] > 0.05 else 'higher values go with lower prices' if num['spearman'] < -0.05 else 'no steady rise or fall'}."
        if bars and card.get("nominal") or (bars and not num):
            priced = [b for b in bars["items"] if b[4] is not None]
            if len(priced) >= 2:
                high = max(priced, key=lambda b: b[4])
                low = min(priced, key=lambda b: b[4])
                a6 += (f" Among its common values, {high[0]}{' (' + high[1] + ')' if high[1] else ''} sells highest ({high[4]:+.1f}%)"
                       f" and {low[0]}{' (' + low[1] + ')' if low[1] else ''} lowest ({low[4]:+.1f}%).")
        if card.get("expected"):
            a6 += " Research expectation: see the box on the left."

    similar = [s for s in card.get("similar") or [] if abs(s[1]) >= 0.7]
    if name == "SalePrice":
        a7 = "The label. The sales file holds no tax or appraised values, so nothing in these three files gives the price away."
    elif similar:
        a7 = "Overlaps strongly with " + ", ".join(f"{s[0]} ({s[1]})" for s in similar) + ". Keeping both adds little."
    elif card.get("nested_in"):
        a7 = "Sits inside " + ", ".join(n[0] for n in card["nested_in"]) + ": every value of this column belongs to one value of each of those, so this column is the finer, more detailed one."
    else:
        a7 = "No other column repeats it."
    if name in ("SqFt1stFloor", "SqFt2ndFloor", "SqFtHalfFloor", "SqFtUpperFloor", "SqFtFinBasement"):
        a7 += f" Together with the other floors it adds up exactly to SqFtTotLiving on {percent(summary['living_area_identity_pct'])} of sales."
    if name not in ("SalePrice",):
        a7 += " No leakage: it is known before the sale and is not computed from the price."
    return [a1, a2, a3, a4, a5, a6, a7]


def main():
    data = json.load(open(STATS))
    summary = data["summary"]
    by_name = {c["name"]: c for c in data["cards"]}
    families, cards = [], []
    for family, file_name, columns in FAMILIES:
        families.append({"name": family, "columns": [c[0] for c in columns]})
        for name, kind, meaning in columns:
            card = by_name[name]
            card.update({"family": family, "file": file_name, "kind": kind, "meaning": meaning, "label": LABELS[name],
                         "meaning_source": ("County lookup file (matched by content) plus the column name."
                                            if card.get("decode") else
                                            "Column name and the values seen; the county's record-format document could not be fetched to confirm it."),
                         "expected": EXPECTED.get(name)})
            if name == "SalePrice":
                card["relation"] = None
            if name == "DocumentDate":
                card["bars"] = {"items": [[str(y), f"median {money(p)}", n, round(n / summary['train_sales'] * 100, 2), None]
                                          for y, n, p in card["years"]], "rest_rows": 0, "rest_distinct": 0, "rare_share": 0}
            verdict, reason = suggestion(card, kind, summary)
            card["suggestion"] = {"verdict": verdict, "reason": plain(reason)}
            card["answers"] = [plain(text) for text in answers(card, kind, family, summary)]
            cards.append(card)
    missing = set(by_name) - {c["name"] for c in cards}
    if missing:
        raise SystemExit(f"columns without a card: {sorted(missing)}")
    payload = {"summary": {k: v for k, v in summary.items() if k not in ("label",)}, "families": families, "cards": cards}
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = open(TEMPLATE).read().replace("/*DATA*/null", text)
    open(OUT, "w").write(html)
    counts = {}
    for card in cards:
        counts[card["suggestion"]["verdict"]] = counts.get(card["suggestion"]["verdict"], 0) + 1
    print(f"{len(cards)} cards, {len(html) / 1e6:.2f} MB, suggestions {counts}")


main()
