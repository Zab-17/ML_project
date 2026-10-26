### Transform (1)

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Land use and zoning | `PresentUse` | Present use (the property type) | use present use as the property type, grouping rare types into Other. |

### Keep (13)

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Lot and services | `WaterSystem` | Water supply | kept; homes with this feature sold well above or below the typical home. |
| Lot and services | `SewerSystem` | Sewer type | kept (only the no-effect group was removed); small price difference. |
| Lot and services | `Access` | Road access | kept; homes with this feature sold well above or below the typical home. |
| Lot and services | `StreetSurface` | Street surface | kept (only the no-effect group was removed); small price difference. |
| Lot and services | `InadequateParking` | Parking adequate or not | kept; homes with this feature sold well above or below the typical home. |
| Nuisances | `PowerLines` | Power lines nearby | kept; homes with this feature sold well above or below the typical home. |
| Legal restrictions | `AdjacentGolfFairway` | Next to a golf course | kept; homes with this feature sold well above or below the typical home. |
| Legal restrictions | `Easements` | Easement on the land | kept; homes with this feature sold well above or below the typical home. |
| Legal restrictions | `DeedRestrictions` | Deed restrictions | kept (only the no-effect group was removed); small price difference. |
| Legal restrictions | `NativeGrowthProtEsmt` | Protected native-plants strip | kept; homes with this feature sold well above or below the typical home. |
| Legal restrictions | `OtherDesignation` | Other legal designation | kept; homes with this feature sold well above or below the typical home. |
| Legal restrictions | `DNRLease` | Land leased from the state | kept (only the no-effect group was removed); small price difference. |
| Legal restrictions | `WaterProblems` | Water problems | kept (only the no-effect group was removed); small price difference. |

### Remove (83)

| Family | Column | Plain name | Reason |
|---|---|---|---|
| Sale record: filters and paperwork | `ExciseTaxNbr` | Sale form number | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `Major` | Parcel number (first half) | removed as a model input, but kept in the data as the key that joins the files and splits train and test by home. |
| Sale record: filters and paperwork | `Minor` | Parcel number (second half) | removed as a model input, but kept in the data as the key that joins the files and splits train and test by home. |
| Sale record: filters and paperwork | `RecordingNbr` | Deed recording number | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `Volume` | Old record book volume | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `Page` | Old record book page | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PlatNbr` | Subdivision map number (sale record) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PlatType` | Subdivision map type (sale record) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PlatLot` | Lot number (sale record) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PlatBlock` | Block number (sale record) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SellerName` | Seller's name | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `BuyerName` | Buyer's name | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PropertyType` | Property type on the sale form | replaced by present use as the property type. |
| Sale record: filters and paperwork | `PrincipalUse` | Principal use (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleInstrument` | Deed type | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleReason` | Sale reason (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `PropertyClass` | Property class (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `SaleWarning` | Sale warnings (sale filter) | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `AFForestLand` | Sale form: forest land? | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `AFCurrentUseLand` | Sale form: farm or open-space programme? | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `AFNonProfitUse` | Sale form: non-profit use? | no effect on price (same for every sale, empty, a record number or a name). |
| Sale record: filters and paperwork | `AFHistoricProperty` | Sale form: historic property? | no effect on price (same for every sale, empty, a record number or a name). |
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
| Location | `PlatLot (parcel file)` | Lot number (parcel record) | lot number inside a subdivision map, a label with no price meaning. |
| Location | `PlatBlock (parcel file)` | Block number (parcel record) | lot number inside a subdivision map, a label with no price meaning. |
| Location | `Address` | Full address | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `BuildingNumber` | House number | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `Fraction` | Fraction in the house number | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `DirectionPrefix` | Direction before the street name | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `StreetName` | Street name | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `StreetType` | Street type | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `DirectionSuffix` | Direction after the street name | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Location | `PropName` | Property name | address piece; appraisal area, ZIP code and subdivision name carry location. |
| Land use and zoning | `HBUAsIfVacant` | Best use if the land were empty | the county's own bookkeeping; almost every house has the same value. |
| Land use and zoning | `HBUAsImproved` | Best use as it stands now | the county's own bookkeeping; almost every house has the same value. |
| Land use and zoning | `PropType` | Parcel property type | the county's own bookkeeping; almost every house has the same value. |
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
| Nuisances | `OtherNuisances` | Other nuisances | no effect on price (same for every sale, empty, a record number or a name). |
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
