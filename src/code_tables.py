"""Decode tables that turn Zillow's numeric type codes into readable category names.

The professor asked for named columns rather than bare identifiers. Zillow ships a
data dictionary whose extra sheets map seven of the identifier columns to names;
this module reads the two we keep straight out of that workbook, so the names in
the report come from the source rather than from memory.

The workbook is read with the standard library, so no spreadsheet package is needed.
"""
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

DICTIONARY_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "zillow_data_dictionary.xlsx"
SPREADSHEET_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATIONSHIP_NAMESPACE = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

SHEET_PER_COLUMN = {
    "heatingorsystemtypeid": "HeatingOrSystemTypeID",
    "propertylandusetypeid": "PropertyLandUseTypeID",
}
# Federal Information Processing Standard county codes for the three counties in the file.
COUNTY_NAMES = {"6037": "Los Angeles", "6059": "Orange", "6111": "Ventura"}


def _shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    return ["".join(part.text or "" for part in entry.iter(f"{SPREADSHEET_NAMESPACE}t")) for entry in root]


def _sheet_paths(workbook: zipfile.ZipFile) -> dict[str, str]:
    relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    target_per_id = {item.get("Id"): item.get("Target") for item in relationships}
    sheets = ET.fromstring(workbook.read("xl/workbook.xml")).find(f"{SPREADSHEET_NAMESPACE}sheets")
    return {sheet.get("name"): "xl/" + target_per_id[sheet.get(f"{RELATIONSHIP_NAMESPACE}id")].lstrip("/")
            for sheet in sheets}


def _rows(workbook: zipfile.ZipFile, path: str, strings: list[str]) -> list[list[str]]:
    sheet = ET.fromstring(workbook.read(path))
    rows = []
    for row in sheet.iter(f"{SPREADSHEET_NAMESPACE}row"):
        values = []
        for cell in row:
            value = cell.find(f"{SPREADSHEET_NAMESPACE}v")
            text = "" if value is None else value.text
            values.append(strings[int(text)] if cell.get("t") == "s" and text else text)
        rows.append(values)
    return rows


def decode_tables() -> dict[str, dict[str, str]]:
    """Map each decodable column to its {code: name} table, including the county codes."""
    tables = {"fips": dict(COUNTY_NAMES)}
    with zipfile.ZipFile(DICTIONARY_PATH) as workbook:
        strings = _shared_strings(workbook)
        paths = _sheet_paths(workbook)
        for column, sheet_name in SHEET_PER_COLUMN.items():
            rows = _rows(workbook, paths[sheet_name], strings)
            tables[column] = {code: name for code, name, *_ in rows[1:] if code}
    return tables
