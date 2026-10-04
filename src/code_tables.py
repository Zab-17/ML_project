"""Turns Zillow's numeric type codes into readable names.

Zillow ships a data dictionary workbook whose extra sheets list the meaning of
each code. We read the two sheets we need from it, plus the county codes, so the
names come from the source instead of being typed in by hand.

An .xlsx file is a zip of XML files, so we open it with the standard library and
avoid needing a spreadsheet package.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.
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
# County codes (Federal Information Processing Standard) for the three counties in the file.
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
    """Returns {column: {code: name}} for the columns we can decode."""
    tables = {"fips": dict(COUNTY_NAMES)}
    with zipfile.ZipFile(DICTIONARY_PATH) as workbook:
        strings = _shared_strings(workbook)
        paths = _sheet_paths(workbook)
        for column, sheet_name in SHEET_PER_COLUMN.items():
            rows = _rows(workbook, paths[sheet_name], strings)
            tables[column] = {code: name for code, name, *_ in rows[1:] if code}
    return tables
