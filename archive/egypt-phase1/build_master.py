"""Merge the Stage 1 developer lists into one master list with stable developer IDs.

Stage 1b (web-verified list) is the backbone. Stage 1a (names seen in listing
datasets) contributes listing counts, and any name 1b could not verify is kept
with verified=False so Stage 2 can retry it.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
STAGE1 = HERE / "stage1"
MASTER_PATH = HERE / "developers_master.csv"

GENERIC_WORDS = r"\b(developments?|developers?|group|holding|properties|real estate|company|egypt|misr|urban|international)\b"


def normalise(name: str) -> str:
    lowered = re.sub(GENERIC_WORDS, " ", str(name).lower())
    return re.sub(r"[^a-z0-9]", "", lowered)


def all_names(row: pd.Series) -> list[str]:
    aliases = str(row.get("aliases_en", "") or "")
    parent = str(row.get("parent_group", "") or "")
    names = [row["name_en"]] + [a for a in aliases.split("|") if a and a != "nan"]
    if parent and parent != "nan":
        names.append(parent)
    return [n for n in names if not re.search(r"[()?]", n)]


def build_alias_index(web_list: pd.DataFrame) -> dict[str, int]:
    index = {}
    for position, row in web_list.iterrows():
        for name in all_names(row):
            index.setdefault(normalise(name), position)
    return index


def attach_listing_counts(web_list: pd.DataFrame, dataset_list: pd.DataFrame) -> pd.DataFrame:
    alias_index = build_alias_index(web_list)
    merged = web_list.assign(listing_count=0, seen_in_listings=False)
    unmatched_rows = []
    for _, row in dataset_list.iterrows():
        match = next((alias_index[k] for k in map(normalise, all_names(row)) if k in alias_index), None)
        if match is None:
            unmatched_rows.append(row)
            continue
        merged.loc[match, "listing_count"] += row["listing_count"]
        merged.loc[match, "seen_in_listings"] = True
    return merged, pd.DataFrame(unmatched_rows)


def as_unverified_rows(unmatched: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "name_en": unmatched["name_en"],
        "listing_count": unmatched["listing_count"],
        "seen_in_listings": True,
        "verified": False,
        "notes": "seen in listing data only; web verification failed in Stage 1b",
    })


def assign_developer_ids(master: pd.DataFrame) -> pd.DataFrame:
    ordered = master.sort_values(["listing_count", "verified", "name_en"],
                                 ascending=[False, False, True]).reset_index(drop=True)
    ordered.insert(0, "developer_id", [f"DEV{i:03d}" for i in range(1, len(ordered) + 1)])
    return ordered


def main() -> None:
    web_list = pd.read_csv(STAGE1 / "stage1b_developers.csv")
    dataset_list = pd.read_csv(STAGE1 / "stage1a_developers.csv")
    merged, unmatched = attach_listing_counts(web_list, dataset_list)
    master = pd.concat([merged, as_unverified_rows(unmatched)], ignore_index=True)
    master = assign_developer_ids(master)
    master.to_csv(MASTER_PATH, index=False)
    print(f"{len(master)} developers | verified {int(master.verified.sum())} | "
          f"seen in listings {int(master.seen_in_listings.sum())} | "
          f"listing-only unverified {len(unmatched)}")
    print("unmatched:", ", ".join(unmatched["name_en"]) if len(unmatched) else "none")


if __name__ == "__main__":
    main()
