"""Merge the six Stage 2 slice files into the final developer tables.

Outputs, next to this script:
  developers.csv  one row per developer
  projects.csv    one row per project
  sources.csv     one row per fetched source
Safe to rerun: slices still being written are picked up in their latest saved state.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
STAGE2 = HERE / "stage2"
SLICE_COUNT = 6
NOT_A_DEVELOPER = {
    "DEV084",  # "Rabaa Investment": a Nasr City place name, not a company
    "DEV163",  # MRB: a facility-management consultancy, per Daily News Egypt
    "DEV038",  # Olayan Group: its own site lists real estate in Saudi Arabia only
}
MERGED_INTO = {
    "DEV156": "DEV155",  # Landmark Developments is LMD, confirmed by press (slice 1)
}
DUPLICATE_PROJECTS = {
    "DEV025-P02",  # NEWGIZA listed under Qatari Diar; kept once under DEV024 NEWGIZA
}
ARABIC = re.compile(r"[؀-ۿ]")


def read_slices(table: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for k in range(1, SLICE_COUNT + 1)
              if (path := STAGE2 / f"{table}_{k}.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def drop_non_developers(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["developer_id"].isin(NOT_A_DEVELOPER)]


def merge_duplicate_developers(developers: pd.DataFrame, projects: pd.DataFrame):
    kept_developers = developers[~developers["developer_id"].isin(MERGED_INTO)]
    moved = projects.assign(developer_id=projects["developer_id"].replace(MERGED_INTO))
    name_key = moved["project_name_en"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    moved = moved[~moved.assign(name_key=name_key).duplicated(["developer_id", "name_key"])]
    return kept_developers, renumber_projects(moved)


def renumber_projects(projects: pd.DataFrame) -> pd.DataFrame:
    position = projects.groupby("developer_id").cumcount() + 1
    new_ids = projects["developer_id"] + "-P" + position.map("{:02d}".format)
    return projects.assign(project_id=new_ids)


def fill_tickers_from_master(developers: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(HERE / "developers_master.csv").set_index("developer_id")["egx_ticker"]
    return developers.assign(egx_ticker=developers["egx_ticker"].fillna(developers["developer_id"].map(master)))


def count_arabic_cells(frame: pd.DataFrame) -> int:
    return int(frame.astype(str).apply(lambda column: column.str.contains(ARABIC)).sum().sum())


def find_unresolved_source_ids(frame: pd.DataFrame, known_ids: set[str]) -> set[str]:
    cited = frame["source_ids"].dropna().astype(str).str.split("|").explode().str.strip()
    return set(cited[cited != ""]) - known_ids


def summarise(developers: pd.DataFrame, projects: pd.DataFrame, sources: pd.DataFrame) -> None:
    priced = projects["price_type"].notna()
    print(f"developers {len(developers)} | projects {len(projects)} | sources {len(sources)}")
    print(f"projects with a price: {int(priced.sum())}")
    print(projects.loc[priced, "price_type"].value_counts().to_string())
    print("projects by category:\n" + projects["project_category"].value_counts(dropna=False).to_string())
    print("delivery status:\n" + projects["delivery_status"].value_counts(dropna=False).to_string())


def main() -> None:
    developers = drop_non_developers(read_slices("developers")).drop_duplicates("developer_id")
    projects = drop_non_developers(read_slices("projects")).drop_duplicates("project_id")
    projects = projects[~projects["project_id"].isin(DUPLICATE_PROJECTS)]
    developers, projects = merge_duplicate_developers(developers, projects)
    developers = fill_tickers_from_master(developers)
    sources = read_slices("sources").drop_duplicates("source_id")

    known_ids = set(sources["source_id"])
    for name, frame in [("developers", developers), ("projects", projects), ("sources", sources)]:
        print(f"{name}: arabic cells {count_arabic_cells(frame)}", end="")
        if name != "sources":
            print(f", unresolved source ids {len(find_unresolved_source_ids(frame, known_ids))}", end="")
        print()

    developers.to_csv(HERE / "developers.csv", index=False)
    projects.to_csv(HERE / "projects.csv", index=False)
    sources.to_csv(HERE / "sources.csv", index=False)
    summarise(developers, projects, sources)


if __name__ == "__main__":
    main()
