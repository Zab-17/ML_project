"""Link compound names as written in listings to developer IDs and project IDs.

Input:  stage1/stage1a_compounds.csv (compound -> developer, as seen in listing data)
        developers_master.csv, projects.csv
Output: compound_lookup.csv, one row per listing compound name, with a match status:
  matched   the project name agrees closely with the listing name
  review    a plausible project exists but a person should confirm it
  no_project  the developer is known but no profiled project resembles the name
  no_developer  the developer name could not be linked to a developer ID
  brand_only  the listing names only the developer brand ("Palm Hills"), so the
              compound cannot be identified
"""
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from build_master import all_names, normalise

HERE = Path(__file__).parent
MATCHED_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60
BRAND_ABBREVIATIONS = {"mountain view": ["mv"], "sodic": ["sodic"]}
MIN_CONTAINED_KEY_LENGTH = 4
CONTAINED_SCORE = 0.9
SAME_CITY_BONUS = 0.05  # breaks ties such as iCity New Cairo versus iCity October


def developer_id_index() -> dict[str, str]:
    master = pd.read_csv(HERE / "developers_master.csv")
    index = {}
    for _, row in master.iterrows():
        for name in all_names(row):
            index.setdefault(normalise(name), row["developer_id"])
    return index


def brand_words(developer_name: str) -> set[str]:
    words = set(str(developer_name).lower().split()) - {"developments", "development", "group"}
    for brand, abbreviations in BRAND_ABBREVIATIONS.items():
        if brand in str(developer_name).lower():
            words.update(abbreviations)
    return words


def strip_brand(name: str, developer_name: str) -> str:
    without_brackets = re.sub(r"\(.*?\)", " ", str(name).lower())
    kept = [word for word in re.split(r"[\s\-]+", without_brackets) if word not in brand_words(developer_name)]
    return normalise(" ".join(kept))


def similarity(compound_key: str, project_key: str) -> float:
    if not compound_key or not project_key:
        return 0.0
    shorter, longer = sorted([compound_key, project_key], key=len)
    if len(shorter) >= MIN_CONTAINED_KEY_LENGTH and shorter in longer:
        return max(CONTAINED_SCORE, SequenceMatcher(None, compound_key, project_key).ratio())
    return SequenceMatcher(None, compound_key, project_key).ratio()


def same_city(listing_area: str, project_city: str) -> bool:
    area, city = normalise(listing_area), normalise(project_city)
    return bool(area and city) and (city in area or area in city)


def best_project(compound_key: str, listing_area: str, candidates: pd.DataFrame) -> tuple[str, str, float]:
    best, best_ranking = ("", "", 0.0), 0.0
    for _, project in candidates.iterrows():
        score = similarity(compound_key, project["name_key"])
        ranking = score + (SAME_CITY_BONUS if same_city(listing_area, project["city"]) else 0)
        if ranking > best_ranking:
            best, best_ranking = (project["project_id"], project["project_name_en"], score), ranking
    return best


def match_status(developer_id: str, compound_key: str, score: float) -> str:
    if not developer_id:
        return "no_developer"
    if not compound_key:
        return "brand_only"
    if score >= MATCHED_THRESHOLD:
        return "matched"
    if score >= REVIEW_THRESHOLD:
        return "review"
    return "no_project"


def link_compound(row: pd.Series, developer_ids: dict, projects: pd.DataFrame) -> dict:
    developer_id = developer_ids.get(normalise(row["developer_name_en"]), "")
    candidates = projects[projects["developer_id"] == developer_id]
    compound_key = strip_brand(row["compound_name_en"], row["developer_name_en"])
    project_id, project_name, score = best_project(compound_key, row["city_or_area"], candidates)
    return {
        "listing_compound_name": row["compound_name_en"],
        "listing_city_or_area": row["city_or_area"],
        "listing_count": row["listing_count"],
        "developer_id": developer_id,
        "developer_name_en": row["developer_name_en"],
        "project_id": project_id if score >= REVIEW_THRESHOLD else "",
        "project_name_en": project_name if score >= REVIEW_THRESHOLD else "",
        "match_score": round(score, 2),
        "match_status": match_status(developer_id, compound_key, score),
        "osm_way_id": "",
    }


def main() -> None:
    compounds = pd.read_csv(HERE / "stage1" / "stage1a_compounds.csv")
    projects = pd.read_csv(HERE / "projects.csv")
    developer_names = pd.read_csv(HERE / "developers_master.csv").set_index("developer_id")["name_en"]
    projects = projects.assign(name_key=[
        strip_brand(name, developer_names.get(developer_id, ""))
        for name, developer_id in zip(projects["project_name_en"], projects["developer_id"])])
    developer_ids = developer_id_index()

    lookup = pd.DataFrame([link_compound(row, developer_ids, projects) for _, row in compounds.iterrows()])
    lookup = lookup.sort_values("listing_count", ascending=False)
    lookup.to_csv(HERE / "compound_lookup.csv", index=False)

    by_status = lookup.groupby("match_status")["listing_count"].agg(["count", "sum"])
    by_status["share_of_listings"] = (by_status["sum"] / by_status["sum"].sum()).round(3)
    print(by_status.to_string())


if __name__ == "__main__":
    main()
