"""Writes one results page per model in results/, from the result files the training runs saved in zillow/out/.

Nothing is trained here and no number is typed by hand. The four lecture classifiers (k-nearest neighbours by
majority vote, the linear support vector classifier, the mixture of experts and the Bayesian network) write their
own pages when they run; this script writes the pages of every other model.

Run: python3 training_code/write_results_pages.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
from dataclasses import dataclass

import pandas as pd

import paths
import shared_code  # noqa: F401  (points TABLE_DIR at the Zillow table)
from shared_code import TABLE_DIR

OUT = TABLE_DIR.parent
METRICS = ["R2 (log)", "MSE (log)", "RMSE (log)", "MAE (log)", "R2 ($)", "RMSE ($)", "MAE ($)", "median $ off",
           "MAPE (%)", "median % off", "within 5%", "within 10%", "within 20%"]
HEADERS = ["R² (log)", "MSE (log)", "RMSE (log)", "MAE (log)", "R² ($)", "RMSE ($)", "MAE ($)", "Median $ miss",
           "Average % miss", "Median % miss", "Within 5%", "Within 10%", "Within 20%"]
BAND_METRICS = ["accuracy (%)", "macro F1", "within one band (%)"]
VOTING = "voting (average of neural network, random forest, k-nearest neighbours)"
REGRESSION_NOTE = "Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes."


@dataclass
class ModelPage:
    file_stem: str
    name: str            # the name the result files use
    title: str
    what: str
    search_file: str | None   # five-fold results, relative to zillow/out
    test_file: str
    kind: str = "regression"  # or "bands"


PAGES = [
    ModelPage("baselines", "median baseline", "Average and median baselines",
              "Predict one number for every home: the training average or the training median (no lecture; the floor).",
              None, "results/all_test_results.csv"),
    ModelPage("linear_regression", "linear regression", "Linear regression (Lecture 4)",
              "Ordinary least squares on all 606 columns.", "neural_search/linear_regression_folds.csv",
              "results/all_test_results.csv"),
    ModelPage("neural_network", "neural network", "Neural network (Lecture 7)",
              "Feed-forward network trained by backpropagation: two hidden layers (ReLU), dropout, batches of 1,024, "
              "stopping after 5 passes without improvement.", "neural_search/fold_results.csv",
              "results/all_test_results.csv"),
    ModelPage("knn_regression", "k-nearest neighbours", "K-nearest neighbours, regression form",
              "The average log value of the k nearest training homes (Lecture 3 teaches the majority-vote "
              "classifier, which is in knn_classifier.md).", "slide_models/fold_results.csv",
              "results/all_test_results.csv"),
    ModelPage("decision_tree", "decision tree", "Decision tree, CART regression (Lecture 10)",
              "CART regression tree: binary splits on squared error, minimum leaf size searched.",
              "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("random_forest", "random forest", "Random forest, regression (Lecture 10)",
              "50 CART regression trees on bootstrap samples, each split choosing from a random share of the columns; "
              "the prediction is the average.", "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("bagging", "bagging", "Bagging, regression form (Lecture 13)",
              "50 regression trees on bootstrap samples, every column open at every split; the prediction is the average.",
              "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("svm_regression", "support vector machine", "Support vector machine, regression form (Lecture 12)",
              "Linear support vector regression: misses smaller than the margin cost nothing (Lecture 12 teaches the "
              "classifier, which is in linear_svm_classifier.md).", "slide_models/fold_results.csv",
              "results/all_test_results.csv"),
    ModelPage("adaboost", "AdaBoost", "AdaBoost.R2, regression form (Lecture 13)",
              "50 rounds of regression trees, each round resampling the homes by the previous round's errors; "
              "the prediction is the weighted median.", "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("kmeans_clustering", "k-means clustering", "K-means clustering (Lecture 14)",
              "Mini-batch K-means on the 606 columns without the label; a home is priced at its cluster's median.",
              "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("hierarchical_clustering", "hierarchical clustering", "Hierarchical clustering (Lecture 14)",
              "Ward agglomerative clustering on 10,000 sampled homes; every home joins the nearest cluster centre and "
              "is priced at its cluster's median.", "slide_models/fold_results.csv", "results/all_test_results.csv"),
    ModelPage("voting", VOTING, "Voting, regression form (Lecture 13)",
              "The average of the three regression models with the lowest validation MSE: the neural network, the "
              "random forest and k-nearest neighbours. No search and no validation score of its own.",
              None, "results/all_test_results.csv"),
    ModelPage("logistic_regression", "logistic regression", "Logistic regression (Lecture 5)",
              "Softmax over the five price bands with an L2 penalty C, L-BFGS solver, at most 300 passes.",
              "band_models/fold_results.csv", "band_models/test_results.csv", kind="bands"),
    ModelPage("perceptron", "perceptron", "Perceptron (Lecture 6)",
              "The perceptron rule, one perceptron per band against the rest.",
              "band_models/fold_results.csv", "band_models/test_results.csv", kind="bands"),
    ModelPage("naive_bayes", "naive Bayes", "Naive Bayes (Lecture 11)",
              "A normal distribution per band for the 7 number columns and a yes/no probability per band for each "
              "0/1 column, with Laplace smoothing.", "band_models/fold_results.csv", "band_models/test_results.csv",
              kind="bands"),
]


def page_text(page):
    lines = [f"# {page.title}", "", page.what, ""]
    if page.kind == "regression":
        lines += [REGRESSION_NOTE + " Tested once on all 571,860 test homes.", ""]
        lines += search_section(page, METRICS, HEADERS) + test_section_regression(page) + band_section(page)
    else:
        lines += ["Classifier on the five price bands (equal fifths of the training homes). Trained on all 2,287,442 "
                  "training homes; tested once on all 571,860 test homes. Guessing scores about 20%.", ""]
        lines += search_section(page, BAND_METRICS, ["Accuracy (%)", "Macro F1", "Within one band (%)"])
        lines += test_section_bands(page)
    return "\n".join(lines) + "\n"


def search_section(page, metrics, headers):
    if page.search_file is None:
        return ["No five-fold search: nothing to tune.", ""]
    folds = pd.read_csv(OUT / page.search_file)
    folds = folds[folds["model"] == page.name] if "model" in folds.columns else folds
    setting_column = "setting" if "setting" in folds.columns else None
    grouped = folds.groupby(setting_column if setting_column else (lambda _: "no setting"), sort=False)
    mean = grouped[metrics].mean()
    spread_metric = metrics[1] if page.kind == "regression" else metrics[0]
    mean.insert(0, "spread", grouped[spread_metric].std())
    mean.insert(0, "folds", grouped.size())
    ascending = page.kind == "regression"
    mean = mean.sort_values(spread_metric, ascending=ascending)
    spread_name = "MSE spread" if page.kind == "regression" else "Accuracy spread"
    header = "| Setting | Folds | " + spread_name + " | " + " | ".join(headers) + " |"
    rows = [f"| {readable(setting)} | {int(r['folds'])} | {r['spread']:.4f} | " +
            " | ".join(cell(m, r[m]) for m in metrics) + " |" for setting, r in mean.iterrows()]
    return ["## Search: average over the five validation folds", "", header, "|" + "---|" * (len(headers) + 3)] + rows + [""]


def test_section_regression(page):
    test = pd.read_csv(OUT / page.test_file)
    names = ["average baseline", "median baseline"] if page.file_stem == "baselines" else [page.name]
    rows = test[test["model"].isin(names)]
    header = "| Model | " + " | ".join(HEADERS) + " |"
    lines = [f"| {r['model'].replace(VOTING, 'voting')} | " + " | ".join(cell(m, r[m]) for m in METRICS) + " |"
             for _, r in rows.iterrows()]
    return ["## Test (571,860 homes)", "", header, "|" + "---|" * (len(HEADERS) + 1)] + lines + [""] + winner_line(page)


def winner_line(page):
    if page.file_stem == "neural_network":
        choice = json.loads((OUT / "final_model/choice.json").read_text())
        return [f"Winning setting: {choice['setting']}; final training {choice['training seconds']:.0f} seconds "
                f"for {choice['passes']} passes.", ""]
    slide = pd.read_csv(OUT / "slide_models/test_results.csv")
    row = slide[slide["model"] == page.name]
    if row.empty:
        return []
    r = row.iloc[0]
    return [f"Winning setting: {readable(r['setting'])}; final fit and test prediction {r['seconds']:,.0f} seconds.", ""]


def band_section(page):
    bands = pd.read_csv(OUT / "results/band_results.csv")
    names = ["average baseline", "median baseline"] if page.file_stem == "baselines" else [page.name]
    rows = bands[bands["model"].isin(names)]
    lines = [f"| {r['model'].replace(VOTING, 'voting')} | " + " | ".join(cell(m, r[m]) for m in BAND_METRICS) + " |"
             for _, r in rows.iterrows()]
    return ["## The same test predictions cut into the five price bands", "",
            "| Model | Accuracy (%) | Macro F1 | Within one band (%) |", "|---|---|---|---|"] + lines + [""]


def test_section_bands(page):
    test = pd.read_csv(OUT / page.test_file)
    r = test[test["model"] == page.name].iloc[0]
    return ["## Test (571,860 homes)", "",
            "| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |",
            "|---|---|---|---|---|",
            f"| {readable(r['setting'])} | {r['accuracy (%)']:.2f} | {r['macro F1']:.4f} | "
            f"{r['within one band (%)']:.2f} | {r['seconds']:.1f} |", ""]


def readable(setting):
    try:
        values = json.loads(setting)
        return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in values.items()) or "no setting"
    except (TypeError, json.JSONDecodeError):
        return str(setting)


def cell(metric, value):
    if "$" in metric and "R2" not in metric:
        return f"{value:,.0f}"
    if "%" in metric or "within" in metric:
        return f"{value:.2f}"
    return f"{value:.4f}"


def main():
    paths.RESULTS_DIR.mkdir(exist_ok=True)
    for page in PAGES:
        (paths.RESULTS_DIR / f"{page.file_stem}.md").write_text(page_text(page))
    print(f"wrote {len(PAGES)} results pages in {paths.RESULTS_DIR}")


if __name__ == "__main__":
    main()
