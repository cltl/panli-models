import pandas as pd
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support

LATEX_MAPPING_MODELS = {
    "deberta_mnli": "MultiNLI",
    "deberta_panli": "PANLI",
    "deberta_mnli_panli": "MultiNLI + PANLI",
    "deberta_mnli_h": "MultiNLI \textsuperscript{\textit{H}}",
    "deberta_panli_h": "PANLI \textsuperscript{\textit{H}}",
    "deberta_mnli_panli_h": "MultiNLI + PANLI \textsuperscript{\textit{H}}",
    "deberta_mnli_p": "MultiNLI \textsuperscript{\textit{P}}",
    "deberta_panli_p": "PANLI \textsuperscript{\textit{P}}",
    "deberta_mnli_panli_p": "MultiNLI + PANLI \textsuperscript{\textit{P}}",
    "majority_baseline": "majority baseline",
    # "deberta_panli_h": "hypothesis-only baseline",
}

CATEGORIES: Dict[str, List] = {
    "relation": ["intra-sentence", "inter-sentence"],
    "additional_sources": [False, True],
    "source_type": ["author", "additional_source"],
}


def create_latex_table_data_overview(
    full: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> str:
    df_all = (
        full.groupby(["relation", "source_type"])["label"]
        .value_counts()
        .to_frame("count")
    )

    df_train = (
        train.groupby(["relation", "source_type"])["label"]
        .value_counts()
        .to_frame("count")
        .sort_index()
    )

    df_val = (
        val.groupby(["relation", "source_type"])["label"]
        .value_counts()
        .to_frame("count")
    )

    df_test = (
        test.groupby(["relation", "source_type"])["label"]
        .value_counts()
        .to_frame("count")
    )

    df_concat = pd.concat([df_train, df_val, df_test, df_all], axis=1)
    df_concat.columns = ["train", "validation", "test", "total"]

    return df_concat.to_latex()


def create_latex_table_class_counts(
    all: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> str:
    labels_3 = ["entailment", "neutral", "contradiction", "total"]
    labels_4 = ["agree", "partially_agree", "disagree", "uncertain", "total"]
    columns = ["train", "development", "test", "total"]

    df = pd.DataFrame(
        {
            "total_count": all["label"].value_counts(),
            "train_count": train["label"].value_counts(),
            "development_count": val["label"].value_counts(),
            "test_count": test["label"].value_counts(),
            "total_perc": all["label"].value_counts(normalize=True) * 100,
            "train_perc": train["label"].value_counts(normalize=True) * 100,
            "development_perc": val["label"].value_counts(normalize=True) * 100,
            "test_perc": test["label"].value_counts(normalize=True) * 100,
        }
    )

    # add total row and convert count columns back to int (sum converts to float)
    df.loc["total"] = df.sum()
    count_columns = [c for c in df.columns if c.endswith("_count")]
    df[count_columns] = df[count_columns].astype(int)

    # combine count + perc columns into single str of format "count (perc%)"
    for c in columns:
        df[c] = (
            df[f"{c}_count"].astype(str)
            + " ("
            + df[f"{c}_perc"].round().astype(int).astype(str)
            + "%)"
        )

    # select columns and order index
    df = df[columns]
    if "agree" in df.index:
        df.index = pd.Categorical(df.index, labels_4)
    else:
        df.index = pd.Categorical(df.index, labels_3)
    df = df.sort_index()

    return df.to_latex()


def create_latex_table_partial_models(
    df_panli_test: pd.DataFrame,
    df_panli_val: pd.DataFrame,
    df_mnli_m: pd.DataFrame,
    df_mnli_mm: pd.DataFrame,
) -> str:

    dfs = {
        "PANLI test": df_panli_test,
        "PANLI development": df_panli_val,
        "MNLI_m": df_mnli_m,
        "MNLI_mm": df_mnli_mm,
    }
    new_dfs = {}
    for name, df in dfs.items():
        df = df * 100
        df = df.round(1)
        df = df[["accuracy", "f1"]]
        new_dfs[name] = df

    # create combined str for MultiNLI m-mm
    acc = (
        new_dfs["MNLI_m"]["accuracy"].astype(str)
        + "/"
        + new_dfs["MNLI_mm"]["accuracy"].astype(str)
    )
    f1 = (
        new_dfs["MNLI_m"]["f1"].astype(str) + "/" + new_dfs["MNLI_mm"]["f1"].astype(str)
    )

    results_mnli = {"accuracy": acc, "f1": f1}
    df_mnli = pd.DataFrame(results_mnli)

    # concatenate dataframe to create multi-index
    d = {
        "PANLI development": new_dfs["PANLI development"],
        "PANLI test": new_dfs["PANLI test"],
        "MultiNLI-m/mm": df_mnli,
    }
    df = pd.concat(d.values(), axis=1, keys=d.keys())

    # rename models
    df.index = [LATEX_MAPPING_MODELS[old] for old in df.index]

    return df.to_latex(escape=False)


def create_latex_results_table(
    df_test: pd.DataFrame, df_val: pd.DataFrame, models: List[str]
) -> str:
    # concatenate two dataframes into one with MultiIndex columns
    d = {"development": df_val, "test": df_test}
    d = {"PANLI test": df_test}
    df = pd.concat(d.values(), axis=1, keys=d.keys())
    df = df[df.index.isin(models)]

    # round accuracies
    df = df * 100
    df = df.round(1)

    # rename models
    df.index = [LATEX_MAPPING_MODELS[model_name] for model_name in df.index]

    return df.to_latex(escape=False)


def get_performance_per_category(df: pd.DataFrame, model: str, only_f1: bool = False):
    results = []
    for category in ["relation", "additional_sources", "source_type"]:
        for key in CATEGORIES[category]:

            group = df[df[category] == key]

            y_true = group["label"]
            y_pred = group[model]
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted"
            )
            if only_f1:
                result = {
                    "category": category,
                    "key": key,
                    "F1": f1,
                }
            else:
                result = {
                    "category": category,
                    "key": key,
                    "precision": p,
                    "recall": r,
                    "F1": f1,
                }
            results.append(result)

    df_results = pd.DataFrame(results).set_index(["category", "key"])
    return df_results


def create_latex_table_results_per_category(
    df_predictions: pd.DataFrame, models: List[str], only_f1: bool = False,
) -> str:
    dfs = {}
    for model in models:
        model_name = LATEX_MAPPING_MODELS[model]
        dfs[model_name] = get_performance_per_category(
            df_predictions, model=model, only_f1=only_f1
        )

    df = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    df = df.droplevel(level=1, axis=1)

    df["delta_abs"] = df.diff(axis=1).iloc[:, 1]
    df["delta_incr"] = df.pct_change(axis=1).iloc[:, 1]

    df = df * 100
    df = df.round(1)

    df["delta_abs"] = "+" + df["delta_abs"].astype(str)
    df["delta_incr"] = "+" + df["delta_incr"].astype(str)

    return df.to_latex(escape=False)


def create_latex_table_results_per_category_all(
    df_predictions: pd.DataFrame, models: List[str], only_f1: bool = False,
) -> str:
    dfs = {}
    for model in models:
        model_name = LATEX_MAPPING_MODELS[model]
        dfs[model_name] = get_performance_per_category(
            df_predictions, model=model, only_f1=only_f1
        )
    print(dfs)
    df = pd.concat(dfs.values(), axis=1, keys=dfs.keys())  # .T

    df = df * 100
    df = df.round(1)

    return df.to_latex(escape=False)
