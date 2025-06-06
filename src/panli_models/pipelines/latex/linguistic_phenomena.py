import pandas as pd
from sklearn.metrics import f1_score
from .nodes import LATEX_MAPPING_MODELS


def get_label_uas(df):
    def _create_tuples(row):
        return [
            (label, score) for label, score in zip(row["labels"], row["uas_scores"])
        ]

    df_uas = df.apply(_create_tuples, axis=1).explode().to_frame("label_uas")
    df_uas[["label", "uas"]] = pd.DataFrame(
        df_uas["label_uas"].tolist(), index=df_uas.index
    )
    df_uas = df_uas.drop(columns=["label_uas"]).reset_index()

    df_uas = df_uas.pivot(index="unit_id", columns="label")["uas"].add_prefix("uas_")
    df_uas.columns.name = None

    return df_uas


def get_all_incorrect(df, models):
    list_ids = []
    for model in models:
        incorrect_model = df[df["label"] != df[model]]
        list_ids.append(set(incorrect_model.index))
    shared_incorrect = set.intersection(*list_ids)
    return shared_incorrect


def get_results_per_type(df, category, models):
    selected = df[df[category]]
    mean_uqs = selected["uqs"].mean()
    mean_uas_ent = selected["uas_entailment"].mean()
    mean_uas_neut = selected["uas_neutral"].mean()
    mean_uas_cont = selected["uas_contradiction"].mean()

    # get overall frequency of samples with marker in dev
    freq = df[category].value_counts()[True]
    perc = df[category].value_counts(normalize=True)[True] * 100
    str_freq = f"{freq} ({round(perc)}%)"

    # get frequency of samples predicted wrong by all models
    freq_incorrect = selected["all_incorrect"].value_counts()[True]
    perc_incorrect = selected["all_incorrect"].value_counts(normalize=True)[True] * 100
    str_incorrect_freq = f"{freq_incorrect} ({round(perc_incorrect)}%)"

    dict_results = {
        "subtype": category,
        "frequency": str_freq,
        "mean uqs": round(mean_uqs, 2),
        "uas entailment": round(mean_uas_ent, 2),
        "uas neutral": round(mean_uas_neut, 2),
        "uas contradiction": round(mean_uas_cont, 2),
        "all incorrect": str_incorrect_freq,
    }

    for model in models:
        y_true = selected["label"]
        y_pred = selected[model]

        performance = f1_score(y_true, y_pred, average="weighted")
        dict_results[model] = round(performance * 100, 1)

    return dict_results


def map_categories_to_types(df_results, df_lexicon):
    fact_types = df_lexicon["factuality_type_full"].unique()
    sips = [c for c in fact_types if c.startswith("SIP_")]
    nsips = [c for c in fact_types if c.startswith("NSIP_")]
    mapping = {
        "lexical_relations": ["antonyms", "synonyms", "hyponyms", "hypernyms"],
        # "SIP/NSIP": ["SIP", "NSIP"],
        "lexical types": [
            "negation",
            "modals",
            "quantifiers",
            "wh_terms",
            "comparatives",
            "pronouns",
        ],
        "SIP": sips,
        "NSIP": nsips,
    }

    flattened = {
        value: key for key, list_values in mapping.items() for value in list_values
    }
    df_results["type"] = df_results["subtype"].map(flattened)
    return df_results


def create_latex_table_linguistic_phenomena(
    df_annotations, df_pred, df_lexicon, models
):
    # get uas scores per label for each unit
    df_aus = get_label_uas(df_pred)

    # concatenate dataframes with relevant columns
    columns = ["label", "uqs"] + models
    df_pred = df_pred[columns]
    df = pd.concat([df_pred, df_annotations, df_aus], axis=1)

    # add column with "all_incorrect"
    all_incorrect = get_all_incorrect(df, models)
    df["all_incorrect"] = df.index.isin(all_incorrect)

    # create list with results per category
    results = []
    for category in df_annotations.columns:
        dict_results = get_results_per_type(df, category, models)
        results.append(dict_results)

    # create dataframe
    columns = [
        "subtype",
        "frequency",
        "mean uqs",
        "uas entailment",
        "uas neutral",
        "uas contradiction",
    ] + models
    df_results = pd.DataFrame(results, columns=columns)

    # add column with broader type of lexical phenomenon
    df_results = map_categories_to_types(df_results, df_lexicon)

    # sort values
    index_cols = ["type", "subtype"]
    df_results = df_results.sort_values(["type", "mean uqs"])
    for c in index_cols:
        df_results[c] = df_results[c].str.upper()
    # optional: drop type column
    df_results = df_results.drop(columns="type")
    df_results = df_results.set_index("subtype")  # or type + subtype
    df_results.columns = [LATEX_MAPPING_MODELS.get(c, c) for c in df_results.columns]

    # optional: drop type column

    return df_results.to_latex()
