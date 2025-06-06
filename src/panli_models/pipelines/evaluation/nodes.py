import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# confusion_matrix,; plot_confusion_matrix,; roc_curve,; auc,
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def read_and_merge_results(
    all_predictions: Dict[str, Callable], df_panli: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Reads the partitioned dataset with all predictions and creates
    a dictionary of format {name_model: df_results}. Adds some columns
    from PANLI to the dataframe with results for analysis.

    Args:
        all_predictions: Kedro partitioned dataset with all predictions
        df_panli: PANLI dataset

    Returns:
        Dict[str, pd.DataFrame]: dictionary with names of models as keys,
            and dataframe with results as values
    """
    panli_cols = ["source_type", "additional_sources", "relation", "uqs", "uas"]
    df_panli = df_panli[panli_cols]

    dict_results: Dict[str, pd.DataFrame] = {}
    for name_model, dataset_load_funct in all_predictions.items():
        df_results = dataset_load_funct()
        df_results = df_results.merge(
            df_panli, how="left", left_index=True, right_index=True
        )

        dict_results[name_model] = df_results

    return dict_results


def create_report(
    df_results: pd.DataFrame, evaluation_params: Dict[str, Any]
) -> pd.DataFrame:
    """Creates a classification report dataframe.

    Args:
        df_results: predictions of a model
        evaluation_params: dictionary with parameters for evaluation,
            including labels in correct order

    Returns:
        pd.DataFrame: classification report
    """

    # unpack parameters
    labels = evaluation_params["labels"]

    # get true/predicted labels
    y_true = df_results["true"]
    y_pred = df_results["predicted"]

    # print classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    logger.info(f"\n{report}")

    # create dataframe from classification report to store
    dict_report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(dict_report).transpose()

    return df_report


def calculate_accuracy(
    df_results: pd.DataFrame,
    column: Optional[str] = None,
    subset: Optional[str] = None,
    weighted: bool = False,
) -> float:
    """Calculates the weighted or unweighted accuracy score, optionally for a given
    subset (intra-sentence or inter-sentence units).

    Args:
        df_results: predictions of a model
        subset: should be "intra-sentence" or "inter-sentence" to select either subset;
            if unspecified, accuracy is calculated on all units
        weighted: whether or not accuracy score should be weighted by UAS of units

    Returns:
        float:
    """
    # select subset: inter- or intra-sentence relation
    if column is not None and subset is not None:
        mask = df_results[column] == subset
        df_results = df_results[mask]

    # true labels, predicted labels and weights
    y_true = df_results["true"]
    y_pred = df_results["predicted"]
    weights = df_results["uas"]

    # calculate either unweighted or weighted accuracy
    if weighted:
        accuracy = accuracy_score(y_true, y_pred, sample_weight=weights)
    else:
        accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def create_accuracy_dataframe(
    dict_results: Dict[str, pd.DataFrame], weighted: bool = False
) -> pd.DataFrame:
    """Calculates the weighted or unweighted accuracy scores of all models for
    all units, inter-sentence and intra-sentence units and returns the result
    as a dataframe.

    Args:
        dict_results: dictionary with names of models as keys and predictions
            of the model as values
        weighted: whether or not accuracy score should be weighted by UAS of units

    Returns:
        pd.DataFrame: dataframe with accuracy scores of all models for
            all units, inter-sentence and intra-sentence units
    """

    # iterate over each individual predictions file for evaluation
    data = []
    for df_results in dict_results.values():

        # calculate accuracy of all units
        accuracy = calculate_accuracy(df_results, weighted=weighted)

        # calculate accuracy per relation type (intra-sentence versus inter-sentence)
        accuracy_intra = calculate_accuracy(
            df_results, column="relation", subset="intra-sentence", weighted=weighted
        )
        accuracy_inter = calculate_accuracy(
            df_results, column="relation", subset="inter-sentence", weighted=weighted
        )

        # calculate accuracy per source type (author versus additional source)
        accuracy_author = calculate_accuracy(
            df_results, column="source_type", subset="author", weighted=weighted
        )
        accuracy_additional_source = calculate_accuracy(
            df_results,
            column="source_type",
            subset="additional_source",
            weighted=weighted,
        )

        # calculate accuracy presence additional sources (with or without)
        df_results["additional_sources"] = df_results["additional_sources"].astype(str)
        accuracy_multi = calculate_accuracy(
            df_results, column="additional_sources", subset="True", weighted=weighted
        )
        accuracy_single = calculate_accuracy(
            df_results,
            column="additional_sources",
            subset="False",
            weighted=weighted,
        )

        data.append(
            [
                accuracy,
                accuracy_intra,
                accuracy_inter,
                accuracy_author,
                accuracy_additional_source,
                accuracy_single,
                accuracy_multi,
            ]
        )

    # create dataframe from data, columns and index
    columns = [
        "PANLI-all",
        "PANLI-intra",
        "PANLI-inter",
        "PANLI-author",
        "PANLI-additional_source",
        "PANLI-single",
        "PANLI-multi",
    ]
    index = list(dict_results.keys())
    df_accuracy = pd.DataFrame(data, columns=columns, index=index)

    return df_accuracy


def create_accuracies_table_panli(
    all_predictions: Dict[str, Callable],
    df_panli: pd.DataFrame,
) -> pd.DataFrame:
    """Calculates the weighted and unweighted accuracy scores of all models for
    all units, inter-sentence and intra-sentence units and returns the result
    as a dataframe.

    Args:
        all_predictions: Kedro partitioned dataset with all predictions
        df_panli: PANLI dataset

    Returns:
        pd.DataFrame: dataframe with accuracy scores of all models for
            all units, inter-sentence and intra-sentence units
    """
    # create a dictionary of format {name_model: df_results}
    dict_results = read_and_merge_results(all_predictions, df_panli)

    # calculate accuracies
    df_accuracy = create_accuracy_dataframe(dict_results, weighted=False)

    # calculate weighted accuracies
    df_accuracy_weighted = create_accuracy_dataframe(dict_results, weighted=True)

    # concatenate two dataframes into one with MultiIndex columns
    d = {"accuracy": df_accuracy, "weighted accuracy": df_accuracy_weighted}
    df_concat = pd.concat(d.values(), axis=1, keys=d.keys())

    return df_concat


def create_classification_reports(
    all_predictions: Dict[str, Callable],
    df_panli: pd.DataFrame,
    evaluation_params: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Creates classification reports for the results of all models on PANLI.

    Args:
        all_predictions: Kedro partitioned dataset with all predictions
        df_panli: PANLI dataset
        evaluation_params: dictionary with parameters for evaluation,
            including labels in correct order

    Returns:
        Dict[str, pd.DataFrame]: dictionary with names of models as keys and predictions
            of the model as values, to be stored as Kedro partitions.PartitionedDataset
    """
    # create a dictionary of format {name_model: df_results}
    dict_results = read_and_merge_results(all_predictions, df_panli)

    # create partitioned dataset for classification reports
    classification_reports = {}

    for name_model, df_results in dict_results.items():

        logger.info(f"Creating classification report for {name_model}")
        df_report = create_report(df_results, evaluation_params)
        classification_reports[name_model] = df_report

    return classification_reports


def merge_predictions(
    all_predictions: Dict[str, Callable],
    df_test: pd.DataFrame,
    models: Optional[List[str]],
) -> pd.DataFrame:

    if models is not None:
        all_predictions = {
            model_name: df_loader
            for model_name, df_loader in all_predictions.items()
            if model_name in models
        }

    for model_name, df_loader in all_predictions.items():
        df_predictions = df_loader()
        df_predictions = df_predictions[["predicted"]]
        df_predictions = df_predictions.rename(columns={"predicted": model_name})
        df_test = df_test.merge(df_predictions, left_index=True, right_index=True)

    return df_test


def create_results_table(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    results = []
    y_true = df["label"]
    for model in models:

        # get scores
        y_pred = df[model]
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        # create dict and add to list
        result = {
            "model": model,
            "accuracy": accuracy,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        results.append(result)

    # create dataframe
    df_results = pd.DataFrame(results).set_index("model")
    return df_results
