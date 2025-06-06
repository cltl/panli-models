import logging
import operator
from typing import Any, Dict, List, Tuple

import pandas as pd
import spacy
from sklearn.model_selection import GroupShuffleSplit
from spacy.language import Language

from panli_models.config.columns import (
    COL_ADDITIONAL_SOURCES,
    COL_HYPOTHESIS,
    COL_HYPOTHESIS_SENT_IDS,
    COL_LABEL,
    COL_LABELS,
    COL_PAIR_ID,
    COL_PREMISE,
    COL_PREMISE_SENT_ID,
    COL_RELATION,
    COL_SCORES,
    COL_SOURCE_TEXT,
    COL_SOURCE_TYPE,
    COL_STATEMENT,
    COL_UAS_DICT,
    COL_UAS_SINGLE,
    COL_UNIT_ID,
    COLUMNS_PANLI,
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


DICT_PRONOUNS = {
    "i": "me",
    "my": "me",
    "he": "him",
    "his": "him",
    "she": "her",
    "they": "them",
    "their": "them",
    "we": "us",
    "our": "us",
}

COLUMNS_TO_RENAME = {
    "unit": COL_UNIT_ID,
    "input.pair_id": COL_PAIR_ID,
    "input.sentence": COL_PREMISE,
    "input.statement": COL_STATEMENT,
    "input.source_text": COL_SOURCE_TEXT,
    "input.sent_id": COL_PREMISE_SENT_ID,
    "input.statement_sent_ids": COL_HYPOTHESIS_SENT_IDS,
}


def split_train_test_validation_by_group(
    df: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    group_col: str,
    id_col: str,
    random_state: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Separates a dataset into train, validation and test sets according to
    specified sizes, while ensuring that all samples from the same group will
    be in the same set.

    Args:
        df: dataframe to split
        train_size: size of training set (ratio)
        validation_size: size of validation set (ratio)
        test_size: size of test set (ratio)
        group_col: name of column to group samples by
        id_col: name of column to get ids from
        random_state: random state

    Returns:
        List[str]: train ids
        List[str]: validation ids
        List[str]: test ids
    """

    # get test set (split full data into train/test)
    groups = df[group_col]
    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_ids, test_ids = next(gss.split(df, groups=groups))

    # get train/validation sets (split train into train/validation)
    df_train = df.iloc[train_ids].copy()
    groups = df_train[group_col]
    gss = GroupShuffleSplit(
        test_size=validation_size / (train_size + validation_size),
        random_state=random_state,
    )
    train_ids, val_ids = next(gss.split(df_train, groups=groups))

    # get unit ids
    final_test_ids = df.iloc[test_ids][id_col].to_list()
    final_train_ids = df_train.iloc[train_ids][id_col].to_list()
    final_val_ids = df_train.iloc[val_ids][id_col].to_list()

    return final_train_ids, final_val_ids, final_test_ids


def split_panli(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the PANLI dataset into train, validation and test.

    Args:
        df: PANLI dataset
        params: parameters specifying train, validation and test sizes

    Returns:
        pd.DataFrame: train set
        pd.DataFrame: validation set
        pd.DataFrame: test set
    """

    # unpack parameters
    train_size = params["train_size"]
    validation_size = params["validation_size"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    # get unit ids of train/test/validation splits
    train_ids, val_ids, test_ids = split_train_test_validation_by_group(
        df,
        train_size,
        validation_size,
        test_size,
        group_col=COL_PAIR_ID,
        id_col=COL_UNIT_ID,
        random_state=random_state,
    )

    # split data into subsets
    df_train = df[df[COL_UNIT_ID].isin(train_ids)].copy()
    df_val = df[df[COL_UNIT_ID].isin(val_ids)].copy()
    df_test = df[df[COL_UNIT_ID].isin(test_ids)].copy()

    df_train = df_train.set_index(COL_UNIT_ID, drop=True)
    df_val = df_val.set_index(COL_UNIT_ID, drop=True)
    df_test = df_test.set_index(COL_UNIT_ID, drop=True)

    # log subset sizes
    logger.info(f"Train size: {len(df_train)}")
    logger.info(f"Validation size: {len(df_val)}")
    logger.info(f"Test size: {len(df_test)}")

    return df_train, df_val, df_test


def determine_relation_pair(row: pd.Series) -> str:
    """Determines whether the premise-hypothesis pair is inter-sentence
    or intra-sentence.

    Args:
        row: premise-hypothesis pair in dataframe

    Returns:
        str: intra-sentence or inter-sentence
    """
    if row["input.sent_id"] in row["input.statement_sent_ids"]:
        return "intra-sentence"
    return "inter-sentence"


def determine_source_type(row: pd.Series) -> str:
    """Determines whether the premise-hypothesis pair is inter-sentence
    or intra-sentence.

    Args:
        row: premise-hypothesis pair in dataframe

    Returns:
        str: intra-sentence or inter-sentence
    """
    if row["input.source_index"] == 0:
        return "author"
    else:
        return "additional_source"


def create_hypothesis(row: pd.Series, nlp: Language) -> str:
    """Creates the hypothesis for the PANLI dataset from the source + statement.

    Args:
        row: single row in PANLI dataset
        nlp: spaCy model

    Returns:
        str: hypothesis
    """
    source = row[COL_SOURCE_TEXT]
    source_type = row[COL_SOURCE_TYPE]
    statement = row[COL_STATEMENT]

    # return full statement for author's perspective
    if source_type == "author" or statement.lower().startswith("according to"):
        return statement

    # parse source & statement (for POS tags)
    parsed_source = nlp(source)
    parsed_statement = nlp(statement)

    # lowercase statement if first word is not proper noun
    if not parsed_statement[0].tag_ in ["NNPS", "NNP"]:
        statement = statement[0].lower() + statement[1:]

    # lowercase source if first word it is not proper noun
    if not parsed_source[0].tag_ in ["NNPS", "NNP"]:
        source = source[0].lower() + source[1:]

    # transform pronouns (e.g. I > me)
    if source.lower() in DICT_PRONOUNS:
        source = DICT_PRONOUNS[source.lower()]

    hypothesis = f"According to {source}, {statement}"
    return hypothesis


def split_labels_scores(
    unit_annotation_score: List[Tuple[str, float]],
) -> Tuple[List[str], List[float]]:
    """Separates labels and scores into two separate lists.

    Args:
        unit_annotation_score: labels with their scores

    Returns:
        List[str]: labels
        List[float]: float
    """
    labels, scores = list(zip(*unit_annotation_score))
    return list(labels), list(scores)


def treat_uas_columns(units: pd.DataFrame) -> pd.DataFrame:
    """Treats the unit annotation score columns by separating
    labels and scores.

    Args:
        units: PANLI units

    Returns:
        pd.DataFrame: PANLI units with treated uas columns
    """
    # convert four-way classification to three-way classification
    uas_dict = units[COL_UAS_DICT]

    # sort unit_annotation_score column into list of tuples
    uas_dict = uas_dict.apply(
        lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    )

    # separate labels and scores
    units[COL_LABELS], units[COL_SCORES] = zip(*uas_dict.apply(split_labels_scores))
    units[COL_LABELS] = units[COL_LABELS].apply(lambda x: list(x))
    units[COL_SCORES] = units[COL_SCORES].apply(lambda x: list(x))

    # get label and unit annotation score for best label
    units[COL_UAS_SINGLE] = uas_dict.apply(lambda x: x[0][1])
    units[COL_LABEL] = uas_dict.apply(lambda x: x[0][0])

    return units


def preprocess_panli(
    units: pd.DataFrame, nlp_model: str = "en_core_web_lg"
) -> pd.DataFrame:
    """Preprocesses the PANLI dataset by:
        - selecting/renaming columns
        - treating the unit annotation score columns (separating scores/labels)
        - creating the hypotheses

    Args:
        units: CrowdTruth units
        nlp_model: name of spaCy model

    Returns:
        pd.DataFrame: pre-processed PANLI
    """
    units = units.reset_index()

    # add additional columns
    units[COL_ADDITIONAL_SOURCES] = units["input.n_sources"].apply(lambda x: x > 0)
    units[COL_RELATION] = units.apply(determine_relation_pair, axis=1)
    units[COL_SOURCE_TYPE] = units.apply(determine_source_type, axis=1)

    # treat columns
    units = treat_uas_columns(units)

    # rename columns
    units = units.rename(columns=COLUMNS_TO_RENAME)

    # simplify source_text column
    units[COL_SOURCE_TEXT] = units[COL_SOURCE_TEXT].replace(
        "John (the author)", "the author"
    )

    # create hypotheses from statement + source
    nlp = spacy.load(nlp_model)
    units[COL_HYPOTHESIS] = units.apply(create_hypothesis, args=(nlp,), axis=1)

    # sort rows and select columns
    units = units.sort_values(COL_UNIT_ID)
    units = units[COLUMNS_PANLI]

    return units
