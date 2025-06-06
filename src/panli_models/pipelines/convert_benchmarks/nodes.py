import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from panli_models.config.columns import COL_LABEL, COL_UNIT_ID

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def split_train_validation(df: pd.DataFrame, params: Dict[str, Any]):
    """Splits a dataframe into train, validation & test sets.

    Args:
        df: dataset to split
        params: parameters with train, val and test size

    Returns:
        pd.DataFrame: train set
        pd.DataFrame: validation set
        pd.DataFrame: test set
    """
    # unpack parameters from parameters.yml
    validation_size = params["validation_size"]
    random_state = params["random_state"]

    # split into train/validation
    train, val = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    # log subset sizes
    logger.info(f"Train size: {len(train)}")
    logger.info(f"Validation size: {len(val)}")

    return train, val


def concat_dataframes(*list_dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates multiple pandas DataFrames into a single DataFrame.

    Args:
        *list_dataframes: A variable number of pandas DataFrames to concatenate.

    Returns:
        pd.DataFrame: A single DataFrame resulting from the concatenation of
            the input DataFrames.
    """
    df = pd.concat(list_dataframes)
    return df


def convert_benchmark(
    df: pd.DataFrame,
    mapping_columns: Dict[str, str],
    mapping_labels: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Renames and selects columns of a dataframe based on the
    specified column mapping.

    Args:
        df: benchmark set (train, test, val)
        mapping_columns: mapping of original columns to new columns
        mapping_labels: mapping of original labels to new labels

    Returns:
        pd.DataFrame: benchmark set with renamed/selected columns
    """
    # rename columns
    df = df.rename(columns=mapping_columns)

    # add any missing columns (fill with NaN)
    for column in mapping_columns.values():
        if column not in df.columns:
            df[column] = np.nan

    # select columns
    df = df[mapping_columns.values()]

    # set index
    df = df.set_index(COL_UNIT_ID)

    # replace values in label column
    if mapping_labels is not None:
        df[COL_LABEL] = df[COL_LABEL].replace(mapping_labels)

        # drop rows with unknown labels
        valid_labels = mapping_labels.values()
        df = df[df[COL_LABEL].isin(valid_labels)]

    return df
