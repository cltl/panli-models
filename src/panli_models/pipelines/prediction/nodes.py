import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import BatchEncoding

from panli_models.config.columns import COL_LABEL
from panli_models.pipelines.finetuning.nodes import tokenize_nli_data

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def create_validation_dataloader(
    inputs: BatchEncoding, true_labels: List[int], batch_size: int = 32
) -> DataLoader:
    """Creates a DataLoader object for the validation dataset.

    Args:
        inputs: tokenized sentence pairs
        true_labels: list of true labels
        batch_size: batch size for validation

    Returns:
        DataLoader: validation DataLoader
    """

    # get input_ids, attention_masks and true_labels as tensors
    # to create TensorDataset
    input_ids = inputs["input_ids"]
    attention_masks = inputs["attention_mask"]
    tensor_true_labels = torch.tensor(true_labels)
    dataset = TensorDataset(input_ids, attention_masks, tensor_true_labels)

    # create DataLoader
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def predict_batches(
    dataloader: DataLoader, model: PreTrainedModel
) -> Tuple[List[float], List[int], List[int]]:
    """Makes predictions on the validation dataset.

    Args:
        dataloader: torch DataLoader with validation data
        model: transformer model

    Returns:
        List[float]: predicted probabilities
        List[int]: predicted labels
        List[int]: true labels
    """
    logits = []
    true_labels = []

    # move model to GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for batch in tqdm(dataloader):

        # Add batch to GPU/CPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up prediction
        with torch.no_grad():

            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        # Move logits and labels to CPU
        b_logits = outputs.logits.detach().cpu().numpy()
        b_label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        logits.append(b_logits)
        true_labels.append(b_label_ids)

    # Flatten lists
    true_labels = [item for sublist in true_labels for item in sublist]
    logits = [item for sublist in logits for item in sublist]

    # Get the corresponding probabilities & predictions
    probabilities = softmax(logits, axis=1).tolist()
    predictions = np.argmax(logits, axis=1).flatten().tolist()

    return probabilities, predictions, true_labels


def predict_with_sequence_classification(
    test: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Uses a pretrained Sequence Classification model to predict inference
    labels on the PANLI dataset.

    Args:
        panli: test dataset
        model: pretrained model
        tokenizer: pretrained tokenizer

    Returns:
        pd.DataFrame: dataframe with probabilities and predicted labels
    """
    # unpack parameters
    hypothesis_only = params["hypothesis_only"]
    premise_only = params["premise_only"]
    model_path = params["model_path"]

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # get labels and ids (lowercase for compatability PANLI)
    label2id = {label.lower(): idx for label, idx in model.config.label2id.items()}
    labels = list(label2id.keys())

    # create dataloader
    inputs, encoded_labels = tokenize_nli_data(
        test, tokenizer, label2id, hypothesis_only, premise_only
    )
    dataloader = create_validation_dataloader(inputs, encoded_labels)

    # make predictions
    logger.info("Predicting inference labels")
    probabilities, predicted_labels, true_labels = predict_batches(dataloader, model)

    # create dataframe with class probabilities
    prob_columns = [f"p_{label}" for label in labels]
    df_results = pd.DataFrame(probabilities, columns=prob_columns, index=test.index)

    # add true labels and predicted labels to dataframe
    df_results["true"] = [labels[i] for i in true_labels]
    df_results["predicted"] = [labels[i] for i in predicted_labels]

    return df_results


def predict_majority(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:

    train_labels = train[COL_LABEL]
    test_true = test[COL_LABEL]
    unique_labels = train_labels.unique().tolist()
    most_frequent = train_labels.value_counts().index[0]
    test_predicted = [most_frequent] * len(test)

    # prob_columns = [f"p_{label}" for label in unique_labels]
    df_results = pd.DataFrame(
        {"true": test_true, "predicted": test_predicted}, index=test.index
    )
    for label in unique_labels:
        if label == most_frequent:
            prob = 1.0
        else:
            prob = 0

        df_results[f"p_{label}"] = prob

    return df_results
