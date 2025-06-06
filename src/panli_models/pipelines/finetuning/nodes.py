import logging
from typing import Any, Dict

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

from panli_models.config.columns import COL_HYPOTHESIS, COL_LABEL, COL_PREMISE

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_nli_data(
    df: pd.DataFrame,
    tokenizer,
    label2id: Dict[str, int],
    hypothesis_only: bool = False,
    premise_only: bool = False,
    max_length: int = 128,
):
    # create lists of premises, hypotheses and encoded labels
    premises = df[COL_PREMISE].tolist()
    hypotheses = df[COL_HYPOTHESIS].tolist()
    encoded_labels = df[COL_LABEL].apply(lambda x: label2id[x]).tolist()

    # tokenize inputs
    if hypothesis_only:
        inputs = tokenizer(
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
    elif premise_only:
        inputs = tokenizer(
            premises,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
    else:
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )

    return inputs, encoded_labels


def compute_metrics(pred) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def finetune_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_params: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    logging_dir: str,
) -> None:
    """Fine-tunes a Sequence Classification model on an NLI dataset.

    Args:
        train: PANLI training set
        validation: PANLI validation set
        model_params: parameters specifying base model, output directory, etc.
        hyperparameters: hyperparameters to use for training
        logging_dir: directory to store logs

    Returns: None
    """
    # unpack parameters
    model_name = model_params["model_name"]
    model_path = model_params["model_path"]
    hypothesis_only = model_params["hypothesis_only"]
    premise_only = model_params["premise_only"]

    # get id2label and label2id (use alphabetical sorting of labels)
    labels = sorted(list(train[COL_LABEL].unique()))
    id2label: Dict[int, str] = {idx: label for idx, label in enumerate(labels)}
    label2id: Dict[str, int] = {label: idx for idx, label in id2label.items()}

    # load tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        return_dict=True,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create inputs and labels
    logger.info("Creating training dataset")
    train_inputs, train_labels = tokenize_nli_data(
        train, tokenizer, label2id, hypothesis_only, premise_only
    )
    logger.info("Creating validation dataset")
    val_inputs, val_labels = tokenize_nli_data(
        validation, tokenizer, label2id, hypothesis_only, premise_only
    )
    logger.info(f"Shape input ids: {train_inputs.input_ids.shape}")

    # create TrainingArguments and Trainer instance
    # https://huggingface.co/transformers/custom_datasets.html#ft-trainer
    training_args = TrainingArguments(
        output_dir=model_path,
        logging_dir=logging_dir,
        num_train_epochs=hyperparameters["num_train_epochs"],
        weight_decay=hyperparameters["weight_decay"],
        adam_epsilon=hyperparameters["adam_epsilon"],
        adam_beta1=hyperparameters["adam_beta1"],
        adam_beta2=hyperparameters["adam_beta2"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        learning_rate=hyperparameters["learning_rate"],
        warmup_steps=hyperparameters["warmup_steps"],
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        eval_accumulation_steps=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # instantiate a Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=NLIDataset(train_inputs, train_labels),
        eval_dataset=NLIDataset(val_inputs, val_labels),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train model
    logger.info("Training model")
    trainer.train()

    # save model
    logger.info(f"Saving to {model_path}")
    trainer.save_model()
