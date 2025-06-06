from typing import Any, Dict

import pandas as pd
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from panli_models.config.columns import COL_HYPOTHESIS, COL_PREMISE


def get_shap_values(
    test: pd.DataFrame,
    params: Dict[str, Any],
):
    # unpack parameters
    hypothesis_only = params["hypothesis_only"]
    premise_only = params["premise_only"]
    model_path = params["model_path"]

    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    pred = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )
    explainer = shap.Explainer(pred)

    if hypothesis_only:
        shap_values = explainer(test[COL_HYPOTHESIS])
        return shap_values
    elif premise_only:
        shap_values = explainer(test[COL_PREMISE])
        return shap_values
    else:
        return None
