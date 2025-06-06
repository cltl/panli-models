from kedro.pipeline import Pipeline, node

from .nodes import get_shap_values


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="get_shap_values_h",
                func=get_shap_values,
                inputs=["panli_val", "params:deberta_panli_h"],
                outputs="shap_values_h",
            ),
            node(
                name="get_shap_values_p",
                func=get_shap_values,
                inputs=["panli_val", "params:deberta_panli_p"],
                outputs="shap_values_p",
            ),
        ]
    )
