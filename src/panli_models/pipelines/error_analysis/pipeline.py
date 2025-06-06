from kedro.pipeline import Pipeline, node

from .nodes import main


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="error_analysis",
                func=main,
                inputs=["panli_val", "factuality_lexicon"],
                outputs="panli_val_annotations",
            ),
        ]
    )
