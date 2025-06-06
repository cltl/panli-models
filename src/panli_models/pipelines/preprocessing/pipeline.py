from kedro.pipeline import Pipeline, node

from .nodes import split_panli, preprocess_panli


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # preprocess_panli (creating hypotheses)
            node(
                name="preprocess_panli",
                func=preprocess_panli,
                inputs="units_three_labels",
                outputs="panli_three_labels",
            ),
            # splitting into train/test/val
            node(
                name="split_panli",
                func=split_panli,
                inputs=["panli_three_labels", "params:split_train_val_test"],
                outputs=["panli_train", "panli_val", "panli_test"],
            ),
            node(
                name="split_panli_4",
                func=split_panli,
                inputs=["panli_four_labels", "params:split_train_val_test"],
                outputs=["panli_train_4", "panli_val_4", "panli_test_4"],
            ),
        ]
    )
