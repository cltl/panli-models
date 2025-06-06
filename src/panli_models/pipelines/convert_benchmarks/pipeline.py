from kedro.pipeline import Pipeline, node

from .nodes import concat_dataframes, convert_benchmark


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # convert MNLI
            node(
                name="concat_mnli_dev",
                func=concat_dataframes,
                inputs=[
                    "mnli_dev_matched_original",
                    "mnli_dev_mismatched_original",
                ],
                outputs="mnli_val_original",
                tags=["mnli", "convert_mnli"],
            ),
            node(
                name="convert_mnli_train",
                func=convert_benchmark,
                inputs=[
                    "mnli_train_original",
                    "params:mapping_columns_mnli",
                    "params:mapping_labels_mnli",
                ],
                outputs="mnli_train",
                tags=["mnli", "convert_mnli"],
            ),
            node(
                name="convert_mnli_val",
                func=convert_benchmark,
                inputs=[
                    "mnli_val_original",
                    "params:mapping_columns_mnli",
                    "params:mapping_labels_mnli",
                ],
                outputs="mnli_val",
                tags=["mnli", "convert_mnli"],
            ),
            node(
                name="convert_mnli_test_matched",
                func=convert_benchmark,
                inputs=[
                    "mnli_dev_matched_original",
                    "params:mapping_columns_mnli",
                    "params:mapping_labels_mnli",
                ],
                outputs="mnli_test_matched",
                tags=["mnli", "convert_mnli"],
            ),
            node(
                name="convert_mnli_test_mismatched",
                func=convert_benchmark,
                inputs=[
                    "mnli_dev_mismatched_original",
                    "params:mapping_columns_mnli",
                    "params:mapping_labels_mnli",
                ],
                outputs="mnli_test_mismatched",
                tags=["mnli", "convert_mnli"],
            ),
            # concat MNLI / PANLI
            node(
                name="concat_panli_mnli_train",
                func=concat_dataframes,
                inputs=[
                    "panli_train",
                    "mnli_train",
                ],
                outputs="mnli_panli_train",
            ),
            node(
                name="concat_panli_mnli_val",
                func=concat_dataframes,
                inputs=[
                    "panli_val",
                    "mnli_val",
                ],
                outputs="mnli_panli_val",
            ),
        ]
    )
