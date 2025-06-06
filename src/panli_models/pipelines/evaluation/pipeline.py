from kedro.pipeline import Pipeline, node

from .nodes import (
    create_results_table,
    merge_predictions,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="merge_predictions_test",
                func=merge_predictions,
                inputs=[
                    "all_predictions_panli_test",
                    "panli_test",
                    "params:all_models",
                ],
                outputs="predictions_panli_test",
            ),
            node(
                name="merge_predictions_val",
                func=merge_predictions,
                inputs=["all_predictions_panli_val", "panli_val", "params:all_models"],
                outputs="predictions_panli_val",
            ),
            node(
                name="merge_predictions_mnli_m",
                func=merge_predictions,
                inputs=[
                    "all_predictions_mnli_matched",
                    "mnli_test_matched",
                    "params:all_models",
                ],
                outputs="predictions_mnli_test_matched",
            ),
            node(
                name="merge_predictions_mnli_mm",
                func=merge_predictions,
                inputs=[
                    "all_predictions_mnli_mismatched",
                    "mnli_test_mismatched",
                    "params:all_models",
                ],
                outputs="predictions_mnli_test_mismatched",
            ),
            node(
                name="create_results_table_test",
                func=create_results_table,
                inputs=["predictions_panli_test", "params:all_models"],
                outputs="results_test",
            ),
            node(
                name="create_results_table_val",
                func=create_results_table,
                inputs=["predictions_panli_val", "params:all_models"],
                outputs="results_val",
            ),
            node(
                name="create_results_table_mnli_matched",
                func=create_results_table,
                inputs=["predictions_mnli_test_matched", "params:all_models"],
                outputs="results_mnli_matched",
            ),
            node(
                name="create_results_table_mnli_mismatched",
                func=create_results_table,
                inputs=["predictions_mnli_test_mismatched", "params:all_models"],
                outputs="results_mnli_mismatched",
            ),
        ]
    )
