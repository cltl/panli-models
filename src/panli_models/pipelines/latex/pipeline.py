from kedro.pipeline import Pipeline, node

from .nodes import (
    create_latex_table_data_overview,
    create_latex_table_partial_models,
    create_latex_results_table,
    create_latex_table_class_counts,
    # create_latex_table_results_per_category,
    create_latex_table_results_per_category_all,
)

from .linguistic_phenomena import create_latex_table_linguistic_phenomena


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="create_latex_table_class_counts",
                func=create_latex_table_class_counts,
                inputs=["panli_three_labels", "panli_train", "panli_val", "panli_test"],
                outputs="latex_table_class_counts",
            ),
            node(
                name="create_latex_table_class_counts_4",
                func=create_latex_table_class_counts,
                inputs=[
                    "panli_four_labels",
                    "panli_train_4",
                    "panli_val_4",
                    "panli_test_4",
                ],
                outputs="latex_table_class_counts_4",
            ),
            node(
                name="create_latex_table_data_overview",
                func=create_latex_table_data_overview,
                inputs=["panli_three_labels", "panli_train", "panli_val", "panli_test"],
                outputs="latex_table_data_overview",
            ),
            node(
                name="create_latex_table_data_overview_4",
                func=create_latex_table_data_overview,
                inputs=[
                    "panli_four_labels",
                    "panli_train_4",
                    "panli_val_4",
                    "panli_test_4",
                ],
                outputs="latex_table_data_overview_4",
            ),
            node(
                name="create_latex_table_partial_models",
                func=create_latex_table_partial_models,
                inputs=[
                    "results_test",
                    "results_val",
                    "results_mnli_matched",
                    "results_mnli_mismatched",
                ],
                outputs="latex_table_partial_models",
            ),
            node(
                name="create_latex_results_table",
                func=create_latex_results_table,
                inputs=["results_test", "results_val", "params:selected_models"],
                outputs="latex_table_results",
            ),
            node(
                name="create_latex_table_results_per_category_all",
                func=create_latex_table_results_per_category_all,
                inputs=[
                    "predictions_panli_val",
                    "params:core_models",
                    "params:only_f1",
                ],
                outputs="latex_table_results_per_category",
            ),
            node(
                name="create_latex_table_linguistic_phenomena",
                func=create_latex_table_linguistic_phenomena,
                inputs=[
                    "panli_val_annotations",
                    "predictions_panli_val",
                    "factuality_lexicon",
                    "params:core_models",
                ],
                outputs="latex_table_results_per_linguistic_phenomenon",
            ),
            # node(
            #     name="create_latex_table_results_per_category_f1",
            #     func=create_latex_table_results_per_category,
            #     inputs=[
            #         "predictions_panli_test",
            #         "params:core_models",
            #         "params:only_f1",
            #     ],
            #     outputs="latex_table_results_per_category_f1",
            # ),
        ]
    )
