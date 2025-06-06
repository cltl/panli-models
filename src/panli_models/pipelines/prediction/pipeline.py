from kedro.pipeline import Pipeline, node

from .nodes import predict_with_sequence_classification, predict_majority


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="predict_deberta_panli",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_panli"],
                outputs="predictions_deberta_panli",
            ),
            node(
                name="predict_deberta_panli_4",
                func=predict_with_sequence_classification,
                inputs=["panli_test_4", "params:deberta_panli_4"],
                outputs="predictions_deberta_panli_4",
                tags=["panli_4"],
            ),
            node(
                name="predict_deberta_mnli",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli"],
                outputs="predictions_deberta_mnli",
            ),
            node(
                name="predict_deberta_mnli_panli",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli_panli"],
                outputs="predictions_deberta_mnli_panli",
            ),
            # most-frequent baseline
            node(
                name="panli_majority_baseline_test",
                func=predict_majority,
                inputs=["panli_train", "panli_test"],
                outputs="predictions_majority_baseline_panli_test",
            ),
            node(
                name="panli_majority_baseline_test_4",
                func=predict_majority,
                inputs=["panli_train_4", "panli_test_4"],
                outputs="predictions_majority_baseline_panli_test_4",
            ),
            node(
                name="mnli_m_majority_baseline",
                func=predict_majority,
                inputs=["mnli_train", "mnli_test_matched"],
                outputs="predictions_majority_baseline_mnli_m",
            ),
            node(
                name="mnli_mm_majority_baseline",
                func=predict_majority,
                inputs=["mnli_train", "mnli_test_mismatched"],
                outputs="predictions_majority_baseline_mnli_mm",
            ),
            # validation set
            node(
                name="panli_majority_baseline_val",
                func=predict_majority,
                inputs=["panli_train", "panli_val"],
                outputs="predictions_majority_baseline_panli_val",
            ),
            node(
                name="panli_majority_baseline_val_4",
                func=predict_majority,
                inputs=["panli_train_4", "panli_val_4"],
                outputs="predictions_majority_baseline_panli_val_4",
                tags=["panli_4"],
            ),
            node(
                name="predict_deberta_panli_val",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_panli"],
                outputs="predictions_deberta_panli_val",
            ),
            node(
                name="predict_deberta_panli_val_4",
                func=predict_with_sequence_classification,
                inputs=["panli_val_4", "params:deberta_panli_4"],
                outputs="predictions_deberta_panli_val_4",
                tags=["panli_4"],
            ),
            node(
                name="predict_deberta_mnli_val",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli"],
                outputs="predictions_deberta_mnli_val",
            ),
            node(
                name="predict_deberta_mnli_panli_val",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli_panli"],
                outputs="predictions_deberta_mnli_panli_val",
            ),
        ]
    )
