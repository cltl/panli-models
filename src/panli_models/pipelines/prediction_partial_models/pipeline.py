from kedro.pipeline import Pipeline, node

from panli_models.pipelines.prediction.nodes import predict_with_sequence_classification


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # PANLI hypothesis-only / premise-only
            node(
                name="panli_predict_deberta_panli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_panli_h"],
                outputs="panli_predictions_deberta_panli_h",
            ),
            node(
                name="panli_predict_deberta_panli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_panli_p"],
                outputs="panli_predictions_deberta_panli_p",
            ),
            node(
                name="panli_predict_deberta_panli_4_h",
                func=predict_with_sequence_classification,
                inputs=["panli_test_4", "params:deberta_panli_4_h"],
                outputs="panli_predictions_deberta_panli_4_h",
            ),
            node(
                name="panli_predict_deberta_panli_4_p",
                func=predict_with_sequence_classification,
                inputs=["panli_test_4", "params:deberta_panli_4_p"],
                outputs="panli_predictions_deberta_panli_4_p",
            ),
            node(
                name="panli_predict_deberta_mnli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli_h"],
                outputs="panli_predictions_deberta_mnli_h",
            ),
            node(
                name="panli_predict_deberta_mnli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli_p"],
                outputs="panli_predictions_deberta_mnli_p",
            ),
            node(
                name="panli_predict_deberta_mnli_panli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli_h"],
                outputs="panli_predictions_deberta_mnli_panli_h",
            ),
            node(
                name="panli_predict_deberta_mnli_panli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_test", "params:deberta_mnli_panli_p"],
                outputs="panli_predictions_deberta_mnli_panli_p",
            ),
            # validation set
            node(
                name="panli_val_predict_deberta_panli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_panli_h"],
                outputs="panli_val_predictions_deberta_panli_h",
            ),
            node(
                name="panli_val_predict_deberta_panli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_panli_p"],
                outputs="panli_val_predictions_deberta_panli_p",
            ),
            node(
                name="panli_val_predict_deberta_panli_4_h",
                func=predict_with_sequence_classification,
                inputs=["panli_val_4", "params:deberta_panli_4_h"],
                outputs="panli_val_predictions_deberta_panli_4_h",
            ),
            node(
                name="panli_val_predict_deberta_panli_4_p",
                func=predict_with_sequence_classification,
                inputs=["panli_val_4", "params:deberta_panli_4_p"],
                outputs="panli_val_predictions_deberta_panli_4_p",
            ),
            node(
                name="panli_val_predict_deberta_mnli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli_h"],
                outputs="panli_val_predictions_deberta_mnli_h",
            ),
            node(
                name="panli_val_predict_deberta_mnli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli_p"],
                outputs="panli_val_predictions_deberta_mnli_p",
            ),
            node(
                name="panli_val_predict_deberta_mnli_panli_h",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli_panli_h"],
                outputs="panli_val_predictions_deberta_mnli_panli_h",
            ),
            node(
                name="panli_val_predict_deberta_mnli_panli_p",
                func=predict_with_sequence_classification,
                inputs=["panli_val", "params:deberta_mnli_panli_p"],
                outputs="panli_val_predictions_deberta_mnli_panli_p",
            ),
            # MNLI: matched
            node(
                name="mnli_m_predict_deberta_panli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_panli_h"],
                outputs="mnli_m_predictions_deberta_panli_h",
            ),
            node(
                name="mnli_m_predict_deberta_panli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_panli_p"],
                outputs="mnli_m_predictions_deberta_panli_p",
            ),
            node(
                name="mnli_m_predict_deberta_mnli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli_h"],
                outputs="mnli_m_predictions_deberta_mnli_h",
            ),
            node(
                name="mnli_m_predict_deberta_mnli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli_p"],
                outputs="mnli_m_predictions_deberta_mnli_p",
            ),
            node(
                name="mnli_m_predict_deberta_mnli_panli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli_panli_h"],
                outputs="mnli_m_predictions_deberta_mnli_panli_h",
            ),
            node(
                name="mnli_m_predict_deberta_mnli_panli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli_panli_p"],
                outputs="mnli_m_predictions_deberta_mnli_panli_p",
            ),
            # MNLI: mismatched
            node(
                name="mnli_mm_predict_deberta_panli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_panli_h"],
                outputs="mnli_mm_predictions_deberta_panli_h",
            ),
            node(
                name="mnli_mm_predict_deberta_panli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_panli_p"],
                outputs="mnli_mm_predictions_deberta_panli_p",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli_h"],
                outputs="mnli_mm_predictions_deberta_mnli_h",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli_p"],
                outputs="mnli_mm_predictions_deberta_mnli_p",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli_panli_h",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli_panli_h"],
                outputs="mnli_mm_predictions_deberta_mnli_panli_h",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli_panli_p",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli_panli_p"],
                outputs="mnli_mm_predictions_deberta_mnli_panli_p",
            ),
            # test full models on mismatched/matched MNLI
            node(
                name="mnli_m_predict_deberta_panli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_panli"],
                outputs="mnli_m_predictions_deberta_panli",
            ),
            node(
                name="mnli_mm_predict_deberta_panli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_panli"],
                outputs="mnli_mm_predictions_deberta_panli",
            ),
            node(
                name="mnli_m_predict_deberta_mnli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli"],
                outputs="mnli_m_predictions_deberta_mnli",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli"],
                outputs="mnli_mm_predictions_deberta_mnli",
            ),
            node(
                name="mnli_m_predict_deberta_mnli_panli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_matched", "params:deberta_mnli_panli"],
                outputs="mnli_m_predictions_deberta_mnli_panli",
            ),
            node(
                name="mnli_mm_predict_deberta_mnli_panli",
                func=predict_with_sequence_classification,
                inputs=["mnli_test_mismatched", "params:deberta_mnli_panli"],
                outputs="mnli_mm_predictions_deberta_mnli_panli",
            ),
        ]
    )
