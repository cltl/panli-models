from kedro.pipeline import Pipeline, node

from .nodes import finetune_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="finetune_deberta_mnli",
                func=finetune_model,
                inputs=[
                    "mnli_train",
                    "mnli_val",
                    "params:deberta_mnli",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_full_models", "finetune_mnli"],
            ),
            node(
                name="finetune_deberta_panli",
                func=finetune_model,
                inputs=[
                    "panli_train",
                    "panli_val",
                    "params:deberta_panli",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_full_models", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_panli_4",
                func=finetune_model,
                inputs=[
                    "panli_train_4",
                    "panli_val_4",
                    "params:deberta_panli_4",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_full_models", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_mnli_panli",
                func=finetune_model,
                inputs=[
                    "mnli_panli_train",
                    "mnli_panli_val",
                    "params:deberta_mnli_panli",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_full_models", "finetune_mnli_panli"],
            ),
            # hypothesis-only
            node(
                name="finetune_deberta_panli_h",
                func=finetune_model,
                inputs=[
                    "panli_train",
                    "panli_val",
                    "params:deberta_panli_h",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_hypothesis-only", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_panli_4_h",
                func=finetune_model,
                inputs=[
                    "panli_train_4",
                    "panli_val_4",
                    "params:deberta_panli_4_h",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_hypothesis-only", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_mnli_h",
                func=finetune_model,
                inputs=[
                    "mnli_train",
                    "mnli_val",
                    "params:deberta_mnli_h",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_hypothesis-only", "finetune_mnli"],
            ),
            node(
                name="finetune_deberta_mnli_panli_h",
                func=finetune_model,
                inputs=[
                    "mnli_panli_train",
                    "mnli_panli_val",
                    "params:deberta_mnli_panli_h",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_hypothesis-only", "finetune_mnli_panli"],
            ),
            # premise-only
            node(
                name="finetune_deberta_panli_p",
                func=finetune_model,
                inputs=[
                    "panli_train",
                    "panli_val",
                    "params:deberta_panli_p",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_premise-only", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_panli_4_p",
                func=finetune_model,
                inputs=[
                    "panli_train_4",
                    "panli_val_4",
                    "params:deberta_panli_4_p",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_premise-only", "finetune_panli"],
            ),
            node(
                name="finetune_deberta_mnli_p",
                func=finetune_model,
                inputs=[
                    "mnli_train",
                    "mnli_val",
                    "params:deberta_mnli_p",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_premise-only", "finetune_mnli"],
            ),
            node(
                name="finetune_deberta_mnli_panli_p",
                func=finetune_model,
                inputs=[
                    "mnli_panli_train",
                    "mnli_panli_val",
                    "params:deberta_mnli_panli_p",
                    "params:hyperparameters",
                    "params:logging_dir",
                ],
                outputs=None,
                tags=["finetune_premise-only", "finetune_mnli_panli"],
            ),
        ]
    )
