from kedro.pipeline import Pipeline, node

from .nodes import run_crowdtruth


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                name="run_crowdtruth_four_labels",
                func=run_crowdtruth,
                inputs=["params:panli_path_prolific", "params:four_labels"],
                outputs=[
                    "units_four_labels",
                    "workers_four_labels",
                    "annotations_four_labels",
                    "jobs_four_labels",
                    "judgments_four_labels",
                ],
            ),
            node(
                name="run_crowdtruth_three_labels",
                func=run_crowdtruth,
                inputs=["params:panli_path_prolific", "params:three_labels"],
                outputs=[
                    "units_three_labels",
                    "workers_three_labels",
                    "annotations_three_labels",
                    "jobs_three_labels",
                    "judgments_three_labels",
                ],
            ),
            node(
                name="run_crowdtruth_two_labels",
                func=run_crowdtruth,
                inputs=["params:panli_path_prolific", "params:two_labels"],
                outputs=[
                    "units_two_labels",
                    "workers_two_labels",
                    "annotations_two_labels",
                    "jobs_two_labels",
                    "judgments_two_labels",
                ],
            ),
        ]
    )
