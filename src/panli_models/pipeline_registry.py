"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from panli_models.pipelines import (
    convert_benchmarks,
    crowdtruth,
    error_analysis,
    evaluation,
    finetuning,
    latex,
    prediction,
    prediction_partial_models,
    preprocessing,
    shap,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    # Create individual pipelines
    crowdtruth_pipeline = crowdtruth.create_pipeline()
    preprocessing_pipeline = preprocessing.create_pipeline()
    conversion_pipeline = convert_benchmarks.create_pipeline()
    finetuning_pipeline = finetuning.create_pipeline()
    prediction_pipeline = prediction.create_pipeline()
    prediction_partial_models_pipeline = prediction_partial_models.create_pipeline()
    evaluation_pipeline = evaluation.create_pipeline()
    error_analysis_pipeline = error_analysis.create_pipeline()
    latex_pipeline = latex.create_pipeline()
    shap_pipeline = shap.create_pipeline()

    pipeline_all = Pipeline(
        [
            crowdtruth_pipeline,
            preprocessing_pipeline,
            conversion_pipeline,
            finetuning_pipeline,
            prediction_pipeline,
            prediction_partial_models_pipeline,
            evaluation_pipeline,
            error_analysis_pipeline,
            shap_pipeline,
            latex_pipeline,
        ]
    )

    return {
        "all": pipeline_all,
        "crowdtruth": crowdtruth_pipeline,
        "preprocessing": preprocessing_pipeline,
        "conversion": conversion_pipeline,
        "finetuning": finetuning_pipeline,
        "prediction": prediction_pipeline,
        "prediction_partial": prediction_partial_models_pipeline,
        "evaluation": evaluation_pipeline,
        "error_analysis": error_analysis_pipeline,
        "shap": shap_pipeline,
        "latex": latex_pipeline,
        "__default__": pipeline_all,
    }
