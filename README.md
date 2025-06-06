# PANLI Models

This repository contains code and resources for evaluating models on the [PANLI dataset](https://github.com/cltl/panli).

## Kedro Pipeline

This project is structured as a [Kedro](https://kedro.org/) pipeline, enabling reproducible, modular, and scalable data workflows. Kedro manages data processing, experiment tracking, and configuration, making it easier to organize and automate the analysis steps for the PANLI dataset. All data transformations and CrowdTruth metric computations are implemented as Kedro nodes and pipelines.

## Repository Structure

- `conf/` - Kedro configuration files for managing project settings and parameters.
- `data/` - Contains the PANLI dataset files and intermediate data generated during pipeline execution.
- `notebooks/` - Jupyter notebooks for data exploration and analysis.
- `src/` - Source code for running the experiments.

## Getting Started

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/panli-models.git
    cd panli-models
    ```

2. Install [Poetry](https://python-poetry.org/) for dependency management:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    Or follow the [official installation guide](https://python-poetry.org/docs/#installation).

3. Install the project dependencies:
    ```bash
    poetry install
    ```

4. Activate the virtual environment:
    ```bash
    poetry shell
    ```

5. **Download NLTK WordNet:**
    ```bash
    python -m nltk.downloader wordnet
    ```

6. **Install spaCy and the English model:**
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Usage

### Running the Kedro Pipelines

You can execute the entire data analysis workflow or run individual pipeline segments using Kedro's command-line interface.

Run these commands after activating the Poetry shell, or by prefixing with `poetry run`.

- **To run the full pipeline:**
    ```bash
    kedro run
    ```

- **To run a specific pipeline or node:**
    ```bash
    kedro run --pipelines=<pipeline_name>
    kedro run --nodes=<node_name>
    ```

Replace `<pipeline_name>` or `<node_name>` with the desired pipeline. Available pipelines, as defined in `pipeline_registry.py`, include:

- `all`: Runs the complete workflow, combining all pipeline stages.
- `crowdtruth`: Computes CrowdTruth metrics to evaluate annotation quality and inter-annotator agreement.
- `preprocessing`: Handles data cleaning and preparation steps.
- `conversion`: Converts data formats as required for downstream tasks.
- `finetuning`: Fine-tunes models on the PANLI dataset.
- `prediction`: Generates predictions using trained models.
- `prediction_partial`: Runs predictions using partial models.
- `evaluation`: Evaluates model performance on the dataset.
- `error_analysis`: Performs error analysis on model outputs.
- `shap`: Generates SHAP explanations for model interpretability.
- `latex`: Produces LaTeX tables and figures for reporting results.


 For more options, see the [Kedro documentation](https://docs.kedro.org/en/stable/index.html).

### Using Kedro with JupyterLab

To interactively explore data and run Kedro pipelines in notebooks, you can use Kedro's JupyterLab integration.

Run the following command after activating the Poetry shell, or by prefixing with `poetry run`:

```bash
kedro jupyter lab
```

This will launch JupyterLab with the Kedro context preloaded, allowing you to access Kedro datasets, pipelines, and configuration directly within your notebooks.

## Related Projects

- [panli](https://github.com/cltl/panli) — The dataset used in this project.
- [panli-crowdtruth](https://github.com/cltl/panli-crowdtruth) — An analysis of the PANLI dataset using the CrowdTruth framework.

## Citation

If you use this repository, please consider citing:

* van Son, C. M. (2024). *Representative Resources for Perspective-Aware Natural Language Inference* (PhD thesis, Vrije Universiteit Amsterdam). [https://doi.org/10.5463/thesis.644](https://doi.org/10.5463/thesis.644)

    <details>
    <summary>BibTeX</summary>
    ```bibtex
    @phdthesis{ba18bc83a2be4b29805c6b91aaa9a152,
        title = "Representative Resources for Perspective-Aware Natural Language Inference",
        author = "{van Son}, {Chantal Michelle}",
        year = "2024",
        month = nov,
        day = "1",
        doi = "10.5463/thesis.644",
        language = "English",
        type = "PhD-Thesis - Research and graduation internal",
        school = "Vrije Universiteit Amsterdam",
    }
    ```
    </details>



## License

This project is licensed under the Apache License 2.0.

