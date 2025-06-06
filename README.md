# PANLI-Models: Transformer Evaluation on PANLI

> Fine-tune and evaluate Transformer models on the PANLI dataset using a scalable Kedro pipeline. Includes benchmarking, error analysis, and SHAP-based interpretability.

## Overview
This repository contains code and resources for **evaluating Transformer-based models** on the PANLI dataset. It includes full support for training, prediction, error analysis, and explainability via SHAP.

Results and methodology are described in:

> van Son, C. M. (2024). *Representative Resources for Perspective-Aware Natural Language Inference* (PhD thesis, Vrije Universiteit Amsterdam). [https://doi.org/10.5463/thesis.644](https://doi.org/10.5463/thesis.644)



## Kedro Pipeline

This project is structured as a [Kedro](https://kedro.org/) pipeline, enabling reproducible, modular, and scalable data workflows. Kedro manages data processing, experiment tracking, and configuration, making it easier to organize and automate the analysis steps for the PANLI dataset. All data transformations and CrowdTruth metric computations are implemented as Kedro nodes and pipelines.

## Repository Structure

```bash
📁 conf/         # Kedro configuration (parameters, catalog, logging)
📁 data/         # Raw data, intermediate files, benchmarks
📁 notebooks/    # Jupyter notebooks for interactive exploration
📁 src/          # Source code for Kedro pipelines and utility functions

```

## Getting Started


1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/panli-models.git
    cd panli-models
    ```

2. **Install [Poetry](https://python-poetry.org/) (if needed)**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    Or follow the [official installation guide](https://python-poetry.org/docs/#installation).

3. **Install dependencies**
    ```bash
    poetry install
    ```

4. **Activate the environment**
    ```bash
    poetry shell
    ```

5. **Download NLTK WordNet**
    ```bash
    python -m nltk.downloader wordnet
    ```

6. **Install spaCy and the English model**
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Downloading MultiNLI

This project uses [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) as an additional benchmark. Follow these steps:

1. **Download the [zip](https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip)**


2. **Extract it , and move to:**

    ```
    data/benchmarks/original/multinli_1.0
    ```

    Ensure the structure looks like:

    ```
    data/
      benchmarks/
         original/
            multinli_1.0/
              multinli_1.0_train.jsonl
              multinli_1.0_dev_matched.jsonl
              multinli_1.0_dev_mismatched.jsonl
              ...
    ```



## Running the Kedro Pipelines

After activating the environment (`poetry shell`), run:

- **Run the full workflow**
    ```bash
    kedro run
    ```

- **Run a specific pipeline**
    ```bash
    kedro run --pipelines=<pipeline_name>
    ```
- **Run a specific node**
    ```bash
    kedro run --pipelines=<pipeline_name>
    kedro run --nodes=<node_name>
    ```

### Available pipelines

Each stage corresponds to a pipeline defined in `pipeline_registry.py`:

| Pipeline             | Description                                         |
| -------------------- | --------------------------------------------------- |
| `all`                | Runs the full workflow                              |
| `crowdtruth`         | Computes CrowdTruth metrics (from PANLI-CrowdTruth) |
| `preprocessing`      | Prepares PANLI and benchmark data                   |
| `conversion`         | Converts data formats (e.g., JSONL → CSV)           |
| `finetuning`         | Fine-tunes Transformer models (e.g., BERT, DeBERTa) |
| `prediction`         | Generates model predictions                         |
| `prediction_partial` | Runs predictions using partial models               |
| `evaluation`         | Evaluates model performance                         |
| `error_analysis`     | Analyzes model errors and failure patterns          |
| `shap`               | Computes SHAP values for interpretability           |
| `latex`              | Exports results as LaTeX tables/figures             |


To run a single experiment (e.g., DeBERTa predictions):

```bash
kedro run --nodes=predict_deberta_panli
```


### JupyterLab Integration

Launch interactive notebooks with Kedro context preloaded:

```bash
kedro jupyter lab
```

This allows direct access to datasets, parameters, and pipeline outputs in your notebooks.


## Citation

If you use this codebase or the PANLI dataset in your research, please cite:

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


## Related projects

- [panli](https://github.com/cltl/panli) — The dataset used in this project.
- [panli-crowdtruth](https://github.com/cltl/panli-crowdtruth) — An analysis of the PANLI dataset using the CrowdTruth framework.