from typing import Dict

import crowdtruth
import pandas as pd

from .classes import ConfigFourLabels, ConfigThreeLabels, ConfigTwoLabels

CONFIG_MAPPING = {
    2: ConfigTwoLabels(),
    3: ConfigThreeLabels(),
    4: ConfigFourLabels(),
}


def fix_annotations(annotations: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:

    annotations["output.answer_value"] = 0

    for idx in judgments.index:
        for k, v in judgments["output.answer_value"][idx].items():
            if v > 0:
                annotations.loc[k, "output.answer_value"] += 1

    annotations = annotations.sort_values(by=["aqs"], ascending=False)
    annotations.round(3)[["output.answer_value", "aqs"]]

    return annotations


def run_crowdtruth(input_file, n_labels=3):

    data, config = crowdtruth.load(file=input_file, config=CONFIG_MAPPING[n_labels])
    results: Dict[str, pd.DataFrame] = crowdtruth.run(data, config)

    # unpack results
    judgments = results["judgments"]
    annotations = results["annotations"]
    units = results["units"]
    workers = results["workers"]
    jobs = results["jobs"]

    # treat annotations (fixing counts)
    annotations = fix_annotations(annotations, judgments)

    return units, workers, annotations, jobs, judgments
