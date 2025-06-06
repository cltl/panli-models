from crowdtruth.configuration import DefaultConfig

INPUT_COLUMNS = [
    "batch_id",
    "list_id",
    "pair_id",
    "sent_id",
    "statement_sent_ids",
    "n_sources",
    "sources",
    "sentence_predicate",
    "sentence",
    "sentence_statement",
    "statement",
    "sim",
    "source_index",
    "source_text",
    "true_answer",
]

OUTPUT_COLUMNS = ["answer_value"]

PROLIFIC_COLUMNS = [
    "judgment_id",  # judgment id
    "question_id",  # unit id
    "worker_id",  # worker id
    "started_time",  # started time
    "submitted_time",  # submitted time
]

MAPPING_THREE_LABELS = {
    "agree": "entailment",
    "disagree": "contradiction",
    "partially_agree": "neutral",
    "uncertain": "neutral",
}

MAPPING_TWO_LABELS = {
    "agree": "entailed",
    "disagree": "not-entailed",
    "partially_agree": "not-entailed",
    "uncertain": "not-entailed",
}

FOUR_LABELS = ["agree", "disagree", "partially_agree", "uncertain"]
THREE_LABELS = ["entailment", "contradiction", "neutral"]
TWO_LABELS = ["entailed", "not-entailed"]


class ConfigFourLabels(DefaultConfig):
    inputColumns = INPUT_COLUMNS
    outputColumns = OUTPUT_COLUMNS
    customPlatformColumns = PROLIFIC_COLUMNS

    # processing of a closed task
    open_ended_task = False
    annotation_vector = FOUR_LABELS


class ConfigThreeLabels(DefaultConfig):
    inputColumns = INPUT_COLUMNS
    outputColumns = OUTPUT_COLUMNS
    customPlatformColumns = PROLIFIC_COLUMNS

    # processing of a closed task
    open_ended_task = False
    annotation_vector = THREE_LABELS

    def processJudgments(self, judgments):
        # pre-process output to match the values in annotation_vector
        for col in self.outputColumns:
            # transform to lowercase
            judgments[col] = judgments[col].replace(MAPPING_THREE_LABELS)
        return judgments


class ConfigTwoLabels(DefaultConfig):
    inputColumns = INPUT_COLUMNS
    outputColumns = OUTPUT_COLUMNS
    customPlatformColumns = PROLIFIC_COLUMNS

    # processing of a closed task
    open_ended_task = False
    annotation_vector = TWO_LABELS

    def processJudgments(self, judgments):
        # pre-process output to match the values in annotation_vector
        for col in self.outputColumns:
            # transform to lowercase
            judgments[col] = judgments[col].replace(MAPPING_TWO_LABELS)
        return judgments
