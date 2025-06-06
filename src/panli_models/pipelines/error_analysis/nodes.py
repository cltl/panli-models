import logging
from typing import Callable, Set

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Token

from panli_models.config.columns import COL_HYPOTHESIS, COL_PREMISE
from panli_models.config.columns_lexicon import (  # COL_CLASS,; COL_SIP_NSIP,; COL_FACTUALITY_TYPE,
    COL_DEPRELS,
    COL_FACTUALITY_TYPE_FULL,
    COL_LEMMA,
    COL_POS,
)

from .lexical_relations import summarize_lexical_relations

logger = logging.getLogger(__name__)

QUANTIFIERS = [
    "much",
    "enough",
    "more",
    "most",
    "less",
    "least",
    "no",
    "none",
    "some",
    "any",
    "many",
    "few",
    "several",
    "almost",
    "nearly",
]

# PTB POS tags
COMP_SUPERL = ["JJR", "JJS", "RBR", "RBS"]
WH_TERMS = ["WDT", "WP", "WP$", "WRB"]
PRONOUNS = ["PRP", "PRP$"]


def contains_modal(doc):
    if any(token.tag_ == "MD" for token in doc):
        return True
    return False


def contains_negation(doc):
    if any(token.dep_ == "neg" for token in doc):
        return True
    return False


def contains_quantifier(doc):
    if any(token.lemma_ in QUANTIFIERS for token in doc):
        return True
    return False


def contains_comparatives(doc):
    if any(token.tag_ in COMP_SUPERL for token in doc):
        return True
    return False


def contains_wh_terms(doc):
    if any(token.tag_ in WH_TERMS for token in doc):
        return True
    return False


def contains_pronouns(doc):
    if any(token.tag_ in PRONOUNS for token in doc):
        return True
    return False


def is_ifcomp(token: Token, deprels: Set[str]) -> bool:
    """Checks whether the Token has an "if" child in its grammatical dependencies.

    Args:
        token: spaCy Token.
        deprels: Set of dependency relations from predicate lexicon.

    Returns:
        bool: Whether or not Token has an "if" child.

    """
    if "ifcomp" in deprels:
        for child in token.children:
            if child.text == ["if", "whether"]:
                return True
    return False


def is_whycomp(token: Token, deprels: Set[str]) -> bool:
    """Checks whether the Token has an "if" child in its grammatical dependencies.

    Args:
        token: spaCy Token.
        deprels: Set of dependency relations from predicate lexicon.

    Returns:
        bool: Whether or not Token has an "if" child.

    """
    if "whycomp" in deprels:
        for child in token.children:
            if child.text == "why":
                return True
    return False


def is_prep(token: Token, deprels: Set[str]) -> bool:
    """Checks whether the Token has any specific prepositions in its
    dependency relations.

    Args:
        token: spaCy Token.
        deprels: Set of dependency relations from predicate lexicon.

    Returns:
        bool: Whether or not Token has an "if" child.

    """
    if token.dep_ == "prep":
        prep = f"{token.dep_}_{token.lemma_}"
        if prep in deprels:
            return True
    return False


def check_factuality_classes(doc: Doc, lexicon: pd.DataFrame) -> pd.Series:
    results = {}
    for token in doc:
        mask_predicate = lexicon[COL_LEMMA] == token.lemma_
        mask_pos = lexicon[COL_POS] == token.pos_
        matches = lexicon[mask_predicate & mask_pos]

        for _, match in matches.iterrows():
            deprels = match[COL_DEPRELS]
            for child in token.children:
                if (
                    child.dep_ in deprels
                    or is_ifcomp(child, deprels)
                    or is_prep(child, deprels)
                    or is_whycomp(child, deprels)
                ):

                    # results[match[COL_SIP_NSIP]] = True
                    # results[match[COL_FACTUALITY_TYPE]] = True
                    results[match[COL_FACTUALITY_TYPE_FULL]] = True

    return pd.Series(results, dtype=bool)


def get_overall_factuality_classes(
    premises: pd.Series, hypotheses: pd.Series, lexicon: pd.DataFrame
) -> pd.DataFrame:
    s1 = premises.apply(check_factuality_classes, args=(lexicon,))
    s2 = hypotheses.apply(check_factuality_classes, args=(lexicon,))
    df = pd.concat([s1, s2]).groupby(level=0).max().fillna(False).astype(bool)
    return df


def get_presence_categories(
    premises: pd.Series, hypotheses: pd.Series, function: Callable
) -> pd.Series:
    s1 = premises.apply(function)
    s2 = hypotheses.apply(function)
    any_modal = np.maximum(s1, s2)
    return any_modal


def create_dataframe_categories(premises, hypotheses):

    category2function = {
        "negation": contains_negation,
        "modals": contains_modal,
        "quantifiers": contains_quantifier,
        "comparatives": contains_comparatives,
        "wh_terms": contains_wh_terms,
        "pronouns": contains_pronouns,
    }
    data = {}
    for category, function in category2function.items():
        data[category] = get_presence_categories(premises, hypotheses, function)

    df = pd.DataFrame(
        data,
        index=premises.index,
    )
    return df


def main(
    df: pd.DataFrame, lexicon: pd.DataFrame, nlp_model: str = "en_core_web_lg"
) -> pd.DataFrame:
    nlp = spacy.load(nlp_model)
    logger.info("Processing premises")
    premises = df[COL_PREMISE].apply(lambda x: nlp(x))
    logger.info("Processing hypotheses")
    hypotheses = df[COL_HYPOTHESIS].apply(lambda x: nlp(x))

    logger.info("Getting presence factuality classes")
    df_fact_classes = get_overall_factuality_classes(premises, hypotheses, lexicon)

    logger.info("Getting presence lexical categories")
    df_categories = create_dataframe_categories(premises, hypotheses)

    logger.info("Getting presence lexical relations")
    df_lexical_relations = summarize_lexical_relations(premises, hypotheses)

    # concatenate dataframes
    df = pd.concat([df_categories, df_lexical_relations, df_fact_classes], axis=1)

    return df
