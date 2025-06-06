import logging
from itertools import permutations, product

import pandas as pd
from nltk.corpus import wordnet as wn
from spacy.tokens import Doc
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_all_synonyms():
    logger.info("Retrieving synonyms from WordNet")
    n_synsets = len(list(wn.all_synsets()))
    synonyms = set()
    for synset in tqdm(wn.all_synsets(), total=n_synsets):
        lemmas = [(lemma.name(), synset.pos()) for lemma in synset.lemmas()]
        for pair in permutations(lemmas, r=2):
            synonyms.add(pair)
    return synonyms


def get_all_hyponyms():
    logger.info("Retrieving hyponyms from WordNet")
    n_synsets = len(list(wn.all_synsets()))
    hyponyms = set()
    for synset in tqdm(wn.all_synsets(), total=n_synsets):
        lemmas1 = [(lemma.name(), synset.pos()) for lemma in synset.lemmas()]
        for hyponym in synset.hyponyms():
            lemmas2 = [(lemma.name(), hyponym.pos()) for lemma in hyponym.lemmas()]
            for pair in product(lemmas1, lemmas2):
                if not pair[0] == pair[1]:
                    hyponyms.add(pair)
    return hyponyms


def get_all_antonyms():
    logger.info("Retrieving antonyms from WordNet")
    n_synsets = len(list(wn.all_synsets()))

    synset_antonyms = set()
    for synset1 in tqdm(wn.all_synsets(), total=n_synsets):
        for lemma in synset1.lemmas():
            for antonym in lemma.antonyms():
                synset2 = antonym.synset()
                synset_antonyms.add((synset1, synset2))

    antonyms = set()
    for synset1, synset2 in synset_antonyms:
        words1 = [(lemma.name(), synset1.pos()) for lemma in synset1.lemmas()]
        words2 = [(lemma.name(), synset2.pos()) for lemma in synset2.lemmas()]
        for pair in product(words1, words2):
            reversed_pair = (pair[1], pair[0])
            antonyms.add(pair)
            antonyms.add(reversed_pair)
    return antonyms


def contains_lexical_relation(doc1: Doc, doc2: Doc, pairs):
    words1 = [(token.lemma_, token.pos_[0].lower()) for token in doc1]
    words2 = [(token.lemma_, token.pos_[0].lower()) for token in doc2]
    return any(pair in pairs for pair in product(words1, words2))


def summarize_lexical_relations(
    premises: pd.Series, hypotheses: pd.Series
) -> pd.DataFrame:
    synonyms = get_all_synonyms()
    antonyms = get_all_antonyms()
    hyponyms = get_all_hyponyms()

    # create lists with bools
    has_hyponyms = [
        contains_lexical_relation(premise, hypothesis, hyponyms)
        for premise, hypothesis in zip(premises, hypotheses)
    ]
    has_hypernyms = [
        contains_lexical_relation(hypothesis, premise, hyponyms)
        for premise, hypothesis in zip(premises, hypotheses)
    ]
    has_antonyms = [
        contains_lexical_relation(premise, hypothesis, antonyms)
        for premise, hypothesis in zip(premises, hypotheses)
    ]
    has_synonyms = [
        contains_lexical_relation(premise, hypothesis, synonyms)
        for premise, hypothesis in zip(premises, hypotheses)
    ]

    # create dataframe
    df_lexical_relations = pd.DataFrame(
        {
            "antonyms": has_antonyms,
            "synonyms": has_synonyms,
            "hypernyms": has_hypernyms,
            "hyponyms": has_hyponyms,
        },
        index=premises.index,
    )

    return df_lexical_relations
