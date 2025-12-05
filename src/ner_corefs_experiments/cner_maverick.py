from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from maverick import Maverick

import spacy
from spacy.tokens.doc import Doc

from pprint import pprint
from copy import deepcopy

from rapidfuzz import fuzz


nlp = spacy.load("en_core_web_sm")

sample4 = "David meets Carrie at a bar where he and his friend Eli are trying to pick up girls. They seem to like each other at first glance but are reluctant to approach each other because neither seems to show much interest. Fate seems to have had other plans though because they soon meet again at a baseball game; and thanks to Carrie's friend\/roommate Zoe, David gets her number and asks her out. Soon they start dating.\nThroughout the story, either Eli, Zoe, David, or Carrie break the fourth wall to express their inner thoughts and feelings.\nCarrie and David's relationship grows from cheap dates to spending nights at the other person's place to the formation of an emotional connection. A sexual encounter at Carrie's workplace soon leads to them being convinced by some kids that they should move in together and the two suddenly confess their love for one another. This devastates Eli as he fears losing his childhood friend.\nSoon after moving into David's apartment, rigorous meetings\/interrogation sessions with each other's parents take place.\nEli calls David one night and reminds him of his deep-rooted fear Of missing out as well as his fear of commitment, which results in Carrie and David growing apart and eventually leads to the two breaking up.\nAfter they both go on dates with people they perceive to be their \"perfect match\", David and Carrie realize how much they miss each other and how truly in love they were, leading to their eventual patch-up.\nSoon after getting back together, David and Carrie often find themselves being compared to older, married couples, with emphasis on the \"married\" part. This causes David to have a nightmare about Carrie marrying another man and he decides to propose to her in the same bar where they first met.\nOn their wedding day during two heart-to-heart conversations between the bride, groom, and their respective friends, it is revealed that Carrie has doubts about her identity after marriage and how things will change which Zoe reassures her with stories about her own life as a married woman, mother, etc. It is also revealed during the talk between David and Eli that Eli's indifference toward David's relationship was purely fueled by the worry that Carrie might not be the one David is meant to end up with and that Eli also has a desire to get married someday...with the right person of course.\nThe story comes to an end with Carrie and David's wedding being overseen by officiants of both the Christian and Jewish faiths."


text_to_resolve = sample4

"""

Utils

"""


def annotate_ner_in_text(text, entities):
    """
    text: original string
    entities: list of dicts like:
      {
        'entity_group': 'PER',
        'score': np.float32(0.913947),
        'word': 'Glenn Tyler',
        'start': 8,
        'end': 19
      }
    Returns: text with '(entity) -> LABEL' inline.
    """
    # Sort by start index descending so we don't mess up indices when inserting
    entities_sorted = sorted(entities, key=lambda e: e["start"], reverse=True)

    annotated = text
    for ent in entities_sorted:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity_group"]
        span = annotated[start:end]
        replacement = f"[{span} -> {label}({ent['score']:.2f})]"
        annotated = annotated[: start + 1] + replacement + annotated[end:]

    return annotated


"""

Models

## CNER Entities     [start_entity, end_entity[
## Maverick Entities [strat_entity, end_entity]
"""

cner_tokenizer = AutoTokenizer.from_pretrained("Babelscape/cner-base")
cner_model = AutoModelForTokenClassification.from_pretrained("Babelscape/cner-base")
cner_pipeline = pipeline(
    "ner", model=cner_model, tokenizer=cner_tokenizer, grouped_entities=True
)

maverick_model = Maverick(
    hf_name_or_path="sapienzanlp/maverick-mes-preco",
    #   hf_name_or_path = "sapienzanlp/maverick-mes-ontonotes",
)
maverick_tokenizer = maverick_model.__get_model_tokenizer__()

"""

The actual work

"""
raw_text_to_resolve = deepcopy(text_to_resolve)
text_to_resolve = "".join(text_to_resolve.splitlines())

doc = nlp(text_to_resolve)

# CNER detection
cner_result = cner_pipeline(text_to_resolve)

# Change entities format from token indices to word indices (gold label is token indexed)
i = 0
cner_entities_word_repr = []
for ent in cner_result:
    s = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
    if s is None:
        raise Exception(
            f"Could not map CNER entity '{ent['word']}'@{(ent['start'], ent['end'])} to doc (len={len(doc.text)})@position: {doc.text[ent['start']: ent['end']]}"
        )
    cner_entities_word_repr.append(
        (s.start, s.end - 1)
    )  # Right boundary is not exclusive in maverick's processing
    print(
        f'{"{"}{i}{"}"}found: {str(s):15s} for ent {ent["word"]:15s} ({doc.text[ent["start"]:ent["end"]]:10s}) at pos: ({ent["start"]},{ent["end"]}) <=> ({s.start},{s.end}) '
    )
    i += 1

# Put all single entities tuples into an array to match Maverick format [[entity_1_occ_1, entity_1_occ_2], [entity_2_occ_1, entity_2_occ_2], ...]
gold_labels = [
    [occ] if type(occ) is not list else occ for occ in cner_entities_word_repr
]


# Coref + NER -- Maverick
resolved_entities = maverick_model.predict(
    raw_text_to_resolve,
    add_gold_clusters=gold_labels,  ## Removing newlines breaks maverick's detection ??
)

# pprint(resolved_entities["clusters_token_text"])

# Entity merging
"""
maverick result:
    clusters_token_text [
        [entit_1_occ_1, entity_1_occ_2]
        [entit_2_occ_1, entity_2_occ_2]
    ]

    clusters_char_offsets [
        [(entit_1_occ_1_char_start_idx, entit_1_occ_1_char_end_idx)), (entity_1_occ_2_char_start_idx, entity_1_occ_2_char_end_idx)]
        [(entit_2_occ_1_char_start_idx, entit_2_occ_1_char_end_idx)), (entity_2_occ_2_char_start_idx, entity_2_occ_2_char_end_idx)]
    ]

    clusters_token_offsets [
        [(entit_1_occ_1_word_start_idx, entit_1_occ_1_word_end_idx)), (entity_1_occ_2_word_start_idx, entity_1_occ_2_word_end_idx)]
        [(entit_2_occ_1_word_start_idx, entit_2_occ_1_word_end_idx)), (entity_2_occ_2_word_start_idx, entity_2_occ_2_word_end_idx)]
    ]

merge result:

{
    "{entity_idx}": {
        "tokens": [entit_1_occ_1, entity_1_occ_2]
        "char_indices": [(entit_1_occ_1_char_start_idx, entit_1_occ_1_char_end_idx)), (entity_1_occ_2_char_start_idx, entity_1_occ_2_char_end_idx)]
        "token_indices": [(entit_1_occ_1_word_start_idx, entit_1_occ_1_word_end_idx)), (entity_1_occ_2_word_start_idx, entity_1_occ_2_word_end_idx)]
        "label": ["PERS"]
    }
}

"""


def average_similarity(list_a, list_b):
    if not list_a or not list_b:
        return 0.0

    scores = []
    for a in list_a:
        for b in list_b:
            s = fuzz.token_set_ratio(a.lower(), b.lower())
            scores.append(s)
    return sum(scores) / len(scores)


# TODO: Handle CNER labels at the end (ordering of maverick result should ease merging)
# TODO: Ignore some entities based on CNER label (for dedupe) ?

entities = {}
duplicates_indices = []
similarity_threshold = 75
for ndx, cluster in enumerate(resolved_entities["clusters_token_text"]):

    if ndx in duplicates_indices:
        continue

    # Handle current entity
    entities[ndx] = {
        "tokens": deepcopy(resolved_entities["clusters_token_text"][ndx]),
        "char_indices": deepcopy(resolved_entities["clusters_char_offsets"][ndx]),
        "token_indices": deepcopy(
            list(resolved_entities["clusters_token_offsets"][ndx])
        ),
    }

    # print(entities[ndx])

    # Check for duplicates
    candidate_dup_ndx = ndx + 1

    # Iter through candidate duplicates
    while candidate_dup_ndx < len(resolved_entities["clusters_token_text"]):

        # Skip already processed
        if candidate_dup_ndx in duplicates_indices:
            candidate_dup_ndx += 1
            continue

        avg_sim = average_similarity(
            cluster, resolved_entities["clusters_token_text"][candidate_dup_ndx]
        )

        # Is it a duplicate ?
        if avg_sim > similarity_threshold:
            print(
                f"Found duplicate -- {str(ndx):3s}: {str(resolved_entities['clusters_token_text'][ndx][:3]):30s} | {str(candidate_dup_ndx):3s}:{str(resolved_entities['clusters_token_text'][candidate_dup_ndx][:3]):30s}"
            )
            duplicates_indices.append(candidate_dup_ndx)
            entities[ndx]["tokens"].extend(
                deepcopy(resolved_entities["clusters_token_text"][candidate_dup_ndx])
            )
            entities[ndx]["char_indices"].extend(
                deepcopy(resolved_entities["clusters_char_offsets"][candidate_dup_ndx])
            )
            entities[ndx]["token_indices"].extend(
                deepcopy(resolved_entities["clusters_token_offsets"][candidate_dup_ndx])
            )

        candidate_dup_ndx += 1

pprint(entities)
