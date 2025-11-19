from typing import List, Dict, Tuple, Optional
import uuid
from spacy.tokens import Doc
import networkx as nx

"""
extraction.py

Updated: detection functions now take a spaCy Doc as input (avoid recomputing the doc).
"""

def detect_entities(doc: Doc) -> List[Dict]:
    """
    Detect named entities using a spaCy Doc.

    Returns a list of dicts:
      { 'id', 'text', 'label', 'start', 'end' }
    """
    entities = []
    for ent in doc.ents:
        entities.append({
            "id": str(uuid.uuid4()),
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities


def detect_events(doc: Doc) -> List[Dict]:
    """
    Detect events from a spaCy Doc by extracting verb-centric predicates
    and their core arguments (subjects/objects).

    Returns list of dicts:
      { 'id', 'text', 'lemma', 'start', 'end', 'subject_texts', 'object_texts', 'sentence' }
    """
    events = []
    for sent in doc.sents:
        for token in sent:
            # consider main verbs and other non-copular meaningful verbs
            if token.pos_ == "VERB" and token.lemma_.lower() not in {"be", "have", "do"}:
                left = min(t.i for t in token.subtree)
                right = max(t.i for t in token.subtree)
                span = doc[left : right + 1]
                subj_texts = [child.text for child in token.children if child.dep_.endswith("subj")]
                obj_texts = [child.text for child in token.children if child.dep_.endswith("obj") or child.dep_ == "dobj"]
                events.append({
                    "id": str(uuid.uuid4()),
                    "text": span.text,
                    "lemma": token.lemma_,
                    "start": span.start_char,
                    "end": span.end_char,
                    "subject_texts": subj_texts,
                    "object_texts": obj_texts,
                    "sentence": sent.text
                })
            # also capture nominalizations (nouns that represent events) heuristically
            elif token.pos_ == "NOUN" and any(child.dep_ == "prep" for child in token.children):
                left = min(t.i for t in token.subtree)
                right = max(t.i for t in token.subtree)
                span = doc[left : right + 1]
                events.append({
                    "id": str(uuid.uuid4()),
                    "text": span.text,
                    "lemma": token.lemma_,
                    "start": span.start_char,
                    "end": span.end_char,
                    "subject_texts": [],
                    "object_texts": [],
                    "sentence": sent.text
                })
    # deduplicate by span + lemma
    unique = {}
    for ev in events:
        key = (ev["start"], ev["end"], ev["lemma"])
        if key not in unique:
            unique[key] = ev
    return list(unique.values())


def merge_narrative_segments(events: List[Dict], entities: List[Dict]) -> List[Dict]:
    """
    Merge narrative segments based on overlapping entities and events.
    """
    pass


def create_narrative_segment(entities: List[Dict], events: List[Dict],
                 graph: nx.Graph = None) -> Tuple[List[Dict], nx.Graph]:
    """
    Create node dictionaries for entities, events, and segments.

    Optionally build a networkx graph linking:
      - event -> entity when entity span is inside event span (label: 'involves')
      - segment -> event/entity when contained (label: 'contains')

    Returns:
      nodes (list of dict) and graph (networkx.Graph or None)
    """
    nodes = []
    segment_node = {
    "id": 0,
    "type": "narrative_segment",
    "label": "0",
    }
    nodes.append(segment_node)
    id_map = {"entity": {}, "event": {}, "segment": {}}

    for ent in entities:
        node = {
            "id": ent["id"],
            "type": "entity",
            "label": ent["label"],
            "text": ent["text"],
            "start": ent["start"],
            "end": ent["end"]
        }
        nodes.append(node)
        id_map["entity"][ent["id"]] = node

    for ev in events:
        node = {
            "id": ev["id"],
            "type": "event",
            "lemma": ev.get("lemma"),
            "text": ev["text"],
            "start": ev["start"],
            "end": ev["end"],
            "subjects": ev.get("subject_texts", []),
            "objects": ev.get("object_texts", [])
        }
        nodes.append(node)
        id_map["event"][ev["id"]] = node

    if graph is None:
        graph = nx.DiGraph()

    
    for n in nodes:
        graph.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
        
    # link events -> entities if entity inside event span
    for ev in events:
        graph.add_edge(ev["id"], segment_node["id"], label="membership")
        for ent in entities:
            graph.add_edge(ent["id"], segment_node["id"], label="membership")
            if ent["start"] >= ev["start"] and ent["end"] <= ev["end"]:
                graph.add_edge(ev["id"], ent["id"], label="involves")
        # try to map subject/object texts to entities (exact match first)
        for subj in ev.get("subject_texts", []):
            for ent in entities:
                if subj.lower() == ent["text"].lower():
                    graph.add_edge(ent["id"], ev["id"], label="subject_of")

        for obj in ev.get("object_texts", []):
            for ent in entities:
                if obj.lower() == ent["text"].lower():
                    graph.add_edge(ent["id"], ev["id"], label="object_of")
    return nodes, graph