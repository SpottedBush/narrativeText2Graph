# narrativeText2Graph

## Overview

This project converts natural-language narratives into structured graphs that represent narrative schemes. The resulting graphs make it possible to compare stories and produce similarity scores between two narratives (useful for clustering, retrieval, or evaluation of retellings). This project has been made in the context of the [SemEval2026](https://narrative-similarity-task.github.io/)

## Graph model

Nodes
- Event — an action or change of state (attributes: id, label/text, span, tense?).
- Entity — persons, objects, places (attributes: id, label, types, mentions).
- Narrative Segment — higher-level passages or scenes (attributes: id, label, span, role).

Edges
- Temporal — ordering or temporal relations between events (before, after, during).
- Discourse — rhetorical/discourse relations (contrast, cause, elaboration).
- Coreference — links entity mentions to the same real-world referent.
- Membership — links events/entities to narrative segments.

Nodes and edges should carry provenance (source text offsets) and confidence scores when available.

## Intended workflow

1. Input raw narrative text.
2. Preprocess: tokenization, sentence splitting, coreference resolution, dependency parsing.
3. Extraction: detect events, entities, and narrative segments; create nodes.
4. Relation detection: infer temporal, discourse, coreference, and membership edges.
5. Export graph in a standard format.
6. Compare graphs to compute similarity scores.

## Formats and interoperability

Node schema (JSON):
{
    "id": "e1",
    "type": "Event",
    "text": "opened the door",
    "span": [45, 58],
    "confidence": 0.92
}

Edge schema:
{
    "source": "e1",
    "target": "e2",
    "type": "Temporal",
    "relation": "before",
    "confidence": 0.88
}

## Comparison & similarity

Approaches to compute similarity between two narrative graphs:
- Graph edit distance: penalize node/edge insertions, deletions, substitutions.
- Node/edge embedding + geometric similarity: embed events and entities (e.g., contextual sentence embeddings) then align and score matches.
- Subgraph matching / isomorphism for structural similarity (useful for motif comparison).
- Hybrid: align events by semantic similarity, then evaluate temporal and discourse consistency for a composite score (0–1).

Output: similarity score and alignment mapping (paired nodes with similarity values).