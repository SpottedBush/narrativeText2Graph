from typing import List, Dict, Any, Optional, Iterable
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from gliner2 import GLiNER2

"""
preprocess.py

Provides simple preprocessing utilities from the SpaCy library:
- tokenization
- sentence splitting 
- dependency parsing
- TODO Arnaud: coreference resolution
"""

def tokenize(doc: Doc) -> List[Dict[str, Any]]:
    """
    Tokenize text. Returns a list of token dicts:
    {text, idx (index in doc), start_char, end_char, lemma, pos, tag}
    """
    out = []
    for i, t in enumerate(doc):
        out.append({
            "text": t.text,
            "idx": i,
            "start_char": t.idx,
            "end_char": t.idx + len(t.text),
            "lemma": t.lemma_,
            "pos": t.pos_,
            "tag": t.tag_,
            "is_alpha": t.is_alpha,
            "is_stop": t.is_stop,
        })
    return out


def split_sentences(doc: Doc) -> List[str]:
    """
    Split text into sentences. Returns a list of sentence strings.
    """
    return [sent.text.strip() for sent in doc.sents]


def parse_dependencies(doc: Doc) -> List[List[Dict[str, Any]]]:
    """
    Parse dependencies. Returns a list (per sentence) of token dicts:
    {text, idx, head_idx, head_text, dep, pos, children (list of child idxs)}
    """
    results: List[List[Dict[str, Any]]] = []
    for sent in doc.sents:
        sent_tokens: List[Dict[str, Any]] = []
        token_to_sent_idx = {token.i: idx for idx, token in enumerate(sent)}
        for idx, token in enumerate(sent):
            head_global = token.head.i
            head_in_sent = token_to_sent_idx.get(head_global, None)
            children = [token_to_sent_idx.get(child.i, None) for child in token.children]
            sent_tokens.append({
                "text": token.text,
                "idx": idx,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head_idx": head_in_sent,
                "head_text": token.head.text,
                "children": children,
                "start_char": token.idx,
                "end_char": token.idx + len(token.text),
            })
        results.append(sent_tokens)
    return results



@Language.factory("gliner-ner", default_config={"gliner_model": "fastino/gliner2-base-v1", "entities": ["location", "character"], "threshold": 0.3, "batch_size": 8, "gpu": False})
def create_gliner_component(nlp: Language, name: str, gliner_model: str, entities: Iterable[str], threshold: float, batch_size: int, gpu: bool):
    return GlinerNerComponent(gliner_model, entities, threshold, batch_size, gpu)

class GlinerNerComponent:
    def __init__(self, gliner_model, entities, threshold, batch_size, gpu):
        self.gliner_model = gliner_model
        self.batch_size = batch_size
        self.entities = list(entities)
        self.threshold = threshold
        self.model = GLiNER2.from_pretrained(self.gliner_model)

        if gpu:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

    def __call__(self, doc):

        if not Doc.has_extension("ents"): # Add doc._.ents dict
            Doc.set_extension("ents", default=True) 

        # TODO Arnaud: handle Batch processing -> batch_extract_entities
        out = self.model.extract_entities(
            text=doc._.resolved_text if Doc.has_extension("resolved_text") else doc.text,
            entity_types=self.entities,
            threshold=self.threshold,
            include_confidence=True,
        ) # -> Dict[str, Any]
        
        doc._.ents = out["entities"]

        return doc # Always return doc


# Convenience wrapper
def preprocess(
    text: str,
    *,
    nlp: Optional[Language] = None,
    do_tokenize: bool = True,
    do_sentences: bool = True,
    do_dependencies: bool = True,
    # do_coref: bool = False, TODO Arnaud
) -> Dict[str, Any]:
    """
    Run selected preprocessing steps. Each step can be enabled/disabled via boolean flags.

    Args:
      text: input text
      nlp: optional spacy Language pipeline to reuse, default to "en_core_web_sm"
      do_tokenize: run tokenize()
      do_sentences: run split_sentences()
      do_dependencies: run parse_dependencies()
      do_coref: run coreference resolution (not implemented, returns None)

    Returns a dict with keys: tokens, sentences, dependencies, doc.
    Keys for disabled steps will be None.
    """
    # Only load nlp if at least one step needs it
    if any([do_tokenize, do_sentences, do_dependencies]) and nlp is None:
        nlp = spacy.load("en_core_web_sm")

    result: Dict[str, Any] = {
        "tokens": None,
        "sentences": None,
        "dependencies": None,
        "doc": None,
    }
    doc = nlp(text)

    if do_tokenize:
        result["tokens"] = tokenize(doc)

    result["doc"] = doc

    if do_sentences:
        result["sentences"] = split_sentences(doc)

    if do_dependencies:
        result["dependencies"] = parse_dependencies(doc)

    return result