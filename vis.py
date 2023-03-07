from typing import List
import spacy
from spacy.tokens import Doc
from spacy import displacy
nlp = None

def visualize(tokens:List[str],tags_iob:List[str]) -> Doc:
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = Doc(nlp.vocab, words=tokens, ents=tags_iob)
    return displacy.render(doc, style="ent", page=True , jupyter=False)