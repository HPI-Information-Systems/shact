from typing import List, Tuple
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy
nlp = None

def generate_colors(n:int):
    """
    Automatically generated function
    """
    import colorsys
    #start with grey
    colors=["rgb(128,128,128)"]
    for i in range(n-1):
        hue=i/n
        sat=0.5
        val=0.8
        rgb=colorsys.hsv_to_rgb(hue,sat,val)
        colors.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")
    return colors

def visualize_iob(tokens:List[str],tags_iob:List[str]):
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = Doc(nlp.vocab, words=tokens, ents=tags_iob)
    return displacy.render(doc, style="ent", page=True , jupyter=False)

def visualize_spans(tokens:List[str],spans:List[Tuple[int,int,str]], jupyter=False, colors=None):
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = Doc(nlp.vocab, words=tokens)
    spans_doc=[]
    for span in spans:
        start=span[0]
        end=span[1]+1
        label=span[2]
        spans_doc.append(Span(doc, start, end, label=label))
    doc.spans["sc"]=spans_doc
    return displacy.render(doc, style="span", page=False , jupyter=jupyter, options={"colors":colors})