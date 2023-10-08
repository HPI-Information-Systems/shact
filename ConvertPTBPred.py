from nltk import Tree
from pathlib import Path
import json
from typing import Dict, List
from multiprocessing import Pool
from tqdm import tqdm
import argparse

parser=argparse.ArgumentParser(description="Converts a jsonl file with predicted spans to a file with trees in parenthesis format")
parser.add_argument("input_file",type=str,help="Path to the input file")
parser.add_argument("output_file",type=str,help="Path to the output file")
parser.add_argument("--n_workers",type=int,default=8,help="Number of workers for multiprocessing")
args=parser.parse_args()

def get_tree(line:str) -> Tree:
    d=json.loads(line)
    tokens=d["tokens"]
    #replace ( and ) to -LRB- and -RRB-
    tokens=[token.replace("(","-LRB-").replace(")","-RRB-") for token in tokens]
    spans:List[Dict]=d["spans"]
    #add length to spans
    for span in spans:
        span["length"]=span["end"]-span["start"]
        span["children"]=[]
    #add missing word spans
    for i,token in enumerate(tokens):
        found=False
        for span in spans:
            if span["start"]==i and span["end"]==i+1:
                found=True
                break
        if not found:
            spans.append({"start":i,"end":i+1,"type":"X","length":1, "children":[]})
    #sort by length and start ascending
    spans.sort(key=lambda x: (x["length"],x["start"]))
    #build tree
    while len(spans)>1:
        span=spans.pop(0)
        for span2 in spans:
            if span2["start"]<=span["start"] and span2["end"]>=span["end"]:
                span2["children"].append(span)
                break
    def build_tree(span:Dict) -> Tree:
        type_span=span["type"]
        types_span=[type_span]
        if "+" in type_span:
            types_span=type_span.split("+")
        last_tree=None
        if span["children"]:
            #sort children by start
            span["children"].sort(key=lambda x: x["start"])
            last_tree=Tree(types_span[-1],[build_tree(child) for child in span["children"]])
        else:
            last_tree=Tree(types_span[-1],[tokens[span["start"]]])
        #loop from the second last type to the first
        for type_span in types_span[-2::-1]:
            last_tree=Tree(type_span,[last_tree])
        return last_tree

    return build_tree(spans[0])

def process_line(line:str) -> str:
    return get_tree(line).pformat(margin=1000000)

if __name__=="__main__":
    input_file=Path(args.input_file)
    output_file=Path(args.output_file)
    lines=input_file.read_text().splitlines()
    with Pool(args.n_workers) as p:
        trees=list(tqdm(p.imap(process_line,input_file.read_text().splitlines()),total=len(lines)))
    with open(output_file,"w") as f:
        f.write("\n".join(trees))