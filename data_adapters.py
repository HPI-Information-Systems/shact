"""
Different functions to convert datasets in different formats to a normalized format.
The normalized format contains an id, the text and a list of spans (start, end, label).
Each function return a list of dictionaries and a class label bidict.
"""
import json
from typing import List
from datasets import ClassLabel, Dataset,DatasetDict, Sequence, Value

class Sentence:
    """
    Class to represent a sentence.
    """
    def __init__(self, id, text, spans):
        self.id = id
        self.text = text
        self.spans = spans

    def __repr__(self):
        return f"Sentence(id={self.id}, text={self.text}, spans={self.spans})"

    def __str__(self):
        return f"Sentence(id={self.id}, text={self.text}, spans={self.spans})"

#conll03 converter based on huggingface's conll03.py (id (string)	tokens (sequence)	pos_tags (sequence)	chunk_tags (sequence)	ner_tags (sequence))
def conll03_ner_adapter(source_dataset:DatasetDict) -> DatasetDict:
    """
    Converts a conll03 dataset to the normalized format.
    """
    class_label_map:ClassLabel = source_dataset["train"].features["ner_tags"].feature
    #only keep the O and the class names from the B- tags
    new_class_label_map:ClassLabel = ClassLabel(names=(["O"] + [label[2:] for label in class_label_map.names if label.startswith("B-")])) 
    def map_iob_to_spans(iob_sequences:List[List[int]]):
        all_spans = []
        for iob_sequence in iob_sequences:
            spans = []
            current_entity = None
            for i, int_label in enumerate(iob_sequence):
                str_label = class_label_map.int2str(int_label)
                # check if label is an entity
                if str_label.startswith("B-"):
                    # check if there is an entity
                    if current_entity:
                        spans.append(current_entity)
                    current_entity = {"start":i, "end":i, "label":new_class_label_map.str2int(str_label[2:])}
                elif str_label.startswith("I-"):
                    if current_entity:
                        current_entity["end"] = i
                    else:
                        current_entity = {"start":i, "end":i, "label":new_class_label_map.str2int(str_label[2:])}
                else:
                    if current_entity:
                        spans.append(current_entity)
                        current_entity = None
            if current_entity:
                spans.append(current_entity)
            all_spans.append(spans)

        return all_spans
    # create mapping
    def map_batch(batch):
        spans=[]
        return {"id":batch["id"], "text":batch["tokens"], "spans":map_iob_to_spans(batch["ner_tags"])}
    
    target_dataset = DatasetDict()

    for split, dataset in source_dataset.items():
        target_dataset[split] = dataset.map(map_batch, batched=True, remove_columns=["tokens","pos_tags", "chunk_tags", "ner_tags"],  desc="Mapping to normalized format", keep_in_memory=True)
        target_dataset[split].features["spans"] = Sequence(feature={"start": Value("int32"), "end": Value("int32"), "label": new_class_label_map})

    return target_dataset

#conll03 converter based on huggingface's conll03.py (id (string)	tokens (sequence)	pos_tags (sequence)	chunk_tags (sequence)	ner_tags (sequence))
def conll03_chunk_adapter(source_dataset:DatasetDict) -> DatasetDict:
    """
    Converts a conll03 dataset to the normalized format.
    """
    feature="chunk_tags"
    class_label_map:ClassLabel = source_dataset["train"].features[feature].feature
    #only keep the O and the class names from the B- tags
    new_class_label_map:ClassLabel = ClassLabel(names=(["O"] + [label[2:] for label in class_label_map.names if label.startswith("B-")])) 
    def map_iob_to_spans(iob_sequences:List[List[int]]):
        all_spans = []
        for iob_sequence in iob_sequences:
            spans = []
            current_entity = None
            for i, int_label in enumerate(iob_sequence):
                str_label = class_label_map.int2str(int_label)
                # check if label is an entity
                if str_label.startswith("B-"):
                    # check if there is an entity
                    if current_entity:
                        spans.append(current_entity)
                    current_entity = {"start":i, "end":i, "label":new_class_label_map.str2int(str_label[2:])}
                elif str_label.startswith("I-"):
                    if current_entity:
                        current_entity["end"] = i
                    else:
                        current_entity = {"start":i, "end":i, "label":new_class_label_map.str2int(str_label[2:])}
                else:
                    if current_entity:
                        spans.append(current_entity)
                        current_entity = None
            if current_entity:
                spans.append(current_entity)
            all_spans.append(spans)

        return all_spans
    # create mapping
    def map_batch(batch):
        spans=[]
        return {"id":batch["id"], "text":batch["tokens"], "spans":map_iob_to_spans(batch[feature])}
    
    target_dataset = DatasetDict()

    for split, dataset in source_dataset.items():
        target_dataset[split] = dataset.map(map_batch, batched=True, remove_columns=["tokens","pos_tags", "chunk_tags", "ner_tags"],  desc="Mapping to normalized format", keep_in_memory=True)
        target_dataset[split].features["spans"] = Sequence(feature={"start": Value("int32"), "end": Value("int32"), "label": new_class_label_map})

    return target_dataset