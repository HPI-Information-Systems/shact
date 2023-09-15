"""
Different functions to convert datasets in different formats to a normalized format.
The normalized format contains an id, the text and a list of spans (start, end, label).
Each function return a list of dictionaries and a class label bidict.
"""
import json
from typing import List
from datasets import ClassLabel, Dataset,DatasetDict, Sequence, Value

def convert_iob_dataset(source_dataset: DatasetDict, feature: str) -> DatasetDict:
    """
    Converts a dataset to the normalized format.
    """
    class_label_map: ClassLabel = source_dataset["train"].features[feature].feature
    # Only keep the O and the class names from the B- tags
    new_class_label_map: ClassLabel = ClassLabel(
        names=(["O"] + [label[2:] for label in class_label_map.names if label.startswith("B-")])
    )

    def map_iob_to_spans(iob_sequences: List[List[int]]):
        all_spans = []
        for iob_sequence in iob_sequences:
            spans = []
            current_entity = None
            for i, int_label in enumerate(iob_sequence):
                str_label = class_label_map.int2str(int_label)
                # Check if label is an entity
                if str_label.startswith("B-"):
                    # Check if there is an entity
                    if current_entity:
                        spans.append(current_entity)
                    current_entity = {"start": i, "end": i, "label": new_class_label_map.str2int(str_label[2:])}
                elif str_label.startswith("I-"):
                    if current_entity:
                        current_entity["end"] = i
                    else:
                        current_entity = {"start": i, "end": i, "label": new_class_label_map.str2int(str_label[2:])}
                else:
                    if current_entity:
                        spans.append(current_entity)
                        current_entity = None
            if current_entity:
                spans.append(current_entity)
            all_spans.append(spans)

        return all_spans

    # Create mapping
    def map_batch(batch):
        return {"id": batch["id"], "text": batch["tokens"], "spans": map_iob_to_spans(batch[feature])}

    target_dataset = DatasetDict()

    for split, dataset in source_dataset.items():
        # list all columns that shoudl not be in the output
        columns_to_remove = [col for col in source_dataset["train"].column_names if col not in ["id", "text", "spans"]]
        target_dataset[split] = dataset.map(
            map_batch, batched=True, remove_columns=columns_to_remove,
            desc="Mapping to normalized format", keep_in_memory=True, 
        )
        features=target_dataset[split].features.copy()
        features["spans"]=[{"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": new_class_label_map}]
        target_dataset[split]=target_dataset[split].cast(features)

    return target_dataset

#conll03 converter based on huggingface's conll03.py (id (string)	tokens (sequence)	pos_tags (sequence)	chunk_tags (sequence)	ner_tags (sequence))
def convert_conll03_ner_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a NER dataset to the normalized format.
    """
    feature = "ner_tags"
    return convert_iob_dataset(source_dataset, feature)

def convert_conll03_chunk_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Chunk dataset to the normalized format.
    """
    feature = "chunk_tags"
    return convert_iob_dataset(source_dataset, feature)

def convert_conll00_chunk_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Chunk dataset to the normalized format and splits the train set into train and validation.
    """
    feature = "chunk_tags"
    new_dict=convert_iob_dataset(source_dataset, feature)
    split_dict=new_dict["train"].train_test_split(test_size=0.2,seed=42)
    new_dict["train"]=split_dict["train"]
    new_dict["validation"]=split_dict["test"]
    return new_dict

def convert_genia_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Genia dataset to the normalized format.
    """
    target_dataset=DatasetDict()
    entity_types=["O"]
    all_entities=source_dataset["train"]["entities"]
    #build a set of all entity types efficiently
    for entities in all_entities:
        for entity in entities:
            if entity["type"] not in entity_types:
                entity_types.append(entity["type"])
    entity_types
    #build ClassLabel from entity types
    class_label_obj: ClassLabel = ClassLabel(names=list(entity_types))
    split_sizes=[len(source_dataset["train"]),len(source_dataset["validation"]),len(source_dataset["test"])]
    #cumulative sum
    split_sizes=[sum(split_sizes[:i+1]) for i in range(len(split_sizes))]
    #build an id column for each split
    #get the ranges
    init=[0]+split_sizes[:-1]
    ranges=[range(init[i],split_sizes[i]) for i in range(len(split_sizes))]
    #build the id column
    for i,split in enumerate(["train","validation","test"]):
        target_dataset[split]=source_dataset[split].add_column("id",[str(i) for i in ranges[i]])
    def map_entity(all_entities:List[List[dict]]):
        all_spans=[]
        for entities in all_entities:
            spans=[]
            for entity in entities:
                spans.append({"start":entity["start"], "end":entity["end"], "label":class_label_obj.str2int(entity["type"])})
            all_spans.append(spans)
        return all_spans
    def map_batch(batch):
        return {"id": batch["id"], "text": batch["tokens"], "spans": map_entity(batch["entities"])}
    for split in ["train","validation","test"]:
        #build the spans column using map
        columns_to_remove = [col for col in source_dataset["train"].column_names if col not in ["id", "text", "spans"]]
        target_dataset[split]=target_dataset[split].map(map_batch,batched=True, remove_columns=columns_to_remove,
            desc="Mapping to normalized format", keep_in_memory=True,)
        features=target_dataset[split].features.copy()
        features["spans"]=[{"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": class_label_obj}]
        target_dataset[split]=target_dataset[split].cast(features)
    return target_dataset
        
        


