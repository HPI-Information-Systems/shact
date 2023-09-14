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
def convert_conll_ner_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a NER dataset to the normalized format.
    """
    feature = "ner_tags"
    return convert_iob_dataset(source_dataset, feature)

def convert_conll_chunk_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Chunk dataset to the normalized format.
    """
    feature = "chunk_tags"
    return convert_iob_dataset(source_dataset, feature)

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
            desc="Mapping to normalized format", keep_in_memory=True
        )
        target_dataset[split].features["spans"] = Sequence(
            feature={"start": Value("int32"), "end": Value("int32"), "label": new_class_label_map}
        )

    return target_dataset