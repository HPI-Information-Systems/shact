"""
Different functions to convert datasets in different formats to a normalized format.
The normalized format contains an id, the text and a list of spans (start, end, label).
Start corresponds to the start index of the span, end to the end index of the span plus one and label to the index of the span's label.
The features object for the label is a ClassLabel object containing the label names.
Each function recieves a DatasetDict and returns a DatasetDict.
"""
import json
from typing import Dict, List
from datasets import ClassLabel, Dataset,DatasetDict, Features, Sequence, Value
from nltk import Tree

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
                    current_entity = {"start": i, "end": i+1, "label": new_class_label_map.str2int(str_label[2:])}
                elif str_label.startswith("I-"):
                    if current_entity:
                        current_entity["end"] = i+1
                    else:
                        current_entity = {"start": i, "end": i+1, "label": new_class_label_map.str2int(str_label[2:])}
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
        return {"id": batch["id"], "tokens": batch["tokens"], "spans": map_iob_to_spans(batch[feature])}

    target_dataset = DatasetDict()

    # list all columns that shoudl not be in the output
    columns_to_remove = [col for col in source_dataset["train"].column_names if col not in ["id", "tokens", "spans"]]
    features=Features({"id": Value(dtype="string"), "tokens": Sequence(feature=Value(dtype="string")), "spans": [{"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": new_class_label_map}]})
    #features["spans"]=[{"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": new_class_label_map}]
    target_dataset = source_dataset.map(
        map_batch, batched=True, remove_columns=columns_to_remove,
        desc="Mapping to normalized format", keep_in_memory=True, features=features, 
    )
    return target_dataset

#conll03 converter based on huggingface's conll03.py (id (string)	tokens (sequence)	pos_tags (sequence)	chunk_tags (sequence)	ner_tags (sequence))
def convert_conll03_ner_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a NER dataset to the normalized format.
    """
    feature = "ner_tags"
    return convert_iob_dataset(source_dataset, feature).filter(lambda example: len(example["tokens"])>0)

def convert_conll03_chunk_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Chunk dataset to the normalized format.
    """
    feature = "chunk_tags"
    return convert_iob_dataset(source_dataset, feature).filter(lambda example: len(example["tokens"])>0)

def convert_conll00_chunk_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """
    Converts a Chunk dataset to the normalized format and splits the train set into train and validation.
    """
    feature = "chunk_tags"
    new_dict=convert_iob_dataset(source_dataset, feature)
    split_dict=new_dict["train"].train_test_split(test_size=0.2,seed=42)
    new_dict["train"]=split_dict["train"]
    new_dict["validation"]=split_dict["test"]
    return new_dict.filter(lambda example: len(example["tokens"])>0)

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
        return {"id": batch["id"], "tokens": batch["tokens"], "spans": map_entity(batch["entities"])}
    for split in ["train","validation","test"]:
        #build the spans column using map
        columns_to_remove = [col for col in source_dataset["train"].column_names if col not in ["id", "tokens", "spans"]]
        target_dataset[split]=target_dataset[split].map(map_batch,batched=True, remove_columns=columns_to_remove,
            desc="Mapping to normalized format", keep_in_memory=True,)
        features=target_dataset[split].features.copy()
        features["spans"]=[{"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": class_label_obj}]
        target_dataset[split]=target_dataset[split].cast(features)
    return target_dataset.filter(lambda example: len(example["tokens"])>0)

def pre_process_ontonotes(source_dataset: DatasetDict) -> DatasetDict:
    def map_doc_batch(batch: Dict[str, List]) -> Dict[str, List]:
        new_batch = {"id": [], "sentence": []}
        for document_id, sentences in zip(batch["document_id"], batch["sentences"]):
            for i,sentence in enumerate(sentences):
                new_batch["id"].append(f"{document_id}-{i}")
                new_batch["sentence"].append(sentence)
        return new_batch
    return source_dataset.map(map_doc_batch, batched=True, keep_in_memory=True, remove_columns=["document_id", "sentences"])

def convert_ontonotes_en_ner(source_dataset:DatasetDict) -> DatasetDict:
    """
    Each document is split into sentences and the sentences are converted with IOB NER tags.
    """
    iob_dataset=DatasetDict()
    ne_feature = source_dataset["train"].features["sentences"][0]["named_entities"]
    sentence_dataset=pre_process_ontonotes(source_dataset)
    def map_sentence_batch(batch: Dict[str, List]) -> Dict[str, List]:
        new_batch = {"id": [], "tokens": [], "ner_tags": []}
        for document_id, sentence in zip(batch["id"], batch["sentence"]):
            new_batch["id"].append(document_id)
            new_batch["tokens"].append(sentence["words"])
            new_batch["ner_tags"].append(sentence["named_entities"])
        return new_batch
    features=Features({"id": Value(dtype="string"), "tokens": Sequence(feature=Value(dtype="string")), "ner_tags": ne_feature})
    iob_dataset=sentence_dataset.map(map_sentence_batch, batched=True, keep_in_memory=True, remove_columns=["sentence"], desc="Mapping to IOB format", features=features)
    return convert_iob_dataset(iob_dataset, "ner_tags").filter(lambda example: len(example["tokens"])>0)

def convert_ontonotes_parse_trees(source_dataset:DatasetDict) -> DatasetDict:
    """
    Each document is split into sentences and the sentences are converted with IOB NER tags.
    """
    span_dataset=DatasetDict()
    list_parse_tree_labels=["O","S","SBAR","SBARQ","SINV","SQ","ADJP","ADVP","CONJP","FRAG","INTJ","LST","NAC","NP","NX","PP","PRN","PRT","QP","RRC","UCP","VP","WHADJP","WHAVP","WHNP","WHPP","X","TOP"]
    class_label_obj: ClassLabel = ClassLabel(names=list_parse_tree_labels)
    sentence_dataset=pre_process_ontonotes(source_dataset)
    ignored_labels=set()
    def map_sentence_batch(batch: Dict[str, List]) -> Dict[str, List]:
        def extract_subtree_spans(tree):
            spans = []
            def traverse(node, start_index):
                nonlocal spans
                nonlocal ignored_labels

                # Get the span for the current node
                leaves = node.leaves()
                start = start_index
                end = start_index + len(leaves)
                if node.label() in list_parse_tree_labels:
                    label = class_label_obj.str2int(node.label()) #node.label()
                    # Append the span and label to the list as a dictionary
                    spans.append({"start": start, "end": end, "label": label})
                    #spans.append((start, end, label))
                else:
                    #raise ValueError(f"Label {node.label()} not in {list_parse_tree_labels}")
                    ignored_labels.add(node.label())
                # Recur for each child
                extra=0
                for child in node:
                    if isinstance(child, Tree):
                        child_leaves = traverse(child, start_index + extra)
                        extra+=len(child_leaves)
                return leaves
            # Start the traversal
            traverse(tree, 0)
            return spans
        new_batch = {"id": [], "tokens": [], "spans": []}
        for document_id, sentence in zip(batch["id"], batch["sentence"]):
            try:
                tree=Tree.fromstring(sentence["parse_tree"])
                spans=extract_subtree_spans(tree)
            except Exception as e:
                #print(e)
                #print(f"Ignored {document_id} - {sentence['parse_tree']}")
                continue
            new_batch["id"].append(document_id)
            new_batch["tokens"].append(sentence["words"])
            new_batch["spans"].append(spans)
        return new_batch
    print(f"Ignored labels: {ignored_labels}")
    span_features=Features({"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": class_label_obj})
    features=Features({"id": Value(dtype="string"), "tokens": Sequence(feature=Value(dtype="string")), "spans": list([span_features])})
    span_dataset=sentence_dataset.map(map_sentence_batch, batched=True, keep_in_memory=True, remove_columns=["sentence"], desc="Mapping to spans", features=features)
    return span_dataset.filter(lambda example: len(example["tokens"])>0)

def convert_ptb_parse_trees(source_dataset:DatasetDict) -> DatasetDict:
    """
    Each document has an id and a tree
    """
    span_dataset=DatasetDict()
    list_parse_tree_labels=['O',
                            'ADJP',
                            'ADVP',
                            'CONJP',
                            'FRAG',
                            'INTJ',
                            'LST',
                            'NAC',
                            'NP',
                            'NX',
                            'PP',
                            'PRN',
                            'PRT',
                            'QP',
                            'RRC',
                            'S',
                            'SBAR',
                            'SBARQ',
                            'SINV',
                            'SQ',
                            'TOP',
                            'UCP',
                            'VP',
                            'WHADJP',
                            'WHADVP',
                            'WHNP',
                            'WHPP',
                            'X',
                            'VBG', 'WDT', 'IN', 'WP', 'CC', 'NN', 'DT', 'VB', '.', ':', 'PDT', 'LS', 'RP', 'WP$', 'NNP', 'RBR', 'SYM', 'EX', 'WRB', 'FW', 'JJ', "''", '-RRB-', 'NNPS', 'JJR', 'JJS', '``', 'CD', 'RB', 'NNS', 'POS', 'VBN', 'VBP', ',', '#', 'PRP', 'MD', 'PRP$', 'UH', 'TO', '-LRB-', '$', 'VBZ', 'VBD', 'RBS',
                            'TOP+ADVP', 'ADVP+DT', 'ADVP+VB', 'NX+NX', 'PRT+JJ', 'FRAG+PP+IN', 'VP+NNS', 'UCP+ADJP', 'S+NP', 'TOP+ADJP', 'ADJP+ADVP', 'WHADVP+WDT', 'ADVP+NNP', 'NP+VBZ', 'LST+NNP', 'ADVP+NNS', 'QP+PDT', 'ADJP+RB', 'WHNP+QP', 'TOP+VP', 'NP+PRP$', 'ADVP+ADVP+JJR', 'ADJP+IN', 'WHNP+WDT', 'CONJP+IN', 'NP+NP+QP', 'PRT+RP', 'WHNP+VB', 'ADJP+ADJP', 'PRT+VBP', 'X+CD', 'VP+MD', 'X+NP', 'S+NP+PRP', 'NP+NP+NN', 'NX+NNP', 'NP+ADJP', 'INTJ+DT', 'S+ADJP+VBN', 'VP+NN', 'TOP+NP', 'PRN+FRAG+WHADJP', 'X+DT', 'SQ+VP+VBD', 'WHNP+NP', 'TOP+PRN', 'ADJP+CD', 'PRT+IN', 'UCP+PP', 'FRAG+NP+NNP', 'ADJP+VBP', 'ADVP+ADVP+RBR', 'NP+NNP', 'ADVP+PRT+RP', 'SBAR+S', 'TOP+SBAR', 'S+VP+VBN', 'PRT+NNP', 'VP+VBP', 'FRAG+ADJP', 'NP+SYM', 'SBAR+FRAG+NP', 'S+ADVP+RB', 'SBAR+S+VP+VBN', 'PP+RP', 'TOP+X+S', 'ADJP+JJ', 'ADVP+ADVP+RB', 'NP+IN', 'SBAR+WHNP', 'ADVP+RP', 'ADJP+DT', 'FRAG+ADVP+RB', 'LST+CD', 'NP+INTJ+DT', 'TOP+FRAG+PP', 'ADJP+QP', 'ADVP+RBR', 'TOP+PP', 'FRAG+SBAR', 'X+VP', 'S+ADJP+JJ', 'ADJP+ADJP+JJR', 'S+VP+VBZ', 'NP+UH', 'FRAG+NP+DT', 'WHADVP+JJR', 'SBAR+RB', 'WHNP+NNS', 'TOP+SBARQ', 'S+ADVP+NNP', 'FRAG+WHPP', 'S+ADVP+JJ', 'VP+NNP', 'NP+NP', 'ADVP+PRP', 'NX+S+VP', 'PRN+SBAR', 'ADJP+NN', 'S+VP+VBD', 'INTJ+VB', 'FRAG+DT', 'ADVP+VBZ', 'NX+NNPS', 'S+S', 'NP+NP+VBZ', 'X+SBARQ', 'TOP+NP+NP', 'FRAG+NP+NN', 'WHNP+WP', 'TOP+FRAG+SBAR', 'NP+PRN', 'S+ADJP+NNP', 'S+ADJP+NN', 'PP+IN', 'FRAG+WHADVP+WRB', 'FRAG+SBARQ', 'TOP+NP+VBN', 'ADJP+NNP', 'X+MD', 'TOP+X+ADVP', 'ADVP+IN', 'S+VP+NNS', 'PRT+VB', 'FRAG+NNP', 'NP+FW', 'NP+NP+NNP', 'NP+FRAG', 'S+VP+VB', 'TOP+X+SBARQ', 'NP+RBS', 'VP+ADVP', 'WHNP+WP$', 'FRAG+VP', 'TOP+S+VP+NNP', 'NP+CD', 'SBAR+SBARQ', 'WHADVP+DT', 'NX+S', 'FRAG+ADVP', 'NP+NN', 'S+NP+NNP', 'NP+WP', 'ADJP+VBG', 'ADVP+JJR', 'PP+TO', 'S+VP+TO', 'S+VP+VP', 'VP+VB', 'S+NP+NNPS', 'ADJP+VBD', 'QP+NNS', 'FRAG+WHADVP', 'FRAG+WHNP', 'NP+POS', 'S+ADVP', 'SBAR+WHADVP+WRB', 'VP+VP', 'NP+NP+RBR', 'ADJP+JJR', 'NP+LS', 'S+VP+NN', 'ADVP+UH', 'ADVP+NN', 'FRAG+S+ADJP+JJ', 'FRAG+S+VP', 'X+FW', 'VP+VBG', 'NAC+NN', 'NP+SBAR+S+VP', 'TOP+S', 'PRT+RB', 'NP+NP+RB', 'FRAG+WHADJP', 'NP+JJR', 'TOP+FRAG+WHADVP', 'LST+JJ', 'INTJ+RB', 'SBARQ+WHADVP+WRB', 'INTJ+WRB', 'SBAR+FRAG', 'VP+VBD', 'TOP+X', 'VP+POS', 'ADVP+ADJP+RBR', 'NP+NP+NP', 'VP+TO', 'FRAG+VP+NN', 'WHNP+WHNP+WP', 'TOP+NP+NN', 'TOP+FRAG', 'X+PP+IN', 'NP+NNPS', 'NX+NX+NN', 'TOP+FRAG+ADVP', 'INTJ+NN', 'LST+SYM', 'TOP+FRAG+NP', 'PRT+VBN', 'NP+DT', 'S+VP+MD', 'ADJP+VB', 'LST+LS', 'VP+NP+NN', 'ADVP+ADVP', 'WHADVP+IN', 'VP+VBN', 'FRAG+ADJP+RBR', 'VP+ADVP+RB', 'NP+VBP', 'NP+NP+CD', 'NP+VB', 'VP+JJ', 'NP+NP+PRP', 'S+ADVP+RBR', 'FRAG+ADJP+JJ', 'FRAG+ADJP+VBG', 'PRN+S', 'INTJ+S', 'PRN+S+VP+NNP', 'WHADVP+RB', 'X+SBAR', 'VP+PP', 'X+RB', 'S+VP+VBP', 'WHADVP+WRB', 'RRC+VP', 'TOP+X+NN', 'S+VP+NNP', 'ADJP+ADJP+JJ', 'NP+VBN', 'TOP+FRAG+NNP', 'NP+JJ', 'NAC+NNP', 'S+ADJP+VBG', 'S+VP+VBG', 'S+VP+JJ', 'NP+TO', 'X+TO', 'FRAG+ADJP+DT', 'NP+ADVP+DT', 'QP+CD', 'S+PP', 'ADVP+EX', 'NX+CD', 'TOP+S+VP', 'NP+WRB', 'INTJ+NNP', 'NP+RB', 'WHNP+DT', 'S+ADJP+RB', 'LST+:', 'S+UCP', 'PRN+NP', 'TOP+NP+S', 'WHNP+CD', 'SBAR+SINV', 'NP+PDT', 'X+VBD', 'NP+PP', 'ADVP+JJS', 'FRAG+UCP', 'ADJP+MD', 'NX+QP', 'INTJ+UH', 'ADVP+CC', 'NP+QP', 'ADJP+NNS', 'X+WP', 'ADVP+RB', 'X+SYM', 'X+PP+TO', 'X+NN', 'FRAG+PP', 'PP+JJ', 'WHNP+JJS', 'NX+NNS', 'NX+VBG', 'FRAG+ADJP+VBN', 'TOP+NP+NNP', 'ADJP+RBR', 'NX+JJ', 'FRAG+JJ', 'ADJP+JJS', 'PP+RB', 'VP+IN', 'WHNP+NNP', 'PP+PP', 'WHNP+WRB', 'WHNP+NN', 'X+IN', 'S+ADJP+JJR', 'WHNP+IN', 'WHNP+JJ', 'FRAG+NP', 'X+S', 'ADJP+VBN', 'PRN+SINV', 'S+ADJP', 'NX+NN', 'NP+WDT', 'NP+VBG', 'TOP+SINV', 'ADVP+WRB', 'S+ADVP+JJR', 'X+NNP', 'NP+PRP', 'ADJP+RP', 'S+ADJP+VB', 'VP+FRAG+ADJP+VBN', 'NP+SBAR', 'FRAG+WHNP+WP', 'TOP+UCP', 'NP+NP+NNS', 'S+NP+NNS', 'ADVP+CD', 'INTJ+VBP', 'VP+NNPS', 'VP+SBAR', 'X+ADVP', 'PRT+NN', 'NP+NP+JJR', 'VP+S+VP+NNS', 'QP+NN', 'S+NP+NN', 'PP+NP', 'FRAG+S', 'NP+EX', 'SBAR+SBAR+S', 'TOP+SQ', 'FRAG+RB', 'ADJP+ADJP+RBR', 'NP+MD', 'WHNP+NP+WDT', 'SBAR+WHNP+WDT', 'ADVP+RBS', 'SBAR+S+VP', 'S+VP+ADVP+RB', 'VP+NP', 'ADVP+VBD', 'NP+NNS', 'SQ+VP+VBZ', 'S+VP', 'PRN+PP', 'FRAG+NN', 'NP+S+VP', 'VP+RB', 'FRAG+VP+VB', 'VP+VBZ', 'NP+S', 'FRAG+VP+VBN', 'NP+NP+JJ', 'LST+RB', 'NP+VBD', 'INTJ+JJ', 'NP+JJS', 'QP+RBR', 'TOP+INTJ', 'SQ+VP', 'ADVP+JJ', 'NP+RBR', 'TOP+NP+NNS', 'NP+NP+DT', 'ADJP+NP', 'FRAG+INTJ+JJ',
                            ]
    class_label_obj: ClassLabel = ClassLabel(names=list_parse_tree_labels)
    sentence_dataset=source_dataset
    ignored_labels=set()
    def map_sentence_batch(batch: Dict[str, List]) -> Dict[str, List]:
        def extract_subtree_spans(tree):
            spans_l = []
            def traverse(node, start_index=0):
                nonlocal spans_l
                nonlocal ignored_labels

                # Get the span for the current node
                leaves = node.leaves()
                start = start_index
                end = start_index + len(leaves)
                if node.label() in list_parse_tree_labels:
                    label = node.label()
                    spans_l.append({"start": start, "end": end, "label": label})
                else:
                    ignored_labels.add(node.label())
                # Recur for each child
                extra=0
                for child in node:
                    if isinstance(child, Tree):
                        child_leaves = traverse(child, start_index + extra)
                        extra+=len(child_leaves)
                return leaves
            # Start the traversal
            traverse(tree)
            #compress all spans with the same start and end into one span concatenating the labels with a +
            spans2=[]
            for i,span in enumerate(spans_l):
                start=span["start"]
                end=span["end"]
                label=span["label"]
                for span2 in spans_l[i+1:]:
                    if span2["start"]==start and span2["end"]==end:
                        label=label+"+"+span2["label"]
                #check if span already exists
                already_exists=False
                for span2 in spans2:
                    if span2["start"]==start and span2["end"]==end:#can be more efficient with a set or a dict (hashing)
                        already_exists=True
                        break
                if not already_exists:
                    spans2.append({"start": start, "end": end, "label": label})
            #convert labels to int
            spans=[]
            for span in spans2:
                if span["label"] in list_parse_tree_labels:
                    spans.append({"start": span["start"], "end": span["end"], "label": class_label_obj.str2int(span["label"])})
                else:
                    ignored_labels.add(span["label"])
            return spans
        new_batch = {"id": [], "tokens": [], "spans": []}
        for document_id, tree_s in zip(batch["id"], batch["tree"]):
            try:
                tree=Tree.fromstring(tree_s)
                spans=extract_subtree_spans(tree)
                words=tree.leaves()
                #replace brackets
                words=[word.replace("-LRB-","(").replace("-RRB-",")") for word in words]

            except Exception as e:
                print(e)
                print(f"Ignored {document_id} - {tree_s}")
                continue
            new_batch["id"].append(document_id)
            new_batch["tokens"].append(words)
            new_batch["spans"].append(spans)
        return new_batch
    span_features=Features({"start": Value(dtype="int32"), "end": Value(dtype="int32"), "label": class_label_obj})
    features=Features({"id": Value(dtype="string"), "tokens": Sequence(feature=Value(dtype="string")), "spans": list([span_features])})
    span_dataset=sentence_dataset.map(map_sentence_batch, batched=True, keep_in_memory=True, remove_columns=["tree"], desc="Mapping to spans", features=features)
    print(f"Ignored labels: {ignored_labels}")
    return span_dataset.filter(lambda example: len(example["tokens"])>0)

def assert_columns(dataset:DatasetDict):
    for split in ["train","validation","test"]:
        assert "id" in dataset[split].column_names
        assert "tokens" in dataset[split].column_names
        assert "spans" in dataset[split].column_names
        assert "start" in dataset[split].features["spans"][0]
        assert "end" in dataset[split].features["spans"][0]
        assert "label" in dataset[split].features["spans"][0]
        assert isinstance(dataset[split].features["spans"][0]["label"],ClassLabel)