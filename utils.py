import inspect
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import re
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import PreTrainedTokenizerFast

def filter_kwargs(f:Callable, kwargs: Dict) -> Dict:
    argspec = inspect.getfullargspec(f)
    return {k: v for k, v in kwargs.items() if k in argspec.args}

def build_tensors_for_inference(words:List[str],tokenizer:PreTrainedTokenizerFast) -> Dict[str,torch.Tensor]:
    """
    Prepares a sentence for inference, returns inputs and word_ids
    """
    inputs=tokenizer(words,return_tensors="pt",is_split_into_words=True,padding=True,return_attention_mask=True,add_special_tokens=False,return_special_tokens_mask=True)
    word_ids=torch.tensor([inputs.word_ids()])
    return {"inputs":inputs,
            "all_word_ids":word_ids}

def get_connectivity_matrix(orig_word_ids:List[int]) -> np.ndarray:
    """
    Generate a connectivity matrix for a chain of vectors
    Vectors can be connected to the left and to the right except
    Additionally, subword tokens are connected to the first token of the word or other tokens of the same word
    The first token of a word is connected to the previous and next word
    :param vectors: np.ndarray of shape (num_vectors, vector_dim)
    :return: np.ndarray of shape (num_vectors, num_vectors)
    """
    #remove duplicates
    word_ids=[]
    for wid in orig_word_ids:
        if len(word_ids)==0 or wid!=word_ids[-1]:
            word_ids.append(wid)
    num_vectors = len(word_ids)
    connectivity_m = np.zeros((num_vectors,num_vectors))
    for i,wid in enumerate(word_ids):
        if wid>0:
            prev_w=word_ids.index(word_ids[i-1])
            connectivity_m[i,prev_w]=1
        if wid<max(word_ids):
            next_w=i+1
            connectivity_m[i,next_w]=1
            
        # else:
        #     if wid>0:
        #         prev_w=word_ids.index(word_ids[i-1])
        #         connectivity_m[i,prev_w]=1
        #     if wid<max(word_ids):
        #         ii=i+1
        #         while ii<len(word_ids) and word_ids[ii]==wid:
        #             ii+=1
        #         connectivity_m[i,ii]=1
    return connectivity_m

def get_potential_recall(clusters:List[List[Tuple[int,int]]], batch:Dict) -> Dict[int,float]:
    """
    Computes the potential recall of the clusters in the batch
    :param clusters: List of tuples (start, end) of the clusters reuslting from HAC
    :param batch: Dictionary with keys "all_word_ids", "final_cluster_masks", "labels" and "inputs"
    """
    results={}
    found={}
    total={}
    for gt_clusters,gt_labels,hac_clusters in zip(batch["final_cluster_masks"],batch["labels"],clusters):
        for gt_cluster in gt_clusters:
            indices=torch.argwhere(gt_cluster==1).squeeze(-1)
            if indices.shape[0]==0:
                continue
            min=torch.min(indices).item()
            max=torch.max(indices).item()
            label_type_idx=gt_labels[min].item()
            if total.get(label_type_idx) is None:
                total[label_type_idx]=0
                found[label_type_idx]=0
            total[label_type_idx]+=1
            if (min,max) in hac_clusters:
                found[label_type_idx]+=1
    for label_type_idx in total.keys():
        results[label_type_idx]=found[label_type_idx]/total[label_type_idx]
    return results

def get_potential_recall_nested(clusters:List[List[Tuple[int,int]]], batch:Dict) -> Dict[int,float]:
    """
    Computes the potential recall of the clusters in the batch
    :param clusters: List of tuples (start, end) of the clusters reuslting from HAC
    :param batch: Dictionary with keys "all_word_ids", "final_cluster_masks", "labels" and "inputs"
    """
    #TODO refactor this function to avoid code duplication with get_potential_recall
    results={}
    found={}
    total={}
    for gt_clusters,gt_types,hac_clusters in zip(batch["final_cluster_masks"],batch["types"],clusters):
        for gt_cluster,gt_type in zip(gt_clusters,gt_types):
            indices=torch.argwhere(gt_cluster==1).squeeze(-1)
            if indices.shape[0]==0:
                continue
            min=torch.min(indices).item()
            max=torch.max(indices).item()
            label_type_idx=gt_type.item()
            if total.get(label_type_idx) is None:
                total[label_type_idx]=0
                found[label_type_idx]=0
            total[label_type_idx]+=1
            if (min,max) in hac_clusters:
                found[label_type_idx]+=1
    for label_type_idx in total.keys():
        results[label_type_idx]=found[label_type_idx]/total[label_type_idx]
    return results

def get_confusion_matrix(gt_spans:List[List[Tuple[Tuple[int, int], int, torch.Tensor]]],predictions:List):
    """
    Computes the confusion matrix of the predictions
    :param gt_spans: List of lists of tuples ((start, end),type,type_ohe) of the clusters reuslting from HAC
    :param predictions: List of LSHAC_NER_Prediction with predictions
    """
    confusion_matrix={}
    for gt_clusters,prediction in zip(gt_spans,predictions):
        for gt_cluster in gt_clusters:
            gt_span=gt_cluster[0]
            gt_type=gt_cluster[1]
            if confusion_matrix.get(gt_type) is None:
                confusion_matrix[gt_type]={}
            for (pred_span,pred_type) in prediction.assignments:
                if confusion_matrix[gt_type].get(pred_type) is None:
                    confusion_matrix[gt_type][pred_type]=0
                if gt_span==pred_span:
                    confusion_matrix[gt_type][pred_type]+=1
    return confusion_matrix

def is_sublist(self,sublist,list):
    if len(sublist)>len(list):
        return False
    for ii in range(len(list)-len(sublist)+1):
        if sublist==list[ii:ii+len(sublist)]:
            return True
    return False


regex_extract_type=re.compile(r"[B,I]-(.*)")