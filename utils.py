import inspect
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import re
import torch

def filter_kwargs(f:Callable, kwargs: Dict) -> Dict:
    argspec = inspect.getfullargspec(f)
    return {k: v for k, v in kwargs.items() if k in argspec.args}

def get_connectivity_matrix(word_ids:List[int]) -> np.ndarray:
    """
    Generate a connectivity matrix for a chain of vectors
    Vectors can be connected to the left and to the right except
    Additionally, subword tokens are connected to the first token of the word or other tokens of the same word
    The first token of a word is connected to the previous and next word
    :param vectors: np.ndarray of shape (num_vectors, vector_dim)
    :return: np.ndarray of shape (num_vectors, num_vectors)
    """
    num_vectors = len(word_ids)
    connectivity_m = np.zeros((num_vectors,num_vectors))
    is_subword= lambda x: word_ids.count(word_ids[x])>1
    is_first_subword= lambda x: x==0 or word_ids[x]!=word_ids[x-1]
    for i,wid in enumerate(word_ids):
        if is_first_subword(i):
            if wid>0:
                prev_w=word_ids.index(word_ids[i-1])
                connectivity_m[i,prev_w]=1
            if wid<max(word_ids):
                ii=i+1
                while ii<len(word_ids) and word_ids[ii]==wid:
                    ii+=1
                connectivity_m[i,ii]=1
        if is_subword(i):
            other_tokens_sw=[j for j,w in enumerate(word_ids) if w==wid and j!=i]
            for j in other_tokens_sw:
                connectivity_m[i,j]=1
            
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
    

regex_extract_type=re.compile(r"[B,I]-(.*)")