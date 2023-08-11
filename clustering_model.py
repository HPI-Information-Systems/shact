from typing import List, Set, Tuple
import typing
from sklearn.cluster import AgglomerativeClustering
from numpy import ndarray as np_arr
import numpy as np

def get_word_spans(word_ids:List[int]) -> List[Set[int]]:
    """returns a list of spans derived from the words ids"""
    extra_spans=[]
    word_ids_np=np.array(word_ids)
    max_word_id=word_ids_np.max()
    for j in range(max_word_id+1):
        indices=np.argwhere(word_ids_np==j).squeeze(-1)
        if indices.shape[0]==0:
            continue
        min=indices.min().item()
        max=indices.max().item()
        extra_spans.append(set(range(min,max+1)))
    return extra_spans

def compute_clusters(clust_model:AgglomerativeClustering, word_ids:List[int])->List[Set[int]]:
    """
    Performs clustering on the given vectors and returns all the clusters as sets of indices.
    Only clsuters with complete words are returned
    :param clust_model: AgglomerativeClustering model fitted on the vectors
    :param word_ids: List of word ids
    """
    predicted_clusters=[]
    len_tokens=clust_model.n_leaves_
    for cluster_merge in clust_model.children_:
        cluster_set=set()
        for i in cluster_merge:
            if i<len_tokens:
                cluster_set.add(i)
            else:
                cluster_set=cluster_set.union(predicted_clusters[i-len_tokens])
        predicted_clusters.append(cluster_set)
    #predicted_clusters=[c for c in predicted_clusters if not broken_word(word_ids, c)]
    predicted_clusters.extend([{w_id} for w_id in set(word_ids)])
    # word_spans=get_word_spans(word_ids)
    # for word_span in word_spans:
    #     if word_span not in predicted_clusters:
    #         predicted_clusters.append(word_span)
    return predicted_clusters

def compute_clusters_thread(clustering_model,token_ls_vectors,word_ids_list):
    predicted_clusters=[{0}]
    vectors=dict()
    for ii,word_ids in enumerate(word_ids_list):
        if word_ids!=-1:
            if vectors.get(word_ids) is None:
                vectors[word_ids]=[]
            vectors[word_ids].append(token_ls_vectors[ii])
    for word_ids in vectors.keys():
        if len(vectors[word_ids])>1:
            #average the vectors
            vectors[word_ids]=np.mean(vectors[word_ids],axis=0)
        else:
            vectors[word_ids]=vectors[word_ids][0]
    vector_to_cluster=[]
    all_word_ids=vectors.keys()
    #sort
    all_word_ids=sorted(all_word_ids)
    for word_ids in all_word_ids:
        vector_to_cluster.append(vectors[word_ids])
    assert len(vector_to_cluster)==len(all_word_ids)
    if len(vector_to_cluster)>1:
        predicted_clusters=compute_clusters(clustering_model.fit(vector_to_cluster),word_ids_list)
    return predicted_clusters

def broken_word(word_ids:List[int], cluster:Set[int])->bool:
    """
    Checks if a cluster contains a broken word
    :param word_ids: List of word ids
    :param cluster: Set of indices of the cluster
    """
    cluster_word_ids=set([word_ids[i] for i in cluster])
    return any([(w in cluster_word_ids and i not in cluster) for i,w in enumerate(word_ids)])