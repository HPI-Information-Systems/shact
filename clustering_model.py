from typing import List, Set
import typing
from sklearn.cluster import AgglomerativeClustering
from numpy import ndarray as np_arr

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
        if len(cluster_set)<len_tokens:#ignore complete set
            predicted_clusters.append(cluster_set)
    predicted_clusters=[c for c in predicted_clusters if not broken_word(word_ids, c)]
    return predicted_clusters

def broken_word(word_ids:List[int], cluster:Set[int])->bool:
    """
    Checks if a cluster contains a broken word
    :param word_ids: List of word ids
    :param cluster: Set of indices of the cluster
    """
    cluster_word_ids=set([word_ids[i] for i in cluster])
    return any([(w in cluster_word_ids and i not in cluster) for i,w in enumerate(word_ids)])