from typing import List, Set
from sklearn.cluster import AgglomerativeClustering
from numpy import ndarray as np_arr

def compute_clusters(clust_model:AgglomerativeClustering)->List[Set[int]]:
    """
    Performs clustering on the given vectors and returns all the clusters as sets of indices
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
    return predicted_clusters