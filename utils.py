import inspect
from typing import Callable, Dict
import numpy as np
import re

def filter_kwargs(f:Callable, kwargs: Dict) -> Dict:
    argspec = inspect.getfullargspec(f)
    return {k: v for k, v in kwargs.items() if k in argspec.args}

def connectivity_matrix(vectors:np.ndarray) -> np.ndarray:
    """
    Generate a connectivity matrix for a chain of vectors
    Vectors can be connected to the left and to the right
    :param vectors: np.ndarray of shape (num_vectors, vector_dim)
    :return: np.ndarray of shape (num_vectors, num_vectors)
    """
    num_vectors = vectors.shape[0]
    connectivity_m = np.zeros((num_vectors,num_vectors))
    for i in range(num_vectors):
        if i-1>=0:
            connectivity_m[i,i-1]=1
        if i+1<num_vectors:
            connectivity_m[i,i+1]=1
    return connectivity_m

regex_extract_type=re.compile(r"[B,I]-(.*)")