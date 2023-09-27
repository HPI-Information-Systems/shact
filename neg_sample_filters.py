"""
Task specific filters for negative sampling
They recieve teh ground truth spans and a candidate span and return True if the candidate span is a valid negative sample
"""

from typing import List, Tuple


def chunk_filter(ground_truth_spans:List[Tuple[int,int]], candidate_span:Tuple[int,int])->bool:
    """
    candidate_span can be a negative sample if it is contained in any ground truth span
    """
    for (s,e) in ground_truth_spans:
        if candidate_span[0]<=s and candidate_span[1]>=e:
            return False
    return True

def flat_ner_filter(ground_truth_spans:List[Tuple[int,int]], candidate_span:Tuple[int,int])->bool:
    """
    candidate_span can be a negative sample if it is not contained in any ground truth span
    """
    for (s,e) in ground_truth_spans:
        if candidate_span[0]>=s and candidate_span[1]<=e:
            return False
    return True

def no_filter(ground_truth_spans:List[Tuple[int,int]], candidate_span:Tuple[int,int])->bool:
    """
    any span can be a negative sample
    """
    return True

