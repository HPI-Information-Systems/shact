from evaluate.info import EvaluationModuleInfo
from inference_model import LSHAC_NER_Prediction
from typing import List, Tuple
from evaluate import Metric

class NestedNERMetric():
    """
    Computes F1, precision and recall for nested NER
    """
    def __init__(self, types:List[str], flat:bool=False):
        """
        types: list of type names
        """
        self.types=types
        self.flat=flat

    def compute(self, predictions:List[LSHAC_NER_Prediction]=None, references:List[List[Tuple[Tuple[int,int],int,int]]]=None,zero_division=0) -> dict | None:
        """
        Computes F1, precision and recall for nested NER
        """
        tp=dict()
        fp=dict()
        fn=dict()

        false_topology=dict() # gt spans that are not in the predictions
        true_topology=dict() # gt spans that are in the predictions
        
        for prediction,ref in zip(predictions,references):
            ref_spans=set([(span,span_type) for (span,span_type,_) in ref])
            assignments=prediction.get_token_assignments(flat=self.flat)
            for (span,span_type) in assignments:
                if span_type not in tp:
                    tp[span_type]=0
                    fp[span_type]=0
                    fn[span_type]=0
                if (span,span_type) in ref_spans:
                    tp[span_type]+=1
                else:
                    fp[span_type]+=1
            for (span,span_type) in ref_spans:
                if (span,span_type) not in assignments:
                    if span_type not in tp:
                        tp[span_type]=0
                        fp[span_type]=0
                        fn[span_type]=0
                    fn[span_type]+=1
                if span_type not in true_topology:
                    true_topology[span_type]=0
                    false_topology[span_type]=0
                if prediction.is_cluster(span):
                    true_topology[span_type]+=1
                else:
                    false_topology[span_type]+=1                
                
        results=dict()
        for span_type in tp.keys():
            precision=tp[span_type]/(tp[span_type]+fp[span_type]) if (tp[span_type]+fp[span_type])>0 else zero_division
            recall=tp[span_type]/(tp[span_type]+fn[span_type]) if (tp[span_type]+fn[span_type])>0 else zero_division
            f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else zero_division
            true_topology_count=true_topology.get(span_type,0)
            false_topology_count=false_topology.get(span_type,0)
            potential_recall=true_topology_count/(true_topology_count+false_topology_count) if (true_topology_count+false_topology_count)>0 else zero_division
            results[self.types[span_type]]={"precision":precision,"recall":recall,"f1":f1, "potential_recall":potential_recall}
        all_tp=sum(tp.values())
        all_fp=sum(fp.values())
        all_fn=sum(fn.values())
        all_true_topology=sum(true_topology.values())
        all_false_topology=sum(false_topology.values())
        all_potential_recall=all_true_topology/(all_true_topology+all_false_topology) if (all_true_topology+all_false_topology)>0 else zero_division
        all_precision=all_tp/(all_tp+all_fp) if (all_tp+all_fp)>0 else zero_division
        all_recall=all_tp/(all_tp+all_fn) if (all_tp+all_fn)>0 else zero_division
        all_f1=2*all_precision*all_recall/(all_precision+all_recall) if (all_precision+all_recall)>0 else zero_division
        results["overall_f1"]=all_f1
        results["overall_precision"]=all_precision
        results["overall_recall"]=all_recall
        results["overall_potential_recall"]=all_potential_recall
        return results

                