from typing import List, Tuple
import torch

class LSHAC_NER_Prediction():
    #class with clusters and types for each cluster fro a single sentence
    def __init__(self,clusters:List[Tuple[int,int]],logits:torch.Tensor,types_list:List[str],sentence_mask:torch.Tensor, word_ids:List[int]=None):
        assert len(sentence_mask.shape)==1, "sentence_mask must be a 1D tensor. results are designed for a single sentence"
        self.sentence_mask=sentence_mask.cpu()
        self.seq_length=self.sentence_mask.sum().item()
        self.clusters=clusters
        self.logits=logits.cpu()
        assert len(clusters)==len(logits)
        self.types_list=types_list
        self.assignments=[]
        self.confidence=[]
        self.prelim_not_entities=[]
        self.part_of_entities=[]
        self.probs=torch.softmax(self.logits,dim=1)
        for cluster,prob in zip(clusters,self.probs):
            class_ix=torch.argmax(prob).item()
            if class_ix!=types_list.index("O"):
                self.assignments.append((cluster,class_ix))
                self.confidence.append(prob[class_ix].item())
            else:
                self.prelim_not_entities.append(cluster)

        assignments_with_confidence=[(c,a,conf) for (c,a),conf in zip(self.assignments,self.confidence)]
        #removes overlapping clusters. Keeps the one with the highest confidence
        sorted_awc=[(c,a,conf) for (c,a,conf) in sorted(assignments_with_confidence,key=lambda x: x[2],reverse=True)]
        self.flat_assignments=[]
        is_child=lambda x,y: x[0]>=y[0] and x[1]<=y[1]
        is_child_existing=lambda x: any([is_child(x,(ini,end)) for ((ini,end),_) in self.flat_assignments])
        overlaps=lambda x,y: is_child(x,y) or is_child(y,x)
        overlaps_existing=lambda x: any([overlaps(x,(ini,end)) for ((ini,end),_) in self.flat_assignments])
        for (c,a,conf) in sorted_awc:
            if not overlaps_existing(c):
                self.flat_assignments.append((c,a))
            else:
                if is_child_existing(c):
                    self.part_of_entities.append(c)
        self.not_entities=[c for c in self.prelim_not_entities if c not in self.part_of_entities]
        #removes nested clusters. Keeps the bigger one
        #self.flat_assignments=[(c,a) for (c,a) in self.assignments if (not is_nested(c))]
        self.seq_labels=self._get_seq_labels()
        self.seq_labels_compressed=None
        self.word_ids=word_ids
        if word_ids:
            # if word_ids are provided, we compress the labels to remove the partial word token labels
            # we asume the first token of a word is the one with the label
            self.seq_labels_compressed=[]
            for i,wid in enumerate(word_ids):
                if wid>=0 and (i==0 or wid!=word_ids[i-1]):
                    self.seq_labels_compressed.append(self.seq_labels[i])

    def all_predictions(self)->List[Tuple[Tuple[int,int],int]]:
        return [(c,a) for (c,a) in self.flat_assignments]+[(c,self.types_list.index("O")) for c in self.not_entities]+[(c,self.types_list.index("O")) for c in self.part_of_entities]
        
    def __repr__(self):
        return f"LSHAC_NER_Prediction(assignments={self.assignments},types_list={self.types_list},seq_length={self.seq_length},sentence_mask={self.sentence_mask})"
    
    def _get_seq_labels(self) -> List[Tuple[Tuple[int,int],int]]:
        seq_labels=[]
        for i in range(self.sentence_mask.shape[0]):
            if self.sentence_mask[i]!=1:
                continue
            containing_clusters=[(c,a) for (c,a) in self.flat_assignments if i in range(c[0],c[1]+1)]
            cluster,assignment=(containing_clusters[0][0],containing_clusters[0][1]) if len(containing_clusters)>0 else (None,None)
            if cluster:
                if i==cluster[0]:
                    seq_labels.append("B-"+self.types_list[assignment])
                else:
                    seq_labels.append("I-"+self.types_list[assignment])
            else:
                seq_labels.append("O")
        assert len(seq_labels)==self.seq_length
        return seq_labels
    
    def get_word_spans(self, clusters:List[Tuple[int,int]]=None)->List[Tuple[int,int]]:
        """
        returns all the clusters as word spans
        """
        if clusters is None:
            clusters=self.clusters
        word_spans=[]
        for (s,e) in clusters:
            min_ix=self.word_ids[s]
            max_ix=self.word_ids[e]
            word_spans.append((min_ix,max_ix))
        return word_spans

    
    def get_pydot_tree(self, flat=False):
        import pydot
        assign=self.flat_assignments if flat else self.assignments
        #add single words

        assigned_spans=[x[0] for x in assign]
        for i in range(max(self.word_ids)+1):
            if i not in self.word_ids:
                continue
            min_ix=self.word_ids.index(i)
            max_ix=len(self.word_ids)-self.word_ids[::-1].index(i)-1
            if (min_ix,max_ix) not in assigned_spans:
                assign.append(((min_ix,max_ix),self.types_list.index("O")))
        assigned_spans=[x[0] for x in assign]
        for (s,e) in self.not_entities:
            if (s,e) not in assigned_spans:
                assign.append(((s,e),self.types_list.index("O")))
        min_total=self.word_ids.index(0)
        max_total=len(self.word_ids)-self.word_ids[::-1].index(max(self.word_ids))-1
        assign.append(((min_total,max_total),self.types_list.index("O")))

        assign_by_length_desc=sorted(assign,key=lambda t: (t[0][1]-t[0][0],-t[0][1]),reverse=False)
        assigned_spans=[x[0] for x in assign_by_length_desc]
        G=pydot.Dot(graph_type='digraph',strict=True)
        for a in sorted(assign,key=lambda t: (t[0][0],t[0][1]-t[0][0])):
            class_id=a[1]
            color=class_id%12+1
            if class_id==self.types_list.index("O"):
                color="white"
            node=pydot.Node(str(a[0]),label=f"{a[0]} {self.types_list[a[1]]}", style="filled", fillcolor=color, colorscheme="paired12")
            G.add_node(node)
        already_added=[]
        while len(assign_by_length_desc)>0:
            a=assign_by_length_desc.pop()
            #reverse list
            for a2 in already_added[::-1]:
                if self._is_child(a[0],a2[0]):
                    G.add_edge(pydot.Edge(str(a2[0]),str(a[0])))
                    break
            already_added.append(a)
        return G
    
    def _is_child(self,child,parent):
        return child[0]>=parent[0] and child[1]<=parent[1]