from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from datasets import ClassLabel, load_metric
from transformers import BertModel
from sklearn.cluster import AgglomerativeClustering
from clustering_model import compute_clusters
import data_modules as dm
from latent_space import hac_sl_ratio_loss, hac_sl_ratio_loss_token_based
import utils

class LSHAC_NER_Prediction():
    #class with clusters and types for each cluster fro a single sentence
    def __init__(self,clusters:List[Tuple[int,int]],logits:torch.Tensor,types_list:List[str],sentence_mask:torch.Tensor, word_ids:List[int]=None):
        assert len(sentence_mask.shape)==1, "sentence_mask must be a 1D tensor. results are designed for a single sentence"
        self.seq_length=sentence_mask.sum().item()
        self.sentence_mask=sentence_mask
        self.clusters=clusters
        self.logits=logits
        assert len(clusters)==len(logits)
        self.types_list=types_list
        self.assignments=[]
        for cluster,logit in zip(clusters,logits):
            class_ix=torch.argmax(logit).item()
            if class_ix!=types_list.index("O"):
                self.assignments.append((cluster,torch.argmax(logit).item()))
        self.seq_labels=self._get_seq_labels()
        self.seq_labels_compressed=None
        if word_ids:
            self.seq_labels_compressed=[]
            for i,wid in enumerate(word_ids):
                if wid>=0 and (i==0 or wid!=word_ids[i-1]):
                    self.seq_labels_compressed.append(self.seq_labels[i])

    def __repr__(self):
        return f"LSHAC_NER_Prediction(assignments={self.assignments},types_list={self.types_list},seq_length={self.seq_length},sentence_mask={self.sentence_mask})"
    
    def _get_seq_labels(self) -> List[Tuple[Tuple[int,int],int]]:
        seq_labels=[]
        is_nested=lambda x: any([(x!=(ini,end) and x[0]>=ini and x[1]<=end) for ((ini,end),_) in self.assignments])
        #removes nested clusters. Keeps the bigger one
        pruned_assignments=[(c,a) for (c,a) in self.assignments if (not is_nested(c))]
        for i in range(self.sentence_mask.shape[0]):
            if self.sentence_mask[i]!=1:
                continue
            containing_clusters=[(c,a) for (c,a) in pruned_assignments if i in range(c[0],c[1]+1)]
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

#Pytorch lighning NER model with BERT as the underlying model
class LSHAC_NERModel(pl.LightningModule):
    def __init__(self, transformer_model: BertModel,
                 classes: ClassLabel,
                 lr=1e-3,
                 ls_hidden_size=128, 
                 distance_fn: Callable = torch.cdist, 
                 hac_metric=None,
                 type_weights:Dict[str,float]=None,):
        super().__init__()
        self.transformer_model=transformer_model
        self.orig_classes=classes
        self.types=[]
        orig_label_names=self.orig_classes.names
        self.types,self.class_type_mapping=self._get_types_mapping(self.orig_classes)
        self.ls_hidden_size=ls_hidden_size
        self.fc_classif = nn.Linear(transformer_model.config.hidden_size*2, len(self.types)) # last one for not entities
        full_hidden_size=transformer_model.config.hidden_size*(transformer_model.config.num_hidden_layers+1)
        self.ls_proj=nn.Linear(full_hidden_size,ls_hidden_size)
        self.lr=lr
        self.distance_fn=distance_fn
        self.hac_metric=hac_metric
        if (not hac_metric) and distance_fn==torch.cdist:
            self.hac_metric="euclidean"
        weigths=self._align_weights(type_weights)
        self.loss_fn=nn.CrossEntropyLoss(weight=weigths)  
        self.seqeval_metric=load_metric("seqeval")        

    def save_hyperparameters(self,**kwargs):
        kwargs.setdefault("ignore",[]).append("transformer_model")
        super().save_hyperparameters(**kwargs)

    def _encode(self, **x)->Tuple[torch.Tensor,torch.Tensor]:
        #encodes the input x using the transformer model
        hidden_states=self.transformer_model(**utils.filter_kwargs(self.transformer_model.forward,x),output_hidden_states=True).hidden_states
        h=torch.cat(hidden_states,dim=-1)
        ls=self.ls_proj(h)
        final_layer=hidden_states[-1]
        return final_layer,ls

    def _get_types_mapping(self,class_labels:ClassLabel)->Tuple[List[str],Dict[str,str]]:
        regex=utils.regex_extract_type
        new_types=[]
        mapping={}
        for name in class_labels.names:
            match=regex.match(name)
            if match:
                new_name=match.group(1)
                if new_name not in new_types:
                    new_types.append(match.group(1))
                mapping[name]=new_name
            else:
                new_types.append(name)
                mapping[name]=name
        return new_types,mapping

    def _align_weights(self,type_weights:Dict[str,float]=None) -> Optional[torch.Tensor]:
        if type_weights:
            weights=torch.zeros(len(self.types))
            for n,w in type_weights.items():
                weights[self.types.index(n)]=w
            return weights
        else:
            return None

    def _get_type_idx(self,class_label:int)->int:
        return self.types.index(self.class_type_mapping[self.orig_classes.names[class_label]])
    
    def _get_clustering_model(self, word_ids):
        #returns the clustering model for the given word ids
        connectivity_matrix=utils.get_connectivity_matrix(word_ids)
        return AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='single',distance_threshold=0,metric=self.hac_metric, connectivity=connectivity_matrix)

    def forward(self, x, word_ids, extra_clusters:List[Tuple[int]]=None) -> Tuple[torch.Tensor,List[Tuple[int,int]],List[torch.Tensor]]:
        """
        x: dict of input ids, attention mask, token type ids, special tokens mask
        extra_clusters: list of clusters to be added to the predicted clusters
        returns: clusters, logits
        clusters: list of clusters as (min,max) spans for each sentence
        logits: logits for each cluster
        """
        final,ls=self._encode(**x)
        clusters,logits=[],[]
        sentence_masks=(x["attention_mask"]-x["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        for i,ls_vectors in enumerate(ls):
            input_ids=x["input_ids"][i]
            sentence_mask=sentence_masks[i]
            token_indices=torch.argwhere(sentence_mask).squeeze(-1)
            token_ls_vectors=ls_vectors[token_indices].detach().cpu().numpy()
            projected_word_ids=word_ids[i][token_indices].detach().cpu().numpy().tolist()
            clustering_model=self._get_clustering_model(projected_word_ids)
            predicted_clusters=compute_clusters(clustering_model.fit(token_ls_vectors),projected_word_ids)
            sentence_clusters=[]
            new_input_ids=[]
            spans_set=set()
            for cluster in predicted_clusters:
                cluster_indices=token_indices[list(cluster)].cpu().numpy()
                min=cluster_indices.min()
                max=cluster_indices.max()
                spans_set.add((min,max))
            if extra_clusters:
                spans_set=spans_set.union(set(extra_clusters[i]))
            for (min,max) in spans_set:
                new_ids=list(input_ids.cpu().numpy()[:min])+\
                    [dm.E_START_ID]+\
                    list(input_ids.cpu().numpy()[min:max+1])+\
                    [dm.E_END_ID]+\
                    list(input_ids.cpu().numpy()[max+1:])
                sentence_clusters.append((min,max))
                new_input_ids.append(new_ids)
            clusters.append(sentence_clusters)
            new_input_ids_t=torch.tensor(new_input_ids).to(self.device)
            encoded_sentences,_=self._encode(input_ids=new_input_ids_t)
            vectors_class_concat=[]
            for (min,max),encoded_sentence in zip(sentence_clusters,encoded_sentences):#TODO optimize with tensor operations
                vectors_class=torch.cat([encoded_sentence[min],encoded_sentence[max]],dim=-1)
                vectors_class_concat.append(vectors_class)

            vectors_class_concat_t=torch.stack(vectors_class_concat,dim=0)
            logit=self.fc_classif(vectors_class_concat_t)
            logits.append(logit)
        return ls,clusters,logits

    def class_criterion(self,logits:torch.Tensor, labels_ohe:torch.Tensor)->torch.Tensor:
        """
        Computes the classification loss
        logits: logits for each cluster
        labels_ohe: labels in ground truth as one hot encoded vector
        weights: weights for each class
        """
        loss=self.loss_fn(logits,labels_ohe) #F.cross_entropy(logits,labels_ohe)
        return loss

    def ls_loss(self, ls_vectors:torch.Tensor, x:Dict) -> torch.Tensor:
        """
        Computes the ltent space loss
        ls: latent space vectors of shape (batch_size,seq_len,ls_hidden_size)
        x: dict from 
        """
        sentence_masks=(x["inputs"]["attention_mask"]-x["inputs"]["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        clusters=x["final_cluster_masks"]
        clusters[clusters<=0]=0
        ls_loss1,distances=hac_sl_ratio_loss(distance_fn=self.distance_fn, vectors=ls_vectors, token_mask=sentence_masks, y=clusters)
        if not ls_loss1:
            return None
        ls_loss2,_ = hac_sl_ratio_loss_token_based(distance_fn=self.distance_fn, vectors=ls_vectors, token_mask=sentence_masks, y=clusters)
        return ls_loss1+ls_loss2

    def compute_losses(self, batch, batch_idx):
        inputs=batch["inputs"]
        y=batch["labels"]*inputs["attention_mask"]
        cluster_masks=batch["final_cluster_masks"]
        all_word_ids=batch["all_word_ids"]
        all_extra_spans=[]
        for i in range(len(cluster_masks)):
            gt_clusters=cluster_masks[i]
            word_ids=all_word_ids[i]
            extra_spans=self.get_extra_spans(gt_clusters,word_ids)
            all_extra_spans.append(extra_spans)
        ls_vectors,clusters,logits=self.forward(inputs,all_word_ids,all_extra_spans)
        ls_loss=self.ls_loss(ls_vectors,batch)
        gt_spans_batch=[]
        gt_classes_ohe_batch=[]
        for gt_clusters,gt_labels in zip(batch["final_cluster_masks"],batch["labels"]):
            gt_spans_sentence=[]
            gt_classes_sentence=[]
            for gt_cluster in gt_clusters:
                indices=torch.argwhere(gt_cluster==1).squeeze(-1)
                if indices.shape[0]==0:
                    continue
                min=torch.min(indices).item()
                max=torch.max(indices).item()
                gt_spans_sentence.append((min,max))
                label_type_idx=self._get_type_idx(gt_labels[min].item())
                classes_tensor_ohe=torch.zeros(len(self.types))
                classes_tensor_ohe[label_type_idx]=1
                gt_classes_sentence.append(classes_tensor_ohe)
            gt_spans_batch.append(gt_spans_sentence)
            gt_classes_ohe_batch.append(gt_classes_sentence)
        ohe_no_entity=torch.zeros(len(self.types))
        ohe_no_entity[self.types.index(self.class_type_mapping["O"])]=1
        losses=[]
        for (sentence_clusters,cluster_logits,gt_spans,gt_classes_ohe) in zip(clusters,logits,gt_spans_batch,gt_classes_ohe_batch):
            loss_logits=[]
            targets=[]
            for (cluster,logit) in zip(sentence_clusters,cluster_logits):
                loss_logits.append(logit)
                if cluster not in gt_spans:
                    targets.append(ohe_no_entity)
                else:
                    gt_class=gt_classes_ohe[gt_spans.index(cluster)]
                    targets.append(gt_class)
                    #losses.append(self.class_criterion(logit,gt_class))
            losses.append(self.class_criterion(torch.stack(loss_logits),torch.stack(targets).to(self.device)))
        class_loss=torch.stack(losses).mean()
        return class_loss,ls_loss

    def training_step(self, batch, batch_idx):
        class_loss,ls_loss=self.compute_losses(batch, batch_idx)
        if ls_loss:
            self.log("losses/train_ls_loss",ls_loss)
        self.log("losses/train_class_loss",class_loss)
        loss=class_loss+ls_loss if ls_loss else class_loss
        self.log("losses/train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        class_loss,ls_loss=self.compute_losses(batch, batch_idx)
        if ls_loss:
            self.log("losses/val_ls_loss",ls_loss)
        self.log("losses/val_class_loss",class_loss)
        loss=class_loss+ls_loss if ls_loss else class_loss
        self.log("losses/val_loss",loss)
        prediction_objs=self.predict(batch)
        predictions=[obj.seq_labels for obj in prediction_objs]
        gt=[]
        labels=batch["labels"]
        sentence_masks=(batch["inputs"]["attention_mask"]-batch["inputs"]["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        for i,(label_tensor,sentence_mask) in enumerate(zip(labels,sentence_masks)):
            gt_sentence=[]
            label_array=label_tensor[sentence_mask==1].tolist()
            for label in label_array:
                gt_sentence.append(self.orig_classes.int2str(label))
            gt.append(gt_sentence)
        res=self.seqeval_metric.compute(predictions=predictions, references=gt)
        val_f1=res["overall_f1"]
        self.log("metrics/val_f1",val_f1)
        for k,v in res.items():
            if type(v)==dict:
                #per class performance
                for k_,v_ in v.items():
                    self.log(f"metrics/val_{k}_{k_}",float(v_))
            else:
                if k!="overall_f1" and type(v)==float or type(v)==int:
                    self.log(f"metrics/val_{k}",float(v))
        return loss
    
    def _predict_logits(self, batch) -> Tuple[List[List[Tuple[int,int]]], List[torch.Tensor]]:
        """
        Adds single word clusters and assigns logits for each cluster
        """
        all_word_ids=batch["all_word_ids"]
        all_extra_spans=[]
        for i in range(len(all_word_ids)):
            word_ids=all_word_ids[i]
            extra_spans=self.get_extra_spans([],word_ids)
            all_extra_spans.append(extra_spans)
        _,clusters,logits=self.forward(batch["inputs"],all_word_ids,all_extra_spans)
        return clusters,logits
    
    def predict(self, batch) -> List[LSHAC_NER_Prediction]:
        clusters,logits = self._predict_logits(batch)
        sentence_masks=(batch["inputs"]["attention_mask"]-batch["inputs"]["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        all_word_ids=batch["all_word_ids"]
        predictions = []
        for (sentence_clusters,cluster_logits,sentence_mask,word_ids_t) in zip(clusters,logits,sentence_masks,all_word_ids):
            word_ids=word_ids_t[sentence_mask==1].tolist()
            predictions.append(LSHAC_NER_Prediction(sentence_clusters,cluster_logits,self.types,sentence_mask,word_ids=word_ids))
        return predictions

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.predict(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_extra_spans(self,clusters,word_ids):
        """returns a list of spans derived from the clusters and individual words"""
        extra_spans=[]
        for cluster in clusters:
            indx=torch.argwhere(cluster).squeeze(-1)
            if indx.shape[0]==0:
                continue
            min=int(indx.min())
            max=int(indx.max())
            extra_spans.append((min,max))
        max_word_id=word_ids.max().item()
        for j in range(max_word_id):
            indices=torch.argwhere(word_ids==j).squeeze(-1)
            if indices.shape[0]==0:
                continue
            min=indices.min().item()
            max=indices.max().item()
            extra_spans.append((min,max))
        return extra_spans
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers import AutoModel
    import data_modules as dm
    from data_modules import HFNer_DataModule
    from datasets import load_dataset
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    dm.include_special_tokens(model,tokenizer)
    data=load_dataset("wnut_17")
    data_module=HFNer_DataModule(data,tokenizer=tokenizer,batch_size=2)
    ner_model=LSHAC_NERModel(model,classes=data_module.class_label_obj,lr=1e-3,ls_hidden_size=128,distance_fn=torch.cdist,hac_metric="euclidean")
    val_data=data_module.val_dataloader()
    batch=next(iter(val_data))
    ner_model.validation_step(batch,0)