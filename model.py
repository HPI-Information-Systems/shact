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
from utils import filter_kwargs
from latent_space import hac_sl_ratio_loss, hac_sl_ratio_loss_token_based
import re

#Pytorch lighning NER model with BERT as the underlying model
class LSHAC_NERModel(pl.LightningModule):
    def __init__(self,transformer_model:BertModel,classes:ClassLabel,lr=1e-3,ls_hidden_size=128,distance_fn:Callable=torch.cdist,affinity=None):
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
        if (not affinity) and distance_fn==torch.cdist:
            affinity="euclidean"
        self.clustering_model=AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='single',distance_threshold=0,affinity=affinity)
        self.loss_fn=nn.CrossEntropyLoss()

    def save_hyperparameters(self,**kwargs):
        kwargs.setdefault("ignore",[]).append("transformer_model")
        super().save_hyperparameters(**kwargs)

    def _encode(self, **x)->Tuple[torch.Tensor,torch.Tensor]:
        #encodes the input x using the transformer model
        hidden_states=self.transformer_model(**filter_kwargs(self.transformer_model.forward,x),output_hidden_states=True).hidden_states
        h=torch.cat(hidden_states,dim=-1)
        ls=self.ls_proj(h)
        final_layer=hidden_states[-1]
        return final_layer,ls

    def _get_types_mapping(self,class_labels:ClassLabel)->Tuple[List[str],Dict[str,str]]:
        regex=re.compile(r"[B,I]-(.*)")
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

    def _get_type_idx(self,class_label:int)->int:
        return self.types.index(self.class_type_mapping[self.orig_classes.names[class_label]])

    def forward(self, x, extra_clusters:List[Tuple[int]]=None):
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
            predicted_clusters=compute_clusters(self.clustering_model.fit(token_ls_vectors))
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
        ls_vectors,clusters,logits=self.forward(inputs,all_extra_spans)
        ls_loss=self.ls_loss(ls_vectors,batch)
        gt_spans=[]
        gt_classes_ohe=[]
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
            gt_spans.append(gt_spans_sentence)
            gt_classes_ohe.append(gt_classes_sentence)
        ohe_no_entity=torch.zeros(len(self.types))
        ohe_no_entity[self.types.index(self.class_type_mapping["O"])]=1
        losses=[]
        for (sentence_clusters,cluster_logits) in zip(clusters,logits):
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
        all_word_ids=batch["all_word_ids"]
        all_extra_spans=[]
        for i in range(len(all_word_ids)):
            word_ids=all_word_ids[i]
            extra_spans=self.get_extra_spans([],word_ids)
            all_extra_spans.append(extra_spans)
        _,clusters,logits=self.forward(batch["inputs"],all_extra_spans)
        all_predicted_spans=[]
        for (sentence_clusters,cluster_logits) in zip(clusters,logits):
            predicted_spans=[]
            for (cluster,logit) in zip(sentence_clusters,cluster_logits):
                predicted_class=torch.argmax(logit)
                predicted_spans.append((cluster,predicted_class))
            all_predicted_spans.append(predicted_spans)
        loss=class_loss+ls_loss if ls_loss else class_loss
        self.log("losses/val_loss",loss)
        return loss


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        all_word_ids=batch["all_word_ids"]
        all_extra_spans=[]
        for i in range(len(all_word_ids)):
            word_ids=all_word_ids[i]
            extra_spans=self.get_extra_spans([],word_ids)
            all_extra_spans.append(extra_spans)
        _,clusters,logits=self.forward(batch["inputs"],all_extra_spans)
        all_predicted_spans=[]
        for (sentence_clusters,cluster_logits) in zip(clusters,logits):
            predicted_spans=[]
            for (cluster,logit) in zip(sentence_clusters,cluster_logits):
                predicted_class=torch.argmax(logit)
                predicted_spans.append((cluster,predicted_class))
            all_predicted_spans.append(predicted_spans)
        return all_predicted_spans

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
    ner_model=LSHAC_NERModel(model,classes=data_module.class_label_obj,lr=1e-3,ls_hidden_size=128,distance_fn=torch.cdist,affinity="euclidean")
    val_data=data_module.val_dataloader()
    batch=next(iter(val_data))
    ner_model.validation_step(batch,0)