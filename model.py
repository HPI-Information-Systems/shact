from typing import Any, Callable, List, Optional, Tuple
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from datasets import load_metric
from transformers import BertModel
from sklearn.cluster import AgglomerativeClustering
from clustering_model import compute_clusters
import data_modules as dm
from utils import filter_kwargs

#Pytorch lighning NER model with BERT as the underlying model
class NERModel(pl.LightningModule):
    def __init__(self,transformer_model:BertModel,types:List[str],lr=1e-3,ls_hidden_size=128,distance_fn:Callable=torch.cdist,affinity=None):
        super().__init__()
        self.transformer_model=transformer_model
        self.types=types
        self.ls_hidden_size=ls_hidden_size
        self.fc_classif = nn.Linear(transformer_model.config.hidden_size*2, len(types)+1) # last one for not entities
        full_hidden_size=transformer_model.config.hidden_size*(transformer_model.config.num_hidden_layers+1)
        self.ls_proj=nn.Linear(full_hidden_size,ls_hidden_size)
        self.lr=lr
        if (not affinity) and distance_fn==torch.cdist:
            affinity="euclidean"
        self.clustering_model=AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='single',distance_threshold=0,affinity=affinity)

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

    def forward(self, x, extra_clusters:List[List[Tuple]]=None):
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
            #sentence_mask=x["attention_mask"][i]*\
            #    (torch.where(x["special_tokens_mask"][i] == 1, 0, 1))
            sentence_mask=sentence_masks[i]
            token_indices=torch.argwhere(sentence_mask).squeeze()
            token_ls_vectors=ls_vectors[token_indices].detach().cpu().numpy()
            predicted_clusters=compute_clusters(self.clustering_model.fit(token_ls_vectors))
            all_clusters=predicted_clusters
            if extra_clusters:
                all_clusters.extend(extra_clusters[i])
            sentence_clusters=[]
            new_input_ids=[]
            spans_set=set()
            for cluster in all_clusters:
                cluster_indices=token_indices[list(cluster)].cpu().numpy()
                min=cluster_indices.min()
                max=cluster_indices.max()
                spans_set.add((min,max))
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
        return clusters,logits

    def training_step(self, batch, batch_idx):
        inputs=batch["inputs"]
        y=batch["labels"]*inputs["attention_mask"]
        h=self.forward(inputs)
        sentence_mask=inputs["attention_mask"].bool()
        if "special_tokens_mask" in batch:#if the batch has special tokens
            sentence_mask=(inputs["attention_mask"]-batch["special_tokens_mask"]).bool()
            if (batch["special_tokens_mask"][:, 0]).bool().all():
                h=h[:, 1:]
                sentence_mask=sentence_mask[:, 1:]
                #y=y[:, 1:]
        assert(h.shape[1]==sentence_mask.shape[1])
        assert(h.shape[1]==y.shape[1])
        crf_loss=-self.crf(h,y,sentence_mask).sum()
        self.log("train_loss",crf_loss)
        return crf_loss

    def validation_step(self, batch, batch_idx):
        inputs=batch["inputs"]
        y=batch["labels"]
        h=self.forward(inputs)
        sentence_mask=inputs["attention_mask"].bool()
        if "special_tokens_mask" in batch:#if the batch has special tokens
            sentence_mask=(inputs["attention_mask"]-batch["special_tokens_mask"]).bool()
            if (batch["special_tokens_mask"][:, 0]).bool().all():
                h=h[:, 1:]
                sentence_mask=sentence_mask[:, 1:]
        res=self.crf.viterbi_tags(h,sentence_mask)
        y_hat=[y_hat_i for y_hat_i,_ in res]
        predictions=[]
        ground_truth=[]
        for (y_i,y_hat_i) in zip(y,y_hat):
            y_i=y_i[y_i!=-1].cpu().numpy()
            assert(len(y_i)==len(y_hat_i))
            predictions.append(self.int2str_fn(y_hat_i))
            ground_truth.append(self.int2str_fn(y_i))
        res=self.seqeval_metric.compute(predictions=predictions,references=ground_truth)
        val_loss=res["overall_f1"]
        self.log("validation_f1",val_loss)
        for k,v in res.items():
            if type(v)==dict:
                #per class performance
                for k_,v_ in v.items():
                    self.log(f"val_{k}_{k_}",float(v_))
        return val_loss


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inputs=batch["inputs"]
        y=batch["labels"]
        h=self.forward(inputs)
        sentence_mask=inputs["attention_mask"].bool()
        if "special_tokens_mask" in batch:#if the batch has special tokens
            sentence_mask=(inputs["attention_mask"]-batch["special_tokens_mask"]).bool()
            if (batch["special_tokens_mask"][:, 0]).bool().all():
                h=h[:, 1:]
                sentence_mask=sentence_mask[:, 1:]
        y_hat=self.crf.viterbi_tags(h,sentence_mask)
        predictions=[]
        ground_truth=[]
        for (y_i,(y_hat_i,crf_score)) in zip(y,y_hat):
            y_i=y_i[y_i!=-1].cpu().numpy()
            assert(len(y_i)==len(y_hat_i))
            predictions.append(self.int2str_fn(y_hat_i))
            ground_truth.append(self.int2str_fn(y_i))
        #res=self.seqeval_metric.compute(predictions=predictions,references=ground_truth)
        return predictions,ground_truth
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

#Pytorch lighning NER model with BERT as the underlying model and latent space embedding
class NERModelLatent(NERModel):
    def __init__(self,num_labels,int2str_fn,latent_structures_model,lr=1e-3):
        super().__init__(latent_structures_model.pl_model,num_labels,int2str_fn,lr)
        self.latent_model=latent_structures_model
        self.fc = nn.Linear(self.transformer_model.config.hidden_size+self.latent_model.latent_space_size, num_labels)

    # def save_hyperparameters(self,**kwargs):
    #     kwargs.setdefault("ignore",[]).append("latent_model")
    #     super().save_hyperparameters(**kwargs)

    #override the encode method to include the latent space embedding
    def _encode(self, **x):
        """encodes the input x using the transformer model and latent space embedding
        then concatenates the two"""
        h=super()._encode(**x)
        latent_h=self.latent_model(x)
        latent_h=F.normalize(latent_h,dim=-1).detach()
        return torch.cat([h,latent_h],dim=-1)#(batch_size,seq_len,hidden_size_transformer+hidden_size_latent)

    def training_step(self, batch, batch_idx):
        l1=super().training_step(batch,batch_idx)
        l2=self.latent_model.fw_train(batch,batch_idx)
        return l1+l2





