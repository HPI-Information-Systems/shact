from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from datasets import ClassLabel
from transformers import BertModel
from sklearn.cluster import AgglomerativeClustering
from clustering_model import compute_clusters
import data_modules as dm
from latent_space import hac_sl_ratio_loss, hac_sl_ratio_loss_token_based
import utils
import evaluate
import random
from pytorch_lightning.loggers.wandb import WandbLogger
from inference_model import LSHAC_NER_Prediction


#Pytorch lighning NER model with BERT as the underlying model
class LSHAC_NERModel(pl.LightningModule):
    def __init__(self, transformer_model: BertModel,
                 classes: ClassLabel,
                 neg_sample_size:int,
                 lr=1e-3,
                 ls_hidden_size=128, 
                 distance_fn: Callable = torch.cdist, 
                 hac_metric=None,):
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
        self.loss_fn=nn.CrossEntropyLoss(reduction="sum")
        self.experiment_id=None
        try:
            self.experiment_id=self.logger.experiment.path
        except:
            #random id
            self.experiment_id=str(np.random.randint(1000000))
        self.seqeval_metric=evaluate.load("seqeval", experiment_id=self.experiment_id)#, zero_division=0)
        self.val_classification=None
        self.neg_sample_size=neg_sample_size
        self.warmup=True

    def save_hyperparameters(self,**kwargs):
        kwargs.setdefault("ignore",[]).append("transformer_model")
        super().save_hyperparameters(**kwargs)

    def _full_encode(self, **x)->Tuple[torch.Tensor,torch.Tensor]:
        #encodes the input x using the transformer model
        hidden_states=self.transformer_model(**utils.filter_kwargs(self.transformer_model.forward,x),output_hidden_states=True).hidden_states
        h=torch.cat(hidden_states,dim=-1)
        ls=self.ls_proj(h)
        final_layer=hidden_states[-1]
        return final_layer,ls
    
    def _encode(self, **x)->torch.Tensor:
        #encodes the input x using the transformer model
        rep=self.transformer_model(**utils.filter_kwargs(self.transformer_model.forward,x),output_hidden_states=False).last_hidden_state
        return rep

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
        """
        returns the clustering model for the given word ids
        It uses the connectivity matrix to avoid clustering words that are not connected
        and to avoid clustering tokens that are not in the same word
        """
        connectivity_matrix=utils.get_connectivity_matrix(word_ids)
        return AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='single',distance_threshold=0,metric=self.hac_metric, connectivity=connectivity_matrix)
    
    def _get_clusters(self, x, ls, word_ids,):
        sentence_masks=(x["attention_mask"]-x["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        all_spans=[]
        for i,ls_vectors in enumerate(ls):
            input_ids=x["input_ids"][i]
            sentence_mask=sentence_masks[i]
            token_indices=torch.argwhere(sentence_mask).squeeze(-1)
            token_ls_vectors=ls_vectors[token_indices].detach().cpu().numpy()
            word_ids_list=word_ids[i].detach().cpu().numpy().tolist()
            projected_word_ids=word_ids[i][token_indices].detach().cpu().numpy().tolist()
            clustering_model=self._get_clustering_model(projected_word_ids)
            
            predicted_clusters=[{0}]
            if len(token_ls_vectors)>1:
                predicted_clusters=compute_clusters(clustering_model.fit(token_ls_vectors),word_ids_list)
            spans_set=set()
            for cluster in predicted_clusters:
                cluster_indices=token_indices[list(cluster)].cpu().numpy()
                min=cluster_indices.min()
                max=cluster_indices.max()
                spans_set.add((min,max))
            all_spans.append(spans_set)
        return all_spans

    #refactor forward ands split into 2 functions clustering and classifications
    def _fw_clusters(self, x, ) -> torch.Tensor:
        """
        x: dict of input ids, attention mask, token type ids, special tokens mask
        returns: latent space vectors, clusters
        latent space vectors: Tensor of shape (batch_size,seq_length,ls_hidden_size)
        clusters: list of clusters as (min,max) spans for each sentence
        """
        _,ls=self._full_encode(**x)
        return ls
    
    def _fw_classify(self, x, clusters:List[Tuple[int,int]]) -> List[torch.Tensor]:
        """
        x: dict of input ids, attention mask, token type ids, special tokens mask
        clusters: list of clusters as (min,max) spans, one for each sentence
        returns: list of logit tensors for each clusters
        """
        #logits=[]
        #batch_size=x["input_ids"].shape[0]
        new_input_ids=[]
        for i,(input_ids,(min,max)) in enumerate(zip(x["input_ids"],clusters)):
            new_ids=list(input_ids.cpu().numpy()[:min])+\
                [dm.E_START_ID]+\
                list(input_ids.cpu().numpy()[min:max+1])+\
                [dm.E_END_ID]+\
                list(input_ids.cpu().numpy()[max+1:])
            new_input_ids.append(new_ids)
        new_input_ids_t=torch.tensor(new_input_ids).to(self.device)
        attention_mask_t=torch.where(new_input_ids_t!=0,1,0).to(self.device)
        batched_input_ids=new_input_ids_t#torch.split(new_input_ids_t,batch_size,dim=0)
        batched_attention_mask=attention_mask_t#torch.split(attention_mask_t,batch_size,dim=0)
        encoded_sentences=self._encode(input_ids=batched_input_ids,attention_mask=batched_attention_mask)
        vectors_class_concat=[]
        for (min,max),encoded_sentence in zip(clusters,encoded_sentences):#TODO optimize with tensor operations
            vectors_class=torch.cat([encoded_sentence[min],encoded_sentence[max]],dim=-1)
            vectors_class_concat.append(vectors_class)
        vectors_class_concat_t=torch.stack(vectors_class_concat,dim=0)
        logits=self.fc_classif(vectors_class_concat_t)
        return logits

    def forward(self, x, clusters:List[Tuple[int,int]]):
        """
        x: dict of input ids, attention mask, token type ids, special tokens mask
        clusters: list of clusters as (min,max) spans for each sentence
        returns: latent space vectors, clusters, logits
        latent space vectors: Tensor of shape (batch_size,seq_length,ls_hidden_size)
        clusters: list of clusters as (min,max) spans for each sentence
        logits: logits for each cluster
        """
        ls=self._fw_clusters(x)
        logits=None
        if not self.warmup:
            logits=self._fw_classify(x,clusters)
        return ls,clusters,logits

    def _forward_senteces(self, x, word_ids) -> Tuple[torch.Tensor,List[Tuple[int,int]],List[torch.Tensor]]:
        """
        x: dict of input ids, attention mask, token type ids, special tokens mask
        returns: latent space vectors, clusters, logits
        latent space vectors: Tensor of shape (batch_size,seq_length,ls_hidden_size)
        clusters: list of clusters as (min,max) spans for each sentence
        logits: logits for each cluster
        """
        ls=self._fw_clusters(x)
        predicted_clusters=self._get_clusters(x,ls,word_ids)
        #prediction. classify all predicted clusters
        #dict with same keys as x but with values as empty lists
        batch_size=x["input_ids"].shape[0]
        all_logits=[]
        #if not self.warmup:
        for i,pred_clusters_sentece in enumerate(predicted_clusters):
            pred_clusters_sentece=list(pred_clusters_sentece)
            #partition predicted clusters into batches
            sentence_batches=[pred_clusters_sentece[i:i+batch_size] for i in range(0,len(pred_clusters_sentece),batch_size)]
            sentence_logits=[]
            for cluster_batch in sentence_batches:
                inputs={}
                for key in x.keys():
                    inputs[key]=[]
                for _ in cluster_batch:
                    for key in x.keys():
                        inputs[key].append(x[key][i])
                for key in x.keys():
                    inputs[key]=torch.stack(inputs[key],dim=0)
                logits=self._fw_classify(inputs,cluster_batch)
                sentence_logits.extend(logits)
            all_logits.append(sentence_logits)
        return ls,predicted_clusters,all_logits

    def class_criterion(self,logits:torch.Tensor, labels_ohe:torch.Tensor)->torch.Tensor:
        """
        Computes the classification loss
        logits: logits for each cluster
        labels_ohe: labels in ground truth as one hot encoded vector
        weights: weights for each class
        """
        #if labels are list, convert to tensor
        if isinstance(labels_ohe,list):
            labels_ohe=torch.tensor(labels_ohe).to(self.device)
        loss=self.loss_fn(logits,labels_ohe) #F.cross_entropy(logits,labels_ohe)
        return loss

    def ls_loss(self, ls_vectors:torch.Tensor, x:Dict, types:torch.Tensor) -> torch.Tensor:
        """
        Computes the ltent space loss
        ls: latent space vectors of shape (batch_size,seq_len,ls_hidden_size)
        x: dict from 
        """
        if isinstance(types,list):
            types=torch.tensor(types).to(self.device)
        sentence_masks=(x["inputs"]["attention_mask"]-x["inputs"]["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        clusters=x["final_cluster_masks"]
        clusters[clusters<=0]=0
        ls_vectors_filter=ls_vectors
        clusters_filter=clusters
        sentence_masks_filter=sentence_masks
        if not self.warmup:
            O_type_idx=self._get_type_idx(self.types.index("O"))
            type_filter=types!=O_type_idx # filter out O labels
            ls_vectors_filter=ls_vectors[type_filter]
            clusters_filter=clusters[type_filter]
            sentence_masks_filter=sentence_masks[type_filter]
        assert ls_vectors_filter.shape[0]==clusters_filter.shape[0]
        assert ls_vectors_filter.shape[0]==sentence_masks_filter.shape[0]
        if ls_vectors_filter.shape[0]==0:
            return None
        ls_loss1,distances=hac_sl_ratio_loss(distance_fn=self.distance_fn, vectors=ls_vectors_filter, token_mask=sentence_masks_filter, y=clusters_filter)
        if not ls_loss1:
            return None
        return ls_loss1
        #ls_loss2,_ = hac_sl_ratio_loss_token_based(distance_fn=self.distance_fn, vectors=ls_vectors, token_mask=sentence_masks, y=clusters)
        #return ls_loss1+ls_loss2
    
    def _get_entities_as_spans(self,clusters:torch.Tensor,labels:torch.Tensor) -> List[List[Tuple[Tuple[int,int],int,torch.Tensor]]]:
        """
        Converts clusters as masks and labels as IOB numeric labels to a list of spans
        Each span is a tuple of (min,max) indices, the type index and the one hot encoded type vector
        clusters: tensor of shape (batch_size,max clusters,seq_len)
        labels: tensor of shape (batch_size,seq_len)
        returns: list of list of spans
        """
        gt_spans_batch=[]
        for gt_clusters,gt_labels in zip(clusters,labels):
            gt_spans_sentence=[]
            for gt_cluster in gt_clusters:
                indices=torch.argwhere(gt_cluster==1).squeeze(-1)
                if indices.shape[0]==0:
                    continue
                min=torch.min(indices).item()
                max=torch.max(indices).item()
                label_type_idx=self._get_type_idx(gt_labels[min].item())
                classes_tensor_ohe=torch.zeros(len(self.types))
                classes_tensor_ohe[label_type_idx]=1
                gt_spans_sentence.append(((min,max),label_type_idx,classes_tensor_ohe))
            gt_spans_batch.append(gt_spans_sentence)
        return gt_spans_batch        

    def compute_losses(self, batch, batch_idx):
        inputs=batch["inputs"]
        types=batch["types"]
        #y=batch["labels"]*inputs["attention_mask"]
        cluster_masks=batch["final_cluster_masks"]
        cluster_masks[cluster_masks<=0]=0
        cluster_spans=self.get_extra_spans(cluster_masks)
        ls_vectors,clusters,logits=self.forward(inputs,cluster_spans)
        #only compute ls_loss for entities
        ls_loss=self.ls_loss(ls_vectors,batch,types)
        type_idxs=[self._get_type_idx(type) for type in types] if types else None
        class_loss=self.class_criterion(logits,type_idxs) if not self.warmup else None
        return class_loss,ls_loss

    def training_step(self, batch, batch_idx):
        class_loss,ls_loss=self.compute_losses(batch, batch_idx)
        loss=torch.tensor(0.0).to(self.device)
        if ls_loss:
            self.log("losses/train_ls_loss",ls_loss)
            loss+=ls_loss
        if class_loss:
            self.log("losses/train_class_loss",class_loss)
            loss+=class_loss
        self.log("losses/train_loss",loss)
        return loss
    
    def compute_labels(self, batch, prediction_objs:List[LSHAC_NER_Prediction]) -> Tuple[List[List[str]],List[List[str]]]:
        """
        Computes the predicted and ground truth labels in the IOB format
        batch: batch of data. Used for getting the ground truth labels
        prediction_objs: list of prediction objects
        returns: tuple of predicted labels and ground truth labels
        """
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
        return predictions,gt
    
    def _test_batch(self, batch, prediction_objs):
        predictions,gt=self.compute_labels(batch, prediction_objs)
        res=self.seqeval_metric.compute(predictions=predictions, references=gt, zero_division=0)
        return predictions,gt,res

    def validation_step(self, batch, batch_idx):


        # class_loss,ls_loss=self.compute_losses(batch, batch_idx)
        # if ls_loss:
        #     self.log("losses/val_ls_loss",ls_loss)
        # self.log("losses/val_class_loss",class_loss)
        # loss=class_loss+ls_loss if ls_loss else class_loss
        # self.log("losses/val_loss",loss)

        prediction_objs,_=self.predict(batch)

        prediction_labels,gt_labels,res=self._test_batch(batch,prediction_objs)
        potential_recall=utils.get_potential_recall(clusters=[obj.clusters for obj in prediction_objs],batch=batch)
        for k,v in potential_recall.items():
            class_name=self.class_type_mapping[self.orig_classes.int2str(k)]           
            self.log(f"metrics/val_{class_name}_potential_recall",v)
        
        #if using confusion matrix
        if self.val_classification is not None:
            gt_spans=self._get_entities_as_spans(batch["final_cluster_masks"],batch["labels"])
            flatten_gt_types=[]
            flatten_predicted_types=[]
            for gt_spans_sentence,pred_obj in zip(gt_spans,prediction_objs):
                gt_dict={}
                for (gt_start,gt_end),gt_type,_ in gt_spans_sentence:
                    gt_dict[(gt_start,gt_end)]=gt_type
                predicted_spans=set()
                for (pred_start,pred_end),pred_type in pred_obj.all_predictions():
                    if (pred_start,pred_end) in gt_dict:
                        gt_type=gt_dict[(pred_start,pred_end)]
                        flatten_gt_types.append(gt_type)
                        flatten_predicted_types.append(pred_type)
                    else:
                        flatten_gt_types.append(self.types.index("O"))
                        flatten_predicted_types.append(pred_type)
                    predicted_spans.add((pred_start,pred_end))
                for (gt_start,gt_end),gt_type in gt_dict.items():
                    if (gt_start,gt_end) not in [x[0] for x in pred_obj.assignments]:
                        flatten_gt_types.append(gt_type)
                        flatten_predicted_types.append(self.types.index("O"))
            
            self.val_classification["predicted"].extend(flatten_predicted_types)
            self.val_classification["gt"].extend(flatten_gt_types)
            
        return prediction_objs,prediction_labels,gt_labels
    
    def validation_epoch_end(self, outputs):
        self.log_confusion_matrix()
        self.val_pred_labels=[]
        self.val_gt_labels=[]
        for (prediction_objs,prediction_labels,gt_labels) in outputs:
            self.val_pred_labels.extend(prediction_labels)
            self.val_gt_labels.extend(gt_labels)
        res=self.seqeval_metric.compute(predictions=self.val_pred_labels, references=self.val_gt_labels, zero_division=0)
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
        self.start_confusion_matrix()
        del self.val_pred_labels
        del self.val_gt_labels
    
    def test_step(self, batch, batch_idx, log_trees=True):
        prediction_objs,_=self.predict(batch)
        pred,gt,res=self._test_batch(batch,prediction_objs)
        return res,prediction_objs,pred,gt
    
    def test_epoch_end(self, outputs):
        all_predictions=[]
        all_gt=[]
        for res,prediction_objs,pred,gt in outputs:
            all_predictions.extend(pred)
            all_gt.extend(gt)
        
        res=self.seqeval_metric.compute(predictions=all_predictions, references=all_gt, zero_division=0)
        self.log("metrics/test_f1",res["overall_f1"])
    
    def start_confusion_matrix(self):
        self.val_classification={
            "predicted":[],
            "gt":[]
        }

    def log_confusion_matrix(self):
        if type(self.logger)==WandbLogger:
            if self.trainer.state.stage!="sanity_check":
                import wandb.plot
                wandb.log({"confusion_matrix":wandb.plot.confusion_matrix(probs=None, y_true=self.val_classification["gt"], preds=self.val_classification["predicted"], class_names=self.types)})
    
    def forward_sentences(self, batch) -> Tuple[List[List[Tuple[int,int]]], List[torch.Tensor]]:
        """
        Adds single word clusters and assigns logits for each cluster
        """
        all_word_ids=batch["all_word_ids"]
        _,clusters,logits=self._forward_senteces(batch["inputs"],all_word_ids)
        return clusters,logits
    
    def predict(self, batch) -> Tuple[List[LSHAC_NER_Prediction], Any]:
        """
        
        """
        clusters,logits = self.forward_sentences(batch)
        sentence_masks=(batch["inputs"]["attention_mask"]-batch["inputs"]["special_tokens_mask"])
        sentence_masks[sentence_masks<=0]=0
        all_word_ids=batch["all_word_ids"]
        predictions = []
        for (sentence_clusters,cluster_logits,sentence_mask,word_ids_t) in zip(clusters,logits,sentence_masks,all_word_ids):
            word_ids=word_ids_t[sentence_mask==1].tolist()
            predictions.append(LSHAC_NER_Prediction(sentence_clusters,cluster_logits,self.types,sentence_mask,word_ids=word_ids))
        ret_batch=batch
        if self.warmup:
            #return ids as a list
            ret_batch={"ids":batch["ids"].cpu().numpy().tolist()}
        return predictions, ret_batch

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.predict(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_extra_spans(self,clusters:torch.Tensor) -> List[Tuple[int,int]]:
        """returns a list of spans derived from the clusters masks"""
        extra_spans=[]
        for cluster in clusters:
            indx=torch.argwhere(cluster).squeeze(-1)
            if indx.shape[0]==0:
                continue
            min=int(indx.min())
            max=int(indx.max())
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