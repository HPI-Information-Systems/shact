import inspect
from typing import Dict, List, Optional, Tuple
import typing
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torch.nn import ConstantPad1d
from datasets import ClassLabel, load_dataset
import datasets as hf_datasets
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import random
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import json
from transformers import BertModel
import utils
from typing import Generator, List, Optional, Union, Callable

E_START="[E_START]"
E_END="[E_END]"
E_START_ID=None
E_END_ID=None
class HFNestedSpanDataset(Dataset):
    """
    A class building sentences like Dataset froma nested span dataset based on span start and end offsets
    """
    def __init__(self, hf_examples:hf_datasets.arrow_dataset.Dataset,tokenizer:Tokenizer,feature_name:str,span_generator_fn:Optional[Callable[[Union[List[str],int]],Generator[Tuple[int,int],None,None]]],limit_samples:int=None) -> None:
        raw_data=hf_examples
        self.feature_name=feature_name
        self.tokenizer=tokenizer
        self.limit_samples=limit_samples
        self.sentences=self._broadcast_sentences_spans(raw_data,span_generator_fn)
    
    def _broadcast_sentences_spans(self, raw_data, span_generator_fn):
        """
        For each sentence in the list, create multiple sentences
        each containing only one entity or no entity
        """
        new_sentences=[]
        self.types=raw_data.features[self.feature_name][0]["label"]
        for raw_sentence in tqdm(raw_data,desc="Broadcasting sentences"):
            new_sentence=[]
            entities=raw_sentence[self.feature_name]
            #find spans of entities in the form o f min and max index
            spans=[]
            for ii,entity in enumerate(entities):
                s_idx=entity["start"]
                e_idx=entity["end"]
                label=entity["label"]
                s=(s_idx,e_idx-1)
                new_sentence.append((raw_sentence["tokens"],s,label))
                spans.append(s)
            by_id=False
            if span_generator_fn:
                try:
                    gen_spans=span_generator_fn(raw_sentence["tokens"])
                    while (self.limit_samples is None) or (len(new_sentence)-len(spans))<self.limit_samples:
                        next_span=next(gen_spans)
                        if next_span not in spans:
                            new_sentence.append((raw_sentence["tokens"],next_span,0))
                except StopIteration:
                    pass
                    
            new_sentences.extend(new_sentence)
        return new_sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]
    
    def collate_fn(self,batch):
        all_words=[]
        all_types=[]
        all_spans=[]
        all_word_ids=[]
        all_masks=[]
        for (tokens,span,type) in batch:
            all_words.append(tokens)
            all_spans.append(span)
            all_types.append(type)
        inputs=self.tokenizer(all_words,return_tensors="pt",is_split_into_words=True,padding=True,return_attention_mask=True,add_special_tokens=False,return_special_tokens_mask=True)
        for ii,(min,max) in enumerate(all_spans):
            words_ids=inputs.word_ids(ii)
            words_ids_pad=[-1 if x is None else x for x in words_ids]
            all_word_ids.append(words_ids_pad)
            mask=[1 if x>=min and x<=max else 0 for x in words_ids_pad]
            all_masks.append(mask)
        return {"inputs":inputs,
                "types":all_types,
                "final_cluster_masks":torch.tensor(all_masks,dtype=torch.long,device=inputs.input_ids.device),
                "all_word_ids":torch.tensor(all_word_ids,dtype=torch.long,device=inputs.input_ids.device)}

class HFNestedSentenceDataset(Dataset):
    def __init__(self, hf_examples:hf_datasets.arrow_dataset.Dataset, tokenizer:Tokenizer,feature_name:str="entities") -> None:
        self.raw_data=hf_examples
        self.feature_name=feature_name
        self.tokenizer=tokenizer
        self.types:ClassLabel=hf_examples.features[feature_name][0]["label"]

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return (idx,self.raw_data[idx])
    
    def get_by_id(self,id):
        return self.__getitem__(id)[1]
    
    def collate_fn(self,batch):
        all_words=[]
        all_ids=[i[0] for i in batch]
        batch_data=[i[1] for i in batch]
        all_entity_w_spans=[]
        for example in batch_data:
            all_words.append(example["tokens"])
            entities=example[self.feature_name]
            entities_w_spans=[]
            for entity in entities:
                type=entity["label"]
                span_s=entity["start"]
                span_e=entity["end"]
                entities_w_spans.append((span_s,span_e-1,type))
            all_entity_w_spans.append(entities_w_spans)

        inputs=self.tokenizer(all_words,return_tensors="pt",is_split_into_words=True,padding=True,return_attention_mask=True,add_special_tokens=False,return_special_tokens_mask=True)
        seq_len=inputs.input_ids.shape[1]

        all_masks=[]
        all_word_ids=[]
        max_entites=0
        for ii,w_spans in enumerate(all_entity_w_spans):
            words_ids=inputs.word_ids(ii)
            words_ids_pad=[-1 if x is None else x for x in words_ids]
            all_word_ids.append(words_ids_pad)
            max_entites=max(max_entites,len(w_spans))
            s_entity_masks=[]
            for (min_i,max_i,type) in w_spans:
                mask=[1 if x>=min_i and x<=max_i else 0 for x in words_ids_pad]
                s_entity_masks.append(mask)
            all_masks.append(s_entity_masks)
        padded_masks=torch.zeros((len(batch_data),max_entites,seq_len),dtype=torch.long,device=inputs.input_ids.device)
        #start padded tags with -100
        padded_types=torch.full((len(batch_data),max_entites),-100,dtype=torch.long,device=inputs.input_ids.device)
        all_word_ids=torch.tensor(all_word_ids,dtype=torch.long,device=inputs.input_ids.device)
        all_ids=torch.tensor(all_ids,dtype=torch.long,device=inputs.input_ids.device)
        for ii,s_entity_masks in enumerate(all_masks):
            for jj,mask in enumerate(s_entity_masks):
                padded_masks[ii,jj,:]=torch.tensor(mask,dtype=torch.long,device=inputs.input_ids.device)
                padded_types[ii,jj]=all_entity_w_spans[ii][jj][2]
        
        return {"ids":all_ids,
                "inputs":inputs,
                "types":padded_types,
                "final_cluster_masks":padded_masks,
                "all_word_ids":all_word_ids}
    
class HFNested_DataModule(pl.LightningDataModule):
    def __init__(self,hf_dataset,tokenizer:Tokenizer,batch_size=32,num_workers=None,undersample_rate=None,span_sampler_fn:Optional[Callable[[Union[List[str],int]],Generator[Tuple[int,int],None,None]]]=None,limit_samples:int=None, test_batch_size:int=None):
        super().__init__()
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        self.test_batch_size=test_batch_size if test_batch_size else batch_size
        self.train_data, self.val_data, self.test_data = hf_dataset["train"], hf_dataset["validation"], hf_dataset.get("test",None)
        self.span_sampler_fn=span_sampler_fn
        self.tokenizer = tokenizer
        self.num_workers=num_workers
        self.feature_name="spans"
        if self.num_workers is None or self.num_workers==0:
            self.num_workers=os.cpu_count()
        self.int2str=dict()
        if undersample_rate:
            self.undersample_rate=undersample_rate
            indices_sample=random.sample(list(range(len(self.train_data))),round(undersample_rate*len(self.train_data)))
            self.train_data=self.train_data.select(indices_sample)
        self.limit_samples=limit_samples
        self.dl_train,int2str=self._get_train_loader(self.train_data,self.batch_size)
        self.int2str["train"]=int2str
        self.class_label_obj=self.train_data.features[self.feature_name][0]["label"] #ClassLabel(names=int2str)
        self.num_classes=len(self.class_label_obj.names)

    def resample_train_dataloader(self, span_sampler_fn:Optional[Callable[[List[str]],Generator[Tuple[int,int],None,None]]]=None) -> DataLoader:
        """
        Re creates the train dataloader with the new span sampler
        """
        self.span_sampler_fn=span_sampler_fn
        self.dl_train,_=self._get_train_loader(self.train_data,self.batch_size)
        return self.dl_train
    
    def get_train_dataloder_for_eval(self) -> DataLoader:
        """
        Builds a dataloader for the training set that can be used for evaluation.
        Is is useful for inference on the training set after warmup.
        """
        dl,_=self._get_test_loader(self.train_data,self.test_batch_size)
        return dl

    def train_dataloader(self):
        # dl,int2str=self._getloader(self.train_data,self.batch_size)
        # self.int2str["train"]=int2str
        return self.dl_train

    def val_dataloader(self):
        dl,int2str=self._get_test_loader(self.val_data,self.test_batch_size)
        self.int2str["val"]=int2str
        return dl

    def test_dataloader(self):
        dl,int2str=self._get_test_loader(self.test_data,self.test_batch_size)
        self.int2str["test"]=int2str
        return dl

    def _get_test_loader(self,data,batch_size):
        ds=HFNestedSentenceDataset(data,self.tokenizer,feature_name=self.feature_name)
        return DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=self.num_workers),self.int2str["train"]

    def _get_train_loader(self,data,batch_size):
        ds=HFNestedSpanDataset(data,self.tokenizer, feature_name=self.feature_name, span_generator_fn=self.span_sampler_fn,limit_samples=self.limit_samples)
        int2str=ds.types.int2str
        return DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=self.num_workers, shuffle=True),int2str


def include_special_tokens(model:BertModel,tokenizer:Tokenizer):
    global E_START_ID,E_END_ID
    tokenizer.add_special_tokens({"additional_special_tokens":[E_START,E_END]})
    E_START_ID=tokenizer.convert_tokens_to_ids(E_START)
    E_END_ID=tokenizer.convert_tokens_to_ids(E_END)
    model.resize_token_embeddings(len(tokenizer))