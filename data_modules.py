import inspect
from typing import Dict, List, Optional, Tuple
import typing
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torch.nn import ConstantPad1d
from datasets import ClassLabel, load_dataset
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

def get_tag_format(hf_dataset, feature_name="ner_tags"):
    is_iob=all([n.startswith("B-") or n.startswith("I-") or n=="O" for n in hf_dataset["train"].features[feature_name].feature.names])
    is_iobes=all([n.startswith("B-") or n.startswith("I-") or n.startswith("E-") or n.startswith("S-") or n=="O" for n in hf_dataset["train"].features[feature_name].feature.names])
    if is_iob:
        return "IOB"
    elif is_iobes:
        return "IOBES"
    else:
        return "IO"

class HFNer_DataModule(pl.LightningDataModule):
    def __init__(self,hf_dataset,tokenizer:Tokenizer,batch_size=32,num_workers=None,tag_format="IOB",undersample_rate=None,feature_name="ner_tags",span_sampler_fn:Optional[Callable[[Union[List[str],int]],Generator[Tuple[int,int],None,None]]]=None):
        super().__init__()
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        self.train_data, self.val_data, self.test_data = hf_dataset["train"], hf_dataset["validation"], hf_dataset.get("test",None)

        self.orig_tag_format=get_tag_format(hf_dataset,feature_name)
        self.span_sampler_fn=span_sampler_fn
        self.tokenizer = tokenizer
        self.num_workers=num_workers
        self.feature_name=feature_name
        if self.num_workers is None or self.num_workers==0:
            self.num_workers=os.cpu_count()
        self.tag_format=tag_format
        self.int2str=dict()
        if undersample_rate:
            self.undersample_rate=undersample_rate
            indices_sample=random.sample(list(range(len(self.train_data))),round(undersample_rate*len(self.train_data)))
            self.train_data=self.train_data.select(indices_sample)
        self.dl_train,int2str=self._get_train_loader(self.train_data,self.batch_size)
        self.int2str["train"]=int2str
        self.class_label_obj=self.dl_train.dataset.class_label_obj
        self.num_classes=self.dl_train.dataset.class_label_obj.num_classes

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
        dl,_=self._get_test_loader(self.train_data,self.batch_size)
        return dl

    def train_dataloader(self):
        # dl,int2str=self._getloader(self.train_data,self.batch_size)
        # self.int2str["train"]=int2str
        return self.dl_train

    def val_dataloader(self):
        dl,int2str=self._get_test_loader(self.val_data,self.batch_size)
        self.int2str["val"]=int2str
        return dl

    def test_dataloader(self):
        dl,int2str=self._get_test_loader(self.test_data,self.batch_size)
        self.int2str["test"]=int2str
        return dl

    def estimate_type_frequency(self):
        #estimate the frequency of each type from the training set
        type_freq=dict()
        for sentence in self.train_data:
            for tag in sentence[self.feature_name]:
                tag_str=self.class_label_obj.int2str(tag)
                regex=utils.regex_extract_type
                match=regex.match(tag_str)
                type=tag_str
                if match:
                    type=match.group(1)
                if type not in type_freq:
                    type_freq[type]=1
                else:
                    type_freq[type]+=1
        return type_freq

    def _get_test_loader(self,data,batch_size):
        split_class_label_obj=data.features[self.feature_name].feature
        ds=HFNerIOBDataset(data,self.tokenizer,class_label_obj=split_class_label_obj, feature_name=self.feature_name, tag_format=self.orig_tag_format)
        int2str=ds.class_label_obj.int2str
        return DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=self.num_workers),int2str

    def _get_train_loader(self,data,batch_size):
        split_class_label_obj=data.features[self.feature_name].feature
        ds=HFNerSpanDataset(data,self.tokenizer,class_label_obj=split_class_label_obj, feature_name=self.feature_name, span_generator_fn=self.span_sampler_fn, tag_format=self.orig_tag_format)
        int2str=ds.class_label_obj.int2str
        return DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=self.num_workers, shuffle=True),int2str

# class HFNerDatum():
#     """
#     One element of the training set
#     """
#     def __inint__(self, tokens, span, label):#, all_labels):
#         self.tokens=tokens
#         self.span=span
#         self.label=label
#         #self.all_labels=all_labels

class HFNerDataset(Dataset):
    def __init__(self,hf_examples,tokenizer:Tokenizer,class_label_obj:ClassLabel,feature_name, tag_format):
        super().__init__()
        self.feature_name=feature_name
        self.tokenizer=tokenizer
        self.class_label_obj=class_label_obj
        if tag_format!="IOB":
            if tag_format=="IO":
                self.raw_data=[self._convert_examplo_io_to_iob(example) for example in hf_examples]
                io_names=class_label_obj.names.copy()
                b_names=[]
                for ii,io_name in enumerate(io_names):
                    if not io_name.startswith("I-") and io_name!="O":
                        io_names[ii]="I-"+io_name
                    if io_names[ii].startswith("I-"):
                        b_names.append("B-"+io_names[ii][2:])
                names=io_names+b_names
                self.class_label_obj=ClassLabel(names=names)
        self._build_b_i_dict()
        #remove empty sentences and sentences with only one token.
        self.raw_data=[sentence for sentence in hf_examples if (len(sentence["tokens"])>0)]
        self.id_to_idx={example["id"]:ii for ii,example in tqdm(enumerate(self.raw_data),desc="building id_to_idx")}
        print(f"loaded {len(self.raw_data)} sentences from the original {len(hf_examples)} sentences")

    def _convert_examplo_io_to_iob(self,example):
        new_example=example.copy()
        for k in new_example:
            if k==self.feature_name:
                new_example[k]=self._convert_io_to_iob(example[k])
        return new_example
    
    def _convert_io_to_iob(self,raw_tags):
        tags=raw_tags.copy()
        prev_tag=None
        for ii,tag in enumerate(tags):
            if (prev_tag is not None) and prev_tag!=tag and tag in self.i_b_dict:
                tags[ii]=self.i_b_dict[tag]
            prev_tag=tag
        return tags

    def _build_b_i_dict(self):
        self.b_i_dict=dict()
        self.i_b_dict=dict()
        label_names=self.class_label_obj.names
        for ii,ln in enumerate(label_names):
            if ln.startswith("I-"):
                entity_type=ln[2:]
                b_index=label_names.index("B-"+entity_type)
                self.b_i_dict[b_index]=ii
                self.i_b_dict[ii]=b_index


class HFNerSpanDataset(HFNerDataset):
    def __init__(self,hf_examples,tokenizer:Tokenizer,class_label_obj:ClassLabel,feature_name, span_generator_fn:Optional[Callable[[Union[List[str],int]],Generator[Tuple[int,int],None,None]]], tag_format):
        super().__init__(hf_examples,tokenizer,class_label_obj,feature_name, tag_format)
        self.span_data=self._broadcast_sentences_spans(self.raw_data, span_generator_fn=span_generator_fn)

    def _broadcast_sentences_spans(self, raw_data, span_generator_fn):
        """
        For each sentence in the list, create multiple sentences
        each containing only one entity or no entity
        """
        new_sentences=[]
        for raw_sentence in tqdm(raw_data,desc="Broadcasting sentences"):
            new_sentence=[]
            tags=raw_sentence[self.feature_name]
            #find spans of entities in the form o f min and max index
            spans=[]
            types=[]
            for ii,tag in enumerate(tags):
                if tag in self.b_i_dict:
                    spans.append((ii,ii))
                    types.append(tag)
                elif tag in self.i_b_dict:
                    spans[-1]=(spans[-1][0],ii)
            for s,t in zip(spans,types):
                new_sentence.append((raw_sentence["tokens"],s,t))
            #get n random non-entity spans
            n=len(raw_sentence["tokens"])
            max_spans=n*(n+1)/2
            potential_negative_spans=max_spans-len(spans)
            #num_neg_samples=min(neg_sample_rate,potential_negative_spans)
            by_id=False
            if span_generator_fn:
                all_spans=[]
                #check if the first argument of fn is an int by type hints
                if len(inspect.signature(span_generator_fn).parameters)==1:
                    type_hints=typing.get_type_hints(span_generator_fn)
                    if len(type_hints)==1:
                        par_type=list(typing.get_type_hints(span_generator_fn).values())[0]
                        if par_type==int:
                            by_id=True
                if by_id:
                    all_spans=span_generator_fn(int(raw_sentence["id"]))#Much faster if read from cache
                else:
                    all_spans=span_generator_fn(raw_sentence["tokens"])#This is much slower
                for a_span in all_spans:
                    if a_span not in spans:
                        new_sentence.append((raw_sentence["tokens"],a_span,0))
            #new_sentence=(raw_sentence, new_sentence)
            new_sentences.extend(new_sentence)
        return new_sentences
    
    def __len__(self):
        return len(self.span_data)
    
    def __getitem__(self,idx):
        return self.span_data[idx]
    
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


class HFNerIOBDataset(HFNerDataset):
    def __init__(self,hf_examples,tokenizer:Tokenizer,class_label_obj:ClassLabel,feature_name, tag_format):
        super().__init__(hf_examples,tokenizer,class_label_obj,feature_name, tag_format)
    
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self,idx):
        return self.raw_data[idx]

    def _get_ne_masks(self,tags,sentence_mask,num_masks=None,pad_value=-1,only_multi_token_ne=False):
        """
        Generates masks for the sentence and tags one row for each named entity with 1s in the NE tokens and 0s in the rest of the sentence
        Padding is done with pad_value
        :param sentence: list of tokens
        :param tags: list of tags
        :param sentence_mask: list of 0s and 1s where 1s indicate that the token is not padding
        :param num_masks: number of masks to generate, if None then no horizontal padding is done
        :param pad_value: value to pad with
        :return: list of masks (list of lists of 0s,1s and pad_value)
        """        
        masks=[]
        current_tag=None
        start=None
        for ii,tag in enumerate(tags):
            if sentence_mask[ii]==0:
                if current_tag and start:
                    len_ne=ii-start
                    if (only_multi_token_ne and len_ne>1) or not only_multi_token_ne:
                        masks.append(self._get_mask(start,ii-1,sentence_mask,pad_value))
                current_tag=None
                start=None
                continue
            if tag in self.b_i_dict:#B-
                if start:
                    len_ne=ii-start
                    if only_multi_token_ne and len_ne==1:
                        current_tag=None
                        start=None
                        continue
                    masks.append(self._get_mask(start,ii-1,sentence_mask,pad_value))
                start=ii
                current_tag=self.b_i_dict[tag]
                continue
            if not current_tag or tag==current_tag:#I- or O after O
                continue
            if tag!=current_tag:#O after I- or B-
                len_ne=ii-start
                if only_multi_token_ne and len_ne==1:
                    current_tag=None
                    start=None
                    continue
                masks.append(self._get_mask(start,ii-1,sentence_mask,pad_value))
                current_tag=None
                start=None
                continue
        if current_tag and start:
            len_ne=ii-start
            if (only_multi_token_ne and len_ne>1) or not only_multi_token_ne:
                masks.append(self._get_mask(start,ii-1,sentence_mask,pad_value))
        if num_masks:
            while len(masks)<num_masks:
                masks.append([pad_value]*len(sentence_mask))
        return masks

    def _get_mask(self,start,end,sentence_mask,pad_value):
        mask=[pad_value]*len(sentence_mask)
        for ii,mv in enumerate(sentence_mask):
            if mv:
                if ii>=start and ii<=end:
                    mask[ii]=1
                else:
                    mask[ii]=0
        return mask

    def _get_ner_tags(self,raw_tags):#final representation of tags as indices
        return raw_tags

    def collate_fn(self,batch):
        all_words=[]
        all_ids=[]
        all_tags=[]
        for example in batch:
            all_words.append(example["tokens"])
            all_ids.append(int(example["id"]))
            all_tags.append(self._get_ner_tags(example[self.feature_name]))

        inputs=self.tokenizer(all_words,return_tensors="pt",is_split_into_words=True,padding=True,return_attention_mask=True,add_special_tokens=False,return_special_tokens_mask=True)
        length=inputs.input_ids.shape[1]
        ids=torch.tensor(all_ids,dtype=torch.int,device=inputs.input_ids.device)
        pad_tags=-1
        padded_tags=torch.full(size=(len(batch),length),fill_value=pad_tags,dtype=torch.long,device=inputs.input_ids.device)
        raw_ne_masks_batch=[]
        max_num_masks=0
        ne_mask_padding=-1
        all_word_ids=[]
        for ii,tags in enumerate(all_tags):
            words_ids=inputs.word_ids(ii)
            #change None to -1
            words_ids_pad=[-1 if x is None else x for x in words_ids]
            all_word_ids.append(words_ids_pad)
            prev_word_id = None
            new_tags=[]
            prev_word_id = None
            for word_id in words_ids:
                if word_id is None:
                    new_tags.append(pad_tags)
                else:
                    if prev_word_id is not None and word_id == prev_word_id and new_tags:#Word split by tokenizer
                        if new_tags[-1] in self.b_i_dict:
                            new_tags.append(self.b_i_dict[new_tags[-1]])#B-LOC to I-LOC
                        else:
                            if new_tags[-1] in self.b_i_dict.values():
                                new_tags.append(new_tags[-1])#I-LOC to I-LOC
                            else:
                                new_tags.append(tags[word_id])
                    else:
                        new_tags.append(tags[word_id])
                prev_word_id=word_id
            padded_tags[ii,:len(new_tags)]=torch.tensor(new_tags,dtype=torch.long,device=padded_tags.device)
            raw_ne_masks=self._get_ne_masks(new_tags,inputs.attention_mask[ii],pad_value=ne_mask_padding)
            raw_ne_masks_batch.append(raw_ne_masks)
            max_num_masks=max(max_num_masks,len(raw_ne_masks))
        all_word_ids=torch.tensor(all_word_ids,dtype=torch.long,device=inputs.input_ids.device)

        ne_masks=torch.full(size=(len(batch),max_num_masks,length),fill_value=ne_mask_padding,dtype=torch.short,device=inputs.input_ids.device)
        try:
            for ii,raw_ne_masks in enumerate(raw_ne_masks_batch):
                if len(raw_ne_masks)>0:
                    ne_masks[ii,:len(raw_ne_masks),:]=torch.tensor(raw_ne_masks,dtype=torch.short,device=ne_masks.device)
        except Exception as e:
            print(e)
            raise e

        return {"ids":ids,
                "inputs":inputs,
                "labels":padded_tags,
                "final_cluster_masks":ne_masks,
                "all_word_ids":all_word_ids}
    


def include_special_tokens(model:BertModel,tokenizer:Tokenizer):
    global E_START_ID,E_END_ID
    tokenizer.add_special_tokens({"additional_special_tokens":[E_START,E_END]})
    E_START_ID=tokenizer.convert_tokens_to_ids(E_START)
    E_END_ID=tokenizer.convert_tokens_to_ids(E_END)
    model.resize_token_embeddings(len(tokenizer))

if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained("prajjwal1/bert-tiny",use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens":[E_START,E_END]})
    data=load_dataset("wnut_17")
    dm=HFNer_DataModule(data,tokenizer=tokenizer,batch_size=2)
    dl=dm.train_dataloader()

    for batch in dl:
        print(batch)
        break

    dl_val=dm.val_dataloader()
    for batch in dl_val:
        print(batch)
        break