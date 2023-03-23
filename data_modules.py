from typing import List, Optional
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
    def __init__(self,hf_dataset,tokenizer:Tokenizer,batch_size=32,num_workers=None,tag_format="IOB",undersample_rate=None,only_with_mw_nes=False, feature_name="ner_tags"):
        super().__init__()
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        self.train_data, self.val_data, self.test_data = hf_dataset["train"], hf_dataset["validation"], hf_dataset["test"]
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
        self.only_with_mw_nes=only_with_mw_nes
        self.dl_train,int2str=self._getloader(self.train_data,self.batch_size)
        self.int2str["train"]=int2str
        self.class_label_obj=self.dl_train.dataset.class_label_obj
        self.num_classes=self.dl_train.dataset.class_label_obj.num_classes

    def train_dataloader(self):
        # dl,int2str=self._getloader(self.train_data,self.batch_size)
        # self.int2str["train"]=int2str
        return self.dl_train

    def val_dataloader(self):
        dl,int2str=self._getloader(self.val_data,self.batch_size)
        self.int2str["val"]=int2str
        return dl

    def test_dataloader(self):
        dl,int2str=self._getloader(self.test_data,self.batch_size)
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

    def _getloader(self,data,batch_size):
        split_class_label_obj=data.features[self.feature_name].feature
        int2str=None
        if self.tag_format=="IOB":
            ds=HFNerIOBDataset(data,self.tokenizer,class_label_obj=split_class_label_obj,only_with_mw_nes=self.only_with_mw_nes, feature_name=self.feature_name)
            int2str=ds.class_label_obj.int2str
        elif self.tag_format=="IO":
            ds=HFNerIO_to_IOB_Dataset(data,self.tokenizer,io_class_label_obj=split_class_label_obj,only_with_mw_nes=self.only_with_mw_nes, feature_name=self.feature_name)
            int2str=ds.class_label_obj.int2str
        return DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=self.num_workers),int2str

class HFNerIOBDataset(Dataset):
    def __init__(self,hf_examples,tokenizer:Tokenizer,class_label_obj:ClassLabel,only_with_mw_nes, feature_name):
        super().__init__()
        self.feature_name=feature_name
        self.tokenizer=tokenizer
        self.class_label_obj=class_label_obj
        self._build_b_i_dict()
        #remove empty sentences and sentences with only one token. If only_with_mw_nes is True, remove sentences with no MW_NES
        self.raw_data=[sentence for sentence in hf_examples if (len(sentence["tokens"])>1 and ((not only_with_mw_nes) or self._contains_mw_ner(sentence)))]
        print(f"loaded {len(self.raw_data)} sentences from the original {len(hf_examples)} sentences")

    def _contains_mw_ner(self,sentence):
        pairs=[[b,i] for b,i in self.b_i_dict.items()]
        for pair in pairs:
            if self._is_sublist(pair,sentence[self.feature_name]):
                return True
        return False

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self,idx):
        return self.raw_data[idx]

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

    #function that checks if a list is a sublist of another list
    def _is_sublist(self,sublist,list):
        if len(sublist)>len(list):
            return False
        for ii in range(len(list)-len(sublist)+1):
            if sublist==list[ii:ii+len(sublist)]:
                return True
        return False

class HFNerIO_to_IOB_Dataset(HFNerIOBDataset):
    #ds=HFNerIO_to_IOB_Dataset(data,self.tokenizer,io_class_label_obj=self.class_label_obj,only_with_mw_nes=self.only_with_mw_nes)
    #self,hf_examples,tokenizer:Tokenizer,class_label_obj:ClassLabel,only_with_mw_nes
    def __init__(self,hf_examples,tokenizer:Tokenizer,io_class_label_obj:ClassLabel,only_with_mw_nes,feature_name):
        super().__init__(hf_examples,tokenizer,io_class_label_obj,only_with_mw_nes=only_with_mw_nes,feature_name=feature_name)
        io_names=io_class_label_obj.names.copy()
        b_names=[]
        for ii,io_name in enumerate(io_names):
            if not io_name.startswith("I-") and io_name!="O":
                io_names[ii]="I-"+io_name
            if io_names[ii].startswith("I-"):
                b_names.append("B-"+io_names[ii][2:])
        names=io_names+b_names
        self.class_label_obj=ClassLabel(names=names)
        self._build_b_i_dict()

    #Override
    def _contains_mw_ner(self,sentence):
        O_idx=self.class_label_obj.str2int("O")
        pairs=[[i,i] for i in range(self.class_label_obj.num_classes) if i!=O_idx]
        for pair in pairs:
            if self._is_sublist(pair,sentence[self.feature_name]):
                return True
        return False


    #override _get_ner_tags. Converts IO to IOB
    def _get_ner_tags(self,raw_tags):
        tags=raw_tags.copy()
        prev_tag=None
        for ii,tag in enumerate(tags):
            if (prev_tag is not None) and prev_tag!=tag and tag in self.i_b_dict:
                tags[ii]=self.i_b_dict[tag]
            prev_tag=tag
        return tags

def include_special_tokens(model:BertModel,tokenizer:Tokenizer):
    global E_START_ID,E_END_ID
    tokenizer.add_special_tokens({"additional_special_tokens":[E_START,E_END]})
    E_START_ID=tokenizer.convert_tokens_to_ids(E_START)
    E_END_ID=tokenizer.convert_tokens_to_ids(E_END)
    model.resize_token_embeddings(len(tokenizer))

if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained("bert-base-cased",use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens":[E_START,E_END]})
    data=load_dataset("wnut_17")
    dm=HFNer_DataModule(data,tokenizer=tokenizer,batch_size=2)
    dl=dm.train_dataloader()
    batch=iter(dl).next()
    #first_elem=batch[0]
    #print(first_elem)
    print(batch)