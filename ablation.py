import random
from typing import Callable, Generator, List, Tuple, Union
import pytorch_lightning as pl
from tqdm import tqdm
from model import Ablat_HAC_full_encoder, Ablat_HAC_last_encoder, LSHAC_NestedNERModel, LSHAC_FlatNERModel, LSHAC_NERModel
from transformers import AutoTokenizer,AutoModel,AutoConfig
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os,re
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from data_modules import HFNer_DataModule, HFNerIOBDataset,HFNestedNer_DataModule, get_tag_format, include_special_tokens
from datasets import load_dataset
import wandb
from dotenv import dotenv_values
from latent_space import cosine_distance
from utils import build_tensors_for_inference
from transformers import PreTrainedTokenizerFast
import configparser
import data_adapters as da
#disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def random_span_sampler(words:List[str]) -> Generator[Tuple[int,int],None,None]:
    """Sample 2*length+1 random spans from the text"""
    yielded=set()
    generated=set()
    length=len(words)
    while len(yielded)<(2*length+1) and len(generated)<(length*(length+1)//2):
        span_length=random.randint(1,length)#TODO favor shorter spans
        start=random.randint(0,len(words)-1)
        end=start+span_length-1
        if (start,end) not in generated and end<len(words):
            generated.add((start,end))
        if (start,end) not in yielded and end<len(words):
            yielded.add((start,end))
            yield start,end

def get_otf_ls_span_generator(model:LSHAC_NERModel, tokenizer:PreTrainedTokenizerFast) -> Callable[[Union[List[str],int]],Generator[Tuple[int,int],None,None]]:
    def generator(words:List[str]):
        #run inference
        uni_batch=build_tensors_for_inference(words=words, tokenizer=tokenizer)
        preds,_=model.predict(uni_batch)
        pred=preds[0]
        word_spans=pred.get_word_spans()
        # yield all spans in the clusters
        # TODO favor shorter spans / shuffle the order
        for span in word_spans:
            yield span
    return generator

if __name__ == '__main__':
    env_config = dotenv_values(".env")
    out_folder=env_config["LSHAC_NER_OUTPUT_DIR"]
    use_wandb=False
    wandb_project=env_config["WANDB_PROJECT"]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="LSHAC_NERModel", type=str, help="Name of the model")
    parser.add_argument("--lang_model_name", default="bert-base-uncased", type=str, help="Transformers (Bert) model name")
    #parser.add_argument("--dataset", default="wnut_17", type=str, help="HF Dataset to use")
    #parser.add_argument("--sub_dataset", type=str, help="HF Dataset to use. For example for 'dfki-nlp/few-nerd' it could be 'supervised")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--test_batch_size", type=int, help="batch size for validation and test")
    parser.add_argument("--restart_ls", action="store_true", help="Restart the weights of the latent space")
    parser.add_argument("--patience",  default=5, type=int, help="Patience for early stopping")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--undersample_rate", type=float, help="Percentage of the training data to use")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help="Number of dataloader workers")
    parser.add_argument("--distance", default="cosine", type=str,choices=["cosine","euclidean"] , help="Distance function to use")
    parser.add_argument("--limit_samples", type=int , help="Limit to the number of spans for sentence to use for training the classifier")
    parser.add_argument("--warmup_epochs", type=int , default=5, help="Number of non entity spans for sentence to use for training the classifier")
    #boolean argument for only considering entity or non entity spans
    #parser.add_argument("--only_entities", action="store_true", help="Only consider entity spans")
    #add config ini arg for task
    parser.add_argument("--task_config", type=str, required=True , help="Config file for task")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator="gpu",devices=1,max_epochs=300)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.task_config)
    args.dataset=config["Dataset"].get("hf_dataset_name")
    args.sub_dataset=config["Dataset"].get("hf_subdataset_name")
    args.data_adapter = config["Dataset"].get("data_adapter")

    final_prediction_type=config["Inference"].get("final_prediction") # full or flat
    flat:bool=final_prediction_type=="flat"

    pl.seed_everything(args.seed)
    logger=False
    if use_wandb:
        logger = WandbLogger(project=wandb_project,name=args.model_name,save_dir=os.path.join(out_folder,"wandb_checkpoints"))

    lang_model_name=args.lang_model_name
    print("Using lang model",lang_model_name)

    early_stop = EarlyStopping(monitor="metrics/val_f1",mode="max",patience=args.patience)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="metrics/val_f1",mode="max")
    
    config = AutoConfig.from_pretrained(lang_model_name, output_hidden_states=True, output_attentions=True, output_special_tokens=True)
    transformer_model = AutoModel.from_pretrained(lang_model_name, config=config)

    #instantiate fast tokenizer, add add_prefix_space in case of roberta
    if lang_model_name.startswith("roberta"):
        tokenizer = AutoTokenizer.from_pretrained(lang_model_name, use_fast=True,add_prefix_space=True)
    else:
        tokenizer=AutoTokenizer.from_pretrained(lang_model_name,use_fast=True)
    
    include_special_tokens(transformer_model,tokenizer)

    trainer=pl.Trainer.from_argparse_args(args,logger=logger,callbacks=[early_stop, checkpoint_callback],deterministic=True)

    hf_dataset=None
    if args.sub_dataset:
        hf_dataset=load_dataset(args.dataset,args.sub_dataset)
    else:
        hf_dataset=load_dataset(args.dataset)

    if not hasattr(da,args.data_adapter):
        print("Data adapter",args.data_adapter,"not found")
        exit(1)
    data_adapter_fn=getattr(da,args.data_adapter)
    hf_dataset=data_adapter_fn(hf_dataset)

    assert args.undersample_rate is None or (args.undersample_rate<=1.0 and args.undersample_rate>=0,0)

    dm = HFNestedNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size,
                                num_workers=args.workers, limit_samples=args.limit_samples, test_batch_size=args.test_batch_size)

    distance_fn = cosine_distance if args.distance == "cosine" else torch.cdist
    hac_metric="cosine" if args.distance=="cosine" else "euclidean"
    ner_model = LSHAC_NestedNERModel(transformer_model, classes=dm.class_label_obj, tokenizer=tokenizer,lr=args.lr, ls_hidden_size=128, distance_fn=distance_fn, hac_metric=hac_metric,
                               flat=flat)

    assert ner_model is not None
    

    ner_model.warmup=True
    #freeze backbone
    for param in ner_model.transformer_model.parameters():
        param.requires_grad = False
    warmup_trainer=pl.Trainer.from_argparse_args(args,logger=None,deterministic=True, enable_checkpointing=False, max_epochs=args.warmup_epochs)
    print("Testing before training")
    warmup_trainer.test(ner_model,dataloaders=dm.test_dataloader())
    dm = HFNestedNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size,
                                num_workers=args.workers, limit_samples=args.limit_samples, test_batch_size=args.test_batch_size)
    warmup_trainer.fit(ner_model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())
    print("Starting test (after training)")
    warmup_trainer.test(ner_model,dataloaders=dm.test_dataloader())
    
    print("Starting test (concatentaion)")
    dm = HFNestedNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size,
                                num_workers=args.workers, limit_samples=args.limit_samples, test_batch_size=args.test_batch_size)
    ner_model=Ablat_HAC_full_encoder(transformer_model, classes=dm.class_label_obj, tokenizer=tokenizer,lr=args.lr,ls_hidden_size=128, distance_fn=distance_fn, hac_metric=hac_metric,
                               flat=flat)
    ner_model.warmup=True
    #freeze backbone
    for param in ner_model.transformer_model.parameters():
        param.requires_grad = False
    warmup_trainer=pl.Trainer.from_argparse_args(args,logger=None,deterministic=True, enable_checkpointing=False, max_epochs=args.warmup_epochs)
    warmup_trainer.test(ner_model,dataloaders=dm.test_dataloader())

    
    
    print("Starting test (last layer)")
    dm = HFNestedNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size,
                                num_workers=args.workers, limit_samples=args.limit_samples, test_batch_size=args.test_batch_size)
    ner_model=Ablat_HAC_last_encoder(transformer_model, classes=dm.class_label_obj, tokenizer=tokenizer,lr=args.lr,ls_hidden_size=128, distance_fn=distance_fn, hac_metric=hac_metric,
                               flat=flat)
    ner_model.warmup=True
    #freeze backbone
    for param in ner_model.transformer_model.parameters():
        param.requires_grad = False
    warmup_trainer=pl.Trainer.from_argparse_args(args,logger=None,deterministic=True, enable_checkpointing=False, max_epochs=args.warmup_epochs)
    warmup_trainer.test(ner_model,dataloaders=dm.test_dataloader())