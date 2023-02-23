import pytorch_lightning as pl
import datasets
from model import LSHAC_NERModel
from transformers import AutoTokenizer,AutoModel,AutoConfig
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os,re
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from data_modules import HFNer_DataModule, include_special_tokens
from datasets import load_dataset
import wandb
from dotenv import dotenv_values
from latent_space import cosine_distance
from utils import ConfusionMatrixCallback

def get_tag_format(hf_dataset):
    is_iob=all([n.startswith("B-") or n.startswith("I-") or n=="O" for n in hf_dataset["train"].features["ner_tags"].feature.names])
    is_iobes=all([n.startswith("B-") or n.startswith("I-") or n.startswith("E-") or n.startswith("S-") or n=="O" for n in hf_dataset["train"].features["ner_tags"].feature.names])
    if is_iob:
        return "IOB"
    elif is_iobes:
        return "IOBES"
    else:
        return "IO"

if __name__ == '__main__':
    env_config = dotenv_values(".env")
    out_folder=env_config["LSHAC_NER_OUTPUT_DIR"]
    use_wandb=(env_config["WANDB"] is None) or env_config["WANDB"]=="True"
    wandb_project=env_config["WANDB_PROJECT"]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="LSHAC_NERModel", type=str, help="Name of the model")
    parser.add_argument("--lang_model_name", default="bert-base-uncased", type=str, help="Transformers (Bert) model name")
    parser.add_argument("--dataset", default="wnut_17", type=str, help="HF Dataset to use")
    parser.add_argument("--sub_dataset", type=str, help="HF Dataset to use. For example for 'dfki-nlp/few-nerd' it could be 'supervised")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--restart_ls", action="store_true", help="Restart the weights of the latent space")
    parser.add_argument("--patience",  default=5, type=int, help="Patience for early stopping")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--undersample_rate", type=float, help="Percentage of the training data to use")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help="Number of dataloader workers")
    parser.add_argument("--distance", default="cosine", type=str,choices=["cosine","euclidean"] , help="Distance function to use")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator="gpu",devices=1,max_epochs=300)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    logger=False
    if use_wandb:
        logger = WandbLogger(project=wandb_project,name=args.model_name,save_dir=os.path.join(out_folder,"wandb_checkpoints"))

    lang_model_name=args.lang_model_name
    print("Using lang model",lang_model_name)

    early_stop = EarlyStopping(monitor="losses/val_loss",mode="min",patience=args.patience)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="losses/val_loss", mode="min")
    confusion_matrix_callback=ConfusionMatrixCallback()
    
    config = AutoConfig.from_pretrained(lang_model_name, output_hidden_states=True, output_attentions=True, output_special_tokens=True)
    transformers_model = AutoModel.from_pretrained(lang_model_name, config=config)
    # if not args.fine_tune_lm:
    #     for param in transformers_model.parameters():
    #         param.requires_grad = False
    #instantiate fast tokenizer, add add_prefix_space in case of roberta
    if lang_model_name.startswith("roberta"):
        tokenizer = AutoTokenizer.from_pretrained(lang_model_name, use_fast=True,add_prefix_space=True)
    else:
        tokenizer=AutoTokenizer.from_pretrained(lang_model_name,use_fast=True)
    
    include_special_tokens(transformers_model,tokenizer)

    trainer=pl.Trainer.from_argparse_args(args,logger=logger,callbacks=[early_stop, checkpoint_callback, confusion_matrix_callback],deterministic=True)

    hf_dataset=None
    if args.sub_dataset:
        hf_dataset=load_dataset(args.dataset,args.sub_dataset)
    else:
        hf_dataset=load_dataset(args.dataset)            
    assert args.undersample_rate is None or (args.undersample_rate<=1.0 and args.undersample_rate>=0,0)

    dm=HFNer_DataModule(hf_dataset,tokenizer=tokenizer,batch_size=args.batch_size,num_workers=args.workers,tag_format=get_tag_format(hf_dataset),undersample_rate=args.undersample_rate,only_with_mw_nes=False)
    type_freq=dm.estimate_type_frequency()
    sum_freq=sum(type_freq.values())
    type_weigths={k:sum_freq-v for k,v in type_freq.items()}
    distance_fn=cosine_distance if args.distance=="cosine" else torch.cdist
    hac_metric="cosine" if args.distance=="cosine" else "euclidean"
    ner_model = LSHAC_NERModel(transformers_model, classes=dm.class_label_obj, lr=args.lr,
                               ls_hidden_size=128, distance_fn=distance_fn, hac_metric=hac_metric, type_weights=type_weigths)

    assert ner_model is not None
    
    if use_wandb:
        logger.watch(ner_model)
        wandb.config.update(vars(args))
    
    #trainer.validate(ner_model,dataloaders=dm.val_dataloader())
    trainer.fit(ner_model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())