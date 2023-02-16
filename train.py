import pytorch_lightning as pl
import datasets
from model import LSHAC_NERModel
from transformers import AutoTokenizer,AutoModel,AutoConfig
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os,re
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from data_modules import HFNer_DataModule, include_special_tokens
from datasets import load_dataset
import wandb

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
    logger = TensorBoardLogger("./ner_tb", name="baseline_ner")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="LSHAC_NERModel", type=str, help="Name of the model")
    parser.add_argument("--wandb_project", default="ls_ner", type=str, help="Name of the project for logging in wandb")
    parser.add_argument("--lang_model_name", default="bert-base-uncased", type=str, help="Transformers (Bert) model name")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use local tensorboard instead of wandb")
    parser.add_argument("--logger_folder", default="/mnt/data/alsier/ls_ner", type=str, help="Where the tensorboard logger will save the models and logs. For wandb, this is the folder where the models will be saved")
    parser.add_argument("--dataset", default="wnut_17", type=str, help="HF Dataset to use")
    parser.add_argument("--sub_dataset", type=str, help="HF Dataset to use. For example for 'dfki-nlp/few-nerd' it could be 'supervised")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--restart_ls", action="store_true", help="Restart the weights of the latent space")
    parser.add_argument("--patience",  default=5, type=int, help="Patience for early stopping")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--undersample_rate", type=float, help="Percentage of the training data to use")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1,max_epochs=300)
    args = parser.parse_args()
    logger=None
    if args.use_tensorboard:
        logger = TensorBoardLogger(args.logger_folder, name=args.model_name)
    else:
        logger = WandbLogger(project=args.wandb_project,name=args.model_name,save_dir=os.path.join(args.logger_folder,"wandb_checkpoints"))
    
    lang_model_name=args.lang_model_name
    print("Using lang model",lang_model_name)

    early_stop = EarlyStopping(monitor="losses/val_loss",mode="max",patience=args.patience)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="losses/val_loss", mode="min")
    
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

    trainer=pl.Trainer.from_argparse_args(args,logger=logger,callbacks=[early_stop, checkpoint_callback])

    hf_dataset=None
    if args.sub_dataset:
        hf_dataset=load_dataset(args.dataset,args.sub_dataset)
    else:
        hf_dataset=load_dataset(args.dataset)            
    assert args.undersample_rate is None or (args.undersample_rate<=1.0 and args.undersample_rate>=0,0)

    dm=HFNer_DataModule(hf_dataset,tokenizer=tokenizer,batch_size=args.batch_size,tag_format=get_tag_format(hf_dataset),undersample_rate=args.undersample_rate,only_with_mw_nes=False)
    #model=LSHAC_NERModel(transformers_model,num_labels=dm.num_classes,int2str_fn=dm.int2str["train"],lr=args.lr)
    ner_model=LSHAC_NERModel(transformers_model,classes=dm.class_label_obj,lr=1e-3,ls_hidden_size=128,distance_fn=torch.cdist,affinity="euclidean")
    
    assert ner_model is not None
    
    if not args.use_tensorboard:
        logger.experiment.config.update(vars(args))
        logger.watch(ner_model)
    
    trainer.fit(ner_model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())