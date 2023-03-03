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
from data_modules import HFNer_DataModule, include_special_tokens, get_tag_format
from datasets import load_dataset
import wandb
from dotenv import dotenv_values
from latent_space import cosine_distance
from utils import ConfusionMatrixCallback
from argparse import Namespace

if __name__ == '__main__':
    env_config = dotenv_values(".env")
    out_folder=env_config["LSHAC_NER_OUTPUT_DIR"]
    use_wandb=(env_config["WANDB"] is None) or env_config["WANDB"]=="True"
    wandb_project=env_config["WANDB_PROJECT"]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_path", type=str, help="Wandb run to use")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--workers", default=os.cpu_count(), type=int, help="Number of dataloader workers")
    parser.add_argument("--use_test", action="store_true", help="Use the test split. Should only be used for the final evaluation")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator="gpu",devices=1,max_epochs=300)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    logger=False
    #if use_wandb:
    api = wandb.Api()
    run = api.run(args.run_path)
    wandb.init(id=run.id, resume="must")
    old_args=Namespace(**run.config)
    logger = WandbLogger(project=wandb_project,name=old_args.model_name,save_dir=os.path.join(out_folder,"wandb_checkpoints"))

    #api = wandb.Api()
    
    lang_model_name=old_args.lang_model_name
    
    config = AutoConfig.from_pretrained(lang_model_name, output_hidden_states=True, output_attentions=True, output_special_tokens=True)
    transformers_model = AutoModel.from_config(config)
    #transformers_model = AutoModel.from_pretrained(lang_model_name, config=config)
    # if not args.fine_tune_lm:
    #     for param in transformers_model.parameters():
    #         param.requires_grad = False
    #instantiate fast tokenizer, add add_prefix_space in case of roberta
    if lang_model_name.startswith("roberta"):
        tokenizer = AutoTokenizer.from_pretrained(lang_model_name, use_fast=True,add_prefix_space=True)
    else:
        tokenizer=AutoTokenizer.from_pretrained(lang_model_name,use_fast=True)
    
    include_special_tokens(transformers_model,tokenizer)

    trainer=pl.Trainer.from_argparse_args(old_args,logger=logger,deterministic=True)

    hf_dataset=None
    if old_args.sub_dataset:
        hf_dataset=load_dataset(old_args.dataset,old_args.sub_dataset)
    else:
        hf_dataset=load_dataset(old_args.dataset)            
    
    dm=HFNer_DataModule(hf_dataset,tokenizer=tokenizer,batch_size=args.batch_size,num_workers=args.workers,tag_format=get_tag_format(hf_dataset),only_with_mw_nes=False)
    
    #distance_fn=cosine_distance if old_args.distance=="cosine" else torch.cdist
    #hac_metric="cosine" if old_args.distance=="cosine" else "euclidean"
    run_spl=args.run_path.split("/")
    assert len(run_spl)==3
    ckpt_dir=os.path.join(logger.save_dir,run_spl[1],run_spl[2],"checkpoints")
    if os.path.exists(ckpt_dir):
        ckpt=[f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        assert len(ckpt)>=0
        ckpt=os.path.join(ckpt_dir,ckpt[-1])
        ner_model = LSHAC_NERModel.load_from_checkpoint(checkpoint_path=ckpt,transformer_model=transformers_model, classes=dm.class_label_obj, neg_sample_size=1)
    else:
        print("No checkpoint found")
        exit(1)


    assert ner_model is not None
        
    dataloader=dm.test_dataloader() if args.use_test else dm.val_dataloader()
    res=trainer.test(ner_model,dataloaders=dataloader)
    print(type(res))
    first_res=res[7][3]
    tree=first_res.get_networkx_tree()
    print(tree)
    