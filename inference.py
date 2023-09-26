import json
from typing import List, Tuple
from inference_model import LSHAC_NER_Prediction
import pytorch_lightning as pl
from tqdm import tqdm
from model import LSHAC_FlatNERModel, LSHAC_NERModel, LSHAC_NestedNERModel
from transformers import AutoTokenizer,AutoModel,AutoConfig
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from data_modules import HFNer_DataModule, HFNestedNer_DataModule, include_special_tokens, get_tag_format
from datasets import load_dataset
import wandb
from dotenv import dotenv_values
from argparse import Namespace
import vis
import re
import data_adapters as da

def is_perfect_prediction(prediction:LSHAC_NER_Prediction,gt:List[Tuple[int,int,str]])->bool:
    """
    Checks if the prediction is perfect
    """
    w_assignments=prediction.get_word_assignments()
    for (s,e,t) in gt:
        type_index=prediction.types_list.index(t)
        found=False
        for ((s2,e2),t2) in w_assignments:
            if s==s2 and e==e2 and type_index==t2:
                found=True
        if not found:
            return False
    return True

if __name__ == '__main__':
    env_config = dotenv_values(".env")
    out_folder=env_config["LSHAC_NER_OUTPUT_DIR"]
    use_wandb=(env_config["WANDB"] is None) or env_config["WANDB"]=="True"
    wandb_project=env_config["WANDB_PROJECT"]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_path", type=str, help="Wandb run to use")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--workers", default=min(os.cpu_count(),64), type=int, help="Number of dataloader workers")
    parser.add_argument("--use_test", action="store_true", help="Use the test split. Should only be used for the final evaluation")
    #parser.add_argument("--clean", action="store_true", help="Delete the images in the wandb run before uploading new ones")
    parser.add_argument("--tree_type", default="errors", type=str, choices=["all","errors","none"] , help="Which trees to generate")
    parser.add_argument("--save_predictions", action="store_true", help="Save predicted results in a file")
    parser.add_argument("--limit", type=int, help="Number batches to predict. Useful for debugging")
    #parser = pl.Trainer.add_argparse_args(parser)
    #parser.set_defaults(accelerator="gpu",devices=1,max_epochs=300)
    args = parser.parse_args()
    #print args to stdout
    print("Arguments:")
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    pl.seed_everything(args.seed)
    logger=False
    #if use_wandb:
    api = wandb.Api()
    run = api.run(args.run_path)
    #wandb.init(id=run.id, project=wandb_project , resume="must")
    old_config=run.config
    #delet limit keys
    for k in ["limit_test_batches"]:
        if k in old_config:
            del old_config[k]
    if "gpus" in old_config:
        del old_config["gpus"]
    old_args=Namespace(**old_config)
    save_dir=os.path.join(out_folder,"wandb_checkpoints")
    old_args.accelerator="gpu"
    old_args.devices=1
    print("Reusing old config: ",old_args)
    #sys.exit(0)
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

    old_args.limit_predict_batches=args.limit
    old_args.limit_test_batches=args.limit

    trainer=pl.Trainer.from_argparse_args(old_args,logger=logger,deterministic=True)

    hf_dataset=None
    if old_args.sub_dataset:
        hf_dataset=load_dataset(old_args.dataset,old_args.sub_dataset)
    else:
        hf_dataset=load_dataset(old_args.dataset)

    if not hasattr(da,old_args.data_adapter):
        print("Data adapter",old_args.data_adapter,"not found")
        exit(1)
    data_adapter_fn=getattr(da,old_args.data_adapter)
    hf_dataset=data_adapter_fn(hf_dataset)

    dm = HFNestedNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size,
                                num_workers=args.workers, limit_samples=0)#, test_batch_size=args.batch_size)
    model_class = LSHAC_NestedNERModel
    
    run_spl=args.run_path.split("/")
    assert len(run_spl)==3
    ckpt_dir=os.path.join(save_dir,run_spl[1],run_spl[2],"checkpoints")
    if os.path.exists(ckpt_dir):
        ckpt=[f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        assert len(ckpt)>=0
        ckpt=os.path.join(ckpt_dir,ckpt[-1])
        ner_model = LSHAC_NestedNERModel.load_from_checkpoint(checkpoint_path=ckpt,transformer_model=transformers_model, classes=dm.class_label_obj, neg_sample_size=1, tokenizer=tokenizer)
    else:
        print("No checkpoint found")
        exit(1)

    assert ner_model is not None
    ner_model.warmup=False
    class_label_obj=dm.class_label_obj
    dataloader_for_test=dm.test_dataloader() if args.use_test else dm.val_dataloader()
    dataset_for_test=dataloader_for_test.dataset
    #dataloader_for_test=dm.val_dataloader()
    trainer.test(ner_model,dataloaders=dataloader_for_test)
    res=trainer.predict(ner_model,dataloaders=dataloader_for_test)
    suffix="test" if args.use_test else "val"
    pred_file_name=os.path.join(ckpt_dir,f"pred_{suffix}.jsonl")
    if args.save_predictions:
        with open(pred_file_name,"w") as f:
            pass
    imgs_folder=None
    if args.tree_type!="none":
        imgs_folder=os.path.join(ckpt_dir,"imgs")
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)
    for (predictions, batch) in tqdm(res,desc="Processing predictions"):
        pred_seq,gt_seq=ner_model.compute_results(prediction_objs=predictions,batch=batch)
        ids=batch["ids"]
        if type(ids)==torch.Tensor:
            ids=ids.tolist()
        gt_seq=[]
        words=[]
        ds_ids=[]
        for id in ids:
            item=dataset_for_test[id]
            if "sample_id" in item:
                ds_ids.append(item["sample_id"])
            elif "id" in item:
                ds_ids.append(item["id"])
            else:
                ds_ids.append(id)
            #if item is a tuple of 2 elements, the first is the id
            if type(item)==tuple and len(item)==2:
                item=item[1]
            words.append(item["tokens"])
            gt_raw=item["spans"]
            gt_spans=[(e["start"],e["end"]-1,class_label_obj.int2str(e["label"])) for e in gt_raw]
            gt_seq.append(gt_spans)
        if args.save_predictions:
            with open(pred_file_name,"a") as f:
                for id,ws,ps in zip(ds_ids,words,pred_seq):
                    #generate json
                    json_obj={"id":id,"tokens":ws,"spans":[]}
                    for ((s,e),t) in ps.get_word_assignments():
                        json_obj["spans"].append({"start":s,"end":e+1,"type":ps.types_list[t]})
                    f.write(json.dumps(json_obj)+"\n")
        if args.tree_type!="none":
            for p,g,p_obj,input_ids,sentece_words,id in zip(pred_seq,gt_seq,predictions,batch["inputs"]["input_ids"],words,ids):
                if args.tree_type=="all" or (not is_perfect_prediction(p_obj, g)):
                    colors={t:c for t,c in zip(dm.class_label_obj.names,vis.generate_colors(len(dm.class_label_obj.names)))}
                    tree=p_obj.get_pydot_tree(colors=colors)
                    sentence=tokenizer.decode(input_ids, skip_special_tokens=True)
                    tokens=tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
                    is_leaf=lambda x: not any([edge.get_source()==x.get_name() for edge in tree.get_edges()])
                    leaves=[node for node in tree.get_nodes() if is_leaf(node)]
                    for i,leaf in enumerate(leaves):
                        span=eval(eval(leaf.get_name()))
                        tokens_span=tokens[span[0]:span[1]+1]
                        leaf.set_label(leaf.get_label()+"\n"+" ".join(tokens_span))
                    bytes_svg = tree.create_svg() # Binary string
                    #create html with hg
                    with open(os.path.join(imgs_folder,"tree_"+str(id)+".html"),"w") as file:
                        file.write("<!DOCTYPE html>\n")
                        file.write("<html>\n")
                        file.write("<head>\n")
                        file.write(f"<title>Prediction {str(id)}</title>\n")
                        file.write("</head>\n")
                        file.write("<body>\n")
                        file.write(f"<h1>{str(id)}</h1>\n")
                        p_spans=[(s,e,dm.class_label_obj.int2str(t)) for ((s,e),t) in p_obj.get_word_assignments() if t!=0]
                        html_p=vis.visualize_spans(sentece_words,p_spans, colors=colors)
                        html_g=vis.visualize_spans(sentece_words,g, colors=colors)
                        file.write("<h2>Prediction</h2>\n")
                        file.write(html_p)
                        file.write("<h2>Ground truth</h2>\n")
                        file.write(html_g)
                        file.write("<h2>Tree</h2>\n")
                        file.write("<div>\n")
                        svg_str=bytes_svg.decode("utf-8")
                        #replace width="\d+pt" height="\d+pt" with width="100%" using regex
                        svg_str=re.sub(r'width="\d+pt" height="\d+pt"','width="1024pt"',svg_str)
                        #remove everything before <svg
                        svg_str=svg_str[svg_str.find("<svg"):]
                        file.write(svg_str)
                        file.write("</div>\n")
                        file.write("</body>\n")
                        file.write("</html>\n")
    if args.save_predictions:
        print(f"Saved predictions to {pred_file_name}")
    if imgs_folder:
        print("Images saved in",imgs_folder)

    