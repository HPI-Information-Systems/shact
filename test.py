import pytorch_lightning as pl
import datasets
from tqdm import tqdm
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
from argparse import Namespace
from PIL import Image, ImageDraw, ImageFont
import io
from tabulate import tabulate
import vis
import imgkit

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
    parser.add_argument("--clean", action="store_true", help="Delete the images in the wandb run before uploading new ones")
    parser.add_argument("--tree_type", default="errors", type=str, choices=["all","errors","none"] , help="Which trees to generate")
    parser.add_argument("--limit", type=int, help="Number batches to predict. Useful for debugging")
    #parser = pl.Trainer.add_argparse_args(parser)
    #parser.set_defaults(accelerator="gpu",devices=1,max_epochs=300)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    logger=False
    #if use_wandb:
    api = wandb.Api()
    run = api.run(args.run_path)
    if args.clean:
        print("Deleting old files under media/images/test/")
        for f in run.files():
            if f.name.startswith("media/images/test/"):
                f.delete()
    wandb.init(id=run.id, project=wandb_project , resume="must")
    old_config=run.config
    #delet limit keys
    for k in ["limit_test_batches"]:
        if k in old_config:
            del old_config[k]
    if "gpus" in old_config:
        del old_config["gpus"]
    old_args=Namespace(**old_config)
    logger = WandbLogger(project=wandb_project,name=old_args.model_name,save_dir=os.path.join(out_folder,"wandb_checkpoints"))
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

    dm = HFNer_DataModule(hf_dataset, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.workers,
                          tag_format=get_tag_format(hf_dataset,feature_name=args.feature_name), feature_name=old_args.feature_name)
    
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
    ner_model.warmup=False
    dataloader_for_test=dm.test_dataloader() if args.use_test else dm.val_dataloader()
    dataset_for_test=dataloader_for_test.dataset
    #dataloader_for_test=dm.val_dataloader()
    trainer.test(ner_model,dataloaders=dataloader_for_test)
    if args.tree_type!="none":
        res=trainer.predict(ner_model,dataloaders=dataloader_for_test)
        for (predictions, batch) in tqdm(res,desc="Processing predictions"):
            pred_seq,gt_seq=ner_model.compute_labels(prediction_objs=predictions,batch=batch)
            ids=batch["ids"]
            pred_seq=[p.seq_labels_compressed for p in predictions]
            gt_seq=[]
            words=[]
            for id in ids:
                item=dataset_for_test[id]
                words.append(item["tokens"])
                gt_ints=item[old_args.feature_name]
                gt_seq.append([dm.class_label_obj.int2str(i) for i in gt_ints])
            batch_images=[]
            for p,g,p_obj,input_ids,sentece_words in zip(pred_seq,gt_seq,predictions,batch["inputs"]["input_ids"],words):
                if p!=g or args.tree_type=="all":
                    tree=p_obj.get_pydot_tree()
                    sentence=tokenizer.decode(input_ids, skip_special_tokens=True)
                    tokens=tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
                    #tokens=[tokenizer.convert_tokens_to_string(t).strip() for t in tokens]
                    is_leaf=lambda x: not any([edge.get_source()==x.get_name() for edge in tree.get_edges()])
                    leaves=[node for node in tree.get_nodes() if is_leaf(node)]
                    for i,leaf in enumerate(leaves):
                        span=eval(eval(leaf.get_name()))
                        tokens_span=tokens[span[0]:span[1]+1]
                        leaf.set_label(leaf.get_label()+"\n"+" ".join(tokens_span))
                    bytes_image = tree.create_png()
                    img=Image.open(io.BytesIO(bytes_image))
                    #resize to max 1024 width
                    img_w, img_h = img.size
                    if img_w>1024:
                        img_h=int(img_h*1024/img_w)
                        img_w=1024
                        img=img.resize((img_w,img_h))
                    #draw = ImageDraw.Draw(image)
                    #font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
                    #tab_data=[["Pred"]+p,["GT"]+g]
                    #headers=[""]+[str(i) for i in range(len(p))]
                    #text=tabulate(tab_data, headers=headers, tablefmt="grid")
                    #draw.text((0,img_h), text, font=font, fill=(0,0,0))
                    html_p=vis.visualize(sentece_words,tags_iob=p)
                    png_p=imgkit.from_string(html_p, False, options={"width":img_w, "quiet":None})
                    img_p=Image.open(io.BytesIO(png_p))
                    img_w_p, img_h_p = img_p.size
                    html_g=vis.visualize(sentece_words,tags_iob=g)
                    png_g=imgkit.from_string(html_g, False, options={"width":img_w, "quiet":None})
                    img_g=Image.open(io.BytesIO(png_g))
                    img_w_g, img_h_g = img_g.size
                    image = Image.new('RGBA', (img_w, img_h+img_h_g+img_h_p), (255, 255, 255, 255))
                    image.paste(img, (0,0))
                    image.paste(img_g, (0,img_h))
                    image.paste(img_p, (0,img_h+img_h_g))
                    font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
                    draw = ImageDraw.Draw(image)
                    draw.text((0,img_h), "Ground Truth", font=font, fill=(0,0,0))
                    draw.text((0,img_h+img_h_g), "Prediction", font=font, fill=(0,0,0))
                    wandb.log({"test/trees":wandb.Image(image, caption=sentence)})    

    