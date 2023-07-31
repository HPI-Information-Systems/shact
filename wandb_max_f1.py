"""
script to find the max f1 score from wandb history and log it to wandb.
Theargument is the wandb run id
"""
import wandb
import argparse
F1_FIELD="metrics/val_f1"

def log_max_f1(run_path):
    api=wandb.Api()
    print(f"run path: {run_path}")
    run=api.run(run_path)
    print(f"run name: {run.name}")
    print("history:")
    history=run.scan_history(keys=[F1_FIELD])
    max_f1=None
    for h in history:
        try:
            f1=h[F1_FIELD]
            if f1 and (f1>max_f1 or max_f1 is None):
                max_f1=f1
        except KeyError:
            pass
    if max_f1 is None:
        print("no f1 scores found")
        return
    print(f"max f1: {max_f1}")
    #add to summary
    run.summary.update({"metrics/max_val_f1":max_f1})
    #run.log({"max_val_f1":max_f1})

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run_path")
    args=parser.parse_args()
    log_max_f1(args.run_path)