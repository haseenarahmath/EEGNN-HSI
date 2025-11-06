import argparse

def build_parser(desc=""):
    p = argparse.ArgumentParser(description=desc or "Run EEGNN")
    p.add_argument("--mode", choices=["train", "eval", "bo"], default="train")
    p.add_argument("--dataset", type=str, default="in")        # "in" -> Indian Pines
    p.add_argument("--model", choices=["DeepGCN","BranchyDeepGCN"], default="BranchyDeepGCN")
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--hid", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--confidence", type=float, default=0.75)   # default single threshold
    p.add_argument("--conf_metric", choices=["max_prob","top_pred_diff","entropy"], default="max_prob")
    p.add_argument("--out_dir", type=str, default="results")
    return p
