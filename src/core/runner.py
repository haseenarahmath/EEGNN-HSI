import torch
from pathlib import Path
from src.utils.util import load_data, preprocess_data, write_into_csv
from src.train.evaluate import val_step
from src.train.training import train_mynetwork, valid_threshold_branches_model
from src.models.models import DeepGCN, BranchyDeepGCN
from src.branchy.branchynet import BranchyNet
from src.bo.optimizer import run_bayes_opt

def build_model(name, in_dim, hid, out_dim, dropout, nlayer):
    if name == "DeepGCN":
        return DeepGCN(in_dim, hid, out_dim, dropout, nlayer=nlayer, residual=1, norm_mode="PN-SI", norm_scale=1.0)
    if name == "BranchyDeepGCN":
        return BranchyDeepGCN(in_dim, hid, out_dim, dropout=dropout, nlayer=nlayer, residual=3,
                              norm_mode="PN-SI", norm_scale=15, nBranches=nlayer, exit_type="conv")
    raise ValueError(f"Unknown model {name}")

def run(args):
    # Load & preprocess (dataset short-codes handled in your util loader)
    ALL_X, ALL_Y, ALL_L = load_data(args.dataset)
    data = preprocess_data(ALL_X, ALL_Y, ALL_L, num_classes=16, train_c=695, test_c=10366)

    model = build_model(args.model, data.num_features, args.hid, data.num_classes, args.dropout, args.layers)

    if args.mode == "train":
        # Branchy joint training (weights from high→low exit by default)
        net = BranchyNet(network=model, device="cuda" if torch.cuda.is_available() else "cpu",
                         lr_main=args.lr, lr_branches=args.lr, weight_decay=args.wd, joint=True)
        losses, accs = net.train_main(data.x, data.y, data.adj, data.train_mask)  # warmup single step
        # full training (your original loops)
        # (You can plug in your existing train_main/val_main calls here.)

        torch.save(model, Path(args.out_dir)/f"{args.model}_{args.dataset}.pth")

    elif args.mode == "eval":
        model = torch.load(Path(args.out_dir)/f"{args.model}_{args.dataset}.pth")
        loss_val, acc_val, kappa, cc, _ = val_step(model, data.x, data.val_mask, data.adj, data.y,
                                                   torch.nn.CrossEntropyLoss())
        print(f"[VAL] acc={acc_val:.4f}, kappa={kappa}")
        write_into_csv(Path(args.out_dir)/"eval.csv", ["val", acc_val, kappa, cc])

    elif args.mode == "bo":
        # Bayesian optimization of per-exit thresholds
        run_bayes_opt(data, args, model_name=args.model, conf_metric=args.conf_metric, out_dir=args.out_dir)
    else:
        raise ValueError("Unknown mode")
