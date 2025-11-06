#!/usr/bin/env python3
"""
EEGNN (ISDA 2023) — Evaluation & Threshold Search

- Computes per-exit accuracy under confidence-based early-exit rules
- Summarizes CDF of exits, per-exit % usage, and overall accuracy
- (Optional) Bayesian Optimization over thresholds for a composite objective
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from util import (
    write_into_csv, load_data, preprocess_data,
    accuracy_2, save_features, get_kappa,
    class_wise_accuracies, calculate_metric,
    OUT_PATH,
)

# ------------------------------
# Config & constants
# ------------------------------

DEFAULT_NUM_CLASSES = 16
DEFAULT_SPLIT = dict(train_c=3340, test_c=7721)  # matches your original evaluate.py
DEFAULT_ENERGY_VALUES = {
    0: 0.285977758835701, 1: 0.42878220706856, 2: 0.57158665530142, 3: 0.654391105021869,
    4: 0.704391105021869, 5: 0.754391105021869, 6: 0.807195552510934, 7: 0.857195552510934,
    8: 0.907195552510934, 9: 0.957195552510934, 10: 1.0
}
CONF_MAP = {0: "max_prob", 1: "top_pred_diff", 2: "entropy"}


# ------------------------------
# I/O helpers
# ------------------------------

def load_data_and_model(dataset_name: str, model_name: str, use_cuda: bool):
    """Loads dataset split + trained Branchy model."""
    ALL_X, ALL_Y, ALL_L = load_data(dataset_name)
    data = preprocess_data(
        ALL_X, ALL_Y, ALL_L,
        num_classes=DEFAULT_NUM_CLASSES,
        train_c=DEFAULT_SPLIT["train_c"],
        test_c=DEFAULT_SPLIT["test_c"],
    )

    ckpt = Path(f"results/model/{model_name}_{dataset_name}.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    net = torch.load(ckpt.as_posix())
    net.eval()

    if use_cuda and torch.cuda.is_available():
        data = data.cuda()
        # keep compatibility with your BranchyNet wrappers
        if hasattr(net, "to_gpu"):
            net.to_gpu()
        else:
            net.to("cuda")

    return data, net


# ------------------------------
# Confidence / entropy
# ------------------------------

def row_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Stable per-row entropy from logits (natural log)."""
    p = torch.softmax(logits, dim=1).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1)  # [N]


@torch.no_grad()
def confidence_mask(
    logits: torch.Tensor,
    threshold: float,
    conf_type: str
) -> torch.Tensor:
    """
    Returns a boolean mask [N] indicating which samples exit at this head.
    conf_type: 'max_prob' | 'top_pred_diff' | 'entropy'
    """
    if conf_type == "max_prob":
        conf, _ = torch.softmax(logits, dim=1).max(dim=1)
    elif conf_type == "top_pred_diff":
        sort_pred, _ = torch.sort(torch.softmax(logits, dim=1), dim=1)
        conf = sort_pred[:, -1] - sort_pred[:, -2]
    elif conf_type == "entropy":
        # Normalize entropy to [0,1] by dividing by log(C)
        H = row_entropy_from_logits(logits)
        C = logits.size(1)
        conf = 1.0 - (H / math.log(C))
    else:
        raise ValueError(f"Unknown conf_type: {conf_type}")

    return conf >= float(threshold)


# ------------------------------
# Metrics & summaries
# ------------------------------

def summarize_per_exit(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    thresholds: List[float],
    conf_type: str,
    energy_values: Dict[int, float],
    out_dir: Path,
    ds_name: str,
    model_name: str,
    n_classes: int = DEFAULT_NUM_CLASSES,
) -> Dict[str, object]:
    """
    Calculates per-exit accuracy, overall accuracy, exit CDF/% and saves CSVs.
    """
    exits = len(predictions)
    assert len(thresholds) >= exits, "Threshold list shorter than number of exits."

    correct_total = 0
    total_pred_list: List[int] = []
    per_exit_acc: List[float] = []
    n_accumulated = np.zeros(exits, dtype=np.int64)

    for idx in range(exits):
        logits = predictions[idx].detach().cpu()
        y_true = targets[idx].detach().cpu()

        mask = confidence_mask(logits, thresholds[idx], conf_type)
        correct, total_pred = accuracy_2(logits, mask, y_true)  # <- your util
        total_pred = int(total_pred)

        # per-exit stats
        acc = 0.0 if total_pred == 0 else (float(correct) / float(total_pred)) * 100.0
        per_exit_acc.append(acc)
        total_pred_list.append(total_pred)
        n_accumulated[idx] = total_pred if idx == 0 else n_accumulated[idx - 1] + total_pred
        correct_total += int(correct)

        # Optional per-exit dumps
        save_features(logits.numpy(), f"{model_name}_{idx}_{ds_name}", 10)
        kappa = get_kappa(y_true, torch.argmax(logits, dim=1), n_classes)
        cc = class_wise_accuracies(torch.argmax(logits, dim=1), y_true, n_classes)
        P1, R1, F1 = calculate_metric(y_true, torch.argmax(logits, dim=1))
        write_into_csv(
            OUT_PATH + "csv/accuracies.csv",
            [
                f"{model_name}_Evaluation", "exit_idx", idx,
                "kappa", kappa, "P", P1, "R", R1, "F1", F1, "dataset", ds_name, "classwise", cc
            ],
        )

    grand_total = max(1, sum(total_pred_list))
    overall_acc = correct_total / grand_total
    cdf_exits = 100.0 * (n_accumulated / grand_total)
    pct_exit = 100.0 * (np.array(total_pred_list) / grand_total)

    write_into_csv(
        (out_dir / "accuracies.csv").as_posix(),
        [
            "per_exit_acc(%)", per_exit_acc,
            "overall_acc", overall_acc,
            "cdf_exits(%)", cdf_exits.tolist(),
            "n_accumulated", n_accumulated.tolist(),
            "per_exit_counts", total_pred_list,
            "per_exit_pct(%)", pct_exit.tolist(),
            "conf_type", conf_type,
            "thresholds", thresholds[:exits],
        ],
    )

    return dict(
        per_exit_acc=per_exit_acc,
        overall_acc=overall_acc,
        cdf_exits=cdf_exits,
        per_exit_counts=total_pred_list,
        per_exit_pct=pct_exit,
    )


# ------------------------------
# Composite BO objective (optional)
# ------------------------------

def _normalize(arr: List[float], t_min: float, t_max: float) -> List[float]:
    mn, mx = min(arr), max(arr)
    if mx - mn < 1e-12:
        return [0.5 * (t_min + t_max) for _ in arr]
    return [(((v - mn) * (t_max - t_min)) / (mx - mn)) + t_min for v in arr]


def composite_objective(
    per_exit_acc: List[float],
    per_exit_pct: np.ndarray,
    energy_values: Dict[int, float],
    overall_acc: float,
    ld: float,
) -> float:
    """
    Weighted sum of (normalized exit-acc * usage %) and (1 - energy) with decay ld^x,
    plus overall accuracy bonus.
    """
    exits = len(per_exit_acc)
    acc_component = _normalize((np.array(per_exit_acc) * per_exit_pct).tolist(), 1.0, 2.0)
    score = 0.0
    for x in range(exits):
        theta = ld ** x
        energy_term = (1.0 - float(energy_values.get(x, 1.0)))
        score += (acc_component[x] + energy_term) * theta
    score += float(overall_acc)
    return 0.0 if math.isnan(score) else score


def run_bayes_opt(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    conf_type: str,
    out_dir: Path,
    energy_values: Dict[int, float],
    init_points: int = 25,
    n_iter: int = 25,
):
    """Runs Bayesian Optimization over up to first 5 exits' thresholds in [0.7, 0.99]."""
    try:
        from bayes_opt import BayesianOptimization
    except Exception as e:
        print(f"[WARN] bayesian-optimization not installed: {e}")
        return

    exits = min(5, len(predictions))  # tune first few exits for practicality
    bounds = {f"c{i+1}": (0.7, 0.99) for i in range(exits)}
    bounds["ld"] = (0.65, 0.65)  # fixed decay as in your code (can widen later)

    def _objective(**kwargs):
        ld = float(kwargs.pop("ld"))
        # thresholds per exit
        thr = [float(kwargs[f"c{i+1}"]) for i in range(exits)]
        # fill any remaining exits with last threshold
        thr_full = thr + [thr[-1]] * (len(predictions) - exits)
        summary = summarize_per_exit(
            predictions, targets, thr_full, conf_type, energy_values, out_dir,
            ds_name="bo", model_name="eegnn"
        )
        return composite_objective(
            summary["per_exit_acc"], summary["per_exit_pct"], energy_values, summary["overall_acc"], ld
        )

    bo = BayesianOptimization(f=_objective, pbounds=bounds, random_state=12, allow_duplicate_points=True)
    bo.maximize(init_points=init_points, n_iter=n_iter)

    # Persist BO results
    write_into_csv((out_dir / "csv" / "bo.csv").as_posix(), bo.res)
    write_into_csv((out_dir / "csv" / "bomax.csv").as_posix(), [bo.max])


# ------------------------------
# CLI
# ------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="EEGNN — Evaluate early exits & search thresholds")
    p.add_argument("--dataset", type=str, default="in", help="dataset short code (e.g., in, pav)")
    p.add_argument("--model", type=str, default="BranchyDeepGCN")
    p.add_argument("--conf_type", choices=["max_prob", "top_pred_diff", "entropy"], default="max_prob")
    p.add_argument("--thresholds", type=float, nargs="*", default=None,
                   help="per-exit thresholds; if not given, defaults to 0.75 for all")
    p.add_argument("--bo", action="store_true", help="run Bayesian Optimization over thresholds")
    p.add_argument("--out_dir", type=str, default="results")
    return p


def main():
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    data, net = load_data_and_model(args.dataset, args.model, use_cuda=(device == "cuda"))
    exits = len(net.network.stages) if hasattr(net, "network") else getattr(net, "nBranches", 1)
    print(f"[INFO] exits detected: {exits}")

    # Get per-exit logits/targets from your Branchy API
    prediction, true_values = net.evaluate_branches(data.x, data.y, data.adj, data.val_mask)
    assert len(prediction) == len(true_values) == exits

    # thresholds: default 0.75 for all exits unless provided
    if args.thresholds is None or len(args.thresholds) == 0:
        thresholds = [0.75] * exits
    else:
        thresholds = list(args.thresholds) + [args.thresholds[-1]] * max(0, exits - len(args.thresholds))

    # Summarize with given thresholds
    summary = summarize_per_exit(
        prediction, true_values, thresholds, args.conf_type,
        energy_values=DEFAULT_ENERGY_VALUES, out_dir=out_dir,
        ds_name=args.dataset, model_name=args.model, n_classes=DEFAULT_NUM_CLASSES
    )

    print("\n=== Summary ===")
    print(f"Per-exit accuracy (%): {summary['per_exit_acc']}")
    print(f"Overall accuracy     : {summary['overall_acc']:.4f}")
    print(f"CDF of exits   (%)   : {summary['cdf_exits'].tolist()}")
    print(f"Exit usage     (%)   : {summary['per_exit_pct'].tolist()}")

    # Optional: BO
    if args.bo:
        print("\n[INFO] Running Bayesian Optimization over thresholds...")
        run_bayes_opt(
            prediction, true_values, args.conf_type, out_dir,
            energy_values=DEFAULT_ENERGY_VALUES, init_points=200, n_iter=25
        )
        print("[OK] BO complete.")

if __name__ == "__main__":
    main()
