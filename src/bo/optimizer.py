#!/usr/bin/env python3
"""
EEGNN — Bayesian Optimization over Early-Exit Thresholds

- Tunes per-exit confidence thresholds in [low, high] to maximize a composite
  objective that balances accuracy, exit usage, and estimated energy.
- Confidence metrics: max_prob | top_pred_diff | entropy (normalized).

Writes:
  results/csv/bo.csv         -> all BO trials
  results/csv/bomax.csv      -> best params + score
  results/bo_RESULTS.csv     -> readable summary for the best thresholds
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from util import (
    write_into_csv, load_data, preprocess_data,
    accuracy_2,  # (logits, mask, labels) -> (correct, total_pred)
    OUT_PATH,
)

# ------------------------------
# Defaults (match your paper setup)
# ------------------------------
DEFAULT_NUM_CLASSES = 16
DEFAULT_SPLIT = dict(train_c=695, test_c=10366)   # your bo.py used test mask
DEFAULT_ENERGY_VALUES = {                         # per-exit energy proxy
    0: 0.285977758835701,
    1: 0.42878220706856,
    2: 0.57158665530142,
    3: 0.714391105021869,
    4: 0.857195552510934,
    5: 1.0,
}

CONF_MAP = {0: "max_prob", 1: "top_pred_diff", 2: "entropy"}


# ------------------------------
# Data / model loading
# ------------------------------

def load_data_and_model(dataset: str, model: str, use_cuda: bool):
    ALL_X, ALL_Y, ALL_L = load_data(dataset)
    data = preprocess_data(
        ALL_X, ALL_Y, ALL_L,
        num_classes=DEFAULT_NUM_CLASSES,
        train_c=DEFAULT_SPLIT["train_c"],
        test_c=DEFAULT_SPLIT["test_c"],
    )

    ckpt = Path(f"results/model/{model}_{dataset}.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    net = torch.load(ckpt.as_posix())
    net.eval()

    if use_cuda and torch.cuda.is_available():
        data = data.cuda()
        if hasattr(net, "to_gpu"):
            net.to_gpu()
        else:
            net.to("cuda")

    return data, net


# ------------------------------
# Confidence / entropy helpers
# ------------------------------

def row_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Stable per-row entropy from logits (natural log)."""
    p = torch.softmax(logits, dim=1).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1)


@torch.no_grad()
def confidence_mask(logits: torch.Tensor, threshold: float, conf_type: str) -> torch.Tensor:
    """
    Boolean mask [N] for samples that meet the exit criterion at this head.
    conf_type ∈ { 'max_prob', 'top_pred_diff', 'entropy' }
    """
    if conf_type == "max_prob":
        conf, _ = torch.softmax(logits, dim=1).max(dim=1)
    elif conf_type == "top_pred_diff":
        s, _ = torch.sort(torch.softmax(logits, dim=1), dim=1)
        conf = s[:, -1] - s[:, -2]
    elif conf_type == "entropy":
        # normalize entropy to [0,1] by dividing by log(C)
        H = row_entropy_from_logits(logits)
        C = logits.size(1)
        conf = 1.0 - (H / math.log(C))
    else:
        raise ValueError(f"Unknown conf_type: {conf_type}")
    return conf >= float(threshold)


# ------------------------------
# Objective and scoring
# ------------------------------

def _normalize(arr: List[float], t_min: float, t_max: float) -> List[float]:
    mn, mx = min(arr), max(arr)
    if mx - mn < 1e-12:
        return [0.5 * (t_min + t_max) for _ in arr]
    return [(((v - mn) * (t_max - t_min)) / (mx - mn)) + t_min for v in arr]


def composite_objective(
    per_exit_acc: List[float],       # [%]
    per_exit_pct: np.ndarray,        # usage percentage per exit (0..100)
    energy_values: Dict[int, float],
    overall_acc: float,              # [0..1]
    ld: float,                       # geometric decay
) -> float:
    """
    Weighted sum of (normalized exit-acc * usage %) and (1 - energy) with decay ld^x,
    plus an overall accuracy bonus.
    """
    exits = len(per_exit_acc)
    # combine accuracy with usage %, then normalize
    acc_component = _normalize((np.array(per_exit_acc) * per_exit_pct).tolist(), 1.0, 2.0)
    score = 0.0
    for x in range(exits):
        theta = ld ** x
        energy_term = 1.0 - float(energy_values.get(x, 1.0))
        score += (acc_component[x] + energy_term) * theta
    score += float(overall_acc)
    return score


def summarize_with_thresholds(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    thresholds: List[float],
    conf_type: str,
) -> dict:
    """
    Compute per-exit accuracy, usage %, CDF, and overall accuracy given thresholds.
    """
    exits = len(predictions)
    assert len(thresholds) >= exits
    correct_total = 0
    total_pred_list: List[int] = []
    per_exit_acc: List[float] = []
    n_accumulated = np.zeros(exits, dtype=np.int64)

    for idx in range(exits):
        logits = predictions[idx].detach().cpu()
        y_true = targets[idx].detach().cpu()
        mask = confidence_mask(logits, thresholds[idx], conf_type)

        correct, total_pred = accuracy_2(logits, mask, y_true)  # from util.py
        total_pred = int(total_pred)
        acc = 0.0 if total_pred == 0 else (float(correct) / float(total_pred)) * 100.0

        per_exit_acc.append(acc)
        total_pred_list.append(total_pred)
        n_accumulated[idx] = total_pred if idx == 0 else n_accumulated[idx - 1] + total_pred
        correct_total += int(correct)

    grand_total = max(1, sum(total_pred_list))
    overall_acc = correct_total / grand_total
    cdf_exits = 100.0 * (n_accumulated / grand_total)
    pct_exit = 100.0 * (np.array(total_pred_list) / grand_total)

    return dict(
        per_exit_acc=per_exit_acc,
        overall_acc=overall_acc,
        per_exit_pct=pct_exit,
        cdf_exits=cdf_exits,
        counts=total_pred_list,
    )


# ------------------------------
# Bayesian Optimization driver
# ------------------------------

def run_bo(
    dataset: str,
    model: str,
    conf_type: str,
    low: float,
    high: float,
    init_points: int,
    n_iter: int,
    ld: float,
    tune_first_k: int,
    out_dir: Path,
    energy_values: Dict[int, float] = DEFAULT_ENERGY_VALUES,
):
    try:
        from bayes_opt import BayesianOptimization
    except Exception as e:
        raise RuntimeError(
            "bayesian-optimization is not installed. "
            "pip install bayesian-optimization"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, net = load_data_and_model(dataset, model, use_cuda=(device == "cuda"))

    # Evaluate on TEST MASK (as in your original bo.py)
    prediction, true_values = net.evaluate_branches(data.x, data.y, data.adj, data.test_mask)
    exits = len(prediction)
    K = min(max(1, tune_first_k), exits)

    # Parameter bounds for BO
    bounds = {f"c{i+1}": (low, high) for i in range(K)}

    def _objective(**kwargs):
        # thresholds for first K exits
        thr = [float(kwargs[f"c{i+1}"]) for i in range(K)]
        # extend remaining exits with last threshold
        thr_full = thr + [thr[-1]] * (exits - K)

        summary = summarize_with_thresholds(prediction, true_values, thr_full, conf_type)
        score = composite_objective(
            per_exit_acc=summary["per_exit_acc"],
            per_exit_pct=summary["per_exit_pct"],
            energy_values=energy_values,
            overall_acc=summary["overall_acc"],
            ld=ld,
        )
        return score

    bo = BayesianOptimization(f=_objective, pbounds=bounds, random_state=345, allow_duplicate_points=True)
    bo.maximize(init_points=init_points, n_iter=n_iter)

    # Persist all trials & the best
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    write_into_csv((out_dir / "csv" / "bo.csv").as_posix(), bo.res)
    write_into_csv((out_dir / "csv" / "bomax.csv").as_posix(), [bo.max])

    # Pretty summary row mirroring your earlier file
    best_thr = [bo.max["params"][f"c{i+1}"] for i in range(K)]
    best_thr_full = best_thr + [best_thr[-1]] * (exits - K)
    summary = summarize_with_thresholds(prediction, true_values, best_thr_full, conf_type)
    write_into_csv(
        (out_dir / "bo_RESULTS.csv").as_posix(),
        [
            "objective_score", bo.max["target"],
            "best_thresholds", best_thr_full,
            "conf_type", conf_type,
            "overall_acc", summary["overall_acc"],
            "per_exit_acc(%)", summary["per_exit_acc"],
            "per_exit_pct(%)", summary["per_exit_pct"].tolist(),
            "cdf_exits(%)", summary["cdf_exits"].tolist(),
        ],
    )

    print("\n[BO] Done.")
    print(f"Best thresholds (first {K} exits): {best_thr}")
    print(f"Overall acc: {summary['overall_acc']:.4f}")
    print(f"Per-exit acc (%): {summary['per_exit_acc']}")
    print(f"Exit usage (%):  {summary['per_exit_pct'].tolist()}")


# ------------------------------
# CLI
# ------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="EEGNN — Bayesian Optimization of exit thresholds")
    p.add_argument("--dataset", type=str, default="in", help="dataset code (e.g., in, pav)")
    p.add_argument("--model", type=str, default="BranchyDeepGCN")
    p.add_argument("--conf_type", choices=["max_prob", "top_pred_diff", "entropy"], default="max_prob")
    p.add_argument("--low", type=float, default=0.7, help="lower bound of threshold")
    p.add_argument("--high", type=float, default=0.99, help="upper bound of threshold")
    p.add_argument("--init_points", type=int, default=25)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--ld", type=float, default=0.65, help="geometric decay weight")
    p.add_argument("--tune_first_k", type=int, default=5, help="tune first K exits; others reuse Kth threshold")
    p.add_argument("--out_dir", type=str, default="results")
    return p


def main():
    args = build_parser().parse_args()
    run_bo(
        dataset=args.dataset,
        model=args.model,
        conf_type=args.conf_type,
        low=args.low,
        high=args.high,
        init_points=args.init_points,
        n_iter=args.n_iter,
        ld=args.ld,
        tune_first_k=args.tune_first_k,
        out_dir=Path(args.out_dir),
        energy_values=DEFAULT_ENERGY_VALUES,
    )


if __name__ == "__main__":
    main()
