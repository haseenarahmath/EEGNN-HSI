# util.py
# EEGNN (ISDA 2023) — utilities for data I/O, metrics, logging

from __future__ import annotations
import csv
import os
from os.path import exists
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import scipy.io as scio
import sklearn
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from torch_geometric.data import Data

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
OUT_PATH = "results/"
Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUT_PATH, "features")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUT_PATH, "csv")).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def _append_or_create_csv(filename: str, header: List[str] | None, row: List[object]) -> None:
    file_exists = exists(filename)
    mode = "a" if file_exists else "w"
    with open(filename, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if (not file_exists) and header:
            writer.writerow(header)
        writer.writerow(row)


def write_into_csv1(rows: List[object]) -> None:
    fields = ["train/val", "epochs", "time", "loss", "acc", "kappa", "class_acc"]
    _append_or_create_csv(os.path.join(OUT_PATH, "time_analysis.csv"), fields, rows)


def write_into_csv2(rows: List[object]) -> None:
    fields = ["train/val", "Training samples", "Testing", "epochs", "time", "loss", "acc", "kappa", "class_acc"]
    _append_or_create_csv(os.path.join(OUT_PATH, "result.csv"), fields, rows)


def write_into_csv11(
    model_name: str, hidden_dim: int, lr: float, nlayer: int, dropout: float, norm_mode: str,
    best_acc: float, best_loss: float, best_kappa: float,
    best_acc_dict, best_loss_dict, best_kappa_dict, best_aa_dict, P1: float, R1: float, F1: float, name: str
) -> None:
    fields = [
        "model", "hidden_dim", "lr", "nlayer", "dropout", "norm_mode", "best_acc", "best_loss", "best_kappa",
        "best_acc_dict", "best_loss_dict", "best_kappa_dict", "best_aa_dict", "Precision", "Recall", "F1score", "name"
    ]
    rows = [
        model_name, hidden_dim, lr, nlayer, dropout, norm_mode, best_acc, best_loss, best_kappa, best_acc_dict,
        best_loss_dict, best_kappa_dict, best_aa_dict, P1, R1, F1, name
    ]
    _append_or_create_csv(os.path.join(OUT_PATH, "results_records_GCN.csv"), fields, rows)


def write_into_csv(filename: str, data: List[object] | Tuple[object, ...]) -> None:
    """Append a single CSV row to `filename` (creates file if needed)."""
    file_exists = exists(filename)
    mode = "a" if file_exists else "w"
    Path(os.path.dirname(filename) or ".").mkdir(parents=True, exist_ok=True)
    with open(filename, mode, newline="") as csvfile:
        csv.writer(csvfile).writerow(list(data))


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy where labels are one-hot."""
    preds = output.argmax(dim=1)
    labels_ = labels.argmax(dim=1)
    return float((preds == labels_).sum().item()) / int(labels_.numel())


def accuracy_2(prediction_: torch.Tensor, confidence: torch.Tensor, labels: torch.Tensor):
    """
    Compute (correct, total_pred) among samples selected by `confidence` mask.
    - prediction_: logits or probabilities [N, C] (torch tensor)
    - confidence : boolean mask [N] (torch tensor of dtype=bool or 0/1)
    - labels     : one-hot [N, C] (torch tensor)
    """
    if prediction_.is_cuda:
        prediction_ = prediction_.detach().cpu()
    if labels.is_cuda:
        labels = labels.detach().cpu()
    if confidence.is_cuda:
        confidence = confidence.detach().cpu()

    conf = confidence.bool()
    pred_arg_max = prediction_.argmax(dim=1)[conf]
    true_arg_max = labels.argmax(dim=1)[conf]

    correct = (pred_arg_max == true_arg_max).double().sum()
    total_pred = conf.sum()
    return correct, total_pred


def calculate_metric(y: torch.Tensor | np.ndarray, output: torch.Tensor | np.ndarray):
    """
    Weighted Precision/Recall/F1 using numpy arrays.
    Accepts logits/probabilities or one-hot for `output`, one-hot for `y`.
    """
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_pred = np.argmax(output, axis=1)
    y_true = np.argmax(y, axis=1)
    P1 = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    R1 = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    F1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return P1, R1, F1


def class_wise_accuracies(
    predicted: torch.Tensor, labels: torch.Tensor, n_classes: int, input_is_logits: bool = True
) -> List[float]:
    """
    Returns per-class accuracy list + [micro_avg, overall_%].
    - If input_is_logits=True: will argmax both predicted & labels.
    """
    if input_is_logits:
        predicted = predicted.argmax(dim=1)
        labels = labels.argmax(dim=1)

    predicted = predicted.detach().cpu()
    labels = labels.detach().cpu()

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.float32)
    for t, p in zip(labels, predicted):
        confusion[t.long(), p.long()] += 1.0

    per_class = []
    total_correct = 0.0
    for i in range(n_classes):
        row_sum = confusion[i].sum()
        acc_i = (confusion[i, i] / row_sum) if row_sum > 0 else torch.tensor(0.0)
        per_class.append(float(acc_i))
        total_correct += float(confusion[i, i])

    micro_avg = total_correct / float(confusion.sum() + 1e-12)
    overall_pct = 100.0 * micro_avg
    per_class.append(float(micro_avg))
    per_class.append(float(overall_pct))
    return per_class


def get_kappa(true_labels: torch.Tensor, predictions: torch.Tensor, n_c: int = 16) -> float:
    """
    Cohen's kappa for one-hot true labels and logits/probs predictions.
    """
    if predictions.is_cuda:
        predictions = predictions.detach().cpu()
    if true_labels.is_cuda:
        true_labels = true_labels.detach().cpu()

    y_pred = predictions.argmax(dim=1).numpy()
    y_true = true_labels.argmax(dim=1).numpy()
    return float(cohen_kappa_score(y_pred, y_true))


# ---------------------------------------------------------------------
# Feature dump
# ---------------------------------------------------------------------
def save_features(pred_: Union[torch.Tensor, np.ndarray, Iterable], model_name: str, nlayer: int) -> None:
    """
    Save predictions/features to MATLAB .mat for later analysis.
    Accepts:
      - torch.Tensor [N, C] or list/iterable of tensors
      - numpy.ndarray [N, C]
    """
    if isinstance(pred_, torch.Tensor):
        payload = pred_.detach().cpu().numpy()
    elif isinstance(pred_, np.ndarray):
        payload = pred_
    elif isinstance(pred_, Iterable):
        # list of tensors/arrays
        payload = [ (p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)) for p in pred_ ]
    else:
        payload = np.asarray(pred_)

    name = f"features_{model_name}_{nlayer}.mat"
    mdic = {"features": payload}
    Path(os.path.join(OUT_PATH, "features")).mkdir(parents=True, exist_ok=True)
    scio.savemat(os.path.join(OUT_PATH, "features", name), mdic)


# ---------------------------------------------------------------------
# Misc small helpers
# ---------------------------------------------------------------------
def sample_mask(idx: np.ndarray, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mask[idx] = True
    return mask


def convert_to_one_hot(Y: np.ndarray, C: int) -> np.ndarray:
    return np.eye(C, dtype=np.float32)[Y.reshape(-1)]  # [N, C]


# ---------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------
def load_data(ds: str):
    """Load preprocessed .mat bundles from data/ for given short code."""
    if ds == "in":
        AX = scio.loadmat("data/ALL_X.mat")
        AY = scio.loadmat("data/ALL_Y.mat")
        AL = scio.loadmat("data/ALL_L.mat")
    elif ds == "pav":
        AX = scio.loadmat("data/New_Pav_ALL_X.mat")
        AY = scio.loadmat("data/New_Pav_ALL_Y.mat")
        AL = scio.loadmat("data/New_Pav_ALL_L.mat")
    elif ds == "sl":
        AX = scio.loadmat("data/SALINAS_X.mat")
        AY = scio.loadmat("data/SALINAS_Y.mat")
        AL = scio.loadmat("data/SALINAS_L.mat")
    elif ds == "bt":
        AX = scio.loadmat("data/Botswana_X.mat")
        AY = scio.loadmat("data/Botswana_Y.mat")
        AL = scio.loadmat("data/Botswana_L.mat")
    else:
        raise ValueError("Unknown dataset code. Use one of: 'in', 'pav', 'sl', 'bt'.")
    return AX, AY, AL


def preprocess_data(
    ALL_X, ALL_Y, ALL_L, num_classes: int, train_c: int, test_c: int
) -> Data:
    """
    Builds a torch_geometric Data object with:
      x:    [N, B] spectral features
      y:    [N, C] one-hot labels
      adj:  [N, N] sparse adjacency (from ALL_L)
      train_mask / val_mask / test_mask
    """
    # adjacency (sparse)
    ALL_L = torch.from_numpy(ALL_L["ALL_L"].todense()).float()
    # features
    ALL_X = torch.from_numpy(ALL_X["ALL_X"]).float()
    # labels (integer class ids starting at 1; convert to one-hot)
    ALL_Y = ALL_Y["ALL_Y"]

    N = ALL_Y.shape[0]
    train_mask = torch.from_numpy(sample_mask(np.arange(0, train_c), N))
    test_mask = torch.from_numpy(sample_mask(np.arange(train_c, test_c), N))

    # shuffle masks together (keeps sizes)
    train_mask, test_mask = sklearn.utils.shuffle(train_mask, test_mask, random_state=0)

    Y_onehot = convert_to_one_hot(ALL_Y - 1, num_classes)  # -> [N, C]
    Y_onehot = torch.from_numpy(Y_onehot).float()

    n_x = ALL_X.shape[1]
    n_y = Y_onehot.shape[1]

    data_ = Data(
        num_features=n_x,
        num_classes=n_y,
        x=ALL_X,
        y=Y_onehot,
        adj=ALL_L,
        train_mask=train_mask,
        val_mask=test_mask,   # original code uses same split for val/test
        test_mask=test_mask,
    )
    return data_
