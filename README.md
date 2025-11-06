# EEGNN — Adaptive Early-Exit Inference in GNN-based Hyperspectral Image Classification

**“Adaptive Early-Exit Inference in Graph Neural Networks Based Hyperspectral Image Classification.”**  
This work augments a deep GNN with **intermediate exit branches** so **easy HSIs exit early**, reducing compute while preserving—or improving—accuracy.

## Highlights
- **Early-Exit Deep GNN (EEGNN):** exit heads attached after each graph block (Fig. 2 in the paper). 
- **Joint Training:** main + exits optimized together with a weighted loss vector **W**.
- **Confidence-based Inference:** exit decisions via **max_prob**, **top_pred_diff**, or **entropy**. 
- **Bayesian Optimization of Thresholds:** per-exit confidence thresholds **T** tuned to maximize an efficiency–accuracy objective.

# Folder layout 

```
EEGNN-HSI/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                         # CLI entry
├── src/
│   ├── core/
│   │   ├── parser.py               # argparse (modes: train / eval / bo)
│   │   └── runner.py               # training / evaluation orchestrator
│   ├── train/
│   │   ├── training.py             # (training )
│   │   └── evaluate.py             # validation helpers 
│   ├── models/
│   │   ├── models.py               # DeepGCN, BranchyDeepGCN, etc.
│   │   ├── early_exit.py           # ExitBlock and branching wiring
│   │   └── layers.py               # GraphConv, GAT-style layers, PairNorm
│   ├── branchy/
│   │   └── branchynet.py           # BranchyNet trainer (joint training)
│   ├── bo/
│   │   └── optimizer.py            # Bayesian threshold search (objective)
│   └── utils/
│       └── util.py                 # data I/O, metrics, CSV logging
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_bo.sh
└── results/                        # logs, checkpoints, CSVs
```

## Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Train EEGNN (BranchyDeepGCN) on Indian Pines
python main.py --mode train --dataset in --model BranchyDeepGCN --layers 10 --hid 128 --dropout 0.1

# 2) Evaluate exits (accuracy per exit, exit CDF)
python main.py --mode eval  --dataset in --model BranchyDeepGCN

# 3) Bayesian threshold optimization (max_prob / diff / entropy)
python main.py --mode bo    --dataset in --model BranchyDeepGCN --conf_metric max_prob
````
## Data

Place .mat files under `data/` (e.g., Indian Pines, Pavia). The loader in `utils/util.py` handles preprocessing and graph construction. 

## Citation

If you use this code or refer to this work, please cite:

> **Haseena Rahmath P** and **Kuldeep Chaurasia**.  
> *Adaptive Early-Exit Inference in Graph Neural Networks Based Hyperspectral Image Classification.*  
> *Proceedings of the International Conference on Intelligent Systems Design and Applications (ISDA 2023)*,  
> Springer LNCS, pp. 444–453.  
> DOI: [10.1007/978-3-031-64847-2_41](https://doi.org/10.1007/978-3-031-64847-2_41)


````
