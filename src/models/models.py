# models.py
# EEGNN
# - DeepGCN / MultiGCN (dense-adj GraphConv backbone)
# - GCN / GCN4 (PyG edge_index backbone)
# - GCNII
# - BranchyDeepGCN with early-exit branches
# Notes:
# * Returns logits by default (use log_softmax in the loss if needed).
# * Supports dense or sparse adjacency for GraphConv via _matmul_adj.

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from early_exit import ExitBlock
from layers import PairNorm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _matmul_adj(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multiply adjacency and features; works for dense or sparse adj."""
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


def _row_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise entropy (base e) from logits (stable)."""
    # softmax then sum_i (-p_i log p_i)
    p = torch.softmax(logits, dim=1).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1)


# ---------------------------------------------------------------------------
# GraphConv (dense-adj)
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Dense-adj graph conv: H = A (XW) + b."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # xavier uniform like
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x @ self.weight
        out = _matmul_adj(adj, h)
        return out + self.bias if self.bias is not None else out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}->{self.out_features})"


# ---------------------------------------------------------------------------
# DeepGCN (dense-adj)
# ---------------------------------------------------------------------------

class DeepGCN(nn.Module):
    """Deep GCN with PairNorm/residual; outputs logits."""
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        nlayer: int = 2,
        residual: int = 0,
        norm_mode: str = "None",
        norm_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert nlayer >= 1, "nlayer must be >= 1"

        self.fcs = nn.Linear(nfeat, nhid)
        self.hidden_layers = nn.ModuleList([GraphConv(nhid, nhid) for _ in range(nlayer)])
        self.out_layer = GraphConv(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)

        self.norm = PairNorm(norm_mode, norm_scale)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_last = nn.Dropout(p=0.4)
        self.skip = residual

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.fcs(x)
        res = torch.zeros_like(x)

        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and (i % self.skip == 0):
                x = x + res
                res = x

        x = self.dropout_last(x)
        x = self.out_layer(x, adj)
        logits = self.fc(x)
        return logits  # apply log_softmax in the loss if needed


# ---------------------------------------------------------------------------
# PyG GCN variants (edge_index)
# ---------------------------------------------------------------------------

class GCN4(nn.Module):
    """Two-layer GCN with BatchNorm (edge_index)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float, n_graph_layers: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.conv1(x, edge_index)
        x = self.bn2(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        return x  # logits


class GCN(nn.Module):
    """Two-layer GCN (edge_index)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float, n_graph_layers: int):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout_p)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_p)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv2(x, edge_index)
        return x  # logits


# ---------------------------------------------------------------------------
# GCNII (dense-adj variant)
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """GCNII layer per original formulation (dense-adj path)."""
    def __init__(self, in_features: int, out_features: int, residual: bool = False, variant: bool = False):
        super().__init__()
        self.variant = variant
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h0: torch.Tensor,
        lamda: float,
        alpha: float,
        layer_idx: int,
    ) -> torch.Tensor:
        theta = math.log(lamda / layer_idx + 1.0)
        hi = _matmul_adj(adj, x)  # A * X

        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        out = theta * (support @ self.weight) + (1 - theta) * r
        if self.residual:
            out = out + x
        return out


class GCNII(nn.Module):
    """GCNII with configurable nlayers; returns logits."""
    def __init__(self, nfeat: int, nhidden: int, nclass: int, dropout: float, nlayers: int,
                 lamda: float, alpha: float, variant: bool):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(nfeat, nhidden), nn.Linear(nhidden, nclass)])
        self.convs = nn.ModuleList([GraphConvolution(nhidden, nhidden, variant=variant) for _ in range(nlayers)])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_last = nn.Dropout(p=0.4)
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h0 = self.fcs[0](x)
        h = h0
        for i, conv in enumerate(self.convs, start=1):
            h = self.dropout(conv(h, adj, h0, self.lamda, self.alpha, i))
            h = self.relu(h)
        h = self.dropout_last(h)
        logits = self.fcs[-1](h)
        return logits


# ---------------------------------------------------------------------------
# MultiGCN (dense-adj)
# ---------------------------------------------------------------------------

class MultiGCN(nn.Module):
    """GCN stack with linear head (dense-adj)."""
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        nlayer: int = 2,
        residual: int = 0,
        norm_mode: str = "None",
        norm_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert nlayer >= 1
        self.fcs = nn.Linear(nfeat, nhid)
        self.hidden_layers = nn.ModuleList([GraphConv(nhid, nhid) for _ in range(nlayer)])
        self.out_layer = nn.Linear(nhid, nclass)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_last = nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.fcs(x)
        for layer in self.hidden_layers:
            h = self.dropout(h)
            h = layer(h, adj)
            h = self.relu(h)
        h = self.dropout_last(h)
        logits = self.out_layer(h)
        return logits


# ---------------------------------------------------------------------------
# BranchyDeepGCN (dense-adj) — Early-Exit model
# ---------------------------------------------------------------------------

class BranchyDeepGCN(nn.Module):
    """
    Early-Exit Deep GCN
    - Builds stages from input->graph blocks with a per-stage ExitBlock.
    - forward_main: run full stack and return final logits + predicted class.
    - forward_exits: run through stages and return per-exit outputs & confidence.
    """
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float = 0.06,
        nlayer: int = 2,
        residual: int = 0,
        norm_mode: str = "None",
        norm_scale: float = 1.0,
        nBranches: int = 2,
        exit_type: str = "conv",
        **kwargs,
    ):
        super().__init__()
        assert nlayer >= 1
        self.nBranches = max(1, nBranches)
        self.skip = residual

        self.fcs = nn.Linear(nfeat, nhid)
        self.hidden_layers = nn.ModuleList([GraphConv(nhid, nhid) for _ in range(nlayer)])

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.softmax = nn.Softmax(dim=1)

        # One ExitBlock per stage (including final)
        self.exit_branches = nn.ModuleList([ExitBlock(nhid, nclass, dropout, exit_type)
                                            for _ in range(self.nBranches)])

        # Build stages: each stage is a small Sequential of ops (fcs + [drop, conv, norm, relu] ...)
        self.stages: nn.ModuleList = nn.ModuleList()
        self.exits: nn.ModuleList = nn.ModuleList()
        self._build_early_exit()

        # Final classifier alias (last exit)
        self.classifier = self.exits[-1]

    # ---- construction helpers ----
    def _build_early_exit(self):
        layers = nn.ModuleList([self.fcs])
        branch_idx = 0
        res_count = 0

        for i, conv in enumerate(self.hidden_layers):
            layers.append(self.dropout)
            layers.append(conv)
            layers.append(self.norm)
            layers.append(self.relu)

            # Attach an exit after each block, until we have nBranches exits
            if branch_idx < self.nBranches:
                self.stages.append(nn.Sequential(*layers))
                self.exits.append(nn.Sequential(self.exit_branches[branch_idx]))
                layers = nn.ModuleList()
                branch_idx += 1
                res_count += 1

        # If for some reason no exits were attached inside loop, attach one at end
        if len(self.exits) == 0:
            self.stages.append(nn.Sequential(*layers))
            self.exits.append(nn.Sequential(self.exit_branches[0]))

    # ---- forward helpers ----
    def _run_stage(self, x: torch.Tensor, adj: torch.Tensor, stage: nn.Sequential, block_index: int) -> torch.Tensor:
        """Run a stage; handles residual across ReLU boundaries."""
        res = torch.zeros_like(x)
        for layer in stage:
            if isinstance(layer, GraphConv):
                x = layer(x, adj)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.skip > 0 and (block_index % self.skip == 0):
                    x = x + res
                    res = x
            else:
                x = layer(x)
        return x

    # ---- public forwards ----
    @torch.no_grad()
    def forward_main(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run through all stages and return (final_logits, predicted_class)."""
        for i, stage in enumerate(self.stages):
            x = self._run_stage(x, adj, stage, i)
        logits = self.exits[-1]([x, adj])
        pred = torch.argmax(self.softmax(logits), dim=1)
        return logits, pred

    def forward_exits(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run stages with exits; returns:
            outputs:   list of per-exit logits
            confid:    list of max-softmax confidence
            classes:   list of predicted classes
            entropies: list of prediction entropies
        """
        outputs: List[torch.Tensor] = []
        confid: List[torch.Tensor] = []
        classes: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for i, stage in enumerate(self.stages):
            x = self._run_stage(x, adj, stage, i)
            logits = self.exits[i]([x, adj])  # ExitBlock expects [x, adj]
            p = self.softmax(logits)
            conf, pred = torch.max(p, dim=1)
            H = _row_entropy_from_logits(logits)

            outputs.append(logits)
            confid.append(conf)
            classes.append(pred)
            entropies.append(H)

        return outputs, confid, classes, entropies

    # Backward-compatible names (if older code called these)
    def forwardMain(self, x: torch.Tensor, adj: torch.Tensor):
        return self.forward_main(x, adj)

    def forward1(self, x: torch.Tensor, adj: torch.Tensor):
        return self.forward_exits(x, adj)
