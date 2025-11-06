# layers.py
# Core graph layers used by EEGNN (ISDA 2023)
# - GraphConv: dense/sparse adjacency support
# - GraphAttConv / GraphAttConvOneHead: sparse GAT-style attention
# - PairNorm: PN / PN-SI / PN-SCS
# - Sparse segment softmax (torch_scatter-based)
from typing import Optional

import math
import torch
import torch.nn as nn
import numpy as np

# Optional sparse backends (install via PyG wheels)
from torch_scatter import scatter_max, scatter_add
try:
    from torch_sparse import spmm as sparse_matmul  # COO spmm(edge_index, value, M, N, mat)
except Exception:
    sparse_matmul = None  # Will fallback to torch.sparse.mm when possible

__all__ = [
    "GraphConv",
    "GraphAttConv",
    "GraphAttConvOneHead",
    "PairNorm",
    "softmax",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _matmul_adj(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Adjacency–feature multiplication supporting dense or sparse adj."""
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


def _xavier_uniform_param(p: torch.Tensor, gain: float = 1.0):
    nn.init.xavier_uniform_(p, gain=gain)


# ---------------------------------------------------------------------------
# GraphConv (dense/sparse adjacency)
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """
    Simple GCN layer with dense/sparse adjacency support:
        H = A (X W) + b
    Adjacency A is expected to be pre-normalized (e.g., row-normalized).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias: Optional[torch.Tensor]
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        _xavier_uniform_param(self.weight)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x @ self.weight                        # [N, out_features]
        out = _matmul_adj(adj, h)                  # [N, out_features]
        return out + self.bias if self.bias is not None else out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}->{self.out_features})"


# ---------------------------------------------------------------------------
# Attention GCN (sparse)
# ---------------------------------------------------------------------------

class GraphAttConvOneHead(nn.Module):
    """
    Sparse GAT-style layer (single head).
    Expects COO sparse adjacency in torch.sparse_coo_tensor format.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, 2 * out_features))
        _xavier_uniform_param(self.weight, gain=nn.init.calculate_gain("relu"))
        _xavier_uniform_param(self.a, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if not adj.is_sparse:
            raise ValueError("GraphAttConv requires a sparse (COO) adjacency tensor.")

        edge_index = adj._indices()                # [2, E]
        N = x.size(0)

        h = x @ self.weight                        # [N, F']
        # build edge-wise features: concat(h_i, h_j)
        edge_h = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1).t()  # [2F', E]
        e = self.leakyrelu(self.a @ edge_h).squeeze(0)                       # [E]

        # normalized attention scores per source node
        alpha = softmax(e, edge_index[0], num_nodes=N)                       # [E]
        alpha = self.dropout(alpha)

        if sparse_matmul is not None:
            out = sparse_matmul(edge_index, alpha, N, N, h)                  # [N, F']
        else:
            # fallback: build a sparse tensor (coalesced)
            A_alpha = torch.sparse_coo_tensor(edge_index, alpha, (N, N), device=h.device).coalesce()
            out = torch.sparse.mm(A_alpha, h)

        return out


class GraphAttConv(nn.Module):
    """
    Multi-head sparse attention GCN.
    out_features must be divisible by heads.
    """
    def __init__(self, in_features: int, out_features: int, heads: int, dropout: float):
        super().__init__()
        assert out_features % heads == 0, "out_features must be divisible by heads"
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.out_per_head = out_features // heads

        self.graph_atts = nn.ModuleList([
            GraphAttConvOneHead(in_features, self.out_per_head, dropout=dropout)
            for _ in range(heads)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = [att(x, adj) for att in self.graph_atts]                         # list of [N, F'/H]
        out = torch.cat(z, dim=1)                                            # [N, F']
        # original GAT uses ELU after concat; leave activation to caller for flexibility
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}->[{self.heads}x{self.out_per_head}])"


# ---------------------------------------------------------------------------
# PairNorm
# ---------------------------------------------------------------------------

class PairNorm(nn.Module):
    """
    PairNorm (Zhao & Akoglu)
    Modes:
      - 'None'  : identity
      - 'PN'    : center columns, then scale by mean row L2 norm
      - 'PN-SI' : center columns, then scale each row individually
      - 'PN-SCS': scale each row individually, then subtract column mean
    """
    def __init__(self, mode: str = "PN", scale: float = 1.0):
        super().__init__()
        assert mode in ["None", "PN", "PN-SI", "PN-SCS"], "Unsupported PairNorm mode"
        self.mode = mode
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "None":
            return x

        eps = 1e-6
        col_mean = x.mean(dim=0, keepdim=False)

        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (eps + x.pow(2).sum(dim=1).mean()).sqrt()
            return self.scale * x / rownorm_mean

        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_ind = (eps + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            return self.scale * x / rownorm_ind

        # PN-SCS
        rownorm_ind = (eps + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        return self.scale * x / rownorm_ind - col_mean


# ---------------------------------------------------------------------------
# Sparse softmax (segment softmax over index)
# ---------------------------------------------------------------------------

def softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    """
    Segment softmax over groups identified by `index`.
    Args:
        src:     [E] unnormalized scores (one per edge or element)
        index:   [E] group ids (typically source nodes)
        num_nodes: number of groups; defaults to max(index)+1
    Returns:
        [E] normalized scores s.t. for each group g: sum_{i in g} softmax(i) = 1
    """
    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1

    # subtract groupwise max for numerical stability
    max_per_group = scatter_max(src, index, dim=0, dim_size=num_nodes)[0]
    out = src - max_per_group[index]
    out = out.exp()
    denom = scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16
    return out / denom
