"""Interaction-aware message-passing GNN predicting pK from the protein-ligand complex graph.

Uses distances only (see data/graph.py), so predictions are invariant to rotating/translating the complex.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

from ..data.graph import EDGE_DIM, NODE_DIM


class InteractionLayer(MessagePassing):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__(aggr="add")
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.upd(torch.cat([x, agg], dim=-1)))

    def message(self, x_i, x_j, edge_attr):
        return self.msg(torch.cat([x_i, x_j, edge_attr], dim=-1))


class AffinityGNN(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4, dropout: float = 0.1,
                 node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(InteractionLayer(hidden, edge_dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, data) -> torch.Tensor:
        h = self.embed(data.x)
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        n_graphs = int(data.batch.max()) + 1 if data.batch is not None else 1
        batch = data.batch if data.batch is not None else torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        lig = data.is_ligand.float().unsqueeze(-1)
        count = global_add_pool(lig, batch, size=n_graphs).clamp(min=1.0)
        lig_pool = global_add_pool(h * lig, batch, size=n_graphs) / count
        all_pool = global_mean_pool(h, batch, size=n_graphs)
        return self.head(torch.cat([lig_pool, all_pool], dim=-1)).squeeze(-1)
