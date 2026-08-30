from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv


class GraphClassifier(nn.Module):
    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        raise NotImplementedError


class GCN(GraphClassifier):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, hidden_channels), GCNConv(hidden_channels, out_channels)]
        )
        self.dropout = dropout

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.convs[0](x, edge_index).relu()
        return nn.functional.dropout(x, p=self.dropout, training=self.training)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.convs[1](self.embed(x, edge_index), edge_index)


class GraphAttentionNetwork(GraphClassifier):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        heads: int,
        layers: int,
        dropout: float,
        v2: bool,
    ):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least two")
        conv = GATv2Conv if v2 else GATConv
        per_head = max(1, hidden_channels // heads)
        self.input = conv(in_channels, per_head, heads=heads, dropout=dropout, add_self_loops=True)
        actual_hidden = per_head * heads
        self.hidden = nn.ModuleList(
            conv(actual_hidden, per_head, heads=heads, dropout=dropout, add_self_loops=True)
            for _ in range(layers - 2)
        )
        self.output = conv(actual_hidden, out_channels, heads=1, concat=False, dropout=dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(actual_hidden) for _ in range(layers - 1))
        self.dropout = dropout

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = nn.functional.elu(self.input(x, edge_index))
        x = self.norms[0](x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for index, layer in enumerate(self.hidden, start=1):
            residual = x
            x = nn.functional.elu(layer(x, edge_index))
            x = self.norms[index](x + residual)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.output(self.embed(x, edge_index), edge_index)


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    *,
    heads: int = 8,
    layers: int = 3,
    dropout: float = 0.5,
) -> GraphClassifier:
    key = name.lower()
    if key == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    if key in {"gat", "gatv2"}:
        return GraphAttentionNetwork(
            in_channels,
            hidden_channels,
            out_channels,
            heads=heads,
            layers=layers,
            dropout=dropout,
            v2=key == "gatv2",
        )
    raise ValueError("model must be one of: gcn, gat, gatv2")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
