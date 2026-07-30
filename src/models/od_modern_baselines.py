from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(eps)


def _dynamic_od_graph(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build origin and destination transition graphs from recent OD history.

    Args:
        x: OD history, shape [B, L, N, N].

    Returns:
        origin_graph: [B, N, N], row-normalized origin interaction graph.
        dest_graph: [B, N, N], row-normalized destination interaction graph.
    """
    flow = x.mean(dim=1)
    eye = torch.eye(flow.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
    origin_graph = _row_normalize(flow + eye)
    dest_graph = _row_normalize(flow.transpose(-1, -2) + eye)
    return origin_graph, dest_graph


class CSTNBaseline(nn.Module):
    """CSTN-style OD baseline with origin/destination local spatial views.

    This is a compact PyTorch adaptation for OD matrices. It keeps the main CSTN
    idea: local spatial context from origin and destination views plus a global
    correlation gate.

    Input:
        x: [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len

        def conv_stack() -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(input_len, d_model, kernel_size=3, padding=1), nn.GELU()]
            for _ in range(max(num_layers - 1, 0)):
                layers.extend(
                    [
                        nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                        nn.BatchNorm2d(d_model),
                        nn.GELU(),
                        nn.Dropout2d(dropout),
                    ]
                )
            return nn.Sequential(*layers)

        self.origin_view = conv_stack()
        self.dest_view = conv_stack()
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * d_model, max(d_model // 2, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(d_model // 2, 8), 2 * d_model, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, pred_len, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        origin_feat = self.origin_view(x)
        dest_feat = self.dest_view(x.transpose(-1, -2)).transpose(-1, -2)
        fused = torch.cat([origin_feat, dest_feat], dim=1)
        fused = fused * self.global_gate(fused)
        pred = F.softplus(self.fuse(fused))
        if return_latent:
            return {"prediction": pred, "features": fused}
        return pred


class GEMLBaseline(nn.Module):
    """GEML-style graph enhanced OD baseline.

    The model derives dynamic origin/destination graphs from recent OD matrices,
    applies dual-view graph propagation, and predicts each OD pair with a shared
    pair-level head.

    Input:
        x: [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.value_proj = nn.Linear(input_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        origin_graph, dest_graph = _dynamic_od_graph(x)
        hidden = self.value_proj(x.permute(0, 2, 3, 1))
        for layer in self.layers:
            origin_msg = torch.einsum("boi,bijf->bojf", origin_graph, hidden)
            dest_msg = torch.einsum("boj,bijf->biof", dest_graph, hidden)
            hidden = hidden + layer(torch.cat([hidden, origin_msg, dest_msg], dim=-1))
        pred = self.head(hidden).permute(0, 3, 1, 2)
        pred = F.softplus(pred)
        if return_latent:
            return {"prediction": pred, "hidden": hidden}
        return pred


class ODSTGCNBaseline(nn.Module):
    """OD-STGCN/DBSTNet-inspired baseline.

    It uses a shared temporal GRU over each OD pair, then applies multi-hop
    origin and destination graph propagation over dynamic OD graphs.

    Input:
        x: [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        d_model: int = 64,
        num_layers: int = 2,
        cheb_order: int = 2,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.cheb_order = max(int(cheb_order), 1)
        self.temporal = nn.GRU(input_size=1, hidden_size=d_model, num_layers=1, batch_first=True)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear((1 + 2 * self.cheb_order) * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def _multi_hop(self, graph: torch.Tensor, hidden: torch.Tensor, along: str) -> list[torch.Tensor]:
        states = []
        current = hidden
        for _ in range(self.cheb_order):
            if along == "origin":
                current = torch.einsum("boi,bijf->bojf", graph, current)
            elif along == "dest":
                current = torch.einsum("boj,bijf->biof", graph, current)
            else:
                raise ValueError(f"Unsupported graph direction: {along}")
            states.append(current)
        return states

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        bsz = x.shape[0]
        pair_seq = x.permute(0, 2, 3, 1).reshape(bsz * self.num_nodes * self.num_nodes, self.input_len, 1)
        _, last_hidden = self.temporal(pair_seq)
        hidden = last_hidden[-1].reshape(bsz, self.num_nodes, self.num_nodes, -1)

        origin_graph, dest_graph = _dynamic_od_graph(x)
        for block in self.blocks:
            origin_states = self._multi_hop(origin_graph, hidden, "origin")
            dest_states = self._multi_hop(dest_graph, hidden, "dest")
            hidden = hidden + block(torch.cat([hidden, *origin_states, *dest_states], dim=-1))

        pred = self.head(hidden).permute(0, 3, 1, 2)
        pred = F.softplus(pred)
        if return_latent:
            return {"prediction": pred, "hidden": hidden}
        return pred
