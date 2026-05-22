from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LastValueBaseline(nn.Module):
    """Repeat the latest observed OD matrix.

    Input:
        x: [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(self, num_nodes: int, input_len: int, pred_len: int, **_: object):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        pred = x[:, -1:].repeat(1, self.pred_len, 1, 1).clamp_min(0.0)
        if return_latent:
            return {"prediction": pred}
        return pred


class HistoricalAverageBaseline(nn.Module):
    """Repeat the average OD matrix over the input window.

    Input:
        x: [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(self, num_nodes: int, input_len: int, pred_len: int, **_: object):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        pred = x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1, 1).clamp_min(0.0)
        if return_latent:
            return {"prediction": pred}
        return pred


class RNNBaseline(nn.Module):
    """LSTM/GRU baseline over flattened OD matrices.

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
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell: str = "gru",
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        flat_dim = num_nodes * num_nodes
        self.input_proj = nn.Linear(flat_dim, d_model)
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * flat_dim),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        bsz = x.shape[0]
        flat = x.reshape(bsz, self.input_len, self.num_nodes * self.num_nodes)
        seq = self.input_proj(flat)
        out, _ = self.rnn(seq)
        context = self.norm(out[:, -1])
        pred = self.head(context).reshape(bsz, self.pred_len, self.num_nodes, self.num_nodes)
        pred = F.softplus(pred)
        if return_latent:
            return {"prediction": pred, "context": context}
        return pred


class TCNBaseline(nn.Module):
    """Temporal convolution baseline over projected flattened OD matrices.

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
        d_model: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        flat_dim = num_nodes * num_nodes
        self.input_proj = nn.Linear(flat_dim, d_model)
        layers: list[nn.Module] = []
        for idx in range(num_layers):
            dilation = 2**idx
            padding = (kernel_size - 1) * dilation
            layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * flat_dim)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        bsz = x.shape[0]
        flat = x.reshape(bsz, self.input_len, self.num_nodes * self.num_nodes)
        seq = self.input_proj(flat).transpose(1, 2)
        hidden = self.tcn(seq)[..., : self.input_len].transpose(1, 2)
        context = self.norm(hidden[:, -1])
        pred = self.head(context).reshape(bsz, self.pred_len, self.num_nodes, self.num_nodes)
        pred = F.softplus(pred)
        if return_latent:
            return {"prediction": pred, "context": context}
        return pred


class FlattenTransformerBaseline(nn.Module):
    """Transformer baseline whose tokens are full flattened OD matrices.

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
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        flat_dim = num_nodes * num_nodes
        self.input_proj = nn.Linear(flat_dim, d_model)
        self.position_embedding = nn.Embedding(input_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_len * flat_dim),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        bsz = x.shape[0]
        flat = x.reshape(bsz, self.input_len, self.num_nodes * self.num_nodes)
        token_ids = torch.arange(self.input_len, device=x.device)
        tokens = self.input_proj(flat) + self.position_embedding(token_ids).unsqueeze(0)
        encoded = self.encoder(tokens)
        context = self.norm(encoded.mean(dim=1))
        pred = self.head(context).reshape(bsz, self.pred_len, self.num_nodes, self.num_nodes)
        pred = F.softplus(pred)
        if return_latent:
            return {"prediction": pred, "encoded_tokens": encoded, "context": context}
        return pred
