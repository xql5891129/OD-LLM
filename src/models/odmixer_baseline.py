from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleInteractModule(nn.Module):
    """Bidirectional trend interaction block from ODMixer.

    Input:
        x: [B, N, N, D]
        y: [B, N, N, D]
    Output:
        output: [B, N, N, D]
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.cross_mixer = nn.Conv1d(2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(shape[0], -1)
        y_flat = y.reshape(shape[0], -1)
        mixed = torch.cat([x_flat.unsqueeze(-1), y_flat.unsqueeze(-1)], dim=-1)
        mixed = self.cross_mixer(mixed.permute(0, 2, 1)).squeeze(1).reshape(shape)
        gate = torch.sigmoid(self.gate_layer(mixed))
        return self.value_layer(x) * gate + x


class BidirectionalTrendLearner(nn.Module):
    """ODMixer bidirectional trend learner between current and previous OD histories."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.up_interact = SingleInteractModule(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.down_interact = SingleInteractModule(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, prev_feat: torch.Tensor, od_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.up_interact(prev_feat, od_feat), self.down_interact(od_feat, prev_feat)


class MixerLayer(nn.Module):
    """MLP mixer layer used by ODMixer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ChannelMixer(nn.Module):
    """ODMixer channel mixer for OD-pair temporal representations.

    Input:
        x: [B, N, N, D]
    Output:
        output: [B, N, N, D]
    """

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mixer = MixerLayer(hidden_dim, 2 * hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm([num_nodes, num_nodes, hidden_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mixer(x) + x)


class MultiViewMixer(nn.Module):
    """ODMixer multi-view mixer over origin and destination dimensions.

    Input:
        x: [B, N, N, D]
    Output:
        output: [B, N, N, D]
    """

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.origin_mixer = MixerLayer(num_nodes, 2 * hidden_dim, num_nodes, dropout)
        self.dest_mixer = MixerLayer(num_nodes, 2 * hidden_dim, num_nodes, dropout)
        self.norm = nn.LayerNorm([num_nodes, num_nodes, hidden_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_feat = self.origin_mixer(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        dest_feat = self.dest_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.norm(origin_feat + dest_feat + x)


class ODMixerInteractionModule(nn.Module):
    """One official ODMixer ODIM block: Channel Mixer followed by Multi-view Mixer."""

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.channel_mixer = ChannelMixer(num_nodes, hidden_dim, dropout)
        self.multi_view_mixer = MultiViewMixer(num_nodes, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multi_view_mixer(self.channel_mixer(x))


class ODMixerBaseline(nn.Module):
    """Official-structure ODMixer baseline adapted to the project trainer.

    The mixer stack follows the official implementation:
    OD pair temporal embedding -> ODIM blocks -> Bidirectional Trend Learners.

    Input:
        x: [B, L, N, N]
        prev_x: optional previous-cycle OD history, [B, L, N, N]
    Output:
        y_pred: [B, H, N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int = 16,
        layer_nums: int = 5,
        dropout: float = 0.1,
        prev_od_mode: str = "lag1",
        output_activation: str = "relu",
        softplus_beta: float = 10.0,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.prev_od_mode = prev_od_mode
        self.output_activation = output_activation
        self.softplus_beta = softplus_beta

        self.emb_layer = nn.Linear(input_len, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [ODMixerInteractionModule(num_nodes, hidden_dim, dropout) for _ in range(layer_nums)]
        )
        self.trend_layers = nn.ModuleList([BidirectionalTrendLearner(hidden_dim, dropout) for _ in range(layer_nums)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.PReLU(),
            nn.Linear(max(hidden_dim // 2, 1), pred_len),
        )

    def _fallback_prev_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.prev_od_mode == "same":
            return x
        if self.prev_od_mode == "mean":
            return x.mean(dim=1, keepdim=True).expand_as(x)
        if self.prev_od_mode == "lag1":
            return torch.cat([x[:, :1], x[:, :-1]], dim=1)
        raise ValueError(f"Unsupported prev_od_mode: {self.prev_od_mode}")

    def _activate(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "none":
            return pred
        if self.output_activation == "relu":
            return F.relu(pred)
        if self.output_activation == "softplus":
            return F.softplus(pred, beta=self.softplus_beta)
        if self.output_activation in {"softplus_shift", "shifted_softplus"}:
            zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
            offset = F.softplus(zero, beta=self.softplus_beta)
            return (F.softplus(pred, beta=self.softplus_beta) - offset).clamp_min(0.0)
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")

    def _encode_branch(self, od: torch.Tensor) -> torch.Tensor:
        # [B, L, N, N] -> [B, N, N, L] -> [B, N, N, D]
        return self.emb_layer(od.permute(0, 2, 3, 1))

    def _project(self, feat: torch.Tensor) -> torch.Tensor:
        # [B, N, N, D] -> [B, N, N, H] -> [B, H, N, N]
        return self.output_layer(feat).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, prev_x: torch.Tensor | None = None, return_auxiliary: bool = False):
        if prev_x is None:
            prev_x = self._fallback_prev_x(x)

        od_feat = self._encode_branch(x)
        prev_feat = self._encode_branch(prev_x)

        for encoder_layer, trend_layer in zip(self.encoder_layers, self.trend_layers):
            od_feat = encoder_layer(od_feat)
            prev_feat = encoder_layer(prev_feat)
            prev_feat, od_feat = trend_layer(prev_feat, od_feat)

        pred = self._activate(self._project(od_feat))
        if not return_auxiliary:
            return pred

        prev_pred = self._activate(self._project(prev_feat))
        return {
            "prediction": pred,
            "prev_prediction": prev_pred,
            "od_feat": od_feat,
            "prev_feat": prev_feat,
        }
