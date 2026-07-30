from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _random_walk_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    degree = adj.sum(dim=1).clamp_min(eps)
    return torch.diag(torch.pow(degree, -1.0)) @ adj


def _chebyshev_polynomials(matrix: torch.Tensor, order: int) -> torch.Tensor:
    supports = [torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)]
    if order >= 1:
        supports.append(matrix)
    for _ in range(2, order + 1):
        supports.append(2 * matrix @ supports[-1] - supports[-2])
    return torch.stack(supports, dim=0)


def _build_support(adj: torch.Tensor, kernel_type: str, cheby_order: int) -> torch.Tensor:
    if kernel_type == "localpool":
        return (torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype) + adj).unsqueeze(0)
    if kernel_type == "random_walk_diffusion":
        return _chebyshev_polynomials(_random_walk_normalize(adj).T, cheby_order)
    if kernel_type == "dual_random_walk_diffusion":
        forward = _chebyshev_polynomials(_random_walk_normalize(adj).T, cheby_order)
        backward = _chebyshev_polynomials(_random_walk_normalize(adj.T).T, cheby_order)
        return torch.cat([forward, backward[1:]], dim=0)
    raise ValueError(f"Unsupported ODCRN kernel_type: {kernel_type}")


class ODConv(nn.Module):
    """Origin-destination graph convolution from ODCRN.

    Input:
        x: [B, N, N, C]
        graph: static support [K, N, N] or dynamic pair ([N, N], [N, N])
    Output:
        h: [B, N, N, hidden_dim]
    """

    def __init__(self, k: int, input_dim: int, hidden_dim: int, use_bias: bool = True, activation=None):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.weight = nn.Parameter(torch.empty(input_dim * (k**2), hidden_dim))
        nn.init.xavier_normal_(self.weight)
        if use_bias:
            self.bias = nn.Parameter(torch.empty(hidden_dim))
            nn.init.constant_(self.bias, 0.0)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, graph: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat_set = []
        if isinstance(graph, tuple):
            origin_graph, dest_graph = graph
            origin_support = _chebyshev_polynomials(origin_graph, self.k - 1)
            dest_support = _chebyshev_polynomials(dest_graph, self.k - 1)
        else:
            origin_support = graph
            dest_support = graph

        for o_idx in range(self.k):
            for d_idx in range(self.k):
                mode_1 = torch.einsum("bncl,nm->bmcl", x, origin_support[o_idx])
                mode_2 = torch.einsum("bmcl,cd->bmdl", mode_1, dest_support[d_idx])
                feat_set.append(mode_2)

        features = torch.cat(feat_set, dim=-1)
        output = torch.einsum("bmdk,kh->bmdh", features, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output) if self.activation is not None else output


class ODCRUCell(nn.Module):
    """Origin-destination convolutional recurrent unit from ODCRN."""

    def __init__(
        self,
        num_nodes: int,
        k: int,
        input_dim: int,
        hidden_dim: int,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gates = ODConv(k, input_dim + hidden_dim, hidden_dim * 2, use_bias, activation)
        self.candidate = ODConv(k, input_dim + hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_nodes, self.num_nodes, self.hidden_dim, device=device, dtype=dtype)

    def forward(self, graph: torch.Tensor | tuple[torch.Tensor, torch.Tensor], x_t: torch.Tensor, h_prev: torch.Tensor):
        xh = torch.cat([x_t, h_prev], dim=-1)
        gates = self.gates(x=xh, graph=graph)
        update_raw, reset_raw = torch.split(gates, self.hidden_dim, dim=-1)
        update = torch.sigmoid(update_raw)
        reset = torch.sigmoid(reset_raw)
        candidate = torch.cat([x_t, reset * h_prev], dim=-1)
        candidate = torch.tanh(self.candidate(x=candidate, graph=graph))
        return (1.0 - update) * h_prev + update * candidate


class DynamicGraphConstructor(nn.Module):
    """ODCRN dynamic origin/destination graph constructor."""

    def __init__(self, num_nodes: int):
        super().__init__()
        self.origin_weight = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.dest_weight = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_normal_(self.origin_weight)
        nn.init.xavier_normal_(self.dest_weight)

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dynamic graph scores are quadratic in OD features, so keep this block
        # in fp32 even when the outer trainer uses mixed precision.
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x_fp32 = x_t.float()
            origin_graph = torch.einsum("bpdh,dd,bqdh->pq", x_fp32, self.origin_weight.float(), x_fp32)
            dest_graph = torch.einsum("boeh,oo,bofh->ef", x_fp32, self.dest_weight.float(), x_fp32)
            origin_graph = torch.softmax(torch.relu(origin_graph), dim=1)
            dest_graph = torch.softmax(torch.relu(dest_graph), dim=1)
        return origin_graph.to(dtype=x_t.dtype), dest_graph.to(dtype=x_t.dtype)


class ODCRUEncoder(nn.Module):
    """ODCRN recurrent encoder."""

    def __init__(
        self,
        num_nodes: int,
        k: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                ODCRUCell(
                    num_nodes=num_nodes,
                    k=k,
                    input_dim=input_dim if layer_idx == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    use_bias=use_bias,
                    activation=activation,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, graph_list: list, x_seq: torch.Tensor) -> list[torch.Tensor]:
        batch_size = x_seq.shape[0]
        hidden = [
            cell.init_hidden(batch_size, device=x_seq.device, dtype=x_seq.dtype)
            for cell in self.cells
        ]
        layer_input = x_seq
        last_hidden = []
        for layer_idx, cell in enumerate(self.cells):
            h_t = hidden[layer_idx]
            outputs = []
            for time_idx in range(x_seq.shape[1]):
                graph = graph_list[layer_idx](layer_input[:, time_idx]) if callable(graph_list[layer_idx]) else graph_list[layer_idx]
                h_t = cell(graph=graph, x_t=layer_input[:, time_idx], h_prev=h_t)
                outputs.append(h_t)
            layer_input = torch.stack(outputs, dim=1)
            last_hidden.append(h_t)
        return last_hidden


class ODCRUDecoder(nn.Module):
    """ODCRN recurrent decoder."""

    def __init__(
        self,
        num_nodes: int,
        k: int,
        output_dim: int,
        hidden_dim: int,
        pred_len: int,
        num_layers: int,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.cells = nn.ModuleList(
            [
                ODCRUCell(
                    num_nodes=num_nodes,
                    k=k,
                    input_dim=output_dim if layer_idx == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    use_bias=use_bias,
                    activation=activation,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(
        self,
        graph_list: list,
        decoder_input: torch.Tensor,
        hidden: list[torch.Tensor],
        output_layer: nn.Module,
    ) -> torch.Tensor:
        outputs = []
        for _ in range(self.pred_len):
            layer_input = decoder_input
            next_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                graph = graph_list[layer_idx](layer_input) if callable(graph_list[layer_idx]) else graph_list[layer_idx]
                h_t = cell(graph=graph, x_t=layer_input, h_prev=hidden[layer_idx])
                next_hidden.append(h_t)
                layer_input = h_t
            hidden = next_hidden
            decoder_input = output_layer(layer_input)
            outputs.append(decoder_input)
        return torch.stack(outputs, dim=1)


class ODCRNCore(nn.Module):
    """Official ODCRN encoder-decoder core."""

    def __init__(
        self,
        num_nodes: int,
        k: int,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        num_layers: int,
        use_dynamic_graph: bool = True,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.use_dynamic_graph = use_dynamic_graph
        self.dynamic_graph = DynamicGraphConstructor(num_nodes) if use_dynamic_graph else None
        self.encoder = ODCRUEncoder(num_nodes, k, input_dim, hidden_dim, num_layers, use_bias, activation)
        self.decoder = ODCRUDecoder(num_nodes, k, input_dim, hidden_dim, pred_len, num_layers, use_bias, activation)
        self.linear = nn.Linear(hidden_dim, input_dim, bias=use_bias)

    def _graphs(self, support: torch.Tensor, num_layers: int) -> list:
        if self.use_dynamic_graph and self.dynamic_graph is not None:
            return [self.dynamic_graph] + [support for _ in range(num_layers - 1)]
        return [support for _ in range(num_layers)]

    def forward(self, support: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        graph_list = self._graphs(support, self.encoder.num_layers)
        hidden = self.encoder(graph_list=graph_list, x_seq=x_seq)
        decoder_input = torch.zeros_like(x_seq[:, 0])
        return self.decoder(
            graph_list=graph_list,
            decoder_input=decoder_input,
            hidden=hidden,
            output_layer=self.linear,
        )


class ODCRNBaseline(nn.Module):
    """Official-structure ODCRN baseline adapted to `[B, L, N, N]`.

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
        hidden_dim: int = 32,
        num_layers: int = 2,
        cheby_order: int = 2,
        kernel_type: str = "random_walk_diffusion",
        static_graph_mode: str = "batch_od",
        use_dynamic_graph: bool = True,
        output_activation: str = "softplus",
        softplus_beta: float = 10.0,
        input_transform: str = "log1p",
        inverse_output: bool = True,
        log_output_max: float = 8.0,
        **_: object,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.cheby_order = cheby_order
        self.kernel_type = kernel_type
        self.static_graph_mode = static_graph_mode
        self.output_activation = output_activation
        self.softplus_beta = softplus_beta
        self.input_transform = input_transform
        self.inverse_output = inverse_output
        self.log_output_max = log_output_max
        if kernel_type == "dual_random_walk_diffusion":
            support_k = cheby_order * 2 + 1
        elif kernel_type == "localpool":
            support_k = 1
        else:
            support_k = cheby_order + 1
        self.core = ODCRNCore(
            num_nodes=num_nodes,
            k=support_k,
            input_dim=1,
            hidden_dim=hidden_dim,
            pred_len=pred_len,
            num_layers=num_layers,
            use_dynamic_graph=use_dynamic_graph,
            activation=None,
        )

    def _static_adj(self, x: torch.Tensor) -> torch.Tensor:
        if self.static_graph_mode == "identity":
            return torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)
        if self.static_graph_mode == "fully_connected":
            return torch.ones(self.num_nodes, self.num_nodes, device=x.device, dtype=x.dtype)
        if self.static_graph_mode == "batch_od":
            flow = x.mean(dim=(0, 1))
            adj = flow + flow.T
            adj = adj / adj.max().clamp_min(1.0)
            return adj + torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)
        raise ValueError(f"Unsupported static_graph_mode: {self.static_graph_mode}")

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

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_transform == "none":
            return x
        if self.input_transform == "log1p":
            return torch.log1p(x.clamp_min(0.0))
        raise ValueError(f"Unsupported input_transform: {self.input_transform}")

    def _inverse_output(self, pred: torch.Tensor) -> torch.Tensor:
        if not self.inverse_output:
            return pred
        if self.input_transform == "log1p":
            return torch.expm1(pred.clamp(min=0.0, max=self.log_output_max)).clamp_min(0.0)
        return pred

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        x_model = self._transform_input(x)
        support = _build_support(self._static_adj(x_model), self.kernel_type, self.cheby_order)
        output = self.core(support=support, x_seq=x_model.unsqueeze(-1)).squeeze(-1)
        pred = self._inverse_output(self._activate(output))
        if return_latent:
            return {"prediction": pred, "support": support}
        return pred
