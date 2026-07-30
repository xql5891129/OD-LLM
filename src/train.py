from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ODDataset
from data.line_dataset import LineODDataset
from losses.sparse_od_loss import sparse_od_loss
from models.od_llm import ODLLM
from models.odcrn_baseline import ODCRNBaseline
from models.od_modern_baselines import CSTNBaseline, GEMLBaseline, ODSTGCNBaseline
from models.odmixer_baseline import ODMixerBaseline
from models.simple_baselines import (
    FlattenTransformerBaseline,
    HistoricalAverageBaseline,
    LastValueBaseline,
    RNNBaseline,
    TCNBaseline,
)
from models.transformer_baseline import ODTensorTransformer
from utils.config import load_config, prepare_output_dirs
from utils.metrics import ODMetricAccumulator, compute_metrics, format_metrics
from utils.seed import get_device, set_seed


def build_dataset(data_cfg: dict, split: str):
    dataset_type = str(data_cfg.get("dataset_type", "od")).lower()
    if dataset_type in {"line", "line_bus", "bus_line"}:
        return LineODDataset(data_cfg, split)
    return ODDataset(data_cfg, split)


def load_projection_init(model_cfg: dict, num_nodes: int) -> tuple[str, tuple[torch.Tensor, torch.Tensor] | None]:
    """Load optional precomputed tokenizer bases without touching checkpoint warm starts."""
    mode = str(model_cfg.get("projection_mode", "learnable")).lower()
    if mode == "learnable":
        return mode, None
    if mode not in {"svd_fixed", "svd_trainable"}:
        raise ValueError(
            "model.projection_mode must be one of learnable, svd_fixed, or svd_trainable; "
            f"got {mode!r}"
        )

    path_value = model_cfg.get("projection_path")
    if not path_value:
        raise ValueError(f"model.projection_path is required for projection_mode={mode}.")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Precomputed projection file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "Po" not in payload or "Pd" not in payload:
        raise ValueError(f"Projection file must contain tensor keys 'Po' and 'Pd': {path}")
    po = torch.as_tensor(payload["Po"], dtype=torch.float32)
    pd = torch.as_tensor(payload["Pd"], dtype=torch.float32)
    rank = int(model_cfg.get("rank", 4))
    expected_shape = (num_nodes, rank)
    if tuple(po.shape) != expected_shape or tuple(pd.shape) != expected_shape:
        raise ValueError(
            f"Projection file {path} is incompatible with model dimensions: expected {expected_shape}, "
            f"got Po={tuple(po.shape)}, Pd={tuple(pd.shape)}"
        )
    print(f"Loaded {mode} projection bases from {path}")
    return mode, (po, pd)


def build_dataloader(dataset, cfg: dict, shuffle: bool, device: torch.device) -> DataLoader:
    train_cfg = cfg["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", device.type == "cuda")),
        "drop_last": bool(train_cfg.get("drop_last", False)) if shuffle else False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def resolve_model_time_feature_indices(
    cfg: dict,
    columns: list[str] | tuple[str, ...] | None,
    feature_dim: int,
) -> list[int] | None:
    """Select model-only time/context features while leaving metrics unchanged."""
    model_cfg = cfg.get("model", {})
    mode = model_cfg.get("time_features_mode")
    selected_columns = model_cfg.get("time_feature_columns")
    if mode is None and not selected_columns:
        return None

    names = list(columns or [f"feature_{idx}" for idx in range(feature_dim)])
    mode = str(mode or "selected").lower()
    if mode in {"all", "full"}:
        return None
    if mode in {"calendar", "time", "time_only"}:
        return list(range(min(5, feature_dim)))
    if mode in {"core_weather", "weather_core"}:
        keep = {
            "weather_temperature_c_z",
            "weather_precip_mm_z",
            "weather_wind_speed_ms_z",
            "weather_is_rainy",
        }
        wanted = [*names[: min(5, len(names))], *[name for name in names[5:] if name in keep]]
    elif mode in {"selected", "custom"}:
        if not selected_columns:
            raise ValueError("model.time_features_mode=selected requires model.time_feature_columns.")
        wanted = list(selected_columns)
    else:
        raise ValueError(f"Unsupported model.time_features_mode: {mode}")

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in wanted if name not in name_to_idx]
    if missing:
        raise ValueError(f"Model time feature columns not found: {missing}")
    return [name_to_idx[name] for name in wanted]


def select_model_time_features(tensor: torch.Tensor, cfg: dict) -> torch.Tensor:
    indices = cfg.get("_runtime", {}).get("model_time_feature_indices")
    if indices is None:
        return tensor
    index = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(dim=-1, index=index)


def read_history_state(history_path: Path, monitor_metric: str) -> tuple[int, float, int]:
    if not history_path.exists():
        return 0, float("inf"), 0

    last_epoch = 0
    best_val = float("inf")
    best_epoch = 0
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = int(record.get("epoch", 0))
            last_epoch = max(last_epoch, epoch)
            if record.get("monitor_metric", monitor_metric) == monitor_metric:
                value = float(record.get("monitor_value", float("inf")))
                if value < best_val:
                    best_val = value
                    best_epoch = epoch
    bad_epochs = max(last_epoch - best_epoch, 0) if best_epoch > 0 else 0
    return last_epoch, best_val, bad_epochs


def resolve_resume_checkpoint(cfg: dict, ckpt_dir: Path) -> Path | None:
    resume_value = cfg["train"].get("resume_checkpoint", cfg["train"].get("resume", False))
    if isinstance(resume_value, bool):
        if not resume_value:
            return None
        candidates = [ckpt_dir / "last.pt", ckpt_dir / "best.pt"]
    else:
        text = str(resume_value).strip()
        if text.lower() in {"", "false", "none", "no", "0"}:
            return None
        if text.lower() in {"true", "auto"}:
            candidates = [ckpt_dir / "last.pt", ckpt_dir / "best.pt"]
        elif text.lower() in {"best", "auto_best"}:
            candidates = [ckpt_dir / "best.pt", ckpt_dir / "last.pt"]
        else:
            path = Path(text)
            candidates = [path if path.is_absolute() else Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_init_checkpoint(cfg: dict) -> Path | None:
    value = cfg["train"].get("init_checkpoint")
    if value is None or str(value).strip().lower() in {"", "false", "none", "0"}:
        return None
    text = str(value).strip()
    if text.lower() in {"previous_best", "auto_previous"}:
        previous_experiment = str(cfg["train"].get("init_experiment", "od_llm"))
        output_root = Path(cfg.get("outputs", {}).get("root", "outputs"))
        candidate = output_root / previous_experiment / "checkpoints" / "best.pt"
    else:
        path = Path(text)
        candidate = path if path.is_absolute() else Path.cwd() / path
    return candidate if candidate.exists() else None


def load_matching_model_state(model: torch.nn.Module, checkpoint_path: Path) -> tuple[int, int]:
    state = torch.load(checkpoint_path, map_location="cpu")
    source = state["model"]
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    model.load_state_dict(matched, strict=False)
    return len(matched), len(source) - len(matched)


def build_model(
    cfg: dict,
    num_nodes: int,
    time_feature_dim: int | None = None,
    poi_features=None,
    poi_feature_dim: int = 0,
) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "od_tensor_transformer")
    use_poi_features = bool(model_cfg.get("use_poi_features", False))
    dynamic_poi_features = bool(model_cfg.get("dynamic_poi_features", False))
    poi_tensor = None if poi_features is None else torch.as_tensor(poi_features, dtype=torch.float32)
    if use_poi_features and poi_tensor is None and not dynamic_poi_features:
        raise ValueError("model.use_poi_features=true requires data.use_poi_features=true and poi_features.npy.")
    projection_mode, projection_init = load_projection_init(model_cfg, num_nodes)
    common = dict(
        num_nodes=num_nodes,
        input_len=int(data_cfg["input_len"]),
        pred_len=int(data_cfg["pred_len"]),
        rank=int(model_cfg.get("rank", 4)),
        d_model=int(model_cfg["d_model"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        max_tokens=int(model_cfg.get("max_tokens", 4096)),
        decoder_mode=str(model_cfg.get("decoder_mode", "low_rank")),
        decoder_scale=float(model_cfg.get("decoder_scale", 1.0)),
        zero_init_decoder=bool(model_cfg.get("zero_init_decoder", False)),
        use_pair_trend_head=bool(model_cfg.get("use_pair_trend_head", False)),
        pair_trend_mode=str(model_cfg.get("pair_trend_mode", "shared")),
        pair_trend_scale=float(model_cfg.get("pair_trend_scale", 1.0)),
        zero_init_pair_trend=bool(model_cfg.get("zero_init_pair_trend", True)),
        residual_activation=str(model_cfg.get("residual_activation", "relu")),
        softplus_beta=float(model_cfg.get("softplus_beta", 10.0)),
        seasonal_blend_init=float(model_cfg.get("seasonal_blend_init", 0.7)),
        learnable_seasonal_blend=bool(model_cfg.get("learnable_seasonal_blend", True)),
        context_pooling=str(model_cfg.get("context_pooling", "mean")),
        horizon_attention_heads=model_cfg.get("horizon_attention_heads"),
        poi_features=poi_tensor,
        poi_feature_dim=int(poi_feature_dim),
        use_poi_features=use_poi_features,
        poi_projection_scale=float(model_cfg.get("poi_projection_scale", 0.1)),
    )
    if name == "od_tensor_transformer":
        return ODTensorTransformer(
            **common,
            n_heads=int(model_cfg["n_heads"]),
            num_layers=int(model_cfg["num_layers"]),
        )
    if name == "last_value":
        return LastValueBaseline(**common)
    if name == "historical_average":
        return HistoricalAverageBaseline(**common)
    if name in {"lstm_baseline", "gru_baseline"}:
        return RNNBaseline(
            **common,
            cell="lstm" if name == "lstm_baseline" else "gru",
            num_layers=int(model_cfg.get("num_layers", 2)),
        )
    if name == "tcn_baseline":
        return TCNBaseline(
            **common,
            num_layers=int(model_cfg.get("num_layers", 3)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
        )
    if name == "transformer_flatten":
        return FlattenTransformerBaseline(
            **common,
            n_heads=int(model_cfg.get("n_heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
        )
    if name == "cstn_baseline":
        return CSTNBaseline(
            **common,
            num_layers=int(model_cfg.get("num_layers", 3)),
        )
    if name == "geml_baseline":
        return GEMLBaseline(
            **common,
            num_layers=int(model_cfg.get("num_layers", 2)),
        )
    if name == "od_stgcn_baseline":
        return ODSTGCNBaseline(
            **common,
            num_layers=int(model_cfg.get("num_layers", 2)),
            cheb_order=int(model_cfg.get("cheb_order", 2)),
        )
    if name == "odmixer":
        return ODMixerBaseline(
            **common,
            hidden_dim=int(model_cfg.get("hidden_dim", model_cfg.get("d_model", 16))),
            layer_nums=int(model_cfg.get("layer_nums", model_cfg.get("num_layers", 5))),
            prev_od_mode=str(model_cfg.get("prev_od_mode", "lag1")),
            output_activation=str(model_cfg.get("output_activation", "relu")),
        )
    if name == "odcrn":
        return ODCRNBaseline(
            **common,
            hidden_dim=int(model_cfg.get("hidden_dim", model_cfg.get("d_model", 32))),
            num_layers=int(model_cfg.get("num_layers", 2)),
            cheby_order=int(model_cfg.get("cheby_order", 2)),
            kernel_type=str(model_cfg.get("kernel_type", "random_walk_diffusion")),
            static_graph_mode=str(model_cfg.get("static_graph_mode", "batch_od")),
            use_dynamic_graph=bool(model_cfg.get("use_dynamic_graph", True)),
            output_activation=str(model_cfg.get("output_activation", "softplus_shift")),
            input_transform=str(model_cfg.get("input_transform", "log1p")),
            inverse_output=bool(model_cfg.get("inverse_output", True)),
            log_output_max=float(model_cfg.get("log_output_max", 8.0)),
        )
    if name in {"od_llm", "od_llm_gpt2"}:
        return ODLLM(
            **common,
            llm_model=str(model_cfg.get("llm_model", "mini_gpt")),
            llm_dim=int(model_cfg.get("llm_dim", 768)),
            llm_layers=int(model_cfg.get("llm_layers", 6)),
            llm_heads=int(model_cfg.get("llm_heads", model_cfg.get("n_heads", 8))),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_path=model_cfg.get("pretrained_path"),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            attn_implementation=model_cfg.get("attn_implementation"),
            freeze_llm=bool(model_cfg.get("freeze_llm", True)),
            llm_trainable_layers=int(model_cfg.get("llm_trainable_layers", 0)),
            llm_trainable_layer_norm=bool(model_cfg.get("llm_trainable_layer_norm", False)),
            llm_trainable_fp32=bool(model_cfg.get("llm_trainable_fp32", True)),
            llm_gradient_checkpointing=bool(model_cfg.get("llm_gradient_checkpointing", False)),
            use_reprogramming=bool(model_cfg.get("use_reprogramming", False)),
            num_virtual_prompt_tokens=int(model_cfg.get("num_virtual_prompt_tokens", 8)),
            num_source_tokens=int(model_cfg.get("num_source_tokens", 1000)),
            use_time_features=bool(model_cfg.get("use_time_features", False)),
            time_feature_dim=int(model_cfg.get("time_feature_dim", time_feature_dim or 0)),
            token_value_transform=str(model_cfg.get("token_value_transform", "none")),
            projection_mode=projection_mode,
            projection_init=projection_init,
            delta_limit=float(model_cfg.get("delta_limit", 0.0)),
            use_context_base_gate=bool(model_cfg.get("use_context_base_gate", False)),
            context_base_gate_init=model_cfg.get("context_base_gate_init"),
            pair_hidden_dim=int(model_cfg.get("pair_hidden_dim", 64)),
            pair_input_transform=str(model_cfg.get("pair_input_transform", "log1p")),
            use_decoder_pair_gate=bool(model_cfg.get("use_decoder_pair_gate", False)),
            decoder_pair_gate_init=float(model_cfg.get("decoder_pair_gate_init", 0.5)),
            use_full_rank_pair_decoder=bool(model_cfg.get("use_full_rank_pair_decoder", False)),
            full_rank_hidden_dim=int(model_cfg.get("full_rank_hidden_dim", 64)),
            full_rank_base_gate_init=float(model_cfg.get("full_rank_base_gate_init", 0.8)),
            full_rank_delta_scale=float(model_cfg.get("full_rank_delta_scale", 1.0)),
            full_rank_delta_limit=float(model_cfg.get("full_rank_delta_limit", 8.0)),
            use_marginal_features=bool(model_cfg.get("use_marginal_features", False)),
            use_relative_pair_delta=bool(model_cfg.get("use_relative_pair_delta", False)),
            relative_pair_delta_limit=float(model_cfg.get("relative_pair_delta_limit", 1.0)),
            use_sparse_hurdle_head=bool(model_cfg.get("use_sparse_hurdle_head", False)),
            occurrence_init=float(model_cfg.get("occurrence_init", 0.2)),
            occurrence_prior_scale=float(model_cfg.get("occurrence_prior_scale", 1.0)),
            occurrence_temperature=float(model_cfg.get("occurrence_temperature", 1.0)),
            output_calibration_scale=model_cfg.get("output_calibration_scale", 1.0),
            occurrence_calibration_bias=model_cfg.get("occurrence_calibration_bias", 0.0),
            occurrence_calibration_temperature=model_cfg.get("occurrence_calibration_temperature", 1.0),
            occurrence_probability_threshold=model_cfg.get("occurrence_probability_threshold", 0.0),
            prediction_calibration_power=model_cfg.get("prediction_calibration_power", 1.0),
            prediction_value_threshold=model_cfg.get("prediction_value_threshold", 0.0),
            evidence_calibration_weights=model_cfg.get("evidence_calibration_weights"),
            evidence_calibration_threshold=model_cfg.get("evidence_calibration_threshold", 0.0),
            use_local_od_mixer=bool(model_cfg.get("use_local_od_mixer", False)),
            local_mixer_hidden_dim=int(model_cfg.get("local_mixer_hidden_dim", 32)),
            local_mixer_layers=int(model_cfg.get("local_mixer_layers", 5)),
            local_mixer_softplus_beta=float(model_cfg.get("local_mixer_softplus_beta", 10.0)),
            freeze_local_od_mixer=bool(model_cfg.get("freeze_local_od_mixer", True)),
        )
    raise NotImplementedError(f"Model is not implemented: {name}")


def resolve_amp_dtype(name: str | None) -> torch.dtype:
    value = str(name or "bf16").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported train.amp_dtype: {name}")


def describe_device(device: torch.device) -> None:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1024**3
        print(
            f"Device: cuda:{device.index or 0} | {props.name} | "
            f"capability={props.major}.{props.minor} | memory={total_gb:.1f} GB"
        )
    else:
        print(f"Device: {device}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def assert_amp_trainable_dtypes(model: torch.nn.Module, use_amp: bool, amp_dtype: torch.dtype) -> None:
    """GradScaler requires trainable parameters to keep fp32 master gradients.

    The forward pass can still run in fp16 autocast. This check only prevents
    HuggingFace fp16 weights from being unfrozen directly, which raises
    "Attempting to unscale FP16 gradients" on the first optimizer step.
    """
    if not use_amp or amp_dtype != torch.float16:
        return
    fp16_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.dtype == torch.float16
    ]
    if fp16_names:
        preview = ", ".join(fp16_names[:8])
        suffix = " ..." if len(fp16_names) > 8 else ""
        raise ValueError(
            "FP16 AMP with GradScaler found trainable FP16 parameters: "
            f"{preview}{suffix}. Keep model.llm_trainable_fp32=true or load "
            "trainable backbone parameters in fp32."
        )


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    base_lr = float(cfg["train"]["learning_rate"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    llm_lr = cfg["train"].get("llm_learning_rate")
    if llm_lr is None:
        return torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    llm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llm_model."):
            llm_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if llm_params:
        groups.append({"params": llm_params, "lr": float(llm_lr), "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters found for optimizer.")
    return torch.optim.AdamW(groups)


def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        stats = {
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype).replace("torch.", ""),
            "finite_ratio": float(finite.float().mean().cpu().item()) if detached.numel() else 1.0,
        }
        if finite.any():
            values = detached[finite].float()
            stats.update(
                {
                    "min": float(values.min().cpu().item()),
                    "max": float(values.max().cpu().item()),
                    "mean": float(values.mean().cpu().item()),
                }
            )
        return stats


def _batch_debug_info(batch: dict, limit: int = 6) -> dict:
    info = {}
    for key in ["sample_index", "target_start", "line_index", "direction", "num_stops", "prev_valid"]:
        if key in batch and torch.is_tensor(batch[key]):
            info[key] = batch[key].detach().cpu().reshape(-1).tolist()[:limit]
    for key in ["line_no", "line_dir"]:
        if key in batch:
            value = batch[key]
            info[key] = list(value[:limit]) if isinstance(value, (list, tuple)) else value
    return info


def _sanitize_tensor(name: str, tensor: torch.Tensor, cfg: dict, batch_idx: int) -> torch.Tensor:
    if not bool(cfg["train"].get("sanitize_nonfinite", True)):
        return tensor
    if bool(torch.isfinite(tensor).all().item()):
        return tensor
    replacement = float(cfg["train"].get("nonfinite_replacement", 0.0))
    print(
        f"Warning: non-finite values in {name} at batch {batch_idx}; "
        f"stats={_tensor_debug_stats(name, tensor)}. Replacing with {replacement}."
    )
    return torch.nan_to_num(tensor, nan=replacement, posinf=replacement, neginf=replacement)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    train: bool,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
    teacher_model=None,
) -> tuple[float, dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    collect_metrics = (not train) or bool(cfg["train"].get("compute_train_metrics", False))
    stream_metrics = collect_metrics and bool(cfg["train"].get("stream_metrics", False))
    metric_accumulator = ODMetricAccumulator(**cfg["metrics"]) if stream_metrics else None
    losses = []
    preds = []
    trues = []
    y_times = []
    valid_masks = []
    loss_cfg = cfg["loss"]
    grad_accum_steps = max(int(cfg["train"].get("grad_accum_steps", 1)), 1)
    max_batches_key = "max_train_batches_per_epoch" if train else "max_eval_batches_per_epoch"
    configured_max_batches = int(cfg["train"].get(max_batches_key, 0) or 0)
    active_batches = len(loader)
    if configured_max_batches > 0:
        active_batches = min(active_batches, configured_max_batches)

    nonfinite_batches = 0
    max_nonfinite_batches = int(cfg["train"].get("max_nonfinite_batches", 3))
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False, total=active_batches)
    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= active_batches:
            break
        x = _sanitize_tensor("x", batch["x"].float().to(device), cfg, batch_idx)  # [B, L, N, N]
        y = _sanitize_tensor("y", batch["y"].float().to(device), cfg, batch_idx)  # [B, H, N, N]
        valid_mask = None
        if bool(loss_cfg.get("use_od_mask", cfg["metrics"].get("use_od_mask", False))) and "od_mask" in batch:
            valid_mask = batch["od_mask"].to(device=device, dtype=torch.bool)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.expand_as(y)
        poi_features = None
        if bool(cfg["model"].get("dynamic_poi_features", False)) and "poi_features" in batch:
            poi_features = _sanitize_tensor("poi_features", batch["poi_features"].float().to(device), cfg, batch_idx)
        model_output = None

        do_update = train and optimizer is not None
        if do_update and batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(do_update):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                auxiliary_loss_weight = float(cfg["model"].get("auxiliary_loss_weight", 0.0))
                model_name = str(cfg["model"].get("name", ""))
                if str(cfg["model"].get("name", "")) == "odmixer":
                    model_output = model(
                        x,
                        prev_x=batch["prev_x"].float().to(device),
                        return_auxiliary=auxiliary_loss_weight > 0.0,
                    )
                elif bool(cfg["model"].get("use_time_features", False)):
                    x_time = _sanitize_tensor("x_time", batch["x_time"].float().to(device), cfg, batch_idx)
                    y_time = _sanitize_tensor("y_time", batch["y_time"].float().to(device), cfg, batch_idx)
                    kwargs = {
                        "x_time": select_model_time_features(x_time, cfg),
                        "y_time": select_model_time_features(y_time, cfg),
                    }
                    if poi_features is not None:
                        kwargs["poi_features"] = poi_features
                    if str(cfg["model"].get("name", "")) in {"od_llm", "od_llm_gpt2"}:
                        if "prev_x" in batch:
                            kwargs["prev_x"] = _sanitize_tensor(
                                "prev_x", batch["prev_x"].float().to(device), cfg, batch_idx
                            )
                        decoder_mode = str(cfg["model"].get("decoder_mode", ""))
                        if "seasonal" in decoder_mode and "prev_y" in batch:
                            kwargs["prev_y"] = _sanitize_tensor(
                                "prev_y", batch["prev_y"].float().to(device), cfg, batch_idx
                            )
                            kwargs["prev_valid"] = batch["prev_valid"].to(device)
                    model_output = model(x, **kwargs)
                elif poi_features is not None and model_name in {
                    "od_llm",
                    "od_llm_gpt2",
                    "od_tensor_transformer",
                }:
                    model_output = model(x, poi_features=poi_features)
                else:
                    model_output = model(x)
                pred = model_output["prediction"] if isinstance(model_output, dict) else model_output
                prediction_clip = float(cfg["train"].get("prediction_clip", 0.0) or 0.0)
                if prediction_clip > 0.0:
                    pred = pred.clamp(min=0.0, max=prediction_clip)
                loss, parts = sparse_od_loss(
                    pred,
                    y,
                    alpha=float(loss_cfg["alpha"]),
                    beta=float(loss_cfg["beta"]),
                    gamma=float(loss_cfg["gamma"]),
                    topk=int(loss_cfg["topk"]),
                    nonzero_threshold=float(loss_cfg.get("nonzero_threshold", 0.0)),
                    mse_weight=float(loss_cfg.get("mse_weight", 0.0)),
                    nonzero_mse_weight=float(loss_cfg.get("nonzero_mse_weight", 0.0)),
                    topk_mse_weight=float(loss_cfg.get("topk_mse_weight", 0.0)),
                    valid_mask=valid_mask,
                    occurrence_logits=(
                        model_output.get("occurrence_logits") if isinstance(model_output, dict) else None
                    ),
                    positive_magnitude=(
                        model_output.get("positive_magnitude") if isinstance(model_output, dict) else None
                    ),
                    occurrence_loss_weight=float(loss_cfg.get("occurrence_loss_weight", 0.0)),
                    occurrence_positive_weight=float(loss_cfg.get("occurrence_positive_weight", 0.7)),
                    magnitude_loss_weight=float(loss_cfg.get("magnitude_loss_weight", 0.0)),
                    magnitude_mse_weight=float(loss_cfg.get("magnitude_mse_weight", 0.0)),
                )
                distill_weight = float(cfg["train"].get("distill_weight", 0.0))
                distill_mse_weight = float(cfg["train"].get("distill_mse_weight", 0.0))
                if train and teacher_model is not None and (distill_weight > 0.0 or distill_mse_weight > 0.0):
                    with torch.no_grad():
                        teacher_pred = teacher_model(
                            x,
                            prev_x=batch["prev_x"].float().to(device),
                        ).float()
                    student_pred = pred.float()
                    teacher_error = (teacher_pred - y.float()).abs()
                    temperature = max(float(cfg["train"].get("distill_temperature", 2.0)), 1e-6)
                    confidence = torch.exp(-teacher_error / temperature)
                    flow_boost = float(cfg["train"].get("distill_flow_boost", 0.0))
                    if flow_boost > 0.0:
                        confidence = confidence * (1.0 + flow_boost * torch.log1p(y.float().clamp_min(0.0)))
                    distill_mask = valid_mask if valid_mask is not None else torch.ones_like(y, dtype=torch.bool)
                    weights = confidence[distill_mask]
                    weight_sum = weights.sum().clamp_min(1e-6)
                    distill_diff = (student_pred - teacher_pred)[distill_mask]
                    distill_mae = (distill_diff.abs() * weights).sum() / weight_sum
                    distill_mse = (distill_diff.square() * weights).sum() / weight_sum
                    loss = loss + distill_weight * distill_mae + distill_mse_weight * distill_mse
                    parts["loss_distill"] = float(distill_mae.detach().cpu().item())
                    parts["loss_distill_mse"] = float(distill_mse.detach().cpu().item())
                if (
                    isinstance(model_output, dict)
                    and auxiliary_loss_weight > 0.0
                    and "prev_prediction" in model_output
                    and "prev_y" in batch
                ):
                    aux_loss, _ = sparse_od_loss(
                        model_output["prev_prediction"],
                        batch["prev_y"].float().to(device),
                        alpha=float(loss_cfg["alpha"]),
                        beta=float(loss_cfg["beta"]),
                        gamma=float(loss_cfg["gamma"]),
                        topk=int(loss_cfg["topk"]),
                        nonzero_threshold=float(loss_cfg.get("nonzero_threshold", 0.0)),
                        mse_weight=float(loss_cfg.get("mse_weight", 0.0)),
                        nonzero_mse_weight=float(loss_cfg.get("nonzero_mse_weight", 0.0)),
                        topk_mse_weight=float(loss_cfg.get("topk_mse_weight", 0.0)),
                        valid_mask=valid_mask,
                    )
                    loss = loss + auxiliary_loss_weight * aux_loss
                    parts["loss_aux"] = float(aux_loss.detach().cpu().item())
                if not torch.isfinite(loss):
                    nonfinite_batches += 1
                    message = (
                        f"Non-finite loss encountered at batch {batch_idx}: {parts}; "
                        f"batch={_batch_debug_info(batch)}; "
                        f"pred={_tensor_debug_stats('pred', pred)}; "
                        f"y={_tensor_debug_stats('y', y)}; "
                        f"x={_tensor_debug_stats('x', x)}"
                    )
                    if train and bool(cfg["train"].get("skip_nonfinite_batches", False)):
                        print(f"Warning: {message}")
                        if optimizer is not None:
                            optimizer.zero_grad(set_to_none=True)
                        if nonfinite_batches > max_nonfinite_batches:
                            raise FloatingPointError(
                                f"Exceeded max_nonfinite_batches={max_nonfinite_batches}. Last error: {message}"
                            )
                        continue
                    raise FloatingPointError(message)
            if do_update:
                loss_for_backward = loss / grad_accum_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                    should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == active_batches
                    if should_step:
                        grad_clip = cfg["train"].get("grad_clip")
                        if grad_clip is not None:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                            if not torch.isfinite(grad_norm):
                                print(f"Warning: non-finite grad_norm at batch {batch_idx}; skipping optimizer step.")
                                optimizer.zero_grad(set_to_none=True)
                                scaler.update()
                                continue
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss_for_backward.backward()
                    should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == active_batches
                    if should_step:
                        grad_clip = cfg["train"].get("grad_clip")
                        if grad_clip is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                            if not torch.isfinite(grad_norm):
                                print(f"Warning: non-finite grad_norm at batch {batch_idx}; skipping optimizer step.")
                                optimizer.zero_grad(set_to_none=True)
                                continue
                        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        if collect_metrics:
            if metric_accumulator is not None:
                metric_accumulator.update(pred, y, y_time=batch["y_time"], valid_mask=valid_mask)
            else:
                preds.append(pred.detach().cpu())
                trues.append(y.detach().cpu())
                y_times.append(batch["y_time"].detach().cpu())
                if valid_mask is not None:
                    valid_masks.append(valid_mask.detach().cpu())
        iterator.set_postfix(loss=f"{losses[-1]:.4f}", **parts)

    if not collect_metrics:
        return sum(losses) / max(len(losses), 1), {}
    if metric_accumulator is not None:
        return sum(losses) / max(len(losses), 1), metric_accumulator.compute()

    pred_all = torch.cat(preds, dim=0)
    true_all = torch.cat(trues, dim=0)
    y_time_all = torch.cat(y_times, dim=0)
    valid_mask_all = torch.cat(valid_masks, dim=0) if valid_masks else None
    metrics = compute_metrics(pred_all, true_all, y_time=y_time_all, valid_mask=valid_mask_all, **cfg["metrics"])
    return sum(losses) / max(len(losses), 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/common/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    describe_device(device)

    train_set = build_dataset(cfg["data"], "train")
    val_set = build_dataset(cfg["data"], "val")
    test_set = build_dataset(cfg["data"], "test")
    line_count = getattr(train_set, "num_lines", None)
    if line_count is None:
        print(f"Dataset: N={train_set.num_nodes}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    else:
        print(
            f"Dataset: lines={line_count}, Nmax={train_set.num_nodes}, "
            f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
        )

    train_loader = build_dataloader(
        train_set,
        cfg,
        shuffle=bool(cfg["train"].get("shuffle_train", True)),
        device=device,
    )
    val_loader = build_dataloader(val_set, cfg, shuffle=False, device=device)
    test_loader = build_dataloader(test_set, cfg, shuffle=False, device=device)

    dataset_time_feature_dim = int(train_set.time_features.shape[1]) if train_set.time_features.ndim == 2 else 0
    model_time_feature_indices = resolve_model_time_feature_indices(
        cfg,
        getattr(train_set, "time_feature_columns", None),
        dataset_time_feature_dim,
    )
    cfg.setdefault("_runtime", {})["model_time_feature_indices"] = model_time_feature_indices
    time_feature_dim = (
        dataset_time_feature_dim
        if model_time_feature_indices is None
        else len(model_time_feature_indices)
    )
    model = build_model(
        cfg,
        train_set.num_nodes,
        time_feature_dim=time_feature_dim,
        poi_features=train_set.poi_features,
        poi_feature_dim=train_set.poi_feature_dim,
    ).to(device)
    init_path = resolve_init_checkpoint(cfg)
    if init_path is not None:
        matched, skipped = load_matching_model_state(model, init_path)
        print(f"Warm-started from {init_path} | matched={matched}, skipped={skipped}")
    local_init_experiment = cfg["train"].get("init_local_experiment")
    if local_init_experiment:
        local_model = getattr(model, "local_od_mixer", None)
        if local_model is None:
            raise ValueError("train.init_local_experiment requires model.use_local_od_mixer=true.")
        local_path = (
            Path(cfg.get("outputs", {}).get("root", "outputs"))
            / str(local_init_experiment)
            / "checkpoints"
            / "best.pt"
        )
        if not local_path.exists():
            raise FileNotFoundError(f"Local OD mixer checkpoint not found: {local_path}")
        local_state = torch.load(local_path, map_location="cpu")
        local_model.load_state_dict(local_state["model"], strict=True)
        print(f"Initialized local OD mixer from {local_path}")
    if bool(cfg["model"].get("train_local_global_gate_only", False)):
        gate = getattr(model, "local_global_gate_head", None)
        if gate is None:
            raise ValueError("train_local_global_gate_only requires model.use_local_od_mixer=true.")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in gate.parameters():
            parameter.requires_grad = True
        print("Training only the Qwen-conditioned local-global OD gate.")
    teacher_model = None
    distill_weight = float(cfg["train"].get("distill_weight", 0.0))
    distill_mse_weight = float(cfg["train"].get("distill_mse_weight", 0.0))
    if distill_weight > 0.0 or distill_mse_weight > 0.0:
        teacher_experiment = str(cfg["train"].get("distill_teacher_experiment", "odmixer"))
        teacher_path = Path(cfg["outputs"]["root"]) / teacher_experiment / "checkpoints" / "best.pt"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Distillation teacher checkpoint not found: {teacher_path}")
        teacher_state = torch.load(teacher_path, map_location="cpu")
        teacher_cfg = teacher_state.get("config")
        if not isinstance(teacher_cfg, dict):
            raise ValueError(f"Teacher checkpoint has no config: {teacher_path}")
        teacher_model = build_model(
            teacher_cfg,
            train_set.num_nodes,
            time_feature_dim=dataset_time_feature_dim,
            poi_features=train_set.poi_features,
            poi_feature_dim=train_set.poi_feature_dim,
        ).to(device)
        teacher_model.load_state_dict(teacher_state["model"])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(
            f"Distillation teacher: {teacher_path} | "
            f"mae_weight={distill_weight}, mse_weight={distill_mse_weight}"
        )
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    amp_dtype = resolve_amp_dtype(cfg["train"].get("amp_dtype", "bf16"))
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    assert_amp_trainable_dtypes(model, use_amp=use_amp, amp_dtype=amp_dtype)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = None
    if trainable_params:
        optimizer = build_optimizer(model, cfg)
    else:
        print("Model has no trainable parameters; training epochs will run as deterministic evaluation.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    grad_accum_steps = max(int(cfg["train"].get("grad_accum_steps", 1)), 1)
    effective_batch = int(cfg["train"]["batch_size"]) * grad_accum_steps
    print(
        f"AMP: enabled={use_amp}, dtype={str(amp_dtype).replace('torch.', '')} | "
        f"grad_accum_steps={grad_accum_steps}, effective_batch={effective_batch}"
    )
    max_train_batches = int(cfg["train"].get("max_train_batches_per_epoch", 0) or 0)
    max_eval_batches = int(cfg["train"].get("max_eval_batches_per_epoch", 0) or 0)
    if max_train_batches > 0 or max_eval_batches > 0:
        print(
            "Batch caps: "
            f"train={max_train_batches if max_train_batches > 0 else 'full'}, "
            f"eval={max_eval_batches if max_eval_batches > 0 else 'full'} "
            "(use only for speed tests, not final paper metrics)."
        )

    ckpt_dir = Path(cfg["outputs"]["checkpoints"])
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history_path = Path(cfg["outputs"]["logs"]) / "train_history.jsonl"
    resume_path = resolve_resume_checkpoint(cfg, ckpt_dir)
    resume_enabled = resume_path is not None
    if history_path.exists() and not resume_enabled:
        history_path.unlink()
    best_val = float("inf")
    bad_epochs = 0
    start_epoch = 1
    monitor_metric = str(cfg["train"].get("monitor_metric", "val_loss"))

    if resume_path is not None:
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        resume_model_only = bool(cfg["train"].get("resume_model_only", False))
        if not resume_model_only and optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if not resume_model_only and scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        history_epoch, history_best, history_bad = read_history_state(history_path, monitor_metric)
        start_epoch = max(int(state.get("epoch", 0)), history_epoch) + 1
        best_val = history_best if resume_model_only else float(state.get("best_val", history_best))
        bad_epochs = history_bad if resume_model_only else int(state.get("bad_epochs", history_bad))
        if not torch.isfinite(torch.tensor(best_val)):
            best_val = history_best
        print(
            f"Resumed from {resume_path} | start_epoch={start_epoch}, "
            f"best_{monitor_metric}={best_val:.4f}, bad_epochs={bad_epochs}"
        )

    if resume_path is None and bool(cfg["train"].get("evaluate_initial_checkpoint", False)):
        initial_val_loss, initial_val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            cfg=cfg,
            train=False,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        initial_monitor = (
            initial_val_loss
            if monitor_metric == "val_loss"
            else float(initial_val_metrics[monitor_metric])
        )
        best_val = initial_monitor
        initial_record = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": initial_val_loss,
            "monitor_metric": monitor_metric,
            "monitor_value": initial_monitor,
            "train_metrics": {},
            "val_metrics": initial_val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(initial_record) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": 0,
                "best_val": best_val,
                "bad_epochs": 0,
                "config": cfg,
                "num_nodes": train_set.num_nodes,
            },
            best_path,
        )
        print(
            f"Saved initialized checkpoint to {best_path} "
            f"({monitor_metric}={initial_monitor:.4f})"
        )

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            train=True,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            teacher_model=teacher_model,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, optimizer, device, cfg, train=False, use_amp=use_amp, amp_dtype=amp_dtype
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
            f"val {format_metrics(val_metrics)}"
        )
        if monitor_metric == "val_loss":
            monitor_value = val_loss
        else:
            if monitor_metric not in val_metrics:
                raise KeyError(f"train.monitor_metric={monitor_metric} is not in validation metrics")
            monitor_value = float(val_metrics[monitor_metric])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "monitor_metric": monitor_metric,
            "monitor_value": monitor_value,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if monitor_value < best_val:
            best_val = monitor_value
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch,
                    "best_val": best_val,
                    "bad_epochs": bad_epochs,
                    "config": cfg,
                    "num_nodes": train_set.num_nodes,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path} ({monitor_metric}={monitor_value:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["train"].get("patience", 5)):
                print("Early stopping")
                break
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "config": cfg,
                "num_nodes": train_set.num_nodes,
            },
            last_path,
        )

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_metrics = run_epoch(
        model, test_loader, optimizer=None, device=device, cfg=cfg, train=False, use_amp=use_amp, amp_dtype=amp_dtype
    )
    metrics_path = Path(cfg["outputs"]["logs"]) / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)
    print(f"Test loss={test_loss:.4f} | {format_metrics(test_metrics)}")
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
