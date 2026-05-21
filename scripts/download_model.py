from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_REGISTRY = {
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "modelscope_id": "Qwen/Qwen2.5-0.5B",
        "llm_dim": 896,
        "llm_heads": 14,
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "modelscope_id": "Qwen/Qwen2.5-1.5B",
        "llm_dim": 1536,
        "llm_heads": 12,
    },
    "qwen2.5-1.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "modelscope_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "llm_dim": 1536,
        "llm_heads": 12,
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B",
        "modelscope_id": "Qwen/Qwen2.5-3B",
        "llm_dim": 2048,
        "llm_heads": 16,
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "modelscope_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "llm_dim": 1536,
        "llm_heads": 12,
    },
    "deepseek-r1-distill-qwen-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "modelscope_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "llm_dim": 3584,
        "llm_heads": 28,
    },
}


def download_from_huggingface(model_id: str, target_dir: Path, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing huggingface_hub. Run `uv sync --extra download` first.") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(target_dir),
    )
    return target_dir


def download_from_modelscope(model_id: str, cache_dir: Path, revision: str | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing modelscope. Run `uv sync --extra download` first.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"model_id": model_id, "cache_dir": str(cache_dir)}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def write_local_config(
    model_key: str,
    local_path: Path,
    config_output: Path,
    llm_layers: int,
    rank: int,
    d_model: int,
) -> None:
    spec = MODEL_REGISTRY[model_key]
    experiment_name = f"od_llm_{model_key.replace('.', '_').replace('-', '_')}_local"
    path_text = local_path.resolve().as_posix()
    config_text = f"""inherits: default.yaml

experiment:
  name: {experiment_name}

model:
  name: od_llm
  rank: {rank}
  d_model: {d_model}
  n_heads: 4
  dim_feedforward: 128
  dropout: 0.1
  max_tokens: 4096

  llm_model: {model_key}
  pretrained_path: "{path_text}"
  pretrained: true
  local_files_only: true
  trust_remote_code: true

  llm_dim: {spec["llm_dim"]}
  llm_layers: {llm_layers}
  llm_heads: {spec["llm_heads"]}
  freeze_llm: true
  use_reprogramming: true
  num_virtual_prompt_tokens: 8
  num_source_tokens: 1000
"""
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(config_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a local LLM backbone for OD-LLM.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="qwen2.5-1.5b")
    parser.add_argument("--provider", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--output-dir", type=str, default="hf_models")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--config-output", type=str, default=None)
    parser.add_argument("--llm-layers", type=int, default=6)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--print-registry", action="store_true")
    args = parser.parse_args()

    if args.print_registry:
        print(json.dumps(MODEL_REGISTRY, indent=2))
        return

    spec = MODEL_REGISTRY[args.model]
    output_root = Path(args.output_dir)
    target_dir = output_root / args.model

    if args.provider == "huggingface":
        local_path = download_from_huggingface(spec["hf_id"], target_dir, args.revision)
    else:
        local_path = download_from_modelscope(spec["modelscope_id"], output_root, args.revision)

    config_output = Path(args.config_output or f"configs/local_{args.model.replace('.', '_').replace('-', '_')}.yaml")
    write_local_config(
        model_key=args.model,
        local_path=local_path,
        config_output=config_output,
        llm_layers=args.llm_layers,
        rank=args.rank,
        d_model=args.d_model,
    )

    print(f"Downloaded {args.model} to: {local_path.resolve()}")
    print(f"Wrote OD-LLM config to: {config_output.resolve()}")
    print(f"Train with: uv run --no-sync python src/train.py --config {config_output.as_posix()}")


if __name__ == "__main__":
    main()
