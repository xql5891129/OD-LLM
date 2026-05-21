# OD-LLM

OD-LLM is a research codebase for bus origin-destination demand forecasting and explanation.

Current stage:

- OD matrix dataset loading from `.npy` and long-form `.csv`
- Learnable two-sided OD tensor tokenizer
- Transformer encoder sanity-check backbone
- OD latent reconstruction head
- Sparse OD loss for all, nonzero, and top-k OD pairs
- Unified forecasting metrics

The first runnable model intentionally avoids an LLM backbone. It validates the OD data pipeline,
latent tokenizer, decoder, loss, and metrics before adding GPT-2 / Time-LLM style adapters.

## Environment With uv

This project includes `pyproject.toml` for uv-based environment migration.
`uv.toml` pins the uv cache to the project-local `.uv-cache` directory, which
helps on Windows machines where the default user cache may be permission-limited.

```bash
cd OD-LLM
uv sync
```

Then run commands inside the uv environment:

```bash
uv run python scripts/generate_toy_data.py --output data/toy/od.npy
uv run python src/train.py --config configs/default.yaml
```

One-click setup plus model download:

Windows PowerShell:

```powershell
.\scripts\setup_env.ps1 -Model qwen2.5-1.5b -Provider huggingface
```

Windows cmd:

```bat
scripts\setup_env.bat -Model qwen2.5-1.5b -Provider huggingface
```

Linux/macOS:

```bash
bash scripts/setup_env.sh --model qwen2.5-1.5b --provider huggingface
```

For users in mainland China, ModelScope may be smoother:

```powershell
.\scripts\setup_env.ps1 -Model qwen2.5-1.5b -Provider modelscope
```

Supported model keys:

```bash
uv run python scripts/download_model.py --print-registry
```

If you use Qwen or DeepSeek, put the model on disk and set `pretrained_path` in
`configs/od_llm_qwen.yaml` or `configs/od_llm_deepseek.yaml`. Keeping
`local_files_only: true` makes experiments reproducible and avoids silent model
downloads.

## Quick Start

```bash
cd OD-LLM
python scripts/generate_toy_data.py --output data/toy/od.npy
python src/train.py --config configs/default.yaml
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/od_tensor_transformer_toy/checkpoints/best.pt
```

Offline OD-LLM smoke test:

```bash
python src/train.py --config configs/od_llm_tiny.yaml
```

Collect all finished run metrics:

```bash
python src/compare.py --root outputs --output outputs/comparison.csv
```

`configs/od_llm_qwen.yaml` and `configs/od_llm_deepseek.yaml` are reserved for
real local Qwen / DeepSeek checkpoints. They expect `transformers` and
cached/local weights.

## Data Formats

Numpy:

```text
od.npy shape: [T, N, N]
```

CSV:

```text
time,origin,destination,flow
2024-01-01 06:00:00,0,3,12
```

Splits are chronological. Samples are constructed as:

```text
x = OD[t-input_len:t]       shape [L, N, N]
y = OD[t:t+pred_len]        shape [H, N, N]
```

## Roadmap

1. Minimal OD tensor tokenizer + Transformer baseline.
2. GPT-2 / Time-LLM adapter over OD latent tokens.
3. Baselines and ablations.
4. Latent-space analysis and basis heatmaps.
5. Explanation prompt generation.
