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
latent tokenizer, decoder, loss, and metrics before adding Qwen / DeepSeek / Time-LLM style adapters.

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
uv run --no-sync python src/train.py --config configs/default.yaml
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

GPU server setup for RTX 5090D / CUDA 13.1 driver:

```powershell
.\scripts\setup_env.ps1 -SkipModel -TorchBackend cu130
.\scripts\setup_env.ps1 -Model qwen2.5-1.5b -Provider huggingface -TorchBackend cu130
```

Linux:

```bash
bash scripts/setup_env.sh --skip-model --torch-backend cu130
bash scripts/setup_env.sh --model qwen2.5-1.5b --provider huggingface --torch-backend cu130
```

`nvidia-smi` reports the maximum CUDA runtime supported by the driver. A
CUDA 13.1 driver can run a PyTorch `cu130` wheel. If your package mirror does
not yet provide `cu130`, use `cu128`:

```powershell
.\scripts\setup_env.ps1 -SkipModel -TorchBackend cu128
```

After installing a CUDA torch wheel with the setup script, prefer `uv run
--no-sync ...` for experiments. This prevents uv from re-syncing the lockfile
and replacing the CUDA wheel with the default CPU wheel.

After setup, run the smoke workflow first:

```powershell
.\scripts\run_smoke.ps1
```

Linux:

```bash
bash scripts/run_smoke.sh
```

Then start the local Qwen OD-LLM experiment:

```powershell
uv run --no-sync python src/train.py --config configs/local_qwen2_5_1_5b.yaml
```

## Public OD Data

For this project, use MetroFlow first. It is the closest public proxy for bus
OD forecasting because it is public transit, station-level, directional OD, and
continuous over time.

Download MetroFlow, then put these files under `data/raw/metroflow/`:

```text
stationInfo.csv
metroData_InOutFlow.csv
metroData_ODFlow.csv
shanghai_weatherHourly.csv
work_calendar.csv
```

Convert the OD flow file to a manageable top-80 station test set:

```powershell
uv run --no-sync python scripts/prepare_metroflow.py `
  --raw-dir data/MetroFlow `
  --output-dir data/metroflow_top80 `
  --top-n 80 `
  --flow-type total `
  --overwrite
```

The full 302-station dense array is larger, but supported:

```powershell
uv run --no-sync python scripts/prepare_metroflow.py `
  --raw-dir data/MetroFlow `
  --output-dir data/metroflow_full `
  --top-n 0 `
  --flow-type total `
  --overwrite
```

Run MetroFlow experiments:

```powershell
uv run --no-sync python src/train.py --config configs/metroflow_top80_transformer.yaml
uv run --no-sync python src/train.py --config configs/metroflow_top80_qwen.yaml
```

For a metadata-only check before scanning the 12GB OD file:

```powershell
uv run --no-sync python scripts/prepare_metroflow.py `
  --raw-dir data/MetroFlow `
  --output-dir outputs/metroflow_profile `
  --top-n 80 `
  --profile-only `
  --overwrite
```

For other public trip files, put downloaded files under `data/raw/...`, then
convert them to `od.npy`:

```powershell
uv run --no-sync python scripts/prepare_public_od.py `
  --source citibike `
  --input data/raw/citibike `
  --output-dir data/public_citibike `
  --freq 30min `
  --top-n 80 `
  --max-rows 500000
```

For NYC taxi parquet files:

```powershell
uv run --no-sync python scripts/prepare_public_od.py `
  --source nyc_taxi `
  --input data/raw/nyc_taxi `
  --output-dir data/public_nyc_taxi `
  --freq 30min `
  --top-n 80 `
  --max-rows 1000000
```

Then point a config to the generated file:

```yaml
data:
  format: npy
  path: data/public_citibike/od.npy
```

Ready-made configs are also provided:

```powershell
uv run --no-sync python src/train.py --config configs/public_citibike_transformer.yaml
uv run --no-sync python src/train.py --config configs/public_citibike_qwen.yaml
uv run --no-sync python src/train.py --config configs/public_nyc_taxi_transformer.yaml
uv run --no-sync python src/train.py --config configs/public_nyc_taxi_qwen.yaml
```

## Experiment Suites

Run all baseline experiments:

```powershell
.\scripts\run_suite.ps1 -Suite configs/suite_baselines.yaml -BaseConfig configs/real_od_transformer.yaml
```

Run OD-LLM ablations:

```powershell
.\scripts\run_suite.ps1 -Suite configs/suite_ablation.yaml -BaseConfig configs/real_od_qwen.yaml
```

Run the full paper experiment pipeline:

```powershell
.\scripts\run_paper_experiments.ps1
```

For a quick syntax check without training:

```powershell
.\scripts\run_suite.ps1 -Suite configs/suite_baselines.yaml -DryRun
```

For users in mainland China, ModelScope may be smoother:

```powershell
.\scripts\setup_env.ps1 -Model qwen2.5-1.5b -Provider modelscope
```

Supported model keys:

```bash
uv run --no-sync python scripts/download_model.py --print-registry
```

If you use Qwen or DeepSeek, put the model on disk and set `pretrained_path` in
`configs/od_llm_qwen.yaml` or `configs/od_llm_deepseek.yaml`. Keeping
`local_files_only: true` makes experiments reproducible and avoids silent model
downloads.

## Quick Start

```bash
cd OD-LLM
uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy
uv run --no-sync python src/train.py --config configs/default.yaml
uv run --no-sync python src/evaluate.py --config configs/default.yaml --checkpoint outputs/od_tensor_transformer_toy/checkpoints/best.pt
```

Offline OD-LLM smoke test:

```bash
uv run --no-sync python src/train.py --config configs/od_llm_tiny.yaml
```

Collect all finished run metrics:

```bash
uv run --no-sync python src/compare.py --root outputs --output outputs/comparison.csv
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
2. Qwen / DeepSeek / Time-LLM adapter over OD latent tokens.
3. Baselines and ablations.
4. Latent-space analysis and basis heatmaps.
5. Explanation prompt generation.
