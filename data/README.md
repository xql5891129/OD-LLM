# Data Directory

Put bus OD data here.

Supported formats:

- `od.npy` with shape `[T, N, N]`
- Long-form CSV with columns `time, origin, destination, flow`

For a toy smoke test:

```bash
uv run --no-sync python scripts/generate_toy_data.py --output data/toy/od.npy
```

Public trip data can be converted to the same `od.npy` format. Examples:

```bash
# Shanghai MetroFlow OD file. Use top-n for a manageable first experiment.
uv run --no-sync python scripts/prepare_metroflow.py \
  --raw-dir data/MetroFlow \
  --output-dir data/metroflow_top80 \
  --top-n 80 \
  --flow-type total \
  --overwrite

# Full 302-station MetroFlow. This creates a large dense OD array.
uv run --no-sync python scripts/prepare_metroflow.py \
  --raw-dir data/MetroFlow \
  --output-dir data/metroflow_full \
  --top-n 0 \
  --flow-type total \
  --overwrite

# Citi Bike / Capital Bikeshare monthly CSV or ZIP files
uv run --no-sync python scripts/prepare_public_od.py \
  --source citibike \
  --input data/raw/citibike \
  --output-dir data/public_citibike \
  --freq 30min \
  --top-n 80 \
  --max-rows 500000

# NYC TLC taxi parquet files
uv run --no-sync python scripts/prepare_public_od.py \
  --source nyc_taxi \
  --input data/raw/nyc_taxi \
  --output-dir data/public_nyc_taxi \
  --freq 30min \
  --top-n 80 \
  --max-rows 1000000

# Generic long-form OD CSV
uv run --no-sync python scripts/prepare_public_od.py \
  --source generic \
  --input data/raw/od.csv \
  --output-dir data/public_generic \
  --time-col time \
  --origin-col origin \
  --destination-col destination \
  --flow-col flow \
  --freq 30min
```
