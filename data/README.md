# Data Directory

Put bus OD data here.

Supported formats:

- `od.npy` with shape `[T, N, N]`
- Long-form CSV with columns `time, origin, destination, flow`

For a toy smoke test:

```bash
python scripts/generate_toy_data.py --output data/toy/od.npy
```

