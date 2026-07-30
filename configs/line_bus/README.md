# Line Bus Experiment Configs

This folder is for line-direction bus OD experiments.

- `*_60_base.yaml`: 60-minute line OD setting, default `L=12, H=3`.
- `*_30_base.yaml`: 30-minute line OD setting, default `L=12, H=6`.
- `suite_baselines.yaml`: `ha`, `lstm`, `transformer`, `odmixer`, `odcrn`.
- `suite_main.yaml`: the single default `od_llm` model.
- `suite_ablation.yaml`: final OD-LLM component ablations.
- `suite_projection_ablation.yaml`: learnable and SVD projection variants.

Main metrics keep the names `mae`, `rmse`, and `wape`. For line-level data,
these are computed on feasible downstream OD pairs indicated by `od_mask`.
