# Line Bus Scripts

Use these scripts for line-direction bus OD experiments.

```powershell
# 1. Prepare one region, 60-minute OD tensors.
.\scripts\line_bus\prepare_line_od.ps1 -Region guangdianyuan -Granularity 60

# 2. Run baselines.
.\scripts\line_bus\run_baselines.ps1 -Region guangdianyuan -Granularity 60

# 3. Run the single OD-LLM main model.
.\scripts\line_bus\run_main.ps1 -Region guangdianyuan -Granularity 60

# 4. Run Qwen OD-LLM ablations.
.\scripts\line_bus\run_ablation.ps1 -Region guangdianyuan -Granularity 60
```

For a quick preprocessing smoke test:

```powershell
.\scripts\line_bus\prepare_line_od.ps1 -Region guangdianyuan -Granularity 60 -MaxLines 2 -MaxCardFiles 2 -NoProgress
```

Linux / AutoDL equivalents:

```bash
bash scripts/line_bus/run_baselines.sh --region guangdianyuan --granularity 60
bash scripts/line_bus/run_main.sh --region guangdianyuan --granularity 60
bash scripts/line_bus/run_ablation.sh --region guangdianyuan --granularity 60
```
