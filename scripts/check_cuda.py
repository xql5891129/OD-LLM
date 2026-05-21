from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is not visible to PyTorch in this environment.")
        print("On the 5090D server, rerun setup with: -TorchBackend cu130")
        print("If cu130 is unavailable on your mirror, try: -TorchBackend cu128")
        return

    print(f"cuda device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"device {idx}: {props.name}, capability={props.major}.{props.minor}, memory={total_gb:.1f} GB")

    device = torch.device("cuda:0")
    x = torch.randn(256, 256, device=device)
    y = x @ x.T
    torch.cuda.synchronize(device)
    print(f"cuda matmul smoke test: ok, mean={y.mean().item():.6f}")


if __name__ == "__main__":
    main()
