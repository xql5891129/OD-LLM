from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_toy_od(total_steps: int, num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    od = np.zeros((total_steps, num_nodes, num_nodes), dtype=np.float32)
    stations = np.arange(num_nodes)
    home = stations[: num_nodes // 2]
    work = stations[num_nodes // 2 :]

    base_pairs = []
    for o in home:
        for d in work:
            if rng.random() < 0.35:
                base_pairs.append((o, d, rng.uniform(3.0, 10.0)))
                base_pairs.append((d, o, rng.uniform(2.0, 8.0)))

    for t in range(total_steps):
        slot = t % 48
        day = (t // 48) % 7
        weekday_scale = 1.0 if day < 5 else 0.55
        morning = np.exp(-0.5 * ((slot - 16) / 3.0) ** 2)
        evening = np.exp(-0.5 * ((slot - 36) / 3.5) ** 2)
        midday = 0.35 * np.exp(-0.5 * ((slot - 25) / 7.0) ** 2)

        for o, d, strength in base_pairs:
            direction_boost = morning if o in home else evening
            lam = weekday_scale * strength * (0.15 + direction_boost + midday)
            if rng.random() < 0.02:
                lam *= rng.uniform(2.0, 4.0)
            flow = rng.poisson(lam)
            if flow > 0:
                od[t, o, d] = flow

        noise_pairs = rng.integers(0, num_nodes, size=(num_nodes // 2, 2))
        for o, d in noise_pairs:
            if o != d and rng.random() < 0.08:
                od[t, o, d] += rng.poisson(1.0)

    idx = np.arange(num_nodes)
    od[:, idx, idx] = 0.0
    return od


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/toy/od.npy")
    parser.add_argument("--total_steps", type=int, default=240)
    parser.add_argument("--num_nodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    od = generate_toy_od(args.total_steps, args.num_nodes, args.seed)
    np.save(output, od)
    print(f"Saved toy OD data to {output} with shape {od.shape}")


if __name__ == "__main__":
    main()

