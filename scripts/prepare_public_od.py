from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_DEFAULTS = {
    "citibike": {
        "time_col": "started_at",
        "origin_col": "start_station_id",
        "destination_col": "end_station_id",
        "flow_col": None,
    },
    "capital_bikeshare": {
        "time_col": "started_at",
        "origin_col": "start_station_id",
        "destination_col": "end_station_id",
        "flow_col": None,
    },
    "nyc_taxi": {
        "time_col": "tpep_pickup_datetime",
        "origin_col": "PULocationID",
        "destination_col": "DOLocationID",
        "flow_col": None,
    },
    "chicago_taxi": {
        "time_col": "trip_start_timestamp",
        "origin_col": "pickup_community_area",
        "destination_col": "dropoff_community_area",
        "flow_col": None,
    },
    "mta_subway_od": {
        "time_col": "time",
        "origin_col": "origin",
        "destination_col": "destination",
        "flow_col": "flow",
    },
    "generic": {
        "time_col": "time",
        "origin_col": "origin",
        "destination_col": "destination",
        "flow_col": "flow",
    },
}


def iter_input_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".zip", ".parquet"}))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No input files found.")
    return files


def read_table(path: Path, columns: list[str] | None, max_rows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns, nrows=max_rows)
    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path, columns=columns)
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pyarrow. Install it with `uv add pyarrow`.") from exc
        return df.head(max_rows) if max_rows else df
    if suffix == ".zip":
        frames = []
        with zipfile.ZipFile(path) as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV file found inside {path}")
            remaining = max_rows
            for name in names:
                with zf.open(name) as f:
                    nrows = remaining if remaining is not None else None
                    frame = pd.read_csv(f, usecols=columns, nrows=nrows)
                    frames.append(frame)
                    if remaining is not None:
                        remaining -= len(frame)
                        if remaining <= 0:
                            break
        return pd.concat(frames, ignore_index=True)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def normalize_mta_subway_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort normalization for MTA Subway OD ridership exports."""
    lower_to_col = {col.lower().strip(): col for col in df.columns}
    candidates = {
        "origin": [
            "origin_station_complex_id",
            "origin station complex id",
            "origin_station_id",
            "origin",
        ],
        "destination": [
            "destination_station_complex_id",
            "destination station complex id",
            "destination_station_id",
            "destination",
        ],
        "flow": [
            "estimated_average_ridership",
            "estimated average ridership",
            "ridership",
            "flow",
        ],
    }
    rename: dict[str, str] = {}
    for target, names in candidates.items():
        for name in names:
            if name in lower_to_col:
                rename[lower_to_col[name]] = target
                break
    df = df.rename(columns=rename)

    if "time" not in df.columns:
        month_col = lower_to_col.get("month")
        day_col = lower_to_col.get("day_of_week") or lower_to_col.get("day of week")
        hour_col = lower_to_col.get("hour_of_day") or lower_to_col.get("hour of day")
        if month_col and day_col and hour_col:
            # MTA OD is aggregated by month, weekday, and hour. This creates a
            # sorted pseudo-time index suitable for pipeline tests, not a true
            # continuous daily time series.
            df["time"] = (
                df[month_col].astype(str)
                + "_dow"
                + df[day_col].astype(str)
                + "_hour"
                + df[hour_col].astype(str).str.zfill(2)
            )
    return df


def load_trip_tables(args: argparse.Namespace) -> pd.DataFrame:
    defaults = SOURCE_DEFAULTS[args.source]
    time_col = args.time_col or defaults["time_col"]
    origin_col = args.origin_col or defaults["origin_col"]
    dest_col = args.destination_col or defaults["destination_col"]
    flow_col = args.flow_col or defaults["flow_col"]

    columns = [time_col, origin_col, dest_col]
    if flow_col:
        columns.append(flow_col)
    if args.source == "mta_subway_od":
        columns = None

    frames = []
    remaining = args.max_rows
    for path in iter_input_files(args.input):
        frame = read_table(path, columns=columns, max_rows=remaining)
        if args.source == "mta_subway_od":
            frame = normalize_mta_subway_columns(frame)
            time_col, origin_col, dest_col, flow_col = "time", "origin", "destination", "flow"
        frame = frame.rename(columns={time_col: "time", origin_col: "origin", dest_col: "destination"})
        if flow_col and flow_col in frame.columns:
            frame = frame.rename(columns={flow_col: "flow"})
        else:
            frame["flow"] = 1.0
        frames.append(frame[["time", "origin", "destination", "flow"]])
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["time", "origin", "destination"])
    if args.drop_self_loops:
        df = df[df["origin"].astype(str) != df["destination"].astype(str)]
    return df


def build_od_array(df: pd.DataFrame, freq: str, top_n: int | None) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if parsed_time.notna().all():
        df = df.assign(time=parsed_time.dt.floor(freq).astype(str))
    else:
        df = df.assign(time=df["time"].astype(str))

    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0.0).astype(float)

    if top_n is not None and top_n > 0:
        node_flow = pd.concat(
            [
                df.groupby("origin")["flow"].sum(),
                df.groupby("destination")["flow"].sum(),
            ],
            axis=1,
        ).fillna(0.0)
        node_flow["total"] = node_flow.sum(axis=1)
        keep_nodes = set(node_flow.sort_values("total", ascending=False).head(top_n).index.astype(str))
        df = df[df["origin"].isin(keep_nodes) & df["destination"].isin(keep_nodes)]

    times = sorted(df["time"].unique())
    nodes = sorted(pd.unique(pd.concat([df["origin"], df["destination"]], ignore_index=True)))
    time_to_idx = {time: idx for idx, time in enumerate(times)}
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    od = np.zeros((len(times), len(nodes), len(nodes)), dtype=np.float32)
    grouped = df.groupby(["time", "origin", "destination"], as_index=False)["flow"].sum()
    for row in grouped.itertuples(index=False):
        od[time_to_idx[row.time], node_to_idx[row.origin], node_to_idx[row.destination]] = float(row.flow)

    time_df = pd.DataFrame({"time_idx": range(len(times)), "time": times})
    node_df = pd.DataFrame({"node_idx": range(len(nodes)), "node_id": nodes})
    return od, time_df, node_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public trip/OD data to OD-LLM od.npy format.")
    parser.add_argument("--source", choices=sorted(SOURCE_DEFAULTS), default="generic")
    parser.add_argument("--input", nargs="+", required=True, help="CSV, ZIP, parquet file, or directory.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--freq", type=str, default="30min")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--origin-col", type=str, default=None)
    parser.add_argument("--destination-col", type=str, default=None)
    parser.add_argument("--flow-col", type=str, default=None)
    parser.add_argument("--drop-self-loops", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_trip_tables(args)
    od, time_df, node_df = build_od_array(df, freq=args.freq, top_n=args.top_n)

    np.save(output_dir / "od.npy", od)
    time_df.to_csv(output_dir / "times.csv", index=False)
    node_df.to_csv(output_dir / "nodes.csv", index=False)
    metadata = {
        "source": args.source,
        "shape": list(od.shape),
        "freq": args.freq,
        "top_n": args.top_n,
        "max_rows": args.max_rows,
        "total_flow": float(od.sum()),
        "nonzero_ratio": float((od > 0).mean()) if od.size else 0.0,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved OD array: {output_dir / 'od.npy'}")
    print(f"Shape: {od.shape}, total_flow={metadata['total_flow']:.2f}, nonzero_ratio={metadata['nonzero_ratio']:.6f}")
    print(f"Saved nodes: {output_dir / 'nodes.csv'}")
    print(f"Saved times: {output_dir / 'times.csv'}")


if __name__ == "__main__":
    main()
