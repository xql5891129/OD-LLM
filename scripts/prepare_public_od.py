from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_DEFAULTS = {
    "metroflow": {
        "time_col": "time",
        "origin_col": "origin",
        "destination_col": "destination",
        "flow_col": "flow",
    },
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


SOURCE_NOTES = {
    "metroflow": {
        "mode": "metro public transit",
        "closeness": "highest",
        "note": "Shanghai MetroFlow. Best public proxy for bus OD forecasting: station-level transit OD, 10-minute resolution, weather/workday metadata. Download manually from Figshare/GitHub, then convert if needed.",
    },
    "mta_subway_od": {
        "mode": "metro public transit",
        "closeness": "high",
        "note": "NYC MTA subway OD ridership. Public transit OD, but aggregated by month/day/hour rather than a continuous daily time series.",
    },
    "generic": {
        "mode": "any long-form OD",
        "closeness": "depends",
        "note": "Use for your own bus OD CSV or any file with time/origin/destination/flow columns.",
    },
    "nyc_taxi": {
        "mode": "taxi",
        "closeness": "medium",
        "note": "Zone-level OD with real urban demand, good for large OD stress tests but not public transit.",
    },
    "chicago_taxi": {
        "mode": "taxi",
        "closeness": "medium",
        "note": "Zone/community-area OD, smaller than NYC taxi and easy to test.",
    },
    "citibike": {
        "mode": "bike sharing",
        "closeness": "low-medium",
        "note": "Station OD and easy CSV format, useful for code tests but travel behavior differs from bus.",
    },
    "capital_bikeshare": {
        "mode": "bike sharing",
        "closeness": "low-medium",
        "note": "Same role as Citi Bike: convenient station OD engineering test.",
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


def _first_existing_column(lower_to_col: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        key = name.lower().strip()
        if key in lower_to_col:
            return lower_to_col[key]
    return None


def normalize_metroflow_columns(df: pd.DataFrame, flow_type: str = "total") -> pd.DataFrame:
    """Normalize Shanghai MetroFlow OD data columns.

    The OD flow file described by MetroFlow uses Date, Slot, StartTime,
    EndTime, O-Station, D-Station, CFlow, HFlow, and NFlow. This function is
    deliberately tolerant to minor header variants so exported CSV files remain
    easy to process.
    """
    lower_to_col = {col.lower().strip(): col for col in df.columns}
    slot_col = _first_existing_column(lower_to_col, ["Slot", "slot"])
    date_col = _first_existing_column(lower_to_col, ["Date", "date"])
    start_col = _first_existing_column(lower_to_col, ["StartTime", "Start Time", "start_time", "starttime"])
    origin_col = _first_existing_column(lower_to_col, ["O-Station", "O_Station", "OStation", "origin", "o_station"])
    dest_col = _first_existing_column(
        lower_to_col,
        ["D-Station", "D_Station", "DStation", "destination", "d_station"],
    )
    if origin_col is None or dest_col is None:
        raise ValueError(f"Cannot find MetroFlow OD station columns in: {list(df.columns)}")

    normalized = pd.DataFrame()
    if date_col and start_col:
        normalized["time"] = df[date_col].astype(str).str.strip() + " " + df[start_col].astype(str).str.strip()
    elif slot_col:
        normalized["time"] = pd.to_numeric(df[slot_col], errors="coerce")
    else:
        time_col = _first_existing_column(lower_to_col, ["time", "timestamp", "datetime"])
        if time_col is None:
            raise ValueError(f"Cannot find MetroFlow time columns in: {list(df.columns)}")
        normalized["time"] = df[time_col]

    normalized["origin"] = df[origin_col]
    normalized["destination"] = df[dest_col]

    flow_type = flow_type.lower()
    flow_candidates = {
        "c": ["CFlow", "cflow", "commuting_flow"],
        "h": ["HBOFlow", "HFlow", "hboflow", "hflow", "home_based_other_flow"],
        "n": ["NHBFlow", "NFlow", "nhbflow", "nflow", "non_home_based_flow", "none_home_based_flow"],
        "total": ["Flow", "flow", "ODFlow", "od_flow", "total_flow"],
    }
    if flow_type in {"c", "h", "n"}:
        col = _first_existing_column(lower_to_col, flow_candidates[flow_type])
        if col is None:
            raise ValueError(f"Cannot find MetroFlow {flow_type.upper()}Flow column in: {list(df.columns)}")
        normalized["flow"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    else:
        total_col = _first_existing_column(lower_to_col, flow_candidates["total"])
        if total_col:
            normalized["flow"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
        else:
            flow_cols = [
                _first_existing_column(lower_to_col, flow_candidates[name])
                for name in ["c", "h", "n"]
            ]
            flow_cols = [col for col in flow_cols if col is not None]
            if not flow_cols:
                raise ValueError(f"Cannot find MetroFlow flow columns in: {list(df.columns)}")
            normalized["flow"] = df[flow_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    return normalized


def load_trip_tables(args: argparse.Namespace) -> pd.DataFrame:
    defaults = SOURCE_DEFAULTS[args.source]
    time_col = args.time_col or defaults["time_col"]
    origin_col = args.origin_col or defaults["origin_col"]
    dest_col = args.destination_col or defaults["destination_col"]
    flow_col = args.flow_col or defaults["flow_col"]

    columns = [time_col, origin_col, dest_col]
    if flow_col:
        columns.append(flow_col)
    if args.source in {"mta_subway_od", "metroflow"}:
        columns = None

    frames = []
    remaining = args.max_rows
    for path in iter_input_files(args.input):
        frame = read_table(path, columns=columns, max_rows=remaining)
        if args.source == "mta_subway_od":
            frame = normalize_mta_subway_columns(frame)
            time_col, origin_col, dest_col, flow_col = "time", "origin", "destination", "flow"
        elif args.source == "metroflow":
            frame = normalize_metroflow_columns(frame, flow_type=args.metroflow_flow)
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
    numeric_time = pd.to_numeric(df["time"], errors="coerce")
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if numeric_time.notna().all():
        df = df.assign(time=numeric_time.astype(int).astype(str), _time_order=numeric_time.astype(float))
    elif parsed_time.notna().all():
        floored_time = parsed_time.dt.floor(freq)
        df = df.assign(time=floored_time.astype(str), _time_order=floored_time.astype("int64"))
    else:
        df = df.assign(time=df["time"].astype(str))
        ordered = {value: idx for idx, value in enumerate(sorted(df["time"].unique()))}
        df["_time_order"] = df["time"].map(ordered).astype(float)

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

    times = (
        df[["time", "_time_order"]]
        .drop_duplicates()
        .sort_values(["_time_order", "time"])["time"]
        .tolist()
    )
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
    parser.add_argument("--input", nargs="+", default=None, help="CSV, ZIP, parquet file, or directory.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--freq", type=str, default="30min")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--origin-col", type=str, default=None)
    parser.add_argument("--destination-col", type=str, default=None)
    parser.add_argument("--flow-col", type=str, default=None)
    parser.add_argument("--metroflow-flow", choices=["total", "c", "h", "n"], default="total")
    parser.add_argument("--drop-self-loops", action="store_true")
    parser.add_argument("--print-sources", action="store_true")
    args = parser.parse_args()

    if args.print_sources:
        print(json.dumps(SOURCE_NOTES, indent=2, ensure_ascii=False))
        return

    if not args.input:
        raise ValueError("--input is required unless --print-sources is used.")
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --print-sources is used.")

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
