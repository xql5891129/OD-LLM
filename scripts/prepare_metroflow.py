from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FLOW_COLUMNS = {
    "total": ["Flow"],
    "c": ["CFlow"],
    "hbo": ["HBOFlow"],
    "nhb": ["NHBFlow"],
    "components": ["CFlow", "HBOFlow", "NHBFlow"],
}

TIME_FEATURE_COLUMNS = [
    "sin_day",
    "cos_day",
    "sin_week",
    "cos_week",
    "is_weekend",
    "is_workday",
    "temperature_2m",
    "apparent_temperature",
    "rain",
    "wind_speed_10m",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def default_paths(raw_dir: Path) -> dict[str, Path]:
    return {
        "od": raw_dir / "metroData_ODFlow.csv",
        "inout": raw_dir / "metroData_InOutFlow.csv",
        "station": raw_dir / "stationInfo.csv",
        "weather": raw_dir / "MetaData" / "shanghai_weatherHourly.csv",
        "calendar": raw_dir / "MetaData" / "workday_calendar.csv",
    }


def parse_metro_datetime(df: pd.DataFrame) -> pd.Series:
    date = df["date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    start = df["startTime"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return pd.to_datetime(date + start, format="%Y%m%d%H%M%S", errors="coerce")


def filter_time_range(dt: pd.Series, start_date: str | None, end_date: str | None) -> pd.Series:
    mask = dt.notna()
    if start_date:
        mask &= dt >= pd.Timestamp(start_date)
    if end_date:
        mask &= dt < pd.Timestamp(end_date)
    return mask


def read_station_info(path: Path) -> pd.DataFrame:
    station = clean_columns(pd.read_csv(path))
    if "stationID" not in station.columns:
        raise ValueError(f"stationInfo.csv must contain stationID, got {station.columns.tolist()}")
    station["stationID"] = pd.to_numeric(station["stationID"], errors="coerce").astype("Int64")
    station = station.dropna(subset=["stationID"]).copy()
    station["stationID"] = station["stationID"].astype(int)
    if "Unnamed: 0" in station.columns:
        station = station.drop(columns=["Unnamed: 0"])
    return station.drop_duplicates("stationID")


def scan_inout(
    inout_path: Path,
    freq: str,
    start_date: str | None,
    end_date: str | None,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.Series]:
    times: set[pd.Timestamp] = set()
    station_scores: dict[int, float] = {}
    usecols = ["date", "startTime", "station", "inFlow", "outFlow"]

    for chunk_id, chunk in enumerate(pd.read_csv(inout_path, chunksize=chunksize, skipinitialspace=True, usecols=usecols), 1):
        chunk = clean_columns(chunk)
        dt = parse_metro_datetime(chunk)
        mask = filter_time_range(dt, start_date, end_date)
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            continue

        bucket = dt.loc[mask].dt.floor(freq)
        times.update(bucket.dropna().tolist())

        station = pd.to_numeric(chunk["station"], errors="coerce")
        flow = (
            pd.to_numeric(chunk["inFlow"], errors="coerce").fillna(0.0)
            + pd.to_numeric(chunk["outFlow"], errors="coerce").fillna(0.0)
        )
        grouped = flow.groupby(station).sum()
        for station_id, value in grouped.items():
            if pd.isna(station_id):
                continue
            station_scores[int(station_id)] = station_scores.get(int(station_id), 0.0) + float(value)

        if chunk_id % 10 == 0:
            print(f"[scan_inout] chunks={chunk_id}, unique_times={len(times)}, scored_stations={len(station_scores)}")

    if not times:
        raise ValueError("No valid MetroFlow time slots found in InOutFlow file.")

    sorted_times = sorted(times)
    time_df = pd.DataFrame({"time_idx": range(len(sorted_times)), "time": sorted_times})
    time_df["time_key"] = time_df["time"].astype("int64")
    score_series = pd.Series(station_scores, dtype=float).sort_values(ascending=False)
    return time_df, score_series


def select_nodes(
    station_info: pd.DataFrame,
    station_scores: pd.Series,
    top_n: int | None,
    station_order: str,
) -> pd.DataFrame:
    station = station_info.copy()
    station["station_flow"] = station["stationID"].map(station_scores).fillna(0.0)
    station["flow_rank"] = station["station_flow"].rank(method="first", ascending=False).astype(int)

    if top_n is not None and top_n > 0:
        keep_ids = set(station.sort_values("station_flow", ascending=False).head(top_n)["stationID"].tolist())
        station = station[station["stationID"].isin(keep_ids)].copy()

    if station_order == "flow_rank":
        station = station.sort_values(["station_flow", "stationID"], ascending=[False, True])
    else:
        station = station.sort_values("stationID")
    station = station.reset_index(drop=True)
    station.insert(0, "node_idx", np.arange(len(station), dtype=int))
    return station


def build_time_features(
    time_df: pd.DataFrame,
    weather_path: Path | None,
    calendar_path: Path | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    times = pd.to_datetime(time_df["time"])
    minute_of_day = times.dt.hour.to_numpy() * 60 + times.dt.minute.to_numpy()
    day_phase = 2 * np.pi * minute_of_day / 1440.0
    dow = times.dt.dayofweek.to_numpy()
    week_phase = 2 * np.pi * dow / 7.0

    aligned = time_df[["time_idx", "time"]].copy()
    aligned["date_int"] = times.dt.strftime("%Y%m%d").astype(int)
    aligned["hour_time"] = times.dt.floor("h")

    if calendar_path and calendar_path.exists():
        calendar = clean_columns(pd.read_csv(calendar_path))
        work_col = "isWorday" if "isWorday" in calendar.columns else "isWorkday"
        calendar["date_int"] = pd.to_numeric(calendar["date"], errors="coerce").astype("Int64")
        calendar = calendar.dropna(subset=["date_int"]).copy()
        calendar["date_int"] = calendar["date_int"].astype(int)
        calendar[work_col] = pd.to_numeric(calendar[work_col], errors="coerce").fillna(0).astype(int)
        aligned = aligned.merge(calendar[["date_int", work_col]], on="date_int", how="left")
        fallback_workday = pd.Series((dow < 5).astype(int), index=aligned.index)
        is_workday = aligned[work_col].where(aligned[work_col].notna(), fallback_workday).to_numpy(dtype=np.float32)
    else:
        is_workday = (dow < 5).astype(np.float32)

    is_weekend = 1.0 - is_workday
    weather_values = np.zeros((len(time_df), 4), dtype=np.float32)
    weather_cols = ["temperature_2m", "apparent_temperature", "rain", "wind_speed_10m"]
    if weather_path and weather_path.exists():
        weather = clean_columns(pd.read_csv(weather_path))
        weather["hour_time"] = pd.to_datetime(weather["date"], format="%Y%m%d %H:%M:%S", errors="coerce")
        for col in weather_cols:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
        aligned = aligned.merge(weather[["hour_time", *weather_cols]], on="hour_time", how="left")
        aligned[weather_cols] = aligned[weather_cols].ffill().bfill().fillna(0.0)
        weather_values = aligned[weather_cols].to_numpy(dtype=np.float32)

    features = np.column_stack(
        [
            np.sin(day_phase),
            np.cos(day_phase),
            np.sin(week_phase),
            np.cos(week_phase),
            is_weekend,
            is_workday,
            weather_values,
        ]
    ).astype(np.float32)

    export_time = time_df[["time_idx", "time"]].copy()
    export_time["date"] = times.dt.strftime("%Y%m%d")
    export_time["start_time"] = times.dt.strftime("%H:%M:%S")
    export_time["hour"] = times.dt.hour
    export_time["day_of_week"] = times.dt.dayofweek
    export_time["is_weekend"] = is_weekend
    export_time["is_workday"] = is_workday
    for idx, col in enumerate(weather_cols):
        export_time[col] = weather_values[:, idx]
    return features, export_time


def flow_values(chunk: pd.DataFrame, flow_type: str) -> pd.Series:
    if flow_type == "total":
        if "Flow" in chunk.columns:
            return pd.to_numeric(chunk["Flow"], errors="coerce").fillna(0.0)
        return sum(pd.to_numeric(chunk[col], errors="coerce").fillna(0.0) for col in FLOW_COLUMNS["components"])
    cols = FLOW_COLUMNS[flow_type]
    return sum(pd.to_numeric(chunk[col], errors="coerce").fillna(0.0) for col in cols)


def write_od_memmap(
    od_path: Path,
    od_file: Path,
    time_df: pd.DataFrame,
    node_df: pd.DataFrame,
    freq: str,
    flow_type: str,
    start_date: str | None,
    end_date: str | None,
    chunksize: int,
    drop_self_loops: bool,
    max_rows: int | None,
) -> tuple[float, int]:
    shape = (len(time_df), len(node_df), len(node_df))
    od = np.lib.format.open_memmap(od_path, mode="w+", dtype=np.float32, shape=shape)
    od[:] = 0.0

    time_to_idx = dict(zip(time_df["time_key"].astype(np.int64), time_df["time_idx"].astype(int)))
    node_to_idx = dict(zip(node_df["stationID"].astype(int), node_df["node_idx"].astype(int)))
    usecols = ["date", "startTime", "originStation", "destinationStation", *FLOW_COLUMNS["components"]]
    if flow_type == "total":
        usecols.append("Flow")
    usecols = sorted(set(usecols))

    total_flow = 0.0
    processed = 0
    kept = 0
    for chunk_id, chunk in enumerate(pd.read_csv(od_file, chunksize=chunksize, skipinitialspace=True, usecols=usecols), 1):
        chunk = clean_columns(chunk)
        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        processed += len(chunk)

        dt = parse_metro_datetime(chunk)
        mask = filter_time_range(dt, start_date, end_date)
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            continue

        bucket_key = dt.loc[mask].dt.floor(freq).astype("int64")
        chunk["time_idx"] = bucket_key.map(time_to_idx)
        chunk["origin_idx"] = pd.to_numeric(chunk["originStation"], errors="coerce").map(node_to_idx)
        chunk["destination_idx"] = pd.to_numeric(chunk["destinationStation"], errors="coerce").map(node_to_idx)
        chunk["flow"] = flow_values(chunk, flow_type)
        chunk = chunk.dropna(subset=["time_idx", "origin_idx", "destination_idx"])
        if drop_self_loops:
            chunk = chunk[chunk["origin_idx"] != chunk["destination_idx"]]
        if chunk.empty:
            continue

        grouped = (
            chunk.groupby(["time_idx", "origin_idx", "destination_idx"], as_index=False)["flow"]
            .sum()
            .astype({"time_idx": int, "origin_idx": int, "destination_idx": int})
        )
        t = grouped["time_idx"].to_numpy(dtype=np.int64)
        o = grouped["origin_idx"].to_numpy(dtype=np.int64)
        d = grouped["destination_idx"].to_numpy(dtype=np.int64)
        f = grouped["flow"].to_numpy(dtype=np.float32)
        np.add.at(od, (t, o, d), f)
        total_flow += float(f.sum())
        kept += len(chunk)

        if chunk_id % 10 == 0:
            print(f"[write_od] chunks={chunk_id}, processed_rows={processed:,}, kept_rows={kept:,}, total_flow={total_flow:.0f}")

    od.flush()
    return total_flow, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Shanghai MetroFlow as OD-LLM od.npy data.")
    parser.add_argument("--raw-dir", type=str, default="data/MetroFlow")
    parser.add_argument("--output-dir", type=str, default="data/metroflow_top80")
    parser.add_argument("--top-n", type=int, default=80, help="Use <=0 for all stations.")
    parser.add_argument("--freq", type=str, default="10min")
    parser.add_argument("--flow-type", choices=["total", "c", "hbo", "nhb"], default="total")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--station-order", choices=["station_id", "flow_rank"], default="station_id")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--drop-self-loops", action="store_true")
    parser.add_argument("--skip-nonzero-count", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    paths = default_paths(raw_dir)
    for name, path in paths.items():
        if name in {"weather", "calendar"}:
            continue
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    od_path = output_dir / "od.npy"
    if od_path.exists() and not args.overwrite and not args.profile_only:
        raise FileExistsError(f"{od_path} already exists. Use --overwrite to rebuild it.")

    print("Reading station info and scanning time axis from InOutFlow...")
    station_info = read_station_info(paths["station"])
    time_df, station_scores = scan_inout(
        paths["inout"],
        freq=args.freq,
        start_date=args.start_date,
        end_date=args.end_date,
        chunksize=args.chunksize,
    )
    top_n = None if args.top_n <= 0 else args.top_n
    node_df = select_nodes(station_info, station_scores, top_n=top_n, station_order=args.station_order)
    time_features, export_time_df = build_time_features(time_df, paths["weather"], paths["calendar"])

    print(f"Time steps: {len(time_df):,}")
    print(f"Stations: {len(node_df):,}")
    print(f"Output shape: ({len(time_df):,}, {len(node_df):,}, {len(node_df):,})")
    print(f"Estimated dense float32 size: {len(time_df) * len(node_df) * len(node_df) * 4 / 1024**3:.2f} GiB")

    node_df.to_csv(output_dir / "nodes.csv", index=False)
    export_time_df.to_csv(output_dir / "times.csv", index=False)
    np.save(output_dir / "time_features.npy", time_features)
    with (output_dir / "time_features_columns.json").open("w", encoding="utf-8") as f:
        json.dump(TIME_FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "source": "metroflow",
        "raw_dir": str(raw_dir),
        "shape": [int(len(time_df)), int(len(node_df)), int(len(node_df))],
        "freq": args.freq,
        "top_n": top_n,
        "flow_type": args.flow_type,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "drop_self_loops": args.drop_self_loops,
        "time_features": TIME_FEATURE_COLUMNS,
    }

    if args.profile_only:
        metadata["profile_only"] = True
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print("Profile only: metadata, nodes, times, and time_features were written; od.npy was not created.")
        return

    print("Streaming ODFlow into dense od.npy. This can take a while for the 12GB file...")
    total_flow, kept_rows = write_od_memmap(
        od_path=od_path,
        od_file=paths["od"],
        time_df=time_df,
        node_df=node_df,
        freq=args.freq,
        flow_type=args.flow_type,
        start_date=args.start_date,
        end_date=args.end_date,
        chunksize=args.chunksize,
        drop_self_loops=args.drop_self_loops,
        max_rows=args.max_rows,
    )

    metadata["total_flow"] = total_flow
    metadata["kept_rows"] = kept_rows
    if not args.skip_nonzero_count:
        od = np.load(od_path, mmap_mode="r")
        metadata["nonzero_count"] = int(np.count_nonzero(od))
        metadata["nonzero_ratio"] = float(metadata["nonzero_count"] / od.size)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved OD array: {od_path}")
    print(f"Saved nodes: {output_dir / 'nodes.csv'}")
    print(f"Saved times: {output_dir / 'times.csv'}")
    print(f"Saved time features: {output_dir / 'time_features.npy'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
