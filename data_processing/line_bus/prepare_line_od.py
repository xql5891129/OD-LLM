from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


CARD_USECOLS = {
    "TXNDATE",
    "LINENO",
    "direction",
    "boarding_station_name",
    "boarding_station_num",
    "alighting_station_name",
    "alighting_station_num",
}


@dataclass
class CardStats:
    min_time: pd.Timestamp
    max_time: pd.Timestamp
    total_rows: int
    valid_rows: int
    invalid_rows: int


def read_csv_auto(path: Path, *args: object, **kwargs: object):
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, *args, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, *args, **kwargs)


def normalize_station(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", "", text)
    return text


def card_files(region_dir: Path) -> list[Path]:
    return sorted((region_dir / "card_data").glob("*/*.csv"))


def scan_cards(files: list[Path], chunksize: int, show_progress: bool) -> CardStats:
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    total_rows = 0
    valid_rows = 0
    iterator = tqdm(files, desc="scan card time", unit="file") if show_progress else files
    for path in iterator:
        reader = read_csv_auto(path, usecols=["TXNDATE"], chunksize=chunksize, on_bad_lines="skip")
        for chunk in reader:
            total_rows += int(len(chunk))
            tx_time = pd.to_datetime(chunk["TXNDATE"], errors="coerce")
            valid = tx_time.dropna()
            valid_rows += int(len(valid))
            if valid.empty:
                continue
            local_min = valid.min()
            local_max = valid.max()
            min_time = local_min if min_time is None else min(min_time, local_min)
            max_time = local_max if max_time is None else max(max_time, local_max)
    if min_time is None or max_time is None:
        raise ValueError("No valid TXNDATE values found in card files.")
    return CardStats(
        min_time=pd.Timestamp(min_time),
        max_time=pd.Timestamp(max_time),
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid_rows=total_rows - valid_rows,
    )


def build_time_index(
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
    interval_minutes: int,
    start_time: str | None,
    end_time: str | None,
) -> pd.DatetimeIndex:
    freq = f"{int(interval_minutes)}min"
    start = pd.Timestamp(start_time) if start_time else pd.Timestamp(min_time).floor(freq)
    if end_time:
        end = pd.Timestamp(end_time) - pd.Timedelta(minutes=interval_minutes)
        end = end.floor(freq)
    else:
        end = pd.Timestamp(max_time).floor(freq)
    return pd.date_range(start=start, end=end, freq=freq)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return ((values - mean) / std).astype(np.float32)


def build_weather(region_dir: Path, out_dir: Path, times: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    weather_files = sorted(region_dir.glob("weather*.csv"))
    weather_cols = [
        "lon",
        "lat",
        "pressure_pa",
        "temperature_c",
        "dewpoint_c",
        "precip_mm",
        "wind_speed_ms",
        "wind_dir_deg",
        "solar_mj_m2",
    ]
    if not weather_files:
        weather = pd.DataFrame(0.0, index=times, columns=weather_cols)
    else:
        frames = []
        for path in weather_files:
            raw = read_csv_auto(path)
            if raw.shape[1] < 10:
                continue
            frame = pd.DataFrame(
                {
                    "time": pd.to_datetime(raw.iloc[:, 2], errors="coerce"),
                    "lon": pd.to_numeric(raw.iloc[:, 0], errors="coerce"),
                    "lat": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
                    "pressure_pa": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
                    "temperature_c": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
                    "dewpoint_c": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
                    "precip_mm": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
                    "wind_speed_ms": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
                    "wind_dir_deg": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
                    "solar_mj_m2": pd.to_numeric(raw.iloc[:, 9], errors="coerce"),
                }
            )
            frames.append(frame.dropna(subset=["time"]))
        if frames:
            weather = pd.concat(frames, ignore_index=True)
            weather = weather.groupby("time", as_index=True)[weather_cols].mean().sort_index()
            weather = weather.reindex(weather.index.union(times)).sort_index()
            weather = weather.interpolate(method="time").ffill().bfill().reindex(times)
        else:
            weather = pd.DataFrame(0.0, index=times, columns=weather_cols)
    weather = weather.astype(np.float32)
    weather.to_csv(out_dir / "weather.csv", index_label="time", encoding="utf-8-sig")
    np.save(out_dir / "weather.npy", weather[weather_cols].to_numpy(np.float32))
    with (out_dir / "weather_columns.json").open("w", encoding="utf-8") as f:
        json.dump(weather_cols, f, ensure_ascii=False, indent=2)
    return weather, weather_cols


def build_time_features(
    out_dir: Path,
    times: pd.DatetimeIndex,
    weather: pd.DataFrame,
    weather_cols: list[str],
) -> None:
    minute_of_day = times.hour.to_numpy() * 60 + times.minute.to_numpy()
    day_phase = 2 * np.pi * minute_of_day / 1440.0
    dow = times.dayofweek.to_numpy()
    week_phase = 2 * np.pi * dow / 7.0
    calendar = np.stack(
        [
            np.sin(day_phase),
            np.cos(day_phase),
            np.sin(week_phase),
            np.cos(week_phase),
            (dow >= 5).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    wind_dir = np.deg2rad(weather["wind_dir_deg"].to_numpy(np.float32))
    weather_features = np.stack(
        [
            _zscore(weather["pressure_pa"].to_numpy(np.float32)),
            _zscore(weather["temperature_c"].to_numpy(np.float32)),
            _zscore(weather["dewpoint_c"].to_numpy(np.float32)),
            _zscore(weather["precip_mm"].to_numpy(np.float32)),
            _zscore(weather["wind_speed_ms"].to_numpy(np.float32)),
            np.sin(wind_dir).astype(np.float32),
            np.cos(wind_dir).astype(np.float32),
            _zscore(weather["solar_mj_m2"].to_numpy(np.float32)),
            (weather["precip_mm"].to_numpy(np.float32) > 0.05).astype(np.float32),
        ],
        axis=-1,
    )
    features = np.concatenate([calendar, weather_features], axis=-1).astype(np.float32)
    columns = [
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "weather_pressure_pa_z",
        "weather_temperature_c_z",
        "weather_dewpoint_c_z",
        "weather_precip_mm_z",
        "weather_wind_speed_ms_z",
        "weather_wind_dir_sin",
        "weather_wind_dir_cos",
        "weather_solar_mj_m2_z",
        "weather_is_rainy",
    ]
    np.save(out_dir / "time_features.npy", features)
    with (out_dir / "time_features_columns.json").open("w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=False, indent=2)


def load_line_stops(region_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((region_dir / "line").glob("*.csv")):
        df = read_csv_auto(path)
        if len(df.columns) >= 8:
            rename = {
                df.columns[0]: "line_name",
                df.columns[1]: "line_no",
                df.columns[2]: "direction",
                df.columns[3]: "station_name",
                df.columns[4]: "stop_no",
                df.columns[5]: "type",
                df.columns[6]: "lng",
                df.columns[7]: "lat",
            }
            df = df.rename(columns=rename)
        for col in ["line_no", "direction", "station_name", "stop_no", "type", "lng", "lat"]:
            if col not in df.columns:
                df[col] = np.nan
        df["source_file"] = path.name
        frames.append(df[["line_no", "direction", "station_name", "stop_no", "type", "lng", "lat", "source_file"]])
    if not frames:
        return pd.DataFrame(columns=["line_no", "direction", "station_name", "stop_no", "type", "lng", "lat"])
    line_df = pd.concat(frames, ignore_index=True)
    line_df["line_no"] = line_df["line_no"].astype(str)
    line_df["direction"] = pd.to_numeric(line_df["direction"], errors="coerce")
    line_df["stop_no"] = pd.to_numeric(line_df["stop_no"], errors="coerce")
    line_df["station_name"] = line_df["station_name"].map(normalize_station)
    line_df["lng"] = pd.to_numeric(line_df["lng"], errors="coerce")
    line_df["lat"] = pd.to_numeric(line_df["lat"], errors="coerce")
    return line_df


REGION_SLUGS = {
    "光电园": "guangdianyuan",
    "大学城": "daxuecheng",
    "解放碑": "jiefangbei",
    "guangdianyuan": "guangdianyuan",
    "daxuecheng": "daxuecheng",
    "jiefangbei": "jiefangbei",
}


@dataclass
class LineEntry:
    line_no: str
    direction: int
    line_dir: Path
    nodes: pd.DataFrame
    stop_to_id: dict[int, int]
    name_to_id: dict[str, int]
    mask: np.ndarray
    od: np.memmap
    counted_rows: int = 0
    filtered_rows: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return self.line_no, self.direction

    @property
    def num_stops(self) -> int:
        return int(len(self.nodes))

    @property
    def total_flow(self) -> float:
        return float(self.od.sum())


def region_slug(region_name: str) -> str:
    return REGION_SLUGS.get(region_name, re.sub(r"[^0-9A-Za-z]+", "_", region_name).strip("_").lower() or "region")


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "line"


def line_files_from_cards(files: list[Path]) -> set[str]:
    line_nos = set()
    for path in files:
        parent = path.parent.name
        if parent:
            line_nos.add(str(parent))
    return line_nos


def load_poi_records(region_dir: Path, top_categories: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for path in sorted((region_dir / "poi").glob("*.csv")):
        df = read_csv_auto(path)
        if len(df.columns) >= 11:
            rename = {
                df.columns[1]: "line_no",
                df.columns[2]: "direction",
                df.columns[3]: "stop_no",
                df.columns[4]: "station_name",
                df.columns[5]: "poi_type",
                df.columns[-1]: "distance_m",
            }
            df = df.rename(columns=rename)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["line_no", "direction", "stop_no", "station_name", "poi_type", "distance_m"]), []

    poi = pd.concat(frames, ignore_index=True)
    for col in ["line_no", "direction", "stop_no", "station_name", "poi_type", "distance_m"]:
        if col not in poi.columns:
            poi[col] = np.nan

    def first_series(frame: pd.DataFrame, column: str) -> pd.Series:
        value = frame[column]
        if isinstance(value, pd.DataFrame):
            return value.iloc[:, 0]
        return value

    poi["line_no"] = first_series(poi, "line_no").astype(str)
    poi["direction"] = pd.to_numeric(first_series(poi, "direction"), errors="coerce").astype("Int64")
    poi["stop_no"] = pd.to_numeric(first_series(poi, "stop_no"), errors="coerce").astype("Int64")
    poi["station_name"] = first_series(poi, "station_name").map(normalize_station)
    poi["distance_m"] = pd.to_numeric(first_series(poi, "distance_m"), errors="coerce")
    poi_type = first_series(poi, "poi_type").fillna("unknown").astype("string").fillna("unknown")
    poi["major_category"] = poi_type.str.split(";", n=1).str[0].replace("", "unknown").fillna("unknown")
    categories = poi["major_category"].value_counts().head(int(top_categories)).index.astype(str).tolist()
    return poi, categories


def build_line_poi_features(
    poi: pd.DataFrame,
    categories: list[str],
    nodes: pd.DataFrame,
    line_no: str,
    direction: int,
    out_dir: Path,
) -> None:
    rows = []
    for _, node in nodes.iterrows():
        station_name = str(node["station_name"])
        stop_no = int(node["stop_no"])
        group = poi[
            (poi["line_no"].astype(str) == str(line_no))
            & (poi["direction"].astype("Int64") == int(direction))
            & (poi["stop_no"].astype("Int64") == stop_no)
        ]
        if group.empty:
            group = poi[poi["station_name"] == station_name]
        row = {
            "node_id": int(node["node_id"]),
            "stop_no": stop_no,
            "station_name": station_name,
            "poi_total_log1p": float(np.log1p(len(group))),
            "poi_min_distance_m": float(group["distance_m"].min()) if group["distance_m"].notna().any() else 0.0,
            "poi_mean_distance_m": float(group["distance_m"].mean()) if group["distance_m"].notna().any() else 0.0,
        }
        counts = group["major_category"].astype(str).value_counts().to_dict()
        for category in categories:
            row[f"poi_count_{category}_log1p"] = float(np.log1p(counts.get(category, 0)))
        rows.append(row)

    feature_df = pd.DataFrame(rows).fillna(0.0)
    feature_cols = [col for col in feature_df.columns if col not in {"node_id", "stop_no", "station_name"}]
    feature_df.to_csv(out_dir / "poi_features.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "poi_features.npy", feature_df[feature_cols].to_numpy(np.float32))
    with (out_dir / "poi_feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)


def make_line_entries(
    line_df: pd.DataFrame,
    out_dir: Path,
    times: pd.DatetimeIndex,
    line_nos: set[str] | None,
    min_stops: int,
    max_lines: int,
) -> dict[tuple[str, int], LineEntry]:
    entries: dict[tuple[str, int], LineEntry] = {}
    if line_df.empty:
        raise ValueError("line folder has no usable line stop CSV files.")

    line_df = line_df.dropna(subset=["line_no", "direction", "stop_no"]).copy()
    if line_nos:
        line_df = line_df[line_df["line_no"].astype(str).isin(line_nos)].copy()

    groups = []
    for (line_no, direction), group in line_df.groupby(["line_no", "direction"], sort=True):
        stops = group.sort_values("stop_no").drop_duplicates(subset=["stop_no"], keep="first").copy()
        stops = stops[stops["station_name"].astype(str) != ""].copy()
        if len(stops) < min_stops:
            continue
        groups.append((str(line_no), int(direction), stops))
    groups = sorted(groups, key=lambda item: (item[0], item[1]))
    if max_lines > 0:
        groups = groups[:max_lines]

    for line_no, direction, stops in groups:
        line_dir_name = f"line_{safe_name(line_no)}_dir{direction}"
        line_dir = out_dir / line_dir_name
        line_dir.mkdir(parents=True, exist_ok=True)
        nodes = stops[["line_no", "direction", "stop_no", "station_name", "lng", "lat", "type"]].copy()
        nodes = nodes.reset_index(drop=True)
        nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
        nodes.to_csv(line_dir / "nodes.csv", index=False, encoding="utf-8-sig")

        stop_to_id = {int(row.stop_no): int(row.node_id) for row in nodes.itertuples(index=False)}
        name_to_id = {str(row.station_name): int(row.node_id) for row in nodes.itertuples(index=False)}
        stop_count = len(nodes)
        mask = np.triu(np.ones((stop_count, stop_count), dtype=bool), k=1)
        np.save(line_dir / "mask.npy", mask)
        od = np.lib.format.open_memmap(
            line_dir / "od.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(times), stop_count, stop_count),
        )
        od[:] = 0.0
        entries[(line_no, direction)] = LineEntry(
            line_no=line_no,
            direction=direction,
            line_dir=line_dir,
            nodes=nodes,
            stop_to_id=stop_to_id,
            name_to_id=name_to_id,
            mask=mask,
            od=od,
        )
    return entries


def count_line_od(
    files: list[Path],
    entries: dict[tuple[str, int], LineEntry],
    times: pd.DatetimeIndex,
    interval_minutes: int,
    chunksize: int,
    show_progress: bool,
) -> None:
    start = times[0]
    interval_ns = pd.Timedelta(minutes=interval_minutes).value
    iterator = tqdm(files, desc="build line OD", unit="file") if show_progress else files
    for path in iterator:
        reader = read_csv_auto(
            path,
            usecols=lambda col: col in CARD_USECOLS,
            chunksize=chunksize,
            on_bad_lines="skip",
        )
        for chunk in reader:
            chunk["line_no"] = chunk["LINENO"].astype(str)
            chunk["direction_i"] = pd.to_numeric(chunk["direction"], errors="coerce").astype("Int64")
            chunk["origin_stop"] = pd.to_numeric(chunk["boarding_station_num"], errors="coerce").astype("Int64")
            chunk["dest_stop"] = pd.to_numeric(chunk["alighting_station_num"], errors="coerce").astype("Int64")
            chunk["origin_name"] = chunk["boarding_station_name"].map(normalize_station)
            chunk["dest_name"] = chunk["alighting_station_name"].map(normalize_station)
            txn_time = pd.to_datetime(chunk["TXNDATE"], errors="coerce")
            txn_ns = txn_time.to_numpy(dtype="datetime64[ns]").astype("int64")
            bin_idx = pd.Series((txn_ns - start.value) // interval_ns, index=chunk.index, dtype="Int64")
            time_valid = txn_time.notna() & bin_idx.notna() & (bin_idx >= 0) & (bin_idx < len(times))

            for (line_no, direction), group_idx in chunk[time_valid].groupby(["line_no", "direction_i"], sort=False).groups.items():
                key = (str(line_no), int(direction))
                entry = entries.get(key)
                if entry is None:
                    continue
                group = chunk.loc[group_idx].copy()
                group["t"] = bin_idx.loc[group_idx].astype(np.int64)
                group["o"] = group["origin_stop"].map(entry.stop_to_id)
                group["d"] = group["dest_stop"].map(entry.stop_to_id)
                missing = group["o"].isna() | group["d"].isna()
                if missing.any():
                    group.loc[missing, "o"] = group.loc[missing, "origin_name"].map(entry.name_to_id)
                    group.loc[missing, "d"] = group.loc[missing, "dest_name"].map(entry.name_to_id)
                valid = group["o"].notna() & group["d"].notna()
                valid = valid & (group["o"].astype(float) < group["d"].astype(float))
                entry.filtered_rows += int((~valid).sum())
                if not valid.any():
                    continue
                compact = pd.DataFrame(
                    {
                        "t": group.loc[valid, "t"].astype(np.int64).to_numpy(),
                        "o": group.loc[valid, "o"].astype(np.int64).to_numpy(),
                        "d": group.loc[valid, "d"].astype(np.int64).to_numpy(),
                    }
                )
                grouped = compact.groupby(["t", "o", "d"], sort=False).size().reset_index(name="flow")
                entry.od[
                    grouped["t"].to_numpy(),
                    grouped["o"].to_numpy(),
                    grouped["d"].to_numpy(),
                ] += grouped["flow"].to_numpy(np.float32)
                entry.counted_rows += int(grouped["flow"].sum())

    for entry in entries.values():
        entry.od.flush()


def write_line_metadata(
    out_dir: Path,
    entries: dict[tuple[str, int], LineEntry],
    min_total_flow: float,
) -> None:
    rows = []
    for idx, entry in enumerate(sorted(entries.values(), key=lambda item: (item.line_no, item.direction))):
        total_flow = entry.total_flow
        enabled = total_flow >= float(min_total_flow)
        metadata = {
            "line_no": entry.line_no,
            "direction": entry.direction,
            "num_stops": entry.num_stops,
            "valid_od_pairs": int(entry.mask.sum()),
            "total_flow": total_flow,
            "counted_rows": int(entry.counted_rows),
            "filtered_rows": int(entry.filtered_rows),
            "enabled": bool(enabled),
        }
        with (entry.line_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        rows.append(
            {
                "line_index": idx,
                "line_dir": entry.line_dir.name,
                "relative_dir": entry.line_dir.name,
                **metadata,
            }
        )
    index_df = pd.DataFrame(rows).sort_values(["enabled", "total_flow"], ascending=[False, False])
    index_df.to_csv(out_dir / "line_index.csv", index=False, encoding="utf-8-sig")


def prepare_region(args: argparse.Namespace, region_dir: Path) -> None:
    slug = region_slug(region_dir.name)
    out_dir = args.output_root / slug / f"{args.interval_minutes}min"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = card_files(region_dir)
    if args.max_card_files > 0:
        files = files[: args.max_card_files]
    if not files:
        raise FileNotFoundError(f"No card CSV files under {region_dir / 'card_data'}")

    line_nos = set(str(item) for item in args.line_nos) if args.line_nos else None
    if line_nos is None and args.use_card_line_dirs:
        line_nos = line_files_from_cards(files)

    print(f"\n===== Preparing line OD {region_dir.name} -> {out_dir} =====")
    stats = scan_cards(files, args.chunksize, show_progress=not args.no_progress)
    times = build_time_index(stats.min_time, stats.max_time, args.interval_minutes, args.start_time, args.end_time)
    pd.DataFrame({"time": times.astype(str)}).to_csv(out_dir / "times.csv", index=False, encoding="utf-8-sig")
    weather, weather_cols = build_weather(region_dir, out_dir, times)
    build_time_features(out_dir, times, weather, weather_cols)

    line_df = load_line_stops(region_dir)
    entries = make_line_entries(
        line_df=line_df,
        out_dir=out_dir,
        times=times,
        line_nos=line_nos,
        min_stops=int(args.min_stops),
        max_lines=int(args.max_lines),
    )
    if not entries:
        raise ValueError(f"No line-direction entries created for {region_dir}")

    poi, categories = load_poi_records(region_dir, top_categories=int(args.poi_top_categories))
    for entry in entries.values():
        build_line_poi_features(poi, categories, entry.nodes, entry.line_no, entry.direction, entry.line_dir)

    count_line_od(
        files=files,
        entries=entries,
        times=times,
        interval_minutes=int(args.interval_minutes),
        chunksize=int(args.chunksize),
        show_progress=not args.no_progress,
    )
    write_line_metadata(out_dir, entries, min_total_flow=float(args.min_total_flow))

    metadata = {
        "dataset_type": "line_bus",
        "region": region_dir.name,
        "region_slug": slug,
        "raw_dir": str(region_dir),
        "interval_minutes": int(args.interval_minutes),
        "num_lines": int(len(entries)),
        "enabled_lines": int(sum(entry.total_flow >= float(args.min_total_flow) for entry in entries.values())),
        "min_total_flow": float(args.min_total_flow),
        "min_stops": int(args.min_stops),
        "num_time_steps": int(len(times)),
        "start_time": str(times[0]),
        "end_time_inclusive": str(times[-1]),
        "card_csv_files": int(len(files)),
        "max_card_files": int(args.max_card_files),
        "scan_total_rows": int(stats.total_rows),
        "scan_valid_rows": int(stats.valid_rows),
        "scan_invalid_rows": int(stats.invalid_rows),
        "time_features": "time_features.npy",
        "line_index": "line_index.csv",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare line-direction bus OD tensors from raw card records.")
    parser.add_argument("--root", type=Path, default=Path("data") / "Busdata")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "processed_line_bus")
    parser.add_argument("--regions", nargs="*", default=None, help="Region folder names. Default: all regions.")
    parser.add_argument("--line-nos", nargs="*", default=None, help="Optional line numbers to keep.")
    parser.add_argument("--interval-minutes", type=int, default=60, choices=[30, 60])
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--min-stops", type=int, default=4)
    parser.add_argument("--min-total-flow", type=float, default=100.0)
    parser.add_argument("--poi-top-categories", type=int, default=30)
    parser.add_argument("--max-lines", type=int, default=0, help="Debug only. Keep at most this many line-directions.")
    parser.add_argument("--max-card-files", type=int, default=0, help="Debug only. Process at most this many card CSVs per region.")
    parser.add_argument("--start-time", type=str, default=None)
    parser.add_argument("--end-time", type=str, default=None, help="Exclusive end time, e.g. 2022-01-01 00:00:00.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--use-card-line-dirs", action="store_true", help="Restrict to line folders that exist under card_data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Busdata root not found: {args.root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.regions) if args.regions else None
    regions = sorted([path for path in args.root.iterdir() if path.is_dir()], key=lambda path: path.name)
    for region_dir in regions:
        if selected is not None and region_dir.name not in selected and region_slug(region_dir.name) not in selected:
            continue
        prepare_region(args, region_dir)


if __name__ == "__main__":
    main()
