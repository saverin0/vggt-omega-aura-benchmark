"""Choose, download and inspect FZI-AURA blocks.

Every dataset path, flag, field and method used here was read from the SDK
source at the pinned commit (pins.AURA_SDK_COMMIT) or from the release
metadata at the pinned dataset revision. Nothing is guessed.

Facts this module relies on:

- The release is published in blocks of up to 20 scenes. A block is the
  smallest download unit.
- The downloader writes dataset.json for the REQUESTED scenes only. Asking
  for one scene fetches its whole block but leaves the 19 neighbours unusable
  by the SDK. So a block is always requested by listing all of its scene IDs.
- A download command must describe the complete dataset root wanted. Adding
  LiDAR later means repeating the camera layer in the same command.
- Road type, weather and speed live in ego/vehicle_signals.parquet, not in
  scene.json. The dataset has no lighting field.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import pins

RELEASE_VERSION = "v1.0"
CHUNKS_FILE = f"metadata/{RELEASE_VERSION}/chunks.parquet"
SCENE_BLOCKS_FILE = f"metadata/{RELEASE_VERSION}/scene_blocks.parquet"
BASE_LAYER = "base_keyframes"
CONTEXT_COLUMNS = ["timestamp_ns", "speed_kph", "road_type", "osm_highway_tag", "weather"]


# ---------------------------------------------------------------- choosing a block

def fetch_release_tables(cache_dir, revision: str = pins.AURA_DATASET_REVISION):
    """Download the two small release tables. Returns (chunks, scene_blocks, hub_files)."""
    from huggingface_hub import HfApi, hf_hub_download

    tables = []
    for filename in (CHUNKS_FILE, SCENE_BLOCKS_FILE):
        path = hf_hub_download(pins.AURA_DATASET_REPO, filename, repo_type="dataset",
                               revision=revision, local_dir=str(cache_dir))
        tables.append(pd.read_parquet(path))
    hub_files = set(HfApi().list_repo_files(pins.AURA_DATASET_REPO, repo_type="dataset", revision=revision))
    return tables[0], tables[1], hub_files


DATASET_INDEX_FILE = f"metadata/{RELEASE_VERSION}/dataset.json"


def fetch_excluded_scene_ids(cache_dir, revision: str = pins.AURA_DATASET_REVISION) -> dict:
    """Scenes the dataset's maintainers exclude, as {scene_id: reason}.

    The release's block table still LISTS these scenes, but the SDK refuses to open them
    (SceneNotFoundError), and rightly so: the reasons are faults such as "PTP synchronization issue;
    objects doubled" or "Too many dropped frames". At the pinned revision there are 8, all in train blocks.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(pins.AURA_DATASET_REPO, DATASET_INDEX_FILE, repo_type="dataset",
                           revision=revision, local_dir=str(cache_dir))
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(entry["scene_id"]): str(entry.get("reason", "")) for entry in index.get("consumer_excluded_scenes", [])}


def with_base_layer(layers) -> list[str]:
    layers = [layer for layer in layers if layer != BASE_LAYER]
    return [BASE_LAYER, *layers]


def available_blocks(chunks: pd.DataFrame, scene_blocks: pd.DataFrame, hub_files, layers) -> pd.DataFrame:
    """Blocks whose every chunk, for every wanted layer, is ready and present on the Hub.

    Mirrors the completeness rule in fzi_aura.download. One row per block with
    its scene count, the size in GB per layer, and the total.
    """
    layers = with_base_layer(layers)
    wanted = chunks[chunks["layer"].isin(layers)].copy()
    wanted["usable"] = (wanted["status"] == "ready") & wanted["chunk_path"].isin(set(hub_files))
    keys = ["split", "scene_block"]

    sizes = (wanted.groupby(keys + ["layer"])["size_bytes"].sum() / 1e9).unstack("layer")
    usable = wanted.groupby(keys + ["layer"])["usable"].all().unstack("layer")
    complete = usable.reindex(columns=layers).fillna(False).all(axis=1)

    table = sizes.reindex(columns=layers).round(2)
    table["total_gb"] = table.sum(axis=1).round(2)
    table.insert(0, "n_scenes", scene_blocks.groupby(keys).size())
    return table[complete].sort_index()


def _block_rows(scene_blocks: pd.DataFrame, split: str, block: int, excluded=()) -> pd.DataFrame:
    rows = scene_blocks[(scene_blocks["split"] == split) & (scene_blocks["scene_block"] == block)]
    if rows.empty:
        raise ValueError(f"no scenes in split={split!r} block={block}")
    rows = rows[~rows["scene_id"].astype(str).isin(set(excluded))]
    return rows.sort_values("scene_ordinal")


def block_scene_ids(scene_blocks: pd.DataFrame, split: str, block: int, excluded=()) -> list[str]:
    """Scene ids of a block, without the scenes the maintainers exclude (pass fetch_excluded_scene_ids())."""
    return _block_rows(scene_blocks, split, block, excluded)["scene_id"].astype(str).tolist()


def block_scene_names(scene_blocks: pd.DataFrame, split: str, block: int, excluded=()) -> list[str]:
    """Folder names of a block's scenes, in the same order as block_scene_ids.

    Taken from the release table, so they are known BEFORE anything is downloaded. That is what lets
    the pipeline look for a block's saved results first. (The name is the scene id with "|" as "_".)
    """
    return _block_rows(scene_blocks, split, block, excluded)["scene"].astype(str).tolist()


# ---------------------------------------------------------------- downloading

def build_download_command(data_root, split: str, scene_ids_file, layers,
                           revision: str = pins.AURA_DATASET_REVISION,
                           jobs: int = 8, dry_run: bool = False) -> list[str]:
    command = [
        sys.executable, "-m", "fzi_aura.download", str(data_root),
        "--revision", revision,
        "--splits", split,
        "--scene-ids-file", str(scene_ids_file),
        "--layers", ",".join(layer for layer in layers if layer != BASE_LAYER) or BASE_LAYER,
        "--jobs", str(jobs),
        "--verify",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def download_block(data_root, split: str, block: int, scene_ids: list[str], layers,
                   revision: str = pins.AURA_DATASET_REVISION, jobs: int = 8, dry_run: bool = False,
                   fast_unpack: str = "auto") -> int:
    """Download and extract one block. `layers` must list EVERY layer wanted in data_root.

    fast_unpack: "auto" decompresses the LiDAR files on all cores when the `xz` tool allows it (see
    fast_download.py; 5.3x faster on Colab, byte-identical files), "never" is the toolkit's own single-core way.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if fast_unpack not in ("auto", "never"):
        raise ValueError(f"fast_unpack must be 'auto' or 'never', not {fast_unpack!r}")
    if fast_unpack == "auto" and not dry_run:
        from . import fast_download

        if fast_download.parallel_xz_available():
            print(f"  downloading with the toolkit, decompressing with xz on all {os.cpu_count()} cores")
            try:
                fast_download.download_block_fast(data_root, split, scene_ids, layers, revision, jobs)
                return 0
            except Exception as error:     # never lose a block to the shortcut: clean up, then do it the toolkit's way
                removed = fast_download.remove_swapped_archives(data_root)
                print(f"  FAST UNPACK FAILED ({type(error).__name__}: {error}); removed {removed} half-done archive(s), "
                      "falling back to the toolkit's own unpacking")
        else:
            print("  no xz tool of version 5.4 or newer here: unpacking the toolkit's single-core way")
    ids_file = data_root / f"_scene_ids_{split}_block{block:06d}.txt"
    ids_file.write_text("".join(f"{scene_id}\n" for scene_id in scene_ids), encoding="utf-8")
    command = build_download_command(data_root, split, ids_file, layers, revision, jobs, dry_run)
    print("$", " ".join(command))
    return subprocess.run(command, check=True).returncode


def usable_scene_ids(dataset, scene_ids) -> tuple[list, list]:
    """(scenes the SDK will open, scenes it excludes). Second line of defence behind fetch_excluded_scene_ids."""
    excluded = set(getattr(dataset, "consumer_excluded_scene_ids", ()))
    kept = [scene_id for scene_id in scene_ids if scene_id not in excluded]
    return kept, [scene_id for scene_id in scene_ids if scene_id in excluded]


def block_on_disk(data_root, scene_ids, layers) -> bool:
    """True when data_root already holds these scenes with these layers.

    Archives are deleted after extraction, so re-running the downloader would
    fetch them again. This check lets a notebook skip that.
    """
    data_root = Path(data_root)
    if not (data_root / "dataset.json").is_file() or not (data_root / "available_data.json").is_file():
        return False
    from fzi_aura import FZIAURADataset

    dataset = FZIAURADataset(data_root)
    if not set(with_base_layer(layers)).issubset(dataset.available_layers or ()):
        return False
    try:  # get_scene applies the SDK's own ID normalisation
        scenes = [dataset.get_scene(scene_id) for scene_id in usable_scene_ids(dataset, scene_ids)[0]]
    except Exception:
        return False
    return all((scene.path / "scene.json").is_file() for scene in scenes)


# ---------------------------------------------------------------- inspecting

def describe_weather(value) -> str:
    """First OpenWeather condition as text. Same rule as the SDK's own example notebook."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        return str(value.get("description") or value.get("main") or "unknown")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    return str(value)


_STAMP = re.compile(r"(\d{9,11})_(\d{9})$")


def payload_timestamp_ns(path):
    """Timestamp encoded in a payload file name such as 1780482739_200002668.jpg.

    The pattern is taken from the examples in the SDK's docs/data-format.md.
    It is used only to READ a time, never to construct a path. Returns None
    when a name does not match.
    """
    match = _STAMP.search(Path(path).stem)
    return int(match.group(1)) * 1_000_000_000 + int(match.group(2)) if match else None


def keyframe_spacing_m(odom_from_base_poses) -> np.ndarray:
    """Distance in metres between consecutive ego poses (4x4, base_link to odom)."""
    positions = np.asarray([np.asarray(pose)[:3, 3] for pose in odom_from_base_poses], dtype=np.float64)
    if len(positions) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def iter_frames(frame_dataset):
    """FrameDataset has __len__ and __getitem__ but no __iter__, so index it explicitly."""
    for index in range(len(frame_dataset)):
        yield frame_dataset[index]


def _mode(series: pd.Series) -> str:
    series = series.dropna()
    return "unknown" if series.empty else str(series.mode().iloc[0])


def lighting_from(timestamp_ns, sunrise, sunset, twilight_minutes: float = 30.0) -> str:
    """day / twilight / night from the dataset's own sunrise and sunset values.

    The values come from OpenWeather and are expected to be Unix seconds. Anything
    that does not look like Unix seconds gives "unknown" rather than a guess.
    """
    try:
        sunrise, sunset = float(sunrise), float(sunset)
    except (TypeError, ValueError):
        return "unknown"
    if not (1e9 < sunrise < 3e9 and 1e9 < sunset < 3e9 and sunrise < sunset):
        return "unknown"
    now, margin = float(timestamp_ns) / 1e9, twilight_minutes * 60.0
    if min(abs(now - sunrise), abs(now - sunset)) <= margin:
        return "twilight"
    return "day" if sunrise < now < sunset else "night"


def weather_group(description) -> str:
    """Wet or dry, from an OpenWeather description. Fine-grained weather has too few scenes per value."""
    text = str(description).lower()
    if text in ("unknown", "nan", "none", ""):
        return "unknown"
    return "wet" if any(word in text for word in ("rain", "drizzle", "thunderstorm", "snow", "sleet")) else "dry"


def speed_band(speed_kph) -> str:
    if not np.isfinite(speed_kph):
        return "unknown"
    if speed_kph < 5:
        return "under 5 km/h"
    if speed_kph < 20:
        return "5-20 km/h"
    return "20-50 km/h" if speed_kph < 50 else "over 50 km/h"


def _numeric_median(frame, column):
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def scene_summary(scene) -> dict:
    """One row of facts about a scene. Needs base_keyframes and camera_keyframes on disk."""
    calibration = scene.calibration()
    keyframes = list(iter_frames(scene.frames(sample_filter="any_label")))
    box_counts = [frame.boxes_3d_count or 0 for frame in keyframes]
    spacing = keyframe_spacing_m([frame.load_ego_pose() for frame in keyframes])
    gaps_s = np.diff([frame.timestamp_ns for frame in keyframes]) / 1e9 if len(keyframes) > 1 else np.zeros(0)

    # Load every column, then pick. Asking the SDK for a named column that a scene
    # lacks raises, and only "visibility" is treated as optional by the SDK.
    all_signals = scene.load_vehicle_signals()
    missing_context = [column for column in CONTEXT_COLUMNS if column not in all_signals.columns]
    signals = all_signals.reindex(columns=CONTEXT_COLUMNS)
    return {
        "scene_id": scene.scene_id,
        "samples_10hz": len(scene),
        "keyframes": len(keyframes),
        "keyframe_gap_s_median": float(np.median(gaps_s)) if len(gaps_s) else float("nan"),
        "keyframe_spacing_m_median": float(np.median(spacing)) if len(spacing) else float("nan"),
        "keyframe_spacing_m_max": float(spacing.max()) if len(spacing) else float("nan"),
        "context_columns_missing": missing_context,
        "signal_columns": sorted(str(column) for column in all_signals.columns),
        "speed_kph_median": float(signals["speed_kph"].median()),
        "road_type": _mode(signals["road_type"]),
        "road_type_values": sorted(str(v) for v in signals["road_type"].dropna().unique()),
        "weather": _mode(signals["weather"].map(describe_weather)),
        "weather_group": weather_group(_mode(signals["weather"].map(describe_weather))),
        "speed_band": speed_band(float(signals["speed_kph"].median())),
        "sunrise_raw": _numeric_median(all_signals, "sunrise"),
        "sunset_raw": _numeric_median(all_signals, "sunset"),
        "lighting": lighting_from((scene.start_timestamp_ns + scene.end_timestamp_ns) / 2,
                                  _numeric_median(all_signals, "sunrise"), _numeric_median(all_signals, "sunset")),
        "start_hour_utc": int(pd.Timestamp(scene.start_timestamp_ns, unit="ns", tz="UTC").hour),
        "boxes_total": int(sum(box_counts)),
        "keyframes_with_zero_boxes": int(sum(count == 0 for count in box_counts)),
        "cameras_on_disk": list(scene.available_cameras()),
        "cameras_calibrated": list(calibration.sensor_ids("camera")),
        "lidars_calibrated": list(calibration.sensor_ids("lidar")),
        "radars_calibrated": list(calibration.sensor_ids("radar")),
    }


def calibration_report(scene) -> dict:
    """Raw calibration entries plus derived checks, for every camera of a scene."""
    calibration = scene.calibration()
    frame = next(iter_frames(scene.frames(sample_filter="any_label")))
    report = {"top_level_keys": sorted(calibration.data.keys()), "sensor_keys": sorted(calibration.sensors.keys()),
              "cameras": {}}
    for camera_id in calibration.sensor_ids("camera"):
        entry = calibration.sensor(f"camera/{camera_id}")
        projection = calibration.camera_projection_matrix(camera_id)
        base_from_camera = calibration.base_from_sensor(f"camera/{camera_id}")
        item = {
            "entry_keys": sorted(entry.keys()),
            "intrinsics_keys": sorted(entry.get("intrinsics", {}).keys()),
            "fx_fy_cx_cy": [float(projection[0, 0]), float(projection[1, 1]),
                            float(projection[0, 2]), float(projection[1, 2])],
            "P_fourth_column": projection[:, 3].round(4).tolist(),
            "position_in_base_link_m": base_from_camera[:3, 3].round(3).tolist(),
            # Third column of the rotation = camera +Z (optical axis) expressed in base_link.
            "optical_axis_in_base_link": base_from_camera[:3, 2].round(3).tolist(),
        }
        if frame.has_camera(camera_id):
            height, width = frame.load_camera(camera_id).shape[:2]
            item["image_width_height"] = [int(width), int(height)]
            item["cx_cy_over_width_height"] = [round(float(projection[0, 2]) / width, 3),
                                               round(float(projection[1, 2]) / height, 3)]
            stamp = payload_timestamp_ns(frame.camera_path(camera_id))
            item["image_minus_sample_ms"] = None if stamp is None else round((stamp - frame.timestamp_ns) / 1e6, 3)
        report["cameras"][camera_id] = item
    return report


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path
