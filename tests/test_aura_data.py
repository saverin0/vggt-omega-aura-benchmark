import sys

import numpy as np
import pandas as pd
import pytest

from vggt_aura import aura_data as ad
from vggt_aura.session import UPSTREAM, missing_upstream


def release_tables():
    """Two val blocks. Block 1 has a LiDAR part that is not on the Hub yet."""
    rows = []
    for block, lidar_on_hub in ((0, True), (1, False)):
        rows.append(("val", block, "base_keyframes", f"b{block}.tar", 2e9, "ready"))
        rows.append(("val", block, "camera_keyframes", f"c{block}.tar", 3e9, "ready"))
        rows.append(("val", block, "lidar_motion_compensated_keyframes", f"l{block}a.tar.xz", 4e9, "ready"))
        rows.append(("val", block, "lidar_motion_compensated_keyframes", f"l{block}b.tar.xz", 1e9, "ready"))
    chunks = pd.DataFrame(rows, columns=["split", "scene_block", "layer", "chunk_path", "size_bytes", "status"])
    hub_files = set(chunks["chunk_path"]) - {"l1b.tar.xz"}
    scene_blocks = pd.DataFrame({
        "split": ["val"] * 5,
        "scene_block": [0, 0, 0, 1, 1],
        "scene_ordinal": [2, 1, 3, 4, 5],
        "scene_id": ["s|2", "s|1", "s|3", "s|4", "s|5"],
    })
    return chunks, scene_blocks, hub_files


def test_camera_only_selection_sees_both_blocks():
    chunks, scene_blocks, hub = release_tables()
    table = ad.available_blocks(chunks, scene_blocks, hub, ["camera_keyframes"])
    assert list(table.index) == [("val", 0), ("val", 1)]
    assert table.loc[("val", 0), "n_scenes"] == 3
    assert table.loc[("val", 0), "total_gb"] == 5.0   # base is always counted


def test_block_with_a_missing_part_is_not_offered():
    chunks, scene_blocks, hub = release_tables()
    table = ad.available_blocks(chunks, scene_blocks, hub, ["camera_keyframes", "lidar_motion_compensated_keyframes"])
    assert list(table.index) == [("val", 0)]
    assert table.loc[("val", 0), "lidar_motion_compensated_keyframes"] == 5.0  # parts are summed


def test_not_ready_chunk_excludes_the_block():
    chunks, scene_blocks, hub = release_tables()
    chunks.loc[chunks["chunk_path"] == "c0.tar", "status"] = "pending"
    table = ad.available_blocks(chunks, scene_blocks, hub, ["camera_keyframes"])
    assert list(table.index) == [("val", 1)]


def test_block_scene_ids_follow_scene_ordinal():
    _, scene_blocks, _ = release_tables()
    assert ad.block_scene_ids(scene_blocks, "val", 0) == ["s|1", "s|2", "s|3"]
    with pytest.raises(ValueError):
        ad.block_scene_ids(scene_blocks, "test", 0)


def test_download_command_uses_only_flags_the_sdk_has(tmp_path):
    command = ad.build_download_command(tmp_path, "val", tmp_path / "ids.txt",
                                        ["base_keyframes", "camera_keyframes"], revision="abc", dry_run=True)
    assert command[:4] == [sys.executable, "-m", "fzi_aura.download", str(tmp_path)]
    flags = [part for part in command if part.startswith("--")]
    assert flags == ["--revision", "--splits", "--scene-ids-file", "--layers", "--jobs", "--verify", "--dry-run"]
    assert command[command.index("--layers") + 1] == "camera_keyframes"   # base is added by the SDK itself
    assert command[command.index("--revision") + 1] == "abc"


def test_download_command_with_only_base_layer(tmp_path):
    command = ad.build_download_command(tmp_path, "val", tmp_path / "ids.txt", ["base_keyframes"])
    assert command[command.index("--layers") + 1] == "base_keyframes"
    assert "--dry-run" not in command


def test_describe_weather_handles_every_shape():
    assert ad.describe_weather([{"main": "Clouds", "description": "scattered clouds"}]) == "scattered clouds"
    assert ad.describe_weather(np.array([{"main": "Rain"}], dtype=object)) == "Rain"
    assert ad.describe_weather(None) == "unknown"
    assert ad.describe_weather(float("nan")) == "unknown"
    assert ad.describe_weather([]) == "[]"


def test_payload_timestamp_matches_the_documented_example():
    assert ad.payload_timestamp_ns("camera/front_medium/1780482739_200002668.jpg") == 1780482739200002668
    assert ad.payload_timestamp_ns("camera/front_medium/frame_0001.jpg") is None


def test_image_offset_from_the_documented_example_is_microseconds():
    offset_ns = ad.payload_timestamp_ns("1780482739_200002668.jpg") - 1780482739200000000
    assert offset_ns == 2668


def test_keyframe_spacing():
    def pose(x, y):
        matrix = np.eye(4)
        matrix[:3, 3] = [x, y, 0.0]
        return matrix
    spacing = ad.keyframe_spacing_m([pose(0, 0), pose(3, 4), pose(3, 4)])
    assert spacing.tolist() == [5.0, 0.0]
    assert ad.keyframe_spacing_m([pose(0, 0)]).size == 0


def test_to_jsonable_handles_numpy_and_paths(tmp_path):
    payload = {"a": np.float32(1.5), "b": np.arange(3), "c": (1, 2), "d": tmp_path}
    path = ad.save_json(tmp_path / "out" / "report.json", payload)
    assert path.is_file()
    assert ad.to_jsonable(payload)["b"] == [0, 1, 2]


def test_missing_upstream_reports_only_absent_packages():
    present = {"einops", "safetensors"}
    missing = missing_upstream(find_spec=lambda name: object() if name in present else None)
    assert missing == [name for name in UPSTREAM if name not in present]


def test_model_package_is_installed_without_dependencies():
    assert UPSTREAM["vggt_omega"][0] == "--no-deps"


def test_lighting_from_sunrise_and_sunset():
    sunrise, sunset = 1752550000, 1752608000                       # Unix seconds, a July day
    noon_ns = 1752585602 * 1_000_000_000                           # the real scene read in phase 2: 13:20 UTC
    assert ad.lighting_from(noon_ns, sunrise, sunset) == "day"
    assert ad.lighting_from((sunset + 3 * 3600) * 10**9, sunrise, sunset) == "night"
    assert ad.lighting_from((sunrise - 3 * 3600) * 10**9, sunrise, sunset) == "night"
    assert ad.lighting_from((sunset - 600) * 10**9, sunrise, sunset) == "twilight"
    assert ad.lighting_from((sunrise + 1500) * 10**9, sunrise, sunset) == "twilight"


def test_lighting_refuses_to_guess():
    noon_ns = 1752585602 * 1_000_000_000
    assert ad.lighting_from(noon_ns, None, None) == "unknown"
    assert ad.lighting_from(noon_ns, float("nan"), 1752608000) == "unknown"
    assert ad.lighting_from(noon_ns, 5.5, 21.0) == "unknown"                 # hours of the day, not Unix seconds
    assert ad.lighting_from(noon_ns, 1752608000, 1752550000) == "unknown"    # sunrise after sunset


def test_weather_and_speed_groups():
    assert ad.weather_group("light rain") == "wet" and ad.weather_group("Drizzle") == "wet"
    assert ad.weather_group("overcast clouds") == "dry" and ad.weather_group("clear sky") == "dry"
    assert ad.weather_group("unknown") == "unknown"
    bands = [ad.speed_band(v) for v in (1.0, 9.3, 30.5, 88.0, float("nan"))]
    assert bands == ["under 5 km/h", "5-20 km/h", "20-50 km/h", "over 50 km/h", "unknown"]


def test_excluded_scenes_are_left_out_of_a_block():
    table = pd.DataFrame({"split": ["train"] * 3, "scene_block": [77] * 3, "scene_ordinal": [1, 2, 3],
                          "scene_id": ["a|44", "a|45", "a|73"], "scene": ["a_44", "a_45", "a_73"]})
    excluded = {"a|44": "Severe repeated LiDAR dropout", "a|73": "PTP synchronization fault"}
    assert ad.block_scene_ids(table, "train", 77) == ["a|44", "a|45", "a|73"]          # unchanged when nothing is passed
    assert ad.block_scene_ids(table, "train", 77, excluded) == ["a|45"]
    assert ad.block_scene_names(table, "train", 77, excluded) == ["a_45"]               # ids and names stay in step


def test_usable_scene_ids_is_the_second_line_of_defence():
    class FakeDataset:
        consumer_excluded_scene_ids = frozenset({"a|73"})
    assert ad.usable_scene_ids(FakeDataset(), ["a|45", "a|73"]) == (["a|45"], ["a|73"])
    assert ad.usable_scene_ids(object(), ["a|45"]) == (["a|45"], [])                    # a dataset without that attribute
