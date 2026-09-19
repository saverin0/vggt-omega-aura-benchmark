"""Ground-truth depth: LiDAR points projected into a camera image.

DECISIONS, fixed before any metric exists (docs/decisions.md has the reasons):

1. Sweep stage: motion_compensated. Its timestamp IS the sample timestamp, the
   cameras fire within 1.4 ms of it, and the semantic labels follow its point
   order. Raw sweeps are stamped at sweep START and would need our own
   ego-motion correction.

2. Resolution: points are projected DIRECTLY at the model's input size, with
   the calibration scaled per axis (geometry.scale_intrinsics). No depth map is
   ever resized, so no interpolation rule is needed. A pixel's centre sits at
   its integer coordinate, the same convention as the model's own unprojection.
   Aggregation rule: when several visible points fall in one pixel, the
   NEAREST is that pixel's ground truth.

3. Classes: `noise` and `ego` points are dropped. `unlabeled` points are kept
   (their geometry is real) and counted.

4. Occlusion. The LiDARs sit elsewhere on the car, so they see background that
   the camera cannot. Projected naively, that background lands INSIDE the
   outline of foreground objects. Three modes are implemented so they can be
   compared; "two_sided" is the one used:

   - "none":      plain projection.
   - "window":    drop a point if anything much nearer lies within a small
                  window. Simple, and WRONG on slanted surfaces: on a road near
                  the horizon depth changes by 25%+ across two pixel rows, so
                  it deletes most road points beyond about 14 m.
   - "two_sided": drop a point only if much-nearer points lie on BOTH sides of
                  it, above and below in its own pixel COLUMN, or left and right
                  in its own pixel ROW, or in its own pixel. A slanted surface
                  only ever has nearer points on ONE side along a line, so it
                  survives. Background leaking into an object's outline has
                  object points around it, so it is removed. Known limit: a
                  leak within one LiDAR ring of an outline's edge can survive.

                  The search lines are exactly one pixel wide (half_width = 0)
                  ON PURPOSE. A wider strip breaks the rule on roads: a
                  left/right strip three rows tall contains the row BELOW,
                  where the road is nearer on both sides, so far road gets
                  deleted again. Found in rehearsal on a dense synthetic road;
                  tests/test_ground_truth.py keeps a test that reproduces it.

   "Much nearer" means nearer by more than max(abs_tol, rel_tol * depth).

Everything below is NumPy on plain arrays, so the C++ port (cpp/) can be
tested against it number for number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

EXCLUDED_CLASS_NAMES = ("noise", "ego")
OCCLUSION_MODES = ("none", "window", "two_sided")
STAGE = "motion_compensated"


@dataclass(frozen=True)
class OcclusionParams:
    mode: str = "two_sided"
    radius: int = 6          # pixels searched on each side, at the model's input size
    half_width: int = 0      # keep 0: wider search lines delete roads (see the module docstring)
    rel_tol: float = 0.10
    abs_tol: float = 0.5     # metres
    min_depth: float = 1.0   # metres; nearer returns are the ego vehicle or sensor artefacts

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ projection

def project_pinhole(points_cam, fx, fy, cx, cy):
    """Camera-frame points (X right, Y down, Z forward) to pixel coordinates and Z-depth."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    z = points_cam[:, 2]
    safe = np.where(z > 1e-9, z, np.nan)
    u = fx * points_cam[:, 0] / safe + cx
    v = fy * points_cam[:, 1] / safe + cy
    return u, v, z


def unproject_pinhole(u, v, depth, fx, fy, cx, cy) -> np.ndarray:
    """Pixels plus Z-depth back to camera-frame points, shape (N, 3). The inverse of project_pinhole.

    Same convention as the model's own unprojection: a pixel's centre is its
    integer coordinate, and `depth` is Z, not the distance along the ray.
    """
    u, v, depth = (np.asarray(a, dtype=np.float64) for a in (u, v, depth))
    return np.column_stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])


def pixel_indices(u, v, width: int, height: int):
    """Nearest pixel for each projected point, and which points fall inside the image."""
    with np.errstate(invalid="ignore"):
        ui = np.floor(u + 0.5)
        vi = np.floor(v + 0.5)
        inside = np.isfinite(ui) & np.isfinite(vi) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = np.where(inside, ui, 0).astype(np.int64)
    vi = np.where(inside, vi, 0).astype(np.int64)
    return ui, vi, inside


# ------------------------------------------------------------------ occlusion

def min_depth_buffer(ui, vi, z, width: int, height: int) -> np.ndarray:
    """Per-pixel nearest depth. Empty pixels hold +inf."""
    buffer = np.full((height, width), np.inf)
    np.minimum.at(buffer, (vi, ui), z)
    return buffer


def _shifted(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """result[v, u] = array[v - dy, u - dx], with +inf where that falls outside."""
    height, width = array.shape
    result = np.full_like(array, np.inf)
    src_v = slice(max(0, -dy), min(height, height - dy))
    dst_v = slice(max(0, dy), min(height, height + dy))
    src_u = slice(max(0, -dx), min(width, width - dx))
    dst_u = slice(max(0, dx), min(width, width + dx))
    result[dst_v, dst_u] = array[src_v, src_u]
    return result


def directional_minima(buffer: np.ndarray, radius: int, half_width: int) -> dict:
    """Nearest depth found in a strip on each side of every pixel.

    "above" looks at image rows v - radius .. v - 1 (smaller v is higher in the
    image), in columns u - half_width .. u + half_width. The other three are
    the same strip rotated. The pixel itself is never included.
    """
    across_u = buffer.copy()
    across_v = buffer.copy()
    for k in range(1, half_width + 1):
        across_u = np.minimum(across_u, np.minimum(_shifted(buffer, 0, k), _shifted(buffer, 0, -k)))
        across_v = np.minimum(across_v, np.minimum(_shifted(buffer, k, 0), _shifted(buffer, -k, 0)))
    sides = {name: np.full_like(buffer, np.inf) for name in ("above", "below", "left", "right")}
    for k in range(1, radius + 1):
        sides["above"] = np.minimum(sides["above"], _shifted(across_u, k, 0))    # takes row v - k
        sides["below"] = np.minimum(sides["below"], _shifted(across_u, -k, 0))   # takes row v + k
        sides["left"] = np.minimum(sides["left"], _shifted(across_v, 0, k))      # takes column u - k
        sides["right"] = np.minimum(sides["right"], _shifted(across_v, 0, -k))   # takes column u + k
    return sides


VISIBLE, HIDDEN_SAME_PIXEL, HIDDEN_BY_NEIGHBOURS = 0, 1, 2


def occlusion_reason(ui, vi, z, width: int, height: int, params: OcclusionParams) -> np.ndarray:
    """Per point: VISIBLE, HIDDEN_SAME_PIXEL or HIDDEN_BY_NEIGHBOURS.

    HIDDEN_SAME_PIXEL: a much nearer point shares the pixel. It covers both a
    dense occluder and a grazing surface (one pixel spans a long stretch of
    road). It never changes the result, because only the nearest point per
    pixel becomes ground truth anyway. It is kept as a separate label so that
    HIDDEN_BY_NEIGHBOURS, the occlusion rule proper and the part that can be
    wrong, can be inspected without that noise.
    """
    if params.mode not in OCCLUSION_MODES:
        raise ValueError(f"mode must be one of {OCCLUSION_MODES}, got {params.mode!r}")
    z = np.asarray(z, dtype=np.float64)
    reason = np.full(len(z), VISIBLE, dtype=np.uint8)
    if params.mode == "none" or len(z) == 0:
        return reason

    buffer = min_depth_buffer(ui, vi, z, width, height)
    threshold = z - np.maximum(params.abs_tol, params.rel_tol * z)   # an occluder must be nearer than this
    sides = directional_minima(buffer, params.radius, params.half_width)
    above, below = sides["above"][vi, ui] < threshold, sides["below"][vi, ui] < threshold
    left, right = sides["left"][vi, ui] < threshold, sides["right"][vi, ui] < threshold
    by_neighbours = (above | below | left | right) if params.mode == "window" else ((above & below) | (left & right))
    reason[by_neighbours] = HIDDEN_BY_NEIGHBOURS
    reason[buffer[vi, ui] < threshold] = HIDDEN_SAME_PIXEL           # something much nearer in the same pixel
    return reason


def visible_mask(ui, vi, z, width: int, height: int, params: OcclusionParams) -> np.ndarray:
    """True for points the camera can see, under the chosen occlusion mode."""
    return occlusion_reason(ui, vi, z, width, height, params) == VISIBLE


def nearest_per_pixel(ui, vi, z, width: int) -> np.ndarray:
    """Indices of the nearest point in each occupied pixel (the aggregation rule)."""
    if len(z) == 0:
        return np.zeros(0, dtype=np.int64)
    flat = vi * width + ui
    order = np.lexsort((z, flat))                 # by pixel, then by depth
    first = np.ones(len(order), dtype=bool)
    first[1:] = flat[order][1:] != flat[order][:-1]
    return np.sort(order[first])


# ------------------------------------------------------------------ AURA glue

def excluded_class_ids(class_names: dict) -> set:
    """IDs of classes to drop, looked up by NAME in the scene's classes.json."""
    wanted = set(EXCLUDED_CLASS_NAMES)
    return {int(class_id) for class_id, name in class_names.items() if name in wanted}


def load_points_in_camera(frame, camera_id: str, lidar_ids) -> dict:
    """Every LiDAR's points for one frame, moved into the camera frame, with labels.

    All motion-compensated sweeps of a sample share that sample's timestamp, and
    the sensor mounts are rigid, so one static transform per LiDAR is enough:
        camera_from_lidar = camera_from_base @ base_from_lidar
    """
    calibration = frame.calibration()
    camera_from_base = calibration.sensor_from_base(f"camera/{camera_id}")
    points, semantic, instance, sensor = [], [], [], []
    for index, lidar_id in enumerate(lidar_ids):
        if not frame.has_lidar(lidar_id, STAGE):
            continue
        if frame.has_semantics(lidar_id):
            cloud, labels = frame.load_lidar_semantic_pair(lidar_id, stage=STAGE)   # the SDK checks the lengths match
            semantic_id, instance_id = labels.semantic_id, labels.instance_id
        else:                                                                       # unlabelled sweep: id 0 = "unlabeled"
            cloud = frame.load_lidar(lidar_id, stage=STAGE)
            semantic_id = instance_id = np.zeros(len(cloud), dtype=np.uint16)
        camera_from_lidar = camera_from_base @ calibration.base_from_sensor(f"lidar/{lidar_id}")
        xyz = np.asarray(cloud.xyz, dtype=np.float64)
        points.append(xyz @ camera_from_lidar[:3, :3].T + camera_from_lidar[:3, 3])
        semantic.append(np.asarray(semantic_id, dtype=np.uint16))
        instance.append(np.asarray(instance_id, dtype=np.uint16))
        sensor.append(np.full(len(xyz), index, dtype=np.uint8))
    if not points:
        return {"points_cam": np.zeros((0, 3)), "semantic_id": np.zeros(0, np.uint16),
                "instance_id": np.zeros(0, np.uint16), "sensor_index": np.zeros(0, np.uint8)}
    return {"points_cam": np.concatenate(points), "semantic_id": np.concatenate(semantic),
            "instance_id": np.concatenate(instance), "sensor_index": np.concatenate(sensor)}


BACKENDS = ("python", "cpp", "auto")
_IDENTITY = np.eye(4)


def _cpp_module():
    from .cpp_build import import_module
    return import_module()


def resolve_backend(backend: str = "auto") -> str:
    """"python" or "cpp". "auto" means C++ when the compiled module is available, otherwise Python."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "python":
        return "python"
    available = _cpp_module() is not None
    if backend == "cpp" and not available:
        raise RuntimeError("backend='cpp' was asked for but the C++ module is not built: python -m vggt_aura.cpp_build")
    return "cpp" if available else "python"


def build_ground_truth(loaded: dict, intrinsics, width: int, height: int,
                       params: OcclusionParams, drop_class_ids=(), backend: str = "python") -> dict:
    """Sparse ground-truth depth for one frame at the model's input size.

    `intrinsics` is (fx, fy, cx, cy) ALREADY scaled to width x height.
    Returns one entry per occupied pixel, plus counts that explain what was dropped.

    `backend`: "python" is the reference. "cpp" runs projection, the occlusion check and the per-pixel
    choice in the C++ core instead; it returns exactly the same pixels and verdicts (differential tests,
    and all 40 real frames of a scene in notebook 06). "auto" uses C++ when the module is built.
    """
    fx, fy, cx, cy = intrinsics
    wanted_class = ~np.isin(loaded["semantic_id"], list(drop_class_ids))
    used = resolve_backend(backend)
    if used == "cpp":
        points = np.ascontiguousarray(loaded["points_cam"], dtype=np.float64)
        out = _cpp_module().lidar_to_image(points, _IDENTITY, fx, fy, cx, cy, width, height, params.mode, params.radius,
                                           params.half_width, params.rel_tol, params.abs_tol, params.min_depth,
                                           keep=wanted_class.astype(np.uint8))
        z = points[:, 2]
        index, reason, chosen = out["candidate_index"], out["reason"], out["chosen"]
        visible = reason == VISIBLE
        chosen_u, chosen_v = out["u"], out["v"]
        in_front = int(out["in_image_in_front"])
    else:
        u, v, z = project_pinhole(loaded["points_cam"], fx, fy, cx, cy)
        ui, vi, inside = pixel_indices(u, v, width, height)
        candidate = inside & (z >= params.min_depth) & wanted_class
        index = np.flatnonzero(candidate)
        reason = occlusion_reason(ui[index], vi[index], z[index], width, height, params)
        visible = reason == VISIBLE
        seen = index[visible]
        chosen = seen[nearest_per_pixel(ui[seen], vi[seen], z[seen], width)]
        chosen_u, chosen_v = ui[chosen], vi[chosen]
        in_front = int((inside & (z >= params.min_depth)).sum())
    return {
        "u": np.asarray(chosen_u).astype(np.int16), "v": np.asarray(chosen_v).astype(np.int16),
        "depth_m": z[chosen].astype(np.float32), "backend": used,
        "xyz_cam": loaded["points_cam"][chosen].astype(np.float32),   # the 3D point itself, for the box membership test
        "semantic_id": loaded["semantic_id"][chosen], "instance_id": loaded["instance_id"][chosen],
        "sensor_index": loaded["sensor_index"][chosen],
        # bookkeeping for the trust check; index arrays refer to `loaded`
        "candidate_index": index, "visible": visible, "reason": reason,
        "counts": {
            "points_loaded": int(len(z)),
            "in_image_in_front": in_front,
            "dropped_by_class": in_front - int(len(index)),
            "candidates": int(len(index)),
            "dropped_as_occluded": int((reason == HIDDEN_BY_NEIGHBOURS).sum()),
            "dropped_same_pixel": int((reason == HIDDEN_SAME_PIXEL).sum()),
            "pixels_with_ground_truth": int(len(chosen)),
        },
    }


def removal_table(loaded: dict, truth: dict, class_names: dict, depth_edges=(0, 10, 20, 40, 80, 1e9)):
    """Share of candidate points dropped by the occlusion rule, by class and by depth band.

    `occluded_share` counts only HIDDEN_BY_NEIGHBOURS, the rule proper. This is
    the check on the rule itself: a rule that drops many far `road` points is
    deleting a slanted surface, not an occlusion. Same-pixel drops are reported
    separately because they are harmless.
    """
    import pandas as pd

    index = truth["candidate_index"]
    z = loaded["points_cam"][index, 2]
    names = np.array([class_names.get(int(c), str(int(c))) for c in loaded["semantic_id"][index]])
    bands = pd.cut(z, list(depth_edges), right=False,
                   labels=[f"{int(a)}-{'inf' if b > 1e8 else int(b)} m" for a, b in zip(depth_edges[:-1], depth_edges[1:])])
    table = pd.DataFrame({"class": names, "band": bands,
                          "occluded": truth["reason"] == HIDDEN_BY_NEIGHBOURS,
                          "same_pixel": truth["reason"] == HIDDEN_SAME_PIXEL})

    def summarise(group):
        return group.agg(points=("occluded", "size"), occluded=("occluded", "sum"), same_pixel=("same_pixel", "sum"))

    by_class = summarise(table.groupby("class")).sort_values("points", ascending=False)
    by_band = summarise(table.groupby("band", observed=False))
    return by_class, by_band


# ------------------------------------------------------------------ persistence

PER_PIXEL_KEYS = ("u", "v", "depth_m", "xyz_cam", "semantic_id", "instance_id", "sensor_index")


def truth_paths(persist_root, scene_name: str, camera_id: str, tag=None) -> tuple[Path, Path]:
    """Where a scene's ground truth lives. `tag` separates ground truth built at another input size,
    for example "518x322" for VGGT. Without a tag this is the VGGT-Omega 640x400 file, as before."""
    folder = Path(persist_root) / "ground_truth" / scene_name
    stem = camera_id if tag is None else f"{camera_id}_{tag}"
    return folder / f"{stem}.npz", folder / f"{stem}.json"


def save_scene_truth(npz_path, json_path, frames: list, timestamps_ns, meta: dict) -> float:
    """Store a scene's sparse ground truth: all frames concatenated, with offsets. Returns size in MB.

    Frame i owns rows frame_offsets[i] : frame_offsets[i + 1] of every per-pixel array.
    """
    npz_path, json_path = Path(npz_path), Path(json_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(f["depth_m"]) for f in frames])]).astype(np.int64)
    stored = {key: np.concatenate([np.asarray(f[key]) for f in frames]) if frames else np.zeros(0) for key in PER_PIXEL_KEYS}
    stored["frame_offsets"] = offsets
    stored["timestamps_ns"] = np.asarray(timestamps_ns, dtype=np.int64)
    temporary = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **stored)
    temporary.replace(npz_path)
    meta = {**meta, "per_frame_counts": [f["counts"] for f in frames]}
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return npz_path.stat().st_size / 1e6


def load_scene_truth(npz_path, json_path) -> tuple[list, np.ndarray, dict]:
    """Inverse of save_scene_truth: (list of per-frame dicts, timestamps_ns, meta)."""
    with np.load(npz_path) as data:
        offsets = data["frame_offsets"]
        frames = [{key: data[key][offsets[i]:offsets[i + 1]] for key in PER_PIXEL_KEYS}
                  for i in range(len(offsets) - 1)]
        timestamps_ns = data["timestamps_ns"]
    return frames, timestamps_ns, json.loads(Path(json_path).read_text(encoding="utf-8"))


def build_scene_truth(frames, camera_id: str, lidar_ids, intrinsics, width: int, height: int,
                      params: OcclusionParams, drop_class_ids=(), backend: str = "auto") -> list:
    """Ground truth for every frame of a scene. One call, so every notebook builds it the same way.

    Uses the C++ core when it is built ("auto"). Reading the LiDAR files stays in Python: the C++ core
    takes numbers in and gives numbers out, and never opens a file.
    """
    return [build_ground_truth(load_points_in_camera(frame, camera_id, lidar_ids), intrinsics, width, height,
                               params, drop_class_ids, backend) for frame in frames]


# ------------------------------------------------------------------ build once, reuse for ever

LIDAR_POLICIES = ("ouster_only", "all")


def select_lidars(available, policy: str = "ouster_only") -> list:
    """Which LiDARs feed the ground truth.

    "ouster_only" (default) drops the Aeva FMCW sensors that only the 12-LiDAR
    recordings carry, so that ground truth has the same sensor set, and so a
    comparable density, in every scene. The SDK itself recognises those sensors
    by "aeva" in the sensor id (fzi_aura/frame.py, load_lidar).
    """
    if policy not in LIDAR_POLICIES:
        raise ValueError(f"policy must be one of {LIDAR_POLICIES}, got {policy!r}")
    chosen = sorted(available)
    return [lidar for lidar in chosen if "aeva" not in lidar.lower()] if policy == "ouster_only" else chosen


def truth_tag(width: int, height: int):
    """File tag for a ground truth built at width x height. 640x400 keeps the untagged name the first saved files had."""
    return None if (width, height) == (640, 400) else f"{width}x{height}"


def saved_truth_matches(meta: dict, timestamps_saved, timestamps_ns, lidar_ids, width, height, params: OcclusionParams) -> bool:
    """A saved ground truth is reused only if it was built for exactly this request."""
    return (meta.get("occlusion") == params.as_dict() and list(meta.get("model_wh", [])) == [width, height]
            and list(meta.get("lidars", [])) == list(lidar_ids)
            and [int(v) for v in timestamps_saved] == [int(v) for v in timestamps_ns])


class GroundTruthUnavailable(RuntimeError):
    """No saved ground truth fits the request, and the LiDAR layer is not on disk to build one."""


def load_or_build_scene_truth(persist_root, scene_id: str, scene_name: str, frames, camera_id: str, lidar_ids,
                              intrinsics, width: int, height: int, params: OcclusionParams, drop_class_ids=(),
                              dataset_revision: str = "", can_build: bool = True) -> tuple[list, str]:
    """Ground truth for a scene: from persistent storage when a matching file exists, otherwise built and saved.

    `lidar_ids` is the INTENDED sensor set, taken from the calibration file (always on disk), not from
    what happens to be downloaded. `can_build` says whether the LiDAR layer is on disk right now.
    Returns (frames of ground truth, "loaded" or "built"). This is what makes the slow LiDAR layer a
    once-per-block cost: after the first build, no notebook needs LiDAR on disk for this scene again.
    """
    npz_path, json_path = truth_paths(persist_root, scene_name, camera_id, tag=truth_tag(width, height))
    timestamps_ns = [int(frame.timestamp_ns) for frame in frames]
    if npz_path.is_file() and json_path.is_file():
        try:
            truth, saved_stamps, meta = load_scene_truth(npz_path, json_path)
        except KeyError:                       # a file from before a field was added: rebuild it
            truth = None
        if truth is not None and saved_truth_matches(meta, saved_stamps, timestamps_ns, lidar_ids, width, height, params):
            return truth, "loaded"
    if not can_build or not lidar_ids:
        raise GroundTruthUnavailable(f"No saved ground truth matches for {scene_id}, and no LiDAR is on disk to build it. "
                           "Download the lidar_motion_compensated_keyframes layer for this block.")
    truth = build_scene_truth(frames, camera_id, lidar_ids, intrinsics, width, height, params, drop_class_ids)
    backend = truth[0].get("backend", "python") if truth else "python"
    save_scene_truth(npz_path, json_path, truth, timestamps_ns,
                     {"scene_id": scene_id, "camera": camera_id, "lidars": list(lidar_ids), "model_wh": [width, height],
                      "occlusion": params.as_dict(), "dataset_revision": dataset_revision, "built_with": backend})
    return truth, f"built ({backend})"
