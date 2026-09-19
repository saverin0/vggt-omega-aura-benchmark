"""Depth, pose and intrinsics metrics, with the alignment protocol fixed in one place.

THE PROTOCOL (docs/decisions.md has the reasons). The model's depth and camera
translation share one unknown scale per sequence, so every error in metres
needs an alignment. Five are reported, always under these names:

- "sequence_scale"    PRIMARY. One number per sequence: median(gt / pred) over
                      all valid pixels of all frames. The strictest honest
                      choice: it grants the model only the one degree of
                      freedom it truly lacks.
- "frame_scale"       One median scale per frame. Forgives scale drift.
- "frame_scale_shift" Median scale AND shift per frame. This is the authors'
                      own protocol (eval/metrics.py), kept for comparability.
                      Usually the most forgiving, but NOT guaranteed to be: it
                      is built from medians, not a best fit, and can score worse
                      than a single scale when many pixels are off (tested).
- "pose_scale"        The scale that best maps the PREDICTED CAMERA PATH onto
                      the true path, applied to depth. It asks whether depth
                      and pose agree on one scale, as they should.
- "unaligned"         Raw output against metres. Meaningless by construction
                      (the model never promises metres); reported only because
                      "aligned and unaligned, both" is the rule of this project.

Ground truth is used where DEPTH_MIN_M <= depth <= DEPTH_MAX_M. Scales are
fitted on all such pixels, moving objects included: they are about 2% of
pixels and cannot move a median.
"""

from __future__ import annotations

import numpy as np

from .geometry import angle_between_deg, invert_se3, rotation_angle_deg, to_4x4, umeyama_similarity

DEPTH_MIN_M, DEPTH_MAX_M = 1.0, 80.0
DEPTH_BAND_EDGES = (1.0, 10.0, 20.0, 40.0, 80.0)
PROTOCOLS = ("sequence_scale", "frame_scale", "frame_scale_shift", "pose_scale", "unaligned")
PRIMARY_PROTOCOL = "sequence_scale"
MIN_PIXELS_PER_FRAME = 10          # same floor as the authors' evaluation
MIN_BASELINE_M = 0.5               # below this, a pair's translation DIRECTION is noise
NEAR_STATIONARY_PATH_M = 10.0      # a scene that moves less than this has no usable trajectory


# ------------------------------------------------------------------ depth

def median_scale(pred, gt) -> float:
    return float(np.median(np.asarray(gt, dtype=np.float64) / np.asarray(pred, dtype=np.float64)))


def median_scale_shift(pred, gt) -> tuple[float, float]:
    """The authors' per-frame alignment, reproduced from vggt-omega eval/metrics.py."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    gt_centered = gt - np.median(gt) + 1e-8
    pred_centered = pred - np.median(pred) + 1e-8
    scale = float(np.median(gt_centered / pred_centered))
    shift = float(np.median(gt - scale * pred))
    return scale, shift


def align_depth(pred, gt, frame_index, protocol: str, pose_scale: float = float("nan")) -> np.ndarray:
    """Predicted depth in metres under one protocol. NaN marks pixels that cannot be aligned."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if protocol == "unaligned":
        return pred.copy()
    if protocol == "sequence_scale":
        return pred * median_scale(pred, gt) if len(pred) else pred.copy()
    if protocol == "pose_scale":
        return pred * pose_scale
    aligned = np.full(len(pred), np.nan)
    for frame in np.unique(frame_index):
        rows = frame_index == frame
        if rows.sum() < MIN_PIXELS_PER_FRAME:
            continue
        if protocol == "frame_scale":
            aligned[rows] = pred[rows] * median_scale(pred[rows], gt[rows])
        else:
            scale, shift = median_scale_shift(pred[rows], gt[rows])
            aligned[rows] = np.clip(pred[rows] * scale + shift, 1e-6, None)
    return aligned


def depth_summary(aligned, gt) -> dict:
    """AbsRel, delta < 1.25 and RMSE over the pixels given. NaN-aligned pixels are skipped."""
    aligned, gt = np.asarray(aligned, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    ok = np.isfinite(aligned) & (aligned > 0)
    if ok.sum() == 0:
        return {"n_pixels": 0, "abs_rel": float("nan"), "delta125": float("nan"), "rmse_m": float("nan")}
    a, g = aligned[ok], gt[ok]
    ratio = np.maximum(a / g, g / a)
    return {"n_pixels": int(ok.sum()), "abs_rel": float(np.mean(np.abs(a - g) / g)),
            "delta125": float(np.mean(ratio < 1.25)), "rmse_m": float(np.sqrt(np.mean((a - g) ** 2)))}


def depth_band_labels(gt) -> np.ndarray:
    edges = DEPTH_BAND_EDGES
    names = np.array([f"{int(a)}-{int(b)} m" for a, b in zip(edges[:-1], edges[1:])])
    index = np.clip(np.digitize(gt, edges[1:-1]), 0, len(names) - 1)
    return names[index]


# ------------------------------------------------------------------ pose

def predicted_camera0_from_camera(extrinsics) -> np.ndarray:
    """Model extrinsics are camera_from_world with world = camera 0. Invert to match the ground truth's form."""
    return invert_se3(to_4x4(extrinsics))


def pairwise_pose_errors(pred_c0_from_c, gt_c0_from_c) -> dict:
    """Relative-pose errors over every frame pair i < j.

    For a pair, the relative pose is camera_i_from_camera_j. Rotation error is
    the angle of R_gt.T @ R_pred. Translation error is the angle between the
    two relative translation DIRECTIONS; magnitude is never used, so this is
    scale-free. Two variants are returned:

    - "translation_deg":        the true angle, 0 to 180. Driving forward when
                                the truth is backward scores 180.
    - "translation_deg_unsigned": min(angle, 180 - angle), what the authors'
                                code computes. It cannot tell forward from
                                backward. Kept only for comparability.
    """
    pred, gt = to_4x4(pred_c0_from_c), to_4x4(gt_c0_from_c)
    n = len(pred)
    i, j = np.triu_indices(n, k=1)
    rel_pred = invert_se3(pred[i]) @ pred[j]
    rel_gt = invert_se3(gt[i]) @ gt[j]
    rotation = rotation_angle_deg(np.swapaxes(rel_gt[:, :3, :3], -1, -2) @ rel_pred[:, :3, :3])
    baseline = np.linalg.norm(rel_gt[:, :3, 3], axis=1)
    translation = angle_between_deg(rel_pred[:, :3, 3], rel_gt[:, :3, 3])
    translation = np.where(baseline >= MIN_BASELINE_M, translation, np.nan)
    return {"i": i, "j": j, "rotation_deg": rotation, "translation_deg": translation,
            "translation_deg_unsigned": np.minimum(translation, 180.0 - translation), "baseline_m": baseline}


def auc(rotation_deg, translation_deg, threshold: int) -> float:
    """Area under the accuracy curve up to `threshold` degrees, as in the authors' eval/metrics.py."""
    rotation_deg, translation_deg = np.asarray(rotation_deg), np.asarray(translation_deg)
    ok = np.isfinite(rotation_deg) & np.isfinite(translation_deg)
    if ok.sum() == 0:
        return float("nan")
    errors = np.maximum(rotation_deg[ok], translation_deg[ok])
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(errors))) * 100.0)


def trajectory_errors(pred_c0_from_c, gt_c0_from_c) -> dict:
    """Absolute trajectory error of the camera centres, two ways.

    Both paths start at the origin with the same axes (camera 0), so the ONLY
    freedom the model lacks is scale. "scale_only" grants exactly that.
    "sim3" is the customary similarity alignment; it may also rotate and shift
    the path, which can hide a real heading error.
    """
    pred = to_4x4(pred_c0_from_c)[:, :3, 3]
    gt = to_4x4(gt_c0_from_c)[:, :3, 3]
    path = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    out = {"gt_path_m": path, "near_stationary": bool(path < NEAR_STATIONARY_PATH_M), "frames": int(len(gt))}
    denominator = float((pred * pred).sum())
    scale = float((pred * gt).sum() / denominator) if denominator > 1e-18 else float("nan")
    out["metres_per_model_unit"] = scale
    out["ate_scale_only_m"] = float(np.sqrt(np.mean(np.sum((scale * pred - gt) ** 2, axis=1))))
    if len(gt) >= 3 and path > 1e-6:
        s, rotation, translation = umeyama_similarity(pred, gt)
        out["ate_sim3_m"] = float(np.sqrt(np.mean(np.sum((s * pred @ rotation.T + translation - gt) ** 2, axis=1))))
        out["sim3_scale"] = s
    else:
        out["ate_sim3_m"], out["sim3_scale"] = float("nan"), float("nan")
    out["ate_scale_only_pct_of_path"] = 100.0 * out["ate_scale_only_m"] / path if path > 1e-6 else float("nan")
    return out


def pose_summary(pred_c0_from_c, gt_c0_from_c) -> dict:
    pairs = pairwise_pose_errors(pred_c0_from_c, gt_c0_from_c)
    out = trajectory_errors(pred_c0_from_c, gt_c0_from_c)
    out.update({
        "pairs": int(len(pairs["i"])), "pairs_with_baseline": int(np.isfinite(pairs["translation_deg"]).sum()),
        "rotation_deg_median": float(np.median(pairs["rotation_deg"])),
        "translation_deg_median": float(np.nanmedian(pairs["translation_deg"])) if np.isfinite(pairs["translation_deg"]).any() else float("nan"),
        "auc3": auc(pairs["rotation_deg"], pairs["translation_deg"], 3),
        "auc30": auc(pairs["rotation_deg"], pairs["translation_deg"], 30),
        "auc30_authors_unsigned": auc(pairs["rotation_deg"], pairs["translation_deg_unsigned"], 30),
    })
    return out


# ------------------------------------------------------------------ intrinsics

def intrinsics_summary(pred_intrinsics, gt_fx: float, gt_fy: float) -> dict:
    """Relative focal-length error. Positive means the model's focal length is too long."""
    fx, fy = np.asarray(pred_intrinsics)[:, 0, 0], np.asarray(pred_intrinsics)[:, 1, 1]
    return {"fx_rel_err_median": float(np.median(fx / gt_fx - 1.0)), "fy_rel_err_median": float(np.median(fy / gt_fy - 1.0)),
            "fx_spread_rel": float((fx.max() - fx.min()) / np.median(fx)), "pred_fx_over_fy": float(np.median(fx / fy)),
            "gt_fx_over_fy": float(gt_fx / gt_fy)}


# ------------------------------------------------------------------ aggregation over scenes

def bootstrap_mean_ci(values, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 95% interval, resampling SCENES (frames of one scene are not independent)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)
