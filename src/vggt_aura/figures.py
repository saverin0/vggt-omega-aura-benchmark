"""Error pictures for single scenes: what the model saw, what it predicted, where it is wrong.

Conventions, the same as everywhere in this project:
- predicted depth is Z-depth in model units; it becomes metres through ONE scale per scene, the median of
  ground truth over prediction at the LiDAR pixels (the primary "sequence_scale" protocol);
- relative error is signed: (prediction - truth) / truth. Positive (red) = predicted TOO FAR,
  negative (blue) = predicted TOO NEAR;
- trajectories are drawn from above in the first camera's frame: X to the right, Z forward (OpenCV axes).

Plotting functions take plain arrays, so they are tested without the dataset. `scene_bundle` is the only
function that touches the dataset and persistent storage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ground_truth as gtm, inference as inf, metrics as mt, objects as ob
from .geometry import to_4x4

ERROR_CLIP = 0.5            # the error colour scale runs from -50% to +50%
DEPTH_RANGE_M = (1.0, 80.0)


def sequence_scale(depth, truth_frames) -> float:
    """Metres per model unit: median of truth / prediction over every LiDAR pixel of the scene in the evaluated range."""
    ratios = []
    for index, truth in enumerate(truth_frames):
        gt = truth["depth_m"].astype(np.float64)
        pred = depth[index, truth["v"].astype(np.int64), truth["u"].astype(np.int64)].astype(np.float64)
        ok = (gt >= mt.DEPTH_MIN_M) & (gt <= mt.DEPTH_MAX_M) & np.isfinite(pred) & (pred > 0)
        ratios.append(gt[ok] / pred[ok])
    ratios = np.concatenate(ratios) if ratios else np.zeros(0)
    return float(np.median(ratios)) if len(ratios) else float("nan")


def frame_errors(depth_m, truth: dict) -> dict:
    """LiDAR pixels of one frame inside the evaluated range, with the signed relative error of the prediction."""
    gt = truth["depth_m"].astype(np.float64)
    u, v = truth["u"].astype(np.int64), truth["v"].astype(np.int64)
    pred = depth_m[v, u].astype(np.float64)
    ok = (gt >= mt.DEPTH_MIN_M) & (gt <= mt.DEPTH_MAX_M) & np.isfinite(pred) & (pred > 0)
    return {"u": u[ok], "v": v[ok], "gt": gt[ok], "pred": pred[ok], "rel": (pred[ok] - gt[ok]) / gt[ok], "kept": ok}


def frame_table(depth, truth_frames, labels, scale: float) -> pd.DataFrame:
    """One line per frame: is the error spread over the scene, or do a few frames carry it?

    scale_ratio = this frame's own best scale over the scene's scale. Far from 1 means the model changed its
    idea of the scene's size in this frame, which one scale per scene cannot absorb.
    """
    out = []
    for index, truth in enumerate(truth_frames):
        e = frame_errors(depth[index] * scale, truth)
        moving = labels[index]["motion"][e["kept"]] == ob.MOVING if labels is not None else np.zeros(len(e["gt"]), bool)
        out.append({"frame": index, "lidar_pixels": int(len(e["gt"])),
                    "abs_rel": float(np.mean(np.abs(e["rel"]))) if len(e["gt"]) else float("nan"),
                    "median_signed_error": float(np.median(e["rel"])) if len(e["gt"]) else float("nan"),
                    "scale_ratio": float(np.median(e["gt"] / e["pred"])) if len(e["gt"]) else float("nan"),
                    "moving_share": float(moving.mean()) if len(moving) else 0.0,
                    "abs_rel_moving": float(np.mean(np.abs(e["rel"][moving]))) if moving.any() else float("nan")})
    return pd.DataFrame(out)


def scene_figure(images, depth, truth_frames, labels, scale: float, frame_indices, title: str):
    """One row per chosen frame: image | predicted depth | LiDAR depth | signed relative error (moving objects ringed)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    frame_indices = list(frame_indices)
    figure, axes = plt.subplots(len(frame_indices), 4, figsize=(19, 3.1 * len(frame_indices) + 0.9), squeeze=False,
                                layout="constrained")
    depth_norm = LogNorm(*DEPTH_RANGE_M)
    for row, index in enumerate(frame_indices):
        image, depth_m = np.asarray(images[index]), depth[index] * scale
        e = frame_errors(depth_m, truth_frames[index])
        ax_image, ax_pred, ax_truth, ax_error = axes[row]
        ax_image.imshow(image)
        ax_image.set_ylabel(f"frame {index}", fontsize=10)
        shown = ax_pred.imshow(np.clip(depth_m, *DEPTH_RANGE_M), cmap="turbo", norm=depth_norm)
        ax_truth.imshow(image, alpha=0.55)
        ax_truth.scatter(e["u"], e["v"], c=np.clip(e["gt"], *DEPTH_RANGE_M), cmap="turbo", norm=depth_norm, s=1.2, linewidths=0)
        ax_error.imshow(image, alpha=0.35)
        errors = ax_error.scatter(e["u"], e["v"], c=np.clip(e["rel"], -ERROR_CLIP, ERROR_CLIP), cmap="coolwarm",
                                  vmin=-ERROR_CLIP, vmax=ERROR_CLIP, s=1.6, linewidths=0)
        if labels is not None:
            moving = labels[index]["motion"][e["kept"]] == ob.MOVING
            # every sixth moving pixel gets a ring: enough to find the object, few enough to still see its colour
            ax_error.scatter(e["u"][moving][::6], e["v"][moving][::6], s=16, facecolors="none", edgecolors="black",
                             linewidths=0.4, alpha=0.8)
        abs_rel = float(np.mean(np.abs(e["rel"]))) if len(e["rel"]) else float("nan")
        ax_error.set_xlabel(f"AbsRel of this frame {abs_rel:.3f} | {len(e['rel'])} LiDAR pixels", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
        if row == 0:
            for axis, name in zip(axes[row], ("camera image", "predicted depth (m)", "LiDAR ground truth (m)",
                                              "signed relative error at the LiDAR pixels")):
                axis.set_title(name, fontsize=10)
    figure.colorbar(shown, ax=axes[:, 1:3].ravel().tolist(), fraction=0.015, pad=0.01, label="depth (m)")
    figure.colorbar(errors, ax=axes[:, 3].ravel().tolist(), fraction=0.03, pad=0.01, label="(pred - truth) / truth")
    figure.suptitle(title + chr(10) + "one scale per scene | error: red = predicted too far, blue = too near | black rings = moving objects",
                    fontsize=11)
    matplotlib.rcParams["figure.max_open_warning"] = 0
    return figure


def path_length_scale(pred_centres, gt_centres) -> float:
    """Metres per model unit from the LENGTH of the two paths. Always positive, on purpose: a trajectory
    predicted backwards then shows up as backwards instead of being flipped back by a negative scale."""
    pred = np.linalg.norm(np.diff(pred_centres, axis=0), axis=1).sum()
    gt = np.linalg.norm(np.diff(gt_centres, axis=0), axis=1).sum()
    return float(gt / pred) if pred > 0 else float("nan")


def trajectory_figure(extrinsics, gt_camera0_from_camera, title: str):
    """Camera path from above, in the first camera's frame: ground truth against prediction (scaled by path length)."""
    import matplotlib.pyplot as plt

    pred = mt.predicted_camera0_from_camera(extrinsics)[:, :3, 3]
    gt = to_4x4(np.asarray(gt_camera0_from_camera))[:, :3, 3]
    scale = path_length_scale(pred, gt)
    pred_m = pred * (scale if np.isfinite(scale) else 1.0)
    figure, (top, along) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for centres, colour, name in ((gt, "black", "ground truth"), (pred_m, "tab:red", "predicted (scaled to the same path length)")):
        top.plot(centres[:, 0], centres[:, 2], "-o", color=colour, markersize=2.5, linewidth=1, label=name)
        top.scatter(centres[:1, 0], centres[:1, 2], color=colour, s=45, marker="s")
        along.plot(np.arange(len(centres)), centres[:, 2], "-o", color=colour, markersize=2.5, linewidth=1, label=name)
    top.set_xlabel("X, to the right of the first camera (m)")
    top.set_ylabel("Z, forward of the first camera (m)")
    top.set_aspect("equal", adjustable="datalim")
    top.set_title("from above (square = first frame)", fontsize=10)
    top.legend(fontsize=8)
    along.set_xlabel("frame")
    along.set_ylabel("Z, forward (m)")
    along.set_title("distance driven forward, frame by frame", fontsize=10)
    along.axhline(0, color="grey", linewidth=0.5)
    figure.suptitle(title, fontsize=12)
    return figure


def scene_bundle(persist_root, dataset, scene_id: str, camera_id: str, prediction_folder: str) -> dict:
    """Everything the pictures need for one scene, from saved predictions and saved ground truth.

    Needs the camera layer on disk (for the images and the boxes). Builds nothing: a scene that was never
    scored raises, with the missing piece named.
    """
    from PIL import Image

    from . import pipeline as pl

    scene = dataset.get_scene(scene_id)
    frames = inf.select_frames(scene, camera_id)
    stamps = [int(frame.timestamp_ns) for frame in frames]
    arrays = pl.load_saved_predictions(persist_root, scene.name, camera_id, prediction_folder, stamps)
    if arrays is None:
        raise RuntimeError(f"{scene_id}: no saved predictions under {prediction_folder}")
    height, width = (int(v) for v in arrays["input_hw"])
    npz_path, json_path = gtm.truth_paths(persist_root, scene.name, camera_id, gtm.truth_tag(width, height))
    if not npz_path.is_file():
        raise RuntimeError(f"{scene_id}: no saved ground truth at {npz_path}")
    truth, truth_stamps, _ = gtm.load_scene_truth(npz_path, json_path)
    if not np.array_equal(np.asarray(truth_stamps, dtype=np.int64), np.asarray(stamps, dtype=np.int64)):
        raise RuntimeError(f"{scene_id}: saved ground truth is for other frames than the saved predictions")
    labels = ob.scene_motion_labels(frames, camera_id, truth)
    images = [np.asarray(Image.open(frame.camera_path(camera_id)).convert("RGB").resize((width, height), Image.BILINEAR))
              for frame in frames]
    scale = sequence_scale(arrays["depth"], truth)
    return {"scene_id": scene_id, "scene_name": scene.name, "arrays": arrays, "truth": truth, "labels": labels,
            "images": images, "scale": scale, "frames": len(frames)}
