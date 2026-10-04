"""Is the error on moving objects inside them, or only along their outlines?

Why this matters. Moving objects score worse than parked ones. Two explanations look the same in a table:

- the MODEL is worse on things that move (a finding about the model), or
- the GROUND TRUTH is worse on things that move: the LiDAR sweep and the camera exposure are not the same
  instant, so the points of an object crossing the view land a few pixels beside it in the image. That
  error sits along the object's outline. The trap-3 bound does not cover it: it only bounds motion ALONG
  the viewing direction.

The test. Split every object's LiDAR pixels into BOUNDARY (within `radius` pixels of a LiDAR pixel that
belongs to something else) and INTERIOR (the rest). Misregistration by a few pixels cannot reach the
interior of an object. So:

- moving INTERIOR against parked INTERIOR is the clean comparison;
- parked objects are the control for the outline itself: they cannot be misregistered by their own motion,
  so parked boundary minus parked interior is what the model's soft depth edges cost, and whatever moving
  objects lose at their boundary BEYOND that is the part misregistration can explain.

Known limit: LiDAR returns nothing from the sky, so an object's upper outline against the sky has no
"something else" pixel next to it and counts as interior. That dilutes both interiors alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as mt, objects as ob

DEFAULT_RADIUS = 6                    # pixels, at the model's input size (640 wide for VGGT-Omega)
NO_PIXEL, BACKGROUND_OWNER = -2, -1
MIN_PIXELS = 200                      # as in evaluation.MIN_PIXELS_PER_STRATUM


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation of a boolean image by `radius` pixels (rows, then columns)."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for axis in (0, 1):
        source = out.copy()
        for shift in range(1, radius + 1):
            for direction in (-1, 1):
                moved = np.roll(source, direction * shift, axis=axis)
                edge = [slice(None), slice(None)]          # np.roll wraps around: blank what came from the other side
                edge[axis] = slice(0, shift) if direction == 1 else slice(-shift, None)
                moved[tuple(edge)] = False
                out |= moved
    return out


def boundary_mask(u, v, owner, width: int, height: int, radius: int = DEFAULT_RADIUS) -> np.ndarray:
    """For every LiDAR pixel: True when it belongs to an object and lies within `radius` pixels of a LiDAR pixel
    that belongs to something else (the background or another object). Background pixels are always False.

    `owner`: per pixel, the index of the object it was labelled with, or BACKGROUND_OWNER.
    """
    u, v, owner = np.asarray(u, np.int64), np.asarray(v, np.int64), np.asarray(owner, np.int64)
    image = np.full((height, width), NO_PIXEL, dtype=np.int64)
    image[v, u] = owner
    out = np.zeros(len(owner), dtype=bool)
    for k in np.unique(owner[owner >= 0]):
        near_other = dilate((image != k) & (image != NO_PIXEL), radius)
        mine = owner == k
        out[mine] = near_other[v[mine], u[mine]]
    return out


def edge_rows(scene_id: str, depth, truth_frames: list, labels: list, radius: int = DEFAULT_RADIUS) -> pd.DataFrame:
    """Per scene: AbsRel of object pixels by motion (moving, parked), zone (interior, boundary, all), object
    category and depth band, under the primary one-scale-per-scene alignment. Background is included for reference.

    `labels[i]` needs "motion", "box_index" and "category" per LiDAR pixel (objects.scene_motion_labels).
    """
    height, width = depth.shape[1:]
    parts = []
    for index, (truth, label) in enumerate(zip(truth_frames, labels)):
        if len(truth["depth_m"]) == 0:
            continue
        u, v = truth["u"].astype(np.int64), truth["v"].astype(np.int64)
        parts.append(pd.DataFrame({
            "frame": index, "gt": truth["depth_m"].astype(np.float64), "pred": depth[index, v, u].astype(np.float64),
            "motion": label["motion"], "category": np.asarray(label["category"], dtype=object),
            "boundary": boundary_mask(u, v, label["box_index"], width, height, radius)}))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    table = table[(table["gt"] >= mt.DEPTH_MIN_M) & (table["gt"] <= mt.DEPTH_MAX_M)
                  & np.isfinite(table["pred"]) & (table["pred"] > 0)].reset_index(drop=True)
    if table.empty:
        return pd.DataFrame()
    aligned = mt.align_depth(table["pred"].to_numpy(), table["gt"].to_numpy(), table["frame"].to_numpy(), mt.PRIMARY_PROTOCOL)
    table["abs_err"] = np.abs(aligned - table["gt"].to_numpy()) / table["gt"].to_numpy()
    table["depth_band"] = mt.depth_band_labels(table["gt"].to_numpy())
    table["motion_name"] = table["motion"].map({ob.MOVING: "moving", ob.PARKED: "parked"})
    table["zone"] = np.where(table["boundary"], "boundary", "interior")

    rows = []

    def add(part, motion, zone, category, band):
        if len(part):
            rows.append({"scene_id": scene_id, "motion": motion, "zone": zone, "category": category, "depth_band": band,
                         "abs_rel": float(part["abs_err"].mean()), "median_abs_rel": float(part["abs_err"].median()),
                         "n_pixels": int(len(part))})

    background = table[table["motion"] == ob.BACKGROUND]
    add(background, "background", "all", "all", "all")
    for band, part in background.groupby("depth_band"):
        add(part, "background", "all", "all", str(band))
    objects = table[table["motion"].isin([ob.MOVING, ob.PARKED])]
    for motion, by_motion in objects.groupby("motion_name"):
        for zone, by_zone in [("all", by_motion), *by_motion.groupby("zone")]:
            add(by_zone, motion, zone, "all", "all")
            for band, part in by_zone.groupby("depth_band"):
                add(part, motion, zone, "all", str(band))
            for category, part in by_zone.groupby("category"):
                add(part, motion, zone, str(category), "all")
    return pd.DataFrame(rows)


def edge_summary(rows: pd.DataFrame, category: str = "all", depth_band: str = "all") -> pd.DataFrame:
    """Mean over scenes (95% bootstrap) for each motion and zone, for one category and one depth band."""
    part = rows[(rows["category"] == category) & (rows["depth_band"] == depth_band) & (rows["n_pixels"] >= MIN_PIXELS)]
    out = []
    for (motion, zone), group in part.groupby(["motion", "zone"]):
        mean, low, high = mt.bootstrap_mean_ci(group["abs_rel"].to_numpy())
        out.append({"motion": motion, "zone": zone, "n_scenes": int(len(group)), "pixels": int(group["n_pixels"].sum()),
                    "abs_rel": mean, "ci_low": low, "ci_high": high, "median_over_scenes": float(group["abs_rel"].median())})
    return pd.DataFrame(out)


COMPARISONS = (
    ("moving", "all", "parked", "all", "moving minus parked, whole objects (what the report showed)"),
    ("moving", "interior", "parked", "interior", "moving minus parked, INTERIOR only (misregistration cannot reach here)"),
    ("parked", "boundary", "parked", "interior", "parked: boundary minus interior (cost of soft depth edges alone)"),
    ("moving", "boundary", "moving", "interior", "moving: boundary minus interior (soft edges PLUS misregistration)"),
)


def edge_comparisons(rows: pd.DataFrame, category: str = "all", depth_band: str = "all") -> pd.DataFrame:
    """Paired differences over scenes that have BOTH sides with enough pixels."""
    part = rows[(rows["category"] == category) & (rows["depth_band"] == depth_band) & (rows["n_pixels"] >= MIN_PIXELS)].copy()
    part["key"] = part["motion"] + "|" + part["zone"]
    wide = part.pivot_table(index="scene_id", columns="key", values="abs_rel", aggfunc="first")
    out = []
    for motion_a, zone_a, motion_b, zone_b, meaning in COMPARISONS:
        a, b = f"{motion_a}|{zone_a}", f"{motion_b}|{zone_b}"
        if a not in wide or b not in wide:
            continue
        both = wide[[a, b]].dropna()
        if both.empty:
            continue
        mean, low, high = mt.bootstrap_mean_ci((both[a] - both[b]).to_numpy())
        out.append({"comparison": meaning, "n_scenes": int(len(both)), "mean_a": float(both[a].mean()), "mean_b": float(both[b].mean()),
                    "difference": mean, "ci_low": low, "ci_high": high, "scenes_where_a_is_worse": int((both[a] > both[b]).sum())})
    return pd.DataFrame(out)
