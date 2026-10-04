"""Evaluate one scene, then aggregate over scenes without reporting noise.

Unit of analysis: the SCENE. Pixels and frames inside a scene are strongly
correlated, so every mean and every interval is taken over scenes. A stratum
counts for a scene only if it holds MIN_PIXELS_PER_STRATUM pixels there, and an
aggregate is flagged "thin" when fewer than `min_scenes` scenes support it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as mt
from .objects import MOTION_NAMES

MIN_PIXELS_PER_STRATUM = 200
DEFAULT_MIN_SCENES = 5

SEMANTIC_GROUPS = {
    "flat": ("road", "sidewalk", "parking", "rail track", "terrain", "ground"),
    "vehicle": ("car", "truck", "bus", "on rails", "motorcycle", "bicycle", "portable", "caravan", "trailer"),
    "human": ("person", "rider", "bicyclist", "motorcyclist", "portable-rider"),
    "structure": ("building", "wall", "fence", "guard rail", "bridge", "tunnel"),
    "thin": ("pole", "traffic sign", "traffic light"),
    "vegetation": ("vegetation",),
}
_GROUP_OF_NAME = {name: group for group, names in SEMANTIC_GROUPS.items() for name in names}


PIXEL_TABLE_COLUMNS = {"frame": "int64", "gt": "float64", "pred": "float64", "conf": "float64", "motion": "uint8",
                       "motion_bound_m": "float32", "semantic_id": "uint16", "motion_name": "object",
                       "depth_band": "object", "semantic_group": "object", "confidence_quartile": "object"}


def pixel_table(depth, conf, truth_frames: list, motion_labels: list, class_names: dict) -> pd.DataFrame:
    """One row per ground-truth pixel of the scene, inside the evaluated depth range."""
    parts = []
    for index, (truth, labels) in enumerate(zip(truth_frames, motion_labels)):
        if len(truth["depth_m"]) == 0:
            continue
        v, u = truth["v"].astype(np.int64), truth["u"].astype(np.int64)
        parts.append(pd.DataFrame({
            "frame": index, "gt": truth["depth_m"].astype(np.float64),
            "pred": depth[index, v, u].astype(np.float64), "conf": conf[index, v, u].astype(np.float64),
            "motion": labels["motion"], "motion_bound_m": labels["motion_bound_m"],
            "semantic_id": truth["semantic_id"],
        }))
    if not parts:       # a scene with no ground truth at all: keep every column so callers need no special case
        return pd.DataFrame({column: pd.Series(dtype=kind) for column, kind in PIXEL_TABLE_COLUMNS.items()})
    table = pd.concat(parts, ignore_index=True)
    table = table[(table["gt"] >= mt.DEPTH_MIN_M) & (table["gt"] <= mt.DEPTH_MAX_M)
                  & np.isfinite(table["pred"]) & (table["pred"] > 0)].reset_index(drop=True)
    table["motion_name"] = table["motion"].map(MOTION_NAMES)
    table["depth_band"] = mt.depth_band_labels(table["gt"].to_numpy())
    names = table["semantic_id"].map(lambda i: class_names.get(int(i), "other"))
    table["semantic_group"] = names.map(lambda name: _GROUP_OF_NAME.get(name, "other"))
    if len(table) >= 4:
        ranks = table["conf"].rank(method="first")
        table["confidence_quartile"] = pd.qcut(ranks, 4, labels=["Q1 least sure", "Q2", "Q3", "Q4 most sure"]).astype(str)
    else:
        table["confidence_quartile"] = "too few pixels"
    return table


STRATA = ("motion_name", "depth_band", "semantic_group", "confidence_quartile")


def evaluate_scene(scene_id: str, arrays: dict, truth_frames: list, motion_labels: list,
                   class_names: dict, gt_fx: float, gt_fy: float) -> tuple[pd.DataFrame, dict]:
    """Depth rows (long form: protocol x stratum) and one dict of pose, intrinsics and bookkeeping."""
    pred_c0_from_c = mt.predicted_camera0_from_camera(arrays["extrinsics"])
    pose = mt.pose_summary(pred_c0_from_c, arrays["gt_camera0_from_camera"])
    table = pixel_table(arrays["depth"], arrays["depth_conf"], truth_frames, motion_labels, class_names)

    rows = []
    for protocol in mt.PROTOCOLS:
        aligned = mt.align_depth(table["pred"].to_numpy(), table["gt"].to_numpy(), table["frame"].to_numpy(),
                                 protocol, pose_scale=pose["metres_per_model_unit"])
        rows.append({"scene_id": scene_id, "protocol": protocol, "stratum_type": "all", "stratum": "all",
                     **mt.depth_summary(aligned, table["gt"].to_numpy())})
        for stratum_type in STRATA:
            for stratum, index in table.groupby(stratum_type).indices.items():
                rows.append({"scene_id": scene_id, "protocol": protocol, "stratum_type": stratum_type,
                             "stratum": str(stratum), **mt.depth_summary(aligned[index], table["gt"].to_numpy()[index])})

    moving = table[table["motion_name"] == "moving object"]
    scene = {
        "scene_id": scene_id, **pose, **mt.intrinsics_summary(arrays["intrinsics"], gt_fx, gt_fy),
        "pixels_evaluated": int(len(table)),
        "sequence_scale_m_per_unit": mt.median_scale(table["pred"], table["gt"]) if len(table) else float("nan"),
        "moving_pixels": int(len(moving)),
        # how wrong can the moving-object GROUND TRUTH itself be, relative to its depth? (objects.py, motion bound)
        "moving_gt_bound_rel_median": float((moving["motion_bound_m"] / moving["gt"]).median()) if len(moving) else float("nan"),
        "moving_gt_bound_rel_p90": float((moving["motion_bound_m"] / moving["gt"]).quantile(0.9)) if len(moving) else float("nan"),
    }
    # Do depth and pose agree on the scale? 1.0 means perfectly.
    depth_scale = scene["sequence_scale_m_per_unit"]
    scene["pose_scale_over_depth_scale"] = scene["metres_per_model_unit"] / depth_scale if np.isfinite(depth_scale) and depth_scale > 0 else float("nan")
    return pd.DataFrame(rows), scene


def aggregate(rows: pd.DataFrame, value: str = "abs_rel", min_scenes: int = DEFAULT_MIN_SCENES) -> pd.DataFrame:
    """Mean over scenes with a bootstrap interval, per protocol and stratum."""
    usable = rows[rows["n_pixels"] >= MIN_PIXELS_PER_STRATUM]
    out = []
    for (protocol, stratum_type, stratum), group in usable.groupby(["protocol", "stratum_type", "stratum"]):
        mean, low, high = mt.bootstrap_mean_ci(group[value].to_numpy())
        out.append({"protocol": protocol, "stratum_type": stratum_type, "stratum": stratum, "n_scenes": int(len(group)),
                    "pixels": int(group["n_pixels"].sum()), value: mean, "ci_low": low, "ci_high": high,
                    "median": float(np.nanmedian(group[value].to_numpy(dtype=float))),
                    "thin": bool(len(group) < min_scenes)})
    return pd.DataFrame(out)


def paired_difference(rows: pd.DataFrame, stratum_type: str, a: str, b: str, protocol: str = mt.PRIMARY_PROTOCOL,
                      value: str = "abs_rel") -> dict:
    """Mean of (a minus b) over scenes that have BOTH strata. The honest way to compare two strata."""
    usable = rows[(rows["protocol"] == protocol) & (rows["stratum_type"] == stratum_type)
                  & (rows["n_pixels"] >= MIN_PIXELS_PER_STRATUM)]
    wide = usable.pivot(index="scene_id", columns="stratum", values=value)
    if a not in wide or b not in wide:
        return {"a": a, "b": b, "n_scenes": 0, "mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    both = wide[[a, b]].dropna()
    mean, low, high = mt.bootstrap_mean_ci((both[a] - both[b]).to_numpy())
    return {"a": a, "b": b, "n_scenes": int(len(both)), "mean_difference": mean, "ci_low": low, "ci_high": high,
            "scenes_where_a_is_worse": int((both[a] > both[b]).sum())}


def aggregate_by_scene_attribute(rows: pd.DataFrame, scene_table: pd.DataFrame, attribute: str,
                                 protocol: str = mt.PRIMARY_PROTOCOL, value: str = "abs_rel",
                                 min_scenes: int = DEFAULT_MIN_SCENES) -> pd.DataFrame:
    """Scene-level strata such as road_type or weather: mean of each scene's overall score, per attribute value."""
    overall = rows[(rows["protocol"] == protocol) & (rows["stratum_type"] == "all")][["scene_id", value]]
    merged = overall.merge(scene_table[["scene_id", attribute]], on="scene_id", how="left")
    out = []
    for level, group in merged.groupby(attribute, dropna=False):
        mean, low, high = mt.bootstrap_mean_ci(group[value].to_numpy())
        out.append({attribute: level, "n_scenes": int(len(group)), "n_recordings": n_recordings(group["scene_id"]),
                    value: mean, "ci_low": low, "ci_high": high,
                    "median": float(np.nanmedian(group[value].to_numpy(dtype=float))), "thin": bool(len(group) < min_scenes)})
    return pd.DataFrame(out)


def recording_of(scene_id) -> str:
    """The drive a scene was cut from. Scene ids read "<recording>|<number>"."""
    return str(scene_id).split("|")[0]


def n_recordings(scene_ids) -> int:
    """Scenes cut from one drive share road, weather and light, so they are not independent evidence.
    The intervals in these tables resample SCENES; a stratum with many scenes but one or two recordings
    is therefore narrower than it deserves. This count makes that visible."""
    return int(len({recording_of(scene_id) for scene_id in scene_ids}))


def cross_table(rows: pd.DataFrame, scene_table: pd.DataFrame, attribute_a: str, attribute_b: str,
                protocol: str = mt.PRIMARY_PROTOCOL, value: str = "abs_rel",
                min_scenes: int = DEFAULT_MIN_SCENES) -> pd.DataFrame:
    """Two scene attributes at once, for example weather by lighting.

    Needed because attributes travel together: if nearly every wet scene is also a dark one, the two
    one-attribute tables show the same scenes twice and cannot say which attribute matters.
    """
    overall = rows[(rows["protocol"] == protocol) & (rows["stratum_type"] == "all")][["scene_id", value]]
    merged = overall.merge(scene_table[["scene_id", attribute_a, attribute_b]], on="scene_id", how="left")
    out = []
    for (level_a, level_b), group in merged.groupby([attribute_a, attribute_b], dropna=False):
        mean, low, high = mt.bootstrap_mean_ci(group[value].to_numpy())
        out.append({attribute_a: level_a, attribute_b: level_b, "n_scenes": int(len(group)),
                    "n_recordings": n_recordings(group["scene_id"]), value: mean, "ci_low": low, "ci_high": high,
                    "median": float(np.nanmedian(group[value].to_numpy(dtype=float))),
                    "thin": bool(len(group) < min_scenes)})
    return pd.DataFrame(out)


def compare_splits(rows: pd.DataFrame, scene_table: pd.DataFrame, split: str = "test",
                   attributes=("weather_group", "lighting"), min_scenes: int = DEFAULT_MIN_SCENES) -> pd.DataFrame:
    """Like for like: one split against all other scenes, condition by condition.

    The test split is almost all daytime, so its overall number must not be set against an overall number
    that includes the night drives. Inside one condition (say dry and day) the two groups are comparable.
    """
    a, b = attributes
    inside = scene_table[scene_table["split"] == split]
    outside = scene_table[scene_table["split"] != split]
    if inside.empty or outside.empty:
        return pd.DataFrame()
    keep = [a, b, "n_scenes", "n_recordings", "abs_rel", "ci_low", "ci_high", "median"]
    left = cross_table(rows[rows["scene_id"].isin(inside["scene_id"])], inside, a, b, min_scenes=min_scenes)[keep]
    right = cross_table(rows[rows["scene_id"].isin(outside["scene_id"])], outside, a, b, min_scenes=min_scenes)[keep]
    merged = left.merge(right, on=[a, b], how="outer", suffixes=(f"_{split}", "_others"))
    merged["difference"] = merged[f"abs_rel_{split}"] - merged["abs_rel_others"]
    # separated = the two 95% intervals do not overlap (a blunt but honest reading of "clearly different")
    merged["intervals_overlap"] = ~((merged[f"ci_low_{split}"] > merged["ci_high_others"])
                                    | (merged[f"ci_high_{split}"] < merged["ci_low_others"]))
    return merged


OVERVIEW_CONTEXT = ("split", "block", "road_type", "weather_group", "lighting", "speed_band", "speed_kph_median")
OVERVIEW_METRICS = ("rotation_deg_median", "translation_deg_median", "ate_scale_only_pct_of_path",
                    "pose_scale_over_depth_scale", "fx_rel_err_median", "near_stationary", "pixels_evaluated")


def scene_overview(rows: pd.DataFrame, scene_table: pd.DataFrame, protocol: str = mt.PRIMARY_PROTOCOL) -> pd.DataFrame:
    """One line per scene: where it is, what it is like, and how the model did on depth, moving objects and pose."""
    primary = rows[rows["protocol"] == protocol]
    overall = primary[primary["stratum_type"] == "all"].set_index("scene_id")["abs_rel"]
    motion = primary[(primary["stratum_type"] == "motion_name") & (primary["n_pixels"] >= MIN_PIXELS_PER_STRATUM)]
    motion = motion.pivot(index="scene_id", columns="stratum", values="abs_rel")
    table = scene_table[["scene_id", *[c for c in OVERVIEW_CONTEXT if c in scene_table]]].copy()
    table.insert(1, "recording", table["scene_id"].map(recording_of))
    table["abs_rel"] = table["scene_id"].map(overall)
    for name, column in (("abs_rel_background", "background"), ("abs_rel_moving", "moving object")):
        table[name] = table["scene_id"].map(motion[column]) if column in motion else float("nan")
    for column in OVERVIEW_METRICS:
        if column in scene_table:
            table[column] = scene_table[column].to_numpy()
    if "pose_scale_over_depth_scale" in table:
        # 0 = pose and depth agree on the scale; 0.69 = they differ by a factor of two, in either direction
        ratio = pd.to_numeric(table["pose_scale_over_depth_scale"], errors="coerce")
        table["scale_disagreement"] = np.abs(np.log(ratio.where(ratio > 0)))
    return table.reset_index(drop=True)


def worst_scenes(overview: pd.DataFrame, by: str, n: int = 12, moving_only: bool = False) -> pd.DataFrame:
    """The n scenes with the highest value of `by`. moving_only drops scenes that barely move (no usable trajectory)."""
    table = overview
    if moving_only and "near_stationary" in table:
        table = table[~table["near_stationary"].astype(bool)]
    return table.dropna(subset=[by]).sort_values(by, ascending=False).head(n).reset_index(drop=True)


SCENE_ATTRIBUTES = ("road_type", "weather_group", "lighting", "speed_band")
CROSS_TABLES = (("weather_group", "lighting"), ("road_type", "weather_group"))
POSE_COLUMNS = ("rotation_deg_median", "translation_deg_median", "auc3", "auc30", "auc30_authors_unsigned",
                "ate_scale_only_m", "ate_scale_only_pct_of_path", "ate_sim3_m")
OTHER_COLUMNS = ("fx_rel_err_median", "fy_rel_err_median", "fx_spread_rel", "pose_scale_over_depth_scale",
                 "moving_gt_bound_rel_median", "moving_gt_bound_rel_p90")


def _scene_means(scene_table: pd.DataFrame, columns) -> pd.DataFrame:
    out = []
    for column in columns:
        if column in scene_table:
            mean, low, high = mt.bootstrap_mean_ci(scene_table[column].to_numpy(dtype=float))
            out.append({"metric": column, "n_scenes": int(scene_table[column].notna().sum()),
                        "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(out)


REVERSED_DEG = 90.0       # translation direction off by more than this: the model has the car driving the other way


def pose_typical_and_failures(scene_table: pd.DataFrame) -> pd.DataFrame:
    """The typical scene (median over scenes) and how many scenes fail outright.

    The mean in the pose table is pulled far up by a handful of scenes where the direction of travel comes
    out reversed (errors near 180 degrees). Median and failure count say the same thing without that distortion.
    """
    out = []
    for column in ("rotation_deg_median", "translation_deg_median", "ate_scale_only_pct_of_path"):
        if column in scene_table:
            values = pd.to_numeric(scene_table[column], errors="coerce").dropna()
            out.append({"metric": f"{column}, median over scenes", "n_scenes": int(len(values)),
                        "value": float(values.median()) if len(values) else float("nan")})
    if "translation_deg_median" in scene_table:
        values = pd.to_numeric(scene_table["translation_deg_median"], errors="coerce").dropna()
        out.append({"metric": f"scenes with direction of travel reversed (over {REVERSED_DEG:.0f} deg)",
                    "n_scenes": int(len(values)), "value": float((values > REVERSED_DEG).sum())})
    if "pose_scale_over_depth_scale" in scene_table:
        ratio = pd.to_numeric(scene_table["pose_scale_over_depth_scale"], errors="coerce").dropna()
        out.append({"metric": "pose scale over depth scale, median over scenes", "n_scenes": int(len(ratio)),
                    "value": float(ratio.median()) if len(ratio) else float("nan")})
        out.append({"metric": "scenes where the two scales differ by more than a factor of 1.5 (or the sign flips)",
                    "n_scenes": int(len(ratio)), "value": float(((ratio <= 0) | (ratio > 1.5) | (ratio < 1 / 1.5)).sum())})
    return pd.DataFrame(out)


def build_report(rows: pd.DataFrame, scene_table: pd.DataFrame, min_scenes: int = DEFAULT_MIN_SCENES) -> dict:
    """Every table of the writeup, from the per-scene rows. A pure function, so it is tested locally."""
    depth = aggregate(rows, "abs_rel", min_scenes)
    delta = aggregate(rows, "delta125", min_scenes)[["protocol", "stratum_type", "stratum", "delta125"]]
    depth = depth.merge(delta, on=["protocol", "stratum_type", "stratum"], how="left")
    primary = depth[depth["protocol"] == mt.PRIMARY_PROTOCOL]

    def stratum(name):
        return primary[primary["stratum_type"] == name].drop(columns=["protocol", "stratum_type"]).reset_index(drop=True)

    # A scene that barely moves has no usable trajectory, so it is kept out of the pose table.
    with_motion = scene_table[~scene_table["near_stationary"].astype(bool)] if "near_stationary" in scene_table else scene_table
    report = {
        "headline_by_protocol": depth[depth["stratum_type"] == "all"].drop(columns=["stratum_type", "stratum"]).reset_index(drop=True),
        "motion": stratum("motion_name"),
        "depth_band": stratum("depth_band"),
        "semantic_group": stratum("semantic_group"),
        "confidence_quartile": stratum("confidence_quartile"),
        "moving_vs_background": pd.DataFrame([paired_difference(rows, "motion_name", "moving object", "background"),
                                              paired_difference(rows, "motion_name", "parked object", "background"),
                                              paired_difference(rows, "motion_name", "moving object", "parked object")]),
        "pose_moving_scenes_only": _scene_means(with_motion, POSE_COLUMNS),
        "pose_typical_and_failures": pose_typical_and_failures(with_motion),
        "intrinsics_scale_and_trap3": _scene_means(scene_table, OTHER_COLUMNS),
    }
    for attribute in SCENE_ATTRIBUTES:
        if attribute in scene_table:
            report[f"by_{attribute}"] = aggregate_by_scene_attribute(rows, scene_table, attribute, min_scenes=min_scenes)
    for a, b in CROSS_TABLES:
        if a in scene_table and b in scene_table:
            report[f"by_{a}_and_{b}"] = cross_table(rows, scene_table, a, b, min_scenes=min_scenes)
    return report


COMPARED_SCENE_METRICS = ("rotation_deg_median", "translation_deg_median", "auc30", "ate_scale_only_pct_of_path",
                          "fx_rel_err_median", "fy_rel_err_median", "pose_scale_over_depth_scale")


def compare_models(rows_a: pd.DataFrame, scenes_a: pd.DataFrame, rows_b: pd.DataFrame, scenes_b: pd.DataFrame,
                   name_a: str, name_b: str, protocol: str = mt.PRIMARY_PROTOCOL) -> pd.DataFrame:
    """Paired comparison of two models on the SAME scenes. difference = a minus b, per scene, then averaged.

    For AbsRel, rotation, translation, ATE: negative means model a is better. For AUC: positive means a is better.
    Each model is scored against ground truth built at its OWN input size, so the pixel sets differ slightly;
    the scenes, the LiDAR points and every rule are identical.
    """
    out = []

    def add(quantity, a: pd.Series, b: pd.Series):
        both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        mean, low, high = mt.bootstrap_mean_ci((both["a"] - both["b"]).to_numpy())
        out.append({"quantity": quantity, "n_scenes": int(len(both)), name_a: float(both["a"].mean()) if len(both) else float("nan"),
                    name_b: float(both["b"].mean()) if len(both) else float("nan"), "difference_a_minus_b": mean,
                    "ci_low": low, "ci_high": high, "scenes_where_a_is_lower": int((both["a"] < both["b"]).sum())})

    def depth(rows, stratum_type, stratum):
        chosen = rows[(rows["protocol"] == protocol) & (rows["stratum_type"] == stratum_type) & (rows["stratum"] == stratum)
                      & (rows["n_pixels"] >= MIN_PIXELS_PER_STRATUM)]
        return chosen.set_index("scene_id")["abs_rel"]

    add("AbsRel, all pixels", depth(rows_a, "all", "all"), depth(rows_b, "all", "all"))
    for stratum in ("background", "parked object", "moving object"):
        add(f"AbsRel, {stratum}", depth(rows_a, "motion_name", stratum), depth(rows_b, "motion_name", stratum))
    moving_a = scenes_a[~scenes_a["near_stationary"].astype(bool)].set_index("scene_id")
    moving_b = scenes_b[~scenes_b["near_stationary"].astype(bool)].set_index("scene_id")
    for metric in COMPARED_SCENE_METRICS:
        if metric in moving_a and metric in moving_b:
            add(metric, moving_a[metric].astype(float), moving_b[metric].astype(float))
    return pd.DataFrame(out)
