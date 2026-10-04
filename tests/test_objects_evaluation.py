from types import SimpleNamespace

import numpy as np
import pandas as pd

from vggt_aura import evaluation as ev, metrics as mt, objects as ob

# Optical camera mounted facing forward: camera Z = base_link X, camera X = -Y (right), camera Y = -Z (down).
BASE_FROM_CAMERA = np.eye(4)
BASE_FROM_CAMERA[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
HALF_SECOND = 500_000_000


def box(object_id, center, size=(4.6, 1.8, 1.6), rotation=(0, 0, 0, 1)):
    return SimpleNamespace(object_id=object_id, center=np.array(center, float), size_lwh=np.array(size, float),
                           rotation_xyzw=np.array(rotation, float))


# ------------------------------------------------------------------ speeds from tracks

def observations(object_id, centres, start=0, gap=1):
    return [(start + k * gap, (start + k * gap) * HALF_SECOND, object_id, c) for k, c in enumerate(centres)]


def test_speed_comes_from_the_objects_own_track():
    moving = observations("car:1", [[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]])           # 2.5 m per 0.5 s = 5 m/s
    parked = observations("car:2", [[9, 1, 0], [9.05, 1, 0], [9.0, 1, 0]])          # 5 cm of annotation jitter
    velocities = ob.track_velocities(moving + parked)
    assert np.allclose(velocities[(1, "car:1")], [5, 0, 0]) and np.allclose(velocities[(0, "car:1")], [5, 0, 0])
    assert ob.motion_label(velocities[(1, "car:1")]) == ob.MOVING
    assert ob.motion_label(velocities[(1, "car:2")]) == ob.PARKED


def test_in_between_speeds_and_missing_tracks_stay_out_of_both_strata():
    assert ob.motion_label(np.array([0.7, 0, 0])) == ob.UNCERTAIN
    assert ob.motion_label(None) == ob.UNCERTAIN
    assert ob.track_velocities(observations("car:3", [[0, 0, 0]]))[(0, "car:3")] is None      # seen once
    far_apart = ob.track_velocities(observations("car:4", [[0, 0, 0], [50, 0, 0]], gap=10))   # 5 s apart
    assert far_apart[(0, "car:4")] is None


# ------------------------------------------------------------------ labelling points

def test_points_inside_a_moving_box_are_labelled_with_the_right_bound():
    points_cam = np.array([[0.0, 0.0, 10.0],      # 10 m ahead: base (10, 0, 0) -> inside the box
                           [0.0, 0.0, 30.0]])     # 30 m ahead: background
    towards = ob.label_points(points_cam, BASE_FROM_CAMERA, np.eye(4), [box("car:1", [10, 0, 0])],
                              {"car:1": np.array([5.0, 0.0, 0.0])})                # driving away along the view axis
    assert towards["motion"].tolist() == [ob.MOVING, ob.BACKGROUND]
    assert np.isclose(towards["motion_bound_m"][0], 0.5) and towards["motion_bound_m"][1] == 0   # 5 m/s x 0.1 s
    across = ob.label_points(points_cam, BASE_FROM_CAMERA, np.eye(4), [box("car:1", [10, 0, 0])],
                             {"car:1": np.array([0.0, 5.0, 0.0])})                 # crossing sideways
    assert across["motion"][0] == ob.MOVING and np.isclose(across["motion_bound_m"][0], 0.0)     # depth is not affected


def test_bound_uses_the_ego_heading():
    quarter_turn = np.eye(4)
    quarter_turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                      # ego faces odom +Y
    out = ob.label_points(np.array([[0.0, 0.0, 10.0]]), BASE_FROM_CAMERA, quarter_turn, [box("car:1", [10, 0, 0])],
                          {"car:1": np.array([0.0, 5.0, 0.0])})                    # odom +Y is now straight ahead
    assert np.isclose(out["motion_bound_m"][0], 0.5)


def test_first_box_wins_and_no_boxes_means_background():
    point = np.array([[0.0, 0.0, 10.0]])
    out = ob.label_points(point, BASE_FROM_CAMERA, np.eye(4), [box("a", [10, 0, 0]), box("b", [10.2, 0, 0])],
                          {"a": np.zeros(3), "b": np.array([9.0, 0, 0])})
    assert out["motion"].tolist() == [ob.PARKED]
    assert ob.label_points(point, BASE_FROM_CAMERA, np.eye(4), [], {})["motion"].tolist() == [ob.BACKGROUND]
    assert len(ob.label_points(np.zeros((0, 3)), BASE_FROM_CAMERA, np.eye(4), [box("a", [1, 0, 0])], {})["motion"]) == 0


# ------------------------------------------------------------------ one synthetic scene, end to end

H, W, FRAMES = 40, 64, 6
CLASS_NAMES = {1: "road", 7: "car", 16: "building", 22: "pole"}


def synthetic_scene(moving_error=1.25, seed=0):
    """Predictions equal to truth / 50, except on moving-object pixels where depth is inflated."""
    rng = np.random.default_rng(seed)
    depth = np.ones((FRAMES, H, W))
    conf = rng.uniform(1.0, 30.0, (FRAMES, H, W))
    truth, labels = [], []
    for f in range(FRAMES):
        v, u = np.meshgrid(np.arange(0, H, 2), np.arange(0, W, 2), indexing="ij")
        v, u = v.ravel(), u.ravel()
        gt = rng.uniform(3.0, 70.0, len(v))
        moving = u < 16
        depth[f, v, u] = gt / 50.0 * np.where(moving, moving_error, 1.0)
        truth.append({"u": u.astype(np.int16), "v": v.astype(np.int16), "depth_m": gt.astype(np.float32),
                      "semantic_id": np.where(moving, 7, 1).astype(np.uint16)})
        labels.append({"motion": np.where(moving, ob.MOVING, ob.BACKGROUND).astype(np.uint8),
                       "motion_bound_m": np.where(moving, 0.4, 0.0).astype(np.float32)})
    centres = np.column_stack([np.zeros(FRAMES), np.zeros(FRAMES), np.arange(FRAMES) * 4.0])
    gt_poses = np.tile(np.eye(4), (FRAMES, 1, 1))
    gt_poses[:, :3, 3] = centres
    extrinsics = np.tile(np.eye(4), (FRAMES, 1, 1))
    extrinsics[:, :3, 3] = -centres / 50.0                                  # camera_from_world, so t = -centre
    intrinsics = np.tile(np.array([[497.0, 0, 32], [0, 498.0, 20], [0, 0, 1]]), (FRAMES, 1, 1))
    arrays = {"depth": depth, "depth_conf": conf, "extrinsics": extrinsics[:, :3], "intrinsics": intrinsics,
              "gt_camera0_from_camera": gt_poses}
    return arrays, truth, labels


def test_scene_evaluation_finds_the_planted_moving_object_error():
    arrays, truth, labels = synthetic_scene()
    rows, scene = ev.evaluate_scene("s|1", arrays, truth, labels, CLASS_NAMES, 538.2, 564.7)
    primary = rows[rows["protocol"] == "sequence_scale"].set_index(["stratum_type", "stratum"])
    assert primary.loc[("motion_name", "background"), "abs_rel"] < 1e-6     # ground truth is stored as float32
    assert np.isclose(primary.loc[("motion_name", "moving object"), "abs_rel"], 0.25, atol=1e-6)
    assert np.isclose(primary.loc[("semantic_group", "vehicle"), "abs_rel"], 0.25, atol=1e-6)
    assert set(rows["protocol"]) == set(mt.PROTOCOLS)
    assert np.isclose(scene["metres_per_model_unit"], 50.0) and np.isclose(scene["sequence_scale_m_per_unit"], 50.0)
    assert np.isclose(scene["pose_scale_over_depth_scale"], 1.0)             # depth and pose agree on the scale
    assert scene["ate_scale_only_m"] < 1e-6 and scene["moving_pixels"] > 0
    assert 0 < scene["moving_gt_bound_rel_median"] < 0.2                     # trap 3, carried through


def test_pixels_outside_the_depth_range_are_not_evaluated():
    arrays, truth, labels = synthetic_scene()
    truth[0]["depth_m"][:5] = [0.5, 0.9, 81.0, 200.0, 500.0]
    table = ev.pixel_table(arrays["depth"], arrays["depth_conf"], truth, labels, CLASS_NAMES)
    assert table["gt"].between(mt.DEPTH_MIN_M, mt.DEPTH_MAX_M).all()
    assert len(table) == sum(len(t["depth_m"]) for t in truth) - 5


def test_aggregation_is_over_scenes_and_flags_thin_cells():
    all_rows = []
    for k in range(6):
        arrays, truth, labels = synthetic_scene(moving_error=1.10 + 0.05 * k, seed=k)
        all_rows.append(ev.evaluate_scene(f"s|{k}", arrays, truth, labels, CLASS_NAMES, 538.2, 564.7)[0])
    rows = pd.concat(all_rows)
    table = ev.aggregate(rows).set_index(["protocol", "stratum_type", "stratum"])
    moving = table.loc[("sequence_scale", "motion_name", "moving object")]
    assert moving["n_scenes"] == 6 and not moving["thin"]
    assert np.isclose(moving["abs_rel"], np.mean([0.10 + 0.05 * k for k in range(6)]), atol=1e-6)
    assert moving["ci_low"] < moving["abs_rel"] < moving["ci_high"]
    assert ev.aggregate(rows[rows["scene_id"].isin(["s|0", "s|1"])])["thin"].all()            # 2 scenes: thin

    paired = ev.paired_difference(rows, "motion_name", "moving object", "background")
    assert paired["n_scenes"] == 6 and paired["scenes_where_a_is_worse"] == 6 and paired["ci_low"] > 0
    assert ev.paired_difference(rows, "motion_name", "moving object", "no such stratum")["n_scenes"] == 0


def test_tiny_strata_do_not_count_for_a_scene():
    arrays, truth, labels = synthetic_scene()
    rows, _ = ev.evaluate_scene("s|1", arrays, truth, labels, CLASS_NAMES, 538.2, 564.7)
    rows.loc[rows["stratum"] == "moving object", "n_pixels"] = ev.MIN_PIXELS_PER_STRATUM - 1
    assert "moving object" not in set(ev.aggregate(rows)["stratum"])


def test_scene_level_strata():
    all_rows = [ev.evaluate_scene(f"s|{k}", *synthetic_scene(seed=k), CLASS_NAMES, 538.2, 564.7)[0] for k in range(6)]
    scenes = pd.DataFrame({"scene_id": [f"s|{k}" for k in range(6)], "road_type": ["urban"] * 5 + ["overland"]})
    out = ev.aggregate_by_scene_attribute(pd.concat(all_rows), scenes, "road_type").set_index("road_type")
    assert out.loc["urban", "n_scenes"] == 5 and not out.loc["urban", "thin"]
    assert out.loc["overland", "n_scenes"] == 1 and out.loc["overland", "thin"]


def test_report_builds_every_table_from_synthetic_scenes():
    all_rows, scenes = [], []
    for k in range(6):
        rows, scene = ev.evaluate_scene(f"s|{k}", *synthetic_scene(moving_error=1.2, seed=k), CLASS_NAMES, 538.2, 564.7)
        all_rows.append(rows)
        scenes.append({**scene, "road_type": "urban" if k < 5 else "overland", "weather_group": "dry",
                       "lighting": "day", "speed_band": "20-50 km/h"})
    report = ev.build_report(pd.concat(all_rows), pd.DataFrame(scenes))
    assert set(report) >= {"headline_by_protocol", "motion", "depth_band", "semantic_group", "confidence_quartile",
                           "moving_vs_background", "pose_moving_scenes_only", "intrinsics_scale_and_trap3",
                           "by_road_type", "by_weather_group", "by_lighting", "by_speed_band"}
    headline = report["headline_by_protocol"].set_index("protocol")
    assert set(headline.index) == set(mt.PROTOCOLS)
    assert headline.loc["unaligned", "abs_rel"] > headline.loc["sequence_scale", "abs_rel"]
    # The authors' scale-and-shift is median-based, not a best fit. With 25% of pixels corrupted,
    # as here, it scores WORSE than one scale per sequence. "More freedom" does not mean "lower error".
    assert headline.loc["frame_scale_shift", "abs_rel"] > headline.loc["sequence_scale", "abs_rel"]
    motion = report["motion"].set_index("stratum")
    assert np.isclose(motion.loc["moving object", "abs_rel"], 0.2, atol=1e-6) and motion.loc["moving object", "n_scenes"] == 6
    first = report["moving_vs_background"].iloc[0]
    assert first["n_scenes"] == 6 and first["ci_low"] > 0
    assert report["by_road_type"].set_index("road_type").loc["overland", "thin"]
    fx = report["intrinsics_scale_and_trap3"].set_index("metric").loc["fx_rel_err_median"]
    assert fx["mean"] < 0 and fx["n_scenes"] == 6
    assert report["pose_moving_scenes_only"].set_index("metric").loc["auc30", "n_scenes"] == 6


def confounded_run():
    """Eight scenes from three drives. Every wet scene is dark except one: the situation of the first batch."""
    ids = ["driveA|1", "driveA|2", "driveA|3", "driveB|1", "driveB|2", "driveB|3", "driveC|1", "driveC|2"]
    errors = [1.05, 1.05, 1.05, 1.40, 1.40, 1.40, 1.40, 1.10]
    all_rows, scenes = [], []
    for k, (scene_id, error) in enumerate(zip(ids, errors)):
        rows, scene = ev.evaluate_scene(scene_id, *synthetic_scene(moving_error=error, seed=k), CLASS_NAMES, 538.2, 564.7)
        all_rows.append(rows)
        scenes.append({**scene, "split": "train", "block": k // 3, "road_type": "urban",
                       "weather_group": "dry" if k < 3 else "wet", "lighting": "day" if k < 3 or k == 7 else "night",
                       "speed_band": "20-50 km/h"})
    return pd.concat(all_rows), pd.DataFrame(scenes)


def test_cross_table_separates_what_the_single_tables_mix_up_and_counts_recordings():
    rows, scenes = confounded_run()
    cross = ev.cross_table(rows, scenes, "weather_group", "lighting").set_index(["weather_group", "lighting"])
    assert set(cross.index) == {("dry", "day"), ("wet", "night"), ("wet", "day")}          # no dry night scene exists
    assert cross.loc[("dry", "day"), "n_scenes"] == 3 and cross.loc[("dry", "day"), "n_recordings"] == 1
    assert cross.loc[("wet", "night"), "n_scenes"] == 4 and cross.loc[("wet", "night"), "n_recordings"] == 2
    assert cross.loc[("wet", "day"), "n_scenes"] == 1 and cross.loc[("wet", "day"), "thin"]
    assert cross["n_scenes"].sum() == 8
    single = ev.aggregate_by_scene_attribute(rows, scenes, "weather_group").set_index("weather_group")
    assert single.loc["wet", "n_scenes"] == 5 and single.loc["wet", "n_recordings"] == 2
    report = ev.build_report(rows, scenes)
    assert {"by_weather_group_and_lighting", "by_road_type_and_weather_group"} <= set(report)


def test_a_few_reversed_scenes_do_not_hide_the_typical_one():
    scenes = pd.DataFrame({"scene_id": [f"d|{k}" for k in range(10)],
                           "rotation_deg_median": [1.0] * 10, "ate_scale_only_pct_of_path": [2.0] * 10,
                           "translation_deg_median": [1.0] * 8 + [170.0, 120.0],
                           "pose_scale_over_depth_scale": [1.0] * 6 + [1.4, 1.6, -3.0, 0.5]})
    table = ev.pose_typical_and_failures(scenes).set_index("metric")["value"]
    assert scenes["translation_deg_median"].mean() > 29                   # what the mean alone would suggest
    assert table["translation_deg_median, median over scenes"] == 1.0     # what a typical scene looks like
    assert table["scenes with direction of travel reversed (over 90 deg)"] == 2
    assert table["scenes where the two scales differ by more than a factor of 1.5 (or the sign flips)"] == 3
    rows, confounded = confounded_run()
    assert "pose_typical_and_failures" in ev.build_report(rows, confounded)


def test_scene_overview_and_worst_scenes():
    rows, scenes = confounded_run()
    scenes.loc[2, "pose_scale_over_depth_scale"] = 0.5          # off by a factor of two, downwards
    scenes.loc[5, "pose_scale_over_depth_scale"] = 3.0          # off by a factor of three, upwards
    scenes["near_stationary"] = [False] * 7 + [True]
    overview = ev.scene_overview(rows, scenes)
    assert len(overview) == 8 and overview["scene_id"].is_unique
    assert list(overview["recording"].unique()) == ["driveA", "driveB", "driveC"]
    overall = rows[(rows["protocol"] == mt.PRIMARY_PROTOCOL) & (rows["stratum_type"] == "all")].set_index("scene_id")["abs_rel"]
    assert np.allclose(overview.set_index("scene_id")["abs_rel"], overall.loc[overview["scene_id"]])
    assert np.isclose(overview.loc[3, "abs_rel_moving"], 0.4, atol=1e-6)
    worst = ev.worst_scenes(overview, "scale_disagreement", n=2)
    assert list(worst["scene_id"]) == ["driveB|3", "driveA|3"]                          # log scale: 3x beats 1/2x
    assert np.isclose(worst.loc[1, "scale_disagreement"], np.log(2))
    assert "driveC|2" not in set(ev.worst_scenes(overview, "abs_rel", n=8, moving_only=True)["scene_id"])
    assert ev.worst_scenes(overview, "abs_rel", n=3)["abs_rel"].is_monotonic_decreasing


def test_a_scene_with_no_ground_truth_is_scored_as_empty_not_as_a_crash():
    arrays, truth, labels = synthetic_scene()
    empty_truth = [{"u": np.zeros(0, np.int16), "v": np.zeros(0, np.int16), "depth_m": np.zeros(0, np.float32),
                    "semantic_id": np.zeros(0, np.uint16)} for _ in truth]
    empty_labels = [{"motion": np.zeros(0, np.uint8), "motion_bound_m": np.zeros(0, np.float32)} for _ in truth]
    rows, scene = ev.evaluate_scene("s|empty", arrays, empty_truth, empty_labels, CLASS_NAMES, 538.2, 564.7)
    assert set(rows["stratum_type"]) == {"all"} and (rows["n_pixels"] == 0).all()
    assert scene["pixels_evaluated"] == 0 and np.isnan(scene["sequence_scale_m_per_unit"])
    assert np.isnan(scene["pose_scale_over_depth_scale"]) and scene["ate_scale_only_m"] < 1e-6   # pose is still scored
    normal = ev.evaluate_scene("s|0", *synthetic_scene(), CLASS_NAMES, 538.2, 564.7)[0]
    assert "s|empty" not in set(ev.aggregate(pd.concat([rows, normal]))["stratum"])              # and never enters a mean
    assert ev.aggregate(pd.concat([rows, normal])).query("stratum == 'all' and protocol == 'sequence_scale'")["n_scenes"].iloc[0] == 1


def test_model_against_model_is_paired_by_scene():
    rows_a, rows_b, scenes_a, scenes_b = [], [], [], []
    for k in range(6):
        ra, sa = ev.evaluate_scene(f"s|{k}", *synthetic_scene(moving_error=1.10, seed=k), CLASS_NAMES, 538.2, 564.7)
        rb, sb = ev.evaluate_scene(f"s|{k}", *synthetic_scene(moving_error=1.30, seed=k), CLASS_NAMES, 538.2, 564.7)
        rows_a.append(ra); rows_b.append(rb); scenes_a.append(sa); scenes_b.append(sb)
    extra, extra_scene = ev.evaluate_scene("s|only_in_a", *synthetic_scene(seed=99), CLASS_NAMES, 538.2, 564.7)
    table = ev.compare_models(pd.concat(rows_a + [extra]), pd.DataFrame(scenes_a + [extra_scene]),
                              pd.concat(rows_b), pd.DataFrame(scenes_b), "omega", "vggt").set_index("quantity")
    moving = table.loc["AbsRel, moving object"]
    assert moving["n_scenes"] == 6                                   # the scene only one model has is ignored
    assert np.isclose(moving["omega"], 0.10, atol=1e-6) and np.isclose(moving["vggt"], 0.30, atol=1e-6)
    assert np.isclose(moving["difference_a_minus_b"], -0.20, atol=1e-6) and moving["scenes_where_a_is_lower"] == 6
    assert abs(table.loc["AbsRel, background", "difference_a_minus_b"]) < 1e-6
    assert "auc30" in table.index and table.loc["auc30", "n_scenes"] == 6


def test_tracks_loaded_once_give_the_same_labels_as_loading_them_per_model():
    class Calibration:
        def base_from_sensor(self, name):
            return np.eye(4)

    class Frame:
        def __init__(self, k):
            self.timestamp_ns, self.k, self.loads = int(k * 1e8), k, 0

        def calibration(self):
            return Calibration()

        def load_ego_pose(self):
            pose = np.eye(4)
            pose[0, 3] = 1.0 * self.k
            return pose

        def load_boxes(self, frame):
            self.loads += 1
            return [box("car", (10.0 + 1.5 * self.k, 0.0, 0.0)), box("parked", (20.0 - 1.0 * self.k, 3.0, 0.0))]

    frames = [Frame(k) for k in range(5)]
    rng = np.random.default_rng(0)
    truth = [{"xyz_cam": np.column_stack([rng.uniform(5, 25, 400), rng.uniform(-1, 4, 400), rng.uniform(-1, 1, 400)])}
             for _ in frames]
    plain = ob.scene_motion_labels(frames, "front_medium", truth)
    tracks = ob.scene_tracks(frames)
    shared = ob.scene_motion_labels(frames, "front_medium", truth, tracks)
    shared_again = ob.scene_motion_labels(frames, "front_medium", truth, tracks)          # the second model
    for a, b, c in zip(plain, shared, shared_again):
        for key in ("motion", "speed_mps", "motion_bound_m"):
            assert np.array_equal(a[key], b[key]) and np.array_equal(a[key], c[key])
    assert sum((label["motion"] != ob.BACKGROUND).sum() for label in plain) > 0
    assert [frame.loads for frame in frames] == [2] * 5          # once for `plain`, once for both shared calls together


def test_medians_sit_next_to_the_means_and_resist_one_wild_scene():
    rows, scenes = confounded_run()
    wild = rows["scene_id"] == "driveB|1"
    rows.loc[wild, "abs_rel"] = rows.loc[wild, "abs_rel"] + 5.0
    table = ev.aggregate_by_scene_attribute(rows, scenes, "weather_group").set_index("weather_group")
    assert table.loc["wet", "abs_rel"] > 1.0 and table.loc["wet", "median"] < 0.2          # the mean is dragged, the median is not
    assert "median" in ev.aggregate(rows) and "median" in ev.cross_table(rows, scenes, "weather_group", "lighting")


def test_one_split_is_compared_with_the_others_condition_by_condition():
    rows, scenes = confounded_run()
    scenes["split"] = ["test", "train", "train", "test", "train", "train", "train", "test"]
    table = ev.compare_splits(rows, scenes, "test").set_index(["weather_group", "lighting"])
    assert table.loc[("dry", "day"), "n_scenes_test"] == 1 and table.loc[("dry", "day"), "n_scenes_others"] == 2
    assert np.isclose(table.loc[("dry", "day"), "difference"], 0.0, atol=1e-6)              # same planted error in both
    assert table.loc[("wet", "night"), "n_scenes_test"] == 1 and table.loc[("wet", "night"), "n_scenes_others"] == 3
    assert np.isnan(table.loc[("wet", "day"), "abs_rel_others"])                            # a condition only the test split has
    assert ev.compare_splits(rows, scenes[scenes["split"] == "train"], "test").empty
