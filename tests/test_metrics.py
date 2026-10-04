import numpy as np
import pytest

from vggt_aura import geometry as g, metrics as mt


def rot_y(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def poses_from_centres(centres, rotations=None):
    out = np.tile(np.eye(4), (len(centres), 1, 1))
    out[:, :3, 3] = centres
    if rotations is not None:
        out[:, :3, :3] = rotations
    return out


def forward_path(n=10, step=4.0):
    """Camera 0 frame: Z forward. A gentle right-hand bend so the path is not a straight line."""
    z = np.arange(n) * step
    return np.column_stack([0.02 * z ** 1.5, np.zeros(n), z])


# ------------------------------------------------------------------ quaternions, boxes, alignment

def test_quaternion_identity_and_a_quarter_turn_about_z():
    assert np.allclose(g.quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3))
    s = np.sqrt(0.5)
    assert np.allclose(g.quat_xyzw_to_matrix([0, 0, s, s]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)   # x goes to y: CCW about +Z


def test_quaternion_order_matters_and_we_use_xyzw():
    s = np.sqrt(0.5)
    as_xyzw = g.quat_xyzw_to_matrix([0, 0, s, s])
    as_if_wxyz = g.quat_xyzw_to_matrix([s, 0, 0, s])        # the same numbers read in the other order
    assert not np.allclose(as_xyzw, as_if_wxyz)


def test_quaternion_matches_the_sdk():
    sdk = pytest.importorskip("fzi_aura.geometry")
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert np.allclose(sdk.transform_matrix([0, 0, 0], q)[:3, :3], g.quat_xyzw_to_matrix(q), atol=1e-12)


def test_real_aura_box_quaternion_is_a_yaw():
    # rotation_xyzw of the first box read in phase 2: almost pure rotation about Z, about 90.7 degrees.
    rotation = g.quat_xyzw_to_matrix([0.0104627, -0.0104856, 0.71135477, 0.70267701])
    assert abs(g.rotation_angle_deg(rotation) - 90.7) < 0.5
    assert rotation[2, 2] > 0.999                          # Z stays Z: the box is upright


def test_points_in_a_rotated_box():
    s = np.sqrt(0.5)                                        # box turned 90 degrees: its LENGTH now lies along Y
    points = np.array([[10.0, 2.0, 0.0], [12.0, 0.0, 0.0], [10.0, 0.0, 0.9], [10.0, 0.0, 1.1]])
    inside = g.points_in_box(points, [10, 0, 0], [4.6, 1.8, 2.0], [0, 0, s, s])
    assert inside.tolist() == [True, False, True, False]
    assert g.points_in_box(points[3:], [10, 0, 0], [4.6, 1.8, 2.0], [0, 0, s, s], margin=0.15).tolist() == [True]


def test_umeyama_recovers_a_known_similarity_and_never_reflects():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    rotation = rot_y(25.0) @ g.quat_xyzw_to_matrix([0.1, 0.2, 0.3, 0.9])
    target = 7.5 * source @ rotation.T + [1.0, -2.0, 3.0]
    scale, found, translation = g.umeyama_similarity(source, target)
    assert np.isclose(scale, 7.5) and np.allclose(found, rotation, atol=1e-9) and np.allclose(translation, [1, -2, 3])
    mirrored = source * [1, 1, -1]
    assert np.isclose(np.linalg.det(g.umeyama_similarity(source, mirrored)[1]), 1.0)


def test_angles():
    assert np.isclose(g.rotation_angle_deg(rot_y(33.0)), 33.0)
    assert np.allclose(g.angle_between_deg([[1, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 1, 0], [-1, 0, 0], [1, 0, 0]])[:2], [90, 180])
    assert np.isnan(g.angle_between_deg([[0, 0, 0]], [[1, 0, 0]])[0])


# ------------------------------------------------------------------ depth alignment

GT = np.array([5.0, 10.0, 20.0, 40.0, 8.0, 16.0, 30.0, 60.0, 12.0, 25.0, 50.0, 70.0] * 2)
FRAME = np.repeat([0, 1], 12)


def abs_rel(pred, protocol, **kw):
    return mt.depth_summary(mt.align_depth(pred, GT, FRAME, protocol, **kw), GT)["abs_rel"]


def test_a_pure_scale_error_vanishes_under_sequence_scale():
    assert abs_rel(GT / 92.0, "sequence_scale") < 1e-12
    assert abs_rel(GT / 92.0, "unaligned") > 0.9            # raw model units against metres: meaningless, as stated


def test_scale_drift_between_frames_is_caught_by_sequence_scale_only():
    drifting = np.where(FRAME == 0, GT / 90.0, GT / 110.0)
    assert abs_rel(drifting, "frame_scale") < 1e-12          # forgiven
    assert abs_rel(drifting, "sequence_scale") > 0.05        # not forgiven: this is why it is the primary protocol


def test_scale_and_shift_is_the_most_forgiving():
    shifted = (GT - 2.0) / 50.0
    assert abs_rel(shifted, "frame_scale_shift") < 1e-6      # not exactly 0: the authors add 1e-8 inside the formula
    assert abs_rel(shifted, "frame_scale") > 0.02


def test_scale_shift_matches_the_authors_formula():
    rng = np.random.default_rng(2)
    pred, gt = rng.uniform(0.05, 1.0, 500), rng.uniform(2.0, 70.0, 500)
    gt_c, pred_c = gt - np.median(gt) + 1e-8, pred - np.median(pred) + 1e-8      # copied from vggt-omega eval/metrics.py
    scale = np.median(gt_c / pred_c)
    shift = np.median(gt - scale * pred)
    assert mt.median_scale_shift(pred, gt) == (float(scale), float(shift))


def test_pose_scale_is_applied_as_given():
    assert abs_rel(GT / 92.0, "pose_scale", pose_scale=92.0) < 1e-12
    assert np.isclose(abs_rel(GT / 92.0, "pose_scale", pose_scale=101.2), 0.1)


def test_frames_with_too_few_pixels_are_left_unaligned():
    frame = np.array([0] * 12 + [1] * 3)
    aligned = mt.align_depth(GT[:15] / 50.0, GT[:15], frame, "frame_scale")
    assert np.isfinite(aligned[:12]).all() and np.isnan(aligned[12:]).all()


def test_depth_summary_known_values():
    ten_percent = mt.depth_summary(GT * 1.10, GT)
    assert np.isclose(ten_percent["abs_rel"], 0.10) and ten_percent["delta125"] == 1.0
    assert mt.depth_summary(GT * 1.30, GT)["delta125"] == 0.0
    assert np.isclose(mt.depth_summary(GT + 3.0, GT)["rmse_m"], 3.0)
    assert mt.depth_summary(np.full(3, np.nan), GT[:3])["n_pixels"] == 0


def test_depth_bands():
    assert mt.depth_band_labels(np.array([1.0, 9.99, 10.0, 39.0, 40.0, 80.0])).tolist() == \
        ["1-10 m", "1-10 m", "10-20 m", "20-40 m", "40-80 m", "40-80 m"]


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError):
        mt.align_depth(GT, GT, FRAME, "best_guess")


# ------------------------------------------------------------------ pose

def test_perfect_poses_score_perfectly():
    gt = poses_from_centres(forward_path())
    out = mt.pose_summary(gt, gt)
    assert out["rotation_deg_median"] < 1e-6 and out["translation_deg_median"] < 1e-6
    assert out["auc30"] > 96.0 and out["ate_scale_only_m"] < 1e-9 and np.isclose(out["metres_per_model_unit"], 1.0)


def test_pose_metrics_ignore_the_unknown_scale():
    gt = poses_from_centres(forward_path())
    pred = poses_from_centres(forward_path() / 92.0)
    out = mt.pose_summary(pred, gt)
    assert out["translation_deg_median"] < 1e-5 and out["ate_scale_only_m"] < 1e-9
    assert np.isclose(out["metres_per_model_unit"], 92.0)


def test_sim3_alignment_hides_a_heading_error_that_scale_only_reveals():
    gt = poses_from_centres(forward_path())
    pred = poses_from_centres(forward_path() @ rot_y(10.0).T)            # the whole path swung 10 degrees to one side
    out = mt.pose_summary(pred, gt)
    assert out["ate_sim3_m"] < 1e-6                                      # similarity alignment rotates it back: looks perfect
    assert out["ate_scale_only_m"] > 2.0                                 # the strict alignment shows the error


def test_driving_backwards_is_180_degrees_wrong_but_the_authors_variant_cannot_see_it():
    gt = poses_from_centres(forward_path())
    pred = poses_from_centres(forward_path() * [-1, 1, -1])
    pairs = mt.pairwise_pose_errors(pred, gt)
    assert np.nanmedian(pairs["translation_deg"]) > 170.0
    assert np.nanmedian(pairs["translation_deg_unsigned"]) < 10.0


def test_rotation_error_is_measured():
    gt = poses_from_centres(forward_path(), np.tile(np.eye(3), (10, 1, 1)))
    rotations = np.stack([rot_y(0.5 * k) for k in range(10)])             # heading drifts half a degree per frame
    pairs = mt.pairwise_pose_errors(poses_from_centres(forward_path(), rotations), gt)
    first_to_last = (pairs["i"] == 0) & (pairs["j"] == 9)
    assert np.isclose(pairs["rotation_deg"][first_to_last][0], 4.5, atol=1e-6)


def test_a_parked_car_scene_has_no_translation_score():
    centres = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 0.3, 10)])   # 30 cm in 5 s
    out = mt.pose_summary(poses_from_centres(centres), poses_from_centres(centres))
    assert out["near_stationary"] and out["pairs_with_baseline"] == 0
    assert np.isnan(out["translation_deg_median"]) and np.isnan(out["auc30"])


def test_auc_by_hand():
    # errors 0.5, 1.5, 2.5 degrees, threshold 3: cumulative accuracy per 1-degree bin is 1/3, 2/3, 3/3
    assert np.isclose(mt.auc(np.array([0.5, 1.5, 2.5]), np.zeros(3), 3), 100.0 * (1 / 3 + 2 / 3 + 1) / 3)
    assert np.isclose(mt.auc(np.array([0.5]), np.array([50.0]), 30), 0.0)      # the WORSE of the two errors counts


# ------------------------------------------------------------------ intrinsics and bootstrap

def test_focal_error_sign_matches_phase_3():
    intrinsics = np.tile(np.array([[497.0, 0, 320], [0, 498.0, 200], [0, 0, 1]]), (5, 1, 1))
    out = mt.intrinsics_summary(intrinsics, gt_fx=538.2, gt_fy=564.7)
    assert np.isclose(out["fx_rel_err_median"], 497.0 / 538.2 - 1) and out["fx_rel_err_median"] < 0   # too short = negative
    assert out["fy_rel_err_median"] < out["fx_rel_err_median"]


def test_bootstrap_is_deterministic_and_sane():
    values = np.array([0.08, 0.10, 0.12, 0.09, 0.11, np.nan])
    first, again = mt.bootstrap_mean_ci(values), mt.bootstrap_mean_ci(values)
    assert first == again and np.isclose(first[0], 0.10) and first[1] < 0.10 < first[2]
    assert np.allclose(mt.bootstrap_mean_ci([0.2, 0.2, 0.2]), 0.2)
    assert np.isnan(mt.bootstrap_mean_ci([0.3])[1]) and np.isnan(mt.bootstrap_mean_ci([])[0])
