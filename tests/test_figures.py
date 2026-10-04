"""The pictures must show what the numbers say: same scale, same sign, same pixels as the evaluation."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vggt_aura import evaluation as ev, figures as fg, metrics as mt, objects as ob   # noqa: E402
import test_objects_evaluation as toe                                                  # noqa: E402


def scene(moving_error=1.25):
    arrays, truth, labels = toe.synthetic_scene(moving_error=moving_error)
    images = [np.full((toe.H, toe.W, 3), 90, dtype=np.uint8) for _ in range(toe.FRAMES)]
    return arrays, truth, labels, images


def test_scale_and_frame_table_agree_with_the_evaluation():
    arrays, truth, labels, _ = scene()
    scale = fg.sequence_scale(arrays["depth"], truth)
    rows, record = ev.evaluate_scene("s|1", arrays, truth, labels, toe.CLASS_NAMES, 538.2, 564.7)
    assert np.isclose(scale, record["sequence_scale_m_per_unit"]) and np.isclose(scale, 50.0)
    table = fg.frame_table(arrays["depth"], truth, labels, scale)
    overall = rows[(rows["protocol"] == mt.PRIMARY_PROTOCOL) & (rows["stratum_type"] == "all")]["abs_rel"].iloc[0]
    weighted = np.average(table["abs_rel"], weights=table["lidar_pixels"])
    assert np.isclose(weighted, overall, atol=1e-6)                              # the frames add up to the scene's number
    assert np.allclose(table["abs_rel_moving"], 0.25, atol=1e-6) and np.allclose(table["scale_ratio"], 1.0, atol=1e-6)
    assert len(table) == toe.FRAMES and (table["moving_share"] > 0).all()


def test_the_sign_of_the_error_is_prediction_minus_truth():
    arrays, truth, labels, _ = scene(moving_error=1.25)                          # moving objects predicted 25% too FAR
    e = fg.frame_errors(arrays["depth"][0] * 50.0, truth[0])
    moving = labels[0]["motion"][e["kept"]] == ob.MOVING
    assert np.allclose(e["rel"][moving], 0.25, atol=1e-6) and np.allclose(e["rel"][~moving], 0.0, atol=1e-6)
    truth[0]["depth_m"][:3] = [0.2, 500.0, 90.0]                                 # outside the evaluated range
    assert len(fg.frame_errors(arrays["depth"][0] * 50.0, truth[0])["gt"]) == len(truth[0]["depth_m"]) - 3


def test_a_backwards_trajectory_is_drawn_backwards():
    arrays, _, _, _ = scene()
    gt = arrays["gt_camera0_from_camera"]
    pred = mt.predicted_camera0_from_camera(arrays["extrinsics"])[:, :3, 3]
    assert np.isclose(fg.path_length_scale(pred, gt[:, :3, 3]), 50.0)
    assert fg.path_length_scale(-pred, gt[:, :3, 3]) > 0                         # never negative: it must not un-flip the path
    backwards = arrays["extrinsics"].copy()
    backwards[:, :3, 3] *= -1
    figure = fg.trajectory_figure(backwards, gt, "reversed")
    lines = figure.axes[0].get_lines()
    assert lines[0].get_ydata()[-1] > 0 > lines[1].get_ydata()[-1]               # truth drives forward, prediction the other way
    assert np.isclose(abs(lines[1].get_ydata()[-1]), lines[0].get_ydata()[-1])   # and over the same distance


def test_the_scene_figure_renders(tmp_path):
    arrays, truth, labels, images = scene()
    figure = fg.scene_figure(images, arrays["depth"], truth, labels, 50.0, [0, toe.FRAMES - 1], "synthetic scene")
    assert len(figure.axes) >= 8
    target = tmp_path / "scene.png"
    figure.savefig(target, dpi=60)
    assert target.stat().st_size > 5000
    no_labels = fg.scene_figure(images, arrays["depth"], truth, None, 50.0, [1], "without motion labels")
    assert len(no_labels.axes) >= 4
