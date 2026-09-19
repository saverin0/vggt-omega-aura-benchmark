import numpy as np
import pytest

from vggt_aura import inference as inf, pins


def test_each_checkpoint_is_tied_to_its_own_resolution():
    assert inf.resolution_for(pins.CHECKPOINT_PUBLIC_512) == 512
    assert inf.resolution_for(pins.CHECKPOINT_RETRAINED_416) == 416
    with pytest.raises(ValueError):
        inf.resolution_for("something_else.pt")


def test_prediction_paths_separate_checkpoints_scenes_and_cameras(tmp_path):
    npz, meta = inf.prediction_paths(tmp_path, "2025-06-13-07-09-37_78", "front_medium", "vggt_omega_1b_512.pt")
    assert npz == tmp_path / "predictions" / "vggt_omega_1b_512" / "2025-06-13-07-09-37_78" / "front_medium.npz"
    assert meta.name == "front_medium.json"


def fake_arrays(frames=3, height=4, width=6):
    rng = np.random.default_rng(0)
    return {
        "depth": rng.uniform(0.2, 5.0, (frames, height, width)),
        "depth_conf": rng.uniform(1.0, 30.0, (frames, height, width)),
        "extrinsics": rng.normal(size=(frames, 3, 4)),
        "intrinsics": rng.normal(size=(frames, 3, 3)),
        "pose_enc": rng.normal(size=(frames, 9)),
        "input_hw": np.array([height, width]),
        "gt_camera0_from_camera": rng.normal(size=(frames, 4, 4)),
        "timestamps_ns": np.array([1752585602300000000 + i * 500_000_000 for i in range(frames)], dtype=np.int64),
        "seconds": 1.23,          # not an array key: must be ignored by the npz writer
    }


def test_predictions_round_trip(tmp_path):
    arrays = fake_arrays()
    npz, meta = inf.prediction_paths(tmp_path, "scene", "front_medium", "vggt_omega_1b_512.pt")
    size_mb = inf.save_predictions(npz, meta, arrays, {"scene_id": "a|1", "frames": 3})
    assert size_mb > 0
    loaded, info = inf.load_predictions(npz, meta)
    assert info == {"scene_id": "a|1", "frames": 3}
    assert set(loaded) == set(inf.ARRAY_KEYS)
    assert loaded["depth"].dtype == np.float32
    assert np.allclose(loaded["depth"], arrays["depth"], rtol=1e-6)            # float32 keeps depth
    assert np.allclose(loaded["depth_conf"], arrays["depth_conf"], rtol=2e-3)  # float16 is enough for confidence
    assert loaded["timestamps_ns"].dtype == np.int64
    assert loaded["timestamps_ns"][1] - loaded["timestamps_ns"][0] == 500_000_000   # no float rounding of ns


def test_no_temporary_file_is_left_behind(tmp_path):
    npz, meta = inf.prediction_paths(tmp_path, "scene", "front_medium", "vggt_omega_1b_512.pt")
    inf.save_predictions(npz, meta, fake_arrays(), {})
    assert sorted(p.name for p in npz.parent.iterdir()) == ["front_medium.json", "front_medium.npz"]


class FakeCalibration:
    def base_from_sensor(self, key):
        assert key == "camera/front_medium"
        matrix = np.eye(4)
        matrix[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]   # optical frame mounted facing forward
        matrix[:3, 3] = [1.7, 0.2, 1.5]
        return matrix


class FakeFrame:
    def __init__(self, x):
        self.x = x

    def calibration(self):
        return FakeCalibration()

    def load_ego_pose(self):
        matrix = np.eye(4)
        matrix[0, 3] = self.x
        return matrix


def test_ground_truth_cameras_are_in_metres_relative_to_the_first_frame():
    relative = inf.ground_truth_cameras([FakeFrame(100.0), FakeFrame(104.0), FakeFrame(110.0)], "front_medium")
    assert np.allclose(relative[0], np.eye(4), atol=1e-12)
    assert np.allclose(relative[:, :3, 3], [[0, 0, 0], [0, 0, 4.0], [0, 0, 10.0]], atol=1e-12)   # forward = camera +Z


def test_the_comparison_arm_is_pinned_and_kept_apart():
    assert len(pins.VGGT_COMMIT) == 40 and len(pins.VGGT_HF_REVISION) == 40
    a = inf.prediction_paths("root", "scene", "front_medium", pins.CHECKPOINT_PUBLIC_512)[0]
    b = inf.prediction_paths("root", "scene", "front_medium", inf.VGGT_NAME)[0]
    assert a != b and b.parent.parent.name == "vggt_1b"


def test_uncompressed_and_compressed_predictions_hold_identical_numbers(tmp_path):
    rng = np.random.default_rng(3)
    arrays = {"depth": np.exp(rng.normal(1, 0.5, (4, 20, 32))).astype(np.float32),
              "depth_conf": (1 + np.exp(rng.normal(0, 1, (4, 20, 32)))).astype(np.float32),
              "extrinsics": rng.normal(size=(4, 3, 4)), "intrinsics": rng.normal(size=(4, 3, 3)),
              "pose_enc": rng.normal(size=(4, 9)), "input_hw": np.array([20, 32]),
              "gt_camera0_from_camera": rng.normal(size=(4, 4, 4)), "timestamps_ns": np.arange(4, dtype=np.int64),
              "seconds": 1.0, "image_seconds": 0.5, "save_seconds": 0.1}
    loaded = {}
    for compress in (False, True):
        npz, sidecar = tmp_path / f"{compress}" / "front_medium.npz", tmp_path / f"{compress}" / "front_medium.json"
        inf.save_predictions(npz, sidecar, arrays, {"scene_id": "s|1"}, compress=compress)
        loaded[compress] = inf.load_predictions(npz, sidecar)[0]
        assert not list(npz.parent.glob("*.tmp*"))
    assert set(loaded[False]) == set(loaded[True]) == set(inf.ARRAY_KEYS)             # the timers are never stored
    for key in inf.ARRAY_KEYS:
        assert np.array_equal(loaded[False][key], loaded[True][key]), key
        assert loaded[False][key].dtype == loaded[True][key].dtype, key
    assert np.array_equal(loaded[False]["depth"], arrays["depth"])                       # depth is stored exactly
