"""Differential tests: the C++ core against the Python reference, on identical inputs.

Python is the reference. Each C++ function is fed exactly the arrays the Python
function gets, and the outputs are compared:

- whole-number and yes/no outputs (pixel indices, occlusion verdicts, chosen
  points) must be EXACTLY equal;
- real-number outputs must agree to floating-point tolerance. They are not
  always bit-identical, because NumPy's matrix product and a plain C++ loop may
  add the same three numbers in a different order.

Skipped when the module is not built. On Colab it is built at session start.
To build by hand:  python -m vggt_aura.cpp_build
"""

import numpy as np
import pytest

from vggt_aura import geometry as geo, ground_truth as gt
from vggt_aura.cpp_build import import_module

cpp = import_module()
pytestmark = pytest.mark.skipif(cpp is None, reason="C++ module not built; run: python -m vggt_aura.cpp_build")

W, H = 640, 400
K = (538.2, 564.73, 324.75, 196.27)          # the real front_medium intrinsics at the model's input size
TIGHT = dict(rtol=1e-12, atol=1e-12)


def rigid(rng):
    q = rng.normal(size=4)
    t = np.eye(4)
    t[:3, :3] = geo.quat_xyzw_to_matrix(q)
    t[:3, 3] = rng.uniform(-3, 3, 3)
    return t


def street(rng, n=60000):
    """LiDAR-like points in a source frame with X forward, Y left, Z up, plus the transform to the camera."""
    forward = rng.uniform(-10, 90, n)
    points = np.column_stack([forward, rng.uniform(-25, 25, n), rng.uniform(-1.7, 6.0, n)])
    ground = rng.random(n) < 0.5
    points[ground, 2] = -1.5 + rng.normal(0, 0.02, ground.sum())
    near_wall = rng.random(n) < 0.15                                  # a close surface, so there is real occlusion
    points[near_wall, 0] = 8.0 + rng.normal(0, 0.05, near_wall.sum())
    points[near_wall, 1] = rng.uniform(-4, -1, near_wall.sum())
    camera_from_source = np.eye(4)
    camera_from_source[:3, :3] = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]     # optical frame
    camera_from_source[:3, 3] = [0.19, 1.49, -1.68]
    return points, camera_from_source


# ------------------------------------------------------------------ rotations and rigid transforms

def test_quaternion_to_matrix():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4) * rng.uniform(0.1, 10)                 # un-normalised on purpose
        assert np.allclose(cpp.quat_xyzw_to_matrix(q), geo.quat_xyzw_to_matrix(q), **TIGHT)


def test_compose_and_invert():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = rigid(rng), rigid(rng)
        assert np.allclose(cpp.compose(a, b), a @ b, **TIGHT)
        assert np.allclose(cpp.invert_se3(a), geo.invert_se3(a), **TIGHT)
        assert np.allclose(cpp.compose(a, cpp.invert_se3(a)), np.eye(4), atol=1e-12)


def test_transform_points():
    rng = np.random.default_rng(2)
    t, points = rigid(rng), rng.uniform(-80, 80, (5000, 3))
    assert np.allclose(cpp.transform_points(t, points), points @ t[:3, :3].T + t[:3, 3], **TIGHT)


# ------------------------------------------------------------------ projection

def test_project_and_unproject():
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.uniform(-30, 30, 20000), rng.uniform(-5, 5, 20000), rng.uniform(-5, 90, 20000)])
    points[:3, 2] = [0.0, 1e-9, -2.0]                                  # at and behind the camera plane
    u_c, v_c, z_c = cpp.project_pinhole(points, *K)
    u_p, v_p, z_p = gt.project_pinhole(points, *K)
    assert np.array_equal(np.isnan(u_c), np.isnan(u_p)) and np.isnan(u_c[:3]).all()
    assert np.allclose(u_c, u_p, equal_nan=True, **TIGHT) and np.allclose(v_c, v_p, equal_nan=True, **TIGHT)
    assert np.array_equal(z_c, z_p)
    ok = ~np.isnan(u_p)
    assert np.allclose(cpp.unproject_pinhole(u_p[ok], v_p[ok], z_p[ok], *K),
                       gt.unproject_pinhole(u_p[ok], v_p[ok], z_p[ok], *K), **TIGHT)


def test_pixel_indices_including_the_edges():
    u = np.array([9.49, 9.5, -0.5, -0.51, 639.49, 639.5, np.nan, np.inf, 100.0])
    v = np.array([0.0, 0.0, 0.0, 0.0, 399.49, 399.5, 5.0, 5.0, -0.5000001])
    ui_c, vi_c, inside_c = cpp.pixel_indices(u, v, W, H)
    ui_p, vi_p, inside_p = gt.pixel_indices(u, v, W, H)
    assert np.array_equal(inside_c.astype(bool), inside_p)
    assert np.array_equal(ui_c, ui_p) and np.array_equal(vi_c, vi_p)


# ------------------------------------------------------------------ occlusion: must match EXACTLY

def candidates(seed, n=60000):
    rng = np.random.default_rng(seed)
    points, camera_from_source = street(rng, n)
    cam = points @ camera_from_source[:3, :3].T + camera_from_source[:3, 3]
    u, v, z = gt.project_pinhole(cam, *K)
    ui, vi, inside = gt.pixel_indices(u, v, W, H)
    keep = inside & (z >= 1.0)
    return ui[keep], vi[keep], z[keep]


@pytest.mark.parametrize("mode", ["none", "window", "two_sided"])
@pytest.mark.parametrize("radius, half_width", [(6, 0), (3, 1), (1, 0), (0, 0), (10, 2)])
def test_occlusion_verdicts_are_identical(mode, radius, half_width):
    ui, vi, z = candidates(seed=radius * 10 + half_width)
    params = gt.OcclusionParams(mode=mode, radius=radius, half_width=half_width)
    expected = gt.occlusion_reason(ui, vi, z, W, H, params)
    got = cpp.occlusion_reason(ui, vi, z, W, H, mode, radius, half_width, params.rel_tol, params.abs_tol)
    assert np.array_equal(got, expected)
    if mode != "none":
        assert (expected != gt.VISIBLE).sum() > 100                   # the test scene really contains occlusion


def test_occlusion_on_the_unit_test_scenes():
    # The three hand-built scenes that justified the rule: road, car with leaks, pole.
    import test_ground_truth as scenes
    for ui, vi, z in (scenes.road_scene(), scenes.car_scene()[:3]):
        for mode in ("window", "two_sided"):
            params = gt.OcclusionParams(mode=mode)
            assert np.array_equal(cpp.occlusion_reason(ui, vi, z, W, H, mode, 6, 0, 0.10, 0.5),
                                  gt.occlusion_reason(ui, vi, z, W, H, params))


def test_nearest_per_pixel_including_exact_ties():
    ui, vi, z = candidates(seed=7)
    assert np.array_equal(cpp.nearest_per_pixel(ui, vi, z, W, H), gt.nearest_per_pixel(ui, vi, z, W))
    tie_u, tie_v = np.array([5, 5, 5, 9]), np.array([7, 7, 7, 1])
    tie_z = np.array([4.0, 4.0, 3.0, 2.0])
    assert np.array_equal(cpp.nearest_per_pixel(tie_u, tie_v, tie_z, W, H), gt.nearest_per_pixel(tie_u, tie_v, tie_z, W))
    same = np.array([4.0, 4.0, 4.0, 2.0])                              # exact tie: the lower index must win in both
    assert np.array_equal(cpp.nearest_per_pixel(tie_u, tie_v, same, W, H), gt.nearest_per_pixel(tie_u, tie_v, same, W))


# ------------------------------------------------------------------ the whole LiDAR-to-image step

@pytest.mark.parametrize("seed", [11, 12, 13])
def test_full_pipeline_matches_the_reference(seed):
    rng = np.random.default_rng(seed)
    points, camera_from_source = street(rng, 120000)
    semantic = rng.choice([1, 7, 16, 25, 27, 31], len(points)).astype(np.uint16)
    drop = {27, 31}
    loaded = {"points_cam": points @ camera_from_source[:3, :3].T + camera_from_source[:3, 3],
              "semantic_id": semantic, "instance_id": np.zeros(len(points), np.uint16),
              "sensor_index": np.zeros(len(points), np.uint8)}
    params = gt.OcclusionParams()
    expected = gt.build_ground_truth(loaded, K, W, H, params, drop)
    got = cpp.lidar_to_image(points, camera_from_source, *K, W, H, params.mode, params.radius, params.half_width,
                             params.rel_tol, params.abs_tol, params.min_depth, keep=~np.isin(semantic, list(drop)))
    assert len(expected["depth_m"]) > 10000
    assert np.array_equal(got["candidate_index"], expected["candidate_index"])
    assert np.array_equal(got["reason"], expected["reason"])
    assert np.array_equal(got["u"], expected["u"]) and np.array_equal(got["v"], expected["v"])
    assert np.allclose(got["depth_m"], expected["depth_m"], rtol=1e-6)            # the reference stores float32
    assert np.allclose(got["depth_m"], loaded["points_cam"][got["chosen"], 2], **TIGHT)


def test_bad_input_is_rejected_not_crashed():
    with pytest.raises(ValueError):
        cpp.transform_points(np.eye(3), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        cpp.project_pinhole(np.zeros((4, 2)), *K)
    with pytest.raises(ValueError):
        cpp.occlusion_reason(np.array([700]), np.array([0]), np.array([5.0]), W, H)   # outside the image
    with pytest.raises(ValueError):
        cpp.occlusion_reason(np.array([1]), np.array([1]), np.array([5.0]), W, H, "magic")
    with pytest.raises(ValueError):
        cpp.quat_xyzw_to_matrix(np.zeros(4))


# ------------------------------------------------------------------ the switch the pipeline actually uses

@pytest.mark.parametrize("seed", [21, 22])
@pytest.mark.parametrize("mode", ["none", "window", "two_sided"])
def test_build_ground_truth_gives_the_same_result_on_either_backend(seed, mode):
    rng = np.random.default_rng(seed)
    points, camera_from_source = street(rng, 150000)
    loaded = {"points_cam": points @ camera_from_source[:3, :3].T + camera_from_source[:3, 3],
              "semantic_id": rng.choice([0, 1, 7, 16, 25, 27, 31], len(points)).astype(np.uint16),
              "instance_id": rng.integers(0, 50, len(points)).astype(np.uint16),
              "sensor_index": rng.integers(0, 6, len(points)).astype(np.uint8)}
    params = gt.OcclusionParams(mode=mode)
    a = gt.build_ground_truth(loaded, K, W, H, params, {27, 31}, backend="python")
    b = gt.build_ground_truth(loaded, K, W, H, params, {27, 31}, backend="cpp")
    assert a["backend"] == "python" and b["backend"] == "cpp"
    assert a["counts"] == b["counts"] and a["counts"]["pixels_with_ground_truth"] > 10000
    for key in ("u", "v", "depth_m", "xyz_cam", "semantic_id", "instance_id", "sensor_index", "candidate_index", "visible", "reason"):
        assert np.array_equal(a[key], b[key]), key                      # EXACT, including the float32 depth
        assert a[key].dtype == b[key].dtype, key


def test_auto_picks_cpp_when_it_is_built_and_python_can_always_be_forced():
    assert gt.resolve_backend("auto") == "cpp" and gt.resolve_backend("python") == "python"
    with pytest.raises(ValueError):
        gt.resolve_backend("fortran")


def test_fast_path_and_general_path_agree_with_the_reference():
    ui, vi, z = candidates(seed=5)
    for half_width in (0, 1):                                            # 0 = fast path, 1 = general path
        params = gt.OcclusionParams(half_width=half_width)
        assert np.array_equal(cpp.occlusion_reason(ui, vi, z, W, H, "two_sided", 6, half_width, 0.10, 0.5),
                              gt.occlusion_reason(ui, vi, z, W, H, params))
