import numpy as np
import pytest

from vggt_aura import ground_truth as gt
from vggt_aura.ground_truth import OcclusionParams

W, H = 640, 400
FX, FY, CX, CY = 179.0, 188.0, 320.0, 200.0


def params(mode, **kw):
    return OcclusionParams(mode=mode, **kw)


# ------------------------------------------------------------------ projection conventions

def test_point_on_the_optical_axis_lands_on_the_principal_point():
    u, v, z = gt.project_pinhole([[0.0, 0.0, 10.0]], FX, FY, CX, CY)
    assert (u[0], v[0], z[0]) == (CX, CY, 10.0)


def test_right_is_plus_u_and_down_is_plus_v():
    u, v, _ = gt.project_pinhole([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]], FX, FY, CX, CY)
    assert u[0] > CX and v[0] == CY        # camera +X (right) moves the pixel right
    assert v[1] > CY and u[1] == CX        # camera +Y (down) moves the pixel down


def test_depth_is_z_not_distance_along_the_ray():
    _, _, z = gt.project_pinhole([[6.0, 0.0, 8.0]], FX, FY, CX, CY)
    assert z[0] == 8.0                     # the range would be 10.0


def test_points_behind_the_camera_never_land_in_the_image():
    u, v, _ = gt.project_pinhole([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0]], FX, FY, CX, CY)
    _, _, inside = gt.pixel_indices(u, v, W, H)
    assert not inside.any()


def test_pixel_centres_sit_on_integer_coordinates():
    ui, vi, inside = gt.pixel_indices(np.array([9.49, 9.5, -0.5, -0.51, 639.49, 639.5]),
                                      np.zeros(6), W, H)
    assert ui[:2].tolist() == [9, 10]
    assert inside.tolist() == [True, True, True, False, True, False]


def test_projection_matches_the_sdk():
    sdk = pytest.importorskip("fzi_aura.geometry")
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(-20, 20, 500), rng.uniform(-3, 3, 500), rng.uniform(1, 80, 500)])
    P = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0]], dtype=float)
    uv, depth = sdk.project_points(points, P)
    u, v, z = gt.project_pinhole(points, FX, FY, CX, CY)
    assert np.allclose(u, uv[:, 0], atol=1e-9) and np.allclose(v, uv[:, 1], atol=1e-9)
    assert np.allclose(z, depth, atol=1e-12)


# ------------------------------------------------------------------ occlusion building blocks

def test_directional_minima_look_the_right_way():
    buffer = np.full((20, 20), np.inf)
    buffer[10, 10] = 3.0
    sides = gt.directional_minima(buffer, radius=4, half_width=1)     # width 1 only to test the strip logic
    assert sides["above"][12, 10] == 3.0 and sides["above"][8, 10] == np.inf    # row 12 has it ABOVE
    assert sides["below"][8, 10] == 3.0 and sides["below"][12, 10] == np.inf
    assert sides["left"][10, 12] == 3.0 and sides["right"][10, 8] == 3.0
    assert sides["above"][10, 10] == np.inf                                      # never the pixel itself
    assert sides["above"][12, 11] == 3.0 and sides["above"][12, 12] == np.inf    # strip is 3 wide
    assert sides["above"][15, 10] == np.inf                                      # beyond the radius


def test_same_pixel_keeps_only_the_near_point():
    ui, vi, z = np.array([50, 50]), np.array([60, 60]), np.array([5.0, 30.0])
    assert gt.visible_mask(ui, vi, z, W, H, params("two_sided")).tolist() == [True, False]
    assert gt.nearest_per_pixel(ui, vi, z, W).tolist() == [0]


def test_tolerance_keeps_points_of_similar_depth_in_one_pixel():
    ui, vi, z = np.array([50, 50]), np.array([60, 60]), np.array([20.0, 21.5])   # 1.5 m < 10% of 21.5
    assert gt.visible_mask(ui, vi, z, W, H, params("two_sided")).all()


# ------------------------------------------------------------------ the three scenes that justify the design

def road_scene(camera_height=1.5, row_step=1, column_step=1):
    """A flat road below the horizon: depth = height * fy / (v - cy). A slanted surface, no occlusion.

    DENSE by default: a point in every pixel. An earlier version of this test
    used every 2nd row and so missed a real bug (see the wide-strip test below).
    """
    rows = np.arange(int(CY) + 4, H, row_step)
    vi, ui = np.meshgrid(rows, np.arange(0, W, column_step), indexing="ij")
    vi, ui = vi.ravel(), ui.ravel()
    return ui, vi, camera_height * FY / (vi - CY)


@pytest.mark.parametrize("row_step, column_step", [(1, 1), (2, 3), (5, 1)])
def test_two_sided_rule_keeps_the_whole_road(row_step, column_step):
    ui, vi, z = road_scene(row_step=row_step, column_step=column_step)
    assert gt.visible_mask(ui, vi, z, W, H, params("two_sided")).all()


def test_wide_search_strips_would_delete_a_dense_road():
    # The bug found in rehearsal. A left/right strip 3 rows tall contains the row below,
    # where the road is nearer on BOTH sides, so the far road counts as occluded.
    ui, vi, z = road_scene()
    visible = gt.visible_mask(ui, vi, z, W, H, params("two_sided", half_width=1))
    assert (~visible[z > 20.0]).mean() > 0.3            # measured: 45% of the far road deleted
    assert OcclusionParams().half_width == 0            # so the default must stay 0


def test_road_with_camber_and_camera_roll_survives():
    # Real roads are not perfectly level across the image: depth drifts slowly along a pixel row.
    ui, vi, z = road_scene()
    z = z * (1.0 + 0.10 * (ui - CX) / W)                # 10% depth change across the full image width
    assert gt.visible_mask(ui, vi, z, W, H, params("two_sided")).mean() > 0.999


def test_window_rule_wrongly_deletes_the_far_road():
    # This failure is the reason the two-sided rule exists.
    ui, vi, z = road_scene()
    visible = gt.visible_mask(ui, vi, z, W, H, params("window"))
    far = z > 20.0
    assert far.sum() > 100
    assert (~visible[far]).mean() > 0.9


def car_scene():
    """A car at 8 m, sampled on every 3rd row like LiDAR rings, with background at 30 m leaking through."""
    top, bottom, left, right = 150, 250, 200, 300
    car_v, car_u = np.meshgrid(np.arange(top, bottom + 1, 3), np.arange(left, right + 1), indexing="ij")
    wall_v, wall_u = np.meshgrid(np.arange(100, 300), np.arange(150, 350, 2), indexing="ij")
    wall_v, wall_u = wall_v.ravel(), wall_u.ravel()
    on_car_row = ((wall_v - top) % 3 == 0) & (wall_v >= top) & (wall_v <= bottom)
    keep = ~(on_car_row & (wall_u >= left) & (wall_u <= right))     # do not share pixels with the car
    wall_v, wall_u = wall_v[keep], wall_u[keep]
    ui = np.concatenate([car_u.ravel(), wall_u])
    vi = np.concatenate([car_v.ravel(), wall_v])
    z = np.concatenate([np.full(car_u.size, 8.0), np.full(wall_u.size, 30.0)])
    is_car = np.arange(len(z)) < car_u.size
    deep_inside = (~is_car) & (vi > top + 3) & (vi < bottom - 3) & (ui > left + 3) & (ui < right - 3)
    well_outside = (~is_car) & ((vi < top - 8) | (vi > bottom + 8) | (ui < left - 8) | (ui > right + 8))
    return ui, vi, z, is_car, deep_inside, well_outside


def test_two_sided_rule_removes_background_leaking_into_a_car():
    ui, vi, z, is_car, deep_inside, well_outside = car_scene()
    visible = gt.visible_mask(ui, vi, z, W, H, params("two_sided"))
    assert deep_inside.sum() > 1000
    assert visible[is_car].all()                    # the car itself is never touched
    assert not visible[deep_inside].any()           # every leak inside the outline is removed
    assert visible[well_outside].all()              # background away from the car is kept


def test_no_occlusion_handling_keeps_the_leak():
    ui, vi, z, _, deep_inside, _ = car_scene()
    assert gt.visible_mask(ui, vi, z, W, H, params("none"))[deep_inside].all()


def test_thin_pole_in_front_of_a_wall():
    pole_v = np.arange(100, 300, 2)
    wall_v, wall_u = np.meshgrid(np.arange(101, 300, 2), np.arange(300, 341), indexing="ij")
    ui = np.concatenate([np.full(len(pole_v), 320), wall_u.ravel()])
    vi = np.concatenate([pole_v, wall_v.ravel()])
    z = np.concatenate([np.full(len(pole_v), 6.0), np.full(wall_u.size, 20.0)])
    visible = gt.visible_mask(ui, vi, z, W, H, params("two_sided"))
    wall = np.arange(len(z)) >= len(pole_v)
    interior = (vi > 110) & (vi < 290)
    assert visible[~wall].all()
    assert not visible[wall & (ui == 320) & interior].any()              # wall seen "through" the pole: removed
    assert visible[wall & (ui != 320)].all()                             # every other wall point stays


# ------------------------------------------------------------------ assembling one frame

def test_build_ground_truth_counts_add_up_and_classes_are_dropped():
    points = np.array([[0.0, 0.0, 10.0],     # kept
                       [0.0, 0.0, 40.0],     # same pixel, far: occluded
                       [1.0, 0.5, 12.0],     # kept
                       [0.5, 0.5, 9.0],      # class 27 "noise": dropped
                       [0.0, 0.0, 0.4],      # nearer than min_depth
                       [0.0, 0.0, -3.0],     # behind the camera
                       [900.0, 0.0, 10.0]])  # outside the image
    loaded = {"points_cam": points, "semantic_id": np.array([7, 16, 1, 27, 7, 7, 7], np.uint16),
              "instance_id": np.arange(7, dtype=np.uint16), "sensor_index": np.zeros(7, np.uint8)}
    truth = gt.build_ground_truth(loaded, (FX, FY, CX, CY), W, H, params("two_sided"), drop_class_ids={27, 31})
    counts = truth["counts"]
    assert counts["points_loaded"] == 7
    assert counts["in_image_in_front"] == 4
    assert counts["dropped_by_class"] == 1
    assert counts["candidates"] == 3
    assert counts["dropped_as_occluded"] == 0 and counts["dropped_same_pixel"] == 1
    assert counts["pixels_with_ground_truth"] == 2
    assert sorted(truth["depth_m"].tolist()) == [10.0, 12.0]
    assert sorted(truth["semantic_id"].tolist()) == [1, 7]
    assert truth["depth_m"].dtype == np.float32 and truth["u"].dtype == np.int16
    assert np.allclose(truth["xyz_cam"][:, 2], truth["depth_m"])          # the stored point's Z IS the depth


def test_excluded_classes_are_found_by_name():
    names = {0: "unlabeled", 1: "road", 27: "noise", 31: "ego"}
    assert gt.excluded_class_ids(names) == {27, 31}


def test_empty_input_is_handled():
    loaded = {"points_cam": np.zeros((0, 3)), "semantic_id": np.zeros(0, np.uint16),
              "instance_id": np.zeros(0, np.uint16), "sensor_index": np.zeros(0, np.uint8)}
    truth = gt.build_ground_truth(loaded, (FX, FY, CX, CY), W, H, params("two_sided"))
    assert truth["counts"]["pixels_with_ground_truth"] == 0 and len(truth["depth_m"]) == 0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        gt.visible_mask(np.array([1]), np.array([1]), np.array([5.0]), W, H, params("magic"))


def test_scene_truth_round_trip(tmp_path):
    def frame(n, depth):
        return {"u": np.arange(n, dtype=np.int16), "v": np.arange(n, dtype=np.int16) + 1,
                "depth_m": np.full(n, depth, np.float32), "xyz_cam": np.full((n, 3), depth, np.float32),
                "semantic_id": np.full(n, 7, np.uint16),
                "instance_id": np.zeros(n, np.uint16), "sensor_index": np.zeros(n, np.uint8),
                "counts": {"pixels_with_ground_truth": n}}
    frames = [frame(3, 10.0), frame(0, 0.0), frame(5, 22.5)]            # an empty frame in the middle
    stamps = [1752585602300000000, 1752585602800000000, 1752585603300000000]
    npz, meta = gt.truth_paths(tmp_path, "scene_a", "front_medium")
    assert gt.save_scene_truth(npz, meta, frames, stamps, {"occlusion": {"mode": "two_sided"}}) > 0
    loaded, loaded_stamps, info = gt.load_scene_truth(npz, meta)
    assert [len(f["depth_m"]) for f in loaded] == [3, 0, 5]
    assert loaded[2]["depth_m"].tolist() == [22.5] * 5 and loaded[2]["v"].tolist() == [1, 2, 3, 4, 5]
    assert loaded_stamps.tolist() == stamps
    assert loaded[0]["xyz_cam"].shape == (3, 3) and loaded[1]["xyz_cam"].shape == (0, 3)
    assert info["occlusion"] == {"mode": "two_sided"}
    assert [c["pixels_with_ground_truth"] for c in info["per_frame_counts"]] == [3, 0, 5]
    assert sorted(x.name for x in npz.parent.iterdir()) == ["front_medium.json", "front_medium.npz"]


def test_reasons_separate_same_pixel_from_neighbour_occlusion():
    ui, vi, z, is_car, deep_inside, _ = car_scene()
    reason = gt.occlusion_reason(ui, vi, z, W, H, params("two_sided"))
    assert (reason[deep_inside] == gt.HIDDEN_BY_NEIGHBOURS).all()
    assert (reason[is_car] == gt.VISIBLE).all()
    same = gt.occlusion_reason(np.array([5, 5]), np.array([5, 5]), np.array([4.0, 40.0]), W, H, params("two_sided"))
    assert same.tolist() == [gt.VISIBLE, gt.HIDDEN_SAME_PIXEL]


def test_removal_table_reports_both_kinds_of_drop():
    ui, vi, z, is_car, _, _ = car_scene()
    points = np.column_stack([(ui - CX) * z / FX, (vi - CY) * z / FY, z])
    loaded = {"points_cam": points, "semantic_id": np.where(is_car, 7, 16).astype(np.uint16),
              "instance_id": np.zeros(len(z), np.uint16), "sensor_index": np.zeros(len(z), np.uint8)}
    truth = gt.build_ground_truth(loaded, (FX, FY, CX, CY), W, H, params("two_sided"))
    by_class, by_band = gt.removal_table(loaded, truth, {7: "car", 16: "building"})
    assert by_class.loc["car", "occluded"] == 0
    assert by_class.loc["building", "occluded"] > 1000
    assert list(by_class.columns) == ["points", "occluded", "same_pixel"]
    assert by_band["points"].sum() == len(z)


def test_truth_paths_keep_input_sizes_apart(tmp_path):
    omega = gt.truth_paths(tmp_path, "scene", "front_medium")
    vggt = gt.truth_paths(tmp_path, "scene", "front_medium", tag="518x322")
    assert omega[0].name == "front_medium.npz"                      # unchanged, so notebook `05b_one_block_metrics` keeps working
    assert vggt[0].name == "front_medium_518x322.npz" and vggt[1].name == "front_medium_518x322.json"


def test_unproject_is_the_inverse_of_project():
    rng = np.random.default_rng(4)
    points = np.column_stack([rng.uniform(-20, 20, 300), rng.uniform(-3, 3, 300), rng.uniform(1, 80, 300)])
    u, v, z = gt.project_pinhole(points, FX, FY, CX, CY)
    assert np.allclose(gt.unproject_pinhole(u, v, z, FX, FY, CX, CY), points, atol=1e-9)
    centre = gt.unproject_pinhole([CX], [CY], [12.0], FX, FY, CX, CY)
    assert np.allclose(centre, [[0.0, 0.0, 12.0]])                  # the principal point looks straight ahead


# ------------------------------------------------------------------ build once, reuse for ever

def test_ouster_only_policy_drops_the_aeva_sensors():
    sensors = ["top_left", "aeva_front", "front_right", "Aeva_Rear_Left", "rear_left"]
    assert gt.select_lidars(sensors) == ["front_right", "rear_left", "top_left"]
    assert gt.select_lidars(sensors, "all") == sorted(sensors)
    with pytest.raises(ValueError):
        gt.select_lidars(sensors, "best")


def test_truth_tag_keeps_the_phase_5_file_name():
    assert gt.truth_tag(640, 400) is None and gt.truth_tag(518, 322) == "518x322"


class StampedFrame:
    def __init__(self, timestamp_ns):
        self.timestamp_ns = timestamp_ns


def fake_truth(n=4):
    return [{"u": np.arange(n, dtype=np.int16), "v": np.arange(n, dtype=np.int16), "depth_m": np.full(n, 9.0, np.float32),
             "xyz_cam": np.zeros((n, 3), np.float32), "semantic_id": np.ones(n, np.uint16),
             "instance_id": np.zeros(n, np.uint16), "sensor_index": np.zeros(n, np.uint8), "counts": {}}]


def test_ground_truth_is_built_once_then_loaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gt, "build_scene_truth", lambda *a, **k: calls.append(1) or fake_truth())
    frames, request = [StampedFrame(1000)], dict(persist_root=tmp_path, scene_id="s|1", scene_name="s_1", camera_id="front_medium",
                                                  intrinsics=(FX, FY, CX, CY), width=W, height=H, params=OcclusionParams())
    truth, source = gt.load_or_build_scene_truth(frames=frames, lidar_ids=["top_left"], **request)
    assert source == "built (python)" and len(calls) == 1
    truth, source = gt.load_or_build_scene_truth(frames=frames, lidar_ids=["top_left"], **request)
    assert source == "loaded" and len(calls) == 1 and truth[0]["depth_m"].tolist() == [9.0] * 4

    # anything that changes the answer forces a rebuild: the sensor set, the rule, the frames
    assert gt.load_or_build_scene_truth(frames=frames, lidar_ids=["top_left", "top_right"], **request)[1].startswith("built")
    other_rule = {**request, "params": OcclusionParams(mode="none")}
    assert gt.load_or_build_scene_truth(frames=frames, lidar_ids=["top_left", "top_right"], **other_rule)[1].startswith("built")
    assert gt.load_or_build_scene_truth(frames=[StampedFrame(2000)], lidar_ids=["top_left", "top_right"], **other_rule)[1].startswith("built")
    assert len(calls) == 4


def test_loading_needs_no_lidar_on_disk_but_building_does(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gt, "build_scene_truth", lambda *a, **k: calls.append(1) or fake_truth())
    request = dict(persist_root=tmp_path, scene_id="s|1", scene_name="s_1", camera_id="front_medium", frames=[StampedFrame(1000)],
                   lidar_ids=["top_left"], intrinsics=(FX, FY, CX, CY), width=W, height=H, params=OcclusionParams())
    with pytest.raises(RuntimeError):
        gt.load_or_build_scene_truth(can_build=False, **request)          # nothing saved, and no LiDAR on disk
    assert gt.load_or_build_scene_truth(can_build=True, **request)[1].startswith("built")
    assert gt.load_or_build_scene_truth(can_build=False, **request)[1] == "loaded"   # the whole point: no LiDAR needed now
    assert len(calls) == 1
    with pytest.raises(RuntimeError):                                     # another sensor set is NOT silently reused
        gt.load_or_build_scene_truth(can_build=False, **{**request, "lidar_ids": ["top_left", "top_right"]})


def test_default_backend_is_the_python_reference_and_unknown_ones_are_rejected():
    loaded = {"points_cam": np.array([[0.0, 0.0, 10.0]]), "semantic_id": np.array([7], np.uint16),
              "instance_id": np.zeros(1, np.uint16), "sensor_index": np.zeros(1, np.uint8)}
    assert gt.build_ground_truth(loaded, (FX, FY, CX, CY), W, H, params("two_sided"))["backend"] == "python"
    with pytest.raises(ValueError):
        gt.build_ground_truth(loaded, (FX, FY, CX, CY), W, H, params("two_sided"), backend="fortran")
