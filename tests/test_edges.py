"""Interior against boundary: the split must put a misregistration-like error at the boundary and nowhere else."""

import numpy as np

from vggt_aura import edges, objects as ob

H, W = 120, 200


def frame(shift_error_on_moving_outline=True):
    """A dense grid of LiDAR pixels at 20 m, with a moving object (left) and a parked one (right), both at 10 m.

    The prediction is perfect, except: the MOVING object's outline is predicted at the background's depth, which
    is what ground truth shifted sideways by a few pixels looks like.
    """
    v, u = (a.ravel() for a in np.meshgrid(np.arange(H), np.arange(W), indexing="ij"))
    gt = np.full(len(u), 20.0)
    owner = np.full(len(u), -1)
    motion = np.full(len(u), ob.BACKGROUND, dtype=np.uint8)
    category = np.full(len(u), "", dtype=object)
    for k, (u0, u1, label, name) in enumerate([(30, 80, ob.MOVING, "car"), (120, 170, ob.PARKED, "car")]):
        inside = (u >= u0) & (u < u1) & (v >= 40) & (v < 90)
        gt[inside], owner[inside], motion[inside], category[inside] = 10.0, k, label, name
    depth = gt.reshape(H, W).copy()
    if shift_error_on_moving_outline:
        outline = (owner == 0) & ((u < 33) | (u >= 77) | (v < 43) | (v >= 87))          # a 3-pixel rim
        depth[v[outline], u[outline]] = 20.0
    truth = {"u": u.astype(np.int16), "v": v.astype(np.int16), "depth_m": gt.astype(np.float32)}
    labels = {"motion": motion, "box_index": owner.astype(np.int32), "category": category}
    return depth[None] / 1.0, [truth], [labels]


def test_dilate_is_a_square_window_and_does_not_wrap_around():
    mask = np.zeros((9, 9), bool)
    mask[0, 0] = mask[4, 4] = True
    out = edges.dilate(mask, 2)
    assert out[2:7, 2:7].all() and out[:3, :3].all() and not out[8, 8] and not out[0, 8] and not out[8, 0]
    assert out.sum() == 25 + 9 - 1 and np.array_equal(edges.dilate(mask, 0), mask)      # the two squares share pixel (2, 2)


def test_boundary_is_a_rim_of_the_given_width():
    depth, truth, labels = frame()
    rim = edges.boundary_mask(truth[0]["u"], truth[0]["v"], labels[0]["box_index"], W, H, radius=6)
    owner = labels[0]["box_index"]
    assert not rim[owner == -1].any()                                  # background is never "boundary"
    for k in (0, 1):
        assert rim[owner == k].sum() == 50 * 50 - 38 * 38              # a 6-pixel rim around a 50 x 50 object


def test_an_error_on_the_outline_shows_at_the_boundary_and_not_in_the_interior():
    depth, truth, labels = frame()
    rows = edges.edge_rows("s|1", depth, truth, labels, radius=6)
    table = rows[(rows["category"] == "all") & (rows["depth_band"] == "all")].set_index(["motion", "zone"])["abs_rel"]
    assert table[("moving", "interior")] == 0 and table[("parked", "interior")] == 0 and table[("parked", "boundary")] == 0
    assert table[("moving", "boundary")] > 0.4 and table[("moving", "all")] > 0.1      # whole-object number looks like a model failure
    assert set(rows["category"]) == {"all", "car"} and "10-20 m" in set(rows["depth_band"])


def test_summary_and_paired_comparisons_over_scenes():
    depth, truth, labels = frame()
    rows = __import__("pandas").concat([edges.edge_rows(f"s|{k}", depth, truth, labels, radius=6) for k in range(6)])
    summary = edges.edge_summary(rows).set_index(["motion", "zone"])
    assert summary.loc[("moving", "boundary"), "n_scenes"] == 6 and summary.loc[("background", "all"), "abs_rel"] == 0
    compared = edges.edge_comparisons(rows).set_index("comparison")["difference"]
    assert compared["moving minus parked, whole objects (what the report showed)"] > 0.1
    assert compared["moving minus parked, INTERIOR only (misregistration cannot reach here)"] == 0
    assert compared["parked: boundary minus interior (cost of soft depth edges alone)"] == 0


def test_motion_labels_say_which_box_and_which_category():
    from types import SimpleNamespace
    box = SimpleNamespace(object_id="a", center=np.array([10.0, 0, 0]), size_lwh=np.array([4.0, 2, 2]),
                          rotation_xyzw=np.array([0.0, 0, 0, 1]), category="truck")
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    out = ob.label_points(np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 40.0]]), base_from_camera, np.eye(4), [box], {"a": np.zeros(3)})
    assert out["box_index"].tolist() == [0, -1] and out["category"].tolist() == ["truck", ""]
