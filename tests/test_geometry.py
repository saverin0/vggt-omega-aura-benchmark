import numpy as np
import pytest

from vggt_aura import geometry as g


def rotation_z(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def pose(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def test_inverse_undoes_the_transform():
    transform = pose(rotation_z(37.0), [1.0, -2.0, 3.0])
    assert np.allclose(g.invert_se3(transform) @ transform, np.eye(4), atol=1e-12)
    assert np.allclose(transform @ g.invert_se3(transform), np.eye(4), atol=1e-12)


def test_inverse_matches_numpy_on_a_stack():
    stack = np.stack([pose(rotation_z(d), [d, 2 * d, -d]) for d in (0.0, 10.0, 123.0)])
    assert np.allclose(g.invert_se3(stack), np.linalg.inv(stack), atol=1e-10)


def test_3x4_input_is_accepted():
    transform = pose(rotation_z(20.0), [1.0, 2.0, 3.0])
    assert np.allclose(g.invert_se3(transform[:3]), np.linalg.inv(transform), atol=1e-12)
    with pytest.raises(ValueError):
        g.to_4x4(np.eye(3))


def test_camera_center_is_not_the_translation_column():
    # The trap this project must not fall into: for camera_from_world [R|t],
    # the camera sits at -R.T @ t, not at t.
    world_from_camera = pose(rotation_z(90.0), [5.0, 0.0, 0.0])      # camera placed at x = 5
    camera_from_world = g.invert_se3(world_from_camera)
    assert np.allclose(g.camera_centers(camera_from_world), [5.0, 0.0, 0.0], atol=1e-12)
    assert not np.allclose(camera_from_world[:3, 3], [5.0, 0.0, 0.0])


def test_identity_extrinsic_puts_the_camera_at_the_origin():
    assert np.allclose(g.camera_centers(np.eye(4)[None]), [[0.0, 0.0, 0.0]])


def test_relative_to_first_starts_at_identity_and_keeps_distances():
    world_from_camera = np.stack([pose(rotation_z(30.0), [10.0, 5.0, 0.0]),
                                  pose(rotation_z(30.0), [10.0, 9.0, 0.0])])
    relative = g.relative_to_first(world_from_camera)
    assert np.allclose(relative[0], np.eye(4), atol=1e-12)
    assert np.isclose(np.linalg.norm(relative[1][:3, 3]), 4.0)       # rigid motion preserves length


def test_forward_motion_shows_up_on_the_camera_z_axis():
    # base_link: X forward, Y left, Z up. Optical camera: X right, Y down, Z forward.
    # Columns of base_from_camera are the camera axes written in base_link.
    base_from_camera = pose(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), [1.5, 0.0, 1.5])
    odom_from_base = [pose(np.eye(3), [0.0, 0.0, 0.0]), pose(np.eye(3), [7.0, 0.0, 0.0])]   # drive 7 m forward
    relative = g.relative_to_first(np.stack([m @ base_from_camera for m in odom_from_base]))
    assert np.allclose(relative[1][:3, 3], [0.0, 0.0, 7.0], atol=1e-12)


def test_scale_intrinsics_for_the_real_front_medium_camera():
    fx, fy, cx, cy = g.scale_intrinsics(1614.61267, 1694.19946, 975.26354, 589.81466, (1920, 1200), (640, 400))
    assert np.isclose(fx, 1614.61267 / 3) and np.isclose(fy, 1694.19946 / 3)
    assert np.isclose(cx, (975.26354 + 0.5) / 3 - 0.5)
    assert np.isclose(cy, (589.81466 + 0.5) / 3 - 0.5)


def test_scale_intrinsics_handles_different_x_and_y_factors():
    fx, fy, cx, cy = g.scale_intrinsics(1000.0, 1000.0, 499.5, 249.5, (1000, 500), (500, 125))
    assert (fx, fy) == (500.0, 250.0)
    assert np.isclose(cx, 249.5) and np.isclose(cy, 62.0)            # the centre stays the centre


def test_fov():
    assert np.isclose(g.fov_degrees(focal_px=500.0, size_px=1000.0), 90.0)
