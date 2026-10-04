"""Which ground-truth pixels lie on moving objects, from the 3D boxes.

A pixel's LiDAR point is tested against every annotated 3D box of its frame
(both in base_link, at the same timestamp). A box's speed comes from its own
track: the same object_id seen at neighbouring keyframes, positions compared
in the odom frame so that the ego vehicle's motion cancels out.

Labels:
  BACKGROUND  in no box
  PARKED      in a box moving slower than PARKED_SPEED_MPS
  MOVING      in a box moving faster than MOVING_SPEED_MPS
  UNCERTAIN   in a box with a speed in between, or with no usable track

The gap between the two thresholds is deliberate. Box centres jitter by some
centimetres between annotations 0.5 s apart, which reads as 0.1 to 0.3 m/s of
fake speed; objects in the gap are kept out of BOTH strata.

THE MOTION BOUND ("trap 3" in the project's notes and in the report table `intrinsics_scale_and_trap3`),
stated as a number per pixel. Motion compensation corrects the ego
vehicle's motion only. A moving object is sampled at different instants across
the roughly 0.1 s sweep and is NOT moved to the reference time. Its LiDAR depth
can therefore be off by up to |object velocity along the camera's Z axis| x
SWEEP_S. `motion_bound_m` carries that bound, so the moving-object result can
be read next to the uncertainty of its own ground truth. The bound covers motion ALONG the viewing direction
only; an object crossing the view is displaced sideways in the image instead, which edges.py deals with.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .geometry import invert_se3, points_in_box

BACKGROUND, PARKED, MOVING, UNCERTAIN = 0, 1, 2, 3
MOTION_NAMES = {BACKGROUND: "background", PARKED: "parked object", MOVING: "moving object", UNCERTAIN: "uncertain object"}
MOVING_SPEED_MPS = 1.0
PARKED_SPEED_MPS = 0.5
MAX_TRACK_GAP_S = 2.0      # neighbours further apart than this say little about the speed now
BOX_MARGIN_M = 0.10
SWEEP_S = 0.1


def track_velocities(observations) -> dict:
    """Velocity in odom for every (frame_position, object_id).

    `observations`: iterable of (frame_position, timestamp_ns, object_id, centre_in_odom).
    Central difference where both neighbours exist, one-sided at a track's ends.
    The value is None when no neighbour lies within MAX_TRACK_GAP_S.
    """
    tracks = defaultdict(list)
    for position, timestamp_ns, object_id, centre in observations:
        tracks[object_id].append((int(timestamp_ns), int(position), np.asarray(centre, dtype=np.float64)))
    velocities = {}
    for object_id, track in tracks.items():
        track.sort(key=lambda item: item[0])
        for m, (t_now, position, _) in enumerate(track):
            before = track[m - 1] if m > 0 and (t_now - track[m - 1][0]) / 1e9 <= MAX_TRACK_GAP_S else None
            after = track[m + 1] if m + 1 < len(track) and (track[m + 1][0] - t_now) / 1e9 <= MAX_TRACK_GAP_S else None
            first, last = before or track[m], after or track[m]
            seconds = (last[0] - first[0]) / 1e9
            velocities[(position, object_id)] = (last[2] - first[2]) / seconds if seconds > 0 else None
    return velocities


def motion_label(velocity) -> int:
    if velocity is None:
        return UNCERTAIN
    speed = float(np.linalg.norm(velocity))
    if speed >= MOVING_SPEED_MPS:
        return MOVING
    return PARKED if speed < PARKED_SPEED_MPS else UNCERTAIN


def label_points(xyz_cam, base_from_camera, odom_from_base, boxes, velocities: dict) -> dict:
    """Motion label, speed and depth-error bound for each ground-truth point of one frame.

    `boxes`: objects with .center, .size_lwh, .rotation_xyzw, .object_id, in base_link.
    `velocities`: object_id -> velocity in odom (or None), for THIS frame.
    """
    xyz_cam = np.asarray(xyz_cam, dtype=np.float64).reshape(-1, 3)
    n = len(xyz_cam)
    motion = np.full(n, BACKGROUND, dtype=np.uint8)
    speed = np.zeros(n, dtype=np.float32)
    bound = np.zeros(n, dtype=np.float32)
    box_index = np.full(n, -1, dtype=np.int32)          # position of the box in `boxes`, -1 = background
    category = np.full(n, "", dtype=object)             # that box's category, as the dataset names it
    if n == 0 or not boxes:
        return {"motion": motion, "speed_mps": speed, "motion_bound_m": bound, "box_index": box_index, "category": category}

    points_base = xyz_cam @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    camera_from_odom = invert_se3(odom_from_base @ base_from_camera)[:3, :3]
    free = np.ones(n, dtype=bool)
    for position, box in enumerate(boxes):
        inside = free & points_in_box(points_base, box.center, box.size_lwh, box.rotation_xyzw, BOX_MARGIN_M)
        if not inside.any():
            continue
        velocity = velocities.get(box.object_id)
        motion[inside] = motion_label(velocity)
        box_index[inside] = position
        category[inside] = str(getattr(box, "category", ""))
        if velocity is not None:
            speed[inside] = np.linalg.norm(velocity)
            bound[inside] = abs(float((camera_from_odom @ velocity)[2])) * SWEEP_S
        free &= ~inside
    return {"motion": motion, "speed_mps": speed, "motion_bound_m": bound, "box_index": box_index, "category": category}


def scene_tracks(frames) -> dict:
    """Ego poses, boxes and object velocities of a scene. They do not depend on the model or its image size,
    so a scene scored for two models loads and tracks them once (pass the result to scene_motion_labels)."""
    poses = [frame.load_ego_pose() for frame in frames]
    boxes_per_frame = [list(frame.load_boxes(frame="base_link")) for frame in frames]
    observations = []
    for position, (frame, pose, boxes) in enumerate(zip(frames, poses, boxes_per_frame)):
        for box in boxes:
            centre_odom = pose[:3, :3] @ np.asarray(box.center, dtype=np.float64) + pose[:3, 3]
            observations.append((position, frame.timestamp_ns, box.object_id, centre_odom))
    return {"poses": poses, "boxes_per_frame": boxes_per_frame, "velocities": track_velocities(observations)}


def scene_motion_labels(frames, camera_id: str, truth_frames: list, tracks: dict | None = None) -> list:
    """AURA glue: label every ground-truth pixel of every frame of a scene."""
    base_from_camera = frames[0].calibration().base_from_sensor(f"camera/{camera_id}")
    tracks = scene_tracks(frames) if tracks is None else tracks
    poses, boxes_per_frame, velocities = tracks["poses"], tracks["boxes_per_frame"], tracks["velocities"]
    labels = []
    for position, (pose, boxes, truth) in enumerate(zip(poses, boxes_per_frame, truth_frames)):
        per_frame = {box.object_id: velocities.get((position, box.object_id)) for box in boxes}
        labels.append(label_points(truth["xyz_cam"], base_from_camera, pose, boxes, per_frame))
    return labels
