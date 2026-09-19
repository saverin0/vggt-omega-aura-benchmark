"""Run VGGT-Omega on frames of one AURA camera and persist what it predicts.

Facts read from facebookresearch/vggt-omega at the pinned commit:

- load_and_preprocess_images(paths, image_resolution=512), default mode
  "balanced", resizes WITHOUT cropping for aspect ratios in [0.5, 2.0] and
  keeps about 1024 patch tokens. A 1920x1200 image becomes 640x400, exactly a
  third in both directions. Pixel values are left in [0, 1].
- model(images) returns "depth" (1, S, H, W, 1), "depth_conf" (1, S, H, W) and
  "pose_enc" (1, S, 9). Depth is exp(.), so strictly positive, and is Z-depth
  in the camera frame. Confidence is 1 + exp(.), so always above 1.
- encoding_to_camera gives camera_from_world 3x4 extrinsics (OpenCV axes,
  world = first camera) and 3x3 intrinsics whose principal point is ALWAYS
  the image centre. Only the two focal lengths are predicted.
- The forward pass opens torch.autocast("cuda"). It needs a GPU.
- Depth and translation share one unknown scale per sequence (see
  docs/decisions.md). Nothing here is in metres.

torch and vggt_omega are imported inside functions so the rest of the package
and the local tests work without them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import pins
from .aura_data import iter_frames
from .geometry import relative_to_first

RESOLUTION_FOR_CHECKPOINT = {
    pins.CHECKPOINT_PUBLIC_512: 512,
    pins.CHECKPOINT_RETRAINED_416: 416,
}


def resolution_for(checkpoint_name: str) -> int:
    """The input resolution a checkpoint was trained for. Feeding another size runs but degrades silently."""
    try:
        return RESOLUTION_FOR_CHECKPOINT[checkpoint_name]
    except KeyError:
        raise ValueError(f"unknown checkpoint {checkpoint_name!r}; known: {sorted(RESOLUTION_FOR_CHECKPOINT)}") from None


def select_frames(scene, camera_id: str, max_frames=None, stride: int = 1) -> list:
    """Consecutive annotated keyframes of a scene that have an image from this camera."""
    frames = list(iter_frames(scene.frames(sample_filter="any_label", require_cameras=[camera_id])))
    frames = frames[::stride]
    return frames if max_frames is None else frames[:max_frames]


def ground_truth_cameras(frames, camera_id: str) -> np.ndarray:
    """camera0_from_camera_i, in METRES, from AURA ego poses and calibration.

    odom_from_camera_i = odom_from_base_i @ base_from_camera. The camera mount
    is rigid, so base_from_camera is the same for every frame of a scene.
    """
    base_from_camera = frames[0].calibration().base_from_sensor(f"camera/{camera_id}")
    odom_from_camera = np.stack([frame.load_ego_pose() @ base_from_camera for frame in frames])
    return relative_to_first(odom_from_camera)


def download_checkpoint(checkpoint_name: str, cache_dir) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(pins.VGGT_OMEGA_HF_REPO, checkpoint_name,
                           revision=pins.VGGT_OMEGA_HF_REVISION, local_dir=str(cache_dir))
    return Path(path)


def load_model(checkpoint_path):
    import torch
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    return model.to("cuda")


def prepare_images(image_paths, image_resolution: int):
    """Read and resize the camera images exactly as the model's authors do. CPU only, so it can run ahead
    in a background thread while the GPU is busy with the previous scene."""
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images([str(p) for p in image_paths], image_resolution=image_resolution)


def run_model(model, image_paths, image_resolution: int, images=None) -> dict:
    """One forward pass over all images together. Returns NumPy arrays plus timing and memory.

    `images`: the result of prepare_images, when it was made ahead of time. Same function, same tensor.
    """
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

    loading = time.time()
    images = (prepare_images(image_paths, image_resolution) if images is None else images).to("cuda")
    image_seconds = time.time() - loading
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    torch.cuda.synchronize()
    seconds = time.time() - started

    def numpy(tensor):
        return tensor.detach().float().cpu().numpy()[0]   # drop the batch dimension

    height, width = (int(v) for v in images.shape[-2:])
    return {
        "depth": numpy(predictions["depth"])[..., 0],          # (S, H, W), Z-depth, scale-normalised
        "depth_conf": numpy(predictions["depth_conf"]),        # (S, H, W), > 1
        "extrinsics": numpy(extrinsics),                       # (S, 3, 4), camera_from_world
        "intrinsics": numpy(intrinsics),                       # (S, 3, 3), principal point at centre
        "pose_enc": numpy(predictions["pose_enc"]),            # (S, 9), raw head output
        "input_hw": np.array([height, width]),
        "seconds": seconds,
        "image_seconds": image_seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


ARRAY_KEYS = ("depth", "depth_conf", "extrinsics", "intrinsics", "pose_enc", "input_hw",
              "gt_camera0_from_camera", "timestamps_ns")


def prediction_paths(persist_root, scene_name: str, camera_id: str, checkpoint_name: str) -> tuple[Path, Path]:
    folder = Path(persist_root) / "predictions" / Path(checkpoint_name).stem / scene_name
    return folder / f"{camera_id}.npz", folder / f"{camera_id}.json"


def save_predictions(npz_path, json_path, arrays: dict, meta: dict, compress: bool = False) -> float:
    """Write arrays and a readable sidecar. Returns the npz size in MB.

    Depth stays float32: it feeds relative-error metrics, and float16's three
    significant digits would add error of the size we want to measure.
    Confidence is only ever thresholded, so float16 is enough.

    Not compressed by default. Predicted depth is close to noise in its low bits, so zip compression
    saves under 20% of the size and costs about 3 s per scene and model (measured on the laptop, 40
    frames: 3.4 s for 50 MB against 0.1 s for 61 MB). That was most of a GPU block's non-model time.
    The stored numbers are identical either way, and load_predictions reads both kinds.
    """
    npz_path, json_path = Path(npz_path), Path(json_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    stored = {key: np.asarray(arrays[key]) for key in ARRAY_KEYS if key in arrays}
    stored["depth"] = stored["depth"].astype(np.float32)
    stored["depth_conf"] = stored["depth_conf"].astype(np.float16)
    temporary = npz_path.with_suffix(".tmp.npz")
    (np.savez_compressed if compress else np.savez)(temporary, **stored)
    temporary.replace(npz_path)            # a killed session never leaves a half-written file under the real name
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return npz_path.stat().st_size / 1e6


def load_predictions(npz_path, json_path) -> tuple[dict, dict]:
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["depth_conf"] = arrays["depth_conf"].astype(np.float32)
    return arrays, json.loads(Path(json_path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ comparison arm: VGGT (the predecessor)
#
# Read from facebookresearch/vggt at the pinned commit:
# - load_and_preprocess_images(paths), default mode "crop": width is set to 518, height to the nearest
#   multiple of 14 that keeps the aspect ratio, and only heights above 518 are centre-cropped. A 1920x1200
#   image becomes 518x322, uncropped, with slightly different x and y factors (0.26979 and 0.26833).
# - Outputs have the same form as VGGT-Omega: "depth" (1, S, H, W, 1) = exp(.), "depth_conf" = 1 + exp(.),
#   "pose_enc" decoded by pose_encoding_to_extri_intri into camera_from_world (OpenCV, world = first camera)
#   and intrinsics with the principal point fixed at the image centre.
# - Training normalises scenes to unit average point distance, so depth is scale-normalised here too.
# - Unlike VGGT-Omega, the CALLER must open the autocast context.
# So ground truth, metrics and tables are shared. Only loading, preprocessing and the input size differ.

VGGT_NAME = "vggt_1b"


def load_vggt():
    from vggt.models.vggt import VGGT

    return VGGT.from_pretrained(pins.VGGT_HF_REPO, revision=pins.VGGT_HF_REVISION).to("cuda").eval()


def prepare_images_vggt(image_paths):
    """VGGT's own image preparation (its default "crop" mode). CPU only; see prepare_images."""
    from vggt.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images([str(p) for p in image_paths])


def run_vggt(model, image_paths, images=None) -> dict:
    """One forward pass of VGGT over all images. Same return format as run_model."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    loading = time.time()
    images = (prepare_images_vggt(image_paths) if images is None else images).to("cuda")
    image_seconds = time.time() - loading
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            predictions = model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    torch.cuda.synchronize()
    seconds = time.time() - started

    def numpy(tensor):
        return tensor.detach().float().cpu().numpy()[0]

    height, width = (int(v) for v in images.shape[-2:])
    return {
        "depth": numpy(predictions["depth"])[..., 0], "depth_conf": numpy(predictions["depth_conf"]),
        "extrinsics": numpy(extrinsics), "intrinsics": numpy(intrinsics), "pose_enc": numpy(predictions["pose_enc"]),
        "input_hw": np.array([height, width]), "seconds": seconds, "image_seconds": image_seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
