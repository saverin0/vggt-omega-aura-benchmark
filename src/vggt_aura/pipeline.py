"""The run, block by block: predict on a GPU, score on a CPU, never download a block twice.

Measured before this was written (docs/performance.md): downloading and unpacking a block is two thirds of
a scene's cost. So each block is fetched at most once per step, and everything that is expensive to make is
a file in persistent storage:

- predict_block (GPU): camera layer only; both models' predictions are saved per scene;
- process_block (CPU): camera and LiDAR layers; ground truth is built and saved, scenes are scored in
  parallel processes, one pair of result files is written per block and model;
- load_run and the report functions read those result files and need no dataset at all.

Also here: the census of scene conditions (census_block), the interior-against-outline check for one scene
(edge_scene), and what lets two servers share one block list (claim_block).

Each block gets its own data folder on the runtime disk, removed when the block
is finished, so disk use stays at one block however many are processed.

Results land in   <persist>/metrics/<run_tag>/<model>/blocks/<split>_block<N>_{rows,scenes}.csv
one pair of files per block and model. A block is complete for a model when its
two files exist, which is what makes a dropped session resumable.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import aura_data as ad, edges, evaluation as ev, geometry as geo, ground_truth as gtm, inference as inf, objects as ob, pins

CAMERA_LAYER = "camera_keyframes"
LIDAR_LAYER = "lidar_motion_compensated_keyframes"
MIN_OUSTER_DEFAULT = 6                 # scenes with fewer Ouster LiDARs are skipped, see enough_ouster
CONTEXT_KEYS = ("road_type", "weather", "weather_group", "lighting", "speed_band", "speed_kph_median",
                "sunrise_raw", "sunset_raw", "start_hour_utc", "keyframes", "boxes_total")


class ModelRunner:
    """A model loaded at most once, and only if some scene still needs predicting."""

    def __init__(self, name: str):
        if name not in MODEL_NAMES:
            raise ValueError(f"model must be one of {sorted(MODEL_NAMES)}, got {name!r}")
        self.name, self._model = name, None

    @property
    def folder(self) -> str:                      # the name predictions are stored under
        return pins.CHECKPOINT_PUBLIC_512 if self.name == "vggt_omega_512" else inf.VGGT_NAME

    def ensure_installed(self) -> None:
        """Make the model's package importable. Call from the main thread before any prepare()."""
        import importlib

        if self.name == "vggt_1b":
            from .session import VGGT_PIP_ARGS, install_extra
            install_extra("vggt", VGGT_PIP_ARGS)
        importlib.import_module("vggt.utils.load_fn" if self.name == "vggt_1b" else "vggt_omega.utils.load_fn")

    def prepare(self, image_paths):
        """Read and resize the images for this model, on the CPU. Safe to run in a background thread."""
        if self.name == "vggt_omega_512":
            return inf.prepare_images(image_paths, inf.resolution_for(pins.CHECKPOINT_PUBLIC_512))
        return inf.prepare_images_vggt(image_paths)

    def predict(self, image_paths, images=None) -> dict:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"{self.name}: scenes are left to predict and the model needs a GPU. "
                               "Connect to a GPU server, or run the prediction pass first.")
        if self._model is None:
            started = time.time()
            if self.name == "vggt_omega_512":
                self._model = inf.load_model(inf.download_checkpoint(pins.CHECKPOINT_PUBLIC_512, "/content/checkpoints"))
            else:
                self.ensure_installed()
                self._model = inf.load_vggt()
            print(f"  {self.name} loaded in {time.time() - started:.0f} s")
        if self.name == "vggt_omega_512":
            return inf.run_model(self._model, image_paths, inf.resolution_for(pins.CHECKPOINT_PUBLIC_512), images)
        return inf.run_vggt(self._model, image_paths, images)

    def release(self) -> None:
        if self._model is not None:
            import torch

            self._model = None
            torch.cuda.empty_cache()


MODEL_NAMES = ("vggt_omega_512", "vggt_1b")


def prepared_ahead(jobs, prepare, lookahead: int = 2, threads: int = 2):
    """Yield (job, prepare(job), seconds_waited) in the order of `jobs`, preparing up to `lookahead` jobs
    in background threads while the caller works on the current one.

    Used to read the next scene's images while the GPU runs the present scene. At most `lookahead`
    prepared results exist at a time (one scene's images are about 120 MB). An error in prepare is
    raised here, for the job it belongs to.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    jobs = iter(jobs)
    waiting = deque()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        def top_up():
            while len(waiting) < max(1, lookahead):
                try:
                    job = next(jobs)
                except StopIteration:
                    return
                waiting.append((job, pool.submit(prepare, job)))

        top_up()
        while waiting:
            job, future = waiting.popleft()
            started = time.time()
            prepared = future.result()
            waited = time.time() - started
            top_up()
            yield job, prepared, waited


class _Background:
    """A function running in a thread; .result() waits for it and re-raises its error."""

    def __init__(self, function, *args):
        import threading

        self._value, self._error = None, None

        def run():
            try:
                self._value = function(*args)
            except BaseException as error:          # handed to the caller in result()
                self._error = error

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value


def warm_files(paths, threads: int = 8) -> dict:
    """Read files once and throw the bytes away, several at a time. Returns {"files", "mb", "seconds"}.

    Saved predictions live in persistent storage (Google Drive), where the first read of a file fetches it
    over the network: 50 to 80 s per block when done one by one at scoring time. Done here, in the
    background while the block downloads, the scorers later find the files in the local cache.
    Missing files are skipped; nothing is ever written.
    """
    from concurrent.futures import ThreadPoolExecutor

    def read(path):
        size = 0
        try:
            with open(path, "rb") as stream:
                for piece in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    size += len(piece)
        except OSError:
            return 0
        return size

    started = time.time()
    paths = [Path(p) for p in paths]
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        sizes = list(pool.map(read, paths))
    return {"files": int(sum(size > 0 for size in sizes)), "mb": round(sum(sizes) / 1e6, 1),
            "seconds": round(time.time() - started, 1)}


def block_tag(split: str, block: int) -> str:
    return f"{split}_block{block:06d}"


def block_result_paths(persist_root, run_tag: str, model: str, split: str, block: int) -> tuple[Path, Path]:
    folder = Path(persist_root) / "metrics" / run_tag / model / "blocks"
    return folder / f"{block_tag(split, block)}_rows.csv", folder / f"{block_tag(split, block)}_scenes.csv"


def claim_path(persist_root, run_tag: str, split: str, block: int) -> Path:
    return Path(persist_root) / "metrics" / run_tag / "claims" / f"{block_tag(split, block)}.json"


def claim_block(persist_root, run_tag: str, split: str, block: int, owner: str, max_age_s: float = 3600.0) -> bool:
    """May this server work on the block? False when ANOTHER server claimed it less than max_age_s ago.

    Lets two servers share one block list (one walks it forwards, one backwards). A claim is a small file
    in persistent storage, which takes seconds to show up on the other server, so two servers can still
    grab the same block in the same moment. That wastes one block of work and nothing else: both produce
    the same numbers and result files are written atomically. A claim older than max_age_s is a server
    that died, and is taken over.
    """
    path = claim_path(persist_root, run_tag, split, block)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = None
    if existing and existing.get("owner") != owner and time.time() - float(existing.get("time", 0)) < max_age_s:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{owner}.tmp")
    temporary.write_text(json.dumps({"owner": owner, "time": time.time()}), encoding="utf-8")
    temporary.replace(path)
    return True


def release_claim(persist_root, run_tag: str, split: str, block: int, owner: str) -> None:
    path = claim_path(persist_root, run_tag, split, block)
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("owner") == owner:
            path.unlink()
    except (OSError, ValueError):
        pass


def block_is_done(persist_root, run_tag: str, models, split: str, block: int) -> bool:
    return all(path.is_file() for model in models for path in block_result_paths(persist_root, run_tag, model, split, block))


def block_is_fully_saved(persist_root, scene_names, camera_id: str, models) -> bool:
    """True when every scene of a block already has predictions AND ground truth saved, for every model.

    Then the slow LiDAR layer is not needed: the camera layer (about 1.5 min) is enough. The check is by
    file only and uses nothing from the dataset, so it runs before any download. Whether a saved ground
    truth really fits the request is decided later by load_or_build_scene_truth; if one does not, the
    pipeline fetches LiDAR after all.
    """
    if not scene_names:
        return False
    for name in scene_names:
        for model in models:
            npz_path, json_path = inf.prediction_paths(persist_root, name, camera_id, ModelRunner(model).folder)
            if not (npz_path.is_file() and json_path.is_file()):
                return False
            try:
                height, width = json.loads(json_path.read_text(encoding="utf-8"))["input_hw"]
            except (KeyError, ValueError, TypeError):
                return False
            if not all(p.is_file() for p in gtm.truth_paths(persist_root, name, camera_id, tag=gtm.truth_tag(int(width), int(height)))):
                return False
    return True


def validate_download(data_root) -> dict:
    """Run the SDK's own validator on a downloaded block. Returns its JSON verdict."""
    result = subprocess.run([sys.executable, "-c", "from fzi_aura.cli.validate import main; main()", str(data_root)],
                            capture_output=True, text=True)
    try:
        verdict = json.loads(result.stdout)
    except json.JSONDecodeError:
        verdict = {"ok": False, "errors": [(result.stdout + result.stderr)[-500:]]}
    return {"ok": bool(verdict.get("ok")), "scenes_checked": verdict.get("scene_files_checked"),
            "errors": list(verdict.get("errors", []))[:5]}


def scene_intrinsics(scene, camera_id: str, width: int, height: int):
    calibration = scene.calibration()
    P = calibration.camera_projection_matrix(camera_id)
    native = calibration.sensor(f"camera/{camera_id}")["intrinsics"]
    return geo.scale_intrinsics(P[0, 0], P[1, 1], P[0, 2], P[1, 2], (native["width"], native["height"]), (width, height))


def predictions_for(session, scene, frames, camera_id: str, runner: ModelRunner, images=None) -> tuple[dict, str]:
    """Predictions for a scene: loaded from persistent storage, or made now and saved.

    `images`: this scene's images already prepared for this model (ModelRunner.prepare), or None."""
    npz_path, json_path = inf.prediction_paths(session.persist_root, scene.name, camera_id, runner.folder)
    stamps = np.array([frame.timestamp_ns for frame in frames], dtype=np.int64)
    if npz_path.is_file() and json_path.is_file():
        arrays, _ = inf.load_predictions(npz_path, json_path)
        if np.array_equal(arrays["timestamps_ns"], stamps):
            return arrays, "loaded"
    arrays = runner.predict([frame.camera_path(camera_id) for frame in frames], images)
    arrays["gt_camera0_from_camera"] = inf.ground_truth_cameras(frames, camera_id)
    arrays["timestamps_ns"] = stamps
    saving = time.time()
    inf.save_predictions(npz_path, json_path, arrays, {
        "scene_id": scene.scene_id, "scene_name": scene.name, "camera": camera_id, "frames": len(frames),
        "model": runner.name, "input_hw": arrays["input_hw"].tolist(), "dataset_revision": pins.AURA_DATASET_REVISION,
        "forward_seconds": round(arrays["seconds"], 2), "peak_gpu_gb": round(arrays["peak_gpu_gb"], 2),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    arrays["save_seconds"] = time.time() - saving
    return arrays, "predicted"


def process_block(session, split: str, block: int, scene_ids: list, camera_id: str, models, run_tag: str,
                  lidar_policy: str = "ouster_only", occlusion: gtm.OcclusionParams = gtm.OcclusionParams(),
                  validate: bool = False, keep_data: bool = False, scene_names=None, workers=None,
                  min_ouster: int = MIN_OUSTER_DEFAULT) -> dict:
    """Everything for one block, in one pass. Safe to call again: finished work is skipped.

    Pass `scene_names` (aura_data.block_scene_names) to let a block whose predictions and ground truth
    are all saved already skip the LiDAR layer, which is most of the download time.
    `workers`: how many scenes are scored at once. None = one per CPU core, at most 8. 1 = one at a time.
    The numbers do not depend on it; only the waiting does.
    """
    if block_is_done(session.persist_root, run_tag, models, split, block):
        return {"block": block_tag(split, block), "status": "already done"}
    layers = [CAMERA_LAYER, LIDAR_LAYER]
    if block_is_fully_saved(session.persist_root, scene_names, camera_id, models):
        print("  predictions and ground truth are all saved: fetching the camera layer only")
        try:
            return _process_block(session, split, block, scene_ids, camera_id, models, run_tag, lidar_policy,
                                  occlusion, validate, keep_data, [CAMERA_LAYER], workers, min_ouster, scene_names)
        except gtm.GroundTruthUnavailable as problem:   # a saved ground truth did not fit after all. Nothing else is
            # caught here: a failed validator or a missing prediction must stop the block, not trigger a second download
            print("  ", problem)
            print("  falling back to the full download")
    return _process_block(session, split, block, scene_ids, camera_id, models, run_tag, lidar_policy,
                          occlusion, validate, keep_data, layers, workers, min_ouster, scene_names)


def enough_ouster(lidar_ids, min_ouster: int = MIN_OUSTER_DEFAULT) -> bool:
    """True when a scene has at least `min_ouster` Ouster LiDARs.

    The census found 55 scenes (all in train) with only four. Ground truth from four sensors is thinner than
    from six, so those scenes would not be comparable with the rest. They are skipped, and said so in the log.
    """
    return len(gtm.select_lidars(lidar_ids, "ouster_only")) >= min_ouster


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def predictions_are_saved(persist_root, scene_name: str, camera_id: str, folder: str, timestamps_ns) -> bool:
    """True when saved predictions exist for exactly these frames. Reads ONLY the frame times from the file.

    An .npz is a zip: asking for one array reads that array alone, a few hundred bytes, not the
    50 MB of depth maps. Loading whole files just to check them cost 137 s per block (measured).
    """
    npz_path, json_path = inf.prediction_paths(persist_root, scene_name, camera_id, folder)
    if not (npz_path.is_file() and json_path.is_file()):
        return False
    try:
        with np.load(npz_path) as data:
            saved = data["timestamps_ns"]
    except Exception:                      # unreadable or half-written file: treat as missing and predict again
        return False
    return bool(np.array_equal(saved, np.asarray(timestamps_ns, dtype=np.int64)))


def load_saved_predictions(persist_root, scene_name: str, camera_id: str, folder: str, timestamps_ns):
    """Saved predictions for exactly these frames, or None."""
    npz_path, json_path = inf.prediction_paths(persist_root, scene_name, camera_id, folder)
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    arrays, _ = inf.load_predictions(npz_path, json_path)
    return arrays if np.array_equal(arrays["timestamps_ns"], np.asarray(timestamps_ns, dtype=np.int64)) else None


def score_scene(job: dict) -> dict:
    """Ground truth, labels and scores for ONE scene and every model. Runs in a worker process.

    It takes only plain values and reads everything else from disk, so it can run in parallel with other
    scenes. It never touches a GPU: predictions must already be saved (phase A of _process_block).
    """
    from fzi_aura import FZIAURADataset

    started = time.time()
    scene_id, camera_id, models = job["scene_id"], job["camera_id"], job["models"]
    occlusion = gtm.OcclusionParams(**job["occlusion"])
    scene = FZIAURADataset(job["data_root"], split=job["split"]).get_scene(scene_id)
    frames = inf.select_frames(scene, camera_id)
    stamps = [int(frame.timestamp_ns) for frame in frames]
    calibrated = scene.calibration().sensor_ids("lidar")
    lidars = gtm.select_lidars(calibrated, job["lidar_policy"])
    on_disk = set(scene.available_lidars())
    classes = json.loads((scene.path / "labels" / "semantic" / "classes.json").read_text(encoding="utf-8"))
    class_names = {int(c["id"]): c["name"] for c in classes["semantic_classes"]}
    drop_ids = gtm.excluded_class_ids(class_names)
    context = {k: v for k, v in ad.scene_summary(scene).items() if k in CONTEXT_KEYS}
    context.update({"n_lidars_calibrated": len(calibrated), "n_lidars_used": len(lidars)})

    steps = {"setup": time.time() - started, "predictions": 0.0, "truth": 0.0, "labels": 0.0, "evaluate": 0.0}

    def timed(step, function, *args, **kwargs):
        clock = time.time()
        value = function(*args, **kwargs)
        steps[step] += time.time() - clock
        return value

    tracks = timed("labels", ob.scene_tracks, frames)          # poses, boxes, velocities: the same for every model
    rows, records, notes = {}, {}, []
    for model in models:
        arrays = timed("predictions", load_saved_predictions, job["persist_root"], scene.name, camera_id,
                       ModelRunner(model).folder, stamps)
        if arrays is None:
            raise RuntimeError(f"{scene_id}: no saved predictions for {model}")
        height, width = (int(v) for v in arrays["input_hw"])
        K = scene_intrinsics(scene, camera_id, width, height)
        truth, how_truth = timed("truth", gtm.load_or_build_scene_truth,
                                 job["persist_root"], scene_id, scene.name, frames, camera_id, lidars, K, width, height,
                                 occlusion, drop_ids, pins.AURA_DATASET_REVISION, can_build=set(lidars) <= on_disk)
        labels = timed("labels", ob.scene_motion_labels, frames, camera_id, truth, tracks)
        scene_rows, record = timed("evaluate", ev.evaluate_scene, scene_id, arrays, truth, labels, class_names, K[0], K[1])
        rows[model] = scene_rows
        records[model] = {**record, **context, "split": job["split"], "block": job["block"], "model": model}
        notes.append(f"{model}: truth {how_truth}")
    return {"scene_id": scene_id, "rows": rows, "records": records, "notes": notes, "seconds": time.time() - started,
            "steps": {step: round(seconds, 2) for step, seconds in steps.items()},
            "sensors": {"lidars_calibrated": sorted(calibrated), "lidars_used": lidars,
                        "aeva_present": len(calibrated) != len(gtm.select_lidars(calibrated, "ouster_only"))}}


def frames_for_timestamps(scene, timestamps_ns) -> list:
    """The scene's frames with exactly these timestamps, in this order, whatever layers are on disk."""
    by_stamp = {int(frame.timestamp_ns): frame for frame in ad.iter_frames(scene.frames(sample_filter="sensor_available"))}
    wanted = [int(stamp) for stamp in np.asarray(timestamps_ns).tolist()]
    missing = [stamp for stamp in wanted if stamp not in by_stamp]
    if missing:
        raise RuntimeError(f"{scene.scene_id}: {len(missing)} of {len(wanted)} predicted frames are not in the dataset index")
    return [by_stamp[stamp] for stamp in wanted]


def edge_scene(job: dict) -> dict:
    """One scene of the interior-against-boundary check (edges.py), for every model and every radius asked for.

    Reads SAVED predictions and SAVED ground truth; from the dataset it needs only the base layer (boxes,
    ego poses, calibration). A scene that was never scored is reported as such, not treated as an error.
    """
    from fzi_aura import FZIAURADataset

    started = time.time()
    scene_id, camera_id = job["scene_id"], job["camera_id"]
    scene = FZIAURADataset(job["data_root"], split=job["split"]).get_scene(scene_id)
    tables, notes, tracks, frames, stamps = [], [], None, None, None
    for model in job["models"]:
        npz_path, json_path = inf.prediction_paths(job["persist_root"], scene.name, camera_id, ModelRunner(model).folder)
        if not (npz_path.is_file() and json_path.is_file()):
            notes.append(f"{model}: no saved predictions, left out")
            continue
        arrays, _ = inf.load_predictions(npz_path, json_path)
        if frames is None:
            # NOT inf.select_frames: with only the base layer on disk the toolkit hides every frame that needs a
            # camera image, and the selection comes back empty. The frames that were predicted are found by their
            # timestamps instead, which the saved predictions carry.
            frames = frames_for_timestamps(scene, arrays["timestamps_ns"])
            stamps = [int(frame.timestamp_ns) for frame in frames]
        if not np.array_equal(np.asarray(arrays["timestamps_ns"], dtype=np.int64), np.asarray(stamps, dtype=np.int64)):
            raise RuntimeError(f"{scene_id}: the models' saved predictions are for different frames")
        height, width = (int(v) for v in arrays["input_hw"])
        npz_path, json_path = gtm.truth_paths(job["persist_root"], scene.name, camera_id, gtm.truth_tag(width, height))
        if not npz_path.is_file():
            notes.append(f"{model}: no saved ground truth, left out")
            continue
        truth, truth_stamps, _ = gtm.load_scene_truth(npz_path, json_path)
        if not np.array_equal(np.asarray(truth_stamps, dtype=np.int64), np.asarray(stamps, dtype=np.int64)):
            raise RuntimeError(f"{scene_id}: saved ground truth and saved predictions are for different frames")
        tracks = ob.scene_tracks(frames) if tracks is None else tracks
        labels = ob.scene_motion_labels(frames, camera_id, truth, tracks)
        for radius in job["radii"]:
            table = edges.edge_rows(scene_id, arrays["depth"], truth, labels, int(radius))
            if len(table):
                tables.append(table.assign(model=model, radius=int(radius), split=job["split"], block=job["block"]))
        notes.append(f"{model}: ok")
    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return {"scene_id": scene_id, "rows": rows, "notes": notes, "seconds": time.time() - started}


def _selftest_job(job: dict) -> dict:
    """Stand-in for score_scene in the tests of run_jobs: must live in this module so worker processes can import it."""
    if job.get("fail"):
        raise RuntimeError(f"planted failure in {job['scene_id']}")
    return {"scene_id": job["scene_id"], "value": job["value"] ** 2, "pid": os.getpid()}


def run_jobs(jobs: list, workers=None, function=score_scene, on_result=None) -> list:
    """Run one job per scene and return the results IN JOB ORDER, whatever order they finish in.

    workers = 1 runs them here, one after another. More workers use separate processes, started with
    "spawn": a fresh interpreter each, which is slower to start than a fork but safe next to a process
    that has used CUDA. A failing job raises here, in the caller, with the scene named in the message.
    """
    workers = default_workers() if workers is None else max(1, int(workers))
    workers = min(workers, max(1, len(jobs)))
    results = [None] * len(jobs)
    if workers == 1:
        for index, job in enumerate(jobs):
            results[index] = function(job)
            if on_result:
                on_result(results[index])
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        for index, future in enumerate(futures):        # collected in job order, so output order is stable
            results[index] = future.result()
            if on_result:
                on_result(results[index])
    return results


def _process_block(session, split, block, scene_ids, camera_id, models, run_tag, lidar_policy, occlusion,
                   validate, keep_data, layers, workers=None, min_ouster=MIN_OUSTER_DEFAULT, scene_names=None) -> dict:
    from fzi_aura import FZIAURADataset

    started = time.time()
    data_root = session.data_root / block_tag(split, block)
    # While the block downloads (network and disk), fetch its saved predictions from persistent storage.
    warming = _Background(warm_files, [inf.prediction_paths(session.persist_root, name, camera_id, ModelRunner(model).folder)[0]
                                       for name in (scene_names or []) for model in models])
    if not ad.block_on_disk(data_root, scene_ids, layers):
        ad.download_block(data_root, split, block, scene_ids, layers)
    download_s = time.time() - started
    summary = {"block": block_tag(split, block), "layers": list(layers), "download_s": round(download_s, 1)}
    # The toolkit's validator only reads the data folder, so it runs while the scenes are scored.
    # Its verdict is collected BEFORE any result is written: a block that fails validation saves nothing.
    validating = _Background(validate_download, data_root) if validate else None
    summary["predictions_warmed"] = warming.result()

    dataset = FZIAURADataset(data_root, split=split)
    scene_ids, skipped = ad.usable_scene_ids(dataset, scene_ids)
    if skipped:
        print(f"  excluded by the dataset's maintainers, skipped: {skipped}")

    # Phase A, here and one scene at a time: make sure every prediction is saved. Only this phase can need a GPU.
    phase_a = time.time()
    runners = {model: ModelRunner(model) for model in models}
    usable, made = [], 0
    for scene_id in scene_ids:
        scene = dataset.get_scene(scene_id)
        frames = inf.select_frames(scene, camera_id)
        if len(frames) < 3:
            print(f"  {scene_id}: skipped, only {len(frames)} usable frames")
            continue
        if not enough_ouster(scene.calibration().sensor_ids("lidar"), min_ouster):
            print(f"  {scene_id}: skipped, fewer than {min_ouster} Ouster LiDARs (thinner ground truth than the rest)")
            continue
        stamps = [int(frame.timestamp_ns) for frame in frames]
        for model in models:
            if predictions_are_saved(session.persist_root, scene.name, camera_id, runners[model].folder, stamps):
                continue                   # already there: nothing to load, nothing to run
            predictions_for(session, scene, frames, camera_id, runners[model])
            made += 1
        usable.append(scene_id)
    for runner in runners.values():
        runner.release()
    predict_s = time.time() - phase_a

    # Phase B, in parallel: ground truth, moving-object labels and scores. CPU only.
    phase_b = time.time()
    jobs = [{"scene_id": scene_id, "camera_id": camera_id, "models": list(models), "split": split, "block": block,
             "data_root": str(data_root), "persist_root": str(session.persist_root), "lidar_policy": lidar_policy,
             "occlusion": occlusion.as_dict()} for scene_id in usable]
    used = min(default_workers() if workers is None else max(1, int(workers)), max(1, len(jobs)))
    print(f"  predictions: {made} made now, the rest loaded ({predict_s:.0f} s) | scoring {len(jobs)} scenes with {used} worker(s)")
    results = run_jobs(jobs, workers, on_result=lambda r: print(
        f"  {r['scene_id']:26} {r['seconds']:5.1f} s | " + " | ".join(r["notes"])))
    score_s = time.time() - phase_b
    step_names = sorted({step for r in results for step in r.get("steps", {})})
    summary["score_steps_worker_s"] = {step: round(sum(r.get("steps", {}).get(step, 0.0) for r in results), 1) for step in step_names}

    if validating is not None:
        summary["validation"] = validating.result()
        print("  validator:", summary["validation"])
        if not summary["validation"].get("ok"):
            shutil.rmtree(data_root, ignore_errors=True)       # so that a re-run downloads the block afresh
            raise RuntimeError(f"{block_tag(split, block)}: the dataset toolkit's validator rejected the download, "
                               f"nothing was saved for this block: {summary['validation'].get('errors')}")

    for model in models:
        rows = [r["rows"][model] for r in results]
        records = [r["records"][model] for r in results]
        if not rows:
            continue
        rows_path, scenes_path = block_result_paths(session.persist_root, run_tag, model, split, block)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(rows_path.with_suffix(".tmp"), index=False)
        pd.DataFrame(records).to_csv(scenes_path.with_suffix(".tmp"), index=False)
        rows_path.with_suffix(".tmp").replace(rows_path)          # both files appear only when both are complete
        scenes_path.with_suffix(".tmp").replace(scenes_path)
        for record in records:
            session.manifest.mark_done(record["scene_id"], f"pipeline/{run_tag}/{model}", block=block_tag(split, block))

    if not keep_data:
        shutil.rmtree(data_root, ignore_errors=True)
    summary.update({"status": "done", "scenes": len(results), "workers": used, "predict_s": round(predict_s, 1),
                    "score_s": round(score_s, 1), "sensors": results[0]["sensors"] if results else None,
                    "total_s": round(time.time() - started, 1)})
    return summary


def block_has_predictions(persist_root, scene_names, camera_id: str, models) -> bool:
    """True when every scene of a block has saved predictions for every model. Checked by file, before any download."""
    if not scene_names:
        return False
    return all(path.is_file() for name in scene_names for model in models
               for path in inf.prediction_paths(persist_root, name, camera_id, ModelRunner(model).folder))


def predict_block(session, split: str, block: int, scene_ids: list, camera_id: str, models,
                  scene_names=None, keep_data: bool = False, min_ouster: int = MIN_OUSTER_DEFAULT,
                  runners=None) -> dict:
    """The GPU half of a two-pass run: predictions only, from the camera layer only.

    `runners`: {model name: ModelRunner} made once by the caller. The models then stay loaded from block to
    block (loading both took about 26 s per block) and the CALLER releases them at the end. Without it,
    models are loaded and released here, as before.

    A GPU is busy for about two minutes of a block's half hour; the rest is download and CPU work. So the
    cheap way is to do ONLY the model runs on a GPU server (camera layer, about 1.5 min to fetch), and
    leave ground truth and scoring to process_block on a CPU server, which then finds these predictions
    saved. Nothing is scored here. Safe to call again: scenes already predicted are skipped.
    """
    from fzi_aura import FZIAURADataset

    if block_has_predictions(session.persist_root, scene_names, camera_id, models):
        return {"block": block_tag(split, block), "status": "already predicted"}
    started = time.time()
    data_root = session.data_root / f"predict_{block_tag(split, block)}"
    if not ad.block_on_disk(data_root, scene_ids, [CAMERA_LAYER]):
        ad.download_block(data_root, split, block, scene_ids, [CAMERA_LAYER])
    download_s = time.time() - started
    dataset = FZIAURADataset(data_root, split=split)
    scene_ids, skipped = ad.usable_scene_ids(dataset, scene_ids)
    if skipped:
        print(f"  excluded by the dataset's maintainers, skipped: {skipped}")
    own_runners = runners is None
    runners = {model: ModelRunner(model) for model in models} if own_runners else runners
    counts = {model: {"predicted": 0, "loaded": 0} for model in models}
    gpu_s = image_s = save_s = 0.0
    todo = []                                   # (scene, frames, model) still to predict, in a fixed order
    for scene_id in scene_ids:
        scene = dataset.get_scene(scene_id)
        frames = inf.select_frames(scene, camera_id)
        if len(frames) < 3:
            print(f"  {scene_id}: skipped, only {len(frames)} usable frames")
            continue
        if not enough_ouster(scene.calibration().sensor_ids("lidar"), min_ouster):
            print(f"  {scene_id}: skipped, fewer than {min_ouster} Ouster LiDARs")
            continue
        stamps = [int(frame.timestamp_ns) for frame in frames]
        for model in models:
            if predictions_are_saved(session.persist_root, scene.name, camera_id, runners[model].folder, stamps):
                counts[model]["loaded"] += 1
            else:
                todo.append((scene, frames, model))
    for model in {model for _, _, model in todo}:
        runners[model].ensure_installed()       # in the main thread, before any background image reading

    def prepare(job):
        scene, frames, model = job
        return runners[model].prepare([frame.camera_path(camera_id) for frame in frames])

    # The next scene's images are read and resized in the background while the GPU runs this one.
    for (scene, frames, model), images, waited in prepared_ahead(todo, prepare):
        arrays, how = predictions_for(session, scene, frames, camera_id, runners[model], images)
        counts[model][how] += 1
        gpu_s += float(arrays["seconds"])
        image_s += waited + float(arrays.get("image_seconds", 0.0))     # time the GPU actually waited for images
        save_s += float(arrays.get("save_seconds", 0.0))                # writing the prediction to persistent storage
    if own_runners:
        for runner in runners.values():
            runner.release()
    if not keep_data:
        shutil.rmtree(data_root, ignore_errors=True)
    return {"block": block_tag(split, block), "status": "predicted", "download_s": round(download_s, 1),
            "model_seconds": round(gpu_s, 1), "image_seconds": round(image_s, 1), "save_seconds": round(save_s, 1),
            "total_s": round(time.time() - started, 1),
            **{f"{model}_new": counts[model]["predicted"] for model in models}}


def load_run(persist_root, run_tag: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every finished block of a run, stitched together. Needs no dataset on disk."""
    folder = Path(persist_root) / "metrics" / run_tag / model / "blocks"
    row_files = sorted(folder.glob("*_rows.csv"))
    if not row_files:
        return pd.DataFrame(), pd.DataFrame()
    rows = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True)
    scenes = pd.concat([pd.read_csv(Path(str(path).replace("_rows.csv", "_scenes.csv"))) for path in row_files], ignore_index=True)
    repeated = scenes.loc[scenes["scene_id"].duplicated(), "scene_id"].unique().tolist()
    if repeated:            # a scene counted twice would silently weigh double in every mean
        raise ValueError(f"run {run_tag!r}, model {model!r}: scenes appear in more than one block file: {repeated[:5]}")
    return rows, scenes


# ------------------------------------------------------------------ census: what is in the dataset, from metadata only

CENSUS_COLUMNS = ("n_lidars_ouster", "lidar_ids")     # added after the first census session


def census_path(persist_root, split: str, block: int) -> Path:
    return Path(persist_root) / "census" / f"{block_tag(split, block)}.csv"


def census_is_current(persist_root, split: str, block: int) -> bool:
    """True when a block's census table exists AND has every column the current code writes.

    A table from an older version of the code is counted again and REPLACED by one that holds everything
    it had plus the new columns. Nothing is ever deleted without being rewritten.
    """
    path = census_path(persist_root, split, block)
    if not path.is_file():
        return False
    return set(CENSUS_COLUMNS) <= set(pd.read_csv(path, nrows=0).columns)


def census_block(session, split: str, block: int, scene_ids: list) -> pd.DataFrame:
    """Context of every scene of a block, from the small metadata layer alone. No images, no LiDAR."""
    from fzi_aura import FZIAURADataset

    out = census_path(session.persist_root, split, block)
    if census_is_current(session.persist_root, split, block):
        return pd.read_csv(out)
    data_root = session.data_root / f"census_{block_tag(split, block)}"
    ad.download_block(data_root, split, block, scene_ids, [ad.BASE_LAYER])
    dataset = FZIAURADataset(data_root, split=split)
    scene_ids, skipped = ad.usable_scene_ids(dataset, scene_ids)
    if skipped:
        print(f"  excluded by the dataset's maintainers, skipped: {skipped}")
    records = []
    for scene_id in scene_ids:
        scene = dataset.get_scene(scene_id)
        summary = ad.scene_summary(scene)
        lidars = summary["lidars_calibrated"]
        records.append({"scene_id": scene_id, "split": split, "block": block, "recording": scene_id.split("|")[0],
                        **{k: summary[k] for k in ("keyframes", "road_type", "weather", "weather_group", "lighting", "speed_band",
                                                   "speed_kph_median", "keyframe_spacing_m_median", "start_hour_utc", "boxes_total")},
                        "n_lidars": len(lidars), "has_aeva": any("aeva" in name.lower() for name in lidars),
                        "n_lidars_ouster": len(gtm.select_lidars(lidars, "ouster_only")),
                        "lidar_ids": " ".join(sorted(lidars)),
                        "n_cameras": len(summary["cameras_calibrated"]), "n_radars": len(summary["radars_calibrated"])})
    table = pd.DataFrame(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".tmp"), index=False)
    out.with_suffix(".tmp").replace(out)
    shutil.rmtree(data_root, ignore_errors=True)
    return table
