# Decisions and verified findings

> Notebook names in this log are the working names of the time (`07b_blocks`, `08c_test_report`, ...). `notebooks/README.md` maps them to the published names.

Each entry says what was checked, where, and when. Anything not listed here
is not verified.

## 2026-09-18 Phase 1

### Predicted depth is scale-normalised, not metric

Source: `facebookresearch/vggt-omega` at commit `a3ab0141`.

- `training/data/composed_dataset.py` ends every training sample with
  `normalize_camera_extrinsics_and_points_batch(sample)`.
- `training/train_utils/normalization.py` defines that normalisation: the
  first camera becomes the world origin, then depths, camera translations and
  world points are all divided by one scalar per sequence. The scalar is the
  mean Euclidean distance of the valid 3D points from the first camera.
- `training/config/default.yaml` has `normalize_predictions: false`, so the
  network is trained to output directly in that normalised frame.

Consequence: one unknown scale factor per sequence, shared by depth and
camera translation. Raw depth error in metres is meaningless. Rotations and
intrinsics are unaffected by the scale.

The authors' own evaluation (`eval/metrics.py`):

- Depth: per frame, median scale AND shift alignment, ground truth kept to
  0.1 to 100 m, then AbsRel and delta 1.25.
- Pose: pairwise relative rotation angle and translation direction angle,
  summarised as AUC at 3 and 30 degrees. Translation magnitude is never
  scored, so their pose metric is scale-free.

Not yet checked: Section 3 of the paper (arXiv 2605.15195). The code is
unambiguous, so the paper is a cross-check rather than a blocker.

Open for phase 5: the alignment protocol. Note that scale-and-shift per frame
is more forgiving than scale-only per sequence. Driving has a true metric
ground truth, so a single scale per sequence is the stricter and more
informative choice. Report both.

### The README notice has been replaced by a retrained checkpoint

Source: live README and `reproduction.md` at commit `a3ab0141`, dated
2026-09-17.

- There is no "Important Notice" in the live README.
- An update dated 2026-09-08 announces the training code and a third
  checkpoint, `vggt_omega_1b_416_reproduce.pt`, retrained in Aug 2026 on data
  re-checked for overlap with the evaluation benchmarks.
- The authors state that the retrained checkpoint "supersedes the original
  checkpoint as the reference for future comparisons on these benchmarks".
- The model table now labels `VGGT-Omega-1B-512` as "Recommended for
  in-the-wild applications, no benchmarking".
- The authors also state that the retrained checkpoint "does not consistently
  match the original checkpoint's qualitative performance on in-the-wild
  videos". It was trained on a shortened schedule, at a fixed 416 x 416 pixel
  budget, initialised from the public VGGT checkpoint, with the
  self-supervised stage skipped.
- The retrained checkpoint needs `image_resolution=416` in preprocessing.

Consequence for this project: the premise "the investigation is ongoing" is
out of date. The contamination concern is about the authors' benchmarks and
does not apply to AURA for either checkpoint.

DECISION (project owner, 2026-09-18): evaluate the public checkpoint
`vggt_omega_1b_512.pt` only, with `image_resolution=512`. Reasons: AURA
post-dates the checkpoint, so the contamination concern does not apply, and
the authors recommend this checkpoint for in-the-wild use, which driving
footage is. The retrained 416 checkpoint is not evaluated. The writeup must
say so plainly, and must say that the authors name the retrained checkpoint
as the reference for their own benchmarks. Adding it later as a second arm
stays possible: ground truth and metrics are checkpoint-independent.

### Licence notes

Model: FAIR Noncommercial Research License, read in full at commit
`a3ab0141`.

- 1.b.i: the materials "or any outputs or results" may be used only for
  noncommercial research. This covers predicted depth maps, poses and metrics
  derived from them.
- 1.b.ii: distributing the materials or derivative works requires doing so
  under the same agreement and including a copy. Weights are never committed
  here, so this is not triggered by the repo.
- 1.b.iii: publications must acknowledge use of the materials.
- Nothing in the licence forbids publishing outputs or results. The repo
  README must state the noncommercial restriction on the results.

Dataset: CC BY-SA 4.0. Figures that show AURA imagery are adaptations and
must carry CC BY-SA 4.0 with attribution to FZI. No imagery, point clouds or
labels are committed beyond what figures need.

The licence for this repo's own code is not yet chosen.

### Unverified risk noted for phase 3

`vggt-omega` declares `numpy<2` as a dependency. If the Colab image ships
NumPy 2, a plain `pip install -e .` would downgrade NumPy under the running
kernel. The bootstrap installs the package with `--no-deps` and adds only the
missing pure-Python dependencies. Whether the model code runs on NumPy 2 is
unverified until the first forward pass.

## 2026-09-18 Code delivery and persistence (project owner's decision)

All work is local. The project is NOT cloned on the Colab runtime and git is
not part of the run loop.

- Code reaches the runtime through the sync cell in the bootstrap notebook.
  `python -m vggt_aura.sync` packs `src/`, `cpp/` and `pyproject.toml` into
  that cell as a base64 zip; running the cell unpacks and installs it. The
  zip is deterministic, and the same 12-character hash is printed locally and
  on the runtime, so stale code is detectable. `--check` exits 1 when a
  notebook carries old code. `.env` files are never packed (tested).
- The Colab extension (0.9.6) also has `Upload to Colab` and `Download...`
  commands, which are the manual route for anything else.
- Upstream packages are pip-installed from their public repos at the pinned
  commits. No token is needed for that.
- Persistence modes are `drive` and `runtime`. `runtime` writes to the
  runtime disk, which is wiped at session end: the last notebook cell zips
  `results/` for download. To resume in `runtime` mode, the previous
  `results/` must be uploaded first. `drive` needs neither step, so it is
  preferred if the Drive mount works. That is still unverified.
- Only `HF_TOKEN` is needed. No GitHub token is used anywhere.

## 2026-09-18 Phase 0 runtime probes (measured, notebooks/00_probes.ipynb)

Run by the project owner from VS Code with the Colab extension 0.9.6.

| Item | Result |
|---|---|
| GPU | None attached, on purpose: the probes need no GPU. torch was the CPU build (2.11.0+cpu). A GPU runtime is first needed at phase 3. |
| Disk | 221 GB free of 242 GB at /content |
| Python / NumPy | 3.13.15 / 2.1.3. NumPy 2 confirms the `numpy<2` risk is real; `--no-deps` install stays. |
| Preinstalled | pandas 2.2.3, pyarrow 23.0.1, einops 0.8.2, safetensors 0.8.0, OpenCV 5.0.0 |
| g++ / cmake / make | 13.3.0 / 3.31.10 / 4.3, all present |
| ninja, pybind11, Eigen | missing. pybind11 is now pip-installed by the bootstrap. Eigen is decided at phase 6. |
| Colab secrets | NOT usable: `userdata.get` raised TimeoutException. |
| HF token via hidden prompt | works, account saverino, gated 512 checkpoint reachable |
| Google Drive mount | works, write and read verified |
| Dataset at pinned revision | reachable |
| Server Mounting | not yet reported |

Decisions that follow:

- Persistence mode is `drive`. Results and the resume manifest live in
  `MyDrive/vggt-omega-aura-benchmark/`.
- The Hugging Face token is read from a `.env` file in that same Drive
  folder. Colab secrets are skipped by default because the call stalls.
- The bootstrap has a `REQUIRE_GPU` switch, off until phase 3. When on, it stops at its second cell if no GPU is attached.
- The 242 GB disk comfortably fits several 20-scene blocks with LiDAR.

## 2026-09-18 Bootstrap notebook verified on Colab

First full run of `notebooks/01_bootstrap.ipynb` by the project owner,
CPU runtime, Python 3.13, NumPy 2.1.3. Every cell passed except cell 7, which
was a bug in the notebook (fixed: it tried to zip a results folder that does
not exist in drive mode).

- Sync cell: code hash on the runtime matched the local hash.
- `vggt-omega` installed with `--no-deps` at the pinned commit; `fzi-aura`
  installed at the pinned commit; pybind11 3.1.0 installed.
- Token read from the `.env` on Drive with no prompt, account saverino.
- Persist root on Drive created, manifest empty as expected.

Still unverified: that the model code actually RUNS on NumPy 2. Installing is
not importing, and importing is not a forward pass. That is settled at phase 3.

## 2026-09-18 Phase 2 preparation: facts read from the SDK source

Source: `fzi-aura-sdk` at commit `a36761db`, files `download.py`, `dataset.py`,
`scene.py`, `frame.py`, `frame_dataset.py`, `calibration.py`, and `docs/`.

- The downloader writes `dataset.json` for the REQUESTED scenes only. Asking
  for one scene downloads its whole block but leaves the other 19 invisible to
  the SDK. A block is therefore always requested by listing all its scene IDs.
  Rehearsed live: 20 requested scenes, 0 boundary neighbours, 4.91 GB.
- A download command must describe the complete dataset root wanted. Adding
  LiDAR later means repeating `camera_keyframes` in the same command.
- Archives are deleted after extraction, so a repeated command downloads
  again. `block_on_disk()` guards against that within a session.
- `load_vehicle_signals(columns=[...])` raises when a scene lacks a named
  column; the SDK treats only `visibility` as optional. We load all columns and
  pick, and report which context columns were missing.
- Road type, weather and speed are columns of `ego/vehicle_signals.parquet`.
  `weather` is a list of OpenWeather condition records. There is NO lighting
  field in the dataset: lighting must be derived, for example from local time
  and sun elevation. Method not chosen yet.
- `FrameDataset` has `__len__` and `__getitem__` but no `__iter__`.
- Calibration entries hold `modality`, `sensor_id`, `frame`,
  `transform_base_link_from_sensor` (`translation_xyz`, `rotation_xyzw`) and,
  for cameras, `intrinsics.P` reshaped to 3x4. Whether distortion parameters or
  an image size are present is unknown until a real file is read (notebook 02).
- Motion-compensated LiDAR has no non-keyframe layer at all, so LiDAR ground
  truth exists only at 2 Hz keyframes whichever camera layer is used.

Protocol choice: develop on the `val` split, keep `test` untouched until the
alignment and occlusion rules are frozen. First block: `val` block 11, the
smallest full 20-scene val block (3.47 GB camera, 1.44 GB base, 3.84 GB
motion-compensated LiDAR when that is added).

## 2026-09-18 Phase 2 results: val block 11, measured (notebooks/02_download_inspect.ipynb)

20 scenes, camera_keyframes + base_keyframes, 4.91 GB, downloaded and extracted
in 1.4 min on a CPU runtime. Bandwidth is not a constraint at this size.
Every SDK call from the dataset card worked as documented.

Sensors (identical in all 20 scenes; one calibration file shared by the block):

- Cameras (8): front_medium, front_tele, front_wide, left_forward,
  left_rearward, rear_wide, right_forward, right_rearward.
- LiDARs (6): front_left, front_right, rear_left, rear_right, top_left,
  top_right. 2026 recordings have 12 per the release metadata; not yet seen.
- Radars (3): front_left, front_right, rear_center.

Camera calibration:

| camera | image | fx | fy | cx | cy |
|---|---|---|---|---|---|
| front_medium | 1920x1200 | 1614.6 | 1694.2 | 975.3 | 589.8 |
| front_tele | 1920x1200 | 3426.4 | 3449.4 | 953.2 | 587.4 |
| front_wide | 2592x2048 | 969.8 | 982.9 | 1313.7 | 1017.8 |
| rear_wide | 2592x2048 | 910.2 | 911.6 | 1300.6 | 1033.7 |
| four side cameras | 1920x1200 | 982 to 998 | 1021 to 1032 | about centre | about centre |

- `camera_model` is `pinhole`. The entry holds only `P`, `width`, `height` and
  `distortion_model: plumb_bob`. There are NO distortion coefficients and no
  separate K. P's fourth column is zero. The principal point sits at the image
  centre, so P is for the native image size (trap 4: scale P by the resize
  factors, separately in x and y).
- UNVERIFIED: whether the JPGs are already undistorted. A projection matrix
  with no coefficients suggests rectified images, but that is an inference.
  Phase 4 must check it visually: LiDAR points on poles and building edges
  near the image borders, especially on the two wide cameras.
- fx and fy differ by about 5% on front_medium. Pixels are not square in this
  calibration, so fx and fy are compared separately against the model.
- front_medium optical axis in base_link is (1.0, -0.02, 0.01): it looks
  forward, as expected. Camera frames follow the optical convention
  (z forward), confirmed by the rotation.
- 1920x1200 is 1.6:1, not 3:2. The model's resized shape for this aspect is
  not the 624x416 quoted for 3:2 inputs. Read the real shape in phase 3.

Timing:

- Camera exposure stamps sit within 1.4 ms of the sample timestamp
  (front_medium 0.004 ms, worst side camera -1.35 ms). Motion-compensated
  LiDAR and label files carry the sample timestamp exactly. Camera-to-LiDAR
  reference-time mismatch is therefore centimetres even at motorway speed.
- That does NOT close trap 3. Motion compensation corrects ego motion only. A
  moving object is still sampled across the ~100 ms sweep, so its points can
  be displaced by up to sweep time x object speed. Whether the PCDs carry
  per-point times is unknown until a LiDAR layer is read (phase 4).

Context fields (all in ego/vehicle_signals.parquet, none missing in this block):
`road_type`, `osm_highway_tag`, `weather`, `speed_kph`, and also `sunrise`,
`sunset`, `visibility`, `rain_1h`, `clouds`, `recording_tags`.

- Lighting CAN be derived from the dataset itself: sample time against the
  `sunrise` and `sunset` columns. This replaces the earlier note that lighting
  would need an outside source. Units and timezone of those columns unchecked.
- road_type changes inside 2 of 20 scenes. Scene-level strata use the mode.

Strata in this block (trap 7 confirmed):

- 18 urban, 2 overland, 0 motorway. All daytime (07 to 14 UTC). Weather: 3
  light rain, the rest clear or cloudy. Blocks group scenes from a few
  recordings, so ONE BLOCK IS NOT A SAMPLE OF THE DATASET. Strata counts need
  a census over many blocks. The base layer alone (about 60 GB for all 1,081
  public scenes) holds everything the census needs.
- Keyframe spacing: urban median 2.2 m (0.1 to 4.3), overland about 10 m.
- Several scenes are nearly stationary (1 to 4 km/h, 0.15 to 0.6 m between
  keyframes). With almost no baseline, translation direction is ill-defined
  and any pose-translation score on them is noise. They need their own
  stratum or an explicit exclusion rule, fixed before phase 5.

Labels: all 40 keyframes per scene carry boxes (1,300 to 1,800 boxes per
scene, 28 in the first frame). 35 semantic classes; `unlabeled` = 0,
`noise` = 27, `ego` = 31, plus generic `dynamic` = 29 and `static` = 30. What to
do with unlabeled, noise and ego in ground-truth depth is decided in phase 4.

## 2026-09-18 Phase 3 preparation: facts read from the model source

Source: `facebookresearch/vggt-omega` at commit `a3ab0141`, files
`utils/load_fn.py`, `utils/pose_enc.py`, `models/vggt_omega.py`,
`models/heads/dense_head.py`, `demo_gradio.py`.

- Preprocessing (`mode="balanced"`, resolution 512) resizes without cropping
  for aspect ratios between 0.5 and 2.0. Computed with the upstream function
  itself: 1920x1200 becomes 640x400, exactly one third in x and in y. The two
  wide cameras (2592x2048) become 576x448, with slightly DIFFERENT x and y
  factors (0.2222 and 0.21875), so intrinsics are always scaled per axis.
- The earlier note of 624x416 applies to 3:2 images, not to AURA.
- Depth is `exp(logits)`: positive, and it is Z-depth in the camera frame (the
  authors' own unprojection multiplies it by the normalised ray). Shape
  (1, S, H, W, 1). Confidence is `1 + exp(logits)`, always above 1.
- Extrinsics are camera_from_world, 3x4, OpenCV axes, world = first camera.
  A camera's position is `-R.T @ t`, not `t`.
- Intrinsics: only the two fields of view are predicted. The principal point
  is ALWAYS set to the image centre. AURA's true principal point is off-centre
  by under 1% of the image size; this is a fixed, known model limitation and
  belongs in the writeup, not in the error budget of the focal length.
- The authors' unprojection uses integer pixel coordinates (no half-pixel
  offset). Ground-truth projection in phase 4 must use the same convention, or
  state the half-pixel difference.
- The forward pass opens `torch.autocast("cuda")`, bf16 where supported,
  otherwise fp16 (a T4 has no bf16). Heads run in fp32. A GPU is mandatory.

Storage decision for predictions: one compressed `.npz` per scene, camera and
checkpoint. Depth float32 (it feeds relative-error metrics; float16's three
significant digits would add error of the size being measured). Confidence
float16 (only ever thresholded). A JSON sidecar records every pin, the GPU,
timing and peak memory.

First scene: `2025-06-13-07-09-37|78`, urban, about 30 km/h, about 4.3 m
between keyframes, camera front_medium, all 40 keyframes in one pass.

## 2026-09-18 Phase 3 results: first forward pass (notebooks/03_first_forward_pass.ipynb)

Scene `2025-06-13-07-09-37|78`, camera front_medium, 40 keyframes at 0.5 s,
166.2 m driven. Checkpoint `vggt_omega_1b_512.pt`, A100 40 GB.

- The model RUNS on Colab's NumPy 2.1 with the `--no-deps` install. That risk
  is closed.
- One pass over 40 frames: 2.98 s, peak GPU 7.75 GB. Checkpoint download 13 s.
  Predictions file 50 MB per scene and camera. Inference cost is negligible.
- Input 640x400 as computed. Depth finite and positive everywhere.
- First extrinsic is the identity to 5e-4, confirming world = first camera.
- Scale reading confirmed by behaviour: path length gives about 92 m per model
  unit, and the median depth then maps to about 17 m, a plausible street
  value. 92 m is also about the mean distance of the scene's points from the
  FIRST camera over a 166 m drive, which is what the training normalisation
  divides by. Consequence: the model's unit depends on sequence length, so
  numbers from passes of different length are not comparable unaligned.

Looked at, not measured:

- Depth maps are coherent: road near, vegetation and buildings in order,
  gaps between buildings far. The sky gets arbitrary depth with low
  confidence. LiDAR never returns from sky, so it cannot enter depth metrics,
  but it would pollute any point cloud built without a confidence filter.
- Confidence is high on the road surface and drops along object outlines and
  thin structures (poles, trunks). Those outlines are exactly where the
  occlusion trap and the moving-object stratum live.
- The predicted camera path has the same shape as ground truth: straight,
  then a right-hand bend.
- Focal length is UNDER-estimated: predicted fx 497, fy 498 against a true
  538 and 565 (minus 8% and minus 12%); field of view 65.6 x 43.8 degrees
  against a true 61.5 x 39.0. The model predicts square pixels; this camera's
  calibration is not square. Predicted fx also wanders from 487 to 510 across
  frames of one physical camera.
- Per-pixel Z-depth can be scored without any intrinsics, so the focal error
  does not contaminate the depth metric. It DOES distort any unprojected point
  cloud, and it may couple with forward translation. Phase 5 should report
  intrinsics error as its own line and keep depth and pose metrics free of it.

Resume design: the GPU is now required only when a scene still needs
predicting, so a resumed session works on a CPU server.

### Resume test passed (2026-09-18)

The A100 server was removed, a fresh CPU server was connected, and notebook 03
was run again from the top. Cell 6 printed `RESUMED`, loaded the predictions
made on the A100 from Drive, and did not download the checkpoint or load the
model. Every number in cell 7 and cell 9 matched the first run. The one visible
difference, confidence max 83.94 against 83.95, is the documented float16
storage of confidence; depth is stored in float32 and matched exactly.
Trap 8 (ephemerality) is covered for the predict stage.

## 2026-09-18 Can the Colab kernel see the laptop's files? (checked at the source)

Source: `googlecolab/colab-vscode` issues #210, #223, #261 and the wiki user
guide, maintainer replies by kevineger.

- No automatic access. Issue #210 states the problem directly: with a Colab
  kernel selected, `pd.read_csv("localfile.csv")` on a local path fails,
  because the code runs on the remote server.
- What exists since release 0.1.6 (2025-12-23): `Upload to Colab` (right-click
  a local file or folder, a manual one-way copy to the server) and
  `Mount Server To Workspace` (the SERVER's files appear in VS Code, the
  opposite direction).
- Automatic mounting or streaming of local folders, and a two-way sync folder
  (#261), were requested and are not available. The maintainer cited blockers.

So code still has to be sent to the runtime. This project does it with the
sync cell, which is the same copy as `Upload to Colab` but automatic, repeatable
and hash-checked. Right-click upload of `src/` remains a valid manual alternative.

## 2026-09-18 Phase 4: ground-truth design, fixed BEFORE any metric

Code: `src/vggt_aura/ground_truth.py`. Notebook: `notebooks/04_ground_truth.ipynb`.

1. Sweep stage: `motion_compensated`. Its timestamp is the sample timestamp,
   cameras fire within 1.4 ms of it, semantic labels follow its point order.
   Motion-compensated points are stored in the SENSOR frame, so each LiDAR
   needs `camera_from_base @ base_from_lidar`. Our transform was checked
   against the SDK's own `project_lidar_to_camera` on its fixture: identical.
2. All LiDARs of a scene are fused. They share the sample timestamp and the
   mounts are rigid, so one static transform per LiDAR suffices.
3. Resolution (trap 4): points are projected directly at the model's input
   size with per-axis scaled intrinsics. No depth map is ever resized. Pixel
   centres sit on integer coordinates, as in the model's own unprojection.
   Aggregation rule: the nearest visible point per pixel is the ground truth.
4. Classes: `noise` and `ego` are dropped, found by NAME in each scene's
   classes.json. `unlabeled` is kept and counted.
5. Points nearer than 1.0 m are dropped (ego vehicle, sensor artefacts).
6. Occlusion (trap 1): the "two_sided" rule. A point is dropped only if
   much-nearer points lie on both sides of it along its own pixel column
   (above and below) or its own pixel row (left and right). "Much nearer" is
   nearer by more than max(0.5 m, 10% of depth). Search radius 6 px at the
   model's input size.

Why not the usual rule ("drop a point if anything much nearer is within a
small window")? On a road near the horizon, depth changes by 25% or more
across two pixel rows. Rehearsed on a synthetic road with nothing hiding it:
the window rule deleted 69% of all road points; two_sided deleted 0%.

A bug found and fixed during rehearsal, recorded because it is instructive:
the first two_sided version searched strips 3 px wide. The left/right strip
then contained the row BELOW the point, where a road is nearer on both sides,
and 67 to 87% of road at 20 to 40 m was deleted. The unit test had missed it
because its synthetic road only had points on every second row. Fix: search
lines exactly one pixel wide (`half_width = 0`). The road tests are now dense
and parametrised, one test reproduces the bug on purpose, and one covers road
camber and camera roll.

Known limits of two_sided, accepted and stated:
- A leak within one LiDAR ring of an object's outline edge can survive.
  Phase 5 can report metrics with and without a boundary band.
- A far road point with a nearer object just above it (overhanging canopy
  within 6 px) is wrongly dropped. This loses ground truth, it does not
  corrupt it.
- The rule has never seen real data. Notebook 04's Figure B and its per-class,
  per-depth drop tables are the acceptance test. Phase 5 should also report
  the headline numbers under `none` and `two_sided` to show the sensitivity.

Still open, answered by running notebook 04: whether the JPGs are undistorted
(Figures A1, A2), whether PCDs carry per-point times (cell 5, trap 3), and the
real ground-truth coverage per frame.

## 2026-09-18 Phase 4 results: ground truth on real data (notebooks/04_ground_truth.ipynb)

Scene `2025-06-13-07-09-37|78`, camera front_medium, CPU server.

LiDAR facts:
- 6 sensors, 57k to 69k points each per sweep, 128 rings (`ring` 0 to 127).
  Fields: `x, y, z, t, reflectivity, ring, range`. `range` looks like
  millimetres (880 to 140400).
- Per-point time EXISTS: field `t`, uint32, 0 to about 6.8e7 in the sweep
  read. If nanoseconds, that is 0 to 68 ms, consistent with a 10 Hz sweep.
  Units and zero point (sweep start or end) are NOT yet verified. This is what
  trap 3 needs: a moving object's displacement can be bounded, or corrected,
  per point as object velocity x (t_point - t_reference). Decide in phase 5.
- Mounts: front and rear corner LiDARs at about 1.45 m height, top pair at
  1.71 m; camera at (1.68, 0.19, 1.49). Baselines to the camera reach about
  0.4 m sideways and 0.25 m vertically for the top pair, more for corners.

Occlusion rule, acceptance test on 4 frames (193,817 candidate points):

| rule | dropped by the rule | road dropped | image coverage |
|---|---|---|---|
| none | 0% | 0% | 16.6% |
| window | 14.2% | 9.1% | 14.2% |
| two_sided | 2.7% | 0.0% | 16.1% |

- two_sided drops nothing on `road` (70,688 points) and 0.7% on `parking`.
  Its drops concentrate where occlusion is expected: building 7.1%,
  caravan 8.3%, static 8.6%, vegetation 4.7%, car 3.1%, pole 3.1%.
- window loses 26 to 36% of all points beyond 20 m. In Figure B its drops form
  a band across the far road and horizon. two_sided's drops sit in small
  clusters at object outlines, at the fence on the right and around poles.
- Verdict: two_sided ACCEPTED as the ground-truth rule. Fences are
  see-through, so background seen through one is dropped; that is the
  conservative choice, since such pixels are mixed at model resolution.

Calibration and distortion (Figures A1, A2), judged by eye:
- front_medium: LiDAR points sit on the trees, the fence, the parked car, the
  lamp post and the roof line, including at the left and right image edges.
  No drift toward the borders is visible.
- front_wide: straight building edges are straight in the image and points
  sit on their objects at both edges. The JPGs behave as undistorted pinhole
  images, which matches a calibration that ships P but no coefficients.
- Limit of this check: the figures show about 2 native pixels per figure
  pixel, so misalignment under roughly 5 native pixels (under 2 pixels at the
  model's input size) cannot be excluded. That is below what matters here.
- NEW TRAP for the wide cameras only: front_wide sees the ego vehicle's
  bonnet, and LiDAR road points project onto it. The camera cannot see that
  road. No LiDAR point marks the bonnet (the `ego` class is dropped), so the
  occlusion rule cannot catch it. If front_wide or rear_wide are ever
  evaluated, they need a fixed bonnet mask. front_medium does not see the
  bonnet and is unaffected.

Ground truth saved for all 40 keyframes: 7.6 MB. Median 42,585 pixels with
truth per frame, 16.6% of the 640x400 image. Depth p5 4.8 m, median 13.3 m,
p95 58.1 m, max 260 m. Class share: vegetation 38%, road 37%, building 8%,
sidewalk 8%, car 2%. Cars are a THIN stratum in this scene and mostly parked;
the moving-object analysis will need many scenes (trap 7).

Timing, first real measurement:
- Download and extract of camera + LiDAR layers: 7.5 min for 8.75 GB. Camera
  and base alone took 1.4 min for 4.9 GB, so the xz-compressed LiDAR layer
  costs about 6 min for 3.84 GB. Extraction, not bandwidth, dominates.
- Building ground truth for 40 frames (240 PCD files plus labels, projection,
  occlusion): 6 s. PCD reading is NOT a bottleneck.

## 2026-09-18 Phase 5: evaluation protocol, fixed BEFORE the first scored run

Code: `metrics.py`, `objects.py`, `evaluation.py`. Notebooks: `05a_predict_block`
(GPU, predicts only, reveals nothing) and `05b_metrics` (CPU, gated on
`PREDICTIONS_REGISTERED`).

Depth (trap 5). Ground truth used where 1 m <= depth <= 80 m. Metrics: AbsRel,
delta < 1.25, RMSE. Five alignments, always reported under these names:

| name | what is fitted | why it is there |
|---|---|---|
| sequence_scale | one median scale per sequence | PRIMARY. Grants only the one freedom the model lacks |
| frame_scale | one median scale per frame | shows how much is scale drift |
| frame_scale_shift | median scale and shift per frame | the authors' own protocol, for comparability |
| pose_scale | scale taken from the camera path | do depth and pose agree on one scale? |
| unaligned | nothing | meaningless by construction; reported because the project rule says both |

- Scales are fitted on all valid pixels, moving objects included (about 2% of
  pixels cannot move a median).
- frame_scale_shift is NOT guaranteed to score best. It is median-based, not a
  best fit; a test shows it scoring worse than one scale when 25% of pixels are
  corrupted. So "more freedom" must not be read as "lower error" by default.

Pose. Predicted extrinsics are inverted to camera0_from_camera, the form of the
ground truth (ego pose composed with the rigid camera mount).
- Pairwise relative rotation error and translation-DIRECTION error over all
  frame pairs; AUC at 3 and 30 degrees computed exactly as the authors do.
- The authors' code takes min(angle, 180 - angle) for translation, so it cannot
  tell driving forward from driving backward. We report the true 0 to 180
  angle as primary and their variant as `auc30_authors_unsigned`.
- ATE of camera centres two ways. `scale_only`: both paths start at the origin
  with the same axes, so scale is the only freedom granted. `sim3`: the
  customary similarity alignment, which may also rotate the path and thereby
  HIDE a heading error (a test demonstrates exactly that).
- Near-stationary rule, decided here as promised in phase 2: frame pairs whose
  true baseline is under 0.5 m get no translation score, and scenes whose true
  path is under 10 m are kept out of the pose table. Their depth still counts.

Intrinsics: relative fx and fy error against the per-axis scaled calibration,
reported on its own line. Per-pixel Z-depth needs no intrinsics, so the known
focal error cannot leak into the depth metric.

Moving vs static (the headline stratum). Each ground-truth pixel keeps its 3D
LiDAR point; the point is tested against the frame's 3D boxes in base_link
(10 cm margin). A box's speed comes from its own object_id track, with
positions compared in odom so ego motion cancels. MOVING is 1.0 m/s or more,
PARKED is under 0.5 m/s, and everything between, or without a usable track
(neighbour keyframe within 2 s), is UNCERTAIN and kept out of BOTH strata,
because annotation jitter alone reads as 0.1 to 0.3 m/s.

Trap 3, as a number. For every moving pixel we carry
|object velocity along camera Z| x 0.1 s, the most the LiDAR depth of that
object can be off because motion compensation corrects ego motion only. It is
reported as a share of depth next to the moving-object result. The per-point
time field `t` could tighten this bound; its units are still unverified, so the
conservative full-sweep bound is used.

Aggregation (trap 7). The unit is the SCENE. Every mean and every 95% bootstrap
interval is over scenes. A stratum counts for a scene only with 200 or more
pixels there. A row supported by fewer than 5 scenes is flagged `thin`. Strata
are compared by PAIRED differences within scenes, not by comparing two means.
Scene-level strata: road_type, weather_group (wet or dry), lighting (from the
dataset's sunrise and sunset, "unknown" unless they look like Unix seconds),
speed_band. With one block (18 urban, 2 overland, 3 wet, all daytime) most of
these WILL be thin; that is expected and is what phase 7 is for.

Sensitivity: every scene is also scored with occlusion handling off, and the
two are shown side by side.

Bugs found by rehearsal before any real run: a scene with no ground-truth
pixels crashed the scorer (fixed, tested). Three tests of mine demanded more
precision than float32 storage gives (loosened to 1e-6).

Comparison arm (open decision, raised now as planned). Checked at the source:
the original `facebook/VGGT-1B` checkpoint is CC BY-NC 4.0 and not gated;
`VGGT-1B-Commercial` is gated under its own licence. Either is usable for
non-commercial research. Not built; the project owner decides.

## 2026-09-18 Comparison arm added: VGGT, the predecessor (project owner's decision)

DECISION: evaluate `facebook/VGGT-1B` on the same scenes, so the result is two
checkpoints on the same post-release data. Licence checked at the source:
CC BY-NC 4.0, not gated. Pins: code `a288dd0f`, weights `860abec7`.

Read from the VGGT source at that commit:
- Outputs have the same form as VGGT-Omega: depth = exp(.), confidence =
  1 + exp(.), extrinsics camera_from_world in OpenCV axes with world = first
  camera, intrinsics with the principal point fixed at the image centre.
  Training also normalises scenes to unit average point distance, so depth is
  scale-normalised. Ground truth, metrics and tables are therefore shared.
- Preprocessing (default "crop" mode) sets width to 518 and height to the
  nearest multiple of 14 that keeps the aspect: 1920x1200 becomes 518x322,
  uncropped, with slightly different x and y factors (0.26979, 0.26833).
- The caller must open the autocast context; VGGT-Omega opens its own.
- It pins `numpy<2` like VGGT-Omega, so it is installed with `--no-deps`.
  Whether it RUNS on NumPy 2 is unverified until notebook 05c runs.

Consequences and rules:
- Each model is scored on ground truth built at its OWN input size (640x400
  and 518x322). No depth map is resized, as everywhere else. The pixel sets
  differ slightly; scenes, LiDAR points and rules are identical. VGGT's ground
  truth is stored under a `518x322` tag so the two never overwrite each other.
- The comparison is PAIRED by scene (`evaluation.compare_models`): per-scene
  difference VGGT-Omega minus VGGT, mean and bootstrap interval over scenes.
- This does NOT test the paper's scaling claim. The two checkpoints differ in
  data, recipe, resolution and architecture details at once.
- Predictions about the comparison are registered in
  `predictions_vggt_comparison.md` before notebook 05d is run.

Process note: notebook 05a was running while this was built. The usual
`python -m vggt_aura.sync` rewrites a cell in every notebook, so it was NOT
used; only the two new notebooks were synced. 05a, 05b and predictions.md were
left byte-for-byte alone (05b and predictions.md timestamps verified unchanged;
05a changed only through the owner's own run and kept its original code hash).
Consequence: 05a and 05b carry code hash `c9c6913865ce`, the new notebooks
`a9c9cd9bbae8`. Both are self-consistent; the later code only ADDS functions
and an optional argument.

## 2026-09-18 Phase 5 results: VGGT-Omega on val block 11 (notebooks/05b_metrics.ipynb)

20 scenes, camera front_medium, 40 keyframes each, 37.5 M evaluated pixels,
depth 1 to 80 m. Means over scenes, 95% bootstrap intervals over scenes.
Predictions were registered first (predictions.md). ONE block: 18 urban, 2
overland, 3 wet, all daytime. Nothing here generalises beyond that yet.

Depth, AbsRel:
| protocol | AbsRel | interval |
|---|---|---|
| sequence_scale (primary) | 0.0542 | 0.0475 to 0.0610 |
| frame_scale | 0.0494 | 0.0426 to 0.0566 |
| frame_scale_shift (authors') | 0.0521 | 0.0443 to 0.0602 |
| pose_scale | 0.0562 | 0.0496 to 0.0630 |
| unaligned | 0.98 | meaningless by construction |

- The strict protocol costs almost nothing: the authors' per-frame scale and
  shift is only 4% lower. Scale drift within a 20 s sequence is small. Their
  protocol is also NOT the most forgiving; plain per-frame scale beats it.
- delta < 1.25 is 0.969.

Moving vs static (the headline stratum), sequence_scale:
| stratum | scenes | AbsRel |
|---|---|---|
| background | 20 | 0.0499 |
| parked object | 18 | 0.0836 |
| moving object | 17 | 0.1754 |
Paired within scenes: moving minus background +0.123 (0.069 to 0.201), worse
in 16 of 17 scenes. Parked minus background +0.035 (0.013 to 0.065), worse in
16 of 18. Moving minus parked +0.083 (0.011 to 0.177), worse in 12 of 15.
- Trap 3 bound on the moving-object ground truth: median 2.5% of depth, 90th
  percentile 5.0%. The 12-point gap is several times larger, so ground-truth
  timing error cannot explain it.
- CAUTION, not yet controlled: moving objects differ from parked ones in more
  than motion. They include pedestrians (the `human` group scores 0.44), and
  they sit at other distances and image positions. The result is CONSISTENT
  with the static-scene assumption failing on moving objects; it does not
  prove it. Phase 7 should compare moving and parked VEHICLES only, matched by
  depth band.

Other depth strata:
- Depth bands 0.049, 0.051, 0.065, 0.074 from near to far. Far is only 1.5x near.
- Semantic groups: flat 0.028, structure 0.051, vegetation 0.086, vehicle
  0.105, thin 0.294, human 0.439 (12 scenes, interval 0.20 to 0.86, very wide).
- Confidence is strongly informative: least-sure quarter 0.129, most-sure
  0.022, a factor of 5.8, monotone across quarters.
- Worst scene `2025-06-11-12-27-55|207` (0.089). In its worst frame the error
  sits on a traffic sign, lamp posts and tree crowns; the road is near zero.

Pose (no scene was near-stationary by the 10 m rule): median rotation error
0.55 deg, translation direction 0.80 deg, AUC@3 84.0, AUC@30 98.2, ATE after
scale-only alignment 1.04% of path (1.68 m), after Sim(3) 0.54 m. The signed
and the authors' unsigned AUC@30 are identical: no pair was ever backwards.

Scale agreement: pose scale over depth scale = 1.0006 (0.995 to 1.006).

Intrinsics: fx -3.3% (-6.1 to -1.0), fy -7.8% (-10.5 to -5.5) on average, but
NOT uniform: from +5% in some urban scenes to -15% and -20% on the two overland
scenes, which also have the lowest AUC@30 (89.7) and the highest ATE (3.9%).
n = 2, so a lead for phase 7, not a finding.

Scene-level strata: urban 0.053 (18), overland 0.068 (2, thin); dry 0.056 (17),
wet 0.045 (3, thin); under 5 km/h 0.049 (4, thin) against 0.053 at 20 to 50 km/h.
Sunrise and sunset ARE Unix seconds, so the lighting rule works; all 20 are day.

Occlusion rule sensitivity: without occlusion handling background AbsRel is
0.0575 instead of 0.0499. Skipping it would overstate the background error by
about 15%. Object strata move by 5 to 9%.

Timing: median 51 s per scene on a Colab CPU (ground truth built twice, for
the sensitivity check). 17 min for the block, after 7.5 min of download.

Predictions scored (predictions.md):
- Both forecasters were too pessimistic on the headline (0.15 and 0.10
  against 0.054; Claude's 80% range 0.06 to 0.16 missed it).
- Abhishek right, Claude wrong: the authors' protocol gains only "a little"
  (Claude said 20 to 40%).
- Claude right, Abhishek wrong or blank: confidence is informative; flat is the
  best group; pose numbers; scale ratio within 0.95 to 1.05.
- Both right in direction: moving worse than background (but it is 3.5x, more
  than either said, and 17 scenes qualified, not "fewer than 10"); near-stationary
  scenes no worse; motion costs more than the object itself.
- Both wrong: far vs near is 1.5x, not 2x. Worst scene is dry and urban, not
  rainy; the single BEST scene is a wet one. Worst group is `human`, not `thin`
  (Claude) or roads (Abhishek). Claude's "focal too short in every scene" is
  wrong: it varies by scene and is sometimes too long. Claude's occlusion
  sensitivity bound (under 0.005) was exceeded (0.0076).

## 2026-09-18 Comparison results: VGGT-Omega against VGGT (notebooks 05c, 05d)

Same 20 scenes, frames, LiDAR points and rules. VGGT-Omega at 640x400, VGGT at
518x322, each on ground truth built at its own size. VGGT ran on NumPy 2
without trouble (1.8 s per scene, 9.75 GB peak). Paired by scene; difference =
VGGT-Omega minus VGGT. Predictions were registered first.

| quantity | VGGT-Omega | VGGT | difference (95% interval) | Omega better in |
|---|---|---|---|---|
| AbsRel, all pixels | 0.0542 | 0.0762 | -0.022 (-0.032 to -0.014) | 18 of 20 |
| AbsRel, background | 0.0499 | 0.0667 | -0.017 (-0.024 to -0.011) | 19 of 20 |
| AbsRel, parked objects | 0.0836 | 0.1570 | -0.073 (-0.139 to -0.024) | 17 of 18 |
| AbsRel, moving objects | 0.1754 | 0.2245 | -0.049 (-0.125 to +0.009) | 8 of 17 |
| rotation error, deg | 0.55 | 0.70 | -0.14 (-0.26 to -0.004) | 17 of 20 |
| translation direction, deg | 0.80 | 0.79 | +0.01 (-0.11 to +0.11) | 8 of 20 |
| AUC@30 | 98.2 | 96.5 | +1.66 (+0.83 to +2.61) | 18 of 20 |
| ATE, % of path | 1.04 | 1.39 | -0.35 (-0.60 to -0.10) | 14 of 20 |
| fx error | -3.3% | -7.7% | | |
| fy error | -7.8% | -11.4% | | |
| pose scale / depth scale | 1.0006 | 1.031 | -0.030 (-0.057 to -0.008) | |

What this block supports:
- VGGT-Omega's depth is better overall by about 29% relative, in 18 of 20
  scenes, interval well clear of zero.
- The gain is largest on PARKED objects (error roughly halved) and clear on
  background. On MOVING objects this block cannot separate the models: the
  interval includes zero and VGGT-Omega is lower in only 8 of 17 scenes. Moving
  objects stay the hard case for BOTH models (0.18 and 0.22). Motion costs
  extra in both: moving minus parked is +0.083 (Omega) and +0.103 (VGGT).
- VGGT-Omega's depth and camera translation agree on one scale (1.0006); VGGT's
  differ by 3%, up to 10 to 19% in single scenes. Consistently, VGGT gains 17%
  from the authors' per-frame scale-and-shift, VGGT-Omega only 4%.
- Rotation and AUC@30 favour VGGT-Omega modestly; translation direction is a tie.
- Both models predict a focal length that is too short; VGGT more so and in
  every scene, VGGT-Omega less and not uniformly.
- VGGT-Omega's confidence is monotone across quarters; VGGT's most-sure quarter
  (0.046) is slightly WORSE than its second (0.042).

One result against the trend, flagged not claimed: on THIN objects (poles,
signs) VGGT scores 0.176 and VGGT-Omega 0.294, intervals not overlapping.
Confounded by input size: the two ground truths contain different pixels for
thin structures, and a pole covers fewer pixels at 518x322. A fair test needs
both models scored on the same LiDAR points. Lead for phase 7.

VGGT's worst scene (`2025-06-11-13-47-25|24`, 0.142 against VGGT-Omega's 0.067)
follows a car closely. In its worst frame the car itself is about right, while
buildings are predicted too near and the roadside too far, and its pose and
depth scales differ by 10%. One dominant moving object appears to disturb
VGGT's estimate of the static scene; VGGT-Omega is not affected in the same
scene. One scene, so an observation, not a finding.

NOT shown by any of this: the paper's scaling claim. The checkpoints differ in
data, recipe, input size and architecture details at once.

Predictions scored (predictions_vggt_comparison.md):
- Claude right: gap 0.01 to 0.04 with interval excluding zero (0.022); Omega
  lower in 15 or more scenes (18); moving-object gap larger in mean but its
  interval includes zero; VGGT's focal length also too short.
- Claude wrong: AUC@30 gain 2 to 8 points (1.66); focal errors within 3 points
  of each other (4.4 and 3.7 apart).
- Abhishek right: VGGT-Omega better overall and on pose. Wrong: VGGT at about 6%
  (7.6%); VGGT focal "probably not" too short (it is, more so); Omega lower in
  about 12 scenes (18).

## 2026-09-18 Phase 6: the C++ geometry core

Files: `cpp/include/vggt_geom/geometry.hpp` (header-only core), `cpp/src/bindings.cpp`
(pybind11, the only file that knows Python), `cpp/CMakeLists.txt`,
`src/vggt_aura/cpp_build.py`, `tests/test_cpp_differential.py`,
`notebooks/06_cpp_core.ipynb`.

- Dependencies: NONE beyond pybind11 for the binding. No Eigen: it is not on the
  Colab image and the functions in scope are small loops. The plan's rule
  ("if a build needs PCL or OpenCV, stop") was never approached.
- Scope held exactly: quaternion to rotation, composition and inversion of rigid
  transforms, point transform, projection, unprojection, pixel indices, the
  occlusion check (all three modes), nearest-per-pixel, and the whole
  LiDAR-to-image step. No file I/O, no point-cloud parsing, no CUDA.
- `unproject_pinhole` was missing from the Python reference although it is in
  the plan's scope; it was added there first, with a round-trip test, and then
  mirrored.
- Differential tests (26): discrete outputs must match EXACTLY, real numbers to
  1e-12. They cover un-normalised quaternions, points at and behind the camera
  plane, pixel-edge rounding, NaN and inf, exact depth ties, 15 occlusion
  parameter combinations, the three hand-built occlusion scenes, the full
  pipeline with a class filter, and rejection of bad input.
- Built with `-ffp-contract=off`, so `a*b+c` is never fused into one rounded
  operation. NumPy does not fuse either, which is what makes exact agreement
  of the discrete outputs achievable.
- Verified LOCALLY before hand-over, although the laptop has no compiler: a
  compiler from a pip package (`ziglang`), installed only in the scratch
  environment, built the real module against the laptop's Python 3.13. All 26
  differential tests pass on that build, and the suite skips cleanly when the
  module is absent (124 passed, 26 skipped). The g++/CMake build on Colab is
  still unverified until notebook 06 runs; `cpp_build` falls back to a direct
  compiler command if CMake cannot find the Python development files.
- Speed, measured locally on a 370k-point synthetic frame: whole step 108 ms in
  Python, 43 ms in C++, 2.5x. Per stage 1.8x to 4.1x. Modest, because the
  Python reference is already vectorised NumPy. At about 5 s per scene the
  Python step was never the bottleneck, so the C++ core is a verification
  exercise and a second opinion on the geometry, NOT a needed optimisation.
- The sync payload now includes `tests/`, so the differential suite runs on the
  runtime. `cpp/build` is never packed.
- Result notebooks (02 to 05d) were deliberately NOT re-synced: each keeps the
  code hash that produced its results.

## 2026-09-18 Phase 6 results on Colab (notebooks/06_cpp_core.ipynb)

- CMake build on the runtime: succeeded first time in 15.6 s (g++ 13.3, CMake
  3.31). CMake found the Python development files; the fallback was not needed.
- Differential suite on the runtime: 26 of 26 passed.
- Real data, all 40 frames of scene `2025-06-13-07-09-37|78` (about 370,000
  points per frame): candidates, occlusion verdicts and chosen pixels are
  IDENTICAL between C++ and Python in every frame. Largest depth difference
  1.5e-5 m, which is the float32 storage of the reference.
- Speed, honestly: the C++ core is NOT meaningfully faster here. Whole scene
  3.28 s in Python against 2.40 s in C++ (1.4x). On the last real frame the C++
  occlusion check was SLOWER than NumPy (51 ms against 31 ms): its plain loops
  over every pixel are not vectorised, NumPy's shifted-array minima are. The
  transform and projection stages are 5x to 8x faster in C++, but they are a
  small part of the step. The C++ core stays what it was meant to be: a second,
  independently written implementation that agrees exactly. It is not used in
  the pipeline and no optimisation is planned, because:

Where one scene's time goes (the measurement asked for before phase 7):

| step | seconds | share |
|---|---|---|
| download and unpack, per-scene share (11.6 min for the block this time) | 34.9 | 66% |
| score the scene, 5 protocols and all strata | 8.0 | 15% |
| moving-object labels from boxes | 3.4 | 7% |
| model inference (A100) | 3.0 | 6% |
| project and occlusion, Python | 2.4 | 5% |
| read 240 LiDAR files | 0.8 | 1.5% |
| read 240 label files | 0.2 | 0.4% |

53 s per scene end to end: 100 scenes 1.5 h, 300 scenes 4.4 h, 1000 scenes 14.6 h.

Conclusions:
- Reading point-cloud files is 1.5% of the time. A C++ PCD reader would break
  the "never parses a PCD" rule and the zero-dependency rule to save under a
  second per scene. NOT proposed. The narrow C++ scope stands.
- Two thirds of the cost is getting a block onto the disk, and it varies
  (7.5 min and 11.6 min for the same block on two days). So the lever is how
  OFTEN a block is fetched, not how fast code runs.

## 2026-09-18 Phase 7 design (approved by the project owner)

1. A block is downloaded ONCE in its life (`pipeline.process_block`): camera +
   LiDAR down, every model run, ground truth built and saved, every scene
   scored, results written, block deleted. Disk use stays at one block. Until
   now the development block was downloaded five times, once per phase.
2. Ground truth is built once and reused (`load_or_build_scene_truth`). A saved
   file is reused only if its occlusion parameters, input size, sensor list and
   frame timestamps all match the request; otherwise it is rebuilt, and if no
   LiDAR is on disk the call fails loudly instead of returning empty ground
   truth. The intended sensor list comes from the calibration file, which is
   always on disk, not from what happens to be downloaded.
3. Reports are built from saved result files only (`07c_report`): no dataset
   download, seconds to run. One pair of CSV files per block and model; a block
   counts as done only when both files of every model exist. A scene appearing
   in two block files is an error, never a silent double count.
4. Census first (`07a_census`): the small metadata layer of every public block,
   giving road type, weather, lighting, speed and the sensor set of all scenes,
   so blocks are CHOSEN to fill thin cells (motorway, wet, not-day) instead of
   taken in order.
5. Drive cache for the development block (`07d_cache_speed_test`): a one-off
   measurement. Adopted only if reading one big file back from Drive clearly
   beats the normal download.
6. Drive has about 3 TB, so full predictions and ground truth stay saved (tens
   of GB at most). The raw dataset is never mirrored to Drive.

Learnings from the dataset card and SDK README, read in full on 2026-09-18, and
how phase 7 uses them:
- The 12-LiDAR recordings add six Aeva FMCW sensors to the six Ouster ones.
  Mixing them would make ground-truth density differ between 2025 and 2026
  recordings. Rule: `LIDAR_POLICY = "ouster_only"` everywhere, the default. The
  census records `n_lidars` and `has_aeva` per scene.
- Aeva sensors measure per-point radial velocity. It is a possible independent
  check of the moving-object labels. NOT built: the field's sign, units and
  whether ego motion is already removed are unverified. First step, once the
  census names a block with Aeva sensors, is to inspect that field on static
  background against moving boxes.
- The toolkit's validator (`fzi_aura.cli.validate`) runs on every newly
  downloaded block (`VALIDATE_DOWNLOAD = True`); its verdict is printed and
  kept in the block summary.
- Attribution for OpenStreetMap and OpenWeather metadata, and the dataset and
  model citations, are now in the README, worded after the dataset's own
  THIRD_PARTY_NOTICES.MD.
- Unresolved: the card names the corner sensors as 64-channel units, yet phase 4
  saw ring numbers up to 127 on `front_left`.

## 2026-09-18 Phase 7 pipeline verified against phase 5 (notebooks 07b, 07c)

Smoke test on the development block, CPU server.
- The pipeline found every prediction and ground truth on Drive, so it fetched
  the camera layer only: 166 s instead of 8 to 12 min. All 40 scene-model
  pairs reported "loaded, truth loaded". No LiDAR was downloaded.
- The dataset toolkit's validator ran for the first time: ok, 20 scenes, no errors.
- `07c_report`, built from the saved result files alone, reproduces phase 5
  EXACTLY, to four decimals, in every table checked: headline AbsRel 0.0542 and
  0.0762, all five protocols, the motion strata, the paired differences, and
  the whole model-against-model table (18 of 20 scenes). The new pipeline is a
  faithful rewrite of notebooks 05b and 05d.
- Cost: about 50 s per scene for two models, so about 17 min of CPU per block,
  more than the 10 min estimated. Whole block 20.4 min without LiDAR.
- Sensor note recorded per block: six Ouster sensors, no Aeva, as expected for
  2025 recordings.

Two-pass option added (`07b0_predict_blocks_gpu`, `pipeline.predict_block`):
GPU server for predictions only, from the camera layer (about 3 to 4 min per
block), then `07b_blocks` on a CPU server with the same BLOCKS list. For 15
blocks that is under an hour of GPU instead of 5 to 6 hours.

## 2026-09-18 Census, first pass, and a bug it exposed

Census of val and test complete (12 blocks, 214 scenes): urban 175, overland 36,
motorway 3; wet 61; night 9 and twilight 5, ALL in val (test has no scene that
is not day); over 50 km/h 23; Aeva sensors in 103 scenes. Night scenes are all
urban and wet, twilight scenes all overland and wet: lighting is confounded with
weather in this data. `train_block000010` is the first block with real motorway
content (3 scenes). The train census stopped at 33 of 56 blocks on the bug below.

BUG (mine, fixed): the release's block table lists 8 scenes that the dataset's
maintainers EXCLUDE (`consumer_excluded_scenes` in dataset.json: "PTP
synchronization issue; objects doubled", "Missing keyframes", "Severe repeated
LiDAR dropout", "Too many dropped frames", Aeva sync faults). The SDK refuses to
open them, so the census crashed with SceneNotFoundError at train block 77. All
8 sit in train blocks 77, 84, 98 and 99, which is why val and test ran clean.
Fix: the excluded list is fetched with the release tables and removed from every
block's scene list (ids and folder names together), and the pipeline filters
again after opening the dataset. Checked live: block 77 goes from 20 to 16
scenes, 84 to 19, 98 to 18, 99 to 6; val 11 is unchanged. This is also the right
behaviour for the benchmark: scenes with doubled objects or dropped frames must
not enter a ground-truth comparison.

Drive cache speed test (`07d`): normal download and unpack 26.1 min this time
(7.5 and 11.6 min on earlier runs: the variance is large), packing to one 13 GB
file on Drive 12.5 min, reading it back 4.5 min. Verdict printed: cache wins
5.8x. Two caveats: the baseline was unusually slow, and the read-back happened in
the session that wrote the file, which may have been served from Colab's local
copy. Against a typical 8 to 12 min download the gain is about 2x, unconfirmed
until measured from a fresh session. Since the pipeline now fetches only the
camera layer (under 3 min) for a block whose results are saved, the cache is a
convenience, not a necessity. Not adopted yet.

## 2026-09-18 Census fix verified on the real block; parallel scoring added

- The excluded-scene fix was run end to end on the laptop against the REAL
  train block 77 (its 1.17 GB metadata layer): 20 scenes listed, 16 after the
  exclusion list, census finished without error, table saved with the new
  columns, temporary download removed. Passing all 20 ids, as the old code did,
  no longer crashes either: the 4 excluded scenes are reported and skipped.
- The new `n_lidars_ouster` column already earns its place: block 77 contains
  scenes with only FOUR Ouster sensors, not six. Under the ouster_only rule
  those scenes get thinner ground truth than the rest. To decide when choosing
  blocks: exclude four-Ouster scenes, or report them separately.
- Block 77 also holds twilight and night scenes from a January 2026 recording,
  all wet, with 10 sensors (6 Ouster + 4 Aeva).
- Scoring now runs scenes in parallel (`pipeline.run_jobs`, `score_scene`).
  Phase A, one scene at a time in the main process: make sure predictions are
  saved (the only part that can need a GPU). Phase B, in worker processes:
  ground truth, moving-object labels and scores. Workers are started with
  "spawn", which is safe next to a process that has used CUDA. Results are
  collected in job order, so output is identical whatever finishes first; a
  failing scene raises in the caller with the scene named. Default: one worker
  per CPU core, at most 8. `notebooks/07e_verify_parallel.ipynb` re-scores the
  development block in parallel and requires every number to be IDENTICAL to
  the one-at-a-time run. Not yet run on Colab.
- Measured facts behind this: a block's metadata and camera layers are always
  ONE file, LiDAR is 1, 2 or 3 files, so raising the downloader's worker count
  above 3 does nothing, and one compressed file unpacks on one core. Scoring
  (about 17 min per block, single core) was the part that could use more cores.
- `07a_census` gained `REVERSE`: a second copy on a second server can count from
  the far end. All notebooks were re-synced in one go while nothing was running.

## 2026-09-18 Parallel scoring verified on Colab (notebooks/07e_verify_parallel.ipynb)

High-RAM CPU server, 8 cores, development block, both models.
- IDENTICAL: all 1,940 and 1,935 score rows and all 40 scene records equal the
  one-at-a-time run exactly (DataFrame equality, no tolerance).
- Scoring 216.5 s with 8 workers against about 1,040 s one at a time: 4.8x.
  Whole block 6.6 min against 20.4 min. Each scene takes longer inside a worker
  (about 73 s against 50 s) because 8 processes share Drive and memory bandwidth.
- Found by that run and fixed: phase A spent 137 s loading every full
  prediction file from Drive only to confirm it existed. It now reads just the
  frame-time array inside each file (`predictions_are_saved`), a few hundred
  bytes. Expected saving about 2 min per block; not yet re-measured.
- High-RAM is now worth choosing for `07b_blocks`. It is NOT useful for the
  census, which scores nothing and unpacks one file on one core.

## 2026-09-18 Census complete: 56 blocks, 1,073 usable scenes

1,073 = the 1,081 published scenes minus the 8 the maintainers exclude. The
second run counted the remaining 23 blocks at about 28 s each; train block 77
gave 16 scenes, as the local check predicted.

| | motorway | overland | urban | wet | night | twilight | 4-Ouster scenes |
|---|---|---|---|---|---|---|---|
| val (107) | 2 | 22 | 83 | 40 | 9 | 5 | 0 |
| test (107) | 1 | 14 | 92 | 21 | 0 | 0 | 0 |
| train (859) | 76 | 197 | 586 | 222 | 56 | 37 | 55 |

What the census says about the promised strata:
- Road type is feasible: 79 motorway scenes exist, nearly all in train blocks.
- Weather is feasible: 283 wet scenes.
- Lighting is feasible only TOGETHER with weather: of 107 night or twilight
  scenes, 91 are also wet. Dry night is 13 scenes in the whole release. The
  writeup must not present a lighting effect as separate from rain.
- The darkest, wettest material is concentrated: train blocks 11, 12 and 13 are
  20 of 20 wet and 20 of 20 not-day. They very likely come from one drive, so
  scenes there are not independent. Report the number of distinct recordings
  behind every stratum, not only the number of scenes.
- The test split has no motorway to speak of (1 scene) and no night at all, so a
  test-only final result cannot cover the conditions this project is about.
- 55 train scenes have only four Ouster sensors. Under the six-Ouster rule their
  ground truth would be thinner. Proposed: leave them out.

Block list proposed to the project owner (not yet approved): val 0, 1, 2, 10, 12
(with val 11 already done); train 11, 12, 13 for dark and wet; train 92, 82, 83,
90 for motorway; all six test blocks as the untouched final set. About 350
scenes, 18 new blocks, about 190 GB streamed through, never stored.

## 2026-09-18 The C++ core now runs in the pipeline (project owner's decision)

Decision: use C++ wherever it is measurably faster, high-RAM CPU servers for
scoring, and the GPU only for the model runs. The project is a mix of both
languages in production, not only in tests.

- `build_ground_truth(..., backend=)` takes "python" (the reference, and the
  default of that function), "cpp", or "auto" (C++ when the module is built).
  `build_scene_truth`, which the pipeline calls, uses "auto". A saved ground
  truth records `built_with`, and the run log prints "truth built (cpp)".
- What runs in C++: projection, pixel rounding, the occlusion check, the
  nearest-per-pixel choice, and the count of in-image points. What stays in
  Python: reading LiDAR files and labels (the C++ core never opens a file),
  class filtering, moving-object labels, scoring.
- Why my first C++ occlusion check was SLOWER than NumPy on real frames
  (51 ms against 31 ms): it built four full-image arrays of side minima with
  per-pixel loops. With one-pixel-wide search lines, "the side minimum is below
  the threshold" equals "some pixel on that side is below it", so the new fast
  path lets each point scan its own row and column and stop at the first hit.
  The general path remains for wider strips; both are tested against Python.
- Exactness: the backend test compares the FULL output of the two backends,
  every array and every count, with exact equality, for all three occlusion
  modes. 182 tests pass with the module, 148 pass and 34 skip without it.
- Measured on the laptop, 370,000-point frame: occlusion check 75 ms to 25 ms
  (3.0x), whole ground-truth step 113 ms to 51 ms (2.2x). Not yet re-measured
  on Colab; notebook 06 will show it when re-run.
- Honest size of the gain: ground truth is built once per scene and model and
  then reused, about 2.4 s per scene in Python. C++ saves roughly a second per
  scene. The large savings today came from parallel scoring (4.8x) and from
  fetching each block once. The C++ is included because it is correct, tested,
  free to use, and part of what this project sets out to show.
- `07b_blocks` and `07e_verify_parallel` now start the session with
  `build_cpp=True`. A failed build is reported and the run continues in Python.

## 2026-09-18 Unpack speed test (07f), six-Ouster rule, development block kept apart

- `07f_unpack_speed_test` measures whether the LiDAR `.tar.xz` files can be
  unpacked on all cores. Verified basis: the release manifest says they were
  compressed with 64 threads. NOT verified until the notebook runs: the xz
  version on Colab, whether the files really contain many independent blocks,
  and the speed. The notebook unpacks the same file three ways (the toolkit's
  Python way, the xz tool on one core, the xz tool on all cores), compares every
  unpacked file by SHA-256, and prints a verdict. Threshold for building it in:
  identical files AND at least 2x faster. Nothing in the pipeline uses it yet.
- Six-Ouster rule (Claude's recommendation, applied as a DEFAULT the project
  owner can change): `pipeline.enough_ouster`, `MIN_OUSTER_DEFAULT = 6`. The
  census found 55 train scenes with only four Ouster sensors. Their ground truth
  would be thinner than the rest, so they are skipped with a log line, both in
  the GPU prediction pass and in scoring. Pass `min_ouster=4` to include them.
  Val block 11 has six sensors in every scene, so existing results are unchanged.
- Development block: `07c_report` has `DEVELOPMENT_BLOCKS = [("val", 11)]`. The
  occlusion rule and the protocol were tuned on that block, so its scenes are
  left out of the final tables and the count left out is printed.
- Block list pre-filled in `07b0` and `07b` with the SMALL option (7 train
  blocks: 11, 12, 13 dark and wet; 92, 82, 83, 90 motorway-rich). This is a
  proposal in a configuration cell, not a result. The val and test blocks are
  listed in a comment as the next steps.

## 2026-09-18 Parallel LiDAR unpacking: measured, then built in

Measured on Colab (notebook 07f, CPU high-RAM, 8 cores, xz 5.4.5), val block 11,
one LiDAR archive of 3.84 GB holding 324 independent xz blocks, 4,798 files,
8.13 GB unpacked:

| way | time |
|---|---|
| Python `tarfile`, as the toolkit does | 419.2 s |
| `xz` tool, one core | 417.3 s |
| `xz` tool, all 8 cores | 79.3 s |

All three gave byte-identical files (SHA-256 of every file). 5.3x. The one-core
xz time equals the Python time, so the gain is the threads, not the tool.

Built in as `src/vggt_aura/fast_download.py`, used by `aura_data.download_block`
(`fast_unpack="auto"`, `"never"` for the toolkit's own way). What is swapped is
ONLY the decompressor. The toolkit still selects, downloads, extracts each
member with its own path-safety checks, deletes the archives and writes the
metadata. Each archive is first checked against the manifest's SHA-256 (the
same check as the toolkit's `--verify`), then `xz -T0` turns the `.tar.xz` into
a plain tar that takes its place; the toolkit opens archives with "r:*", which
reads a plain tar whatever its name. The swap is atomic and marked, so an
interrupted block can be repeated. If the fast way fails for any reason, the
half-done archives are removed and the block is done the toolkit's way; if no
`xz` of version 5.4 or newer is found, likewise.

Verified locally: 191 tests pass (real `xz` 5.8.3, multi-block and single-block
archives, wrong checksum never unpacked, interrupted run repeatable, fall-back);
the `select()` call was run against the real toolkit and the real release
tables and resolves val block 11 to the expected three archives.
NOT yet verified: one whole block through the new path on Colab. The first
07b run does that; it runs the toolkit's validator on each new block
(`VALIDATE_DOWNLOAD = True`). Expected speed is below 5.3x for the whole
unpack step, because the plain tar is written once more before extraction.

## 2026-09-18 Phase 7, first batch: 7 train blocks, 140 scenes (source: notebook 07c output)

Blocks: train 11, 12, 13 (chosen because every scene is wet and not-day) and
train 92, 82, 83, 90 (chosen for motorway scenes). These blocks were picked
BECAUSE they are hard or rare, so the headline below is not an estimate for the
dataset as a whole. Val block 11 (development) is left out. Camera front_medium,
sequence_scale alignment unless stated, 95% bootstrap intervals over scenes.
All 140 scenes had six Ouster sensors; the six-Ouster rule skipped nothing.

VGGT-Omega:
- AbsRel 0.1259 (0.1150 to 0.1380); authors' protocol 0.1119. Val block 11 was 0.0542.
- dry 0.0811 (76 scenes) against wet 0.1792 (64); day 0.0824 (80) against
  night 0.1839 (33) and twilight 0.1842 (27).
  CONFOUNDED: 60 of the 64 wet scenes are the same scenes as the 60 not-day
  ones, and they come from very few recordings. Weather and lighting cannot be
  told apart in this batch. Also unverified: rain adds false LiDAR returns, so
  part of the "error" in wet scenes may be ground-truth error.
- motion: background 0.1111, parked 0.2304, moving 0.3448. Paired moving minus
  background +0.2329 (0.1907 to 0.2788), worse in 126 of 136 scenes. Moving
  minus parked +0.0676 (0.0026 to 0.1283), 71 of 84: the interval barely
  excludes zero.
- trap 3: the bound on ground-truth error from object motion is now 8.8% median
  and 18.6% at the 90th percentile (val 11: 2.5%). At motorway speeds a real
  share of the moving-object error may be ground-truth timing, not the model.
- depth bands: 0.093 (1-10 m), 0.144, 0.166, 0.181 (40-80 m).
- classes: flat 0.054, structure 0.129, vegetation 0.203, other 0.331,
  human 0.349 (37 scenes), vehicle 0.353, thin 0.401.
- confidence quartiles: 0.299, 0.102, 0.059, 0.045. Confidence orders error well.
- road type: motorway 0.111 (28), urban 0.115 (54), overland 0.144 (58);
  not interpretable alone, because overland here is mostly the wet dark drive.
- pose (134 moving scenes): rotation 1.44 deg, translation direction 5.3 deg
  (2.1 to 9.5: a heavy tail, a few scenes fail badly), AUC@30 89.1, ATE 5.4%
  of path. pose_scale alignment 0.68 with a huge interval and
  pose-scale over depth-scale 1.43 (1.02 to 2.12): in some scenes the pose
  scale and the depth scale disagree. On val 11 that ratio was 1.0006.
  The failing scenes are not yet identified.
- focal length underestimated: fx -10.2%, fy -13.5%.

VGGT-Omega against VGGT, paired by scene:
- depth: a TIE. AbsRel difference -0.0004 (-0.0077 to +0.0075), Omega lower in
  85 of 140. On val block 11 Omega was lower in 18 of 20. The depth advantage
  seen on the development block does not hold on these hard scenes.
- moving objects: Omega +0.023 worse, interval spans zero (58 of 136).
- pose: Omega clearly better. Rotation 1.44 against 2.68 deg, translation 5.3
  against 16.5 deg, ATE 5.4% against 9.9% of path, AUC@30 89.1 against 81.1.
- focal: both underestimate; VGGT more (-16.7% fx).

Run cost, measured: 07b0 on A100 about 40 min for 7 blocks (models run about
70 s per block); 07b on CPU high-RAM 44 min (6.5 min per block; validator
passed on every block; fast unpacking worked on every block, no fall-back).

## 2026-09-18 Report additions after the first batch

- `evaluation.cross_table`: two scene attributes at once. `build_report` now
  adds weather by lighting and road type by weather. Reason: in the first batch
  60 of 64 wet scenes are also the dark ones, so the single tables cannot say
  which attribute matters.
- `n_recordings` column in every scene-attribute table. A recording is the
  drive a scene was cut from (scene id = "<recording>|<number>"). The bootstrap
  resamples scenes, which treats scenes of one drive as independent; they are
  not, so an interval over many scenes from one or two recordings is narrower
  than it deserves. The count makes that visible. A bootstrap over recordings
  is the stricter option and is not built yet.
- `evaluation.scene_overview` (one line per scene, saved as
  `<model>_scene_overview.csv`) and `evaluation.worst_scenes`; `07c` cell 6
  lists the 12 worst scenes by depth error, by translation direction, and by
  disagreement between pose scale and depth scale (absolute log of the ratio).
- 193 tests pass. No block needs re-running: `07c` reads saved results only.

## 2026-09-18 First batch, second look (source: 07c re-run with cross tables and worst scenes)

- ALL 60 dark scenes come from TWO recordings of one evening: 2026-01-08-15-27-06
  (27 scenes, twilight) and 2026-01-08-16-15-15 (33 scenes, night). Both wet.
  "Night" in this batch means one drive. Any night or wet-dark number is a
  statement about that evening, not about night driving.
- weather by lighting (VGGT-Omega AbsRel): dry day 0.0811 (76 scenes, 27
  recordings); wet day 0.1070 (4 scenes, 3 recordings, THIN); wet night 0.1839
  (33, 1 recording); wet twilight 0.1842 (27, 1 recording). No dry dark scene.
  Wet day sits between dry day and wet dark, but four scenes carry no finding.
- road type by weather: dry is 0.076 to 0.089 on all three road types; wet is
  0.167 to 0.200 on all three. Road type is not what drives the error here.
- the 12 worst depth scenes of BOTH models all come from those two recordings.
  Several have moving-object AbsRel above 1.0.
- pose: the mean translation error (5.3 deg for Omega, 16.5 deg for VGGT) is
  made by a few scenes where the direction of travel comes out REVERSED (errors
  of 120 to 177 deg, negative pose scale) while rotation stays within a few
  degrees. Omega: 3 such scenes in the worst-12 list; VGGT: all 12 of its
  worst-12 are above 97 deg. All of them: the wet night/twilight drives, mostly
  overland at 60 to 95 km/h. Same code gives under 1 deg elsewhere, so a
  convention error on our side is unlikely, but the cause is NOT established
  (hypothesis only: little static texture in headlights and spray). To look at
  in the error visualisations.
- one dry daytime outlier in both models: recording 2025-06-20-10-08-06
  (scenes 87 and 89, motorway at 8 to 12 km/h, presumably a traffic jam): pose
  scale against depth scale 0.32 (Omega) and 2.85 (VGGT), ATE 11 to 17% of path.
  Hypothesis only: most of the view is other moving vehicles.
- added `evaluation.pose_typical_and_failures`: median over scenes, count of
  reversed scenes, count of scenes whose two scales differ by more than 1.5x.
  The mean alone misdescribes the typical scene. 194 tests pass.

Batch 2 (pre-filled in 07b0 and 07b, appended to the list; finished blocks are
skipped): train 5, 6, 9 (census: 12 wet scenes each, only 2, 1, 1 dark: wet in
DAYLIGHT, which is what separates rain from darkness) and val 0, 1, 2, 10, 12.
Limits known in advance: the whole dataset has only 13 dry night scenes and 3
dry twilight scenes, so "darkness without rain" will stay thin whatever is run;
and the val dark scenes may be the same evening (n_recordings will show it).
The defensible comparisons are dry day against wet day (rain, in daylight) and
wet day against wet dark (darkness, given rain).

## 2026-09-18 GPU notebook: predictions are saved uncompressed; per-step timers

Observed in batch 1: a GPU block took 315 to 395 s, of which the models ran
about 70 s and the download about 60 s. About 190 s were unexplained.
Laptop benchmark with arrays of the real shape (40 frames assumed, 400 x 640):
`np.savez_compressed` 3.4 s per scene and model for a 50 MB file, `np.savez`
0.1 s for 61 MB; reading and resizing 40 camera images 1.3 s. Per block of 20
scenes and 2 models that is about 136 s of compression and 53 s of images,
which matches the unexplained time. Laptop numbers, so indicative only.

- `save_predictions(..., compress=False)` is the new default. Predicted depth
  is noise-like in its low bits, so compression saved under 20%. The stored
  numbers are bit-identical (tested), `load_predictions` reads both kinds, and
  the batch 1 files stay as they are. Cost: about 0.4 GB more Drive space per
  block.
- `predict_block` now reports `image_seconds` and `save_seconds` next to
  `model_seconds`, so the next run shows the real split on Colab.
- Considered, not built: reading the next scene's images in a background thread
  while the GPU works (at most about 50 s per block; decide from the timers),
  and fetching the next block while the current one is scored in 07b.
- No further C++ candidate: what remains is network, disk, decompression
  (already native code on all cores) and the models themselves.
- 195 tests pass.

## 2026-09-18 Two CPU servers can share one block list

- `07b_blocks_second_server.ipynb` is `07b_blocks` with `REVERSE = True`: it
  walks the same list backwards on a second CPU server; the two meet in the
  middle. `07b_blocks` alone still does everything.
- `pipeline.claim_block` / `release_claim`: a small claim file per block in
  persistent storage. A block claimed by the other server is skipped. Drive
  takes seconds to show a file on the other server, so both can still grab the
  same block in the same moment; that wastes one block of work and nothing
  else (identical numbers, atomic result files). Claims are released in a
  `finally`, and one older than 25 minutes (a dead server) is taken over.
- The GPU notebook is not split: it is the short step, and a second A100
  doubles the fixed cost (installs, loading two models) for little gain.
- Known and accepted: the append-only manifest on Drive can lose lines when
  two servers write it at once. Nothing in phase 7 reads it to decide; all
  skipping is decided from the result and prediction files.
- 197 tests pass.

## 2026-09-18 GPU notebook, measured split and two more savings (code ready, not yet run)

Measured on Colab with the new timers (train block 9, A100, 20 scenes, 2 models,
232 s): models 71 s, reading and resizing images 69 s, download + checksum +
extract 56 s, loading both models 26 s, saving 9 s. Saving uncompressed took
the block from about 320 s to about 235 s.

- Models were reloaded for every block. `predict_block(..., runners=)` now
  accepts runners made once by the notebook, which releases them at the end.
- `pipeline.prepared_ahead`: the next scene's images are read and resized in
  background threads while the GPU runs the present scene (same upstream
  function, same tensor; at most two scenes prepared at a time). Order is
  kept, an unreadable image raises at its own scene. `predict_block` now
  decides what is left from `predictions_are_saved` (timestamps only) instead
  of loading whole prediction files.
- Expected, NOT measured: about 150 s per block. The `image_seconds` of the
  next run is the time the GPU really waited for images and will tell.
- Test split at the pinned dataset revision: blocks 0, 1, 2, 3, 11, 12 are
  downloadable (107 scenes; the release tables list 13 blocks, the others were
  not uploaded at that revision). No test block has been run yet.
- 199 tests pass. The 07b0 notebook itself is patched only after the running
  session ends (scratchpad `patch_07b0_runners.py`, then a full sync).

## 2026-09-18 Efficiency review of the scoring notebook (code ready, NOT yet run on Colab)

Measured on Colab, heavy 12-LiDAR block (train 5, 16 GB, 724 s): download 44 s,
checksum + decompress 251 s, extract 123 s, validator about 45 s, fetching saved
predictions from Drive 77 s, scoring 181 s. Scenes take 40 to 85 s each here
against 13 to 32 s in batch 1; WHY is not known (more boxes, more points, or
Drive reads inside the workers), so step timers were added instead of a guess.

Changed, all result-identical and tested (203 tests):
- `fast_download.sha256_many`: all archives of a block are checksummed at once in
  threads (one after the other this was an estimated 40 to 60 s on one core).
- `pipeline.warm_files`: a block's saved predictions are read from Drive in the
  background WHILE the block downloads, so the scorers find them cached.
- The toolkit's validator runs in the background while scenes are scored. Its
  verdict is collected before any result is written, and a block that fails
  validation now RAISES and saves nothing (before, it only printed).
- `objects.scene_tracks`: ego poses, boxes and object velocities are loaded once
  per scene and shared by both models (they were loaded per model).
- `score_scene` reports seconds per step (setup, predictions, truth, labels,
  evaluate); a block's summary sums them as `score_steps_worker_s`.

Estimated, not measured: 2 to 3 minutes off a heavy block of 12.

Considered, NOT built:
- Fetching and unpacking the next block while the current one is scored. Largest
  remaining lever (perhaps a third of a block), but two blocks on disk at once
  can exceed a CPU server's disk with 16 GB blocks, and it interacts with the
  two-server claims. Only worth it if many more blocks are to be run.
- Not extracting the six unused Aeva sensors: the validator may then reject the
  folder (unverified), and the download and decompression cost stays.

## 2026-09-18 Phase 7, batches 1 + 2: 15 blocks, 287 scenes (source: notebook 07c output)

Blocks: train 5, 6, 9, 11, 12, 13, 82, 83, 90, 92 and val 0, 1, 2, 10, 12. Val 11
(development, 20 scenes) left out. Blocks were chosen for rare conditions, so
headline numbers are not dataset-wide estimates. sequence_scale, 95% bootstrap
over scenes; n_recordings shows how few drives some cells rest on.

Rain against darkness (VGGT-Omega AbsRel):
- dry day 0.0806 (148 scenes, 35 recordings); wet day 0.0908 (61 scenes, 9
  recordings); wet night 0.1589 (44, 2 recordings); wet twilight 0.2452 (32,
  1 recording); dry night 0.1213 (2 scenes, thin).
- Rain in daylight costs about 0.010 AbsRel (intervals 0.076-0.086 against
  0.083-0.100, just overlapping). The doubling of error seen in batch 1 belongs
  to the dark scenes, which are three recordings, all wet. "Darkness" and "that
  evening" still cannot be told apart, and darkness without rain is 2 scenes.
- VGGT shows the same pattern: 0.101, 0.110, 0.155, 0.180.
- dry is 0.078 to 0.088 on every road type; overland wet (0.194) is the dark drive.

One scene dominates a cell: 2026-01-08-15-27-06|43 (val 10, twilight, overland)
has AbsRel 1.640 with good pose (0.3 deg rotation). It moved wet twilight from
0.184 to 0.245 and widened its interval to 0.17-0.35. Cause unknown; to be
looked at before it is reported. Means are sensitive to such scenes; a median
over scenes should be reported next to them.

Other VGGT-Omega results: overall 0.1134 (0.1018-0.1276), authors' protocol
0.1030; background 0.104, parked 0.174, moving 0.266; moving minus background
+0.161 (241 of 270 scenes); moving minus parked +0.051 (0.018-0.083, 148 of
189), now clearly above zero; trap-3 bound median 7.3%, p90 15.0%; depth bands
0.092 / 0.123 / 0.135 / 0.145; flat 0.047, structure 0.127, vegetation 0.165,
other 0.264, vehicle 0.272, human 0.334, thin 0.349; confidence quartiles
0.275 / 0.086 / 0.053 / 0.041. Pose, typical scene (median over 269 moving
scenes): rotation 0.74 deg, translation 0.81 deg, ATE 1.4% of path; 4 scenes
with the direction of travel reversed; 23 scenes where pose scale and depth
scale differ by more than 1.5x. Focal length: fx -5.4%, fy -9.2%.

VGGT-Omega against VGGT, paired over 287 scenes:
- depth: Omega lower in 211 scenes (74%); mean difference -0.0070 with interval
  -0.0145 to +0.0018, which still includes zero because of a heavy tail (the
  1.64 scene is Omega's). Batch 1 alone had shown a tie.
- moving objects: Omega better by 0.026 (-0.051 to -0.001, 164 of 270).
- pose: rotation 1.22 against 2.83 deg, translation 3.8 against 10.7 deg, ATE
  3.7% against 7.4%, AUC@30 92.5 against 83.5; reversed scenes 4 against 14.
- focal: -5.7% against -16.4%.

Run cost: 07b0 on A100 33 min for 8 blocks; 07b on two CPU high-RAM servers
about 50 min (heavy 12-LiDAR blocks 8 to 17 min each); all validators passed.

## 2026-09-19 Phase 8: the test split, as its own notebooks

- `08a_test_predict_gpu`, `08b_test_score` (+ optional `08b_test_score_second_server`),
  `08c_test_report`. Generated from the phase-7 notebooks (scratchpad
  `make_nb08.py`), so they run exactly the same code. Blocks: test 0, 1, 2, 3, 11,
  12 (107 scenes), the only test blocks downloadable at the pinned revision.
- Same RUN_TAG as phase 7, so all results live together. `08c` shows the test
  scenes alone, then `evaluation.compare_splits`: test against all other scenes
  INSIDE the same weather and light. The test split has no night scene, so its
  overall number must not be set against an overall number that includes the
  night drives. After phase 8, `07c` covers every scored scene, test included.
- Every scene-level table now carries a `median` next to the mean (one scene
  with AbsRel 1.64 had moved a whole cell).
- Code review before the run found and fixed: (1) a block that fails validation
  stayed on disk, so every re-run would have found "block on disk", skipped the
  download and failed again; it is now removed. (2) background threads imported
  the image-loading modules while the main thread imported the model from the
  same package; both are now imported up front in the main thread.
- Both report notebooks were dry-run locally, cell by cell, on synthetic block
  results (scratchpad `rehearse_08c.py`): all cells ran. 205 tests pass.
- First run on Colab of: kept models, background image reading, parallel
  checksums, predictions warmed during download, validator in the background,
  shared tracks, step timers. None of these has been measured yet.

## 2026-09-19 Phase 8 result: the test split (107 scenes, 6 blocks, 15 recordings; source: 08c output)

All daytime (86 dry, 21 wet), 92 urban, 14 overland, 1 motorway. Nothing was
tuned on these scenes. sequence_scale, 95% bootstrap over scenes.

VGGT-Omega on test:
- AbsRel 0.0815 (0.0765-0.0869), median 0.0762; authors' protocol 0.0765.
- like for like: dry day 0.0835 on test against 0.0806 on train+val (difference
  +0.003, intervals overlap); wet day 0.0734 (21 scenes, 4 recordings) against
  0.0908 (overlap). The daytime numbers found on train+val hold on unseen data.
- moving objects: background 0.077, parked 0.099, moving 0.203. Moving minus
  background +0.126 (0.101-0.156), worse in 99 of 101 scenes. Moving minus
  parked +0.108 (0.077-0.148), 78 of 89. Parked minus background only +0.023.
  The trap-3 bound here is 4.3% median, 9.5% p90: the moving-object gap is
  about three times what ground-truth timing could explain at its median. This
  is the cleanest evidence so far that MOVING, not "being a vehicle", is what
  costs accuracy.
- depth bands 0.066 / 0.082 / 0.094 / 0.106; flat 0.039, structure 0.094,
  vegetation 0.112, vehicle 0.140, other 0.175, human 0.274, thin 0.296;
  confidence quartiles 0.181 / 0.065 / 0.045 / 0.035.
- pose (99 moving scenes): rotation 0.59 deg, translation 0.82 deg, AUC@30 98.0,
  ATE 1.2% of path. ZERO scenes with reversed direction, ZERO scenes where pose
  scale and depth scale disagree by 1.5x. The pose failures of phase 7 are
  confined to the wet dark drives (and one traffic-jam recording).
- focal length: fx -0.5% (interval includes zero), fy -4.9%.

VGGT-Omega against VGGT on test, paired: depth -0.0436 (-0.0503 to -0.0375),
Omega lower in 103 of 107 scenes; moving objects -0.166; rotation 0.59 against
2.44 deg; ATE 1.2% against 3.1%; focal -0.7% against -11.0%.
Together with phase 7: Omega's depth advantage is clear in daylight and shrinks
to a tie on the wet dark drives; its pose advantage holds everywhere.

Caveat: VGGT is WORSE on test dry-day (0.129) than on train+val dry-day (0.101),
intervals apart. Neither model saw any AURA data, so this is not a train/test
effect; the test split is 86% urban, and "dry day" cells differ in what they
contain. Like-for-like by weather and light is therefore only approximately
like for like.

Run cost with every speed-up active: 08a (A100) 13 min for 6 blocks, 2.1 min
per block (GPU waited 3 s per block for images, was 68 s; models loaded once).
08b on two CPU servers 20 min; heavy 12-LiDAR block 9.4 min (was 12 to 17),
light block 4 to 5 min (was 6.5); predictions waited for: 1 s (was 77 s).
Step timers, worker-seconds for test block 0: evaluate 438, truth 379, labels
177, predictions 55, setup 5: scoring time is evaluation and ground truth, not
the motion labels.

## 2026-09-19 Phase 9: error pictures (`src/vggt_aura/figures.py`, `09_error_pictures.ipynb`)

- Per scene and model: a table with one line per frame (AbsRel, signed median
  error, the frame's own scale over the scene's scale, share and error of
  moving-object pixels), a figure (image | predicted depth | LiDAR | signed
  relative error with moving objects ringed) and the driven path from above.
- Same conventions as the evaluation, and tested against it: the per-frame
  numbers, weighted by pixels, add up to the scene's AbsRel; error sign is
  (prediction - truth) / truth; one scale per scene.
- The path figure scales the prediction by PATH LENGTH, which is always
  positive, so a trajectory predicted backwards is drawn backwards. A
  least-squares scale would come out negative and flip it back.
- Uses saved predictions and saved ground truth; only the camera layer is
  downloaded. Scenes: the AbsRel 1.64 outlier (val 10), two reversed-direction
  scenes (train 12, 13), the motorway crawl (train 82), the worst
  moving-object scene of the test split, and the test scene with the median
  error as the reference for "normal".
- The notebook's cells were dry-run locally with synthetic results and a faked
  dataset (scratchpad `rehearse_09.py`); 209 tests pass. Not yet run on Colab.

## 2026-09-19 What the error pictures show (source: 09_error_pictures output, 6 scenes, viewed one by one)

The picture code reproduces the evaluation on real data (outlier scene: AbsRel
1.640 in both). Observations, each from ONE scene unless said otherwise:

1. The AbsRel 1.64 outlier (2026-01-08-15-27-06|43, wet twilight motorway) is
   mostly a GROUND-TRUTH problem, not a model failure. The LiDAR shows (a) large
   round blobs behind every vehicle, far bigger than the vehicle: water spray,
   measured as a surface; (b) scattered returns at 1 to 3 m in the open sky:
   rain drops; (c) slabs of points in image areas that show sky. The model's
   depth looks plausible (road, cars, far background). Per frame the MEDIAN
   signed error is near zero (-0.02) while the mean AbsRel is 2 to 4: a few
   hundred false near points dominate, because relative error divides by a
   small "true" depth. VGGT scores 0.755 on the same scene, for the same reason.
   Consequence: AbsRel in rain is inflated by the ground truth by an unknown
   amount. These points are evidently not all labelled `noise`. The wet-scene
   numbers must be reported with this caveat and with medians.
2. "Direction reversed" (2026-01-08-16-15-15|16, wet night) is really "ego
   motion lost": the predicted path follows the truth for about 10 frames and
   then stalls near 100 m while the car drives 425 m; pairwise directions are
   then noise. The camera image is blown out by oncoming headlights and glare on
   the wet road; oncoming vehicles visible to LiDAR are absent from the
   predicted depth. Depth of the static scene stays reasonable (0.139).
3. The motorway crawl (2025-06-20-10-08-06|89, dry day) confirms the traffic-jam
   hypothesis: the ego car STANDS for 22 frames (truth: 0 m, even -2 m), then
   pulls away. The model predicts steady forward motion from frame 0. Traffic in
   the next lane moves; the model takes the moving vehicles for the static
   world. Depth is excellent (0.074). A pose failure with nothing wrong in depth.
4. Worst moving-object scene of the test split (2025-08-04-11-19-58|52, slow
   urban): errors on moving objects sit largely at their EDGES (a red band
   beside walking pedestrians, fringes on the bus), and static thin poles show
   red/blue fringes too. Two causes cannot be told apart from pictures: the
   model's depth bleeding across object boundaries, or LiDAR points of a moving
   object landing beside it in the image (time offset between sweep and
   exposure). NOTE: the trap-3 bound only covers motion ALONG the viewing axis;
   an object crossing the view sideways has a bound of zero but can be
   misregistered by several pixels. So the bound understates ground-truth error
   for crossing objects, and the moving-object gap is not yet cleanly the
   model's. Proposed check: moving-object error in the INTERIOR of objects
   (LiDAR pixels whose neighbours are all on the same object) against their
   boundary. Needs boxes, so the base layer of the blocks, no GPU.
5. Camera images of the test scene show water smear on the lens although the
   weather field says dry: the weather label comes from a weather service, not
   from the image.

## 2026-09-19 Phase 10 result: moving-object error, interior against boundary (test split, 107 scenes)

First run failed harmlessly: with only the base layer on disk the toolkit hides
every frame that needs a camera image, so `select_frames` came back empty and
every scene was reported as "never scored". Frames are now found by the
timestamps stored in the saved predictions (`pipeline.frames_for_timestamps`).

VGGT-Omega, rim of 6 pixels, AbsRel, mean over scenes (95% bootstrap):

| | whole object | boundary | interior |
|---|---|---|---|
| moving | 0.203 | 0.273 | 0.109 |
| parked | 0.099 | 0.139 | 0.054 |
| background | 0.077 | | |

Paired over scenes:
- moving minus parked, whole objects: +0.108 (0.077 to 0.148), 78 of 89 scenes.
- moving minus parked, INTERIOR only: +0.062 (0.038 to 0.096), 70 of 76 scenes.
  With a 12-pixel rim: +0.052 (0.023 to 0.096). So roughly 60% of the gap
  survives where misregistration cannot reach: motion itself costs the model
  accuracy. The remaining 40% sits on the outlines and cannot be attributed.
- parked: boundary minus interior +0.090 (0.069 to 0.112), 83 of 86 scenes.
  Outlines alone cost more than motion does. For parked objects this is the
  model's soft depth edges plus whatever residual calibration and ego-motion
  error shifts LiDAR points across an edge; the two are not separated here.
- moving: boundary minus interior +0.169. The excess over parked (+0.079) is
  what object motion adds at the outline: misregistration, or worse edges on
  moving objects, not separable.
- same distance (interior, moving minus parked): +0.101 at 1-10 m, +0.025 at
  10-20 m, +0.019 at 20-40 m; every interval excludes zero. Distance is not the
  hidden cause, but the effect is concentrated on NEAR moving objects.
- cars only, interior: +0.036 (0.024 to 0.047), 56 of 66 scenes. Buses: too few
  scenes (4) to say anything.
- VGGT for comparison: interior moving 0.332 against parked 0.102. The older
  model's moving-object error is inside the objects; VGGT-Omega's interior
  moving error is a third of it (0.109).

Limit: LiDAR returns nothing from the sky, so an object's outline against the
sky has no neighbouring pixel of "something else" and counts as interior. This
puts some outline error into both interiors.

Object pixels by category (VGGT-Omega, test split): car 3.6 M moving / 9.3 M
parked, bus 0.74 / 0.34, person 0.47 / 0.25, truck 0.24 / 0.38.

## 2026-09-19 Clean-up before publishing

- README rewritten around the final results; every number in it was checked
  against the notebook output it came from. One sentence was wrong on first
  writing and corrected: of the 4 scenes with reversed direction of travel
  outside the test split, 3 are on the wet night drive and 1 is a dry daytime
  urban scene (2026-06-03-10-44-05|39, depth 0.042) that was not looked at.
- `notebooks/09_error_pictures.ipynb`: embedded pictures removed from the
  committed copy (48 MB to 0.2 MB); seven of them are in `results/figures/` as
  JPEG. They show AURA imagery and are shared under CC BY-SA 4.0.
- Registered predictions moved to `docs/`. Empty `results/` sub-folders removed.
- Lint (pyflakes) clean, 215 tests pass, repository 14 MB without `.git`.
- Scan for secrets in tracked and untracked files: no token-like strings;
  `.env` is ignored. The owner's first name and Hugging Face user name appear
  in the documents and in two notebook outputs, as authorship.
- Still open: a licence for this repository's own code (owner's choice).

## 2026-09-19 Second clean-up and the pre-publication security scan

Clean-up (the first pass had left the notebook folder as it grew):
- `notebooks/` holds the run itself (11 notebooks); phases 0 to 6 moved to
  `notebooks/development/`, the three speed measurements to
  `notebooks/experiments/`; `notebooks/README.md` says what each one is.
- Notebooks are now committed with an EMPTY sync cell
  (`python -m vggt_aura.sync --clear`): the code exists once, in `src/`, `cpp/`
  and `tests/`, not as a base64 copy in each of 24 notebooks. `sync` now also
  covers sub-folders.
- Usage audit of all 183 public functions and classes: none is dead. Three are
  used by tests only (`unproject_pinhole`, `visible_mask`, `_selftest_job`), on
  purpose. pyflakes clean, 216 tests.
- `.vscode/` ignored (its settings hold a local absolute path).

Security scan (scratchpad `security_scan.py`; 77 files that would be published,
plus the whole git history; notebook sources, text outputs and metadata
included; the actual values in `.env` searched for exactly, never printed):
- the Hugging Face token and the GitHub token: not in any file, not in history.
- no token-like strings of any common kind, no private keys, no URLs with
  credentials, no e-mail addresses, no home or user paths, no Drive file ids,
  no IP addresses in the files to be published. `.env` is ignored and was never
  committed. The pictures carry no EXIF or other metadata.
- the commit e-mail is NOT a GitHub noreply address. It appears only as the
  author of the existing commit, not in any file. Every commit pushed with it
  shows it publicly.
- the existing local history (one commit) still contains `.vscode/settings.json`
  and the old notebooks with embedded code bundles. Harmless, but a fresh
  history is cleaner.
- code patterns looked at and judged fine: `verify=False` in `fast_download.py`
  is the toolkit's checksum argument, not TLS (checksums are verified just
  before); `extractall` in the sync cell unpacks the project's own zip;
  `shell=True` occurs only in development and experiment notebooks with fixed
  commands or paths made by this code. Upstream code is pinned by commit hash.

## 2026-09-19 Published names, and what the hashes in this repository are

- Notebooks renamed to plain names (`01_census` ... `09_moving_object_edges`;
  `development/`, `experiments/`); references fixed in notebook sources, code,
  tests and READMEs. Saved OUTPUTS were not edited: they are the record of what
  was printed. `notebooks/README.md` maps the working names used in this log.
- 334 dead download progress-bar widgets removed from saved outputs.
- Hash audit. The "code hash" printed by the sync cell is the first 12 hex
  digits of the SHA-256 of a zip of `src/`, `cpp/`, `tests/` and
  `pyproject.toml`: 37 files, all of them published here, `.env` excluded by
  name. It identifies a code version and cannot be turned back into anything;
  the current bundle and the one bundle found in the git history were decoded
  and searched for the values in `.env`: not present. The other long hex
  strings are pinned public commits and dataset revisions, and random widget
  ids (now removed). Nothing derived from a secret is printed anywhere.

## 2026-09-19 `docs/performance.md`

One page that points to the exact lines where C++ is called, says when it runs
(`backend="auto"`, log line `truth built (cpp)`), how equality is tested, and
what it saved, with every other speed-up next to it: where in the code, the
measured gain, how identity of results was checked. It states plainly that the
C++ gain is about one second per scene and model (2.2x on a step that is 5% of
a scene's time; the rewritten version was timed on a laptop only, the first
version on Colab was 1.4x per scene and slower than NumPy on the occlusion
test). The README wording was corrected to match. All relative links checked.
Owner's decision: no further algorithms or research; ship. Ideas are listed in
the README as possible next steps.

## 2026-09-19 Notebook text brought in line with the published names and with what each notebook does

The rename had only caught full file names. A cell-by-cell audit of all 24
notebooks (markdown cells, code comments, printed messages; saved outputs and
storage keys left alone) found and fixed:
- titles still reading "Phase 7a", "Phase 8b" and so on: every run and
  experiment notebook has a new title and description; development notebooks
  are titled "Development 0" to "Development 6".
- descriptions that no longer matched the code: `03_score` still described the
  first single-pass design (it now says what happens: checksums, all-core
  unpacking, validator, C++ ground truth, parallel scoring, six-Ouster rule,
  second server); time estimates replaced by the measured ones (GPU block 2 min,
  pictures 20 min, edge check 25 min); each run notebook ends with "Next:".
- the `REVERSE` comment, identical in all four scoring notebooks and therefore
  wrong in the two second-server ones; the block-list comments ("batch 2", "the
  six test blocks come last"); a wrong pointer in `05_test_predict_gpu`.
- `RUN_TAG = "phase7_front_medium"` is a storage folder name and stays; it now
  carries a comment saying so.
- `06_cpp_core` now says that its saved output is from the FIRST version of the
  C++ occlusion test (slower than NumPy), which is why it was rewritten.
- each experiment notebook states its outcome and where it went into the code.
- paths to the registered predictions (`docs/`), `tests/` added to the list of
  what the sync cell carries, unnumbered "Start the session" cells numbered.
Checked afterwards: every code cell compiles, every notebook has a title, one
empty sync cell and a session cell; the 15-block list is unchanged (compared
before and after); 216 tests; all relative links and every notebook name used
in the documents resolve; security scan unchanged.

## 2026-09-19 Code fixes before publishing, and a provenance record

Checked from a clean copy of only the publishable files in a fresh environment:
`pip install -e .[dev]` and `pytest` work with nothing else installed (skips:
the C++ tests without a compiler, two cross-checks without the dataset toolkit).
`dev` now also installs matplotlib, pillow and pybind11.

Defect found and fixed: `process_block` tries the camera layer only when
everything is saved, and on failure retried with the full download. It caught
ANY `RuntimeError`, and since a rejected download also raises one, a validator
failure would have been printed as "a saved ground truth did not fit" and
retried. A missing ground truth is now its own error
(`ground_truth.GroundTruthUnavailable`), the only one that triggers the retry;
tested both ways.

Wording from the time of the work removed from the code ("phase 7", "trap 3");
the pipeline's module text described the first single-pass design and was
rewritten. The report table key `intrinsics_scale_and_trap3` is kept, because it
is in the saved outputs and result files, and is now explained in `objects.py`.

Provenance (owner's point: reproducible means "I will run it again"). Emptying
the sync cells for the commit also removed their output, which was the only
record of the code hash a notebook ran with. Now the sync cell hands its hash
to the session, and `start_session` appends one line per session to
`<persist>/provenance.jsonl`: code hash, dataset revision, toolkit and model
commits, weight revisions, package versions, GPU, cores. No secret is in it
(tested). The code itself still has to travel in the cell, because the Colab
runtime cannot read files on the laptop; only its hash can live in a text file.
The README has a "Running it again" section, including how to force a repeat
(new `RUN_TAG`, or a fresh persistent folder) and what may differ between two
runs (GPU arithmetic; not measured here).
220 tests, lint clean, sync round trip checked, sync cells emptied again.

## 2026-09-19 A full re-run is now one setting

Every run notebook has `DRIVE_ROOT` in its first cell and passes it to
`start_session`. With the usual folder, saved work is skipped; with an empty
folder everything is recomputed from the images up. The token is looked for in
the chosen folder first and in the usual folder's `.env` second, so a fresh
folder needs no copy. Development and experiment notebooks keep the usual folder:
they are the record of how the method was built, not part of a re-run.

## 2026-09-19 Re-run check: one block from an empty folder is bit-identical to the published run

`experiments/rerun_check_1_predict_gpu` and `rerun_check_2_score_and_compare`,
val block 12 (7 scenes), persistent storage an empty Drive folder, code hash
19ba183bdad7, A100-SXM4-40GB, torch 2.11.0+cu128, numpy 2.1.3 (from the new
`provenance.jsonl`, which was written in both sessions and holds no secret).
- predictions: depth bit-identical in 7 of 7 scenes for both models; largest
  difference in any pose entry 0.
- ground truth, built again from LiDAR with the C++ core: identical in 7 of 7
  scenes at both input sizes.
- results: 580 + 575 rows compared, identical pixel counts, largest difference
  in AbsRel, delta<1.25, RMSE and the pose metrics: 0. Block mean AbsRel 0.0746
  (VGGT-Omega) and 0.0933 (VGGT) in both runs.
Also exercised for the first time on Colab, all working: the `DRIVE_ROOT`
setting, the token found in the usual folder's `.env` without a copy, the
provenance file, the renamed notebooks, the narrowed error handling, parallel
checksums and the background validator on a fresh folder.
Limit: both runs used the same GPU type and package versions. A different GPU
or PyTorch version was not tried.
Small gap noticed: the provenance line is written at session start, before the
VGGT package is installed, so its version is missing from the GPU session's
line; its commit is in the line anyway.

## 2026-09-19 Licence

Owner's choice: MIT for this repository's own code (`LICENSE`). It covers the
code only: outputs of VGGT-Omega stay under its noncommercial licence and
pictures showing AURA imagery stay CC BY-SA 4.0, as the README says.

## 2026-09-19 Final review of the pushed repository

Read through the GitHub CLI (signed in as the owner), nothing changed by it:
private, default branch `main`, one branch, no tags, releases, deploy keys,
webhooks, workflows or action secrets. Three commits, all with the owner's
GitHub noreply address as author and committer and linked to the account. MIT
detected. The README renders with its table, picture, nine sections and no
e-mail address. No value from `.env` and no token-like string anywhere in the
pushed history. Largest file 2.9 MB. No history rewrite needed.
Found and fixed: every notebook declares format 4.5, which asks for an `id` on
each cell; 125 cells written by the generator scripts had none. Stable ids were
added; nothing else in the notebooks changed.
