# Where the time went, where C++ runs, and what each change saved

Every number here was measured; the source is given with it. Nothing below changes a result: each speed-up was
checked to give identical numbers or byte-identical files before it was used.

## C++: where it is, when it runs, what it saved

**What is written in C++.** The geometry core and nothing else: quaternion to rotation, composing and inverting
rigid transforms, transforming points, pinhole projection and unprojection, pixel rounding, the occlusion test,
the nearest point per pixel, and the whole LiDAR-to-image step built from them. Header-only C++17, no
dependencies: [`cpp/include/vggt_geom/geometry.hpp`](../cpp/include/vggt_geom/geometry.hpp). It never opens a
file and never touches the GPU. The Python bindings (pybind11) are in
[`cpp/src/bindings.cpp`](../cpp/src/bindings.cpp).

**Where the pipeline calls it.** One place:
[`build_ground_truth` in `src/vggt_aura/ground_truth.py`](../src/vggt_aura/ground_truth.py#L260). Its `backend`
argument is `"python"`, `"cpp"` or `"auto"`; `"auto"` ([`resolve_backend`](../src/vggt_aura/ground_truth.py#L248))
uses C++ when the module is built and Python otherwise. The C++ branch is the call to `lidar_to_image`
([line 276](../src/vggt_aura/ground_truth.py#L276)). The scoring notebooks start their session with
`build_cpp=True` ([`start_session`](../src/vggt_aura/session.py#L76)), which compiles the module on the runtime in
about 15 s ([`cpp_build.py`](../src/vggt_aura/cpp_build.py#L75): CMake first, a direct compiler call as the
fall-back). Every scene's log line says which one built its ground truth: `truth built (cpp)` or
`truth built (python)`; the saved ground truth records it too. In the published runs it was C++ throughout.

**How it is known to be right.** [`tests/test_cpp_differential.py`](../tests/test_cpp_differential.py) feeds
both implementations the same arrays. Whole numbers and yes/no outputs must be EXACTLY equal, real numbers equal
to floating-point tolerance, and `build_ground_truth` must return the same arrays, dtypes and counts from either
backend for all three occlusion modes. On real data (40 frames, about 370,000 points each) candidates, occlusion
verdicts and chosen pixels were identical in every frame (`notebooks/development/06_cpp_core.ipynb`).

**What it saved, honestly.** Little.

| measurement | Python | C++ | where measured |
|---|---|---|---|
| first C++ version, whole scene, ground-truth step | 3.28 s | 2.40 s (1.4x) | Colab, `06_cpp_core` |
| first C++ version, occlusion test, one frame | 31 ms | 51 ms (SLOWER) | Colab, `06_cpp_core` |
| after the rewrite of the occlusion test, one frame | 75 ms | 25 ms (3.0x) | laptop |
| after the rewrite, whole ground-truth step, one frame | 113 ms | 51 ms (2.2x) | laptop |

The first version lost to NumPy because it built four full-image arrays of side minima with plain loops, which
NumPy does vectorised. The occlusion rule looks only along a point's own pixel row and column, so the rewrite
lets each point scan those and stop at the first hit
([`occlusion_reason`, the `half_width == 0` path](../cpp/include/vggt_geom/geometry.hpp#L186)). The rewritten
version was not timed again on Colab.

The step it speeds up was 5% of a scene's time (table below). At 2.2x that is roughly one second per scene and
model, a few minutes over the whole project. The C++ core is in the pipeline because it is correct, tested and
free to use; its main value is being a second, independently written implementation that agrees exactly. The
time was won elsewhere.

## Where one scene's time went before any speed-up

Measured on Colab, one block of 20 scenes, sequential (`notebooks/development/06_cpp_core.ipynb`, recorded in
`docs/decisions.md`):

| step | seconds per scene | share |
|---|---|---|
| download and unpack (the scene's share of its block) | 34.9 | 66% |
| score the scene: five alignments, all strata | 8.0 | 15% |
| moving-object labels from boxes | 3.4 | 7% |
| model inference (A100) | 3.0 | 6% |
| project and occlusion test (the step C++ speeds up) | 2.4 | 5% |
| read 240 LiDAR files | 0.8 | 1.5% |

## What each change saved

| change | where in the code | measured | checked how |
|---|---|---|---|
| score scenes in parallel processes | [`pipeline.run_jobs`](../src/vggt_aura/pipeline.py#L479) | 4.8x on 8 cores (1,040 s to 217 s for a block) | identical numbers, `experiments/parallel_scoring_identical` |
| decompress LiDAR archives on all cores | [`fast_download.py`](../src/vggt_aura/fast_download.py#L93) | 5.3x (419 s to 79 s for a 3.8 GB archive) | every unpacked file byte-identical (SHA-256), `experiments/parallel_unpack_speed_test`; checksum before unpacking; automatic fall-back |
| GPU and CPU work in separate notebooks | `02_predict_gpu`, `03_score` | the GPU is busy about 70 s of a block; a CPU server does the rest | same result files |
| do not zip-compress predicted depth | [`inference.save_predictions`](../src/vggt_aura/inference.py#L138) | 3.4 s to 0.1 s per save (laptop); GPU block 5.3 to 3.9 min | stored numbers bit-identical (test) |
| read the next scene's images while the GPU works | [`pipeline.prepared_ahead`](../src/vggt_aura/pipeline.py#L99) | GPU waits 3 s per block instead of 68 s | same upstream function, same tensors |
| keep both models loaded across blocks | `predict_block(runners=...)` | 26 s per block; GPU block 3.9 to 2.1 min with the line above | |
| fetch saved predictions during the download | [`pipeline.warm_files`](../src/vggt_aura/pipeline.py#L155) | wait before scoring 77 s to 1 s | reads only |
| checksum all archives at once; validator in the background | [`fast_download.sha256_many`](../src/vggt_aura/fast_download.py#L83), [`pipeline._Background`](../src/vggt_aura/pipeline.py#L131) | heavy block 12 to 17 min down to about 9 (together with the two lines above) | validator verdict still collected before any result is written |
| two servers share one block list | [`pipeline.claim_block`](../src/vggt_aura/pipeline.py#L196) | about half the wall-clock time | no block scored twice in either run |
| C++ geometry core | see above | about 1 s per scene and model | exact equality |

End to end: a block of 20 scenes went from 20.4 minutes to 6.5 on one CPU server, and a GPU block from 5.3
minutes to 2.1.

Tried and not adopted: keeping a copy of each block on Google Drive instead of downloading it again
(`experiments/drive_cache_speed_test`): faster in one same-session test, not convincing enough to build on.

## Not done

- Downloading and unpacking the next block while the current one is scored: the largest lever left (perhaps a
  third of a block), not built because two 16 GB blocks on disk at once can fill a CPU server's disk.
- The remaining scoring time is the evaluation itself and ground-truth building (step timers, test block 0:
  evaluate 438, ground truth 379, labels 177 worker-seconds for 20 scenes and 2 models). Moving-object labelling
  in C++ would be the next candidate; it was left alone because the C++ scope was fixed on purpose.
