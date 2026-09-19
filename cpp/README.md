# C++ geometry core

A second implementation of the geometry, in plain C++17, exposed to Python as
the module `vggt_geom_cpp`. **Python stays the reference.** If the two ever
disagree, the C++ is wrong.

Scope (fixed): projection, unprojection, rigid transform composition,
quaternion to rotation, LiDAR-to-image projection with the occlusion check.
Numbers in, numbers out. No file I/O, no CUDA, no PCL, no OpenCV, no Eigen.

Where it runs: the pipeline builds the LiDAR ground truth with it when the module is built
(`build_ground_truth(..., backend="auto")` in `src/vggt_aura/ground_truth.py`) and with the Python reference
otherwise. What it saved, and the exact lines: [`docs/performance.md`](../docs/performance.md).

## Layout

- `include/vggt_geom/geometry.hpp`  the whole core. Header-only, zero dependencies.
- `src/bindings.cpp`                the only file that knows about Python (pybind11).
- `CMakeLists.txt`                  the build description.
- `build/`                          created by the build, never committed or synced.

Header-only means the geometry is written entirely in a `.hpp` file, so there is
no library to link: including the header is all it takes to use it.

## Build and test

    python -m vggt_aura.cpp_build                     # build, then report
    pytest tests/test_cpp_differential.py             # skipped when the module is not built

`cpp_build` tries CMake first and falls back to one direct compiler command. On
Colab it takes seconds. The build uses `-ffp-contract=off` so the compiler does
not fuse `a*b+c` into one rounded step; NumPy does not, and the tests compare
number for number.

## What the differential tests demand

Whole-number and yes/no results (pixel indices, occlusion verdicts, chosen
points) must be EXACTLY equal to the Python reference. Real-number results must
agree to floating-point tolerance: NumPy's matrix product and a plain C++ loop
may add the same three numbers in a different order.
