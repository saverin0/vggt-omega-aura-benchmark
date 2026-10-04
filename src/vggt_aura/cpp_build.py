"""Build and load the C++ geometry core (cpp/), a pybind11 module named vggt_geom_cpp.

    python -m vggt_aura.cpp_build          # build, then import and report

Two build routes, tried in this order:

1. CMake, the proper route: cpp/CMakeLists.txt describes the project, and
   CMake generates the real compiler commands for whatever machine it is on.
2. One direct compiler command, as a fallback when CMake or the Python
   development files it looks for are missing. It is the same command CMake
   would end up running, written out by hand.

The module is OPTIONAL everywhere. Python stays the reference implementation,
so a failed build is reported and never fatal.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

MODULE_NAME = "vggt_geom_cpp"


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def build_dir(root=None) -> Path:
    return Path(root or project_dir()) / "cpp" / "build"


def _run(command, cwd=None) -> bool:
    print("$", " ".join(str(part) for part in command))
    result = subprocess.run([str(part) for part in command], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print((result.stdout + result.stderr)[-3000:])
    return result.returncode == 0


def _build_with_cmake(root: Path) -> bool:
    if shutil.which("cmake") is None:
        print("cmake not found")
        return False
    import pybind11

    out = build_dir(root)
    configure = ["cmake", "-S", root / "cpp", "-B", out, "-DCMAKE_BUILD_TYPE=Release",
                 f"-Dpybind11_DIR={pybind11.get_cmake_dir()}", f"-DPython_EXECUTABLE={sys.executable}"]
    return _run(configure) and _run(["cmake", "--build", out, "--config", "Release", "-j"])


def _build_directly(root: Path) -> bool:
    compiler = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    if compiler is None:
        print("no C++ compiler found (looked for c++, g++, clang++)")
        return False
    import pybind11

    out = build_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (MODULE_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, "-O2", "-shared", "-std=c++17", "-fPIC", "-ffp-contract=off", "-Wall", "-Wextra",
               f"-I{sysconfig.get_paths()['include']}", f"-I{pybind11.get_include()}", f"-I{root / 'cpp' / 'include'}",
               root / "cpp" / "src" / "bindings.cpp", "-o", target]
    return _run(command)


def build(root=None) -> bool:
    """Build the module into cpp/build. Returns True when a usable module exists afterwards."""
    root = Path(root or project_dir())
    if not (root / "cpp" / "src" / "bindings.cpp").is_file():
        print("cpp/ has no sources, nothing to build")
        return False
    if importlib.util.find_spec("pybind11") is None:
        print("pybind11 is not installed: pip install pybind11")
        return False
    built = _build_with_cmake(root) or _build_directly(root)
    importlib.invalidate_caches()
    return built and import_module(root) is not None


def import_module(root=None):
    """The compiled module, or None. Looks in cpp/build and in $VGGT_GEOM_CPP_DIR."""
    folders = [build_dir(root), build_dir(root) / "Release"]
    if os.environ.get("VGGT_GEOM_CPP_DIR"):
        folders.insert(0, Path(os.environ["VGGT_GEOM_CPP_DIR"]))
    for folder in folders:
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(MODULE_NAME)
    except ImportError:
        return None


def main() -> int:
    ok = build()
    module = import_module()
    print("built and importable:" if ok and module else "NOT available:", getattr(module, "__file__", None))
    return 0 if module else 1


if __name__ == "__main__":
    sys.exit(main())
