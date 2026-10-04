"""Start a working session on an ephemeral Colab runtime.

Every phase notebook calls start_session() right after its sync cell. It does
what notebooks/01_storage_and_session.ipynb does step by step: optional GPU check, Drive
mount, upstream installs at pinned commits, Hugging Face token, persist root
and resume manifest. Safe to call again in the same session: installs are
skipped when the packages are already importable.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import pins
from .credentials import load_secret
from .persistence import Manifest, resolve_persist_root

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/vggt-omega-aura-benchmark"
DEFAULT_DATA_ROOT = "/content/data/fzi-aura"

# import name -> pip arguments
UPSTREAM = {
    # --no-deps: upstream pins numpy<2, which would downgrade NumPy under the running kernel.
    "vggt_omega": ["--no-deps", f"git+{pins.VGGT_OMEGA_REPO}@{pins.VGGT_OMEGA_COMMIT}"],
    "einops": ["einops"],
    "safetensors": ["safetensors"],
    "pybind11": ["pybind11"],
    "fzi_aura": [f"fzi-aura[download] @ git+{pins.AURA_SDK_REPO}@{pins.AURA_SDK_COMMIT}"],
}


@dataclass
class Session:
    project_dir: Path
    persist_mode: str
    persist_root: Path     # survives the session in drive mode
    data_root: Path        # dataset on the runtime disk, wiped every session
    manifest: Manifest


def gpu_name():
    """Name and memory of the attached GPU, or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def missing_upstream(find_spec=importlib.util.find_spec) -> list[str]:
    """Import names from UPSTREAM that are not installed yet."""
    return [name for name in UPSTREAM if find_spec(name) is None]


def install_upstream() -> None:
    for name in missing_upstream():
        print("installing", name)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *UPSTREAM[name]], check=True)
    importlib.invalidate_caches()


def mount_drive() -> None:
    from google.colab import drive  # only exists on Colab
    drive.mount("/content/drive")


PROVENANCE_PACKAGES = ("numpy", "pandas", "torch", "fzi-aura", "vggt-omega", "vggt", "huggingface-hub")


def provenance_record(gpu=None) -> dict:
    """What a result depends on besides the inputs: the code, the pinned data and models, the versions, the machine.

    The code hash is the one the sync cell printed (first 12 hex digits of the SHA-256 of the zipped src/, cpp/,
    tests/ and pyproject.toml). It identifies a code version and contains nothing else.
    """
    versions = {}
    for name in PROVENANCE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_hash": os.environ.get("VGGT_AURA_CODE_HASH", "unknown (not started from a sync cell)"),
            "dataset_revision": pins.AURA_DATASET_REVISION, "sdk_commit": pins.AURA_SDK_COMMIT,
            "vggt_omega_commit": pins.VGGT_OMEGA_COMMIT, "vggt_omega_weights": pins.VGGT_OMEGA_HF_REVISION,
            "vggt_commit": pins.VGGT_COMMIT, "vggt_weights": pins.VGGT_HF_REVISION,
            "python": sys.version.split()[0], "packages": versions, "gpu": gpu or "none", "cpu_cores": os.cpu_count()}


def write_provenance(persist_root, record: dict) -> Path:
    """Append the record as one JSON line to <persist_root>/provenance.jsonl. Never fatal: a run is not lost over a log."""
    path = Path(persist_root) / "provenance.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + chr(10))
    except OSError as error:
        print("provenance not written:", error)
    return path


def start_session(persist_mode: str = "drive", drive_root: str = DEFAULT_DRIVE_ROOT,
                  data_root: str = DEFAULT_DATA_ROOT, require_gpu: bool = False, build_cpp: bool = False) -> Session:
    gpu = gpu_name()
    if gpu:
        print("GPU:", gpu)
    elif require_gpu:
        raise RuntimeError("No GPU attached. Change the runtime type to a GPU, reconnect, and re-run.")
    else:
        print("GPU: none (fine for download and inspection)")

    if persist_mode == "drive":
        mount_drive()
        Path(drive_root).mkdir(parents=True, exist_ok=True)

    install_upstream()

    os.environ["HF_TOKEN"] = load_secret(
        "HF_TOKEN", env_files=[Path(drive_root) / ".env", Path(DEFAULT_DRIVE_ROOT) / ".env", Path("/content/.env")], prompt=True
    )

    project_dir = Path(__file__).resolve().parents[2]
    persist_root = resolve_persist_root(persist_mode, project_root=project_dir, drive_root=drive_root)
    manifest = Manifest(persist_root / "manifest" / "completed_scenes.jsonl")
    data = Path(data_root)
    data.mkdir(parents=True, exist_ok=True)

    print("persist root:", persist_root)
    print("data root   :", data, "(runtime disk, wiped at session end)")
    record = provenance_record(gpu)
    print("provenance  :", write_provenance(persist_root, record).name, "| code", record["code_hash"],
          "| data", record["dataset_revision"][:12])
    if build_cpp:                                   # optional and never fatal: Python is the reference
        from . import cpp_build
        print("C++ core    :", "built" if cpp_build.build(project_dir) else "NOT built, continuing with Python only")
    return Session(project_dir, persist_mode, persist_root, data, manifest)


def install_extra(import_name: str, pip_args: list) -> None:
    """Install a package only some notebooks need. Skipped when it is already importable."""
    if importlib.util.find_spec(import_name) is None:
        print("installing", import_name)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip_args], check=True)
        importlib.invalidate_caches()


# --no-deps for the same reason as vggt_omega: upstream pins numpy<2 (and, in requirements.txt, an old torch).
VGGT_PIP_ARGS = ["--no-deps", f"git+{pins.VGGT_REPO}@{pins.VGGT_COMMIT}"]
