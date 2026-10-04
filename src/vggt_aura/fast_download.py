"""Download a block with the dataset toolkit, but decompress the LiDAR files on all cores.

Why: the LiDAR layer ships as `.tar.xz`. The toolkit decompresses each file on ONE
core, which was the largest single wait in a block. The files were compressed in
many independent pieces, so the `xz` command-line tool can decompress them on all
cores. Measured on Colab (notebook `parallel_unpack_speed_test`, val block 11, 8 cores): 7.0 min down to
1.3 min, every unpacked file byte-identical.

How, and what is NOT replaced: the toolkit still chooses the files, downloads them,
extracts every member with its own path-safety checks, and writes the dataset
metadata. Only the decompressor is swapped:

1. the toolkit downloads the archives;
2. every archive is checked against the SHA-256 in the release manifest (the same
   check the toolkit's `--verify` does);
3. each `.tar.xz` is decompressed by `xz -T0` into a plain tar that takes the
   archive's place. The toolkit opens archives with "r:*", which reads a plain tar
   whatever the file is called;
4. the toolkit extracts, deletes the archives, and writes the metadata.

Safe to interrupt: the swap in step 3 is atomic, and a marker file written just
before it lets a re-run tell a verified plain tar from an unverified download.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

XZ_MAGIC = b"\xfd7zXZ\x00"
MIN_XZ_VERSION = (5, 4)               # first version that decompresses on several threads
MARKER_SUFFIX = ".verified-plain-tar"


def xz_tool_version():
    """(major, minor) of the `xz` tool on PATH, or None when there is none."""
    tool = shutil.which("xz")
    if tool is None:
        return None
    try:
        text = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parallel_xz_available() -> bool:
    version = xz_tool_version()
    return version is not None and version >= MIN_XZ_VERSION


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for piece in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def is_xz(path) -> bool:
    with open(path, "rb") as stream:
        return stream.read(len(XZ_MAGIC)) == XZ_MAGIC


def marker_for(archive: Path) -> Path:
    return archive.with_name(archive.name + MARKER_SUFFIX)


def verify_archive(archive: Path, expected_sha256: str, actual_sha256: str | None = None) -> None:
    """`actual_sha256`: the file's checksum when it was already computed (sha256_many), else it is computed here."""
    if (actual_sha256 or sha256_file(archive)) != str(expected_sha256):
        raise RuntimeError(f"archive SHA-256 mismatch, not unpacking: {archive}")


def sha256_many(paths, threads: int = 4) -> dict:
    """Checksums of several files at once. hashlib releases the interpreter lock, so threads really run side by side;
    one after the other this was 40 to 60 s of a 16 GB block, on one core, with the others idle."""
    from concurrent.futures import ThreadPoolExecutor

    paths = [Path(p) for p in paths]
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        return dict(zip(paths, pool.map(sha256_file, paths)))


def decompress_in_place(archive, expected_sha256: str, threads: int = 0, actual_sha256: str | None = None) -> str:
    """Verify a `.tar.xz`, then replace it with its plain tar. Returns "decompressed" or "already done".

    threads=0 lets xz use every core.
    """
    archive = Path(archive)
    marker = marker_for(archive)
    if not is_xz(archive):
        if marker.is_file():
            return "already done"                      # verified and swapped in an earlier, interrupted run
        raise RuntimeError(f"not an xz file and never verified, delete it and download again: {archive}")
    verify_archive(archive, expected_sha256, actual_sha256)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            subprocess.run(["xz", f"-T{int(threads)}", "-dc", str(archive)], stdout=output, check=True)
        marker.write_text(str(expected_sha256), encoding="utf-8")
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "decompressed"


def remove_swapped_archives(data_root) -> int:
    """Delete every archive that was already turned into a plain tar, with its marker. Returns how many.

    Used before falling back to the toolkit's own way: its checksum test would reject a swapped file, and
    a missing file is simply downloaded again.
    """
    removed = 0
    for marker in Path(data_root).rglob("*" + MARKER_SUFFIX):
        archive = marker.with_name(marker.name[: -len(MARKER_SUFFIX)])
        if archive.is_file() and not is_xz(archive):
            archive.unlink()
            removed += 1
        marker.unlink()
    return removed


def download_block_fast(data_root, split: str, scene_ids, layers, revision: str, jobs: int = 8, threads: int = 0) -> dict:
    """The toolkit's download-and-extract, with `.tar.xz` files decompressed on all cores. See the module text."""
    from fzi_aura.download import FZIAURADownloader

    data_root = Path(data_root)
    layers = list(dict.fromkeys(["base_keyframes", *layers]))
    downloader = FZIAURADownloader(data_root, revision=revision)
    downloader.fetch_metadata(max_workers=jobs)
    selection = downloader.select(layers=layers, splits=[split], scene_ids=list(scene_ids))
    started = time.time()
    downloader.download(selection, max_workers=jobs)
    download_s = time.time() - started

    chunks = pd.read_parquet(downloader.metadata_dir / "chunks.parquet").set_index("chunk_path")
    started = time.time()
    n_xz = 0
    archives = {chunk_path: downloader.output_dir / chunk_path for chunk_path in selection.chunk_paths}
    # everything still in its downloaded form is checksummed now, all files at once; a file already swapped
    # for its plain tar by an interrupted run was verified then, and is recognised by its marker
    actual = sha256_many([a for p, a in archives.items() if not str(p).endswith(".tar.xz") or is_xz(a)])
    for chunk_path, archive in archives.items():
        expected = str(chunks.loc[chunk_path, "sha256"])
        if str(chunk_path).endswith(".tar.xz"):
            n_xz += decompress_in_place(archive, expected, threads, actual.get(archive)) == "decompressed"
        else:
            verify_archive(archive, expected, actual.get(archive))
    decompress_s = time.time() - started

    started = time.time()
    downloader.extract(selection, jobs=jobs, verify=False, delete_archives=True)    # verified above, file by file
    for chunk_path in selection.chunk_paths:
        marker_for(downloader.output_dir / chunk_path).unlink(missing_ok=True)
    summary = {"archives": len(selection.chunk_paths), "xz_decompressed_on_all_cores": n_xz,
               "download_s": round(download_s, 1), "verify_and_decompress_s": round(decompress_s, 1),
               "extract_s": round(time.time() - started, 1)}
    print("  fast unpack:", summary)
    return summary
