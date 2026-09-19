"""The all-cores decompressor must give exactly what Python's own decompressor gives, and must never
unpack a file that fails its checksum. The toolkit itself is faked here; the real end-to-end check is
notebook `parallel_unpack_speed_test` (byte-identical on a real 3.8 GB LiDAR file) plus the toolkit's validator in `03_score`.
"""

import io
import lzma
import sys
import tarfile
import types

import pandas as pd
import pytest

from vggt_aura import aura_data as ad, fast_download as fd

needs_xz = pytest.mark.skipif(not fd.parallel_xz_available(), reason="no xz tool of version 5.4 or newer on PATH")


def make_tar_xz(path, files, multi_block=True):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    data = raw.getvalue()
    if multi_block:                                    # several independent streams, as a threaded compressor writes
        third = len(data) // 3
        packed = b"".join(lzma.compress(piece) for piece in (data[:third], data[third:2 * third], data[2 * third:]))
    else:
        packed = lzma.compress(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    return data


FILES = {"val/scene_a/lidar/top_left/0001.pcd": bytes(range(256)) * 400,
         "val/scene_a/lidar/top_right/0001.pcd": b"\x00\x01" * 70000,
         "val/scene_b/lidar/top_left/0002.pcd": b"point cloud " * 9000}


def test_version_rule():
    assert (5, 4) >= fd.MIN_XZ_VERSION and (5, 8) >= fd.MIN_XZ_VERSION and not (5, 2) >= fd.MIN_XZ_VERSION


@needs_xz
@pytest.mark.parametrize("multi_block", [True, False])
def test_decompressed_tar_is_byte_identical_and_readable(tmp_path, multi_block):
    archive = tmp_path / "part-000000.tar.xz"
    plain = make_tar_xz(archive, FILES, multi_block)
    sha = fd.sha256_file(archive)
    assert fd.decompress_in_place(archive, sha) == "decompressed"
    assert archive.read_bytes() == plain and not fd.is_xz(archive)
    with tarfile.open(archive, "r:*") as tar:          # how the toolkit opens it: the name no longer matters
        assert {m.name: tar.extractfile(m).read() for m in tar} == FILES
    assert fd.decompress_in_place(archive, sha) == "already done"          # an interrupted run can simply be repeated
    assert not list(tmp_path.glob(".*tmp*"))


@needs_xz
def test_a_wrong_checksum_is_never_unpacked(tmp_path):
    archive = tmp_path / "part-000000.tar.xz"
    make_tar_xz(archive, FILES)
    before = archive.read_bytes()
    with pytest.raises(RuntimeError, match="SHA-256"):
        fd.decompress_in_place(archive, "0" * 64)
    assert archive.read_bytes() == before and not fd.marker_for(archive).exists()


def test_an_unverified_plain_file_is_refused(tmp_path):
    archive = tmp_path / "part-000000.tar.xz"
    archive.write_bytes(b"this is not xz and nobody verified it")
    with pytest.raises(RuntimeError, match="never verified"):
        fd.decompress_in_place(archive, "0" * 64)


@needs_xz
def test_whole_block_with_a_faked_toolkit(tmp_path, monkeypatch):
    """Order of work, checksums for every archive, the toolkit's extract called without its own verify, markers gone."""
    root = tmp_path / "root"
    lidar, camera = "val/lidar/part-000000.tar.xz", "val/camera/part-000000.tar"
    calls = []

    class FakeDownloader:
        def __init__(self, output_dir, revision=None):
            self.output_dir, self.metadata_dir = root, root / "metadata" / "v1.0"

        def fetch_metadata(self, max_workers=8):
            self.metadata_dir.mkdir(parents=True)
            make_tar_xz(root / lidar, FILES)
            (root / camera).parent.mkdir(parents=True)
            (root / camera).write_bytes(b"plain camera tar")
            pd.DataFrame({"chunk_path": [lidar, camera],
                          "sha256": [fd.sha256_file(root / lidar), fd.sha256_file(root / camera)]}
                         ).to_parquet(self.metadata_dir / "chunks.parquet")

        def select(self, layers, splits, scene_ids):
            calls.append(("select", tuple(layers), tuple(splits), tuple(scene_ids)))
            return types.SimpleNamespace(chunk_paths=(lidar, camera))

        def download(self, selection, max_workers=8):
            calls.append(("download",))

        def extract(self, selection, jobs, verify, delete_archives):
            assert not fd.is_xz(root / lidar) and verify is False and delete_archives is True
            with tarfile.open(root / lidar, "r:*") as tar:
                assert {m.name: tar.extractfile(m).read() for m in tar} == FILES
            calls.append(("extract",))

    monkeypatch.setitem(sys.modules, "fzi_aura.download", types.SimpleNamespace(FZIAURADownloader=FakeDownloader))
    summary = fd.download_block_fast(root, "val", ["s1", "s2"], ["lidar_x", "camera_y"], "rev")
    assert [c[0] for c in calls] == ["select", "download", "extract"]
    assert calls[0][1] == ("base_keyframes", "lidar_x", "camera_y") and calls[0][3] == ("s1", "s2")
    assert summary["archives"] == 2 and summary["xz_decompressed_on_all_cores"] == 1
    assert not list(root.rglob("*" + fd.MARKER_SUFFIX))


@needs_xz
def test_a_failed_fast_unpack_cleans_up_and_falls_back(tmp_path, monkeypatch, capsys):
    swapped, untouched = tmp_path / "chunks" / "a.tar.xz", tmp_path / "chunks" / "b.tar.xz"
    make_tar_xz(swapped, FILES)
    make_tar_xz(untouched, FILES)

    fd.decompress_in_place(swapped, fd.sha256_file(swapped))     # the state a block is in when it breaks half way

    def breaks_half_way(data_root, *args):
        raise OSError("disk full")

    ran = []
    monkeypatch.setattr(fd, "parallel_xz_available", lambda: True)
    monkeypatch.setattr(fd, "download_block_fast", breaks_half_way)
    monkeypatch.setattr(ad.subprocess, "run", lambda command, check: ran.append(command) or types.SimpleNamespace(returncode=0))
    assert ad.download_block(tmp_path, "val", 11, ["s1"], ["lidar_motion_compensated_keyframes"]) == 0
    assert not swapped.exists() and fd.is_xz(untouched)          # the toolkit re-downloads the one, keeps the other
    assert not list(tmp_path.rglob("*" + fd.MARKER_SUFFIX))
    assert len(ran) == 1 and "FAST UNPACK FAILED" in capsys.readouterr().out


def test_download_block_falls_back_when_xz_is_missing(tmp_path, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(fd, "parallel_xz_available", lambda: False)
    monkeypatch.setattr(ad.subprocess, "run", lambda command, check: ran.append(command) or types.SimpleNamespace(returncode=0))
    assert ad.download_block(tmp_path, "val", 11, ["s1"], ["camera_keyframes"]) == 0
    assert len(ran) == 1 and "fzi_aura.download" in ran[0] and "--verify" in ran[0]
    assert "single-core" in capsys.readouterr().out
    with pytest.raises(ValueError):
        ad.download_block(tmp_path, "val", 11, ["s1"], ["camera_keyframes"], fast_unpack="sometimes")


def test_checksums_of_many_files_at_once_equal_one_by_one(tmp_path):
    paths = []
    for k in range(6):
        path = tmp_path / f"a{k}.tar"
        path.write_bytes(bytes([k]) * (300_000 + k))
        paths.append(path)
    assert fd.sha256_many(paths, threads=3) == {path: fd.sha256_file(path) for path in paths}
    assert fd.sha256_many([]) == {}
    with pytest.raises(RuntimeError, match="SHA-256"):
        fd.verify_archive(paths[0], fd.sha256_file(paths[1]), actual_sha256=fd.sha256_file(paths[0]))
