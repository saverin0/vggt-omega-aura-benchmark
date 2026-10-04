import pytest

from vggt_aura.persistence import Manifest, resolve_persist_root

SCENE_A = "2026-05-28-16-26-51|94"  # real AURA scene IDs contain a pipe
SCENE_B = "2026-05-28-16-26-51|19"


def test_new_manifest_is_empty(tmp_path):
    m = Manifest(tmp_path / "manifest" / "completed.jsonl")
    assert m.done("predict") == set()
    assert not m.is_done(SCENE_A, "predict")


def test_mark_done_survives_a_new_session(tmp_path):
    path = tmp_path / "completed.jsonl"
    Manifest(path).mark_done(SCENE_A, "predict", checkpoint="x.pt")
    resumed = Manifest(path)  # a fresh object stands in for a fresh session
    assert resumed.is_done(SCENE_A, "predict")
    assert not resumed.is_done(SCENE_B, "predict")


def test_stages_are_independent(tmp_path):
    m = Manifest(tmp_path / "completed.jsonl")
    m.mark_done(SCENE_A, "predict")
    assert not m.is_done(SCENE_A, "metrics")


def test_pending_keeps_order_and_skips_done(tmp_path):
    m = Manifest(tmp_path / "completed.jsonl")
    m.mark_done(SCENE_A, "predict")
    assert m.pending([SCENE_B, SCENE_A], "predict") == [SCENE_B]


def test_truncated_last_line_is_ignored(tmp_path):
    path = tmp_path / "completed.jsonl"
    Manifest(path).mark_done(SCENE_A, "predict")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"scene_id": "2026-05-28-16-26-51|19", "sta')  # killed mid-write
    assert Manifest(path).done("predict") == {SCENE_A}


def test_marking_twice_is_harmless(tmp_path):
    m = Manifest(tmp_path / "completed.jsonl")
    m.mark_done(SCENE_A, "predict")
    m.mark_done(SCENE_A, "predict")
    assert m.done("predict") == {SCENE_A}


def test_runtime_mode_uses_results_dir(tmp_path):
    root = resolve_persist_root("runtime", project_root=tmp_path)
    assert root == tmp_path / "results"
    assert root.is_dir()


def test_drive_mode_fails_loudly_when_not_mounted(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_persist_root("drive", project_root=tmp_path, drive_root=tmp_path / "nodrive" / "proj")


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_persist_root("s3", project_root=tmp_path)
