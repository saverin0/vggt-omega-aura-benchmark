"""Every session leaves a line saying which code, data, versions and machine produced its results."""

import json

from vggt_aura import pins, session


def test_provenance_names_the_code_the_data_and_the_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("VGGT_AURA_CODE_HASH", "0123456789ab")
    record = session.provenance_record(gpu="NVIDIA A100-SXM4-40GB, 40960 MiB")
    assert record["code_hash"] == "0123456789ab" and record["dataset_revision"] == pins.AURA_DATASET_REVISION
    assert record["vggt_omega_weights"] == pins.VGGT_OMEGA_HF_REVISION and "numpy" in record["packages"]
    assert record["gpu"].startswith("NVIDIA") and record["utc"].endswith("+00:00")
    path = session.write_provenance(tmp_path, record)
    session.write_provenance(tmp_path, {**record, "code_hash": "ba9876543210"})
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["code_hash"] for line in lines] == ["0123456789ab", "ba9876543210"]        # appended, never overwritten


def test_provenance_holds_no_secret_and_does_not_fail_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_never_be_written_anywhere")
    monkeypatch.delenv("VGGT_AURA_CODE_HASH", raising=False)
    record = session.provenance_record()
    assert "hf_this_must_never" not in json.dumps(record) and record["code_hash"].startswith("unknown")
    blocked = tmp_path / "a_file_not_a_folder"
    blocked.write_text("x", encoding="utf-8")
    session.write_provenance(blocked, record)                       # cannot create a folder there: prints, does not raise
