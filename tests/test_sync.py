import base64
import io
import json
import zipfile

from vggt_aura.sync import MARKER, build_payload, collect_files, main, render_sync_cell, update_notebook


def make_project(root):
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "cpp").mkdir()
    (root / "notebooks").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "src" / "pkg" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "pkg" / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    (root / "src" / "pkg" / ".env").write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
    (root / "cpp" / "geometry.hpp").write_text("// header\n", encoding="utf-8")
    (root / ".env").write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
    return root


def make_notebook(path, with_sync_cell=True):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# title\n"]}]
    if with_sync_cell:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": 3,
                      "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}],
                      "source": [MARKER + "\n", "print('placeholder')\n"]})
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": ["print('user cell')\n"]})
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")


def unpack(payload):
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    return {name: archive.read(name) for name in archive.namelist()}


def test_collects_code_and_skips_junk_and_secrets(tmp_path):
    names = [p.as_posix() for p in collect_files(make_project(tmp_path))]
    assert names == ["cpp/geometry.hpp", "pyproject.toml", "src/pkg/__init__.py"]


def test_payload_never_contains_a_secret(tmp_path):
    payload, _, _ = build_payload(make_project(tmp_path))
    assert all(b"hf_secret" not in data for data in unpack(payload).values())


def test_payload_round_trips(tmp_path):
    payload, _, count = build_payload(make_project(tmp_path))
    files = unpack(payload)
    assert count == 3
    assert files["src/pkg/__init__.py"] == b"VALUE = 1\n"


def test_hash_is_stable_and_changes_with_code(tmp_path):
    root = make_project(tmp_path)
    first = build_payload(root)[1]
    assert build_payload(root)[1] == first
    (root / "src" / "pkg" / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert build_payload(root)[1] != first


def test_hash_ignores_windows_line_endings(tmp_path):
    root = make_project(tmp_path)
    unix = build_payload(root)[1]
    (root / "src" / "pkg" / "__init__.py").write_bytes(b"VALUE = 1\r\n")
    assert build_payload(root)[1] == unix


def test_rendered_cell_is_valid_python_and_unpacks(tmp_path, monkeypatch):
    payload, digest, count = build_payload(make_project(tmp_path / "project"))
    source = render_sync_cell(payload, digest, count)
    compile(source, "<sync cell>", "exec")
    assert source.startswith(MARKER)
    # The payload embedded in the cell text must decode to the same files.
    embedded = source.split('PAYLOAD = """')[1].split('"""')[0]
    assert unpack(embedded)["cpp/geometry.hpp"] == b"// header\n"


def test_update_replaces_only_the_sync_cell(tmp_path):
    notebook = tmp_path / "nb.ipynb"
    make_notebook(notebook)
    assert update_notebook(notebook, MARKER + "\nprint('new')") == "updated"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    assert "".join(cells[1]["source"]) == MARKER + "\nprint('new')"
    assert cells[1]["outputs"] == [] and cells[1]["execution_count"] is None
    assert "".join(cells[2]["source"]) == "print('user cell')\n"
    assert update_notebook(notebook, MARKER + "\nprint('new')") == "current"


def test_notebook_without_sync_cell_is_left_alone(tmp_path):
    notebook = tmp_path / "nb.ipynb"
    make_notebook(notebook, with_sync_cell=False)
    before = notebook.read_text(encoding="utf-8")
    assert update_notebook(notebook, MARKER + "\nprint('new')") == "no-sync-cell"
    assert notebook.read_text(encoding="utf-8") == before


def test_check_mode_detects_stale_code_without_writing(tmp_path):
    root = make_project(tmp_path)
    notebook = root / "notebooks" / "nb.ipynb"
    make_notebook(notebook)
    assert main(["--root", str(root), "--check"]) == 1      # placeholder cell is stale
    assert "placeholder" in notebook.read_text(encoding="utf-8")  # and was not touched
    assert main(["--root", str(root)]) == 0                 # write it
    assert main(["--root", str(root), "--check"]) == 0      # now current
    (root / "src" / "pkg" / "__init__.py").write_text("VALUE = 9\n", encoding="utf-8")
    assert main(["--root", str(root), "--check"]) == 1      # code changed, stale again


def test_sub_folders_are_synced_and_clear_empties_every_cell(tmp_path, capsys):
    import json

    from vggt_aura import sync

    root = tmp_path
    (root / "pyproject.toml").write_text("[project]" + chr(10) + "name='x'" + chr(10), encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("A = 1" + chr(10), encoding="utf-8")
    for folder in ("notebooks", "notebooks/development"):
        (root / folder).mkdir()
        cell = {"cell_type": "code", "metadata": {}, "execution_count": 3, "outputs": [{"output_type": "stream", "text": ["old"]}],
                "source": [sync.MARKER + chr(10), "old payload"]}
        (root / folder / "nb.ipynb").write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    assert sync.main(["--root", str(root)]) == 0
    assert "development/nb.ipynb: updated" in capsys.readouterr().out
    assert sync.main(["--root", str(root), "--check"]) == 0
    assert sync.main(["--root", str(root), "--clear"]) == 0
    for folder in ("notebooks", "notebooks/development"):
        cell = json.loads((root / folder / "nb.ipynb").read_text(encoding="utf-8"))["cells"][0]
        source = "".join(cell["source"])
        assert source == sync.EMPTY_SYNC_CELL and "PAYLOAD" not in source and cell["outputs"] == []
        compile(source, "cell", "exec")
    assert sync.main(["--root", str(root), "--check"]) == 1                     # emptied notebooks are, rightly, not current


def test_the_sync_cell_hands_its_code_hash_to_the_session():
    from vggt_aura import sync

    cell = sync.render_sync_cell("QUJD", "0123456789ab", 3)
    compile(cell, "sync cell", "exec")
    assert 'os.environ["VGGT_AURA_CODE_HASH"] = "0123456789ab"' in cell
