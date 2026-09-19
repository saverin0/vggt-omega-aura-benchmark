import pytest

from vggt_aura.credentials import load_secret, parse_env_file


def write_env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_skips_comments_blanks_and_empty_values(tmp_path):
    path = write_env(tmp_path, "# note\n\nHF_TOKEN=hf_abc\nGH_TOKEN=\n")
    assert parse_env_file(path) == {"HF_TOKEN": "hf_abc"}


def test_parse_strips_quotes_and_spaces(tmp_path):
    path = write_env(tmp_path, 'HF_TOKEN = "hf_abc" \n')
    assert parse_env_file(path) == {"HF_TOKEN": "hf_abc"}


def test_parse_keeps_equals_signs_inside_the_value(tmp_path):
    path = write_env(tmp_path, "HF_TOKEN=abc=def\n")
    assert parse_env_file(path) == {"HF_TOKEN": "abc=def"}


def test_missing_file_is_empty(tmp_path):
    assert parse_env_file(tmp_path / "nope.env") == {}


def test_environment_variable_wins_over_file(tmp_path, monkeypatch):
    path = write_env(tmp_path, "HF_TOKEN=from_file\n")
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert load_secret("HF_TOKEN", env_files=[path]) == "from_env"


def test_falls_back_to_file(tmp_path, monkeypatch):
    path = write_env(tmp_path, "HF_TOKEN=from_file\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_secret("HF_TOKEN", env_files=[path]) == "from_file"


def test_blank_template_counts_as_missing(tmp_path, monkeypatch):
    path = write_env(tmp_path, "HF_TOKEN=\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        load_secret("HF_TOKEN", env_files=[path])


def test_optional_secret_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert load_secret("GH_TOKEN", env_files=[tmp_path / ".env"], required=False) is None


def test_error_message_never_contains_a_value(tmp_path, monkeypatch):
    path = write_env(tmp_path, "OTHER=super_secret_value\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as err:
        load_secret("HF_TOKEN", env_files=[path])
    assert "super_secret_value" not in str(err.value)
