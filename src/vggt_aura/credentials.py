"""Find a secret without ever printing it.

Lookup order, first hit wins, cheapest first:

1. An environment variable of the same name.
2. A KEY=VALUE line in a .env file (gitignored).
3. Colab secrets (google.colab.userdata), only if asked for.
4. An interactive hidden prompt, if allowed.

Colab secrets are off by default. Measured on 2026-09-18: through the VS Code
Colab extension, userdata.get() raises TimeoutException after a long wait, so
trying it first would stall every session. A .env kept on Google Drive is the
route that works: it survives sessions and needs no typing.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(path) -> dict[str, str]:
    """Read KEY=VALUE lines. Blank lines, comments and empty values are skipped."""
    path = Path(path)
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


def _from_colab(name: str):
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        return userdata.get(name) or None
    except Exception:
        # Secret missing, or notebook access to it not granted.
        return None


def load_secret(name: str, env_files=(), required: bool = True, prompt: bool = False,
                use_colab_secrets: bool = False):
    """Return the secret's value, or None if absent and not required."""
    value = os.environ.get(name) or None
    if value is None:
        for env_file in env_files:
            value = parse_env_file(env_file).get(name)
            if value:
                break
    if value is None and use_colab_secrets:
        value = _from_colab(name)
    if value is None and prompt:
        from getpass import getpass
        value = getpass(f"{name} (input hidden): ").strip() or None
    if value is None and required:
        searched = ", ".join(str(f) for f in env_files) or "none"
        raise RuntimeError(
            f"Secret {name} not found in the environment or in .env files ({searched})."
        )
    return value
