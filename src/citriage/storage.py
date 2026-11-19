from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import ValidationError

from citriage.models import RunMeta
from citriage.redact import redact_strings


def repo_dir(out: Path, repo: str) -> Path:
    return out / repo.replace("/", "__")


def run_dir(out: Path, repo: str, run_id: int, *, attempt: int | None = None) -> Path:
    name = f"{run_id}_a{attempt}" if attempt is not None else str(run_id)
    return repo_dir(out, repo) / name


def read_run_meta(path: Path) -> RunMeta | None:
    try:
        raw = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return RunMeta.model_validate(raw)
    except ValidationError:
        return None


def write_run(
    path: Path,
    meta: RunMeta,
    logs: dict[str, bytes],
    *,
    extra_secrets: tuple[str, ...],
) -> bool:
    if path.exists():
        return False

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=parent))
    try:
        for name, content in logs.items():
            (temp_path / name).write_bytes(content)

        dumped = meta.model_dump(mode="json")
        redacted = redact_strings(dumped, extra_secrets)
        if not isinstance(redacted, dict):
            raise TypeError("meta dump is not an object")
        meta_text = json.dumps(redacted, indent=2, ensure_ascii=False) + "\n"
        (temp_path / "meta.json").write_text(meta_text, encoding="utf-8")

        if path.exists():
            return False
        try:
            os.replace(temp_path, path)
        except OSError:
            if path.exists():
                return False
            raise
        return True
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
