from __future__ import annotations

from pathlib import Path

from citriage.parse.logparse import parse_log


def main() -> int:
    root = Path("data/raw")
    files = sorted(root.rglob("*.log"))
    errors: list[tuple[Path, str]] = []
    for path in files:
        try:
            parse_log(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            errors.append((path, f"{type(exc).__name__}: {exc}"))
    print(f"parsed {len(files)} logs; exceptions: {len(errors)}")
    for path, message in errors:
        print(f"  {path}: {message}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
