"""Re-hash every installed package file against the sha256 in its wheel RECORD.

Detects silent on-disk corruption (a damaged .dll / .pyd crashes Python with an
access violation instead of a readable error).

    .venv\\Scripts\\python scripts\\verify_install.py
"""
import base64
import csv
import hashlib
import importlib.metadata as md
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()


def main(only: set[str]) -> None:
    bad: dict[str, list[str]] = {}
    checked = 0
    for dist in md.distributions():
        if only and dist.metadata["Name"].lower() not in only:
            continue
        record = dist.read_text("RECORD")
        if not record:
            continue
        base = Path(dist.locate_file(""))
        for row in csv.reader(record.splitlines()):
            if len(row) < 2 or not row[1]:
                continue
            path = base / row[0]
            if not path.exists():
                continue
            checked += 1
            if file_hash(path) != row[1]:
                bad.setdefault(dist.metadata["Name"], []).append(row[0])
    print(f"checked {checked} files")
    if not bad:
        print("ALL FILES MATCH")
    for name, files in sorted(bad.items()):
        print(f"CORRUPT {name}: {len(files)} file(s)")
        for f in files[:8]:
            print(f"    {f}")


if __name__ == "__main__":
    import sys

    main({name.lower() for name in sys.argv[1:]})   # optional: only these packages
