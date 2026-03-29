#!/usr/bin/env python3
"""Generate or verify SHA-256 checksums for plot files.

Usage
-----
  # Create / update checksums file
  python scripts/checksum_plots.py

  # Verify existing checksums against current files
  python scripts/checksum_plots.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
CHECKSUM_FILE = PLOTS_DIR / "checksums.sha256"
GLOB_PATTERN = "*.png"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def generate() -> None:
    lines: list[str] = []
    for p in sorted(PLOTS_DIR.glob(GLOB_PATTERN)):
        digest = sha256(p)
        lines.append(f"{digest}  {p.name}")
        print(f"  {p.name}: {digest}")

    if not lines:
        print("No plot files found.", file=sys.stderr)
        sys.exit(1)

    CHECKSUM_FILE.write_text("\n".join(lines) + "\n")
    print(f"\n✔ Checksums written to {CHECKSUM_FILE.relative_to(PLOTS_DIR.parent)}")


def verify() -> None:
    if not CHECKSUM_FILE.exists():
        print(f"Checksum file not found: {CHECKSUM_FILE}", file=sys.stderr)
        sys.exit(1)

    ok, changed, missing = 0, 0, 0
    for line in CHECKSUM_FILE.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = PLOTS_DIR / name
        if not path.exists():
            print(f"  MISSING  {name}")
            missing += 1
            continue
        actual = sha256(path)
        if actual == expected:
            print(f"  OK       {name}")
            ok += 1
        else:
            print(f"  CHANGED  {name}")
            changed += 1

    print(f"\nResults: {ok} ok, {changed} changed, {missing} missing")
    if changed or missing:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot checksum manager")
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify current plots against saved checksums",
    )
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        generate()


if __name__ == "__main__":
    main()
