#!/usr/bin/env python3
"""Run a group of training configs and write a manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run train.py. Defaults to current interpreter.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for config_path in args.configs:
        run_name = config_path.stem
        output_dir = args.output_root / run_name
        command = [
            args.python,
            str(ROOT / "train.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        started = _stamp()
        print(" ".join(command))
        if args.dry_run:
            returncode = 0
            status = "DRY_RUN"
        else:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            returncode = completed.returncode
            status = "PASS" if returncode == 0 else "FAIL"

        ended = _stamp()
        manifest_rows.append({
            "run_name": run_name,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "status": status,
            "returncode": returncode,
            "started_at": started,
            "ended_at": ended,
        })
        _write_manifest(args.output_root / "run_manifest.csv", manifest_rows)

        if returncode != 0 and not args.continue_on_error:
            raise SystemExit(returncode)

    print(f"manifest -> {args.output_root / 'run_manifest.csv'}")


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["run_name", "config", "output_dir", "status", "returncode", "started_at", "ended_at"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
