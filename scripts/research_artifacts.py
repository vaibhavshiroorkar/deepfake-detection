"""Content-bound completion receipts for research orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_payload_digests(inputs: list[Path]) -> dict[str, str]:
    """Bind cache bytes, restricting the inventory to supplied manifest rows."""
    indexes = []
    selected = set()
    for path in inputs:
        if path.suffix.lower() != ".csv":
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            fields = set(rows.fieldnames or ())
            if {"clip_id", "cache_path"} <= fields:
                indexes.append((path, list(rows)))
            elif "clip_id" in fields:
                selected.update(row["clip_id"] for row in rows)
    files = set()
    for index, rows in indexes:
        for row in rows:
            if selected and row["clip_id"] not in selected:
                continue
            path = Path(row["cache_path"])
            files.add(
                path.resolve()
                if path.is_absolute()
                else (index.parent / path).resolve()
            )
    return {str(path): sha256(path) for path in sorted(files)}


def completed(receipt: Path, configuration: dict, outputs: list[Path]) -> bool:
    """Reject partial or changed outputs rather than overwriting prior evidence."""
    if not receipt.exists():
        if any(path.exists() for path in outputs):
            raise ValueError(
                f"Unverified outputs have no completion receipt: {receipt}"
            )
        return False
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        payload.get("configuration") != configuration
        or payload.get("outputs")
        != {str(path): sha256(path) for path in outputs if path.is_file()}
        or not all(path.is_file() for path in outputs)
        or payload.get("cache_payloads", {})
        != cache_payload_digests([path for path in outputs if path.is_file()])
    ):
        raise ValueError(f"Completion receipt mismatch: {receipt}; use a new run path")
    return True


def record_completion(receipt: Path, configuration: dict, outputs: list[Path]) -> None:
    """Write only after every required artifact exists."""
    payload = {
        "configuration": configuration,
        "outputs": {str(path): sha256(path) for path in outputs},
        "cache_payloads": cache_payload_digests(outputs),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_stage(
    receipt: Path, command: list[str], outputs: list[Path], inputs: list[Path]
) -> int:
    root = Path(__file__).resolve().parents[1]
    configuration = {
        "command": command,
        "inputs": {str(path): sha256(path) for path in inputs},
        "cache_payloads": cache_payload_digests(inputs),
        "implementation": {
            path.relative_to(root).as_posix(): sha256(path)
            for directory in (root / "src", root / "scripts")
            for path in sorted(directory.rglob("*.py"))
        },
    }
    if completed(receipt, configuration, outputs):
        print(f"Verified completed stage: {receipt}", flush=True)
        return 0
    # The operator supplies an argument vector. No shell interprets its values.
    code = subprocess.run(command, check=False).returncode  # noqa: S603
    if code == 0:
        record_completion(receipt, configuration, outputs)
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    return run_stage(args.receipt, command, args.outputs, args.inputs)


if __name__ == "__main__":
    sys.exit(main())
