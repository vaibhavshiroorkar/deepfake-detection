"""CLI: build dataset manifests.

    python -m dfd.data.build_manifests --dataset ffpp
    python -m dfd.data.build_manifests --dataset all          # every implemented builder; stubs are listed

Writes data/_manifests/<key>.manifest.csv and data/_manifests/<key>.report.json (measured counts).
Exit code 0 when the manifest was written, 2 when the builder is a stub (layout not yet verified) or failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builders import LayoutNotVerified, asvspoof2019_la, celebdf, dfd, ffpp, forensynths, in_the_wild, stubs
from .manifest import default_data_root, write_manifest

BUILDERS = {
    "ffpp": ffpp.build,
    "asvspoof2019_la": asvspoof2019_la.build,
    "forensynths_progan_val": forensynths.build_progan_val,
    "forensynths_progan_test": forensynths.build_progan_test,
    "forensynths_cnn_subset": forensynths.build_cnn_subset,
    "diffusion_test": forensynths.build_diffusion_test,
    "celebdf_v2": celebdf.build,
    "dfd": dfd.build,
    "in_the_wild": in_the_wild.build,
    # stubs: layout not yet verified
    "celebdf_pp": celebdf.build_pp,
    "lavdf": stubs.build_lavdf,
    "fakeavceleb": stubs.build_fakeavceleb,
    "wilddeepfake": stubs.build_wilddeepfake,
}


def build_one(key: str, data_root: Path, out_dir: Path) -> dict:
    records, report = BUILDERS[key](data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = write_manifest(out_dir / f"{key}.manifest.csv", records)
    report["manifest_rows_written"] = n
    (out_dir / f"{key}.report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help=f"one of {sorted(BUILDERS)} or 'all'")
    ap.add_argument("--data-root", type=Path, default=None, help="default: <repo>/data")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <data-root>/_manifests")
    args = ap.parse_args(argv)
    data_root = (args.data_root or default_data_root()).resolve()
    out_dir = args.out_dir or (data_root / "_manifests")
    keys = sorted(BUILDERS) if args.dataset == "all" else [args.dataset]
    for k in keys:
        if k not in BUILDERS:
            print(f"unknown dataset {k!r}; choose from {sorted(BUILDERS)}", file=sys.stderr)
            return 2
    rc = 0
    for k in keys:
        try:
            rep = build_one(k, data_root, out_dir)
        except LayoutNotVerified as e:
            print(f"[{k}] STUB: {e}")
            rc = 2
            continue
        except (FileNotFoundError, ValueError) as e:
            print(f"[{k}] FAILED: {e}", file=sys.stderr)
            rc = 2
            continue
        print(f"[{k}] wrote {rep['manifest_rows_written']} rows; complete={rep.get('complete')}")
        print(json.dumps({a: b for a, b in rep.items() if a != "ignored_files"}, indent=2, sort_keys=True)[:3000])
    return rc


if __name__ == "__main__":
    sys.exit(main())
