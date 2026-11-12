"""python -m dfd.web --adapter mock|real [--checkpoint DIR] [--port 8000]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local web page for the deepfake detector (127.0.0.1 only).")
    ap.add_argument("--adapter", choices=["mock", "real"], required=True,
                    help="mock results are NOT real and are shown under a MOCK banner; real needs a checkpoint")
    ap.add_argument("--checkpoint", default=None, help="checkpoint folder holding metadata.json (real adapter)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    from dfd.infer.factory import make_adapter
    from dfd.infer.checkpoint import CheckpointError, NoCheckpointError
    try:
        adapter = make_adapter(a.adapter, a.checkpoint)
    except (NoCheckpointError, CheckpointError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    from dfd.web.app import create_app
    if adapter.inference_source != "real":
        print("WARNING: running with the MOCK adapter; every result is a stand-in and is shown under a MOCK banner.", file=sys.stderr)
    print(f"Open http://127.0.0.1:{a.port}/ in a normal browser window (Chrome, Edge or Firefox). "
          "Use the address 127.0.0.1, not localhost, and not an embedded preview pane.", file=sys.stderr)
    create_app(adapter).run(host="127.0.0.1", port=a.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
