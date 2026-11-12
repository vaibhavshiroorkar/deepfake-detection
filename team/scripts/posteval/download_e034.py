"""E-034: download the shards of data/_manifests/posteval/e034_cfeval.shards.json into data/cf_eval_e034/ (protocol: docs/EXPERIMENTS.md E-034).

    python scripts/posteval/download_e034.py [connections]

Same method as scripts/modern_image/download_ranged.py (HTTP Range chunks of 32 MiB over several connections, pinned revision) but a separate
target folder, so nothing of E-028 is written. Each shard is checked against the Hub LFS SHA-256 and size recorded in the shard list before it
is moved into place and logged to data/cf_eval_e034/_meta/download_log.jsonl; a mismatching file is deleted. Downloading opens no image.
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SEL = ROOT / "data" / "_manifests" / "posteval" / "e034_cfeval.shards.json"
BASE = ROOT / "data" / "cf_eval_e034"
LOG = BASE / "_meta" / "download_log.jsonl"
CH = 32 * 1024 * 1024


def main() -> int:
    if not (ROOT / "data" / "_results" / "posteval" / "E-034" / "prereg.json").exists():
        sys.exit("refused: preregistration record of E-034 is missing")
    conns = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sel = json.load(open(SEL))
    (BASE / "_meta").mkdir(parents=True, exist_ok=True)
    (BASE / "shards" / "data").mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["file"] for l in open(LOG) if l.strip() and json.loads(l)["match"]} if LOG.exists() else set()
    todo = [s for s in sel["shards"] if s["file"] not in done]
    print(f"{len(sel['shards'])} shards, {len(done)} verified, {len(todo)} to fetch", flush=True)

    def url(s):
        return f"https://huggingface.co/datasets/{sel['repo']}/resolve/{sel['revision']}/{s['file']}"

    def chunk(s, k):
        lo, hi = k * CH, min((k + 1) * CH, s["size"]) - 1
        for attempt in range(10):
            try:
                r = requests.get(url(s), headers={"Range": f"bytes={lo}-{hi}"}, timeout=60, allow_redirects=True)
                if r.status_code != 206 or len(r.content) != hi - lo + 1:
                    raise RuntimeError(f"status {r.status_code}, {len(r.content)} bytes")
                return k, r.content
            except Exception as e:  # noqa: BLE001
                time.sleep(2 + 2 * attempt)
                last = e
        raise RuntimeError(f"{s['file']} chunk {k}: {last}")

    t0, got = time.time(), 0
    with ThreadPoolExecutor(conns) as ex:
        for s in todo:
            n = (s["size"] + CH - 1) // CH
            dst = BASE / "shards" / s["file"]
            part = dst.with_suffix(".part")
            with open(part, "wb") as f:
                f.truncate(s["size"])
                for k, buf in ex.map(lambda k: chunk(s, k), range(n)):
                    f.seek(k * CH)
                    f.write(buf)
            h = hashlib.sha256(part.read_bytes()).hexdigest()
            ok = h == s["sha256"] and part.stat().st_size == s["size"]
            if ok:
                part.replace(dst)
            else:
                part.unlink()
            with open(LOG, "a") as f:
                f.write(json.dumps({"file": s["file"], "size": s["size"], "expected_sha256": s["sha256"], "local_sha256": h, "match": ok,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "method": "ranged"}) + "\n")
            got += s["size"]
            print(f"{s['file']} match={ok} {got / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
