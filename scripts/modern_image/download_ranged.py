"""Parallel ranged download of the Community Forensics shards that download.py fetched too slowly (about 0.3 MB/s per connection).

Usage (repo root, PYTHONPATH=src): python scripts/modern_image/download_ranged.py eval|small [connections] [shard names...]
Same inputs and the same verification as download.py: the shards of data/cf_<x>/_meta/subset.json at the pinned revision, each checked against the
Hub's LFS SHA-256 before it is moved to data/cf_<x>/shards/ and logged to download_log.jsonl (same fields; a mismatching file is deleted).
Chunks of 32 MiB are fetched with HTTP Range requests over many connections; a part file plus a chunk manifest allow resuming.
With no shard names, every shard of the subset that is not yet logged as matching is downloaded, smallest first.
"""
import hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).parent))
import _env

which = sys.argv[1]
conns = int(sys.argv[2]) if len(sys.argv) > 2 else 14
names = sys.argv[3:]
base = _env.CF_EVAL if which == "eval" else _env.CF_SMALL
sub = json.load(open(base / "_meta" / "subset.json"))
logf = base / "_meta" / "download_log.jsonl"
done = {json.loads(l)["file"] for l in open(logf) if l.strip() and json.loads(l)["match"]} if logf.exists() else set()
todo = [s for s in sorted(sub["shards"], key=lambda s: s["size"]) if s["file"] not in done and (not names or s["file"] in names or Path(s["file"]).name in names)]
CH = 32 * 1024 * 1024
part_dir = base / "shards" / "_ranged"
part_dir.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
state = {}


def url(s):
    return f"https://huggingface.co/datasets/{sub['repo']}/resolve/{sub['revision']}/{s['file']}"


def fetch(s, k):
    part = part_dir / (Path(s["file"]).name + ".part")
    man = part_dir / (Path(s["file"]).name + ".chunks")
    lo, hi = k * CH, min((k + 1) * CH, s["size"]) - 1
    for attempt in range(10):
        try:
            r = requests.get(url(s), headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=60, allow_redirects=True)
            if r.status_code != 206:
                raise RuntimeError(f"status {r.status_code}")
            buf = bytearray()
            for b in r.iter_content(1 << 20):
                buf += b
            if len(buf) != hi - lo + 1:
                raise RuntimeError(f"short chunk {len(buf)} of {hi - lo + 1}")
            with lock:
                with open(part, "r+b") as f:
                    f.seek(lo)
                    f.write(buf)
                with open(man, "a") as f:
                    f.write(f"{k}\n")
                st = state[s["file"]]
                st["left"] -= 1
                if st["left"] == 0:
                    st["event"].set()
            return
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(min(60, 3 * (attempt + 1)))
    raise RuntimeError(f"chunk {k} of {s['file']} failed: {err}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def finish(s):
    part = part_dir / (Path(s["file"]).name + ".part")
    h = sha(part)
    rec = {"file": s["file"], "bytes": os.path.getsize(part), "sha256": h, "sha256_expected": s["sha256"], "match": h == s["sha256"],
           "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "revision": sub["revision"], "method": "download_ranged.py"}
    if rec["match"]:
        dst = base / "shards" / s["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, dst)
    else:
        os.remove(part)
    with lock:
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(rec["file"], rec["bytes"], rec["match"], time.strftime("%H:%M:%S"), flush=True)


t0 = time.time()
jobs = []
for s in todo:
    n = -(-s["size"] // CH)
    part = part_dir / (Path(s["file"]).name + ".part")
    man = part_dir / (Path(s["file"]).name + ".chunks")
    got = {int(x) for x in open(man).read().split()} if man.exists() and part.exists() else set()
    if not part.exists() or not got:
        with open(part, "wb") as f:
            f.truncate(s["size"])
        open(man, "w").close()
        got = set()
    state[s["file"]] = {"left": n - len(got), "event": threading.Event()}
    if state[s["file"]]["left"] == 0:
        state[s["file"]]["event"].set()
    jobs += [(s, k) for k in range(n) if k not in got]
print("shards", [s["file"] for s in todo], "chunks", len(jobs), flush=True)
with ThreadPoolExecutor(conns) as ex:
    futs = [ex.submit(fetch, s, k) for s, k in jobs]
    fin = ThreadPoolExecutor(2)
    pending = list(todo)
    while pending:
        for s in list(pending):
            if state[s["file"]]["event"].is_set():
                pending.remove(s)
                fin.submit(finish, s)
        for f in futs:
            if f.done() and f.exception():
                raise f.exception()
        time.sleep(2)
    fin.shutdown(wait=True)
print("done", round(time.time() - t0), "s")
