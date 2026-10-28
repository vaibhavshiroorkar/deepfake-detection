"""Download the chosen Community Forensics shards at the pinned revision and verify each against the Hub's LFS SHA-256.

Usage: python scripts/modern_image/download.py eval|small [threads]
Run with HF_HUB_DISABLE_XET=1 (the Xet path stalled on 2026-10-27; plain HTTPS gave 3.8 MB/s). Small shards go smallest first.
Reads data/cf_<x>/_meta/subset.json; writes shards under data/cf_<x>/shards/ and one line per shard to
data/cf_<x>/_meta/download_log.jsonl (file, bytes, sha256 local, sha256 expected, match, time). A mismatching file is deleted.
"""
import hashlib, json, sys, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _env
from huggingface_hub import hf_hub_download

which = sys.argv[1]
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
base = _env.CF_EVAL if which == "eval" else _env.CF_SMALL
sub = json.load(open(base / "_meta" / "subset.json"))
out = base / "shards"
logf = base / "_meta" / "download_log.jsonl"
done = set()
if logf.exists():
    done = {json.loads(l)["file"] for l in open(logf) if l.strip() and json.loads(l)["match"]}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def one(s):
    if s["file"] in done:
        return None
    for attempt in range(5):
        try:
            p = hf_hub_download(sub["repo"], s["file"], repo_type="dataset", revision=sub["revision"], local_dir=str(out))
            break
        except Exception as e:
            err = repr(e)
            if "401" in err or "403" in err:
                raise
            time.sleep(10 * (attempt + 1))
    else:
        return {"file": s["file"], "match": False, "error": err}
    h = sha(p)
    rec = {"file": s["file"], "bytes": os.path.getsize(p), "sha256": h, "sha256_expected": s["sha256"], "match": h == s["sha256"],
           "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "revision": sub["revision"]}
    if not rec["match"]:
        os.remove(p)
    return rec


t0 = time.time()
n = 0
with ThreadPoolExecutor(threads) as ex:
    order = sorted(sub["shards"], key=lambda s: s["size"]) if which == "small" else sub["shards"]
    futs = [ex.submit(one, s) for s in order]
    for fu in as_completed(futs):
        rec = fu.result()
        if rec is None:
            continue
        n += 1
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(n, rec["file"], rec.get("bytes"), rec["match"], round(time.time() - t0), "s", flush=True)
print("done", n, round(time.time() - t0), "s")
