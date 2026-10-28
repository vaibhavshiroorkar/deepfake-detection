"""Index the metadata columns (no image bytes) of every parquet shard of Community Forensics-Eval or -Small by HTTP range reads.

Usage: python scripts/modern_image/index_shards.py eval|small
Writes data/cf_<x>/_meta/index.csv (one row per image: shard, row, metadata) and shards.json (per-shard rows and LFS SHA-256 from the Hub).
"""
import csv, json, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _env
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

which = sys.argv[1]
repo, rev, out = (_env.EVAL_REPO, _env.EVAL_REV, _env.CF_EVAL) if which == "eval" else (_env.SMALL_REPO, _env.SMALL_REV, _env.CF_SMALL)
cols = _env.META_COLS + ["prompt"]
info = HfApi().dataset_info(repo, revision=rev, files_metadata=True)
shards = {s.rfilename: {"size": s.size, "sha256": (s.lfs.sha256 if s.lfs else None)} for s in info.siblings if s.rfilename.endswith(".parquet")}
fs = HfFileSystem()


def one(name):
    for attempt in range(4):
        try:
            with fs.open(f"datasets/{repo}@{rev}/{name}", "rb", block_size=1 << 20) as f:
                pf = pq.ParquetFile(f)
                tb = pf.read(columns=cols)
            return name, tb.to_pylist()
        except Exception as e:  # network
            err = e
            time.sleep(3 * (attempt + 1))
    return name, err


t0 = time.time()
rows, failed = [], []
with ThreadPoolExecutor(12) as ex:
    for i, (name, res) in enumerate(ex.map(one, sorted(shards))):
        if isinstance(res, Exception):
            failed.append((name, repr(res)))
            continue
        shards[name]["rows"] = len(res)
        for k, r in enumerate(res):
            r = dict(r)
            r["resolution"] = "x".join(map(str, r["resolution"] or []))
            r["prompt"] = (r.get("prompt") or "")[:300].replace("\n", " ")
            rows.append({"shard": name, "row": k, **r})
        if i % 20 == 0:
            print(i, name, len(rows), round(time.time() - t0), flush=True)
(out / "_meta").mkdir(parents=True, exist_ok=True)
with open(out / "_meta" / "index.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["shard", "row"] + cols)
    w.writeheader()
    w.writerows(rows)
json.dump({"repo": repo, "revision": rev, "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "failed": failed, "shards": shards},
          open(out / "_meta" / "shards.json", "w"), indent=1)
print("done", len(rows), "rows", len(failed), "failed", round(time.time() - t0), "s")
