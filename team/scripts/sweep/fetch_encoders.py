"""Download the approved O-007 encoder weights at pinned revisions and verify their SHA-256 (CLAUDE.md rule 7).

    HF_HUB_CACHE=D:\\cache\\hf\\hub PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/fetch_encoders.py --group image

Expected values are the Hugging Face (LFS) hashes logged in docs/DATA_LOG.md before the download. The result of
every file (expected, local, match) is written to data/_logs/encoder_weights.json. IMG-04 (original Xception) is a
GitHub release asset with no published hash: its local SHA-256 is recorded as the first observed value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

CACHE = Path(r"D:\cache")
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hf" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))

# id, repo, weight file, bytes, expected sha256, revision (from docs/O007_SWEEP.md section 12)
HF = {
    "IMG-01": ("timm/efficientnet_b4.ra2_in1k", "model.safetensors", 77933206, "8030f9c929ed71a06728db4b61323960e573b463e68f310bd1bfb31ed62cbf91", "442c00e15609ff467772913d230768706a406aca"),
    "IMG-07": ("facebook/dinov2-small", "model.safetensors", 88249960, "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1", "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
    "IMG-08": ("facebook/dinov2-base", "model.safetensors", 346345912, "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "IMG-13": ("openai/clip-vit-large-patch14", "model.safetensors", 1710540580, "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    "IMG-15": ("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "model.safetensors", 1710540580, "0fce40ac684817a17ab8aa6014f0a933f602effa13a41c2618b6716667974e09", "1627032197142fbe2a7cfec626f4ced3ae60d07a"),
    "IMG-17": ("google/siglip-base-patch16-224", "model.safetensors", 812672320, "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
    "AU-1": ("facebook/wav2vec2-base", "pytorch_model.bin", 380267417, "3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21", "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"),
    "AU-4": ("microsoft/wavlm-base-plus", "pytorch_model.bin", 377617425, "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51", "4c66d4806a428f2e922ccfa1a962776e232d487b"),
    "AU-6": ("openai/whisper-small", "model.safetensors", 966995080, "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b", "973afd24965f72e36ca33b3055d56a652f456b4d"),
    "AU-8": ("facebook/hubert-base-ls960", "pytorch_model.bin", 377569754, "062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad", "dba3bb02fda4248b6e082697eee756de8fe8aa8a"),
}
XCEPTION_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth"
XCEPTION_BYTES = 91675053
XCEPTION_PATH = CACHE / "weights" / "xception-43020ad28.pth"
LOG = Path("data/_logs/encoder_weights.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(cid: str) -> dict:
    from huggingface_hub import snapshot_download
    repo, fname, size, expected, rev = HF[cid]
    d = Path(snapshot_download(repo, revision=rev, allow_patterns=[fname, "*.json", "*.txt"]))
    f = d / fname
    local = sha256(f)
    return dict(id=cid, repo=repo, revision=rev, file=fname, dir=str(d), bytes=f.stat().st_size, expected_bytes=size,
                sha256_expected=expected, sha256_local=local, match=bool(local == expected and f.stat().st_size == size))


def fetch_xception() -> dict:
    XCEPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not XCEPTION_PATH.is_file() or XCEPTION_PATH.stat().st_size != XCEPTION_BYTES:
        import subprocess  # curl resumes a partial file; urlretrieve did not survive a connection abort
        subprocess.run(["curl", "-L", "--retry", "20", "--retry-delay", "5", "-C", "-", "-o", str(XCEPTION_PATH), XCEPTION_URL], check=True)
    local = sha256(XCEPTION_PATH)
    return dict(id="IMG-04", repo=XCEPTION_URL, revision="release v0.1-cadene", file=XCEPTION_PATH.name, dir=str(XCEPTION_PATH.parent),
                bytes=XCEPTION_PATH.stat().st_size, expected_bytes=XCEPTION_BYTES, sha256_expected="NR (no published hash)",
                sha256_local=local, match=bool(XCEPTION_PATH.stat().st_size == XCEPTION_BYTES))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["image", "audio"])
    ap.add_argument("--only", help="comma separated candidate IDs")
    a = ap.parse_args(argv)
    ids = a.only.split(",") if a.only else [i for i in list(HF) + ["IMG-04"] if a.group and i.startswith("IMG" if a.group == "image" else "AU")]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in json.loads(LOG.read_text())} if LOG.is_file() else {}
    bad = 0
    for cid in ids:
        r = fetch_xception() if cid == "IMG-04" else fetch_hf(cid)
        done[cid] = r
        LOG.write_text(json.dumps(list(done.values()), indent=1))
        print(cid, "match" if r["match"] else "MISMATCH", r["sha256_local"][:16], flush=True)
        bad += not r["match"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
