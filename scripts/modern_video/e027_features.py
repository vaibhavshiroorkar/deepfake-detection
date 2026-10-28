"""E-027 per-frame embeddings of the 14 canonical-v3 frames of every DEVELOPMENT clip (DF26 reals and the six open generators).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/modern_video/e027_features.py IMG-07 IMG-08 IMG-17 IMG-13

Writes data/_results/modern/e027/features_<enc>_dev.npz (emb (n, 14, D) float32, id, generator, y, group). The closed subset is not
read here; e027_closed.py calls `extract` for it after its guard. IMG-07 runs through the live module's DinoRunner (fp32, the same code
path as the product); the others through dfd.sweep.encoders (fp16 autocast, the model card's mean and std; frames are already 224 px).
"""
from __future__ import annotations

import sys
import time

import numpy as np

import e027_lib as L


class Enc:
    def __init__(self, cid: str):
        import torch
        self.cid = cid
        if cid == "IMG-07":
            from dfd.platform.genvideo.features import DinoRunner
            self.r = DinoRunner()
            self.fn = lambda fr: self.r.embed(fr)[1]
        else:
            from dfd.sweep import encoders
            self.e = encoders.load(cid, device="cuda" if torch.cuda.is_available() else "cpu")
            dev = next(self.e.model.parameters()).device

            def fn(fr):
                x = torch.from_numpy(np.ascontiguousarray(fr.transpose(0, 3, 1, 2))).to(dev)
                return self.e(x).cpu().numpy()
            self.fn = fn
            self.revision = self.e.revision

    def __call__(self, frames):
        return np.asarray(self.fn(frames), dtype=np.float32)

    def rev(self):
        return self.r.revision if self.cid == "IMG-07" else self.revision


def extract(rows, cid: str, root=L.CANON, reader=None, log_every=200):
    from dfd.platform.genvideo import canonical
    reader = reader or canonical.read_frames
    enc = Enc(cid)
    embs, keep, bad = [], [], []
    t0 = time.time()
    for k, r in enumerate(rows, 1):
        try:
            fr = reader(root / r["video_path"])
            e = enc(fr)
            if not np.isfinite(e).all():
                raise ValueError("non-finite embedding")
            embs.append(e)
            keep.append(r)
        except Exception as ex:  # noqa: BLE001 - listed, preregistered exclusion
            bad.append({"id": r["video_path"], "error": f"{type(ex).__name__}: {str(ex)[:160]}"})
        if k % log_every == 0:
            print(f"{cid} {k}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    import torch
    vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    return np.stack(embs), keep, bad, {"seconds": round(time.time() - t0, 1), "max_vram_mib": round(vram, 1), "revision": enc.rev()}


def save(path, embs, keep, bad, info, cid):
    np.savez_compressed(path, emb=embs, id=np.array([r["video_path"] for r in keep]), generator=np.array([r["generator"] for r in keep]),
                        y=np.array([r["y"] for r in keep]), group=np.array([r["group"] for r in keep]), encoder=np.array([cid]),
                        revision=np.array([str(info["revision"])]))
    L.jdump({"encoder": cid, "n": len(keep), "failures": bad, **info}, str(path).replace(".npz", "_info.json"))


def main(argv) -> int:
    rows = L.dev_rows()
    assert all(r["y"] == 0 or r["generator"] in L.OPEN_GENS for r in rows)
    L.OUT.mkdir(parents=True, exist_ok=True)
    for cid in argv:
        out = L.OUT / f"features_{cid}_dev.npz"
        if out.exists():
            print(f"{out.name} exists, skipped")
            continue
        embs, keep, bad, info = extract(rows, cid)
        save(out, embs, keep, bad, info, cid)
        print(f"{cid}: {len(keep)} clips, {len(bad)} failures, {info}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
