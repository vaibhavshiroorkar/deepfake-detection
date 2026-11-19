"""CLIP ViT-L/14 embeddings of the website's face crops for every calibration clip, for a generated-video probe.

    .venv/Scripts/python.exe scripts/embed_faces_clip.py [--device cuda]

Reads the 16 face crops per clip that the study cached (runs/p001/cache*), the same inputs the website's
first part reproduces, prepares them as the CLIP face model does (ImageNet-normalized crops back to RGB,
224 x 224 bicubic, CLIP-normalized), and averages the image embeddings of a clip's crops.
Writes runs/gen-probe/embeddings-<corpus>.npz: clip_id, label (clip label), embedding (768).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "gen-probe"
CLIP_DIR = ROOT / "models" / "pretrained" / "clip-vit-large-patch14"


class ClipEmbedder:
    def __init__(self, device: str) -> None:
        from transformers import CLIPVisionModelWithProjection

        from deepfake_detection.webapp.candidates import (
            CLIP_MEAN,
            CLIP_STD,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_DIR, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.stats = [
            s.to(device) for s in (IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD)
        ]

    @torch.inference_mode()
    def __call__(self, crops: torch.Tensor) -> np.ndarray:
        """Mean L2-normalized image embedding of (T, 3, H, W) ImageNet-normalized crops."""
        im_mean, im_std, clip_mean, clip_std = self.stats
        rgb = (crops.to(self.device).float() * im_std + im_mean).clamp(0, 1)
        rgb = F.interpolate(
            rgb, size=(224, 224), mode="bicubic", align_corners=False
        ).clamp(0, 1)
        x = ((rgb - clip_mean) / clip_std).half()
        emb = self.model(pixel_values=x).image_embeds.float()
        emb = F.normalize(emb, dim=1).mean(0)
        return emb.cpu().numpy()


def main() -> int:
    from build_web_calibration import CORPORA, rows_for
    from score_v2_cache import INDEX, _clip_ids

    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    embed = ClipEmbedder(args.device)
    for corpus, (branch_set, sync_stem) in CORPORA.items():
        target = OUT / f"embeddings-{corpus}.npz"
        if target.exists():
            print(f"{corpus}: done already", flush=True)
            continue
        index_file, cache = INDEX[corpus]
        index = read_cache_index(ROOT / "runs" / "p001" / index_file)
        store = CacheStore(ROOT / "runs" / "p001" / cache)
        rows = rows_for(corpus, branch_set, sync_stem, {})
        ids, labels, embs = [], [], []
        began = time.time()
        for clip_id, row in zip(_clip_ids(sync_stem), rows, strict=True):
            if clip_id not in index or row["scores"]["face"] is None:
                continue
            view = store.load(index[clip_id])
            embs.append(embed(torch.from_numpy(view.visual_view)))
            ids.append(clip_id)
            labels.append(row["label"])
            if len(ids) % 500 == 0:
                print(
                    f"{corpus} {len(ids)} clips, {(time.time() - began) / len(ids):.3f} s/clip",
                    flush=True,
                )
        np.savez(
            target,
            clip_id=np.array(ids),
            label=np.array(labels),
            embedding=np.stack(embs),
        )
        print(f"{corpus}: {len(ids)} clips -> {target.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
