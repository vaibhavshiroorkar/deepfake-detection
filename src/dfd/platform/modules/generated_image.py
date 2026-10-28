"""generated_image: the frozen generated-image model (T-I) on the WHOLE image when a face was found (UX-7, D8; D-083).

The frozen still-image path (image_detector) sends an image with a face to the face-swap model and never to the generated-image model, so a
fully generated portrait is examined only for face swaps. This module closes that gap additively: when, and only when, the adapter took the
face-crop path, it scores the whole image with the same frozen T-I encoder, head and temperature through the adapter's own whole-image
function (the one the no-face path uses), so the number is the score the frozen model gives the image. image_detector, its finding, its
threshold and the routing are unchanged.

EXPERIMENTAL. T-I was trained on ProGAN images. E-028 then measured it on 3,119 face-containing images of a public benchmark (AUC 0.79, most
generated faces missed at its threshold), so the module makes no finding: the finding is always ``inconclusive``, the warning code starts with
EXPERIMENTAL (grouped as experimental, the score is never a key figure), and no threshold is reported as a decision line. Making it decide would
need a preregistered validation that it passes and a recorded decision.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from .base import AdapterProvider, shared_provider, weights_hash
from .image_detector import T_I_EVALUATION

EXPERIMENTAL_WARNING = ("EXPERIMENTAL_PORTRAIT_SCOPE",
                        "Experimental: when the generated-image model was tested on images that contain a face it missed most generated ones, so "
                        "no finding is made from it. The number is a score for inspection only.")
ALREADY_EXAMINED = ("no face was found, so the whole image was already examined by the generated-image model in the image analysis; this "
                    "extra check runs only when a face sends the image to the face-swap model")
SCORE_MEANING = ("Score in [0, 1] from the frozen generated-image model: higher means the whole image looks more like the generated training "
                 "images than like the photographs. Its temperature was fitted on ProGAN validation images without faces, so for an image "
                 "with a face it is not a calibrated probability. It is the output of one experimental check, not a verdict on the image.")
# Evaluation evidence on faces, text only: no score, finding or gate depends on it. Numbers: E-028 RESULT (face-routed images, 3,119).
PORTRAIT_EVIDENCE = ("Evaluated by this project (E-028) on images that contain a face: 3,119 images from a public benchmark of 21 recent "
                     "generators. This model reached AUC 0.79 [0.77, 0.80] and 81 percent of the generated images scored below its threshold, "
                     "so most generated faces were missed; the face-swap model of the image analysis reached AUC 0.60 [0.58, 0.62] on the "
                     "same images. The model was trained on ProGAN images, not on portraits.")


def _decode(path: str):
    import cv2
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("the image could not be decoded")
    return img


class GeneratedImageModule:
    module_id = "generated_image"
    section = "authenticity"
    source = "real"
    requires_gpu = True

    def __init__(self, provider: Optional[AdapterProvider] = None, timeout_s: float = 120.0,
                 score_whole_image: Optional[Callable] = None):
        """``score_whole_image(adapter, bgr) -> float`` is injectable for tests; the default is the frozen adapter's own function."""
        self.provider = provider or shared_provider()
        self.timeout_s = timeout_s
        self._score = score_whole_image or (lambda adapter, bgr: adapter._score_whole_image("T-I", bgr))

    @property
    def version(self) -> str:
        try:
            return self.provider.checkpoint_id()
        except Exception:  # noqa: BLE001 - only the label; run() raises the clear error
            return "checkpoint-unavailable"

    def _result(self, **kw) -> ModuleResult:
        return ModuleResult(module_id=self.module_id, version=self.version, section=self.section, source=self.source, **kw)

    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type != "image":
            return Applicability.no("not an image file")
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        res = self.provider.analyse(ctx)          # cached per file: the face search is the one image_detector used, nothing is rescored
        s = res.per_modality.get("image")
        if s is None or not s.available:
            return self._result(status="not_applicable", reason="the image analysis produced no result for this file")
        if s.path != "face_crop":
            return self._result(status="not_applicable", reason=ALREADY_EXAMINED)
        adapter = self.provider.adapter()
        meta = adapter.meta
        if "T-I" not in getattr(adapter, "heads", {"T-I": None}) or "T-I" not in meta.get("tasks", {}):
            return self._result(status="unsupported", reason="this checkpoint has no generated-image model")
        img = _decode(ctx.probe.path)
        with self.provider._lock:                 # the adapter is shared with the other wrappers; one GPU call at a time
            score = float(self._score(adapter, img))
        if not np.isfinite(score):
            raise ValueError("the generated-image model produced a non-finite score")
        thr = float(meta["tasks"]["T-I"]["threshold"]["value"])
        return self._result(
            status="completed", finding="inconclusive",
            score={"value": round(score, 6), "meaning": SCORE_MEANING, "calibrated": False, "threshold": None},
            units_analysed=1,
            tested_scope="path=whole_image_with_face: the whole image (not the face crop) through the frozen CLIP ViT-L/14 encoder and the "
                         "linear head of T-I, the same function the image analysis uses when no face is found.",
            warnings=[{"code": EXPERIMENTAL_WARNING[0], "message": EXPERIMENTAL_WARNING[1]}],
            observations=[{"code": "frozen_threshold_not_applied", "kind": "processing", "value": round(thr, 6),
                           "text": "The frozen T-I threshold, recorded for reference only; it is not applied to images with a face."}],
            limitations=["Trained on ProGAN images against LSUN photographs; not trained on portraits or on face generators.",
                         PORTRAIT_EVIDENCE,
                         "The temperature was fitted on the ProGAN validation split; the score is not calibrated for images with a face.",
                         "It does not look for face swaps; the image analysis does that on the face crop."],
            model={"name": "CLIP ViT-L/14 (openai/clip-vit-large-patch14), frozen, with a linear head, task T-I, applied to the whole image of a "
                           "still image that contains a face",
                   "version": self.provider.checkpoint_id(), "weights_sha256": weights_hash(meta, "head_T-I.npz"),
                   "licence": "No licence stated in the encoder repo; accepted as research-only (D-038); see docs/DATA_LOG.md",
                   "evaluation": T_I_EVALUATION + ". " + PORTRAIT_EVIDENCE})
