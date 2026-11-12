"""image_detector: the frozen still-image path of the v1 adapter (WS2).

A face found: the face crop goes through the video-trained T-V head. No face: the whole image goes through the T-I head (CLIP ViT-L/14 probe).
"""
from __future__ import annotations

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from .base import (LARGEST_FACE_LIMITATION, SCORE_MEANING_V2, AdapterModule, finding_for, threshold_of, warnings_for, weights_hash)
from .face_video import EVALUATION as T_V_EVALUATION

T_I_EVALUATION = ("E-002 (encoder ranking, exploratory), E-004 (head, temperature and threshold fitted on the ProGAN validation split); "
                  "E-014 (sealed test, tag I-4: AUC 0.9316 [0.8933, 0.9694] on the ForenSynths CNN subset, 0.9354 [0.8967, 0.9665] on the "
                  "diffusion test set, 1.0000 [0.9999, 1.0000] on the ProGAN test_in split)")
FACE_CROP_EVALUATION = ("Scored by the T-V head, which was evaluated on video clips only (" + T_V_EVALUATION + "). "
                        "Not evaluated on still face-swapped photos.")


class ImageDetectorModule(AdapterModule):
    module_id = "image_detector"

    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type != "image":
            return Applicability.no("not an image file; video and audio files are handled by other modules")
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        res = self.provider.analyse(ctx)
        meta = self.provider.meta()
        s = res.per_modality.get("image")
        warns = warnings_for(res, "image")
        if s is None or not s.available or s.score is None:
            return self._not_applicable(s.reason if s is not None and s.reason else "no image score available for this file", warns)
        if s.path == "face_crop":
            thr = threshold_of(meta, "T-V")
            limits = ["A video-trained model scored a still image: the face-crop path uses the T-V head, which was trained and evaluated on video "
                      "frames. It was not evaluated on still face-swapped photos.",
                      "The temperature and threshold were fitted on FaceForensics++ validation clips, so the score is not claimed to be "
                      "calibrated for still images.",
                      "The whole-image probe for generated images is not used on this path, because a face was found."]
            if any(w["code"] == "MULTIPLE_FACES" for w in warns):
                limits.insert(0, LARGEST_FACE_LIMITATION)
            return self._result(
                status="completed", finding=finding_for(s.score, thr),
                score={"value": s.score, "meaning": SCORE_MEANING_V2, "calibrated": False, "threshold": thr},
                units_analysed=s.n_units, tested_scope="path=face_crop: the largest face of the image (YuNet, 1.3 enlarged crop, 224 px) scored by "
                "the video-trained SigLIP base head (T-V).",
                limitations=limits, warnings=warns,
                model={"name": "SigLIP base (google/siglip-base-patch16-224), fully fine-tuned, task T-V, applied to a face crop of a still image",
                       "version": self.provider.checkpoint_id(), "weights_sha256": weights_hash(meta, "weights_T-V.pt"),
                       "licence": "Encoder: Apache-2.0 tag (docs/EXPERIMENTS.md encoder table); training data terms: see docs/DATA_LOG.md",
                       "evaluation": FACE_CROP_EVALUATION})
        if s.path == "whole_image":
            thr = threshold_of(meta, "T-I")
            limits = ["Trained on ProGAN images against LSUN photographs; the sealed tests cover other GAN families and diffusion sets "
                      "(E-014), not every generator and not edited or partly generated photos.",
                      "No face was found, so the face-swap path was not used.",
                      "The temperature and threshold were fitted on the ProGAN validation split and are not tuned for other sources."]
            return self._result(
                status="completed", finding=finding_for(s.score, thr),
                score={"value": s.score, "meaning": SCORE_MEANING_V2, "calibrated": True, "threshold": thr},
                units_analysed=s.n_units, tested_scope="path=whole_image: the whole image through the CLIP ViT-L/14 encoder and a linear head (T-I), "
                "for fully generated images.",
                limitations=limits, warnings=warns,
                model={"name": "CLIP ViT-L/14 (openai/clip-vit-large-patch14), frozen, with a linear head, task T-I",
                       "version": self.provider.checkpoint_id(), "weights_sha256": weights_hash(meta, "head_T-I.npz"),
                       "licence": "No licence stated in the encoder repo; accepted as research-only (D-038); see docs/DATA_LOG.md",
                       "evaluation": T_I_EVALUATION})
        raise ValueError(f"unknown image path from the adapter: {s.path!r}")
