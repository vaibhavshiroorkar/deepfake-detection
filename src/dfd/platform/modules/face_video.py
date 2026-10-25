"""face_video: the frozen visual head (T-V, fine-tuned SigLIP base) on the largest face of a video (WS2)."""
from __future__ import annotations

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from .base import (LARGEST_FACE_LIMITATION, SCORE_MEANING_V2, AdapterModule, finding_for, threshold_of, warnings_for, weights_hash)

EVALUATION = ("E-005 (fine-tuning; temperature and threshold fitted on the FaceForensics++ validation split); "
              "E-014 (sealed test, tag I-4: AUC 0.8218 [0.8020, 0.8393] on the Celeb-DF family, 0.8999 [0.8759, 0.9210] on Celeb-DF v2, "
              "0.9899 [0.9826, 0.9959] on the FaceForensics++ test_in split; tag I-5 on LAV-DF test_in: 0.3226 [0.2953, 0.3490])")


class FaceVideoModule(AdapterModule):
    module_id = "face_video"

    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type != "video":
            return Applicability.no("not a video file; still images are handled by the image detector")
        if probe.has_video is False:
            return Applicability.no("the file has no video stream")
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        res = self.provider.analyse(ctx)
        meta = self.provider.meta()
        s = res.per_modality.get("visual")
        warns = warnings_for(res, "visual")
        if s is None or not s.available or s.score is None:
            return self._not_applicable(s.reason if s is not None and s.reason else "no visual score available for this file", warns)
        thr = threshold_of(meta, "T-V")
        limits = ["Trained and sealed-tested on face-swap material (FaceForensics++, Celeb-DF family, DFD). Not evaluated on talking-head "
                  "generated video; on LAV-DF (content-driven edits) the sealed AUC is below 0.5 (E-014).",
                  "A manipulation of the audio only is not visible to this module.",
                  "The score is the mean over the sampled frames that contain a face; frames without a face are left out.",
                  "The temperature and the threshold were fitted on FaceForensics++ validation clips and are not tuned for other sources."]
        if any(w["code"] == "MULTIPLE_FACES" for w in warns):
            limits.insert(0, LARGEST_FACE_LIMITATION)
        return self._result(
            status="completed", finding=finding_for(s.score, thr),
            score={"value": s.score, "meaning": SCORE_MEANING_V2, "calibrated": True, "threshold": thr},
            units_analysed=s.n_units,
            tested_scope=(f"Face manipulation of the largest face in {s.n_units} sampled frames (at most 16, YuNet detector, 1.3 enlarged "
                          "crop, 224 px), scored by the fine-tuned SigLIP base video head."),
            limitations=limits, warnings=warns,
            model={"name": "SigLIP base (google/siglip-base-patch16-224), fully fine-tuned, task T-V", "version": self.provider.checkpoint_id(),
                   "weights_sha256": weights_hash(meta, "weights_T-V.pt"),
                   "licence": "Encoder: Apache-2.0 tag (docs/EXPERIMENTS.md encoder table); training data terms: see docs/DATA_LOG.md",
                   "evaluation": EVALUATION})
