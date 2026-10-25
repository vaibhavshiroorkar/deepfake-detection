"""speech: the frozen audio ensemble (T-A: HuBERT base, WavLM base+, Whisper small, mean of logits) on a speech file or a video's audio track (WS2)."""
from __future__ import annotations

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from .base import SCORE_MEANING_EXPLORATORY, AdapterModule, finding_for, threshold_of, warnings_for, weights_hash

EVALUATION = ("E-007, E-008, E-009 (exploratory development; the calibration is exploratory); E-014 (sealed test, tag I-4: AUC 0.9835 "
              "[0.9798, 0.9872] on ASVspoof 2019 LA test_in, 0.9148 [0.8896, 0.9383] on In-the-Wild test_x; tag I-5 on the audio track of "
              "LAV-DF videos, test_in, 2,000-video exploratory subset: 0.6462 [0.6188, 0.6723])")
HEADS = ("head_T-A_AU-8.npz", "head_T-A_AU-4.npz", "head_T-A_AU-6.npz")


class SpeechModule(AdapterModule):
    module_id = "speech"

    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type == "audio":
            return Applicability.yes()
        if probe.media_type == "video":
            if probe.has_audio is False:
                return Applicability.no("the video has no audio stream")
            return Applicability.yes()
        return Applicability.no("a still image has no audio")

    def run(self, ctx: Context) -> ModuleResult:
        res = self.provider.analyse(ctx)
        meta = self.provider.meta()
        s = res.per_modality.get("audio")
        warns = warnings_for(res, "audio")
        if s is None or not s.available or s.score is None:
            return self._not_applicable(s.reason if s is not None and s.reason else "no audio score available for this file", warns)
        thr = threshold_of(meta, "T-A")
        limits = ["Only the first window of about 4 seconds (64,600 samples at 16 kHz, after trimming leading and trailing silence and "
                  "peak normalisation) is scored; the rest of the file is not analysed.",
                  "The audio score calibration is exploratory (AUDIO_CALIBRATION_EXPLORATORY): read the score as a ranking score, not as a "
                  "reliable probability.",
                  "Trained on ASVspoof 2019 LA speech; the sealed result on another source is lower (In-the-Wild test_x, E-014).",
                  "The audio track of a video was scored on LAV-DF at a sealed AUC of 0.6462 [0.6188, 0.6723] (E-014, tag I-5, 2,000-video "
                  "exploratory subset), so the score is weak evidence on videos of that kind."]
        return self._result(
            status="completed", finding=finding_for(s.score, thr),
            score={"value": s.score, "meaning": SCORE_MEANING_EXPLORATORY, "calibrated": False, "threshold": thr},
            units_analysed=s.n_units,
            tested_scope="Synthetic or converted speech in the first 4 seconds of the audio (file or audio track of a video), scored by the "
                         "mean of the logits of three frozen speech encoders with a linear head each.",
            limitations=limits, warnings=warns,
            model={"name": "logit mean of HuBERT base, WavLM base+ and Whisper small encoders with linear heads, task T-A",
                   "version": self.provider.checkpoint_id(), "weights_sha256": weights_hash(meta, *HEADS),
                   "licence": "HuBERT base: Apache-2.0 tag; WavLM base+: research-only (D-038); Whisper small: see docs/DATA_LOG.md",
                   "evaluation": EVALUATION})
