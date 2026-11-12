"""lip_sync: audio-video temporal agreement of one face track, measured with SyncNet v2 (WS4, cross_modal section).

The network and the offset search are ports of joonson/syncnet_python (code MIT; see ``dfd.platform.syncnet``). The weights file
``data/models/syncnet/syncnet_v2.model`` has no stated licence (docs/LIPSYNC_IMPLEMENTATION_NOTE.md). The module never says that
media is manipulated: the cross_modal vocabulary offers consistent, potential_mismatch and inconclusive only.

Thresholds. The reference code has NO decision threshold (it prints offset, min dist and confidence; run_syncnet.py, SyncNetInstance.py
lines 262-264). The three numbers below (MIN_CONFIDENCE, OFFSET_TOLERANCE_FRAMES, SCORE_DISPLAY_SCALE) and the audio-activity rule are
placeholders chosen for this prototype. They are uncalibrated, not taken from the reference, and have not been evaluated on this
project's data. They are constructor arguments so that a preregistered evaluation can replace them.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WEIGHTS = REPO_ROOT / "data" / "models" / "syncnet" / "syncnet_v2.model"
DEFAULT_YUNET_DIR = REPO_ROOT / "data" / "_prep_bench" / "models"

MODULE_VERSION = "syncnet-v2-port-3"
MODEL_NAME = "SyncNet v2 (Chung and Zisserman, Oxford VGG)"
LICENCE = "code MIT (joonson/syncnet_python); weights licence not stated by the source"
NOT_EVALUATED = "not evaluated"

# Project placeholders (see module docstring): uncalibrated, not from the reference, not evaluated on project data.
# MIN_CONFIDENCE was 1.0 (a pure placeholder) and is now 3.0, CHOSEN ON THE E-022 DEVELOPMENT SET (40 LAV-DF train clips, shifted-audio and
# synthetic non-speech cases): there, correctly in-range cases had confidence >= 4.46 and out-of-range (+-1000 ms) cases <= 2.66. The
# value sits in that gap by inspection, not by optimisation. NO held-out check was done; it is still uncalibrated.
MIN_CONFIDENCE = 3.0              # below this the offset curve has no clear minimum: inconclusive
OFFSET_TOLERANCE_FRAMES = 2       # symmetric tolerance (80 ms at 25 fps), used only when offset_band is None (port-2 behaviour)
# Port-3: asymmetric band of offsets reported as consistent, FITTED in E-025 stage 1 on 230 real clips (150 LAV-DF train, 80 FakeAVCeleb
# development; 2.5th to 97.5th percentile of the confident offsets) and validated on 230 disjoint real clips (false alarms 1.3% [0.4, 3.8]).
# It is a property of VoxCeleb2-derived clips, not a universal tolerance, and it makes the Wav2Lip -3 frame offset (E-024) unflaggable.
OFFSET_BAND_FRAMES = (-6, 2)
SCORE_DISPLAY_SCALE = 10.0        # display value = min(confidence, 10) / 10; 10.02 is the confidence printed in the reference README demo
MIN_SECONDS = 2.0                 # shorter clips: inconclusive (the offset search spans +-15 frames and needs windows beyond that)
MAX_SECONDS = 30.0                # analysed prefix (memory and time bound); the rest of the file is not analysed
MIN_FACE_FRACTION = 0.5           # a face must be found in at least this fraction of frames, else not_applicable
MAX_MULTI_FACE_FRACTION = 0.10    # more frames than this with two or more faces: speaker attribution not established
AUDIO_ACTIVE_RMS = 0.01           # 20 ms blocks above this RMS (full scale 1.0) count as audible
MIN_ACTIVE_FRACTION = 0.20        # fewer audible blocks than this: no usable audio, inconclusive
# Heuristic speech-likelihood gate (NOT a speech detector). Cheap, non-learned: 25 ms frames, band 100-4000 Hz, frames above
# AUDIO_ACTIVE_RMS only. Speech-like if the median spectral flatness (geometric / arithmetic mean of the power spectrum) of active
# frames lies in [SPEECH_FLATNESS_MIN, SPEECH_FLATNESS_MAX] (below: steady tones and chords; above: noise) and the coefficient of
# variation of the active-frame RMS is at least SPEECH_RMS_CV_MIN (steady sounds fail). Values CHOSEN ON THE E-022 DEVELOPMENT SET
# (40 LAV-DF train audio tracks against 50 synthetic white/pink/tone/chord tracks); no held-out check was done. Natural music, singing,
# and speech over music or noise were not tested and may pass or fail wrongly.
SPEECH_FLATNESS_MIN = 0.001
SPEECH_FLATNESS_MAX = 0.28
SPEECH_RMS_CV_MIN = 0.15
SPEECH_NOT_CHECKED = ("Speech is only screened by a cheap heuristic (energy, spectral flatness, loudness variation), not a speech "
                      "detector: some non-speech audio (natural music, singing, speech over noise) may pass it and then carries no "
                      "information about lip movement, and some speech may fail it.")

TESTED_SCOPE = ("Temporal agreement between the audio and the lip region of one tracked face, as the offset in frames (25 fps) within "
                "+-15 frames of zero that minimises SyncNet v2 embedding distance over the analysed prefix (at most 30 s). Smoke-tested "
                "on one clip (data/_results/platform/lip_sync_smoke.json) and checked on controlled cases of LAV-DF train clips (E-022, development set "
                "only, no held-out check), validated once on new clips with the earlier +-2 frame tolerance (E-024: failed three of eight preregistered lines), and "
                "re-fitted and validated in E-025 (offset band -6 to +2 frames fitted on 230 real clips, validated on 230 disjoint real clips; see limitations).")
LIMITATIONS = [
    "It measures audio-visual temporal agreement of one face track. It does not detect manipulation.",
    "A mismatch is not proof of manipulation: dubbed content, bad muxing, variable frame rate and off-sync recordings produce one.",
    "A synchronised fake passes: generators that are trained or tuned for lip synchronisation can yield a consistent result.",
    "The offset search covers +-15 frames (+-0.6 s); a larger shift is not measured: it usually gives a low confidence or an offset at the edge of the range, both reported as inconclusive, but not every case is caught.",
    "The decision rule (confidence minimum 3.0, offset tolerance, speech heuristic) is an uncalibrated rule of this project, not taken from the "
    "reference code. The confidence minimum and the speech heuristic values were chosen on the E-022 development set with no held-out check. "
    "The score is a rescaled confidence, not a probability.",
    "Faces come from YuNet, not from the S3FD detector the reference pipeline uses, and tracking is simpler than the reference's. "
    "The effect on the result has not been measured.",
    "Tested scope: independent validation E-024 (frozen configuration, new clips, 95% Wilson intervals): unmodified synchronised clips were flagged potential_mismatch "
    "in 15.0% [8.1, 26.1] of 60 LAV-DF train clips and 13.3% [6.9, 24.2] of 60 FakeAVCeleb development clips (preregistered limit: 10%), and "
    "25.0% [15.8, 37.2] of clips shifted by only +-40 ms. Roughly one real clip in seven shows a confident offset of 3 to 8 frames, so a "
    "potential_mismatch is weak evidence on its own. In-range shifts of 120 to 480 ms were recovered within 2 frames or flagged in 97.8% [94.4, 99.1]; "
    "shifts of 1000 ms were all reported inconclusive; synthetic non-speech audio was inconclusive in 100 of 100 cases (natural music is untested).",
    "Wav2Lip-based fake clips of FakeAVCeleb (development half) were flagged potential_mismatch in 85 to 93% of cases, almost all at an offset of exactly -3 "
    "frames. This is the opposite of the pre-stated expectation and is not a calibrated detector result: it may reflect a systematic lag in those "
    "clips or in this implementation, and was not investigated or used to change any threshold.",
    "Tested scope: E-025 (preregistered; band fitted on one pool of 230 real VoxCeleb2-derived clips, validated on a disjoint pool of 230): unmodified real clips "
    "were flagged potential_mismatch in 1.3% [0.4, 3.8] (3 of 230) under the band -6 to +2 frames, against 11.7% (27 of 230) under the earlier +-2 frame rule. "
    "The cost: the 160 Wav2Lip-based FakeAVCeleb fakes, which sit at an offset of -3 or -2 frames, are now flagged in 0 of 160 cases (E-024 flagged 85 to 93% with "
    "the old rule), and audio shifted by -120, +120, -240, +240, +-480 ms (sign as in E-022) is flagged in 14, 2, 26, 14 and 26 of 28 clips each (30 LAV-DF clips, those with a confident base offset). The band describes how far real speech clips in this data drift from SyncNet's zero; it is not a "
    "detector of lip-sync fakes and has not been tested on any other media. The LAV-DF half of the fitting and validation pools was split by file, not by speaker, so same-speaker overlap cannot be excluded: the 1.3% is not evidence of speaker-independent generalisation.",
    SPEECH_NOT_CHECKED,
]
SCORE_MEANING = ("SyncNet confidence (median over the 31 candidate shifts of the mean embedding distance, minus its minimum), divided by "
                 f"{SCORE_DISPLAY_SCALE:g} and clipped to [0, 1] for display. It measures how clearly one audio-visual offset stands out; "
                 "it is an audio-visual offset measure, not a probability of manipulation and not a measure of authenticity. The "
                 "scaling constant is a display choice of this project. The raw offset, minimum distance and confidence are listed as "
                 "observations.")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou_free_nearest(prev_c: tuple, faces: list):
    """The face whose centre is nearest to the previous track centre."""
    return min(faces, key=lambda f: (f.x + f.w / 2 - prev_c[0]) ** 2 + (f.y + f.h / 2 - prev_c[1]) ** 2)


def build_track(per_frame_faces: list) -> tuple[Optional[np.ndarray], dict]:
    """One face track from per-frame detections (lists of objects with x, y, w, h).

    Returns (boxes (T, 4) as x1, y1, x2, y2 with gaps linearly interpolated, info). ``boxes`` is None when no single track can be
    formed; ``info['reason']`` is then 'no_face' or 'multiple_faces'.
    """
    t = len(per_frame_faces)
    counts = np.array([len(f) for f in per_frame_faces])
    info = {"n_frames": t, "frames_with_face": int((counts > 0).sum()), "frames_with_several": int((counts > 1).sum())}
    if t == 0 or info["frames_with_face"] < MIN_FACE_FRACTION * t:
        info["reason"] = "no_face"
        return None, info
    if info["frames_with_several"] > MAX_MULTI_FACE_FRACTION * t:
        info["reason"] = "multiple_faces"
        return None, info
    boxes = np.full((t, 4), np.nan)
    prev = None
    for i, faces in enumerate(per_frame_faces):
        if not faces:
            continue
        f = max(faces, key=lambda z: z.w * z.h) if prev is None else iou_free_nearest(prev, faces)
        boxes[i] = (f.x, f.y, f.x + f.w, f.y + f.h)
        prev = (f.x + f.w / 2, f.y + f.h / 2)
    ok = ~np.isnan(boxes[:, 0])
    idx = np.arange(t)
    for k in range(4):
        boxes[:, k] = np.interp(idx, idx[ok], boxes[ok, k])
    info["reason"] = None
    return boxes, info


def speech_like_metrics(wav: np.ndarray, sr: int = 16000) -> dict:
    """Heuristic speech-likelihood features: median spectral flatness (100-4000 Hz) and RMS coefficient of variation of active 25 ms frames."""
    n = int(0.025 * sr)
    k = len(wav) // n
    out = {"active_frames": 0, "flatness_median": None, "rms_cv": None}
    if k == 0:
        return out
    x = wav[:k * n].reshape(k, n).astype(np.float64)
    rms = np.sqrt((x ** 2).mean(1))
    act = rms > AUDIO_ACTIVE_RMS
    out["active_frames"] = int(act.sum())
    if act.sum() < 3:
        return out
    sp = np.abs(np.fft.rfft(x[act] * np.hanning(n), axis=1)) ** 2 + 1e-12
    sp = sp[:, int(100 / sr * n):int(4000 / sr * n)]
    flat = np.exp(np.log(sp).mean(1)) / sp.mean(1)
    out["flatness_median"] = float(np.median(flat))
    out["rms_cv"] = float(rms[act].std() / rms[act].mean())
    return out


def looks_speech_like(m: dict) -> bool:
    """True when the heuristic features are in the speech-like region (see SPEECH_FLATNESS_*). A heuristic, not a speech detector."""
    if m["flatness_median"] is None:
        return False
    return SPEECH_FLATNESS_MIN <= m["flatness_median"] <= SPEECH_FLATNESS_MAX and m["rms_cv"] >= SPEECH_RMS_CV_MIN


def audible_fraction(wav: np.ndarray, sr: int = 16000) -> float:
    """Fraction of 20 ms blocks whose RMS exceeds AUDIO_ACTIVE_RMS."""
    n = int(0.02 * sr)
    k = len(wav) // n
    if k == 0:
        return 0.0
    rms = np.sqrt((wav[:k * n].reshape(k, n).astype(np.float64) ** 2).mean(1))
    return float((rms > AUDIO_ACTIVE_RMS).mean())


def offset_curve_evidence(r, band: tuple, min_confidence: float) -> dict:
    """The offset search the decision was read from (D5): SyncResult.mean_dist, which the scorer computes and the module used to discard.
    ``mean_dist[i]`` belongs to offset ``vshift - i`` (scoring.score_track); here offsets run in ascending order. Display only."""
    n = int(r.vshift)
    md = np.asarray(r.mean_dist, dtype=np.float64)
    offsets = list(range(-n, n + 1))
    return {"kind": "offset_curve", "offsets": offsets, "mean_dist": [round(float(md[n - o]), 4) for o in offsets],
            "best_offset": int(r.offset_frames), "band": [int(band[0]), int(band[1])], "min_confidence": float(min_confidence),
            "confidence": round(float(r.confidence), 4),
            "text": ("Mean SyncNet embedding distance at each candidate audio-visual offset (frames at 25 fps). The lowest point is the best "
                     "alignment; the shaded band is the range reported as consistent. A timing measure, not a manipulation detector.")}


class LipSyncModule:
    module_id = "lip_sync"
    section = "cross_modal"
    source = "real"
    version = MODULE_VERSION

    def __init__(self, weights_path: str | Path | None = None, device: Optional[str] = None, timeout_s: float = 180.0,
                 detector_factory: Optional[Callable[[], object]] = None, min_confidence: float = MIN_CONFIDENCE,
                 offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES, min_seconds: float = MIN_SECONDS,
                 max_seconds: float = MAX_SECONDS, offset_band: Optional[tuple] = OFFSET_BAND_FRAMES):
        self.weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        self.device = device
        self.timeout_s = timeout_s
        self._detector_factory = detector_factory
        self.min_confidence, self.offset_tolerance = min_confidence, offset_tolerance_frames
        self.offset_band = tuple(offset_band) if offset_band is not None else (-offset_tolerance_frames, offset_tolerance_frames)
        self.min_seconds, self.max_seconds = min_seconds, max_seconds
        self._model = None
        self._detector = None
        self._sha: Optional[str] = None
        self._lock = threading.RLock()

    # ---- lazy resources -------------------------------------------------------------------------------------------------------
    def _dev(self) -> str:
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def weights_sha256(self) -> str:
        with self._lock:
            if self._sha is None:
                src = self.weights_path.with_name("SOURCE.json")
                rec = None
                if src.is_file():
                    try:
                        rec = json.loads(src.read_text(encoding="utf-8"))
                    except ValueError:
                        rec = None
                self._sha = _sha256_file(self.weights_path)      # always computed from the file; SOURCE.json is not trusted blindly
                if rec and rec.get("sha256") not in (None, self._sha):
                    raise RuntimeError("syncnet_v2.model does not match the hash recorded in SOURCE.json")
            return self._sha

    def _load_model(self):
        with self._lock:
            if self._model is None:
                if not self.weights_path.is_file():
                    raise FileNotFoundError(f"SyncNet weights not found: {self.weights_path}")
                from ..syncnet.model import S, load_weights
                self._model = load_weights(S(), self.weights_path).to(self._dev()).eval()
            return self._model

    def _get_detector(self):
        with self._lock:
            if self._detector is None:
                if self._detector_factory is not None:
                    self._detector = self._detector_factory()
                else:
                    from dfd.prep.face import make_detector
                    self._detector = make_detector("yunet", DEFAULT_YUNET_DIR)
            return self._detector

    def _model_block(self) -> dict:
        return {"name": MODEL_NAME, "version": MODULE_VERSION, "weights_sha256": self.weights_sha256(), "licence": LICENCE,
                "evaluation": NOT_EVALUATED}

    # ---- contract -------------------------------------------------------------------------------------------------------------
    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type != "video":
            return Applicability.no("lip synchronisation needs a video with sound; this input is " + str(probe.media_type))
        if probe.has_audio is False:
            return Applicability.no("the video has no audio stream")
        if probe.has_video is False:
            return Applicability.no("the file has no video stream")
        if not self._weights_available():
            return Applicability.unsupported(f"the SyncNet weights file is not installed ({self.weights_path.name} not found; see docs/SETUP.md)")
        return Applicability.yes()

    def _weights_available(self) -> bool:
        """A missing weights file is an unsupported configuration, not an inference failure (tests replace this with a constant)."""
        return self.weights_path.is_file()

    def _result(self, **kw) -> ModuleResult:
        return ModuleResult(module_id=self.module_id, version=self.version, section=self.section, source=self.source, **kw)

    def _not_applicable(self, reason: str, warnings: Optional[list] = None) -> ModuleResult:
        return self._result(status="not_applicable", reason=reason, warnings=warnings or [])

    def _inconclusive(self, code: str, message: str, observations: Optional[list] = None, score: Optional[dict] = None,
                      units: Optional[int] = None) -> ModuleResult:
        return self._result(status="completed", finding="inconclusive", score=score, units_analysed=units, model=self._model_block(),
                            tested_scope=TESTED_SCOPE, limitations=list(LIMITATIONS),
                            warnings=[{"code": code, "message": message}], observations=observations or [])

    # ---- analysis steps (separate so that tests can replace them) ---------------------------------------------------------------
    def _load_audio(self, path: str) -> np.ndarray:
        from dfd.prep.audio import extract_audio
        wav = extract_audio(path, 16000, check_stream=True)
        return wav[: int(self.max_seconds * 16000)]

    def _detect_faces(self, path: str) -> list:
        from ..syncnet.preprocess import iter_frames_25fps
        det = self._get_detector()
        return [det.detect(fr) for fr in iter_frames_25fps(path, self.max_seconds)]

    def _make_crops(self, path: str, boxes: np.ndarray) -> np.ndarray:
        from ..syncnet.preprocess import crop_face, iter_frames_25fps, smooth_boxes
        sm = smooth_boxes(boxes)
        crops = []
        for i, fr in enumerate(iter_frames_25fps(path, self.max_seconds)):
            if i >= len(sm):
                break
            crops.append(crop_face(fr, *sm[i]))
        return np.stack(crops)

    def _score(self, crops: np.ndarray, wav: np.ndarray):
        from ..syncnet.preprocess import audio_mfcc_tensor
        from ..syncnet.scoring import score_track
        return score_track(self._load_model(), crops, audio_mfcc_tensor(wav), len(wav), self._dev())

    # ---- run ------------------------------------------------------------------------------------------------------------------
    def run(self, ctx: Context) -> ModuleResult:
        t0 = time.perf_counter()
        path = ctx.probe.path
        wav = self._load_audio(path)
        if wav.size == 0:
            return self._not_applicable("the video has no decodable audio stream")
        secs = len(wav) / 16000.0
        if secs < self.min_seconds:
            return self._inconclusive("CLIP_TOO_SHORT", f"The audio is {secs:.2f} s long; at least {self.min_seconds:g} s is needed.")
        frac = audible_fraction(wav)
        if frac < MIN_ACTIVE_FRACTION:
            return self._inconclusive("NO_AUDIBLE_AUDIO", f"Only {frac:.0%} of the audio is above the silence level "
                                      f"(RMS {AUDIO_ACTIVE_RMS:g}); there is nothing to compare with the lips.")
        sm = speech_like_metrics(wav)
        if not looks_speech_like(sm):
            return self._inconclusive("AUDIO_NOT_SPEECH_LIKE", "A cheap heuristic (spectral flatness and loudness variation; not a speech "
                                      f"detector) finds the audio unlike speech (flatness {sm['flatness_median']}, loudness variation "
                                      f"{sm['rms_cv']}). Lip movement cannot be compared with noise, steady tones or music, so no offset is reported.")
        # Display only (D5, 2026-10-27): records that the screen above passed, so the report can say so. Changes no decision.
        screen = [{"kind": "heuristic", "code": "speech_like_screen", "value": True,
                   "text": "The audio passed the cheap speech-likeness screen (spectral flatness and loudness variation; not a speech detector)"}]
        faces = self._detect_faces(path)
        boxes, info = build_track(faces)
        if boxes is None and info["reason"] == "no_face":
            res = self._not_applicable(f"no face found in at least {MIN_FACE_FRACTION:.0%} of the frames "
                                       f"({info['frames_with_face']} of {info['n_frames']} frames had one)")
            res.observations = screen
            return res
        if boxes is None:
            return self._inconclusive("SPEAKER_ATTRIBUTION_NOT_ESTABLISHED",
                                      f"Two or more faces appear in {info['frames_with_several']} of {info['n_frames']} frames. This "
                                      "module scores one face track and cannot tell which person is speaking, so it does not pick one.",
                                      observations=list(screen))
        if info["n_frames"] < self.min_seconds * 25:
            return self._inconclusive("CLIP_TOO_SHORT", f"Only {info['n_frames']} video frames at 25 fps were decoded.", observations=list(screen))
        crops = self._make_crops(path, boxes)
        r = self._score(crops, wav)         # inference errors propagate: the coordinator records the module as failed
        conf, off = float(r.confidence), int(r.offset_frames)
        obs = [{"kind": "heuristic", "code": "av_offset_frames", "text": "AV offset in frames at 25 fps (reference definition: vshift minus argmin)", "value": off},
               {"kind": "heuristic", "code": "sync_min_dist", "text": "Minimum of the mean embedding distance over the shifts", "value": round(float(r.min_dist), 4)},
               {"kind": "heuristic", "code": "sync_confidence", "text": "Median minus minimum of the mean embedding distance (reference confidence)", "value": round(conf, 4)},
               {"kind": "heuristic", "code": "frames_with_face", "text": "Frames with a detected face / frames analysed", "value": f"{info['frames_with_face']}/{info['n_frames']}"}] + screen
        score = {"value": round(float(min(max(conf, 0.0), SCORE_DISPLAY_SCALE) / SCORE_DISPLAY_SCALE), 4), "meaning": SCORE_MEANING,
                 "calibrated": False, "threshold": None}
        warns = []
        if conf < self.min_confidence:
            res = self._inconclusive("LOW_SYNC_CONFIDENCE", f"The confidence {conf:.2f} is below the placeholder minimum {self.min_confidence:g}: "
                                     "no audio-visual offset stands out clearly (silence, non-speech audio, an unusable face crop or a shift "
                                     "beyond the search range can all cause this).", obs, score, r.n_windows)
        elif abs(off) >= r.vshift:
            res = self._inconclusive("OFFSET_AT_SEARCH_BOUNDARY", f"The best offset ({off} frames) is at the edge of the search range "
                                     f"(+-{r.vshift} frames); the true shift may be larger than the range, so it is not reliable.", obs, score, r.n_windows)
        else:
            finding = "consistent" if self.offset_band[0] <= off <= self.offset_band[1] else "potential_mismatch"
            if finding == "potential_mismatch":
                warns.append({"code": "OFFSET_BEYOND_TOLERANCE", "message": f"The best audio-visual offset is {off} frames "
                              f"({off * 40} ms), outside the band of {self.offset_band[0]} to {self.offset_band[1]} frames fitted on real clips (E-025). This is not proof of manipulation."})
            res = self._result(status="completed", finding=finding, score=score, units_analysed=r.n_windows, model=self._model_block(),
                               tested_scope=TESTED_SCOPE, limitations=list(LIMITATIONS), warnings=warns, observations=obs)
        res.evidence = [offset_curve_evidence(r, self.offset_band, self.min_confidence)]
        res.timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return res


def build_lip_sync_module(weights_path: str | Path | None = None, **kw) -> LipSyncModule:
    return LipSyncModule(weights_path=weights_path, **kw)
