"""The real adapter: frozen encoder, linear head, temperature scaling, threshold, from a checkpoint folder (D-028, D-040).

It refuses to exist without a checkpoint: construction raises ``NoCheckpointError`` when the folder has no metadata.json,
``CheckpointError`` when the metadata or a head file breaks the contract, ``PreprocessMismatchError`` when the checkpoint's
preprocess_id differs from the one in use. It never falls back to the mock and never returns a made-up number.

Paths (the same preprocessing as training, from dfd.prep):
  video  16 uniformly spaced frames, YuNet, largest face enlarged 1.3, 224 px, JPEG quality 95 round trip; frames without a face are
         left out; head on each frame, mean of the frame probabilities (D-024), temperature scaling.
  image  face found: the face crop goes through the video (T-V) head; no face: the whole image goes through the T-I head (D-033);
         the path used is reported in the field `path`.
  audio  the first 64,600 samples at 16 kHz (repeat-padded, dfd.prep.audio), the speech encoder's own extractor, mean and standard
         deviation over time of the last hidden layer, the T-A head, temperature scaling. A video's audio track is scored the same way.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from dfd.infer.adapters import AdapterResult, Media, ModalityScore
from dfd.infer.checkpoint import (CheckpointError, NoCheckpointError, PreprocessMismatchError, config_hash_of,  # noqa: F401
                                  load_checkpoint)
from dfd.prep import audio as A
from dfd.prep import config as P
from dfd.prep import face as F
from dfd.prep import video as V

INFERENCE_SOURCE = "real"
EPS = 1e-6


def _calibrate(p: float, temperature: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    z = np.log(p / (1 - p)) / temperature
    return float(1.0 / (1.0 + np.exp(-z)))


class RealAdapter:
    inference_source = INFERENCE_SOURCE

    def __init__(self, checkpoint_dir: str | Path | None = None, device: Optional[str] = None,
                 encoder_factory: Optional[Callable] = None, detector_factory: Optional[Callable] = None,
                 audio_encoder_factory: Optional[Callable] = None):
        self.meta, self.heads = load_checkpoint(checkpoint_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._encoder_factory = encoder_factory
        self._audio_encoder_factory = audio_encoder_factory
        self._audio_enc: dict = {}
        self._detector_factory = detector_factory
        self._encoders: dict = {}
        self._detector = None
        if encoder_factory is None and audio_encoder_factory is None:
            self._check_weights()

    # ---- construction helpers
    def _check_weights(self) -> None:
        """The encoder weights on disk must be the verified ones the checkpoint was trained with."""
        from dfd.sweep import encoders
        for task, tm in self.meta["tasks"].items():
            rec = encoders._record(tm["encoder"]["cid"])  # raises if the weights are missing or not hash-verified
            if rec["revision"] != tm["encoder"]["revision"]:
                raise CheckpointError(f"{task}: encoder {tm['encoder']['cid']} revision {rec['revision']} differs from the checkpoint's {tm['encoder']['revision']}")

    def _device(self) -> str:
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _encoder(self, task: str):
        cid = self.meta["tasks"][task]["encoder"]["cid"]
        if cid not in self._encoders:
            if self._encoder_factory is not None:
                self._encoders[cid] = self._encoder_factory(cid)
            else:
                from dfd.sweep import encoders
                self._encoders[cid] = encoders.load(cid, self._device())
        return self._encoders[cid]

    def _score_audio(self, media: Media) -> ModalityScore:
        """Calibrated probability of fake from the first 4 seconds of audio (file or the audio track of a video)."""
        if "T-A" not in self.heads:
            return ModalityScore(False, None, None, "this checkpoint has no audio model")
        wav = A.extract_audio(media.path) if media.media_type == "video" else A.load_audio(media.path)
        if wav.size == 0:
            return ModalityScore(False, None, None, "no decodable audio")
        cid = self.meta["tasks"]["T-A"]["encoder"]["cid"]
        if cid not in self._audio_enc:
            if self._audio_encoder_factory is not None:
                self._audio_enc[cid] = self._audio_encoder_factory(cid)
            else:
                from dfd.sweep import encoders
                self._audio_enc[cid] = encoders.load_audio(cid, self._device())
        out = self._audio_enc[cid](A.window_first_repeat(wav)[None, :])
        feat = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        p = float(self.heads["T-A"].prob_rows(feat)[0])
        return ModalityScore(True, round(_calibrate(p, float(self.meta["tasks"]["T-A"]["calibration"]["temperature"])), 6), 1)

    def _det(self):
        if self._detector is None:
            cfg = self.meta["tasks"]["T-V"]["preprocess_config"] if "T-V" in self.meta["tasks"] else {"detector": "yunet"}
            self._detector = (self._detector_factory or (lambda: F.make_detector(cfg["detector"], F.DEFAULT_MODEL_DIR)))()
        return self._detector

    # ---- scoring
    def _score_rgb(self, task: str, rgb: list[np.ndarray]) -> float:
        """Calibrated probability of fake for a list of RGB uint8 images of the size the encoder expects (mean of the image probabilities)."""
        import torch
        enc = self._encoder(task)
        dev = self._device() if self._encoder_factory is None else None
        feats = []
        for k in range(0, len(rgb), 64):
            batch = torch.from_numpy(np.stack(rgb[k:k + 64]).transpose(0, 3, 1, 2).copy())
            if dev:
                batch = batch.to(dev)
            out = enc(batch)
            feats.append(out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out))
        p = float(self.heads[task].prob_rows(np.concatenate(feats)).mean())
        return _calibrate(p, float(self.meta["tasks"][task]["calibration"]["temperature"]))

    def _score_whole_image(self, task: str, bgr: np.ndarray) -> float:
        """Whole image by the encoder's own model-card rule, as in the extraction (E-002); a double without a processor gets a plain resize."""
        import torch
        from PIL import Image
        enc = self._encoder(task)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        proc = getattr(enc, "processor", None)
        if proc is None:
            size = int(self.meta["tasks"][task]["preprocess_config"]["fallback_size"])
            return self._score_rgb(task, [np.ascontiguousarray(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1])])
        px = proc(Image.fromarray(rgb))[None]
        if self._encoder_factory is None:
            px = px.to(self._device())
        out = enc.encode_pixels(px)
        p = float(self.heads[task].prob_rows(out.detach().cpu().numpy()).mean())
        return _calibrate(p, float(self.meta["tasks"][task]["calibration"]["temperature"]))

    @staticmethod
    def _jpeg(bgr: np.ndarray, quality: int) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])  # the crops used for training went through JPEG
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr

    def _face_crops(self, frames, cfg: dict):
        """(RGB crops of the largest face per frame, any frame with several faces)."""
        det, crops, multi = self._det(), [], False
        for frame in frames:
            faces = det.detect(frame)
            multi = multi or len(faces) > 1
            crop, _, _ = F.crop_largest_face(frame, faces, cfg["enlarge"], cfg["size"])
            if crop is not None:
                crops.append(np.ascontiguousarray(self._jpeg(crop, cfg["quality"])[:, :, ::-1]))
        return crops, multi

    def analyse(self, media: Media, progress: Optional[Callable[[str], None]] = None) -> AdapterResult:
        tick = progress or (lambda stage: None)
        per: dict[str, ModalityScore] = {}
        warnings: list[dict] = []
        used: list[str] = []
        stage = {"detect": 0.0, "score": 0.0, "calibrate": 0.0}
        threshold = 0.5
        tick("detect")
        t = time.perf_counter()
        if media.media_type == "video":
            if "T-V" not in self.heads:
                per["visual"] = ModalityScore(False, None, None, "this checkpoint has no visual model")
            else:
                cfg = self.meta["tasks"]["T-V"]["preprocess_config"]
                info = V.probe(media.path)
                idx = V.sample_indices(info.n_frames, cfg["n_frames"])
                crops, multi = self._face_crops((f for _, f in V.iter_frames(media.path, backend=cfg["backend"], indices=idx, info=info)), cfg)
                stage["detect"] = time.perf_counter() - t
                if multi:
                    warnings.append({"code": "MULTIPLE_FACES", "message": "some frames show several faces; the largest face is scored.", "modality": "visual"})
                if not crops:
                    warnings.append({"code": "NO_FACE_DETECTED", "message": "no face was found in the sampled frames; the visual score is not available.", "modality": "visual"})
                    per["visual"] = ModalityScore(False, None, 0, "no face detected")
                else:
                    tick("score")
                    t = time.perf_counter()
                    per["visual"] = ModalityScore(True, round(self._score_rgb("T-V", crops), 6), len(crops))
                    stage["score"] = time.perf_counter() - t
                    used.append("T-V")
            if media.has_audio:
                tick("score")
                t = time.perf_counter()
                per["audio"] = self._score_audio(media)
                stage["score"] += time.perf_counter() - t
                if per["audio"].available:
                    used.append("T-A")
            else:
                per["audio"] = ModalityScore(False, None, None, "video has no audio track")
        elif media.media_type == "image":
            img = cv2.imdecode(np.fromfile(str(media.path), dtype=np.uint8), cv2.IMREAD_COLOR)
            faces = self._det().detect(img) if "T-V" in self.heads else []
            stage["detect"] = time.perf_counter() - t
            if len(faces) > 1:
                warnings.append({"code": "MULTIPLE_FACES", "message": "the image shows several faces; the largest face is scored.", "modality": "image"})
            tick("score")
            t = time.perf_counter()
            if faces:
                cfg = self.meta["tasks"]["T-V"]["preprocess_config"]
                crop, _, _ = F.crop_largest_face(img, faces, cfg["enlarge"], cfg["size"])
                per["image"] = ModalityScore(True, round(self._score_rgb("T-V", [np.ascontiguousarray(self._jpeg(crop, cfg["quality"])[:, :, ::-1])]), 6), 1, None, "face_crop")
                used.append("T-V")
                threshold = float(self.meta["tasks"]["T-V"]["threshold"]["value"])
            elif "T-I" in self.heads:
                per["image"] = ModalityScore(True, round(self._score_whole_image("T-I", img), 6), 1, None, "whole_image")
                used.append("T-I")
                threshold = float(self.meta["tasks"]["T-I"]["threshold"]["value"])
            else:
                per["image"] = ModalityScore(False, None, None, "no face found and this checkpoint has no whole-image model")
            stage["score"] = time.perf_counter() - t
        else:
            tick("score")
            t = time.perf_counter()
            per["audio"] = self._score_audio(media)
            stage["score"] = time.perf_counter() - t
            if per["audio"].available:
                used.append("T-A")
                threshold = float(self.meta["tasks"]["T-A"]["threshold"]["value"])
        tick("calibrate")  # temperature scaling is applied inside the scoring call above; its time is part of "score"
        return AdapterResult(per, warnings, self._model_block(used), threshold, {k: v * 1000 for k, v in stage.items()})

    def _model_block(self, used: list[str]) -> dict:
        ts = used or list(self.meta["tasks"])
        tm = self.meta["tasks"]
        per_task = {t: {"encoder": tm[t]["encoder"]["cid"], "revision": tm[t]["encoder"]["revision"], "preprocess_id": tm[t]["preprocess_id"],
                        "calibration": tm[t]["calibration"], "threshold": tm[t]["threshold"]} for t in ts}
        one = len(ts) == 1
        return {"checkpoint_id": self.meta["checkpoint_id"], "config_hash": config_hash_of(self.meta),
                "preprocess_id": tm[ts[0]]["preprocess_id"] if one else {t: tm[t]["preprocess_id"] for t in ts},
                "git_commit": self.meta["git_commit"], "iteration": self.meta["iteration"], "protocol": self.meta["protocol"],
                "calibration": tm[ts[0]]["calibration"] if one else {t: tm[t]["calibration"] for t in ts},
                "threshold": tm[ts[0]]["threshold"] if one else {t: tm[t]["threshold"] for t in ts}, "tasks": per_task}
