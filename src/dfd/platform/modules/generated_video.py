"""generated_video: whole-video detector for AI-generated video, independent of faces (authenticity section).

The clip is passed through the canonical re-encode (first 2.0 s, 7 fps, 224 px, no metadata, ``genvideo.canonical``), 8 frames go through frozen
DINOv2-small, and a small head trained by this project (E-020) scores the features. What the head is, what it was measured to do and whether it
counts as validated is read from the installed head file (``meta``), never hard-coded here. A head whose meta does not say ``validated: true``
is EXPERIMENTAL: the module still reports its score but the finding is always ``inconclusive``, with a warning. Without a head file the module
reports ``unsupported``; it never invents a result.

Validation gate (post-audit). The ``validated`` flag in the head file is a claim by the file itself and authorises nothing on its own. A definitive
finding needs ALL of: the flag is the boolean true; the SHA-256 of the head file is listed in ``APPROVED_HEADS`` (code-controlled, in this source
file, reviewed with the code; editing the head's metadata changes its hash and removes the approval); the head's own evaluation, tested-scope and
limitation text does not say it is unvalidated or failed to generalise; and the score is a finite number. Anything else fails closed to
``inconclusive`` with a warning naming the reason. ``APPROVED_HEADS`` is empty: no head has passed an independent preregistered validation, so
every installed head, including the E-020 C2 head, is experimental. To approve a future head, see docs/SETUP.md section 8.
"""
from __future__ import annotations

import hashlib
import io
import math
import tempfile
import threading
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from ..genvideo import canonical
from ..genvideo.features import DinoRunner, shared_runner
from ..genvideo.model import Head, load_head
from ..module import Applicability, Context, Probe
from ..schema import NOT_EVALUATED, ModuleResult

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HEAD = REPO_ROOT / "checkpoints" / "genvideo-E020" / "head.npz"
INDICATES, NO_INDICATION = "indicates_manipulation", "no_indication_within_tested_scope"
EXPERIMENTAL_WARNING = ("EXPERIMENTAL_UNVALIDATED", "Experimental capability: the installed head was not validated as a detector, so no manipulation "
                        "finding is made from it. The number below is a ranking score for inspection only.")
# SHA-256 of head files approved for definitive findings -> the id of the preregistered validation experiment that approved it.
# EMPTY ON PURPOSE. Adding an entry is a reviewed code change that must cite a validation experiment (docs/SETUP.md section 8).
APPROVED_HEADS: Mapping[str, str] = {}
UNVALIDATED_PHRASES = ("not validated", "unvalidated", "did not generalise", "did not generalize", "not generalise", "not generalize",
                       "chance level", "experimental")
FEATURE_DIM = {"restrav21": 21, "cls_mean": 384}
DEFAULT_SCORE_MEANING = ("Uncalibrated ranking score in [0, 1] from the installed head: higher means the clip looks more like the generated training "
                         "clips than like the real training clips. It is not a probability.")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class GeneratedVideoModule:
    module_id = "generated_video"
    section = "authenticity"
    source = "real"
    timeout_s = 180.0
    requires_gpu = True

    def __init__(self, head_path: str | Path | None = None, runner: Optional[DinoRunner] = None,
                 canonicalise: Callable = canonical.canonicalise, read_frames: Callable = canonical.read_frames,
                 approved_heads: Optional[Mapping[str, str]] = None):
        self.approved_heads = APPROVED_HEADS if approved_heads is None else approved_heads
        self.head_path = Path(head_path) if head_path else DEFAULT_HEAD
        self._runner = runner
        self._canon, self._read = canonicalise, read_frames
        self._head: Optional[Head] = None
        self._head_sha: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        try:
            return str(self._load_head().meta.get("head_id", "unknown"))
        except Exception:  # noqa: BLE001 - label only; applicability reports the reason
            return "head-unavailable"

    def _load_head(self) -> Head:
        with self._lock:
            if self._head is None:
                # One read: the hash the gate checks and the weights/metadata that are used come from the same bytes (no swap between them).
                data = self.head_path.read_bytes()
                self._head = load_head(io.BytesIO(data))
                self._head_sha = hashlib.sha256(data).hexdigest()
            return self._head

    def _validation_state(self, head: Head, score: float) -> tuple[bool, Optional[str]]:
        """(definitive findings allowed, reason code when not). Fails closed on every doubt."""
        meta = head.meta
        if meta.get("validated") is not True:
            return False, "EXPERIMENTAL_UNVALIDATED"
        if not math.isfinite(score) or not math.isfinite(head.threshold):
            return False, "VALIDATION_NONFINITE"
        sha = self._head_sha
        if sha not in self.approved_heads:
            return False, "VALIDATION_NOT_APPROVED"
        text = " ".join(str(meta.get(k, "")) for k in ("evaluation", "tested_scope")) + " " + " ".join(map(str, meta.get("limitations", [])))
        if any(ph in text.lower() for ph in UNVALIDATED_PHRASES) or not str(meta.get("evaluation", "")).strip():
            return False, "VALIDATION_CONTRADICTED"
        return True, None

    def applicability(self, probe: Probe) -> Applicability:
        if probe.media_type != "video" or probe.has_video is False:
            return Applicability.no("not a video file")
        if not self.head_path.is_file():
            return Applicability.unsupported(f"no trained generated-video detector is installed ({self.head_path.name} not found)")
        try:
            head = self._load_head()
        except Exception as e:  # noqa: BLE001 - unreadable head is an unsupported capability, not a crash
            return Applicability.unsupported(f"the installed generated-video head cannot be read ({type(e).__name__})")
        key, proto = str(head.meta.get("feature", "restrav21")), head.meta.get("protocol")
        if key not in FEATURE_DIM or head.mean.shape[0] != FEATURE_DIM[key]:
            return Applicability.unsupported(f"the installed head is incompatible with this module (feature {key!r}, input size {head.mean.shape[0]})")
        if proto != canonical.PROTOCOL_ID:
            return Applicability.unsupported(f"the installed head was trained under preprocessing protocol {proto!r}, this module applies {canonical.PROTOCOL_ID!r}")
        if probe.duration_s is not None and probe.duration_s < canonical.WINDOW_S:
            return Applicability.no(f"the clip is shorter than the {canonical.WINDOW_S:.1f} s window this detector analyses")
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        head = self._load_head()
        runner = self._runner or shared_runner()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "canonical.mp4"
            self._canon(ctx.probe.path, dst)
            frames = self._read(dst)
        feats = runner.features(frames)
        key = str(head.meta.get("feature", "restrav21"))
        if key not in FEATURE_DIM:
            raise ValueError(f"head feature {key!r} cannot be computed by this module")
        if not np.isfinite(np.asarray(feats[key], dtype=np.float64)).all():
            raise ValueError("the feature vector contains non-finite values")
        score = float(head.score(feats[key])[0])
        meta = head.meta
        if not math.isfinite(score):
            raise ValueError("the head produced a non-finite score")
        validated, why_not = self._validation_state(head, score)
        enc_rev = runner.revision or "unknown"
        return ModuleResult(
            module_id=self.module_id, version=str(meta.get("head_id", "unknown")), section=self.section, source=self.source, status="completed",
            finding=(INDICATES if score >= head.threshold else NO_INDICATION) if validated else "inconclusive",
            score={"value": round(score, 6), "meaning": str(meta.get("score_meaning", DEFAULT_SCORE_MEANING)), "calibrated": False,
                   "threshold": round(head.threshold, 6) if validated else None},
            units_analysed=int(frames.shape[0]),
            warnings=[] if validated else [{"code": EXPERIMENTAL_WARNING[0], "message": EXPERIMENTAL_WARNING[1]}] + (
                [{"code": why_not, "message": "The head file claims or implies validation but the gate refused it (" + why_not + "); no finding is made."}]
                if why_not != EXPERIMENTAL_WARNING[0] else []),
            model={"name": str(meta.get("model_name", "generated-video head (identity not recorded in the head file)")),
                   "version": str(meta.get("head_id", "unknown")),
                   "weights_sha256": f"head={self._head_sha};dinov2-small-revision={enc_rev}",
                   "licence": str(meta.get("licence", "DINOv2-small Apache-2.0 (model card, DATA_LOG IMG-07); head trained by this project")),
                   "evaluation": str(meta.get("evaluation", NOT_EVALUATED))},
            tested_scope=str(meta.get("tested_scope", "")) or "No evaluation recorded for this detector.",
            limitations=list(meta.get("limitations", [])) + [
                "Only the first 2.0 s of the clip is analysed, after a canonical re-encode (7 fps, 224 px); the rest of the file is not seen.",
                "Face-independent: it does not detect face swaps or lip-sync edits and is not a substitute for those modules.",
                "No indication does not mean the video is camera-captured."])
