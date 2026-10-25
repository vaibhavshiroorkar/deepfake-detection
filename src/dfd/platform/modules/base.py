"""Shared helpers for the modules that wrap the frozen v1 detectors (WS2).

The frozen detectors are not changed. One ``RealAdapter`` is built lazily per process and checkpoint folder, ``adapter.analyse(Media)`` runs once per
file, and its ``AdapterResult`` is cached in ``ctx.cache`` under the file's sha256, so the three modules never rescore the same file. An adapter
can be injected (tests). A missing checkpoint raises ``NoCheckpointError`` from the adapter; there is no fallback to a mock. Adapter exceptions
propagate: the coordinator turns them into failed entries.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from dfd.infer.adapters import AdapterResult, Media
from dfd.infer.schema import SCORE_MEANING

from ..module import Context, Probe
from ..schema import ModuleResult

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "I-5-freeze-candidate"
RECORDED_ELSEWHERE = "see docs/DATA_LOG.md"
SCORE_MEANING_V2 = SCORE_MEANING + " It is the output of one detector module, not an overall verdict on the media."
SCORE_MEANING_EXPLORATORY = ("Probability of fake from the speech model of this project, with an exploratory calibration: read it as a ranking "
                             "score, not as a reliable probability. It is the output of one detector module, not an overall verdict on the media.")
INDICATES, NO_INDICATION = "indicates_manipulation", "no_indication_within_tested_scope"
LARGEST_FACE_LIMITATION = "only the largest face is scored"


class AdapterProvider:
    """Builds the adapter once (lazily, thread-safe) and exposes the checkpoint metadata."""

    def __init__(self, checkpoint_dir: str | Path | None = None, adapter: Any = None, factory: Optional[Callable[[Path], Any]] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT
        self._adapter = adapter
        self._factory = factory
        self._lock = threading.RLock()
        self.n_analyses = 0

    def adapter(self):
        with self._lock:
            if self._adapter is None:
                if self._factory is not None:
                    self._adapter = self._factory(self.checkpoint_dir)
                else:
                    from dfd.infer.real_adapter import RealAdapter     # heavy imports only when a module really runs
                    self._adapter = RealAdapter(self.checkpoint_dir)   # raises NoCheckpointError or CheckpointError, never a mock
            return self._adapter

    def meta(self) -> dict:
        return self.adapter().meta

    def checkpoint_id(self) -> str:
        """Cheap: reads metadata.json without building the adapter, unless an adapter was injected."""
        if self._adapter is not None:
            return str(self._adapter.meta["checkpoint_id"])
        return str(json.loads((self.checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))["checkpoint_id"])

    def analyse(self, ctx: Context) -> AdapterResult:
        key = "dfd_adapter_result:" + ctx.probe.sha256
        with self._lock:
            if key not in ctx.cache:
                ctx.cache[key] = self.adapter().analyse(media_from_probe(ctx.probe))
                self.n_analyses += 1
            return ctx.cache[key]


_SHARED: dict[str, AdapterProvider] = {}
_SHARED_LOCK = threading.Lock()


def shared_provider(checkpoint_dir: str | Path | None = None) -> AdapterProvider:
    """One provider (hence one adapter) per process and checkpoint folder."""
    key = str(Path(checkpoint_dir if checkpoint_dir is not None else DEFAULT_CHECKPOINT).resolve())
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = AdapterProvider(key)
        return _SHARED[key]


def media_from_probe(p: Probe) -> Media:
    """The v1 Media for the adapter. An unknown audio stream (None) is passed as present: the adapter then reports an undecodable track."""
    if p.media_type == "audio":
        has_audio = True
    else:
        has_audio = p.media_type == "video" and p.has_audio is not False
    return Media(Path(p.path), p.filename, p.media_type, p.mime_type, p.size_bytes, p.sha256, p.duration_s, has_audio, p.width, p.height)


def finding_for(score: float, threshold: float) -> str:
    return INDICATES if score >= threshold else NO_INDICATION


def threshold_of(meta: dict, task: str) -> float:
    return float(meta["tasks"][task]["threshold"]["value"])


def weights_hash(meta: dict, *files: str) -> str:
    """Hashes recorded in the checkpoint's metadata (``files``); nothing is computed or guessed."""
    rec = meta.get("files", {})
    got = [(f, rec.get(f)) for f in files]
    if any(h is None for _, h in got):
        return RECORDED_ELSEWHERE
    return got[0][1] if len(got) == 1 else ";".join(f"{f}={h}" for f, h in got)


def warnings_for(res: AdapterResult, modality: str) -> list:
    return [{"code": w["code"], "message": w["message"]} for w in res.warnings if w.get("modality") == modality]


class AdapterModule:
    """Common parts of the three wrappers."""
    source = "real"
    section = "authenticity"
    module_id = ""

    def __init__(self, provider: Optional[AdapterProvider] = None, timeout_s: float = 120.0):
        self.provider = provider or shared_provider()
        self.timeout_s = timeout_s

    @property
    def version(self) -> str:
        try:
            return self.provider.checkpoint_id()
        except Exception:  # noqa: BLE001 - only the label; run() raises the clear error
            return "checkpoint-unavailable"

    def _result(self, **kw) -> ModuleResult:
        return ModuleResult(module_id=self.module_id, version=self.version, section=self.section, source=self.source, **kw)

    def _not_applicable(self, reason: str, warnings: Optional[list] = None) -> ModuleResult:
        return self._result(status="not_applicable", reason=reason, warnings=warnings or [])
