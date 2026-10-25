"""File to report: validate and probe a file, run every registered module through the coordinator (WS6).

Uses the v1 ``inspect`` function read-only (extension whitelist, decode check, ffprobe) and the frozen detectors only through the
wrapper modules. Capabilities with no implementation are registered as ``PlannedModule`` so that the report's coverage lists them as
unsupported with the reason, instead of leaving them out and implying they ran.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Optional

from dfd.infer.library import inspect          # raises dfd.infer.library.InputError for unusable files

from .coordinator import Coordinator
from .module import Applicability, Probe, probe_from_media
from .provenance.modules import build_provenance_modules


class PlannedModule:
    """A capability that is designed but not built. Never runs; always reported as unsupported with its reason."""
    version = "not-implemented"
    timeout_s = 1.0
    source = "real"

    def __init__(self, module_id: str, section: str, reason: str):
        self.module_id, self.section, self._reason = module_id, section, reason

    def applicability(self, probe: Probe) -> Applicability:
        return Applicability.unsupported(self._reason)

    def run(self, ctx):                         # pragma: no cover - the coordinator never reaches it
        raise RuntimeError("planned module has no implementation")


def planned_modules() -> list:
    return [
        PlannedModule("generated_video", "authenticity", "not implemented: no detector for fully generated video exists yet "
                      "(plan in docs/GENERATED_VIDEO_PLAN.md)"),
        PlannedModule("lip_sync", "cross_modal", "not implemented: audio-video synchronisation module not built yet "
                      "(decision note in docs/LIPSYNC_DECISION_NOTE.md)"),
    ]


def default_modules() -> list:
    from .modules import build_default_modules
    return build_default_modules() + planned_modules() + build_provenance_modules()


def analyse_file(path: str | Path, filename: Optional[str] = None, modules: Optional[list] = None,
                 request_id: Optional[str] = None, progress: Optional[Callable[[str], None]] = None) -> dict:
    p = Path(path)
    media = inspect(p, filename or p.name)
    probe = probe_from_media(media)
    mods = default_modules() if modules is None else modules
    return Coordinator(mods).analyse(probe, request_id or str(uuid.uuid4()), progress)
