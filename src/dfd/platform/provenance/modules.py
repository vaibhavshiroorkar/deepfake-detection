"""Provenance-section modules built on ``metadata.py`` (WS3). Source is "real" (a real local computation) but neither module is a
detector: neither has a model block or a calibrated score, and neither can say a file is authentic or fake.
"""
from __future__ import annotations

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from . import metadata as M

_NO_MODEL_NOTE = "Local file inspection with ffprobe and a byte scan; no model is involved."


class MediaMetadataModule:
    """Ordinary container and stream description plus descriptive heuristic flags. Always ``inconclusive``: metadata describes a
    file, it does not establish where the content came from."""
    module_id = "media_metadata"
    version = "1"
    section = "provenance"
    timeout_s = 30.0
    source = "real"

    def applicability(self, probe: Probe) -> Applicability:
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        info = M.ffprobe(ctx.probe.path)
        obs = [o for o in M.describe(info) + M.heuristics(info)]
        flags = [o for o in obs if o["kind"] == "heuristic"]
        return ModuleResult(
            module_id=self.module_id, version=self.version, section=self.section, source=self.source, status="completed",
            finding="inconclusive", observations=obs,
            warnings=[{"code": o["code"], "message": o["text"]} for o in flags],
            tested_scope=_NO_MODEL_NOTE + " Reports container, stream and tag properties and a few descriptive flags.",
            limitations=["Metadata can be stripped, rewritten or copied; ordinary editing, transcoding and sharing change it.",
                         "Missing or unusual metadata does not show a file is fake; present metadata does not show it is genuine.",
                         "The heuristic flags are descriptive and have not been evaluated as evidence of manipulation."])


class ContentCredentialsModule:
    """Looks for embedded C2PA/JUMBF marker bytes. Marker presence is reported as unverified (``inconclusive``). Cryptographic
    verification needs a C2PA library that is not installed, so ``credentials_valid`` and ``credentials_invalid`` are never produced."""
    module_id = "content_credentials"
    version = "1-marker-scan"
    section = "provenance"
    timeout_s = 60.0
    source = "real"

    def applicability(self, probe: Probe) -> Applicability:
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        scan = M.scan_c2pa_markers(ctx.probe.path)
        verifier = M.c2pa_verifier_available()
        limits = ["No credential was verified: signatures, certificate chains and content hashes are not checked "
                  "(no C2PA verifier is installed in this environment)." if not verifier else
                  "A verifier package is importable but is not wired into this module; nothing was verified.",
                  "Credentials held in a sidecar file or a cloud store cannot be found by a byte scan.",
                  "Absence of credentials does not show that a file is fake, and presence does not show that it is authentic.",
                  "SynthID and other invisible watermarks are not checked."]
        if scan["both_markers"]:
            finding, text, code = "inconclusive", "C2PA/JUMBF marker bytes occur in the file; a manifest may be embedded but is unverified", "C2PA_MARKER_PRESENT"
        else:
            finding, text, code = "no_credentials_found", "no embedded C2PA/JUMBF marker bytes found" + (
                " in the scanned part of the file" if scan["truncated"] else ""), "C2PA_MARKER_ABSENT"
        return ModuleResult(
            module_id=self.module_id, version=self.version, section=self.section, source=self.source, status="completed",
            finding=finding, units_analysed=scan["scanned_bytes"], observations=[{"kind": "marker_scan", "code": code, "text": text,
                                                                                   "value": scan["found"]}],
            tested_scope=_NO_MODEL_NOTE + " Searched for embedded C2PA manifest marker bytes only.", limitations=limits)


def build_provenance_modules() -> list:
    return [MediaMetadataModule(), ContentCredentialsModule()]
