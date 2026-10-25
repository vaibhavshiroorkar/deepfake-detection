"""Provenance-section modules built on ``metadata.py`` and ``c2pa_verify.py`` (WS3). Source is "real" (a real local computation) but
neither module is a detector: neither has a model block or a calibrated score, and neither can say a file is authentic or fake.
"""
from __future__ import annotations

from ..module import Applicability, Context, Probe
from ..schema import ModuleResult
from . import c2pa_verify as V
from . import metadata as M

_NO_MODEL_NOTE = "Local file inspection with ffprobe and a byte scan; no model is involved."

_LIM_SIDECAR = "Credentials held in a sidecar file or a cloud store cannot be found by a local scan of the file."
_LIM_ABSENCE = ("Absence of credentials proves nothing: credentials are optional and are removed by many editing and sharing paths. "
                "Presence does not show that a file is authentic.")
_LIM_WATERMARK = "SynthID and other invisible watermarks are not checked."
_LIM_VALID = ("A valid credential attests to signed claims about the asset at signing time; it does not prove that every pixel or "
              "sample is authentic or that the content is true.")


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
    """C2PA Content Credentials. When ``c2pa-python`` is importable the manifest is validated by the library (claim signature,
    assertion and data hashes) through ``c2pa_verify.verify``; otherwise, or when the library cannot read the file, only a
    marker-byte scan is reported and nothing is claimed as verified. Not a detector: a credential says who signed which claims,
    not whether content is real.

    Finding map (decided autonomously - needs user review): ``credentials_valid`` only when the library reports validation state
    ``Trusted`` (valid and the signer chains to a configured trust anchor). A manifest that validates but whose signer is on no
    trust list (the default: none is bundled) is ``inconclusive`` with a verification observation saying integrity was checked and
    the signer is untrusted. Failed validation is ``credentials_invalid``; no manifest is ``no_credentials_found``."""
    module_id = "content_credentials"
    version = "2-c2pa"
    section = "provenance"
    timeout_s = 60.0
    source = "real"

    def applicability(self, probe: Probe) -> Applicability:
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        path = ctx.probe.path
        scan = M.scan_c2pa_markers(path)
        res = V.verify(path)
        st = res["status"]
        if st == "unavailable":
            return self._marker_only(scan, "No C2PA verifier is installed in this environment.", res)
        if st == "verified":
            return self._verified(res, scan)
        if st == "absent":
            if scan["both_markers"]:
                return self._marker_only(scan, "The C2PA library found no readable manifest although marker bytes occur.", res)
            return self._build("no_credentials_found", scan, [self._marker_obs("C2PA_MARKER_ABSENT",
                               "no C2PA manifest found by the C2PA library", scan)], [_LIM_ABSENCE, _LIM_SIDECAR, _LIM_WATERMARK], res)
        code = "C2PA_FORMAT_UNSUPPORTED" if st == "unsupported" else "C2PA_READ_ERROR"
        reason = res["reason"] or "the library could not read the file"
        obs = [self._marker_obs(code, reason, scan)]
        if scan["both_markers"]:       # keep the byte-scan fact first: markers occur but the library could not validate them
            obs.insert(0, self._marker_obs("C2PA_MARKER_PRESENT", "C2PA/JUMBF marker bytes occur in the file; a manifest may be "
                                                                  "embedded but is unverified", scan))
        return self._build("inconclusive", scan, obs,
                           [f"No credential was verified: {reason}.", _LIM_SIDECAR, _LIM_ABSENCE, _LIM_WATERMARK], res,
                           warnings=[{"code": code, "message": reason}])

    @staticmethod
    def _marker_obs(code, text, scan) -> dict:
        return {"kind": "marker_scan", "code": code, "text": text, "value": scan["found"]}

    def _build(self, finding, scan, obs, limits, res, warnings=None) -> ModuleResult:
        lib = res.get("library") or {}
        checked = res["status"] != "unavailable"
        scope = "Local file inspection: a byte scan for C2PA marker bytes"
        if checked and lib.get("c2pa_python"):
            scope += (f" and a C2PA manifest check with c2pa-python {lib['c2pa_python']} (C2PA SDK {lib['c2pa_sdk']}); remote "
                      "manifest fetch and OCSP are off, no model is involved.")
        else:
            scope += "; no model is involved and no manifest was validated."
        return ModuleResult(module_id=self.module_id, version=self.version, section=self.section, source=self.source,
                            status="completed", finding=finding, units_analysed=scan["scanned_bytes"], observations=obs,
                            warnings=warnings or [], tested_scope=scope, limitations=limits)

    def _marker_only(self, scan, why, res) -> ModuleResult:
        if scan["both_markers"]:
            finding, code = "inconclusive", "C2PA_MARKER_PRESENT"
            text = "C2PA/JUMBF marker bytes occur in the file; a manifest may be embedded but is unverified"
        else:
            finding, code = "no_credentials_found", "C2PA_MARKER_ABSENT"
            text = "no embedded C2PA/JUMBF marker bytes found" + (" in the scanned part of the file" if scan["truncated"] else "")
        lim = [f"No credential was verified: {why}", _LIM_SIDECAR, _LIM_ABSENCE, _LIM_WATERMARK]
        return self._build(finding, scan, [self._marker_obs(code, text, scan)], lim, res)

    def _verified(self, r, scan) -> ModuleResult:
        state, trust = r["validation_state"], r["trust"]
        signer = r["signer"] or {}
        who = signer.get("common_name") or signer.get("issuer") or "unknown signer"
        codes = r["failure_codes"]
        hard = [c for c in codes if c != V.UNTRUSTED_CODE]
        warns, lim = [], [_LIM_VALID, _LIM_SIDECAR, _LIM_ABSENCE, _LIM_WATERMARK]
        if state == "Invalid" or hard:
            finding, code = "credentials_invalid", "C2PA_VALIDATION_FAILED"
            text = "C2PA validation failed: " + ", ".join(hard or codes) + ". The file or its manifest does not match what was signed."
            lim.insert(1, "A failed check shows a mismatch with what was signed (edit, re-encode or damage); it does not say why or by whom.")
        elif state == "Trusted":
            finding, code = "credentials_valid", "C2PA_VALID_TRUSTED"
            text = (f"manifest signature and hashes match and the signing certificate chains to a trust anchor configured by the operator "
                    f"(signer: {who}); this is relative to that configuration and does not make the content authentic")
            lim.insert(1, "Trust is only as good as the configured trust anchors (DFD_C2PA_TRUST_ANCHORS); none are built in.")
        elif state == "Valid" and trust == "untrusted_signer":
            finding, code = "inconclusive", "C2PA_VALID_SIGNER_UNTRUSTED"
            text = (f"manifest integrity checked (signature and hashes match) but the signer ({who}) is not on a trust list, so who "
                    "signed it is not established")
            warns.append({"code": code, "message": text})
            lim.insert(1, "No trust anchor is configured, or the signer does not chain to a configured one, so the signer identity is not "
                          "verified; anyone can sign a manifest with a self-made certificate.")
        else:
            finding, code = "inconclusive", "C2PA_STATE_UNRECOGNISED"
            text = f"the library reported validation state {state!r}, which this module does not map to a verdict"
        obs = [{"kind": "verification", "code": code, "text": text,
                "value": {"validation_state": state, "trust": trust, "failure_codes": codes, "signer": r["signer"],
                          "claim_generator": r["claim_generator"], "statuses": r["validation_statuses"]}}]
        if r["actions"]:
            obs.append({"kind": "file_metadata", "code": "C2PA_ACTIONS", "value": r["actions"],
                        "text": "actions declared in the manifest: " + ", ".join(a["action"] or "?" for a in r["actions"])})
        for d in r["digital_source_types"]:
            ai = any(t in d for t in V.AI_SOURCE_TYPES)
            obs.append({"kind": "file_metadata", "code": "C2PA_DECLARED_SOURCE_AI" if ai else "C2PA_DECLARED_SOURCE", "value": d,
                        "text": f"manifest declares digitalSourceType {d}" + (
                            " (generative AI): a declared origin stated by the signer, not a detection by this system" if ai else "")})
        return self._build(finding, scan, obs, lim, r, warnings=warns)


def build_provenance_modules() -> list:
    return [MediaMetadataModule(), ContentCredentialsModule()]
