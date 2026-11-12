"""Evidence behind every verdict rule (D-087).

Each entry names a measured quantity, the result file it is stored in, the path inside that file and the value that was verified
against the file on 2026-10-27. ``tests/platform_verdict/test_evidence_registry.py`` re-reads the files and fails if a value
differs, so no sentence of the verdict layer can drift away from a stored result. The text is what the user sees; the numbers in it
are the stored values rounded as written. The files live under ``data/`` (git-ignored); the test skips, and says so, when a file is
absent.
"""
from __future__ import annotations

from typing import Optional

# id -> {"text", "file", "path", "value", "experiment"}; "path" is a "/"-separated key path into the JSON file.
EVIDENCE = {
    "E015_visual_sensitivity": {
        "experiment": "E-015", "file": "data/_results/e015/e015_results.json",
        "path": "results/H1_visual_pooled/sensitivity_at_frozen_threshold", "value": 0.4804,
        "text": "On the sealed half of FakeAVCeleb the frozen video threshold caught 48 percent of the fake videos, so a missed fake is common."},
    "E015_visual_specificity": {
        "experiment": "E-015", "file": "data/_results/e015/e015_results.json",
        "path": "results/H1_visual_pooled/specificity_at_frozen_threshold", "value": 0.996,
        "text": "On the sealed half of FakeAVCeleb the frozen video threshold flagged 0.4 percent of the unmanipulated videos, so a flag was rarely a false alarm there."},
    "E015_audio_sensitivity": {
        "experiment": "E-015", "file": "data/_results/e015/e015_results.json",
        "path": "results/H2_audio_pooled/sensitivity_at_frozen_threshold", "value": 0.9615,
        "text": "On the sealed half of FakeAVCeleb the frozen speech threshold caught 96 percent of the fake audio."},
    "E015_audio_specificity": {
        "experiment": "E-015", "file": "data/_results/e015/e015_results.json",
        "path": "results/H2_audio_pooled/specificity_at_frozen_threshold", "value": 0.6367,
        "text": "On the sealed half of FakeAVCeleb the frozen speech threshold flagged 36 percent of the unmanipulated audio, so a flag alone is weak evidence."},
    "E028_tv_still_fpr": {
        "experiment": "E-028", "file": "data/_results/modern/e028/e028_results.json",
        "path": "face_path/metrics/face_crop_scored/pooled/T-V/fpr/est", "value": 0.2152,
        "text": "On 1,524 camera photographs with a face from a public benchmark, the face-swap model flagged 22 percent."},
    "E028_tv_still_fnr": {
        "experiment": "E-028", "file": "data/_results/modern/e028/e028_results.json",
        "path": "face_path/metrics/face_crop_scored/pooled/T-V/fnr/est", "value": 0.6621,
        "text": "On the generated faces of that benchmark the face-swap model missed 66 percent."},
    "E028_ti_fnr": {
        "experiment": "E-028", "file": "data/_results/modern/e028/e028_results.json",
        "path": "main/pooled/T-I/fnr/est", "value": 0.7395,
        "text": "On 21 modern generators the generated-image model missed 74 percent of the generated images at its threshold, so a low score is not reassuring."},
    "E028_ti_fpr": {
        "experiment": "E-028", "file": "data/_results/modern/e028/e028_results.json",
        "path": "main/pooled/T-I/fpr/est", "value": 0.0418,
        "text": "On the same 21-generator benchmark the generated-image model flagged 4 percent of the photographs overall, and more of some kinds (PNG files and RAISE photographs: 12 and 16 percent)."},
    "E028_ti_face_fnr": {
        "experiment": "E-028", "file": "data/_results/modern/e028/e028_results.json",
        "path": "errors/fnr_by_face_routed/face/fnr", "value": 0.8063,
        "text": "On benchmark images that contain a face the generated-image model missed 81 percent of the generated ones."},
    "E030_speech_eer_itw": {
        "experiment": "E-030", "file": "data/_results/e030/e030_results.json",
        "path": "rows/last/eer_estimate", "value": 0.1578,
        "text": "On 42 sealed In-the-Wild speakers the speech system's equal error rate was 15.8 percent."},
    "E026_installed_video_head_fpr": {
        "experiment": "E-026", "file": "data/_results/genvideo/e026/e026_results.json",
        "path": "sources/pooled R-FF + R-LAV/fpr", "value": 0.7682,
        "text": "The installed generated-video head scored 77 percent of 440 unmodified videos from two other sources as generated-like."},
    "E027_candidate_video_head_fpr": {
        "experiment": "E-027", "file": "data/_results/modern/e027/confound_results.json",
        "path": "sources/pooled/at_t_BA/fpr", "value": 0.8023,
        "text": "A video head trained on more generators (E-027) flagged 80 percent of unmodified videos from those two sources."},
    "E029_candidate_image_fpr": {
        "experiment": "E-029", "file": "data/_results/modern/e029/e029_results.json",
        "path": "main/pooled/IMG-13_seed0/fpr/est", "value": 0.5213,
        "text": "A generated-image head trained on 125 recent generators ranked images better but flagged 52 percent of photographs at its threshold."},
}


def resolve_path(obj, path: str):
    """Follow a ``/`` path; the segment ``last`` selects the last item of a list."""
    cur = obj
    for seg in path.split("/"):
        if isinstance(cur, list):
            cur = cur[-1] if seg == "last" else cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


def text_of(evidence_id: str) -> Optional[dict]:
    e = EVIDENCE.get(evidence_id)
    if e is None:
        return None
    return {"id": evidence_id, "experiment": e["experiment"], "text": e["text"]}
