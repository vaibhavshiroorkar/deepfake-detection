"""Deterministic, evidence-grounded explanations of module results (report schema 2.1, decision D3 of
docs/PRODUCT_REDESIGN_DECISIONS.md).

Pure functions, no model and no I/O. Every sentence is built from fields of the module result (or of another module in the
same report) and names them in ``sources``, so a test can check that each number shown equals a stored value and that each
claim has a field behind it. When a field a template needs is missing, the sentence is left out; it is never filled with a
default. The text describes what a module examined, observed and reports, and what that does not establish. It never
combines modules into one judgement and never turns a score into a probability.

Path grammar of ``sources`` (resolved by ``resolve``): ``a.b``, ``a[2]``, ``a[code=X]`` (first list item whose ``code`` is X),
``a[kind=Y].band``; a leading ``@<module_id>.`` reads another module of the same report; ``probe.x`` reads the report probe.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from . import schema, verdict

TEMPLATE_VERSION = "v2"   # v2 (2026-10-27): processing and decision sentences; v1 explanations stay valid

# Human names and a one-line description of what each module checks. Internal ids stay in the JSON and in technical details.
MODULES = {
    "face_video": ("Face manipulation analysis", "Looks for face-swap manipulation in the largest face of a video."),
    "image_detector": ("Image analysis", "Checks a still image: a face is checked for face-swap signs; an image without a face is checked "
                                         "for signs of being fully generated."),
    "speech": ("Synthetic speech analysis", "Looks for synthetic or converted speech in the first seconds of the audio."),
    "lip_sync": ("Audio-visual synchronisation", "Measures whether the audio and the mouth movements of one face are in step."),
    "generated_image": ("AI-generated image, whole image (experimental)", "When a face was found in a still image, also gives the whole "
                                                                          "image to the generated-image model. It makes no finding."),
    "generated_video": ("AI-generated video (experimental research)", "A research prototype for fully generated video. It makes no finding."),
    "media_metadata": ("File information", "Reads the container, streams and tags of the file."),
    "content_credentials": ("Content credentials (C2PA)", "Looks for embedded C2PA content credentials and checks them."),
    "c2pa": ("Content credentials (C2PA)", "Looks for embedded C2PA content credentials."),
}
# Which kinds of file each check can run on, and what else it needs, for the upload panel. tests/platform_explain checks this table
# against every module's own applicability() so the two cannot drift apart.
APPLIES_TO = {
    "face_video": (("video",), "Videos; needs a visible face"),
    "image_detector": (("image",), "Still images"),
    "speech": (("audio", "video"), "Audio files, and videos with a sound track"),
    "lip_sync": (("video",), "Videos with a sound track and a visible face"),
    "generated_image": (("image",), "Still images in which a face is found (experimental, makes no finding)"),
    "generated_video": (("video",), "Videos (experimental, makes no finding)"),
    "media_metadata": (("image", "video", "audio"), "All files"),
    "content_credentials": (("image", "video", "audio"), "All files"),
}
FINDING_LABEL = {
    "indicates_manipulation": "Indicates manipulation",
    "no_indication_within_tested_scope": "No indication within tested scope",
    "inconclusive": "Inconclusive",
    "consistent": "No mismatch found within tested scope",
    "potential_mismatch": "Potential mismatch",
    "credentials_valid": "Credentials present and valid",
    "credentials_invalid": "Credentials present but invalid",
    "no_credentials_found": "No credentials found",
}
STATUS_LABEL = {"completed": "Completed", "not_applicable": "Not applicable", "unsupported": "Unsupported", "failed": "Failed",
                "timed_out": "Timed out", "skipped": "Skipped"}
# Display groups, in the order the dashboard lists them. A group is a presentation of status and finding, never a verdict.
GROUPS = (
    ("attention", "Merits attention"),
    ("nothing_found", "Nothing found within tested scope"),
    ("credential", "Credential verified"),
    ("inconclusive", "Inconclusive"),
    ("informational", "File information"),
    ("experimental", "Experimental, no finding made"),
    ("not_applicable", "Did not apply to this file"),
    ("unsupported", "Not available"),
    ("did_not_run", "Did not complete"),
)
GROUP_LABEL = dict(GROUPS)
KINDS = ("mock", "processing", "examined", "observation", "model_output", "decision", "interpretation", "not_established", "limitation")
# The questions the dashboard answers, in order, and the sentence kinds under each (roadmap 6.2 and 6.3: processing history, observed
# evidence and interpretation kept apart). "processing" is what the pipeline did, from recorded fields; "decision" is the fixed rule that
# turned the observation into the reported finding.
EXPLANATION_PARTS = (
    ("mock", "Scenario data", ("mock",)),
    ("processing", "What we did", ("processing",)),
    ("examined", "What we examined", ("examined",)),
    ("observed", "What we observed", ("observation", "model_output")),
    ("decision", "Why this was reported", ("decision",)),
    ("interpretation", "What it means", ("interpretation",)),
    ("not_established", "What it does not establish", ("not_established",)),
    ("limitation", "Limitations", ("limitation",)),
)
RESULT_KINDS = {"model_output", "decision", "interpretation", "processing"}   # only a module that completed may carry these
NUMBER_CONSTANTS = {"0", "1"}          # "on a 0 to 1 scale"; any other number must come from a cited field
FORBIDDEN = [r"\bauthentic\w*", r"\bgenuine\b", r"\bsafe\b", r"likely real", r"verified real", r"\b(is|are|looks?|appears?) real\b",
             r"fake probability", r"probability of fake", r"\bproves?\b", r"deepfake detected", r"\bis (a )?fake\b", "%"]
GENERIC_NOT_ESTABLISHED = {
    "authenticity": "A result from one check does not show how the file was made; other checks can disagree.",
    "cross_modal": "A timing result alone does not show whether the file was manipulated.",
    "provenance": "Credentials and metadata describe the file; they do not show whether its content is true.",
}


# ---------------------------------------------------------------------------------------------------------------- paths
_STEP = re.compile(r"([A-Za-z_]\w*)((?:\[[^\]]+\])*)")


def resolve(m: dict, path: str, report: Optional[dict] = None) -> Any:
    """The value at ``path`` in module result ``m`` (see the module docstring), or None when any step is missing."""
    cur: Any = m
    if path.startswith("@"):
        mid, _, path = path[1:].partition(".")
        cur = next((x for x in (report or {}).get("modules", []) if x.get("module_id") == mid), None)
    elif path.startswith("probe."):
        cur, path = (report or {}).get("probe"), path[len("probe."):]
    for part in path.split("."):
        if cur is None:
            return None
        g = _STEP.fullmatch(part)
        if not g:
            return None
        cur = cur.get(g.group(1)) if isinstance(cur, dict) else None
        for sel in re.findall(r"\[([^\]]+)\]", g.group(2)):
            if isinstance(cur, list) and sel.isdigit():
                cur = cur[int(sel)] if int(sel) < len(cur) else None
            elif isinstance(cur, list) and "=" in sel:
                k, v = sel.split("=", 1)
                cur = next((x for x in cur if isinstance(x, dict) and str(x.get(k)) == v), None)
            else:
                return None
    return cur


# ---------------------------------------------------------------------------------------------------------------- formatting
def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def fmt_score(value: float, threshold: Optional[float] = None) -> str:
    """Two decimals; three or four when two would print the score and the threshold alike, so the side of the line stays visible."""
    if threshold is None:
        return fmt(value)
    for d in (2, 3):
        if fmt(value, d) != fmt(threshold, d):
            return fmt(value, d)
    return fmt(value, 4)


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def _sentence(kind: str, text: str, *sources: str, quoted: bool = False) -> dict:
    """``quoted``: the text is a field's own words (a warning, reason or observation), shown verbatim apart from capitalisation."""
    d = {"kind": kind, "text": text, "sources": list(sources)}
    if quoted:
        d["quoted"] = True
    return d


def _quote(kind: str, text: str, source: str) -> dict:
    return _sentence(kind, _cap(_end(text)), source, quoted=True)


def _lim_index(m: dict, needle: str) -> Optional[int]:
    """Index of the first limitation containing ``needle`` (case-insensitive), so a paraphrase can cite the string it summarises."""
    for i, s in enumerate(m.get("limitations") or []):
        if needle.lower() in str(s).lower():
            return i
    return None


def _lim(m: dict, needle: str, kind: str, text: str) -> Optional[dict]:
    i = _lim_index(m, needle)
    return None if i is None else _sentence(kind, text, f"limitations[{i}]")


def _warning(m: dict, code: str) -> Optional[dict]:
    return next((w for w in m.get("warnings") or [] if isinstance(w, dict) and w.get("code") == code), None)


def _obs(m: dict, code: str) -> Optional[dict]:
    return next((o for o in m.get("observations") or [] if isinstance(o, dict) and o.get("code") == code), None)


def _evidence(m: dict, kind: str) -> Optional[dict]:
    return next((e for e in m.get("evidence") or [] if isinstance(e, dict) and e.get("kind") == kind), None)


def _cap(s: str) -> str:
    s = str(s).strip()
    return (s[:1].upper() + s[1:]) if s else s


def _end(s: str) -> str:
    s = str(s).strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def is_experimental(m: dict) -> bool:
    return verdict.is_experimental(m)      # one definition with the verdict layer (D-087)


def group_of(m: dict) -> str:
    st = m.get("status")
    if st == "not_applicable":
        return "not_applicable"
    if st == "unsupported":
        return "unsupported"
    if st != "completed":
        return "did_not_run"
    if is_experimental(m):
        return "experimental"
    f = m.get("finding")
    if f in ("indicates_manipulation", "potential_mismatch", "credentials_invalid"):
        return "attention"
    if f in ("no_indication_within_tested_scope", "consistent", "no_credentials_found"):
        return "nothing_found"
    if f == "credentials_valid":
        return "credential"
    if m.get("module_id") == "media_metadata":
        return "informational"
    return "inconclusive"


def _score_sentences(m: dict, what: str) -> list:
    """Model output and the side of the fixed decision line. Uses score.value and score.threshold only."""
    s = m.get("score") or {}
    v, t = s.get("value"), s.get("threshold")
    if not isinstance(v, (int, float)):
        return []
    if not isinstance(t, (int, float)):
        return [_sentence("model_output", f"The {what} gave a score of {fmt(v)} on a 0 to 1 scale.", "score.value")]
    out = [_sentence("model_output", f"The {what} gave a score of {fmt_score(v, t)} on a 0 to 1 scale; its fixed decision line is "
                                     f"{fmt(t)}.", "score.value", "score.threshold")]
    said = {"indicates_manipulation": "signs of manipulation", "no_indication_within_tested_scope": "no indication within its tested scope"}
    if m.get("finding") in said:
        side = "at or above" if v >= t else "below"
        out.append(_sentence("decision", f"The score is {side} the decision line, so the check reports {said[m['finding']]}.",
                             "score.value", "score.threshold", "finding"))
    return out


# ---------------------------------------------------------------------------------------------------------------- templates
def _not_completed(m: dict) -> tuple[str, list, str]:
    st = m.get("status")
    headline = {"not_applicable": "Did not apply to this file", "unsupported": "Not available for this file",
                "failed": "Did not complete: the check failed", "timed_out": "Did not complete within its time limit",
                "skipped": "Not started"}.get(st, "Did not complete")
    sents = []
    if m.get("reason"):
        sents.append(_sentence("observation", "Reason: " + _end(m["reason"]), "reason", quoted=True))
    if m.get("errors"):
        e = m["errors"][0]
        sents.append(_sentence("observation", f"Error {e.get('code')}: {_end(e.get('message', ''))}", "errors[0]", quoted=True))
    for i, w in enumerate(m.get("warnings") or []):
        if isinstance(w, dict) and w.get("message"):
            sents.append(_quote("observation", w["message"], f"warnings[{i}]"))
    sents.append(_sentence("not_established", "No finding is made by this check for this file. The other checks are reported "
                                              "separately and are not affected.", "status"))
    return headline, sents, "This check gives no information about the file."


def _face_video(m: dict, r: dict) -> tuple[str, list, str]:
    hit = m["finding"] == "indicates_manipulation"
    headline = ("Signs of face manipulation were found in the sampled frames" if hit
                else "No indication of face manipulation was found within the model's tested scope")
    s = []
    n = m.get("units_analysed")
    fs = (_obs(m, "frames_sampled") or {}).get("value")
    if isinstance(fs, int) and isinstance(n, int):
        s.append(_sentence("processing", f"It took {fs} frames at even intervals across the video and searched each for faces with a face "
                                         f"detector (YuNet); {n} of them contained a face.",
                           "observations[code=frames_sampled].value", "units_analysed", "tested_scope"))
    if "1.3 enlarged" in str(m.get("tested_scope", "")):
        s.append(_sentence("processing", "In each of those frames the largest face was cut out with a margin (1.3 times the face box) and "
                                         "resized to 224 by 224 pixels.", "tested_scope"))
    s.append(_lim(m, "mean over the sampled frames", "processing", "The face model scored each face crop, and the result is the mean of "
                                                                   "those scores; frames without a face were left out."))
    if isinstance(n, int):
        s.append(_sentence("examined", f"It examined only the largest face in {n} sampled frames that contained a face (at most 16 "
                                       "frames are sampled); the rest of each frame and the frames in between were not examined.",
                           "units_analysed", "tested_scope"))
        s.append(_sentence("limitation", "The report does not record how many different people appear, or the score of each frame.",
                           "units_analysed"))
    if _warning(m, "MULTIPLE_FACES"):
        s.append(_sentence("observation", "Several faces appeared; only the largest face was scored.", "warnings[code=MULTIPLE_FACES]"))
    s += _score_sentences(m, "face manipulation model")
    s.append(_sentence("interpretation", ("This means the face crops resembled the face-swap material the model was trained on, more than "
                                          "the unaltered faces in it, by enough to cross its decision line." if hit else
                                          "This means the face crops did not resemble the face-swap material the model was trained on "
                                          "closely enough to cross its decision line."), "finding", "model.evaluation"))
    s.append(_lim(m, "Trained and sealed-tested on face-swap", "limitation",
                  "It was trained and tested on face-swap videos. It was not evaluated on fully generated talking-head video, and on "
                  "one dataset of content edits (LAV-DF) it ranked videos worse than chance."))
    s.append(_lim(m, "audio only", "limitation", "A change to the audio alone is not visible to this check."))
    if hit:
        dnm = "This does not show who made the video or how, and other checks can disagree."
    else:
        dnm = "This does not establish that the video is unmanipulated: the model looks for one kind of face manipulation only."
    s.append(_sentence("not_established", dnm, "finding"))
    return headline, s, dnm


def _image_detector(m: dict, r: dict) -> tuple[str, list, str]:
    hit = m["finding"] == "indicates_manipulation"
    scope = str(m.get("tested_scope", ""))
    s = []
    if scope.startswith("path=face_crop"):
        headline = ("Signs of face manipulation were found in the face in this image" if hit
                    else "No indication of face manipulation was found in the face in this image, within the model's tested scope")
        s.append(_sentence("processing", "A face detector (YuNet) searched the image and found a face, so the largest face was cut out "
                                         "with a margin (1.3 times the face box), resized to 224 pixels and scored once by the face-swap "
                                         "model built for video frames.", "tested_scope"))
        if isinstance(m.get("units_analysed"), int):
            s.append(_sentence("examined", f"It examined {m['units_analysed']} face crop: the largest face of the image. The rest of the "
                                           "image was not examined.", "units_analysed", "tested_scope"))
        if _warning(m, "MULTIPLE_FACES"):
            s.append(_sentence("observation", "Several faces appeared; only the largest face was scored.", "warnings[code=MULTIPLE_FACES]"))
        s += _score_sentences(m, "face manipulation model")
        s.append(_sentence("interpretation", ("This means the face resembled the face-swap material the model was trained on by enough to "
                                              "cross its decision line." if hit else "This means the face did not resemble that face-swap "
                                              "material closely enough to cross the decision line."), "finding", "model.evaluation"))
        s.append(_lim(m, "not evaluated on still face-swapped photos", "limitation",
                      "That model was not evaluated on still photos, and its score is not calibrated for them."))
        gi = _module(r, "generated_image")
        if gi and gi.get("status") == "completed" and gi.get("source") != "mock":
            s.append(_sentence("limitation", "This check does not use the generated-image model when a face is found. The separate "
                                             "experimental check on the whole image gives the image to that model, but makes no finding.",
                               "tested_scope", "@generated_image.status"))
        else:
            s.append(_lim(m, "whole-image probe for generated images is not used", "limitation",
                          "The generated-image check did not run on this image because a face was found, so a fully AI-generated "
                          "portrait is not examined by the generated-image model."))
        dnm = ("This does not show how the image was made." if hit else
               "This does not establish that the image is unmanipulated or that it was not generated by AI.")
    elif scope.startswith("path=whole_image"):
        headline = ("Signs that the image is fully generated were found" if hit
                    else "No indication that the image is fully generated was found within the model's tested scope")
        s.append(_sentence("processing", "A face detector found no face, so the whole image was given to the CLIP ViT-L/14 image encoder "
                                         "and a linear head scored its features.", "tested_scope", "model.name"))
        s.append(_sentence("examined", "No face was found, so the whole image was checked for signs of being fully generated.", "tested_scope"))
        s += _score_sentences(m, "generated-image model")
        s.append(_sentence("interpretation", ("This means the image resembled the generated images the model was trained to tell apart from "
                                              "photographs, by enough to cross its decision line." if hit else "This means the image did "
                                              "not resemble those generated images closely enough to cross the decision line."),
                           "finding", "model.evaluation"))
        s.append(_lim(m, "Trained on ProGAN", "limitation",
                      "It was trained on images of one older generator (ProGAN) and tested on some other GAN and diffusion image sets; it "
                      "does not cover every generator, nor edited or partly generated photos."))
        dnm = ("This does not show which tool made the image." if hit else
               "This does not establish that the image was not generated or edited.")
    else:
        return _generic(m, r)
    s.append(_sentence("not_established", dnm, "finding"))
    return headline, s, dnm


def _speech(m: dict, r: dict) -> tuple[str, list, str]:
    hit = m["finding"] == "indicates_manipulation"
    headline = ("Signs of synthetic or converted speech were found in the analysed audio" if hit
                else "No indication of synthetic or converted speech was found within the model's tested scope")
    s = []
    i = _lim_index(m, "about 4 seconds")
    if i is not None:
        src = "probe.media_type"
        where = "the audio track of the video" if (r.get("probe") or {}).get("media_type") == "video" else "the audio"
        if "64,600 samples" in str(m["limitations"][i]):
            s.append(_sentence("processing", "The audio was converted to 16 kHz mono, silence at the start and end was trimmed and the level "
                                             "was normalised; the first 64,600 samples (about 4 seconds) were kept.", f"limitations[{i}]"))
        if "HuBERT" in str((m.get("model") or {}).get("name", "")):
            s.append(_sentence("processing", "Three speech encoders (HuBERT base, WavLM base+ and Whisper small) each scored that window "
                                             "with a linear head, and their outputs were averaged.", "model.name"))
        s.append(_sentence("examined", f"It examined only the first window of about 4 seconds of {where}.", f"limitations[{i}]", src))
    s += _score_sentences(m, "speech model")
    s.append(_sentence("interpretation", ("This means the audio window resembled the synthetic or converted speech the model was trained "
                                          "on, by enough to cross its decision line." if hit else "This means the audio window did not "
                                          "resemble that material closely enough to cross the decision line."), "finding", "model.evaluation"))
    if _warning(m, "AUDIO_CALIBRATION_EXPLORATORY"):
        s.append(_sentence("limitation", "The score's calibration is exploratory: read it as a ranking, not as a probability.",
                           "warnings[code=AUDIO_CALIBRATION_EXPLORATORY]"))
    ls = _lim(m, "does not check whether the audio contains speech", "limitation",
              "It does not check whether the audio contains speech: any sound is scored, so on music, tones or noise the result "
              "carries no information.")
    s.append(ls)
    lip = next((x for x in r.get("modules", []) if x.get("module_id") == "lip_sync"), None)
    if lip is not None and _warning(lip, "AUDIO_NOT_SPEECH_LIKE"):
        s.append(_sentence("observation", "The audio-visual synchronisation check screened this audio with a simple heuristic (not a "
                                          "speech detector) and found it unlike speech, so this result should not be relied on.",
                           "@lip_sync.warnings[code=AUDIO_NOT_SPEECH_LIKE]"))
    s.append(_lim(m, "In-the-Wild", "limitation", "On speech from sources other than its training data, its tested results were weaker."))
    if (r.get("probe") or {}).get("media_type") == "video":
        s.append(_lim(m, "LAV-DF", "limitation", "On the audio tracks of one video dataset (LAV-DF) it separated the classes only weakly "
                                                 "(AUC 0.6462), so on video soundtracks it is weak evidence."))
    dnm = ("This does not show who made the audio or how." if hit else
           "This does not establish that the speech is human or unedited.")
    s.append(_sentence("not_established", dnm, "finding"))
    return headline, s, dnm


def _lip_sync(m: dict, r: dict) -> tuple[str, list, str]:
    f = m["finding"]
    s = []
    if "+-15 frames" in str(m.get("tested_scope", "")):
        s.append(_sentence("processing", "It tracked one face, cut out the mouth region of each frame and turned the audio into short sound "
                                         "features; the SyncNet model then measured how far apart the two were for every shift of the "
                                         "audio from -15 to +15 frames.", "tested_scope"))
    fwf = (_obs(m, "frames_with_face") or {}).get("value")
    if isinstance(fwf, str) and re.fullmatch(r"\d+/\d+", fwf):
        a, b = fwf.split("/")
        s.append(_sentence("processing", f"A face was found in {a} of the {b} frames analysed.", "observations[code=frames_with_face].value"))
    s.append(_sentence("examined", "It compared the audio with the mouth movements of one tracked face over the first part of the "
                                   "video (at most 30 s).", "tested_scope"))
    off = (_obs(m, "av_offset_frames") or {}).get("value")
    conf = (_obs(m, "sync_confidence") or {}).get("value")
    curve = _evidence(m, "offset_curve")
    if f == "consistent":
        headline = "No audio-visual timing mismatch was found within the analysed segment"
        if isinstance(off, int) and curve and isinstance(curve.get("band"), list):
            lo, hi = curve["band"]
            s.append(_sentence("observation", f"The best alignment was at an offset of {_signed(off)} frames, inside the accepted band of "
                                              f"{lo} to {_signed(hi)} frames measured on real speech clips.",
                               "observations[code=av_offset_frames].value", "evidence[kind=offset_curve].band"))
        elif isinstance(off, int):
            s.append(_sentence("observation", f"The best alignment was at an offset of {_signed(off)} frames.", "observations[code=av_offset_frames].value"))
        if curve and isinstance(curve.get("min_confidence"), (int, float)):
            s.append(_sentence("decision", f"Because the best offset is inside the band and its confidence is at least "
                                           f"{fmt(curve['min_confidence'], 1)}, the rule reports the timing as consistent.",
                               "evidence[kind=offset_curve].band", "evidence[kind=offset_curve].min_confidence", "finding"))
        s.append(_sentence("interpretation", "This means the measured audio and the visible mouth movements appeared to be in step.", "finding"))
        s.append(_lim(m, "synchronised fake passes", "not_established", "It does not establish that the video is unmanipulated: a fake "
                                                                         "whose lips were made to match the audio passes this check."))
        s.append(_lim(m, "0 of 160", "limitation", "In testing, it flagged none of 160 lip-sync fakes of one known type (Wav2Lip)."))
        dnm = "Timing that lines up does not establish that the video is unmanipulated."
    elif f == "potential_mismatch":
        headline = "The audio and the mouth movements appear out of step"
        w = _warning(m, "OFFSET_BEYOND_TOLERANCE")
        if w:
            s.append(_quote("observation", w["message"], "warnings[code=OFFSET_BEYOND_TOLERANCE]"))
            s.append(_sentence("decision", "Because the best offset lies outside the accepted band and stood out clearly enough, the rule "
                                           "reports a potential mismatch.", "warnings[code=OFFSET_BEYOND_TOLERANCE]", "finding"))
        s.append(_lim(m, "A mismatch is not proof", "interpretation", "This is a timing observation, not proof of manipulation: dubbing, "
                                                                      "poor muxing and variable frame rates also cause it."))
        s.append(_lim(m, "3 of 230", "limitation", "On unmodified real clips in testing, this happened in 3 of 230 cases."))
        dnm = "A timing mismatch does not show that the video is a deepfake."
    else:
        headline = "The timing check could not reach a conclusion"
        for i, w in enumerate(m.get("warnings") or []):
            if isinstance(w, dict) and w.get("message"):
                s.append(_quote("observation", w["message"], f"warnings[{i}]"))
        s.append(_sentence("decision", "Because the conditions of the rule for a timing result were not met, no timing result is reported.",
                           "finding"))
        dnm = "No timing finding is made for this file."
    if isinstance(conf, (int, float)) and curve and isinstance(curve.get("min_confidence"), (int, float)) and f != "inconclusive":
        s.append(_sentence("observation", f"The best offset stood out with a confidence of {fmt(conf)} (the minimum for a timing result "
                                          f"is {fmt(curve['min_confidence'], 1)}).", "observations[code=sync_confidence].value",
                           "evidence[kind=offset_curve].min_confidence"))
    s.append(_lim(m, "uncalibrated rule", "limitation", "The decision rule is the project's own and is not calibrated; the score is a "
                                                       "rescaled confidence, not a probability."))
    s.append(_sentence("not_established", dnm, "finding"))
    return headline, s, dnm


def _generated_video(m: dict, r: dict) -> tuple[str, list, str]:
    if not is_experimental(m):
        return _generic(m, r)
    headline = "Experimental research output: no finding is made"
    s = []
    # units_analysed counts the decoded canonical frames (14); the encoder receives a subset of them (features.PICK), recorded in
    # the frames evidence. The sentence cites the evidence counts, never units_analysed as "frames analysed".
    fr = _evidence(m, "frames")
    i = _lim_index(m, "first 2.0 s")
    mname = str((m.get("model") or {}).get("name", ""))
    if fr and all(isinstance(fr.get(k), int) for k in ("n_frames", "n_decoded", "size_px")) and i is not None:
        s.append(_sentence("processing", f"The video was re-encoded to a standard form (its first 2.0 s, cropped to the centre square at "
                                         f"{fr['size_px']} by {fr['size_px']} pixels); {fr['n_decoded']} frames were decoded and "
                                         f"{fr['n_frames']} of them were given to an image encoder (DINOv2-small).",
                           f"limitations[{i}]", "evidence[kind=frames].size_px", "evidence[kind=frames].n_decoded",
                           "evidence[kind=frames].n_frames", "model.name"))
    if fr and isinstance(fr.get("n_frames"), int) and "mean" in mname and "logistic regression" in mname:
        s.append(_sentence("processing", f"The {fr['n_frames']} frame embeddings were averaged into one vector, and a logistic regression "
                                         "scored it. Because of the averaging, the order of the frames and the motion between them are not "
                                         "used: the model compares appearance, not movement.", "model.name", "evidence[kind=frames].n_frames"))
    if fr and isinstance(fr.get("n_frames"), int) and i is not None:
        s.append(_sentence("examined", f"The model saw only those {fr['n_frames']} frames from the first 2.0 s of the video, cropped to "
                                       "the centre square; the rest of the video was not examined.",
                           "evidence[kind=frames].n_frames", f"limitations[{i}]"))
    elif i is not None:
        s.append(_sentence("examined", "The model analysed frames from the first 2.0 s of the video, re-encoded and cropped to the "
                                       "centre square; the rest of the video was not seen.", f"limitations[{i}]"))
    v = (m.get("score") or {}).get("value")
    if isinstance(v, (int, float)):
        s.append(_sentence("model_output", f"The experimental model produced a ranking score of {fmt(v)}.", "score.value"))
        s.append(_sentence("decision", "The module is experimental, so by design it makes no finding, whatever the score.",
                           "warnings[code=EXPERIMENTAL_UNVALIDATED]", "finding"))
        s.append(_sentence("interpretation", "The score indicates how closely the clip resembles the generated clips of the model's "
                                             "development dataset. It is not a probability.", "score.meaning"))
    ev = str((m.get("model") or {}).get("evaluation", ""))
    g = re.search(r"DF26 held-out AUC (\d\.\d+)", ev)
    if g:
        s.append(_sentence("limitation", f"On an independent benchmark of generated and real videos (DF26), the model ranked them no "
                                         f"better than chance (AUC {g.group(1)}).", "model.evaluation"))
    s.append(_lim(m, "Real videos also receive high scores", "limitation",
                  "Real videos also receive high scores: in a test on unmodified real clips from two other sources, 338 of 440 scored "
                  "above the model's threshold."))
    s.append(_sentence("not_established", "This score cannot establish whether the video was AI-generated.",
                       "warnings[code=EXPERIMENTAL_UNVALIDATED]"))
    s.append(_lim(m, "Face-independent", "limitation", "It does not look for face swaps or lip-sync edits."))
    return headline, s, "This score cannot establish whether the video was AI-generated."


def _generated_image(m: dict, r: dict) -> tuple[str, list, str]:
    if not is_experimental(m):
        return _generic(m, r)
    headline = "Experimental research output: the whole image was also checked by the generated-image model; no finding is made"
    s = [_sentence("processing", "A face was found, so the image analysis scored only the face crop with the face-swap model. This extra "
                                 "check then gave the whole image, not only the face, to the CLIP ViT-L/14 image encoder and the linear "
                                 "head that the image analysis uses for images without a face.", "tested_scope", "model.name"),
         _sentence("examined", "It examined the whole image once, resized to the encoder's input size; fine detail smaller than that "
                               "is not seen.", "units_analysed", "tested_scope")]
    v = (m.get("score") or {}).get("value")
    if isinstance(v, (int, float)):
        s.append(_sentence("model_output", f"The generated-image model gave a score of {fmt(v)} on a 0 to 1 scale.", "score.value"))
        s.append(_sentence("decision", "The model missed most generated faces when it was tested on images that contain a face, so by design this check makes no "
                                       "finding, whatever the score; the decision line of the image analysis is not applied.",
                           "warnings[code=EXPERIMENTAL_PORTRAIT_SCOPE]", "finding"))
        s.append(_sentence("interpretation", "The score indicates how closely the whole image resembles the generated images the model "
                                             "was trained on, rather than its training photographs. It is not a probability.", "score.meaning"))
    i = _lim_index(m, "on images that contain a face")
    if i is not None:
        s.append(_quote("limitation", m["limitations"][i], f"limitations[{i}]"))
    s.append(_lim(m, "Trained on ProGAN", "limitation", "It was trained on images of one older generator (ProGAN), not on portraits or "
                                                        "on face generators."))
    dnm = "This score cannot establish whether the image was AI-generated, and a low score does not show that it is a photograph."
    s.append(_sentence("not_established", dnm, "warnings[code=EXPERIMENTAL_PORTRAIT_SCOPE]"))
    return headline, s, dnm


_METADATA_FACTS = ("VIDEO_STREAM", "AUDIO_STREAM", "DURATION_S", "TAG_ENCODER", "TAG_CREATION_TIME")


def _media_metadata(m: dict, r: dict) -> tuple[str, list, str]:
    headline = "File information was read. It describes the file and is not evidence either way"
    s = [_sentence("processing", "A file inspector (ffprobe) read the container, stream and tag information, and the bytes of the file "
                                 "were scanned; no model is involved.", "tested_scope"),
         _sentence("decision", "File information is reported as information only: this check never makes a finding.", "finding")]
    for code in _METADATA_FACTS:
        o = _obs(m, code)
        if o and o.get("text"):
            s.append(_quote("observation", o["text"], f"observations[code={code}].text"))
    for o in m.get("observations") or []:
        if isinstance(o, dict) and o.get("kind") == "heuristic" and o.get("text"):
            s.append(_quote("observation", o["text"], f"observations[code={o['code']}].text"))
    s.append(_lim(m, "does not show a file is fake", "not_established",
                  "Missing or present metadata does not show whether the content was manipulated; ordinary editing and sharing change it."))
    return headline, s, "Metadata describes the file, not where its content came from."


def _content_credentials(m: dict, r: dict) -> tuple[str, list, str]:
    f = m["finding"]
    s = [_sentence("processing", "It searched the file for embedded C2PA content credentials" + (
        " and checked any manifest with the C2PA library." if "manifest check" in str(m.get("tested_scope", "")) else "."), "tested_scope")]
    if f == "no_credentials_found":
        headline = "No content credentials were found in the file"
        s.append(_sentence("decision", "No manifest was found, so the check reports that no credentials are present.", "finding"))
        s.append(_lim(m, "Absence of credentials proves nothing", "not_established",
                      "Credentials are optional and are removed by most editing and sharing, so their absence says nothing about the content."))
        dnm = "A missing credential does not show that the content was generated or manipulated."
    elif f == "credentials_valid":
        headline = "Content credentials are present and their signer is trusted by this computer's configuration"
        dnm = "A valid credential confirms signed claims about the file; it does not show that the content is true."
    elif f == "credentials_invalid":
        headline = "Content credentials are present but failed validation"
        s.append(_lim(m, "failed check shows a mismatch", "interpretation",
                      "The file or its manifest does not match what was signed. Editing, re-encoding or damage can cause this; the check "
                      "does not say why or by whom."))
        dnm = "A failed credential does not show that the content is false."
    else:
        headline = ("Content credentials are present and intact, but the signer is not on a trust list"
                    if _warning(m, "C2PA_VALID_SIGNER_UNTRUSTED") else "The content credentials could not be checked")
        dnm = "No conclusion about the credentials is drawn."
    if _obs(m, "C2PA_VALID_TRUSTED"):
        s.append(_sentence("observation", "The manifest's signature and hashes match, and the signing certificate chains to a trust anchor "
                                          "configured by the operator of this computer.", "observations[code=C2PA_VALID_TRUSTED].text"))
    for code in ("C2PA_VALIDATION_FAILED", "C2PA_VALID_SIGNER_UNTRUSTED", "C2PA_MARKER_PRESENT",
                 "C2PA_FORMAT_UNSUPPORTED", "C2PA_READ_ERROR", "C2PA_STATE_UNRECOGNISED"):
        o = _obs(m, code)
        if o and o.get("text"):
            s.append(_quote("observation", o["text"], f"observations[code={code}].text"))
    if _obs(m, "C2PA_DECLARED_SOURCE_AI"):
        s.append(_sentence("observation", "The credential declares the content as made with generative AI. This is a statement by the "
                                          "signer, not a detection by this system.", "observations[code=C2PA_DECLARED_SOURCE_AI].text"))
    if f == "credentials_valid":
        s.append(_lim(m, "valid credential attests", "not_established", dnm))
    s.append(_lim(m, "SynthID", "limitation", "Invisible watermarks such as SynthID are not checked."))
    if f != "credentials_valid":
        s.append(_sentence("not_established", dnm, "finding"))
    return headline, s, dnm


def _generic(m: dict, r: dict) -> tuple[str, list, str]:
    """Any completed result without a dedicated template (mock modules, future modules): quotes fields, adds nothing."""
    label = FINDING_LABEL.get(m.get("finding"), "Inconclusive")
    s = []
    if m.get("tested_scope"):
        s.append(_sentence("examined", "Scope: " + _end(m["tested_scope"]), "tested_scope", quoted=True))
    v = (m.get("score") or {}).get("value")
    if m.get("source") == "mock":
        if isinstance(v, (int, float)):
            s.append(_sentence("model_output", f"Scripted MOCK score: {fmt(v)}. It is not a measurement.", "score.value"))
    else:
        s += _score_sentences(m, "model")
    for i, w in enumerate(m.get("warnings") or []):
        if isinstance(w, dict) and w.get("message"):
            s.append(_quote("observation", w["message"], f"warnings[{i}]"))
    dnm = GENERIC_NOT_ESTABLISHED.get(m.get("section"), GENERIC_NOT_ESTABLISHED["authenticity"])
    s.append(_sentence("not_established", dnm, "section"))
    return label, s, dnm


_TEMPLATES = {"face_video": _face_video, "image_detector": _image_detector, "speech": _speech, "lip_sync": _lip_sync,
              "generated_image": _generated_image, "generated_video": _generated_video, "media_metadata": _media_metadata,
              "content_credentials": _content_credentials}


def _key_figure(m: dict) -> Optional[dict]:
    """The one number a card shows, or None. Experimental scores are never key figures (UX-4): they stay in technical details."""
    if m.get("status") != "completed" or m.get("source") == "mock" or is_experimental(m):
        return None
    if m.get("module_id") == "lip_sync":
        off = (_obs(m, "av_offset_frames") or {}).get("value")
        if isinstance(off, int):
            return {"label": "Best audio-visual offset", "text": f"{_signed(off)} frames", "sources": ["observations[code=av_offset_frames].value"]}
        return None
    s = m.get("score") or {}
    v, t = s.get("value"), s.get("threshold")
    if isinstance(v, (int, float)) and isinstance(t, (int, float)):
        return {"label": "Score", "text": f"{fmt_score(v, t)} (decision line {fmt(t)})", "sources": ["score.value", "score.threshold"]}
    return None


def explain_module(m: dict, report: Optional[dict] = None) -> dict:
    report = report or {"modules": [m]}
    mid = str(m.get("module_id"))
    st = m.get("status")
    title, what = MODULES.get(mid, (mid.replace("_", " ").capitalize(), ""))
    mock = m.get("source") == "mock"
    if st != "completed":
        headline, sents, dnm = _not_completed(m)
        key = f"{st}"
    elif mock:
        headline, sents, dnm = _generic(m, report)
        key = f"completed.{m.get('finding')}.mock"
    else:
        headline, sents, dnm = _TEMPLATES.get(mid, _generic)(m, report)
        key = f"completed.{m.get('finding')}" + (".experimental" if is_experimental(m) else "")
    sents = [x for x in sents if x is not None]
    if mock:
        if st == "completed" and is_experimental(m):
            headline = "Experimental research output, no finding is made"
        headline = "MOCK: " + headline
        sents.insert(0, _sentence("mock", "Invented scenario data: no model ran, and nothing here is a measurement.", "source"))
    g = group_of(m)
    return {"template_id": f"{mid if mid in _TEMPLATES and not mock else 'generic'}.{key}.{TEMPLATE_VERSION}",
            "title": title, "what_it_checks": what, "group": g, "group_label": GROUP_LABEL[g],
            "status_label": STATUS_LABEL.get(st, str(st)),
            "finding_label": ("Information only" if g == "informational" else FINDING_LABEL.get(m.get("finding"))) if st == "completed" else None,
            "headline": headline, "sentences": sents, "does_not_mean": dnm, "key_figure": _key_figure(m)}


def _fallback(m: dict, err: Exception) -> dict:
    """Used only when a template raises: the report still renders, says why, and claims nothing."""
    g = group_of(m)
    mid = str(m.get("module_id"))
    return {"template_id": f"fallback.{TEMPLATE_VERSION}", "title": MODULES.get(mid, (mid, ""))[0], "what_it_checks": "",
            "group": g, "group_label": GROUP_LABEL[g], "status_label": STATUS_LABEL.get(m.get("status"), str(m.get("status"))),
            "finding_label": FINDING_LABEL.get(m.get("finding")) if m.get("status") == "completed" else None,
            "headline": "No explanation could be produced for this result",
            "sentences": [_sentence("observation", f"The explanation template failed ({type(err).__name__}); see technical details for the "
                                                   "raw result.", "status"),
                          _sentence("not_established", "Without an explanation, read the raw result in technical details; no conclusion "
                                                       "is drawn here.", "status")],
            "does_not_mean": "", "key_figure": None}


# ---------------------------------------------------------------------------------------------------------------- report summary
def _module(r: dict, mid: str) -> Optional[dict]:
    return next((m for m in r.get("modules", []) if m.get("module_id") == mid), None)


def _faces_fact(r: dict) -> dict:
    """What the checks established about faces, after they ran (the probe is built before any module runs, so it cannot know)."""
    p = r.get("probe") or {}
    if p.get("media_type") == "audio":
        return {"label": "Faces", "value": None, "text": "Not applicable: an audio file has no pictures.", "sources": ["probe.media_type"]}
    fv, im, ls = _module(r, "face_video"), _module(r, "image_detector"), _module(r, "lip_sync")
    if fv and fv.get("source") != "mock":
        if fv.get("status") == "completed" and isinstance(fv.get("units_analysed"), int):
            return {"label": "Faces", "value": True, "text": f"A face was found in {fv['units_analysed']} sampled frames (face manipulation analysis).",
                    "sources": ["@face_video.units_analysed"]}
        if fv.get("status") == "not_applicable" and _warning(fv, "NO_FACE_DETECTED"):
            return {"label": "Faces", "value": False, "text": "No face was found in the sampled frames (face manipulation analysis).",
                    "sources": ["@face_video.warnings[code=NO_FACE_DETECTED]"]}
    if im and im.get("source") != "mock" and im.get("status") == "completed":
        sc = str(im.get("tested_scope", ""))
        if sc.startswith("path=face_crop"):
            return {"label": "Faces", "value": True, "text": "A face was found in the image (image analysis).", "sources": ["@image_detector.tested_scope"]}
        if sc.startswith("path=whole_image"):
            return {"label": "Faces", "value": False, "text": "No face was found in the image (image analysis).", "sources": ["@image_detector.tested_scope"]}
    o = _obs(ls, "frames_with_face") if ls and ls.get("source") != "mock" else None
    if o:
        return {"label": "Faces", "value": True, "text": f"Frames with a face in the synchronisation check: {o['value']}.",
                "sources": ["@lip_sync.observations[code=frames_with_face].value"]}
    if isinstance(p.get("n_faces"), int):
        return {"label": "Faces", "value": p["n_faces"] > 0, "text": f"Faces stated in the probe: {p['n_faces']}.", "sources": ["probe.n_faces"]}
    return {"label": "Faces", "value": None, "text": "Not established: no check that looks for faces completed on this file.", "sources": []}


def _speech_fact(r: dict) -> dict:
    p = r.get("probe") or {}
    if p.get("media_type") == "image":
        return {"label": "Speech", "value": None, "text": "Not applicable: a still image has no audio.", "sources": ["probe.media_type"]}
    if p.get("has_audio") is False:
        return {"label": "Speech", "value": None, "text": "Not applicable: the file has no audio stream.", "sources": ["probe.has_audio"]}
    ls = _module(r, "lip_sync")
    if ls and ls.get("source") != "mock":
        o = _obs(ls, "speech_like_screen")
        if o is not None:
            return {"label": "Speech", "value": bool(o["value"]),
                    "text": ("Speech-like" if o["value"] else "Not speech-like") + " by a simple heuristic screen in the synchronisation check "
                            "(not a speech detector).", "sources": ["@lip_sync.observations[code=speech_like_screen].value"]}
        if _warning(ls, "AUDIO_NOT_SPEECH_LIKE"):
            return {"label": "Speech", "value": False, "text": "Not speech-like by a simple heuristic screen in the synchronisation check "
                    "(not a speech detector).", "sources": ["@lip_sync.warnings[code=AUDIO_NOT_SPEECH_LIKE]"]}
    if isinstance(p.get("speech_detected"), bool):
        return {"label": "Speech", "value": p["speech_detected"], "text": f"Speech stated in the probe: {'yes' if p['speech_detected'] else 'no'}.",
                "sources": ["probe.speech_detected"]}
    return {"label": "Speech", "value": None, "text": "Not established: no check that screens for speech ran on this file. An audio stream "
            "is present, but that does not mean it contains speech.", "sources": ["probe.has_audio"]}


def _disagreements(r: dict) -> list:
    out = [{"modules": list(c.get("modules", [])), "text": c.get("note", "")} for c in r.get("conflicts") or []]
    done = [m for m in r.get("modules", []) if m.get("status") == "completed" and not is_experimental(m)]
    flagged = [m["module_id"] for m in done if m.get("finding") in ("potential_mismatch", "credentials_invalid")]
    clear = [m["module_id"] for m in done if m.get("section") == "authenticity" and m.get("finding") == "no_indication_within_tested_scope"]
    if flagged and clear:
        out.append({"modules": flagged + clear, "text": "Some checks raised a point that merits attention while detector checks found "
                    "nothing within their tested scope. These checks look at different things and can disagree; neither cancels the other."})
    return out


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def summarise(r: dict) -> dict:
    mods = r.get("modules", [])
    groups = {g: [] for g, _ in GROUPS}
    for m in mods:
        groups[group_of(m)].append(m.get("module_id"))
    c = {g: len(v) for g, v in groups.items()}
    p = r.get("probe") or {}
    kind = {"video": "video", "image": "image", "audio": "audio file"}.get(p.get("media_type"), "file")
    parts = []
    if c["attention"]:
        parts.append(_plural(c["attention"], "check found a point that merits attention", "checks found points that merit attention"))
    if c["nothing_found"]:
        parts.append(_plural(c["nothing_found"], "found nothing within its tested scope", "found nothing within their tested scope"))
    if c["credential"]:
        parts.append(_plural(c["credential"], "verified a content credential", "verified content credentials"))
    if c["inconclusive"]:
        parts.append(_plural(c["inconclusive"], "was inconclusive", "were inconclusive"))
    if c["informational"]:
        parts.append(_plural(c["informational"], "described the file", "described the file"))
    if c["experimental"]:
        parts.append(_plural(c["experimental"], "is experimental and makes no finding", "are experimental and make no finding"))
    ran = len(mods) - c["not_applicable"] - c["unsupported"] - c["did_not_run"]
    head = f"{_plural(ran, 'check', 'checks')} ran on this {kind}."
    if parts:
        head += " " + _cap("; ".join(parts)) + "."
    tail = []
    if c["not_applicable"]:
        tail.append(_plural(c["not_applicable"], "did not apply", "did not apply"))
    if c["unsupported"]:
        tail.append(_plural(c["unsupported"], "was not available", "were not available"))
    if c["did_not_run"]:
        tail.append(_plural(c["did_not_run"], "did not complete", "did not complete"))
    if tail:
        head += " " + _cap(", ".join(tail)) + "."
    return {"headline": head, "counts": c,
            "groups": [{"id": g, "label": GROUP_LABEL[g], "modules": groups[g]} for g, _ in GROUPS if groups[g]],
            "media_facts": [_faces_fact(r), _speech_fact(r)], "disagreements": _disagreements(r),
            "no_overall_score": "No overall score or verdict is given: each check reports only on what it examines."}


def attach(report: dict) -> dict:
    """Adds ``explanation`` to every module and ``summary`` to the report, and marks it schema 2.1. Never raises for a template bug."""
    for m in report.get("modules", []):
        m.pop("assessment", None)
        m["assessment"] = verdict.assess_module(m)      # D-087: categorical verdict with its evidence; never a probability
        try:
            m["explanation"] = explain_module(m, report)
        except Exception as e:  # noqa: BLE001 - an explanation bug must not cost the user the report
            m["explanation"] = _fallback(m, e)
    report["summary"] = summarise(report)
    report["schema_version"] = schema.SCHEMA_VERSION
    return report


# ---------------------------------------------------------------------------------------------------------------- checks
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers_in(value: Any) -> set:
    """Number tokens a cited value licenses: tokens of its text, and 0 to 4 decimal renderings of its numeric parts."""
    out: set = set()
    if value is None:
        return out
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out.add(str(abs(int(value))) if float(value).is_integer() else "")
            for d in range(0, 5):
                out.add(f"{abs(value):.{d}f}")
        return out - {""}
    if isinstance(value, str):
        return set(_NUM.findall(value))
    if isinstance(value, dict):
        for v in value.values():
            out |= _numbers_in(v)
        return out
    if isinstance(value, list):
        for v in value:
            out |= _numbers_in(v)
        return out
    return set(_NUM.findall(json.dumps(value, default=str)))


def check_explanation(m: dict, report: Optional[dict] = None) -> list[str]:
    """Problems with ``m['explanation']``: structure, status integrity, cited fields, numbers, vocabulary, required caveats."""
    mid = m.get("module_id", "?")
    e = m.get("explanation")
    if not isinstance(e, dict):
        return [f"{mid}: explanation missing"]
    bad = []
    for k in ("template_id", "title", "group", "headline", "sentences", "status_label"):
        if not e.get(k) and k != "sentences":
            bad.append(f"{mid}: explanation.{k} missing")
    if e.get("group") != group_of(m):
        bad.append(f"{mid}: explanation group {e.get('group')!r} does not match status/finding ({group_of(m)!r})")
    sents = e.get("sentences")
    if not isinstance(sents, list):
        return bad + [f"{mid}: explanation.sentences must be a list"]
    texts = [str(e.get("headline", "")), str(e.get("does_not_mean", ""))]
    kinds = []
    for i, s in enumerate(sents):
        if not isinstance(s, dict) or s.get("kind") not in KINDS or not s.get("text") or not s.get("sources"):
            bad.append(f"{mid}: sentence {i} needs kind in {KINDS}, text and sources")
            continue
        kinds.append(s["kind"])
        if s.get("quoted"):        # a field's own words: checked to be a real quote, exempt from the template vocabulary rule
            src_text = resolve(m, s["sources"][0], report)
            src_text = (src_text.get("message") if isinstance(src_text, dict) else src_text)
            if not isinstance(src_text, str) or src_text.strip().rstrip(".").lower() not in s["text"].lower():
                bad.append(f"{mid}: sentence {i} is marked quoted but does not contain the text of {s['sources'][0]!r}")
        else:
            texts.append(s["text"])
        allowed = set(NUMBER_CONSTANTS)
        for src in s["sources"]:
            v = resolve(m, src, report)
            if v is None:
                bad.append(f"{mid}: sentence {i} cites {src!r}, which is missing")
            allowed |= _numbers_in(v)
        for tok in _NUM.findall(s["text"]):
            if tok not in allowed:
                bad.append(f"{mid}: sentence {i} shows the number {tok} that none of its sources {s['sources']} contains")
    kf = e.get("key_figure")
    if kf is not None:
        allowed = set(NUMBER_CONSTANTS)
        for src in kf.get("sources", []):
            allowed |= _numbers_in(resolve(m, src, report))
        for tok in _NUM.findall(str(kf.get("text", ""))):
            if tok not in allowed:
                bad.append(f"{mid}: key figure shows {tok}, not in its sources")
    blob = " ".join(texts).lower()
    for pat in FORBIDDEN:
        if re.search(pat, blob):
            bad.append(f"{mid}: explanation uses forbidden wording {pat!r}")
    if m.get("status") != "completed":
        if RESULT_KINDS & set(kinds) or kf is not None:
            bad.append(f"{mid}: a module that did not complete cannot have processing, model output, decision, interpretation or a "
                       "key figure")
    elif m.get("limitations") and not ({"limitation", "not_established"} & set(kinds)):
        bad.append(f"{mid}: a completed result needs at least one limitation or not-established sentence")
    if is_experimental(m) and m.get("status") == "completed":
        if "experimental" not in str(e.get("headline", "")).lower():
            bad.append(f"{mid}: an experimental result must say so in its headline")
        if kf is not None:
            bad.append(f"{mid}: an experimental score must not be a key figure")
    if m.get("source") == "mock" and ("mock" not in kinds or not str(e.get("headline", "")).startswith("MOCK")):
        bad.append(f"{mid}: a mock result must be marked MOCK in its explanation")
    return bad


def check_report_explanations(r: dict) -> list[str]:
    """Checks every explanation and the summary that are present (both optional in schema 2.1, D3)."""
    bad = []
    for m in r.get("modules", []):
        if isinstance(m, dict) and "explanation" in m:
            bad += check_explanation(m, r)
    stripped_mods = [{k: v for k, v in m.items() if k not in ("explanation", "assessment")} for m in r.get("modules", [])]
    for m, base in zip(r.get("modules", []), stripped_mods):
        if isinstance(m, dict) and "assessment" in m and m["assessment"] != verdict.assess_module(base):
            bad.append(f"{m.get('module_id')}: assessment does not match the module result")
        if isinstance(m, dict) and "assessment" in m:
            blob_m = json.dumps(m["assessment"]).lower()
            for pat in FORBIDDEN:
                if re.search(pat, blob_m):
                    bad.append(f"{m.get('module_id')}: assessment uses forbidden wording {pat!r}")
    if "assessment" in r:
        bad.append("report must not carry an assessment: verdicts are module-specific (D-087)")
    if "summary" not in r:
        pass
    elif not isinstance(r["summary"], dict):
        bad.append("summary must be an object")
    else:
        stripped = {**r, "modules": [{k: v for k, v in m.items() if k != "explanation"} for m in r.get("modules", [])]}
        if r["summary"] != summarise(stripped):
            bad.append("summary does not match the module results")
        blob = json.dumps(r["summary"]).lower()
        for pat in FORBIDDEN:
            if re.search(pat, blob):
                bad.append(f"summary uses forbidden wording {pat!r}")
    return bad


def vocabulary() -> dict:
    """Labels the dashboard shows, from one place (D2: no second copy in the browser)."""
    return {"modules": {k: {"title": t, "what_it_checks": w, **({"applies_to": list(APPLIES_TO[k][0]), "needs": APPLIES_TO[k][1]}
                                                                  if k in APPLIES_TO else {})} for k, (t, w) in MODULES.items()},
            "findings": FINDING_LABEL, "statuses": STATUS_LABEL, "groups": [{"id": g, "label": lab} for g, lab in GROUPS],
            "module_order": list(MODULES),
            "explanation_parts": [{"id": i, "label": lab, "kinds": list(k)} for i, lab, k in EXPLANATION_PARTS],
            "schema_versions": list(schema.ACCEPTED_VERSIONS), "statement": schema.STATEMENT, "assessment": verdict.vocabulary()}
