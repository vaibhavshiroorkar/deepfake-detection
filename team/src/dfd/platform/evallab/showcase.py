"""Recommended demonstrations: a curated, read-only view of evaluation results that are already logged.

Every number is read at request time from a logged result file (the E-014 sealed report data/_final/eval/I-4.json, or a
post-freeze results.json of E-033 and E-034) and returned with the file's SHA-256. Nothing here runs a model, reads a
score file, re-analyses a sealed split, writes a file or changes a threshold. The text of each entry (why it is
recommended, its caveats) is fixed here; the numbers are never written here.

Sealed blocks of the E-014 report hold the mean over three seeds of AUROC, EER, balanced accuracy, precision and recall
with a group-bootstrap interval, each at the per-seed validation threshold (D-024). Specificity, false-positive rate,
F1 and accuracy at that threshold are not in the report; they are derived per seed from the logged per-seed values
(specificity = 2 * balanced accuracy - recall, F1 = 2PR / (P + R), accuracy = (recall * n_fake + specificity * n_real) / n),
averaged over seeds, and marked as derived, without an interval.
"""

from __future__ import annotations

import json
from pathlib import Path

from dfd.eval.scorefile import file_sha256

from .common import REPO_ROOT

SEALED_REPORT = "data/_final/eval/I-4.json"
E033_RESULTS = "data/_results/posteval/E-033/results.json"
E034_RESULTS = "data/_results/posteval/E-034/results.json"

KIND_TEXT = {
    "stored_sealed": "Stored evaluation result: logged once at the freeze (E-014) on a sealed test split and displayed "
                     "here as logged. Nothing is recomputed and the sealed split is not reopened.",
    "stored_post_freeze": "Stored evaluation result: logged post-freeze evaluation on previously unread data. The Lab can "
                          "also re-analyse its stored scores (metrics recomputed now; the model is not run again).",
}
STATUS = ("Experimental research results: a frozen checkpoint measured on stated data. They are not a product approval "
          "and no verdict on the Analyse page depends on them.")

_SEALED_COMMON = ["Mean over three training seeds of the same recipe; the product checkpoint ships seed 0 (D-050), "
                  "whose AUROC is shown separately.",
                  "Threshold-based metrics use each seed's threshold of maximum balanced accuracy on the validation split "
                  "(D-024), not a threshold chosen on this test data."]

ENTRIES = [
    # ---------------------------------------------------------------- recommended
    {"id": "speech-itw", "section": "recommended", "modality": "speech", "task": "T-A",
     "title": "In-the-Wild speech deepfakes (42 sealed speakers)", "setting": "Cross-corpus: trained on ASVspoof 2019 LA only",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "audio", "dataset": "in_the_wild", "split": "test_x"},
     "lab": {"dataset": "in_the_wild", "group": "G4"},
     "why": "Strongest evidence that the speech detector carries over to recordings it was not trained on: real-world "
            "celebrity and politician speech, speaker-disjoint from everything used for fitting.",
     "caveats": ["12 other In-the-Wild speakers (dev slice) were used to rank design candidates, so the corpus is not "
                 "entirely new; the 42 test speakers were sealed and read once at the freeze.",
                 "42 of the 54 speakers of the published set, so not directly comparable with published numbers.",
                 "About one third of the clips are fake, so accuracy depends on the class balance; read AUROC and balanced accuracy."]},
    {"id": "video-celebdf", "section": "recommended", "modality": "face_video", "task": "T-V",
     "title": "Celeb-DF v2 face-swap videos", "setting": "Cross-dataset: trained on FaceForensics++ only",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "visual", "dataset": "celebdf_v2", "split": "test_x"},
     "lab": {"dataset": "celebdf_v2", "group": "G4"},
     "why": "A different dataset with higher-quality face swaps than the training data: the standard cross-dataset test for "
            "face-video detectors, with 344 independent identity groups.",
     "caveats": ["15 other Celeb-DF identities (dev slice) were used to rank design candidates; the test identities were sealed.",
                 "Our identity-disjoint split, not the official 518-video list, so published numbers are not on the same protocol.",
                 "Only about 15 percent of the videos are real; accuracy is dominated by the fake class."]},
    {"id": "image-cnn", "section": "recommended", "modality": "image", "task": "T-I",
     "title": "ForenSynths CNN-generated images (6 unseen generators)", "setting": "Cross-generator: trained on ProGAN only",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "image", "dataset": "forensynths_cnn_subset", "split": "test_x"},
     "lab": {"dataset": "forensynths_cnn_subset", "group": "G4"},
     "why": "Images from six generators the image detector never saw (BigGAN, CycleGAN, GauGAN, StyleGAN, StyleGAN2, "
            "Deepfake), with balanced classes.",
     "caveats": ["Older GAN generators: on modern diffusion and commercial generators the same detector misses most fakes "
                 "(Community Forensics, E-028 and E-034).",
                 "Only 16 source groups, so the interval is wide.",
                 "Whole-image path; the product routes still images with a face to the face path."]},
    {"id": "video-ffpp", "section": "recommended", "modality": "face_video", "task": "T-V",
     "title": "FaceForensics++ c23 official test", "setting": "In-dataset: same dataset as training, unseen videos",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "visual", "dataset": "ffpp", "split": "test_in"},
     "why": "Shows what the face-video detector does on the kind of manipulation it was trained on (four face-swap and "
            "reenactment methods, c23 compression).",
     "caveats": ["In-dataset: an upper reference, not evidence of generalisation.",
                 "700 clips in 70 source groups."]},
    {"id": "speech-asv", "section": "recommended", "modality": "speech", "task": "T-A",
     "title": "ASVspoof 2019 LA evaluation set", "setting": "In-corpus: official evaluation set, mostly unseen attacks",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "audio", "dataset": "asvspoof2019_la", "split": "test_in"},
     "why": "The standard benchmark for synthetic speech, 67 speakers and 13 spoofing attacks, scored by the shipped speech "
            "system (three speech encoders, mean logit).",
     "caveats": ["In-corpus: same recording conditions as training.",
                 "About 90 percent of the clips are spoofed, so accuracy and precision are dominated by that class.",
                 "Only the first 64,600 samples (about 4 s) of each clip are scored."]},
    {"id": "lavdf-ta-g4", "section": "recommended", "modality": "speech", "task": "T-A",
     "title": "LAV-DF unread clips, speech head: G4 Very large sample",
     "setting": "Post-freeze evaluation on previously unread data (E-033)",
     "source": {"kind": "stored_post_freeze", "file": E033_RESULTS, "dataset": "lavdf_unread_ta", "group": "G4"},
     "lab": {"dataset": "lavdf_unread_ta", "group": "G4"},
     "why": "The LAV-DF option where the product detects something: the speech head against word-level audio edits, on "
            "3,000 source videos never read before, with a balanced class mix. It can be re-analysed live in the Lab.",
     "caveats": ["Moderate result: the edits are short and the head hears only about 4 s; the diagnostic group G5 "
                 "(edit inside the scored window) shows the effect of that window.",
                 "LAV-DF behaviour was already seen on its dev split and a sealed subset: a larger replication, not an "
                 "independent test.",
                 "Identities are not in the metadata; the bootstrap resamples source videos, so intervals are optimistic."]},
    # ---------------------------------------------------------------- other stored results (full list)
    {"id": "video-dfd", "section": "other", "modality": "face_video", "task": "T-V",
     "title": "DeepFakeDetection (DFD) c23", "setting": "Cross-dataset: trained on FaceForensics++ only",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "visual", "dataset": "dfd", "split": "test_x"},
     "why": "A second cross-dataset face-video test (paid actors, 28 actor groups).",
     "caveats": ["Only 28 actor groups: wide interval.", "About 11 percent real; no official split exists."]},
    {"id": "video-celebdf-family", "section": "other", "modality": "face_video", "task": "T-V",
     "title": "Celeb-DF family (Celeb-DF v2 plus Celeb-DF++ fakes)", "setting": "Cross-dataset: the headline face-video test of the paper",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "visual", "dataset": "celebdf_family", "split": "test_x"},
     "why": "The headline cross-dataset block of the paper; adds the newer Celeb-DF++ manipulations, which are harder.",
     "caveats": ["9.7 percent real: accuracy is not informative at this balance.",
                 "Celeb-DF++ fakes are paired with Celeb-DF v2 reals (our pairing, D-045)."]},
    {"id": "image-diffusion", "section": "other", "modality": "image", "task": "T-I",
     "title": "UnivFD diffusion subset (8 generator folders)", "setting": "Cross-generator: trained on ProGAN only",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "image", "dataset": "diffusion_test", "split": "test_x"},
     "why": "Diffusion and autoregressive generators (LDM, Glide, Guided diffusion, DALL-E) unseen in training.",
     "caveats": ["Only 10 groups: wide interval.", "1,000 images per folder against shared LAION and ImageNet reals."]},
    {"id": "image-progan", "section": "other", "modality": "image", "task": "T-I",
     "title": "ForenSynths ProGAN test", "setting": "In-distribution: same generator as training",
     "source": {"kind": "stored_sealed", "file": SEALED_REPORT, "modality": "image", "dataset": "forensynths_progan_test", "split": "test_in"},
     "why": "In-distribution reference for the image detector.",
     "caveats": ["Same generator as training: near-perfect separation is expected and says nothing about other generators."]},
    {"id": "lavdf-ta-g5", "section": "other", "modality": "speech", "task": "T-A",
     "title": "LAV-DF unread clips, speech head: G5 diagnostic (edit inside the scored window)",
     "setting": "Post-freeze evaluation on previously unread data (E-033), diagnostic group",
     "source": {"kind": "stored_post_freeze", "file": E033_RESULTS, "dataset": "lavdf_unread_ta", "group": "G5"},
     "lab": {"dataset": "lavdf_unread_ta", "group": "G5"},
     "why": "Explains part of the speech head's errors on LAV-DF: it is better when the edit lies inside the roughly 4 s it hears.",
     "caveats": ["Diagnostic subset of G3 selected by edit position; not a general sample of LAV-DF.",
                 "The real originals are 62 percent of this group."]},
    {"id": "cf-g4", "section": "other", "modality": "image", "task": "T-I",
     "title": "Community Forensics-Eval unread shards: G4 Very large sample",
     "setting": "Post-freeze evaluation on previously unread data (E-034)",
     "source": {"kind": "stored_post_freeze", "file": E034_RESULTS, "dataset": "cfeval_unread_ti", "group": "G4"},
     "lab": {"dataset": "cfeval_unread_ti", "group": "G4"},
     "why": "Modern generators (12 of the set): the ranking is above chance, but at the frozen threshold most generated images are missed.",
     "caveats": ["Low recall at the frozen threshold, so accuracy is low; the threshold was fitted on ProGAN and is not retuned.",
                 "Reals and fakes differ in file format and size; see the PNG-only limitation below."]},
    # ---------------------------------------------------------------- limitations to show honestly
    {"id": "lavdf-tv-g4", "section": "limitation", "modality": "face_video", "task": "T-V",
     "title": "LAV-DF unread clips, face-video head: G4 Very large sample",
     "setting": "Post-freeze evaluation on previously unread data (E-033)",
     "source": {"kind": "stored_post_freeze", "file": E033_RESULTS, "dataset": "lavdf_unread_tv", "group": "G4"},
     "lab": {"dataset": "lavdf_unread_tv", "group": "G4"},
     "why": "Known failure: the face-video head was trained on face swaps and reenactment, not on LAV-DF's short lip edits, "
            "and ranks edited clips below untouched ones.",
     "caveats": ["Within produced clips only (diagnostic G5) it is at chance: the below-chance ranking comes from scoring the "
                 "untouched originals higher, not from a response to the edit (E-033 reading)."]},
    {"id": "cf-g5", "section": "limitation", "modality": "image", "task": "T-I",
     "title": "Community Forensics-Eval unread shards: G5 PNG only, format-matched",
     "setting": "Post-freeze evaluation on previously unread data (E-034), diagnostic group",
     "source": {"kind": "stored_post_freeze", "file": E034_RESULTS, "dataset": "cfeval_unread_ti", "group": "G5"},
     "lab": {"dataset": "cfeval_unread_ti", "group": "G5"},
     "why": "When file format cannot help, separation on modern generators drops close to chance.",
     "caveats": ["An observed association with file format; the PNG-only group also differs in content and real sources, "
                 "so it does not prove that the head uses format."]},
]

NOT_SHOWN = [
    {"title": "LAV-DF with in-domain heads and late fusion (I-6)",
     "reason": "Very high numbers that come from a production cue of the dataset pipeline (E-011, E-012), and fusion is not "
               "a product feature (D-059). Not a demonstration of deepfake detection."},
    {"title": "FakeAVCeleb sealed identity half (E-015)",
     "reason": "Visual AUROC is high but explained largely by a codec difference between real and fake videos (E-021). "
               "Its scores are not a Lab score file."},
]

MODALITY_TEXT = {"face_video": "Face video", "speech": "Speech / audio", "image": "Image", "audio_visual": "Audio-visual research"}


class SourceError(Exception):
    pass


def _ci(m: dict | None) -> dict | None:
    if not isinstance(m, dict) or m.get("estimate") is None:
        return None
    return {"estimate": m["estimate"], "low": m.get("low"), "high": m.get("high")}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _sealed(doc: dict, src: dict) -> dict:
    cand = next((c for c in doc.get("candidates", []) if c.get("modality") == src["modality"]), None)
    if cand is None:
        raise SourceError(f"no {src['modality']} candidate in the report")
    blk = next((b for b in cand.get("blocks", []) if b.get("dataset") == src["dataset"] and b.get("split") == src["split"]), None)
    if blk is None:
        raise SourceError(f"no block {src['dataset']} {src['split']} in the report")
    m = blk["metrics"]
    nf, nr = int(blk["n_fake"]), int(blk["n_real"])
    metrics = {k: _ci(m.get(k)) for k in ("auc", "eer", "balanced_accuracy", "precision", "recall")}
    ps = {k: (m.get(k) or {}).get("per_seed") or [] for k in ("balanced_accuracy", "precision", "recall", "auc")}
    derived = {}
    if ps["balanced_accuracy"] and len(ps["balanced_accuracy"]) == len(ps["recall"]) == len(ps["precision"]):
        spec = [2 * b - r for b, r in zip(ps["balanced_accuracy"], ps["recall"])]
        f1 = [(2 * p * r / (p + r)) if (p + r) > 0 else None for p, r in zip(ps["precision"], ps["recall"])]
        acc = [(r * nf + s * nr) / (nf + nr) for r, s in zip(ps["recall"], spec)]
        for key, vals in (("specificity", spec), ("fpr", [1 - s for s in spec]), ("f1", f1), ("accuracy", acc)):
            v = _mean(vals)
            if v is not None:
                derived[key] = {"estimate": v, "low": None, "high": None, "derived": True}
    metrics.update(derived)
    fits = cand.get("fits") or {}
    thresholds = {rid: f.get("threshold") for rid, f in sorted(fits.items())}
    seeds = [f.get("seed") for _, f in sorted(fits.items())]
    seed0 = ps["auc"][seeds.index(0)] if 0 in seeds and len(ps["auc"]) == len(seeds) else None
    return {"n_real": nr, "n_fake": nf, "n_samples": int(blk["n_samples"]), "n_groups": blk.get("n_groups"),
            "metrics": metrics, "seed0_auc": seed0, "n_seeds": len(seeds),
            "threshold": {"values": thresholds,
                          "provenance": "per-seed threshold of maximum balanced accuracy on the validation split (D-024), "
                                        "fitted before the sealed split was read; " + str((doc.get("protocol") or {}).get("threshold_rule", ""))},
            "interval": "95 percent group bootstrap, " + str((doc.get("protocol") or {}).get("n_boot", "")) + " resamples",
            "block": f"{src['dataset']} {src['split']}", "experiment": "E-014 (sealed evaluation at the freeze)",
            "derived_note": "Specificity, false-positive rate, F1 and accuracy are derived per seed from the logged per-seed "
                            "precision, recall and balanced accuracy at the same threshold, then averaged; no interval is logged for them."}


def _post_freeze(doc: dict, src: dict) -> dict:
    ds = (doc.get("datasets") or {}).get(src["dataset"])
    g = ((ds or {}).get("groups") or {}).get(src["group"])
    if g is None:
        raise SourceError(f"no group {src['group']} of {src['dataset']} in the results file")
    d = g.get("data") or {}
    keys = ("auc", "eer", "balanced_accuracy", "precision", "recall", "specificity", "fpr", "fnr", "f1", "accuracy")
    return {"n_real": d.get("n_real"), "n_fake": d.get("n_fake"), "n_samples": d.get("n_samples"), "n_groups": d.get("n_groups"),
            "metrics": {k: _ci((g.get("metrics") or {}).get(k)) for k in keys}, "confusion": g.get("confusion"),
            "lab_run_id": g.get("lab_run_id"), "seed0_auc": None, "n_seeds": 1,
            "threshold": {"values": None, "provenance": "fixed threshold of the frozen product checkpoint (D-024), read from its "
                                                         "metadata; not tuned on this data"},
            "interval": "95 percent cluster bootstrap, " + str(doc.get("n_boot", "")) + " resamples",
            "block": f"{src['dataset']} {src['group']}", "experiment": str(doc.get("experiment", ""))}


class Showcase:
    def __init__(self, root: Path = REPO_ROOT, entries=None):
        self.root = Path(root)
        self.entries = ENTRIES if entries is None else entries

    def _read(self, rel: str, cache: dict):
        if rel not in cache:
            p = self.root / rel
            try:
                cache[rel] = (json.loads(p.read_text(encoding="utf-8")), file_sha256(p))
            except (OSError, ValueError) as e:
                cache[rel] = SourceError(f"{rel} could not be read: {type(e).__name__}")
        if isinstance(cache[rel], Exception):
            raise cache[rel]
        return cache[rel]

    def listing(self) -> dict:
        cache, out = {}, []
        for e in self.entries:
            src = e["source"]
            item = {k: e.get(k) for k in ("id", "section", "modality", "task", "title", "setting", "why", "caveats", "lab")}
            item.update({"modality_label": MODALITY_TEXT.get(e["modality"], e["modality"]), "kind": src["kind"],
                         "kind_text": KIND_TEXT[src["kind"]], "source_file": src["file"], "primary_metric": "auc"})
            try:
                doc, sha = self._read(src["file"], cache)
                item.update(_sealed(doc, src) if src["kind"] == "stored_sealed" else _post_freeze(doc, src))
                item.update({"available": True, "source_sha256": sha})
                if src["kind"] == "stored_sealed":
                    item["caveats"] = list(e.get("caveats") or []) + _SEALED_COMMON
            except (SourceError, KeyError, TypeError, ValueError) as ex:
                item.update({"available": False, "reason": str(ex)})
            out.append(item)
        return {"entries": out, "not_shown": NOT_SHOWN, "status": STATUS, "modalities": MODALITY_TEXT,
                "note": "Stored results are displayed as logged. The Evaluation Lab never runs a detector model: a Lab "
                        "evaluation recomputes metrics from stored scores."}
