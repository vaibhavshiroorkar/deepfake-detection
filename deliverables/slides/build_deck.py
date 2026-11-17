"""Build the final-year presentation from results/numbers.json on the college slide master.

    .venv/Scripts/python.exe deliverables/slides/build_deck.py            final build: stops on a missing number
    .venv/Scripts/python.exe deliverables/slides/build_deck.py --draft    marks missing numbers [pending]

Every number on a slide is a {{key}} placeholder, filled the same way as the paper and the report
(scripts/fill_numbers.py). The deck follows the paper section by section: 14 main slides, a divider,
then backup slides. Output: deliverables/build/Deepfake_Detection_Presentation.pptx.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import slide_common as C  # noqa: E402
from deck_kit import (  # noqa: E402
    F_BLUE,
    F_GREEN,
    F_GREY,
    F_ORANGE,
    F_RED,
    H0,
    Y0,
    Deck,
    set_spacing,
    style_run,
)
from fill_numbers import PATTERN, render  # noqa: E402
from lxml import etree  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu  # noqa: E402
from slide_common import BOT, GREY, L, W, col_x, span_w  # noqa: E402

TEMPLATE = HERE / "college-template.pptx"
FIG = ROOT / "deliverables" / "figures"
BRAND = ROOT / "assets" / "branding"
OUT = ROOT / "deliverables" / "build" / "Deepfake_Detection_Presentation.pptx"

TITLE = "Shortcut-Neutral Multimodal Audio-Visual Deepfake Detection"
MEMBERS = [("Vaibhav Shiroorkar", "16014024805"), ("Adithya Jithesh", "16014024803"), ("Shivansh Gupta", "16014023052")]
GUIDE = "Dr. Tejashree Phatak"
GUIDE_ROLE = "Assistant Professor"


class Numbers:
    """{{key}} -> value from results/numbers.json. Draft mode prints [pending] for a missing key."""

    def __init__(self, draft: bool):
        self.values = json.loads((ROOT / "results" / "numbers.json").read_text(encoding="utf-8"))
        self.draft = draft
        self.missing: set[str] = set()

    def __call__(self, text: str) -> str:
        def one(m):
            key = m.group(1)
            if key in self.values:
                return render(self.values[key]["value"], m.group(2))
            self.missing.add(key)
            return "[pending]"
        return PATTERN.sub(one, text)

    def has(self, *keys: str) -> bool:
        return all(k in self.values for k in keys)


def build(draft: bool) -> int:
    n = Numbers(draft)
    d = Deck(str(TEMPLATE))

    def T(paras):
        """Fill numbers in a list of (kind, text[, size]) paragraphs."""
        return [(p[0], n(p[1]), *p[2:]) for p in paras]

    def card(s, x, y, w, h, paras, **kw):
        return d.add_card(s, x, y, w, h, T(paras), **kw)

    def slide(title, notes, lead=None):
        s = d.new_slide(title, n(notes))
        if lead:
            d.lead(s, n(lead))
        return s

    def figure_or_pending(s, name, x, y, w, h, alt, keys=()):
        path = FIG / name
        if path.exists() and n.has(*keys):
            return d.pic_fit(s, path, x, y, w, h, alt)
        card(s, x, y, w, h, [("h", "Figure"), ("p", "[pending] This figure is drawn when the run that feeds it "
                                                  "finishes.", 16)], name="card_pending")
        return None

    # 1. title ----------------------------------------------------------------------------------------
    d.count += 1
    s = d.prs.slides.add_slide(d.layout["TITLE"])
    d.set_title(s.shapes.title, TITLE, size=24, y=40000, h=820000, check=False)
    s.shapes.title.left, s.shapes.title.width = Emu(2600000), Emu(C.SLIDE_W - 2 * 2600000)
    sub = [p for p in s.placeholders if p.placeholder_format.idx == 1][0]
    sub.left, sub.top, sub.width, sub.height = Emu(L), Emu(880000), Emu(W), Emu(640000)
    tf = sub.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    bp = tf._txBody.find(qn("a:bodyPr"))
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, qn("a:noAutofit"))
    for i, txt in enumerate(["Department of Electronics and Computer Engineering",
                             "K. J. Somaiya School of Engineering, Somaiya Vidyavihar University"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = txt
        style_run(r, 18, False, False, GREY)
        set_spacing(p, 0, 2)
    ph_y, ph_h = 1600000, 2250000
    half = (W - 120000) // 2
    for k, (fn, alt) in enumerate([("campus_aryabhat_building.jpg", "Aryabhat building, K. J. Somaiya campus"),
                                   ("campus_bhaskaracharya_building.jpg", "Bhaskaracharya building, K. J. Somaiya campus")]):
        pic = d.add_pic(s, BRAND / fn, L + k * (half + 120000), ph_y, alt=alt)
        ar_box, ar_img = half / ph_h, pic.image.size[0] / pic.image.size[1]
        pic.width, pic.height = Emu(half), Emu(ph_h)
        pic.left, pic.top = Emu(L + k * (half + 120000)), Emu(ph_y)
        if ar_img < ar_box:
            pic.crop_top = pic.crop_bottom = (1 - ar_img / ar_box) / 2
        else:
            pic.crop_left = pic.crop_right = (1 - ar_box / ar_img) / 2
    d.add_text(s, L, 3930000, W, 400000, [("h", "Group members")], "members_label")
    for k, (nm, roll) in enumerate(MEMBERS):
        d.add_text(s, col_x(4 * k), 4340000, span_w(4), 760000, [("p", f"**{nm}**"), ("s", f"Roll No: {roll}")], "member")
    d.add_text(s, L, 5150000, W, 500000, [("p", f"**Guide:** {GUIDE}, {GUIDE_ROLE}")], "guide")
    d.add_footer(s, 1)
    s.notes_slide.notes_text_frame.text = (
        "This project asks why audio-visual deepfake detectors that score near perfect at home fail on new data. "
        "We show that the benchmark files themselves give the label away, build a protocol that removes those file "
        "cues, and test a lip-sync detector trained on real clips only under that protocol. "
        "Introduce the three members and the guide, then move to the problem.")

    # 2. problem and objectives -------------------------------------------------------------------------
    s = slide("Problem and objectives",
              "Detectors report near perfect AUC on the corpus they trained on and drop toward chance elsewhere. "
              "The usual reading is overfitting to one generator. We test a simpler reading: every fake passes through "
              "a production pipeline, and that pipeline leaves marks in the file, such as a codec, a length or digital "
              "silence. A detector can learn those marks instead of the manipulation, and its benchmark score cannot "
              "tell us which it learned. The four objectives were written down as hypotheses H1 to H4 in a "
              "preregistration before any model was trained.",
              "Detectors score near perfect at home and fail elsewhere. Do the files give the label away?")
    for k, (hd, tx, fl) in enumerate([("Visual", "swapped or re-enacted faces", F_BLUE),
                                      ("Audio", "cloned or synthetic speech", F_ORANGE),
                                      ("Lip sync", "lips that do not match the sound", F_GREEN)]):
        d.dbox(s, col_x(4 * k), Y0, span_w(4), 650000, f"**{hd}**\n{tx}", fill=fl, size=16)
    card(s, col_x(0), Y0 + 800000, span_w(7), H0 - 800000, [
        ("h", "Objectives, preregistered as H1 to H4"),
        ("n", "1.  Measure how much of each benchmark can be read from file cues alone.", 16),
        ("n", "2.  Remove those cues, and check the removal with the same audit (H1).", 16),
        ("n", "3.  Detect fakes from lip and sound agreement, trained on real clips only.", 16),
        ("n", "4.  Hold the false-alarm rate on a new corpus with real clips only (H4).", 16),
        ("n", "5.  Test whether fusion beats the visual path alone (H2).", 16),
        ("s", "H3 (which stream was faked) is the next stage.", 16),
    ], name="card_objectives")
    card(s, col_x(7), Y0 + 800000, span_w(5), H0 - 800000, [
        ("h", "Not claimed"),
        ("b", "A new network architecture", 16),
        ("b", "Modality prediction (H3)", 16),
        ("b", "A comparison with AVH-Align or HAVIC, which we have not yet run on our protocol", 16),
    ], name="card_notclaimed")

    # 3. related work -----------------------------------------------------------------------------------
    s = slide("Related work and the gap",
              "AVH-Align trains an alignment model on real clips only and showed that it survives trimming the leading "
              "silence; our sync path follows that idea and we do not claim it. Smeu et al. and DeepFake Doctor found "
              "the silence cue, and Mueller et al. found it in ASVspoof. What we did not find published is a codec census "
              "per category, a classifier on container metadata alone, and one cleaning step applied to training and "
              "test data that covers codec, both silence edges, loudness and length together. Split conformal and the "
              "Neyman-Pearson umbrella are standard; we apply them, we do not invent them.",
              "The silence cue and the real-only idea are known. The joint cleaning and the codec census are new.")
    rows = [["AVH-Align (Smeu et al., 2025)", "Real-only alignment on frozen AV-HuBERT; trims leading silence at test", "Same idea, measured after cleaning both training and test data"],
            ["DeepFake Doctor (2025)", "Silence cue, Wav2Lip leak, identity split", "Codec census, joint cleaning, HiFi-AVDF audit"],
            ["Mueller et al. (2021)", "Silence alone separates ASVspoof 2019 LA", "The same check for audio-visual corpora"],
            ["AVFF, HAVIC", "Audio-visual fusion; HAVIC introduces HiFi-AVDF", "File-cue ceiling for HiFi-AVDF"],
            ["Bates et al.; Tong et al.", "Conformal and umbrella false-positive control", "Applied to a video detector across corpora"]]
    d.add_table(s, L, Y0, [2900000, 4100000, W - 7000000], ["Work", "What it does", "What we add"],
                rows, size="auto", max_h=H0, bold_first_col=True, fill=H0)

    # 4. shortcut audit ---------------------------------------------------------------------------------
    s = slide("Shortcut audit",
              "The audit reads each clip's container metadata and the edges of its audio track, and nothing else: codecs, "
              "bitrates, frame size, frame rate, sample rate, duration, leading and trailing digital silence and "
              "loudness. Gradient-boosted trees given all those cues, scored out of fold over five folds grouped by "
              "identity, give the shortcut ceiling. The ceiling is the bar a detector must beat on that corpus; 0.5 is "
              "the wrong bar. FakeAVCeleb and HiFi-AVDF are almost fully readable from their files. DFDC is close to "
              "clean on these cues, which may be one reason detectors trained on FakeAVCeleb fall to chance there.",
              "A model that never sees a pixel or hears a word predicts the label from file cues alone.")
    figure_or_pending(s, "fig_census.png", col_x(0), Y0, span_w(7), H0, "Shortcut ceiling per corpus, raw and neutral")
    card(s, col_x(7), Y0, span_w(5), H0, [
        ("h", "Cues read"),
        ("s", "Codecs, bitrates, frame size, frame rate, sample rate, duration, silence at both ends, loudness.", 16),
        ("h", "How to read it"),
        ("s", "ROC-AUC of the cue-only model, out of fold, folds grouped by identity. Higher means a weaker "
              "benchmark. The dashed line is our bar for clean data.", 16),
        ("h", "Neutral ceilings on targets"),
        ("s", "LAV-DF {{audit.lavdf.neutral.clip}}, DFDC {{audit.dfdc.neutral.clip}}, "
              "HiFi-AVDF {{audit.hifiavdf.neutral.clip}}", 16),
    ], name="card_audit")

    # 5. FakeAVCeleb census ------------------------------------------------------------------------------
    s = slide("FakeAVCeleb: the codec gives it away",
              "Every real FakeAVCeleb clip is H.264 video with AAC audio. Generated speech in the real-video category is "
              "stored almost entirely as MP3, and most face swaps are MPEG-4 Part 2. A detector that learns MPEG-4 means "
              "fake, or MP3 means fake, needs no face and no voice. Clips with generated audio also start and end with "
              "digital silence, while real clips start with room noise. To our knowledge the MP3 track and the codec "
              "split per category have not been reported for this corpus.",
              "Real clips are H.264 and AAC. Cloned speech is MP3. Face swaps are mostly MPEG-4.")
    c = "audit.fakeavceleb.raw.census."
    k = "audit.fakeavceleb.neutral.census."
    rows = [["Real video, real audio", n(f"{{{{{c}RealVideo-RealAudio.clips|int}}}}"), "H.264 (all)", "AAC (all)",
             n(f"{{{{{k}RealVideo-RealAudio.clips|int}}}}")],
            ["Real video, fake audio", n(f"{{{{{c}RealVideo-FakeAudio.clips|int}}}}"), "H.264 (all)",
             n(f"**MP3 ({{{{{c}RealVideo-FakeAudio.audio_codec.mp3|int}}}})**, AAC ({{{{{c}RealVideo-FakeAudio.audio_codec.aac|int}}}})"),
             n(f"{{{{{k}RealVideo-FakeAudio.clips|int}}}}")],
            ["Fake video, real audio", n(f"{{{{{c}FakeVideo-RealAudio.clips|int}}}}"),
             n(f"**MPEG-4 ({{{{{c}FakeVideo-RealAudio.video_codec.mpeg4|int}}}})**, H.264 ({{{{{c}FakeVideo-RealAudio.video_codec.h264|int}}}})"),
             "AAC (all)", n(f"{{{{{k}FakeVideo-RealAudio.clips|int}}}}")],
            ["Fake video, fake audio", n(f"{{{{{c}FakeVideo-FakeAudio.clips|int}}}}"), "**MPEG-4 (all)**", "AAC (all)",
             n(f"{{{{{k}FakeVideo-FakeAudio.clips|int}}}}")]]
    _, th = d.add_table(s, L, Y0, [2900000, 1500000, 2700000, 2200000, W - 9300000],
                        ["Category", "Raw clips", "Video codec", "Audio codec", "Kept"], rows, size=16,
                        bold_first_col=True, fill=2400000)
    y = Y0 + th + 150000
    card(s, col_x(0), y, span_w(12), BOT - y, [
        ("h", "Why it matters"),
        ("s", "A detector that learns MPEG-4 means fake, or MP3 means fake, needs no face and no voice. The cue-only "
              "ceiling is {{audit.fakeavceleb.raw.clip}}. Kept: clips left after cleaning; every refused clip carries "
              "generated speech.", 16),
    ], name="card_census")

    # 6. HiFi-AVDF -----------------------------------------------------------------------------------
    s = slide("HiFi-AVDF: generator fingerprints",
              "HiFi-AVDF pairs each real video with a copy made by one of six commercial generators: Kling 2.5, "
              "PixVerse V5, Seedance 1.0, Sora 2, Veo 3.1 and Wan 2.5. Real clips share one frame size and five of the "
              "six generators write another. Sora 2 and Veo 3.1 write every clip at one fixed length. Seedance writes at "
              "a much higher bitrate, and Kling and Seedance start with digital silence. None of these cues needs the "
              "content of the video, and together they give a ceiling close to 1.",
              "A recent corpus of commercial generators is readable from its files: ceiling {{audit.hifiavdf.raw.clip}}.")
    rows = [["Frame size", "Real clips share one size; five of six generators write another"],
            ["Length", "Sora 2 and Veo 3.1 write one fixed length; real lengths vary"],
            ["Bitrate", "Seedance writes far above the real clips"],
            ["Edge silence", "Kling and Seedance start with digital silence; real clips do not"]]
    _, th = d.add_table(s, L, Y0, [2400000, W - 2400000], ["Cue", "What the files show"], rows, size=16,
                        bold_first_col=True, pad=50000)
    y = Y0 + th + 150000
    card(s, col_x(0), y, span_w(12), BOT - y, [
        ("h", "What we do about it"),
        ("s", "Neutral copies get one frame size, the native size of the real clips, besides the common cleaning steps. "
              "{{audit.hifiavdf.raw.clips|int}} clips are audited, folds grouped by real and generated pair."),
    ], name="card_hifi")

    # 7. neutralizer -------------------------------------------------------------------------------------
    s = slide("Removing the cues: neutralizer v4",
              "The first neutralizer cut with an ffmpeg input seek, normalized loudness with a standard filter and "
              "re-encoded. It looked correct, but AAC encoders add priming samples at the start of a stream, and "
              "re-muxing exposed them differently per source, so face swaps gained leading silence that real clips did "
              "not. Cleaning had put the shortcut back. Version 4 never lets ffmpeg touch the audio timeline before "
              "encoding: the gain, the cut and the dither are done on the samples in numpy, then both streams are "
              "re-encoded once.",
              "Version 1 put the silence back through hidden AAC padding. Version 4 works on the samples.")
    d.flow(s, Y0, 900000, ["**Decode**\n16 kHz mono", "**Level**\n-23 dBFS RMS gain", "**Cut**\nfirst sound, exactly 3.00 s",
                           "**Dither**\n-66 dBFS", "**Encode**\nH.264 25 fps, AAC"],
           [F_GREY, F_BLUE, F_GREEN, F_ORANGE, F_RED])
    y = Y0 + 1050000
    card(s, col_x(0), y, span_w(6), BOT - y, [
        ("h", "Why version 1 failed"),
        ("b", "AAC priming samples became leading silence on face swaps only", 16),
        ("b", "Duration and trailing silence still differed by class", 16),
        ("b", "Clip ceiling stayed at {{audit.fakeavceleb.v1.clip}}", 16),
    ], name="card_v1")
    card(s, col_x(6), y, span_w(6), BOT - y, [
        ("h", "What each step removes"),
        ("b", "Fixed length removes duration", 16),
        ("b", "Start at first sound removes leading silence", 16),
        ("b", "Dither removes exact digital zeros in pauses", 16),
        ("b", "One codec and level removes format and loudness", 16),
    ], name="card_v4")

    # 8. H1 ------------------------------------------------------------------------------------------------
    s = slide("H1: the cleaning test was not met",
              "H1 required the clip-label ceiling on neutral FakeAVCeleb to be at most 0.60. Version 4 gives "
              "{{audit.fakeavceleb.neutral.clip}} on {{audit.fakeavceleb.neutral.clips|int}} clips, so H1 is not met "
              "and we say so. The strongest remaining single cue is video bitrate. Under one constant quality setting "
              "bitrate follows image content, so smoother face swaps take fewer bits; that is a property of the "
              "manipulated pixels, but H1 was defined on all cues. Cleaning also refuses "
              "{{neutral.fakeavceleb.refused|int}} clips as too short, all of them generated speech, which is a "
              "selection effect we report beside every result.",
              "Ceiling fell from {{audit.fakeavceleb.raw.clip}} to {{audit.fakeavceleb.neutral.clip}}; the bar was 0.60.")
    figure_or_pending(s, "fig_h1.png", col_x(0), Y0, span_w(7), H0, "Shortcut ceiling before and after neutralization")
    card(s, col_x(7), Y0, span_w(5), H0, [
        ("h", "Reading"),
        ("b", "**Not met:** {{audit.fakeavceleb.neutral.clip}} against 0.60", 16),
        ("b", "Left over: video bitrate, which follows image content", 16),
        ("b", "Audio label ceiling {{audit.fakeavceleb.neutral.audio}}, from edges and loudness", 16),
        ("b", "{{neutral.fakeavceleb.refused|int}} short generated-speech clips refused; no real clip refused", 16),
    ], name="card_h1")

    # 9. sync detector -------------------------------------------------------------------------------------
    s = slide("Real-only lip-sync detector",
              "We crop 88 pixel grey mouth images at 25 frames per second. A frozen AV-HuBERT Base encoder encodes the "
              "lips and the audio in two separate passes. A small head takes a lips frame and an audio frame and "
              "predicts whether they belong together; positives are aligned frames of real clips, negatives are the "
              "same lips with audio shifted by 5 to 20 frames or taken from another clip. The head trains and selects its "
              "epoch on real clips only, so it cannot learn a generator's habits. The clip score is the mean misfit over "
              "two 2 second windows.",
              "Frozen AV-HuBERT features, a small head trained on real clips only.")
    d.flow(s, Y0, 1000000, ["**Neutral clip**\nv4, 3.00 s", "**Mouth crops**\n88 px grey, 25 fps",
                            "**AV-HuBERT**\nfrozen, lips and audio apart", "**Alignment head**\ntrained on real only",
                            "**Score**\nmean misfit, 2 windows"], [F_GREY, F_BLUE, F_BLUE, F_GREEN, F_ORANGE], gap=170000)
    y = Y0 + 1150000
    card(s, col_x(0), y, span_w(6), BOT - y, [
        ("h", "The maths"),
        ("eq", "Window misfit:  m = mean over frames of  -log sigmoid(g)", 16),
        ("eq", "Clip score:  s = mean of m over the two windows", 16),
        ("s", "Reported as 1 - exp(-s), which keeps the order, so AUC is unchanged.", 16),
        ("s", "g: the head's logit for an aligned lips and audio pair.", 16),
    ], name="card_maths")
    card(s, col_x(6), y, span_w(6), BOT - y, [
        ("h", "Protocol"),
        ("b", "70/15/15 split by identity; identity-strict test subset", 16),
        ("b", "Seeds 17, 29, 43; 1,000 bootstrap resamples of people", 16),
        ("b", "Threshold from validation, with a file that records its clips", 16),
    ], name="card_protocol")

    # 10. in-domain results ---------------------------------------------------------------------------------
    s = slide("Sync path on neutral FakeAVCeleb",
              "On the test set the AUC is {{sync.test.seed17.auc}}, {{sync.test.seed29.auc}} and "
              "{{sync.test.seed43.auc}} for the three seeds, and the identity-strict subset, whose people never appear "
              "in training, gives the same picture. Every interval lies above the neutral shortcut ceiling of "
              "{{audit.fakeavceleb.neutral.clip}}, so the path does better than file cues alone on the same copies. The "
              "operating point is weaker than the ranking: at the validation threshold the false-positive rate on real "
              "test clips is between {{sync.test.seed43.fpr}} and {{sync.test.seed17.fpr}}. That false-alarm rate is "
              "what H4 addresses.",
              "Every interval clears the neutral ceiling of {{audit.fakeavceleb.neutral.clip}}. False alarms stay high.")
    rows = []
    for part, label in [("test", "Test"), ("test-identity-strict", "Identity-strict")]:
        for seed in (17, 29, 43):
            k = f"sync.{part}.seed{seed}"
            rows.append([label if seed == 17 else "", str(seed),
                         n(f"{{{{{k}.auc}}}} [{{{{{k}.auc_low}}}}, {{{{{k}.auc_high}}}}]"),
                         n(f"{{{{{k}.fpr}}}}"), n(f"{{{{{k}.balanced_accuracy}}}}")])
    _, th = d.add_table(s, L, Y0, [2400000, 1200000, 3300000, 1900000, W - 8800000],
                        ["Set", "Seed", "ROC-AUC [95% interval]", "FPR", "Balanced acc."], rows, size=16,
                        bold_first_col=True, pad=20000)
    y = Y0 + th + 150000
    d.add_text(s, L, y, W, 300000, T([("cap", "Test: {{sync.test.seed17.n|int}} clips. Identity-strict: "
                                               "{{sync.test-identity-strict.seed17.n|int}} clips. FPR and balanced "
                                               "accuracy at the threshold chosen on validation.")]), "cap")
    y += 400000
    card(s, col_x(0), y, span_w(6), BOT - y, [
        ("h", "Ranking holds"),
        ("s", "Every interval lies above the neutral shortcut ceiling of {{audit.fakeavceleb.neutral.clip}}, so the "
              "path beats file cues alone on the same copies.", 16)], name="card_rank")
    card(s, col_x(6), y, span_w(6), BOT - y, [
        ("h", "Threshold does not"),
        ("s", "At the validation threshold, {{sync.test.seed43.fpr|pct}} to {{sync.test.seed17.fpr|pct}} of real test "
              "clips are flagged. H4 tests a fix.", 16)], name="card_fpr")

    # 11. per-method ----------------------------------------------------------------------------------------
    s = slide("Scores follow the mechanism",
              "Split by manipulation method, the order is the same in all three seeds. Voice cloning replaces the voice "
              "and leaves the real lips, which still move to the original words, so it is caught best. Wav2Lip "
              "regenerates the lips to fit the audio; the fit is imperfect, so it is caught well but less well. A face "
              "swap alone keeps the original lips and voice, so there is little misalignment to find. A detector that "
              "learned file habits would have no reason to rank methods this way; this is evidence, not proof, since "
              "the residual audio cue also sits on generated speech.",
              "Voice cloning is caught best, Wav2Lip next, face swaps that keep the real lips least.")
    figure_or_pending(s, "fig_sync_methods.png", col_x(0), Y0, span_w(8), H0, "Sync path AUC per manipulation method")
    card(s, col_x(8), Y0, span_w(4), H0, [
        ("h", "Seed 17, test AUC"),
        ("b", "Voice cloning: {{sync.test.seed17.method.rtvc}}", 16),
        ("b", "Wav2Lip: {{sync.test.seed17.method.wav2lip}}", 16),
        ("b", "FaceSwap + Wav2Lip: {{sync.test.seed17.method.faceswap-wav2lip}}", 16),
        ("b", "FSGAN: {{sync.test.seed17.method.fsgan}}", 16),
        ("b", "FaceSwap: {{sync.test.seed17.method.faceswap}}", 16),
    ], name="card_methods")

    # 12. cross-corpus ----------------------------------------------------------------------------------------
    s = slide("Cross-corpus: frozen model, new data",
              "The models are frozen. Nothing is retrained, selected or thresholded on a target corpus. Each target is "
              "audited raw, neutralized with version 4 and audited again, and the neutral ceiling is the bar. LAV-DF "
              "replaces single words inside real recordings, our DFDC subset has no fake audio, and HiFi-AVDF makes "
              "whole clips with commercial generators, so each target tests something different. LAV-DF has two rows. "
              "Our neutral copy keeps 3 seconds from the first sound, and in 46 percent of LAV-DF fakes those 3 seconds "
              "hold no edited word, so the file label calls real footage fake. We found this after reading the file-label "
              "result and added the window label in amendment A9; both rows stay on the slide. The DFDC row is after "
              "amendment A10: the face finder first missed small DFDC faces and scored only 921 clips, and locating the "
              "face first raised that to {{sync.dfdc.seed17.scored|int}}, with little change in AUC.",
              "Each locked target, beside its own neutral shortcut ceiling.")
    rows = []
    for name, k, audit in [("LAV-DF, file label", "lavdf-test", "lavdf.neutral"), ("LAV-DF, window label", "lavdf-window", "lavdf.window"),
                           ("DFDC", "dfdc", "dfdc.neutral"), ("HiFi-AVDF", "hifiavdf", "hifiavdf.neutral")]:
        rows.append([name, n(f"{{{{sync.{k}.seed17.scored|int}}}}")]
                    + [n(f"{{{{sync.{k}.seed{sd}.auc}}}}") for sd in (17, 29, 43)]
                    + [n(f"{{{{audit.{audit}.clip}}}}")])
    _, th = d.add_table(s, L, Y0, [2400000, 1500000] + [(W - 6100000) // 3] * 3 + [2200000],
                        ["Target", "Clips", "Seed 17", "Seed 29", "Seed 43", "Neutral ceiling"], rows, size=16,
                        bold_first_col=True, pad=40000)
    y = Y0 + th + 150000
    figure_or_pending(s, "fig_cross_corpus.png", col_x(0), y, span_w(12), BOT - y,
                      "Sync path AUC on each locked target", keys=("sync.hifiavdf.seed17.auc",))

    # 13. H4 ----------------------------------------------------------------------------------------------------
    s = slide("H4: false alarms from real clips only",
              "A platform cannot label fakes in its own traffic, but it has plenty of real clips. For each target and seed "
              "we draw 200 real clips as a calibration set, with a seed fixed in the preregistration, and read no fake "
              "label. Three thresholds are scored on the remaining clips: the source validation threshold carried over "
              "unchanged, the split conformal threshold, and the Neyman-Pearson umbrella threshold. The bar is a "
              "realized false-positive rate of at most 0.07 with the conformal rule on every target. DFDC is excluded "
              "because our subset has fewer than 200 real clips.",
              "Set the threshold from 200 real target clips. Does the false-alarm rate hold at 5%?")
    card(s, col_x(0), Y0, span_w(5), H0, [
        ("h", "Three rules"),
        ("b", "Source: FakeAVCeleb validation threshold, unchanged", 16),
        ("b", "Conformal: the k-th smallest real score, k = ceil((n + 1)(1 - alpha))", 16),
        ("b", "Umbrella: smallest rank with P(FPR > alpha) at most delta", 16),
        ("s", "alpha = delta = 0.05, n = 200. Bar: FPR at most 0.07.", 16),
    ], name="card_rules")
    if n.has("h4.lavdf-test-seed17.conformal_real_only.fpr", "h4.lavdf-window-seed17.conformal_real_only.fpr",
             "h4.hifiavdf-seed17.conformal_real_only.fpr"):
        rows = []
        for name, k in [("LAV-DF, file", "lavdf-test"), ("LAV-DF, window", "lavdf-window"), ("HiFi-AVDF", "hifiavdf")]:
            for sd in (17, 29, 43):
                h = f"h4.{k}-seed{sd}"
                rows.append([name if sd == 17 else "", str(sd), n(f"{{{{{h}.source_validation.fpr}}}}"),
                             n(f"{{{{{h}.conformal_real_only.fpr}}}}"), n(f"{{{{{h}.conformal_real_only.tpr}}}}"),
                             n(f"{{{{{h}.success}}}}")])
        d.add_table(s, col_x(5), Y0, [1300000, 700000, 1100000, 1150000, 1150000, span_w(7) - 5400000],
                    ["Target", "Seed", "Source FPR", "Conf. FPR", "Conf. TPR", "Met"], rows, size=14,
                    bold_first_col=True, max_h=H0)
    else:
        card(s, col_x(5), Y0, span_w(7), H0, [("h", "Result"),
                                              ("p", "[pending] The H4 table and figure appear when the cross-corpus "
                                                    "run finishes.", 16)], name="card_pending")

    # 13b. H2 ---------------------------------------------------------------------------------------------------
    s = slide("H2: does fusion beat visual-only?",
              "The fusion was fixed in amendment A8 before any visual or audio model existed: a logistic regression on the "
              "three standardized branch logits, fit on FakeAVCeleb validation and cross-fitted by identity. The statistic "
              "is AUC of the fusion minus AUC of the visual path, on the clip label, averaged over three seeds, with one "
              "identity bootstrap shared by the seeds. Holm correction runs over DFDC, LAV-DF and the identity-strict "
              "FakeAVCeleb test, and H2 holds if two of the three survive. HiFi-AVDF is reported outside the rule. "
              "It holds, but read the levels. On FakeAVCeleb the gain is the cloned-voice clips the visual path never "
              "trained on. On LAV-DF the visual path is below chance, at {{h2.lavdf-window.auc.visual.min}} to "
              "{{h2.lavdf-window.auc.visual.max}}, and fusion still trails the audio path and the sync path alone, "
              "likely because its weights come from FakeAVCeleb. On DFDC every clip has real audio and fusion adds nothing. "
              "HiFi-AVDF repeats the LAV-DF pattern: fusion beats a visual path near chance but trails audio alone. "
              "So adding audio and sync helps, but this fusion does not transfer. A fusion that can drop a path is "
              "future work.",
              "Fused by a model fixed in advance. Does it beat the visual path on new corpora?")
    rows = [[name, n(f"{{{{h2.{k}.clips|int}}}}"),
             n(f"{{{{h2.{k}.estimate}}}} [{{{{h2.{k}.lower}}}}, {{{{h2.{k}.upper}}}}]"), n(f"{{{{h2.{k}.p}}}}"),
             "n/a" if k == "hifiavdf" else n(f"{{{{h2.{k}.holm}}}}")]
            for name, k in [("FakeAVCeleb, strict", "test-identity-strict"), ("DFDC", "dfdc"),
                            ("LAV-DF, window label", "lavdf-window"), ("HiFi-AVDF (reported)", "hifiavdf")]]
    _, th = d.add_table(s, L, Y0, [3200000, 1400000, 3400000, 1200000, W - 9200000],
                        ["Target", "Clips", "Fusion minus visual [95%]", "p", "Holm"], rows, size=16,
                        bold_first_col=True, pad=40000)
    y = Y0 + th + 150000
    card(s, col_x(0), y, span_w(6), BOT - y, [
        ("h", "Verdict"),
        ("p", "By the preregistered rule, H2 holds: **{{h2.success}}**.", 16),
    ], name="card_h2")
    card(s, col_x(6), y, span_w(6), BOT - y, [
        ("h", "Fusion weights, seed 17"),
        ("s", "Visual {{h2.weights.seed17.visual}}, audio {{h2.weights.seed17.audio}}, sync "
              "{{h2.weights.seed17.sync}}, on standardized logits.", 16),
    ], name="card_weights")

    # 14. limitations and conclusion -----------------------------------------------------------------------
    s = slide("Limitations and conclusion",
              "Audio-visual benchmarks leak their labels through their files, and a detector score on them means little "
              "until it is set beside the file-cue ceiling. Our cleaning lowers the FakeAVCeleb ceiling a lot but misses "
              "the preregistered bar, and we report that. The real-only sync path clears the neutral ceiling and its "
              "errors follow the mechanism. The fusion test was fixed before any branch was trained, and we report it either "
              "way. Next come modality prediction and the published baselines, AVH-Align and HAVIC, on the same protocol.",
              "Benchmarks leak labels through files. Compare every score with the shortcut ceiling.")
    card(s, col_x(0), Y0, span_w(6), H0, [
        ("h", "Limitations"),
        ("b", "H1 missed its bar: {{audit.fakeavceleb.neutral.clip}} against 0.60", 16),
        ("b", "A residual audio cue: ceiling {{audit.fakeavceleb.neutral.audio}}", 16),
        ("b", "Cleaning refuses {{neutral.fakeavceleb.refused|int}} generated-speech clips", 16),
        ("b", "No published baselines yet; H3 not run", 16),
        ("b", "One training corpus: FakeAVCeleb, 2021, about 500 real videos", 16),
        ("b", "Possible speaker overlap with AV-HuBERT pretraining", 16),
    ], name="card_limits")
    card(s, col_x(6), Y0, span_w(6), H0, [
        ("h", "Contributions"),
        ("n", "1.  A shortcut census of four corpora", 16),
        ("n", "2.  A cleaning protocol, with its failure explained", 16),
        ("n", "3.  A real-only lip-sync detector, measured after cleaning", 16),
        ("n", "4.  A false-alarm test from real clips only", 16),
        ("n", "5.  A preregistered fusion test (H2)", 16),
        ("s", "Hypotheses were committed before the runs. Every number has a receipt.", 16),
    ], name="card_contrib")

    # divider ---------------------------------------------------------------------------------------------
    s = slide("Backup slides",
              "The slides after this divider answer likely questions. They cover the preregistration and receipts, the "
              "mistakes we caught, the maths we changed and the data we used. They are not part of the main talk.")
    d.add_text(s, L, Y0, W, 1700000, [("p", "B1  Preregistration and receipts"), ("p", "B2  Mistakes we caught"),
                                       ("p", "B3  How the numbers are computed"), ("p", "B4  Data and access")],
               "backup_contents")

    # B1 ------------------------------------------------------------------------------------------------------
    s = slide("B1  Preregistration and receipts",
              "The hypotheses, success bars, seeds and splits were committed as P-001 before any model was trained on "
              "neutral data, so git order proves the order. Every change is a dated amendment committed before the run it "
              "affects. Every evaluation writes a receipt with its command and the hashes of its inputs, code and outputs, "
              "and rerunning the same code verifies the outputs instead of overwriting them.",
              "Hypotheses before runs, amendments before changes, a receipt for every number.")
    rows = [["H1", "Neutral FakeAVCeleb clip ceiling at most 0.60", "Not met: {{audit.fakeavceleb.neutral.clip}}"],
            ["H2", "Fusion beats visual-only on 2 of 3 targets, after Holm", "Holds: {{h2.success}}"],
            ["H3", "Predict the manipulated modality across corpora", "Next stage"],
            ["H4", "Real-only conformal FPR at most 0.07 on every target", "Running"]]
    d.add_table(s, L, Y0, [900000, 6000000, W - 6900000], ["Test", "Bar", "Status"], T_rows(rows, n), size=18,
                bold_first_col=True, fill=H0)

    # B2 ------------------------------------------------------------------------------------------------------
    s = slide("B2  Mistakes we caught",
              "The project handbook logs every mistake we could trace, with how it was caught, what it cost and the fix. "
              "These six changed results. Each one was found by a check, not by a bad number, and each fix has a test.",
              "Six of the logged mistakes that would have changed a result.")
    rows = [["Identity leak in FF++", "Donor faces crossed train and test", "Split by connected identity component"],
            ["Test-set selection", "An ensemble was picked on a test set", "Select on validation only"],
            ["Fixed 0.5 threshold", "Meaningless on 10:1 fake-heavy data", "Validation threshold file, overlap refused"],
            ["Neutralizer v1", "Re-mux put silence back on face swaps", "Version 4 works on the samples"],
            ["LAV-DF window audit", "One file supplied both labels", "Audit per file"],
            ["Two writers", "Two runs wrote the same outputs", "One chain per output; resume by cut log"]]
    d.add_table(s, L, Y0, [2800000, 3900000, W - 6700000], ["Mistake", "Effect", "Fix"], rows, size=18,
                max_h=H0, bold_first_col=True, fill=H0)

    # B3 ------------------------------------------------------------------------------------------------------
    s = slide("B3  How the numbers are computed",
              "Old and new numbers in this project are not directly comparable, so this slide lists what changed in the "
              "maths. None of these changes was chosen after seeing the result it affects. ROC-AUC itself, the identity "
              "bootstrap and the three seeds are unchanged.",
              "What changed in the maths, and why.")
    rows = [["EER", "Linear interpolation where FPR = FNR", "The nearest ROC point is off on small sets"],
            ["Threshold", "Maximum balanced accuracy on validation", "0.5 means nothing on skewed data"],
            ["Splits", "Connected identity components", "Donor faces leaked across splits"],
            ["Sync score", "Mean misfit s, shown as 1 - exp(-s)", "Monotone, so AUC is unchanged"],
            ["Ceiling", "Boosted trees on cues, out of fold", "The bar a detector must beat"],
            ["H4", "Conformal rank; umbrella bound", "Standard formulas, not new maths"]]
    d.add_table(s, L, Y0, [1900000, 4600000, W - 6500000], ["What", "Now", "Why"], rows, size=18,
                max_h=H0, bold_first_col=True, fill=H0)

    # B4 ------------------------------------------------------------------------------------------------------
    s = slide("B4  Data and access",
              "FakeAVCeleb trains every model. LAV-DF, DFDC and HiFi-AVDF are locked targets that never train, select or "
              "calibrate anything, except the real-only calibration split of H4. AV-Deepfake1M, its successor, "
              "DeepSpeak v2 and KoDF have been requested for the next stage.",
              "One training corpus, three locked targets, four more requested.")
    rows = [["FakeAVCeleb", "Train, validate, test", "{{neutral.fakeavceleb.kept|int}} neutral clips"],
            ["LAV-DF", "Locked target", "{{audit.lavdf.raw.clips|int}} files, sampled by recording"],
            ["DFDC", "Locked target", "{{audit.dfdc.raw.clips|int}} clips, no fake audio"],
            ["HiFi-AVDF", "Locked target", "{{audit.hifiavdf.raw.clips|int}} clips, six generators"],
            ["AV-Deepfake1M, 1M++, DeepSpeak v2, KoDF", "Requested", "Next stage"]]
    d.add_table(s, L, Y0, [3600000, 3000000, W - 6600000], ["Corpus", "Role", "Size"], T_rows(rows, n), size=18,
                bold_first_col=True, fill=H0)

    # save ----------------------------------------------------------------------------------------------------
    if n.missing and not draft:
        print("missing numbers: " + ", ".join(sorted(n.missing)))
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.prs.save(OUT)
    for w in d.warnings:
        print("layout:", w)
    print(f"{d.count} slides -> {OUT.relative_to(ROOT)}; {len(n.missing)} numbers pending")
    return 0


def T_rows(rows, n):
    return [[n(c) for c in row] for row in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="mark missing numbers [pending] instead of stopping")
    raise SystemExit(build(parser.parse_args().draft))
