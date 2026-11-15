"""Build the A0 portrait poster from results/numbers.json.

    .venv/Scripts/python.exe deliverables/poster/build_poster.py            stops on a missing number
    .venv/Scripts/python.exe deliverables/poster/build_poster.py --draft    marks missing numbers [pending]

Output: deliverables/build/Deepfake_Detection_Poster.pptx. The PDF comes from PowerPoint
(deliverables/poster/export.ps1). Numbers, figures and wording follow the paper and the slides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "deliverables" / "slides"))

from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR  # noqa: E402
from pptx.util import Emu, Pt  # noqa: E402

from build_deck import BRAND, FIG, GUIDE, GUIDE_ROLE, MEMBERS, TITLE, Numbers  # noqa: E402
from deck_kit import F_BLUE, F_GREEN, F_GREY, F_ORANGE, F_RED, Deck, rgb  # noqa: E402
import slide_common as C  # noqa: E402
from slide_common import RED  # noqa: E402

OUT = ROOT / "deliverables" / "build" / "Deepfake_Detection_Poster.pptx"
IN = 914400
PW, PH = int(33.11 * IN), int(46.81 * IN)       # A0 portrait
M = int(0.9 * IN)                                # outer margin
GAP = int(0.6 * IN)                              # between columns
COLS = 3
CW = (PW - 2 * M - (COLS - 1) * GAP) // COLS
HEAD_H = int(6.2 * IN)
FOOT_H = int(1.3 * IN)
BODY_TOP = HEAD_H + int(0.6 * IN)
SZ_H, SZ_B, SZ_S = 48, 32, 28                    # section heading, body, secondary (points)


def build(draft: bool) -> int:
    n = Numbers(draft)
    d = Deck(None)
    d.prs.slide_width, d.prs.slide_height = Emu(PW), Emu(PH)
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    d.count = 1

    def T(paras):
        return [(p[0], n(p[1]), *p[2:]) for p in paras]

    def colx(k):
        return M + k * (CW + GAP)

    # header: red band, logos, title, names
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Emu(PW), Emu(HEAD_H))
    band.fill.solid(); band.fill.fore_color.rgb = rgb(RED); band.line.fill.background()
    plate = int(3.6 * IN)
    for k, (fn, alt) in enumerate([("somaiya_vidyavihar_logo.jpg", "Somaiya Vidyavihar University logo"),
                                   ("somaiya_trust_logo.png", "Somaiya Trust logo")]):
        x = M if k == 0 else PW - M - plate
        bg = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(int(0.8 * IN)), Emu(plate), Emu(int(3.0 * IN)))
        bg.fill.solid(); bg.fill.fore_color.rgb = rgb("FFFFFF"); bg.line.fill.background()
        d.pic_fit(s, BRAND / fn, x + int(0.2 * IN), int(1.0 * IN), plate - int(0.4 * IN), int(2.6 * IN), alt)
    tw = PW - 2 * M - 2 * plate - 2 * GAP
    tx = M + plate + GAP
    d.add_text(s, tx, int(0.6 * IN), tw, int(2.7 * IN), [("h", TITLE, 80)], "poster_title", anchor=MSO_ANCHOR.MIDDLE)
    names = "     ".join(nm for nm, _ in MEMBERS)
    d.add_text(s, tx, int(3.3 * IN), tw, int(2.6 * IN), [
        ("p", f"**{names}**", 36),
        ("p", f"Guide: {GUIDE}, {GUIDE_ROLE}", 32),
        ("p", "Department of Electronics and Computer Engineering, K. J. Somaiya School of Engineering, "
              "Somaiya Vidyavihar University", 28),
    ], "poster_names")
    for shp in s.shapes:     # header text is white on the band
        if shp.has_text_frame and shp.name.startswith(("poster_title", "poster_names")):
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = rgb("FFFFFF")

    def section(col, y, h, title, paras):
        """A card; h=None sizes it to its text, and the last card of a column takes the rest."""
        if h is not None and h <= 0:
            raise SystemExit(f"poster column {col + 1} overflows at '{title}'")
        body = T([("h", title, SZ_H)] + list(paras))
        card = d.add_card(s, colx(col), y, CW, h or int(20 * IN), body, inset=int(0.25 * IN))
        if h is None:
            h = int(C.estimate_text_height(card) + 0.15 * IN)
            card.height = Emu(h)
        if y + h > PH - FOOT_H:
            raise SystemExit(f"poster column {col + 1} overflows at '{title}'")
        return y + h + GAP

    def figure(col, y, name, alt, keys=(), max_h=None):
        path = FIG / name
        if not path.exists() or not n.has(*keys):
            d.add_card(s, colx(col), y, CW, int(2 * IN), T([("p", "[pending] " + alt + ": drawn when its run finishes.", SZ_S)]),
                       inset=int(0.25 * IN))
            return y + int(2 * IN) + GAP
        _, _, ph = d.pic_fit(s, path, colx(col), y, CW, max_h or int(12 * IN), alt)
        return y + ph + GAP

    # column 1: problem and audit
    y = BODY_TOP
    y = section(0, y, None, "1. The problem", [
        ("p", "Audio-visual deepfake detectors score near perfect on their own benchmark and fall toward chance on "
              "new data.", SZ_B),
        ("p", "Every fake passes through a production pipeline that leaves marks in the file: a codec, a length, "
              "digital silence. **Do the files give the label away?**", SZ_B),
    ])
    y = section(0, y, None, "2. Shortcut audit", [
        ("p", "Gradient-boosted trees read only file metadata and audio edges: no pixel, no word. Their out-of-fold "
              "AUC is the shortcut ceiling, the bar a real detector must beat.", SZ_B),
    ])
    y = figure(0, y, "fig_census.png", "Shortcut ceiling per corpus, raw and neutral")
    c, k = "audit.fakeavceleb.raw.census.", "audit.fakeavceleb.neutral.census."
    rows = [["Real video, real audio", n(f"{{{{{c}RealVideo-RealAudio.clips|int}}}}"), "H.264, AAC"],
            ["Real video, fake audio", n(f"{{{{{c}RealVideo-FakeAudio.clips|int}}}}"),
             n(f"H.264, **MP3 ({{{{{c}RealVideo-FakeAudio.audio_codec.mp3|int}}}})**")],
            ["Fake video, real audio", n(f"{{{{{c}FakeVideo-RealAudio.clips|int}}}}"),
             n(f"**MPEG-4 ({{{{{c}FakeVideo-RealAudio.video_codec.mpeg4|int}}}})**, AAC")],
            ["Fake video, fake audio", n(f"{{{{{c}FakeVideo-FakeAudio.clips|int}}}}"), "**MPEG-4 (all)**, AAC"]]
    d.add_text(s, colx(0), y, CW, int(0.8 * IN), [("h", "FakeAVCeleb codec census", 40)], "cap")
    y += int(0.9 * IN)
    _, th = d.add_table(s, colx(0), y, [int(CW * 0.42), int(CW * 0.18), CW - int(CW * 0.42) - int(CW * 0.18)],
                        ["Category", "Clips", "Codecs"], rows, size=SZ_S, bold_first_col=True, pad=int(0.12 * IN))
    y += th + GAP
    y = section(0, y, None, "Reading", [
        ("p", "Cloned speech is stored as MP3 and face swaps as MPEG-4, while every real clip is H.264 with AAC. "
              "A detector can learn the codec and never look at a face.", SZ_B),
    ])
    rows = [["FakeAVCeleb", "Train, test", n("{{neutral.fakeavceleb.kept|int}}")],
            ["LAV-DF", "Locked target", n("{{audit.lavdf.raw.clips|int}}")],
            ["DFDC", "Locked target", n("{{audit.dfdc.raw.clips|int}}")],
            ["HiFi-AVDF", "Locked target", n("{{audit.hifiavdf.raw.clips|int}}")]]
    d.add_text(s, colx(0), y, CW, int(0.9 * IN), [("h", "Data", 40)], "cap")
    y += int(1.0 * IN)
    _, th = d.add_table(s, colx(0), y, [int(CW * 0.36), int(CW * 0.36), CW - int(CW * 0.72)],
                        ["Corpus", "Role", "Clips"], rows, size=SZ_S, bold_first_col=True, pad=int(0.12 * IN))
    y += th + GAP
    y = section(0, y, None, "Protocol", [
        ("p", "Splits by identity. Three seeds. 95% intervals from 1,000 bootstrap resamples of people. Thresholds "
              "from validation only. Targets never train, select or calibrate anything.", SZ_S),
    ])
    section(0, y, PH - FOOT_H - y - GAP, "What we learned", [
        ("b", "Cleaning can put a shortcut back: audit the cleaned copies.", SZ_S),
        ("b", "Real-only training cannot learn file habits.", SZ_S),
        ("b", "AUC can hold while the threshold breaks.", SZ_S),
        ("b", "Write hypotheses first; report the one that fails.", SZ_S),
    ])

    # column 2: cleaning and the sync path
    y = BODY_TOP
    y = section(1, y, None, "3. Removing the cues", [
        ("p", "Version 1 put silence back through hidden AAC padding. Version 4 works on the audio samples, then "
              "encodes once.", SZ_B),
    ])
    steps = ["**Decode** 16 kHz mono", "**Level** -23 dBFS RMS", "**Cut** at first sound, exactly 3.00 s",
             "**Dither** at -66 dBFS", "**Encode** H.264 25 fps, AAC"]
    bh, bg_ = int(1.15 * IN), int(0.35 * IN)
    for i, (txt, fl) in enumerate(zip(steps, [F_GREY, F_BLUE, F_GREEN, F_ORANGE, F_RED])):
        d.dbox(s, colx(1), y, CW, bh, txt, fill=fl, size=SZ_S)
        if i < len(steps) - 1:
            d.darrow(s, colx(1) + CW // 2, y + bh, colx(1) + CW // 2, y + bh + bg_)
        y += bh + bg_
    y += GAP - bg_
    y = figure(1, y, "fig_h1.png", "Shortcut ceiling before and after neutralization", max_h=int(9 * IN))
    y = section(1, y, None, "H1: not met", [
        ("p", "The FakeAVCeleb ceiling fell from {{audit.fakeavceleb.raw.clip}} to "
              "**{{audit.fakeavceleb.neutral.clip}}**. The preregistered bar was 0.60. Video bitrate is the main cue "
              "left. {{neutral.fakeavceleb.refused|int}} short generated-speech clips are refused.", SZ_B),
    ])
    y = section(1, y, None, "4. Real-only lip-sync detector", [
        ("p", "Frozen AV-HuBERT encodes 88 px mouth crops and the audio separately. A small head learns whether a "
              "lips frame and an audio frame belong together, **from real clips only**, so it cannot learn a "
              "generator's file habits.", SZ_B),
        ("p", "Clip score: mean of  -log sigmoid(g)  over aligned frames and two 2 s windows.", SZ_S),
    ])
    rows = [[name, n(f"{{{{h2.{k}.estimate}}}} [{{{{h2.{k}.lower}}}}, {{{{h2.{k}.upper}}}}]"),
             "n/a" if k == "hifiavdf" else n(f"{{{{h2.{k}.holm}}}}")]
            for name, k in [("FakeAVCeleb, strict", "test-identity-strict"), ("DFDC", "dfdc"),
                            ("LAV-DF test", "lavdf-test"), ("HiFi-AVDF", "hifiavdf")]]
    y = section(1, y, None, "5. Fusion against visual-only (H2)", [
        ("p", "Logistic fusion of the three paths, fixed before any branch was trained. Mean gain in AUC over the "
              "visual path, three seeds; Holm over the first three rows.", SZ_S),
    ])
    _, th = d.add_table(s, colx(1), y, [int(CW * 0.36), int(CW * 0.44), CW - int(CW * 0.80)],
                        ["Target", "Gain [95%]", "Holm"], rows, size=SZ_S, bold_first_col=True, pad=int(0.08 * IN))
    y += th + int(0.3 * IN)
    d.add_text(s, colx(1), y, CW, int(0.8 * IN), T([("p", "**H2 holds: {{h2.success}}**", SZ_B)]), "h2_verdict")
    y += int(0.8 * IN) + GAP
    if PH - FOOT_H - y - GAP > int(2 * IN):      # the pipeline figure only where it fits
        figure(1, y, "fig_pipeline.png", "Pipeline from clip to score", max_h=PH - FOOT_H - y - GAP)

    # column 3: results
    y = BODY_TOP
    rows = []
    for part, label in [("test", "Test"), ("test-identity-strict", "Strict")]:
        for seed in (17, 29, 43):
            kk = f"sync.{part}.seed{seed}"
            rows.append([label if seed == 17 else "", str(seed),
                         n(f"{{{{{kk}.auc}}}} [{{{{{kk}.auc_low}}}}, {{{{{kk}.auc_high}}}}]"), n(f"{{{{{kk}.fpr}}}}")])
    d.add_text(s, colx(2), y, CW, int(0.9 * IN), [("h", "6. Results on neutral FakeAVCeleb", SZ_H)], "head")
    y += int(1.0 * IN)
    _, th = d.add_table(s, colx(2), y, [int(CW * 0.2), int(CW * 0.14), int(CW * 0.44), CW - int(CW * 0.78)],
                        ["Set", "Seed", "AUC [95%]", "FPR"], rows, size=SZ_S, bold_first_col=True, pad=int(0.06 * IN))
    y += th + int(0.3 * IN)
    d.add_text(s, colx(2), y, CW, int(1.4 * IN), T([("p", "Every interval clears the neutral ceiling "
                                                           "({{audit.fakeavceleb.neutral.clip}}). False alarms at the "
                                                           "validation threshold stay high.", SZ_S)]), "cap")
    y += int(1.6 * IN)
    y = figure(2, y, "fig_sync_methods.png", "Sync path AUC per manipulation method", max_h=int(7.5 * IN))
    y = section(2, y, None, "Scores follow the mechanism", [
        ("p", "Voice cloning is caught best, Wav2Lip next, and face swaps that keep the real lips least, in all "
              "three seeds.", SZ_B),
    ])
    y = section(2, y, None, "7. New corpora and false alarms", [
        ("p", "Frozen model on LAV-DF, DFDC and HiFi-AVDF: AUC {{sync.lavdf-test.seed17.auc}}, "
              "{{sync.dfdc.seed17.auc}} and {{sync.hifiavdf.seed17.auc}} (seed 17), against neutral ceilings of "
              "{{audit.lavdf.neutral.clip}}, {{audit.dfdc.neutral.clip}} and {{audit.hifiavdf.neutral.clip}}.", SZ_S),
    ])
    y = figure(2, y, "fig_cross_corpus.png", "Sync path AUC on each locked target",
               keys=("sync.hifiavdf.seed17.auc",), max_h=int(6 * IN))
    y = section(2, y, None, "False-alarm rules (H4)", [
        ("p", "200 real target clips set the threshold; no fake label is read.", SZ_S),
        ("p", "Conformal: the k-th smallest real score, k = ceil((n + 1)(1 - alpha)).", SZ_S),
        ("p", "Umbrella: the smallest rank with P(FPR > alpha) at most delta.", SZ_S),
        ("p", "Bar: realized FPR at most 0.07 on every target.", SZ_S),
    ])
    y = figure(2, y, "fig_h4.png", "Realized FPR and TPR of the three threshold rules",
               keys=("h4.hifiavdf-seed17.conformal_real_only.fpr",), max_h=int(6 * IN))
    section(2, y, PH - FOOT_H - y - GAP, "Conclusion", [
        ("p", "Benchmarks leak labels through their files. Compare every detector score with the shortcut ceiling. "
              "Hypotheses were committed before the runs, and every number has a receipt.", SZ_B),
    ])

    # footer band
    foot = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Emu(PH - FOOT_H), Emu(PW), Emu(FOOT_H))
    foot.fill.solid(); foot.fill.fore_color.rgb = rgb(RED); foot.line.fill.background()
    ft = d.add_text(s, M, PH - FOOT_H + int(0.3 * IN), PW - 2 * M, int(0.8 * IN),
                    [("p", "Electronics and Computer Engineering", SZ_S)], "footer_text")
    for r in ft.text_frame.paragraphs[0].runs:
        r.font.color.rgb = rgb("FFFFFF")
    if n.missing and not draft:
        print("missing numbers: " + ", ".join(sorted(n.missing)))
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.prs.save(OUT)
    for w in d.warnings:
        print("layout:", w)
    print(f"poster -> {OUT.relative_to(ROOT)}; {len(n.missing)} numbers pending")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="mark missing numbers [pending] instead of stopping")
    raise SystemExit(build(parser.parse_args().draft))
