"""Build the final-year presentation from results/numbers.json on the college slide master.

    .venv/Scripts/python.exe deliverables/slides/build_deck.py            final build: stops on a missing number
    .venv/Scripts/python.exe deliverables/slides/build_deck.py --draft    marks missing numbers [pending]

Every number on a slide is a {{key}} placeholder, filled the same way as the paper and the report
(scripts/fill_numbers.py). The deck is exactly the 13 slides of the college's exhibition structure:
title, introduction, need, literature review, three on methodology, two on results, discussion and
future scope, conclusions, awards and publications, bibliography. Output: deliverables/build/Deepfake_Detection_Presentation.pptx.
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
    F_FACE,
    F_GREY,
    F_RED,
    F_SYNC,
    F_VOICE,
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
from slide_common import BOT, GREY, TOP, L, W, col_x, span_w  # noqa: E402

TEMPLATE = HERE / "college-template.pptx"
FIG = ROOT / "deliverables" / "figures"
BRAND = ROOT / "assets" / "branding"
OUT = ROOT / "deliverables" / "build" / "Deepfake_Detection_Presentation.pptx"

TITLE = "Multimodal Audio-Visual Deepfake Detection"
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

    def s_title():
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
            "This project trains an audio-visual deepfake detector from three models: one for the face, one for the "
            "voice, and one that checks whether the lips match the words, which learns from real videos only. We train "
            "and test them on files cleaned the way an upload cleans them, so they must learn the fake and not the file, "
            "and we test them on three datasets they never saw. "
            "Introduce the three members and the guide, then move to the problem.")

    # Design (researched 11-18): one message per slide, as a sentence headline (assertion-evidence, Alley,
    # Penn State); one visual as its evidence; on-slide words cut to short labels, since printed text that
    # repeats the speaker adds nothing (Mayer, redundancy and coherence). Details live in the speaker notes.

    def headline(s, text):
        """The slide's one message, under the title."""
        d.add_text(s, L, TOP, W, 430000, [("p", "**" + n(text) + "**", 22)], "text_headline")

    def stat(s, x, y, w, h, big, label, fill=F_GREY, name="stat"):
        """A large number with a short label below it."""
        card(s, x, y, w, h, [("p", "**" + big + "**", 36), ("s", label, 16)], fill=fill, name=name,
             anchor=MSO_ANCHOR.MIDDLE)

    def s_intro():
        s = slide("Introduction",
                  "A deepfake is a video in which a person's face, voice or both have been made or changed by a "
                  "generative model. A face can be swapped or its mouth redrawn, a voice can be cloned, and in the hardest "
                  "cases both are faked so that the lips still follow the words. No single model sees all of these, so we "
                  "train three: a face model, a voice model, and a lip-sync model that asks whether the lips match the "
                  "words. Benchmark files give the label away through their codecs, so every model is trained and tested "
                  "on files cleaned the way an upload would clean them.")
        headline(s, "Three trained models: a fake face, a fake voice, and lips that do not match")
        box_h = 2500000
        for k, (head, text, fill) in enumerate([("Face model", "swapped faces, redrawn mouths", F_FACE),
                                                ("Voice model", "cloned or synthesised speech", F_VOICE),
                                                ("Lip-sync model", "lips that do not match the words", F_SYNC)]):
            d.dbox(s, col_x(4 * k), Y0, span_w(4), box_h, f"**{head}**\n{text}", fill=fill, size=24)
        card(s, col_x(0), Y0 + box_h + 300000, W, BOT - Y0 - box_h - 300000,
             [("p", "**Our aim:** a detector that learns the fake, not the file, and still works on data it never saw", 26)],
             anchor=MSO_ANCHOR.MIDDLE, name="card_question")

    def s_need():
        from pptx.chart.data import CategoryChartData
        from pptx.dml.color import RGBColor
        from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
        from pptx.util import Pt

        s = slide("Need for the project",
                  "Deepfakes now move real money. In January 2024 a finance worker at Arup in Hong Kong paid out about "
                  "25.6 million US dollars after a video call in which every other face and voice was generated. The FBI's "
                  "2025 Internet Crime Report counts almost 900 million dollars of AI-related scam losses in the United "
                  "States in one year, and Deloitte expects fraud losses enabled by generative AI there to grow from 12.3 "
                  "to 40 billion dollars between 2023 and 2027. India has required platforms to label synthetic media since "
                  "February 2026. Detectors are needed, and so are tests that show whether a detector really works.")
        headline(s, "Deepfakes now cost real money, and the law has begun to respond")
        data = CategoryChartData()
        data.categories = ["2023", "2027 (projected)"]
        data.add_series("US$ billion", (12.3, 40))
        cw = span_w(7)
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Emu(col_x(0)), Emu(Y0), Emu(cw),
                                Emu(BOT - Y0 - 380000), data)
        gf.name = d.name("chart_losses")
        chart = gf.chart
        chart.has_legend = False
        chart.has_title = True
        chart.chart_title.text_frame.text = "US fraud losses from generative AI, US$ billion"
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = C.FONT
        chart.font.name = C.FONT
        plot = chart.plots[0]
        plot.gap_width = 80
        plot.has_data_labels = True
        plot.data_labels.font.size = Pt(20)
        plot.data_labels.font.bold = True
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        plot.series[0].format.fill.solid()
        plot.series[0].format.fill.fore_color.rgb = RGBColor.from_string(C.RED)
        chart.value_axis.visible = False
        chart.value_axis.has_major_gridlines = False
        chart.category_axis.tick_labels.font.size = Pt(16)
        d.add_text(s, col_x(0), BOT - 360000, cw, 340000, [("s", "Source: Deloitte Center for Financial Services, 2024", 12)],
                   "chart_source")
        tile_h = (BOT - Y0 - 2 * 150000) // 3
        for k, (big, label) in enumerate([("US$25.6 million", "lost in one deepfake video call (Arup, Hong Kong, 2024)"),
                                          ("US$893 million", "AI-related scam losses in the US in 2025 (FBI)"),
                                          ("February 2026", "India requires labels on synthetic media (MeitY)")]):
            card(s, col_x(7), Y0 + k * (tile_h + 150000), span_w(5), tile_h,
                 [("p", "**" + big + "**", 26), ("s", label, 16)], name=f"card_need{k}")

    def s_literature():
        s = slide("Literature review",
                  "AVFF and HAVIC fuse audio and video and train on real and fake clips. AVH-Align, which our lip-sync "
                  "model follows, learns from real clips only and trims the leading silence at test time. DeepFake Doctor "
                  "found the silence clue and proposed splitting by person. None of them removes the file clues from both "
                  "training and test data, and none reports what a classifier gets from the file details alone. Check each "
                  "cell against the papers before quoting it.")
        headline(s, "Earlier detectors score near-perfect, but none removed the file clues first")
        rows = [["AVFF (CVPR 2024)", "No", "No", "No", "Yes"],
                ["AVH-Align (CVPR 2025)", "Yes", "No", "Partly", "Yes"],
                ["DeepFake Doctor (2025)", "No", "Yes", "Partly", "Partly"],
                ["HAVIC (CVPR Findings 2026)", "No", "No", "No", "Yes"],
                ["**This project**", "**Yes**", "**Yes**", "**Yes**", "**Yes**"]]
        d.add_table(s, L, Y0, [3300000, 1850000, 1850000, 1850000, W - 8850000],
                    ["Work", "Learns from real clips only", "Split by person", "Removes file clues",
                     "Tested on unseen data"], rows, size=20, bold_first_col=True, fill=BOT - Y0 - 420000)
        d.add_text(s, L, BOT - 380000, W, 360000,
                   [("s", "Partly: only the silence at the start of the audio, or only at test time. Ours: the lip-sync model "
                          "learns from real clips only.", 14)], "lit_note")

    def s_hardware():
        s = slide("Methodology: hardware and software",
                  "All training and the live demo run on one desktop PC. The GPU trains the models; the website's detector "
                  "also runs on the processor alone, and we checked that it then gives the same scores. The models are "
                  "deliberately modest and frozen where possible, because the question is what the data allow, not how "
                  "large a model can be. Three models at three seeds each are trained on the GPU; the website checks a "
                  "video in 3-second parts, each taking about {{web.seconds_per_part|int}} seconds on the processor.")
        headline(s, "One desktop PC trains the models and runs the live demo")
        rows_hw = [("GPU", "NVIDIA RTX 5070 Ti, 16 GB"), ("Processor", "AMD Ryzen 5 5600X, 6 cores"),
                   ("Memory", "32 GB"), ("System", "Windows 11")]
        rows_sw = [("Language", "Python 3.13, PyTorch 2.12"), ("Media", "FFmpeg, MediaPipe"),
                   ("Models", "EfficientNet-B0, Wav2Vec2, AV-HuBERT"), ("Website", "FastAPI server, HTML and JavaScript")]
        col_h = BOT - Y0 - 1500000
        for k, (head, rows, fill) in enumerate([("Hardware", rows_hw, F_GREY), ("Software", rows_sw, F_GREY)]):
            paras = [("h", head)] + [("p", f"**{a}:** {b}", 18) for a, b in rows]
            card(s, col_x(6 * k), Y0, span_w(6), col_h, paras, fill=fill, name=f"card_{head.lower()}")
        y = Y0 + col_h + 200000
        stat(s, col_x(0), y, span_w(6), BOT - y, "3 models x 3 seeds", "trained on the GPU", name="stat_gpu")
        stat(s, col_x(6), y, span_w(6), BOT - y, n("about {{web.seconds_per_part|int}} s"), "per 3-second part on the processor",
             name="stat_cpu")

    def s_flowchart():
        s = slide("Methodology: flowchart",
                  "The method has five stages. A video comes in and is first cleaned the way an upload would clean it: one "
                  "video codec, three seconds from the first sound, one loudness level and a little noise, so that how the "
                  "file was made cannot give the answer away. Three trained models then score it: the face, the voice, and "
                  "whether the lips match the words. The lip-sync model learned from real videos only. Each score is placed among "
                  "the scores of real videos to give the answer. The file audit at the top is not part of the detector: it "
                  "measures how much the file alone reveals, before and after cleaning.")
        headline(s, "Each clip is cleaned like an upload, then scored by three trained models")
        d.pic_fit(s, FIG / "fig_pipeline.png", L, Y0, W, BOT - Y0,
                  "The method in five stages: input, cleaning, face, voice and lip-sync models, comparison with real "
                  "videos, and the answer, with the file audit above")

    def s_experiment():
        s = slide("Methodology: experimental details",
                  "Every model learned from one dataset, FakeAVCeleb, split by person so that nobody appears in both "
                  "training and test. Three other datasets were locked away and used only for testing: LAV-DF, where "
                  "single words are edited; DFDC, with face swaps; and HiFi-AVDF, made by six commercial video generators. "
                  "The tests and their pass marks were written down before any run, as preregistration P-001. Each model "
                  "was trained three times with different random seeds.")
        headline(s, "Trained on one dataset, then tested on three it never saw")
        lh = 2600000
        d.dbox(s, col_x(0), Y0 + 150000, span_w(4), lh,
               n("**Train and test**\nFakeAVCeleb\n{{neutral.fakeavceleb.kept|int}} clips\nsplit by person"), fill=F_GREY, size=20)
        targets = [("LAV-DF", "{{audit.lavdf.neutral.clips|int}} clips, words edited"),
                   ("DFDC", "{{audit.dfdc.raw.clips|int}} clips, face swaps"),
                   ("HiFi-AVDF", "{{audit.hifiavdf.raw.clips|int}} clips, six AI video tools")]
        th = (lh - 2 * 150000) // 3
        for k, (name, text) in enumerate(targets):
            ty = Y0 + 150000 + k * (th + 150000)
            d.dbox(s, col_x(6), ty, span_w(6), th, n(f"**{name}**  {text}"), fill=F_GREY, size=18, align=PP_ALIGN.LEFT)
            d.darrow(s, col_x(0) + span_w(4), Y0 + 150000 + lh // 2, col_x(6), ty + th // 2)
        y = Y0 + 150000 + lh + 250000
        for k, (big, label) in enumerate([("P-001", "tests fixed before the runs"), ("3 seeds", "per model"),
                                          ("ROC-AUC", "0.5 is guessing, 1.0 is perfect")]):
            stat(s, col_x(4 * k), y, span_w(4), BOT - y, big, label, name=f"stat_exp{k}")

    def s_results_files():
        from pptx.chart.data import CategoryChartData
        from pptx.dml.color import RGBColor
        from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
        from pptx.util import Pt

        s = slide("Results: the detector at home",
                  "On FakeAVCeleb test clips of people the models never saw in training, the face model alone reaches "
                  "{{h2.test-identity-strict.auc.visual.max}} AUC on the real-or-fake label and the three models combined reach {{h2.test-identity-strict.auc.fusion.min}} to {{h2.test-identity-strict.auc.fusion.max}}. The "
                  "lip-sync model reaches {{sync.test.seed17.auc}} to {{sync.test.seed29.auc}} without ever seeing a fake, catches cloned voices best and plain "
                  "face swaps least, as its design predicts, and reacts to audio moved by one frame. The files after "
                  "cleaning give away {{audit.fakeavceleb.neutral.clip}} on their own, so every model's score is real learning, above that line.")
        headline(s, n("Combined, the three models reach up to {{h2.test-identity-strict.auc.fusion.max}} on people never seen in training"))
        keys = [("Face", "visual"), ("Voice", "audio"), ("Lip sync", "sync"), ("Combined", "fusion")]
        data = CategoryChartData()
        data.categories = [k for k, _ in keys]
        data.add_series("AUC", [float(n("{{h2.test-identity-strict.seed17.auc.%s}}" % key)) for _, key in keys])
        cw = span_w(8)
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Emu(col_x(0)), Emu(Y0), Emu(cw), Emu(BOT - Y0), data)
        gf.name = d.name("chart_home")
        chart = gf.chart
        chart.has_legend = False
        chart.has_title = True
        chart.chart_title.text_frame.text = "AUC on the identity-strict test, real or fake (seed 17)"
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = C.FONT
        chart.font.name = C.FONT
        plot = chart.plots[0]
        plot.gap_width = 70
        plot.vary_by_categories = False
        plot.has_data_labels = True
        plot.data_labels.font.size = Pt(20)
        plot.data_labels.font.bold = True
        plot.data_labels.number_format = "0.000"
        plot.data_labels.number_format_is_linked = False
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        colours = ["2A78D6", "EB6834", "1BAF7A", "3D3C39"]      # the figures' face, voice, lip-sync colours; grey for both
        for point, colour in zip(plot.series[0].points, colours):
            point.format.fill.solid()
            point.format.fill.fore_color.rgb = RGBColor.from_string(colour)
        chart.value_axis.visible = False
        chart.value_axis.has_major_gridlines = False
        chart.value_axis.minimum_scale = 0
        chart.value_axis.maximum_scale = 1.1
        chart.category_axis.tick_labels.font.size = Pt(18)
        tile_h = (BOT - Y0 - 2 * 150000) // 3
        stat(s, col_x(8), Y0, span_w(4), tile_h,
             n("{{h2.test-identity-strict.auc.fusion.min}} to {{h2.test-identity-strict.auc.fusion.max}}"),
             "three models combined, three seeds", name="stat_combined")
        stat(s, col_x(8), Y0 + tile_h + 150000, span_w(4), tile_h, n("{{sync.test.seed17.auc}} to {{sync.test.seed29.auc}}"),
             "lip-sync model, never saw a fake", fill=F_SYNC, name="stat_home")
        stat(s, col_x(8), Y0 + 2 * (tile_h + 150000), span_w(4), tile_h, n("{{audit.fakeavceleb.neutral.clip}}"),
             "cleaned files alone: the line to beat", name="stat_clean")

    def s_results_detection():
        s = slide("Results: new data",
                  "On three datasets the models never saw, the combined detector falls to {{h2.lavdf-window.auc.fusion.min}} to {{h2.dfdc.auc.fusion.max}}, and the best single "
                  "model changes with the dataset, so no fixed combination transfers. The lip-sync model fails on Sora 2, "
                  "which makes lips and voice together. A threshold carried over from training flags about half of the "
                  "real HiFi-AVDF clips; setting it from 200 real clips of that dataset brings false alarms back to about "
                  "5 in 100, but few fakes are then caught. Before cleaning, file details alone gave {{audit.fakeavceleb.raw.clip}} on FakeAVCeleb: "
                  "that is why we train on cleaned files.")
        headline(s, "On three datasets it never saw, the detector weakens; real clips fix its false alarms")
        figure_or_pending(s, "fig_cross_corpus.png", col_x(0), Y0, span_w(8), BOT - Y0, "Lip-sync model AUC per dataset")
        tile_h = (BOT - Y0 - 2 * 150000) // 3
        stat(s, col_x(8), Y0, span_w(4), tile_h,
             n("{{h2.lavdf-window.auc.fusion.min}} to {{h2.dfdc.auc.fusion.max}}"), "combined, three unseen datasets",
             name="stat_unseen")
        stat(s, col_x(8), Y0 + tile_h + 150000, span_w(4), tile_h,
             n("{{h4.hifiavdf-seed17.source_validation.fpr|pct}}, then {{h4.hifiavdf-seed17.conformal_real_only.fpr|pct}}"),
             "real clips flagged, old then new line", name="stat_fpr")
        stat(s, col_x(8), Y0 + 2 * (tile_h + 150000), span_w(4), tile_h, n("{{audit.fakeavceleb.raw.clip}}"),
             "raw file details alone", fill=F_RED, name="stat_files")

    def s_discussion():
        s = slide("Discussion and future scope",
                  "At home the three models together work well on people they never saw, above what the cleaned files "
                  "give away. On new data a detector trained on one dataset transfers weakly, and modern generators that "
                  "make picture and sound together are hardest. The next stage trains on several "
                  "datasets with one held out each time, says which stream was faked in both directions, runs more "
                  "published detectors under our cleaning, and puts the website online.")
        headline(s, "Strong at home, weaker on new data; more varied training comes next")
        groups = [("Discussion", F_GREY, ["Three models combined beat the face model alone",
                                          "One training dataset does not transfer far",
                                          "Modern generators keep lips and voice in sync"]),
                  ("Future scope", F_GREY, ["Train on several datasets, one held out",
                                             "Say which stream was faked, both ways",
                                             "Test more published detectors after cleaning",
                                             "Put the website online"])]
        for k, (head, fill, items) in enumerate(groups):
            x = col_x(6 * k)
            d.add_text(s, x, Y0, span_w(6), 420000, [("h", head)], f"head_{k}")
            ih = (BOT - Y0 - 500000 - (len(items) - 1) * 150000) // len(items)
            for j, item in enumerate(items):
                d.dbox(s, x, Y0 + 500000 + j * (ih + 150000), span_w(6), ih, item, fill=fill, size=20, align=PP_ALIGN.LEFT)

    def s_conclusions():
        s = slide("Conclusions",
                  "First, three trained models together detect fakes well on people they never saw, once the training "
                  "files are cleaned so they cannot give the label away. Second, a lip-sync model that learns from real "
                  "clips only adds evidence the face model cannot see, and moving the audio of a real clip raises its "
                  "score, as its design predicts. Third, on new data the detector weakens, and real clips from the new "
                  "source fix its false alarms but not its misses.")
        headline(s, "Three trained models detect fakes at home; new data needs more varied training")
        cw = span_w(4)
        for k, (big, label, fill) in enumerate([
                (n("{{h2.test-identity-strict.auc.fusion.min}} to {{h2.test-identity-strict.auc.fusion.max}}"), "three models combined, on people never seen in training", F_FACE),
                (n("{{sync.test.seed17.auc}} to {{sync.test.seed29.auc}}"), "lip-sync model after cleaning: it reads real lip and voice agreement", F_SYNC),
                (n("{{h4.hifiavdf-seed17.conformal_real_only.fpr|pct}}"), "false alarms with a threshold from real new clips, but few fakes caught", F_GREY)]):
            card(s, col_x(4 * k), Y0 + 300000, cw, 3000000,
                 [("p", "**" + big + "**", 40), ("p", label, 20)], fill=fill, name=f"card_conc{k}",
                 anchor=MSO_ANCHOR.MIDDLE)

    def s_awards():
        s = slide("Awards, publications and copyright",
                  "We have no awards, publications or registered copyright for this project. The research paper written "
                  "for it is submitted with the report. We plan to submit the file-clue audit and the cleaning method to a "
                  "conference after more published detectors have been tested under it.")
        headline(s, "None so far; a conference paper is planned")
        for k, (head, text) in enumerate([("Awards", "None"), ("Publications", "None yet\nPaper planned"),
                                          ("Copyright", "None registered")]):
            d.dbox(s, col_x(4 * k), Y0 + 300000, span_w(4), 2600000, f"**{head}**\n{text}", fill=F_GREY, size=24)

    def s_bibliography():
        s = slide("Bibliography",
                  "These are the main sources the project builds on. The datasets come first, then the detectors we follow "
                  "or compare with, then the work on shortcuts and false-alarm control. The report lists all 35 references "
                  "in full.",
                  "Main references; the full list is in the report.")
        refs = ["H. Khalid et al., FakeAVCeleb, NeurIPS Datasets and Benchmarks, 2021.",
                "Z. Cai et al., LAV-DF, DICTA, 2022.",
                "B. Dolhansky et al., The DeepFake Detection Challenge dataset, arXiv:2006.07397, 2020.",
                "J. Peng et al., HAVIC and HiFi-AVDF, CVPR Findings, 2026.",
                "B. Shi et al., AV-HuBERT, ICLR, 2022.",
                "S. Smeu et al., AVH-Align: circumventing shortcuts in audio-visual deepfake datasets, CVPR, 2025.",
                "T. Oorloff et al., AVFF: audio-visual feature fusion for deepfake detection, CVPR, 2024.",
                "M. Klemt et al., DeepFake Doctor, arXiv:2506.05851, 2025.",
                "N. M. Mueller et al., Speech is silver, silence is golden, ASVspoof Workshop, 2021.",
                "R. Geirhos et al., Shortcut learning in deep neural networks, Nature Machine Intelligence, 2020.",
                "S. Bates et al., Testing for outliers with conformal p-values, Annals of Statistics, 2023.",
                "X. Tong et al., Neyman-Pearson classification algorithms, Science Advances, 2018.",
                "CNN Business, British engineering giant Arup revealed as $25 million deepfake scam victim, 16 May 2024.",
                "FBI Internet Crime Complaint Center, Internet Crime Report 2025, ic3.gov.",
                "Deloitte Center for Financial Services, Generative AI is expected to magnify the risk of deepfakes and other fraud in banking, 2024.",
                "MeitY, IT (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules, 2026, notified 10 February 2026."]
        d.add_text(s, L, Y0, W, H0, [("s", f"[{k + 1}]  {r}", 12) for k, r in enumerate(refs)], "bibliography")

    # The college's 13 slides, in its order, and nothing else.
    for make in (s_title, s_intro, s_need, s_literature, s_hardware, s_flowchart, s_experiment,
                 s_results_files, s_results_detection, s_discussion, s_conclusions, s_awards, s_bibliography):
        make()

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
