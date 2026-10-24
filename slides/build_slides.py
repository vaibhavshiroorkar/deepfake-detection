"""Builds the presentation skeleton (.pptx) from the supplied college deck.

Source file (read only, never modified): Document Template/
"DeepFake_Detection_System_Presentation (2).pptx". From it we keep the slide
master (red bars, Somaiya Vidyavihar University logo, Somaiya Trust logo),
the layouts, the slide size (16:9, 12192000 x 6858000 EMU), the theme and the
title-slide design (two campus photographs). ALL old slide content is
removed: text, tables, diagrams, timeline. Only identity details are reused.

Run (from the repository root):
    uv run --no-project --with python-pptx python slides/build_slides.py

Rules in force (CLAUDE.md): no result is written unless it comes from a logged
experiment E-xxx; result slides carry the tag "RESULT: TO BE FILLED"; only
papers marked READ in docs/RESEARCH.md are cited; model, dataset, framework
and metric choices are open (docs/DECISIONS.md O-001 to O-011); no em-dashes
or en-dashes anywhere.
"""
import copy
import os

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "Document Template", "DeepFake_Detection_System_Presentation (2).pptx")
OUT = os.path.join(HERE, "DeepFake_Detection_Presentation_draft.pptx")

DARK_RED = RGBColor(0xA4, 0x25, 0x30)   # master bar colour
BRIGHT_RED = RGBColor(0xD9, 0x22, 0x2A)  # master bar colour (right block)
GREY = RGBColor(0x3F, 0x3F, 0x3F)        # footer and slide number text
BAND = RGBColor(0xF5, 0xF5, 0xF5)        # table banding in the old deck
FONT = "Times New Roman"

# Identity details, as in the supplied deck and report (CONFIRM items are
# listed as questions in docs/_agents/A.md).
DEPARTMENT = "Department of Electronics and Computer Engineering"  # CONFIRM
GUIDE = "Dr. Tejashree Phatak"
MEMBERS = [("Vaibhav Shiroorkar", "16014024805"), ("Adithya Jithesh", "16014024803"),
           ("Shivansh Gupta", "16014023052")]  # order as in the supplied deck
FOOTER = "LY Project"  # as in the supplied deck, CONFIRM

prs = Presentation(SRC)
layouts = {l.name: l for l in prs.slide_layouts}
TITLE_ONLY = layouts["TITLE_ONLY"]

# --- keep prototypes of the old content slide (title, footer, slide number)
old2 = prs.slides[1]
proto = {}
for shp in old2.shapes:
    if shp.is_placeholder:
        proto[str(shp.placeholder_format.type)] = copy.deepcopy(shp._element)
proto_title = [v for k, v in proto.items() if "TITLE" in k][0]
proto_ftr = [v for k, v in proto.items() if "FOOTER" in k][0]
proto_num = [v for k, v in proto.items() if "SLIDE_NUMBER" in k][0]

# --- title slide: edit slide 1 in place (keeps the two campus photographs)
s1 = prs.slides[0]


def set_runs(shape, lines, size=None, bold=None):
    """Replace the paragraphs of a shape with `lines`, copying the first run format."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]._p
    first_r = p0.find(qn("a:r"))
    rpr = copy.deepcopy(first_r.find(qn("a:rPr"))) if first_r is not None and first_r.find(qn("a:rPr")) is not None else None
    ppr = p0.find(qn("a:pPr"))
    ppr = copy.deepcopy(ppr) if ppr is not None else None
    txbody = tf._txBody
    for p in list(txbody.findall(qn("a:p"))):
        txbody.remove(p)
    for line in lines:
        p = etree.SubElement(txbody, qn("a:p"))
        if ppr is not None:
            p.append(copy.deepcopy(ppr))
        parts = line if isinstance(line, list) else [line]
        for i, part in enumerate(parts):
            if i > 0:
                br = etree.SubElement(p, qn("a:br"))
                if rpr is not None:
                    br.append(copy.deepcopy(rpr))
            r = etree.SubElement(p, qn("a:r"))
            if rpr is not None:
                rp = copy.deepcopy(rpr)
                for a in ("err",):
                    if a in rp.attrib:
                        del rp.attrib[a]
                r.append(rp)
            t = etree.SubElement(r, qn("a:t"))
            t.text = part


for shp in s1.shapes:
    if shp.name.endswith(";80;p2"):      # title
        set_runs(shp, [["Multimodal Deepfake Detection", "for Images, Video and Audio"]])
    elif shp.name.endswith(";81;p2"):    # members
        set_runs(shp, ["Group Members : "] + ["%s (%s)" % m for m in MEMBERS])
    elif shp.name.endswith(";87;p2"):    # department
        set_runs(shp, [DEPARTMENT])
    elif shp.name.endswith(";88;p2"):    # guide
        set_runs(shp, ["Guide : " + GUIDE])
s1.notes_slide.notes_text_frame.text = (
    "Title slide. Introduce the project: a multimodal deepfake detection system for images, videos and the audio inside videos, "
    "built from zero in this repository. Introduce the group and the guide. State the shape of the talk: the problem and scope, "
    "the plan and evaluation protocol, a minimal baseline (MVP), seven iterations that each start from a written hypothesis, "
    "the final comparison, limitations and conclusion. State at once that the working title is a proposal and that every result "
    "slide is marked to be filled until the experiment that supports it has run.")

# --- drop the old content slides (2 to 9): their text, tables and diagrams must not survive
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:]:
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)


# ---------------------------------------------------------------------------
def run_fmt(run, size, bold=False, italic=False, color=None):
    run.font.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:cs", "a:ea"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def set_bullet(p, char="•"):
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", "285750")
    ppr.set("indent", "-285750")
    for t in ("a:buNone", "a:buChar", "a:buFont"):
        for e in ppr.findall(qn(t)):
            ppr.remove(e)
    bf = etree.SubElement(ppr, qn("a:buFont"))
    bf.set("typeface", "Arial")
    bc = etree.SubElement(ppr, qn("a:buChar"))
    bc.set("char", char)


def add_body(slide, blocks, x=737846, y=950000, w=10716308, h=4400000, size=16):
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for kind, text in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if kind == "h":
            p.space_before = Pt(8)
            r = p.add_run(); r.text = text; run_fmt(r, size, bold=True)
        elif kind == "p":
            r = p.add_run(); r.text = text; run_fmt(r, size)
        elif kind == "b":
            set_bullet(p)
            r = p.add_run(); r.text = text; run_fmt(r, size)
        elif kind == "ph":   # placeholder: not yet decided or measured
            r = p.add_run(); r.text = text; run_fmt(r, size, italic=True, color=DARK_RED)
        elif kind == "bph":
            set_bullet(p)
            r = p.add_run(); r.text = text; run_fmt(r, size, italic=True, color=DARK_RED)
    return tb


def add_tag(slide, text="RESULT: TO BE FILLED"):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(9250000), Emu(5400000), Emu(2250000), Emu(380000))
    shp.fill.background()
    shp.line.color.rgb = BRIGHT_RED
    shp.line.width = Pt(2)
    tf = shp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; run_fmt(r, 12, bold=True, color=BRIGHT_RED)
    shp.name = "Tag to be filled"


def add_table(slide, header, rows, widths, x=737846, y=1000000, size=12, row_h=640000):
    gt = slide.shapes.add_table(len(rows) + 1, len(header), Emu(x), Emu(y), Emu(sum(widths)), Emu(row_h * (len(rows) + 1)))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # no style, no grid
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    for i, wd in enumerate(widths):
        tbl.columns[i].width = Emu(wd)

    def fill(cell, text, bold=False, italic=False, color=None, bg=None, sz=size):
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        r = p.add_run(); r.text = text; run_fmt(r, sz, bold, italic, color)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        if bg is not None:
            cell.fill.solid(); cell.fill.fore_color.rgb = bg

    for i, h in enumerate(header):
        fill(tbl.cell(0, i), h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), bg=DARK_RED)
    for ri, row in enumerate(rows, 1):
        for ci, v in enumerate(row):
            ph_ = v.startswith("[")
            fill(tbl.cell(ri, ci), v, italic=ph_, color=DARK_RED if ph_ else None, bg=BAND if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    return gt


_slide_no = [1]


def new_slide(title, notes):
    _slide_no[0] += 1
    s = prs.slides.add_slide(TITLE_ONLY)
    spTree = s.shapes._spTree
    for shp in list(s.shapes):           # remove auto-created title placeholder
        spTree.remove(shp._element)
    t = copy.deepcopy(proto_title)
    off = t.find(".//" + qn("a:off")); ext = t.find(".//" + qn("a:ext"))
    off.set("x", "2650000"); ext.set("cx", "8300000")  # keep clear of the two logos
    f = copy.deepcopy(proto_ftr)
    n = copy.deepcopy(proto_num)
    for i, el in enumerate((t, f, n)):
        el.find(qn("p:nvSpPr")).find(qn("p:cNvPr")).set("id", str(10 + i))
    # title text and size
    size = 3200 if len(title) <= 34 else (2800 if len(title) <= 42 else 2400)
    runs = t.findall(".//" + qn("a:r"))
    runs[0].find(qn("a:t")).text = title
    runs[0].find(qn("a:rPr")).set("sz", str(size))
    for e in t.findall(".//" + qn("a:endParaRPr")):
        e.set("sz", str(size))
    for tt in n.iter(qn("a:t")):
        tt.text = str(_slide_no[0])
    for el in (t, f, n):
        spTree.append(el)
    s.notes_slide.notes_text_frame.text = notes
    return s


# ---------------------------------------------------------------------------
# Slides. Narrative as titles; speaker notes carry what to say and where the
# evidence lives. No numbers of ours appear anywhere: there are no results.
# ---------------------------------------------------------------------------
TODO = "[to be filled from E-xxx]"

# 2 Problem
s = new_slide("The problem: real or manipulated?",
    "Problem. We want to tell real media from manipulated media for three kinds of input: images, videos, and the audio inside videos. "
    "Why it is hard: published detectors are measured under different protocols, so their reported numbers cannot be read side by side. "
    "In one benchmark that used one recipe for all detectors, scores within the training dataset were much higher than scores on other datasets (Yan et al., 2023). "
    "That is that paper's number under its own protocol, not our result. The same gap shows up for Celeb-DF and for the DFDC challenge (Li et al., 2020; Dolhansky et al., 2020). "
    "The aim of this project is therefore a documented path, with our own protocol, not a single headline number. "
    "Evidence in the repo: docs/RESEARCH.md, notes that bear on decisions, items 1 to 3. Open decision: the exact output and labels, O-001.")
add_body(s, [
    ("h", "What is being detected"),
    ("p", "Images, videos, and the audio inside videos that may be manipulated or synthesised: face swaps, talking-face synthesis, cloned voices."),
    ("h", "Why it is hard"),
    ("b", "Published detectors are measured under different protocols, so their numbers do not line up (Yan et al., 2023)."),
    ("b", "A score on the data a detector was trained on says little about new data (Li et al., 2020; Dolhansky et al., 2020)."),
    ("b", "A headline number can hide seed variance (Jung et al., 2021 report spread across seeds)."),
    ("h", "Problem statement"),
    ("p", "Given an input x, produce a score s(x) in [0, 1] for 'manipulated', with a stated uncertainty."),
    ("ph", "[TBD: exact output and labels, open decision O-001]"),
], size=20)

# 3 Scope
s = new_slide("Scope and claims we will not make",
    "Scope. The user decided that inputs are images, videos and audio (D-002) and that success means results comparable with published detectors on a protocol we define and run ourselves, not the best model in the world (D-005). "
    "How each input type is handled is still open (O-002). Claims discipline: we do not say real-time, state of the art, or better than X unless a head-to-head run on the same protocol supports it. "
    "A detector trained on one kind of fake makes no claim about other kinds without a test on them. Compute is local: one GPU with 12 GB, measured as ENV-DESKTOP-GPU (docs/EXPERIMENTS.md).")
add_body(s, [
    ("b", "Inputs: images, videos, and the audio of videos (decision D-002)."),
    ("bph", "[TBD: how each input type is handled, open decision O-002]"),
    ("b", "Goal: results comparable with published detectors on a protocol we define and run ourselves, not the best model in the world (D-005)."),
    ("b", "No 'real-time', no 'state of the art', no 'better than X' without a head-to-head run on the same protocol."),
    ("b", "A detector trained on one kind of fake makes no claim about other kinds without a test on them."),
    ("b", "Compute: one local GPU with 12 GB of memory; every plan must fit inside it."),
], size=22)

# 4 Plan
s = new_slide("Plan: MVP first, then seven iterations",
    "Plan and method. Stage 0 fixes the problem statement, Stage 1 the data and leakage rules, Stage 2 the metrics, splits and seeds before any model exists. Stage 3 is the minimal working detector (MVP). "
    "Stage 4 runs other models in the same harness so the final choice has evidence. Stage 5 is iterations 1 to 7, each from a hypothesis, a success criterion and a failure criterion written before the run in docs/EXPERIMENTS.md. "
    "Stage 6 tests whether heart-rate evidence helps. Stage 7 assembles the paper, report and slides from the logs. Failed runs stay in the log. Source: docs/PLAN.md and D-004.")
add_body(s, [
    ("h", "Stages"),
    ("b", "0 Problem and success   1 Data and leakage rules   2 Metrics, splits, seeds"),
    ("b", "3 Simplest end-to-end MVP   4 Other models in the same harness"),
    ("b", "5 Iterations 1 to 7, each from a hypothesis   6 Heart-rate (rPPG) feasibility   7 Paper, report, slides"),
    ("h", "Every iteration"),
    ("b", "Hypothesis, success criterion and failure criterion are written before the run."),
    ("b", "Result is appended after the run; failed runs stay in the log."),
    ("b", "A frozen choice is never reopened because of test-set results."),
], size=20)

# 5 Protocol
s = new_slide("Protocol fixed before any model",
    "Protocol. Data is split once, by source identity: no person, video source or recording appears on both sides of a split, and each split is a file with an identifier and a hash. "
    "Tuning uses only train and validation. The test split is touched only for a frozen choice, and the evaluation script refuses test data unless the run is flagged final. "
    "Every candidate gets at least three seeds, and we report the mean, the standard deviation and a bootstrap interval that resamples identities or source videos, not frames. "
    "Candidates are compared in a paired way on the same units; if the interval of the difference includes zero, we prefer the simpler or cheaper candidate. Cost is measured next to accuracy. "
    "Metrics, threshold policy and calibration are still open (O-005). Source: CLAUDE.md rules 3 and 4, docs/DECISIONS.md comparison rules.")
add_body(s, [
    ("b", "Split once by source identity; each split stored as a file with a hash (policy: open decision O-004)."),
    ("b", "Tuning on train and validation only; test touched once, when the iteration is frozen."),
    ("b", "At least 3 seeds; mean, standard deviation, and a bootstrap interval over identities or source videos."),
    ("b", "Paired comparison; if the interval of the difference includes zero, prefer the simpler or cheaper candidate."),
    ("b", "Cost next to accuracy: parameters, FLOPs, peak VRAM, seconds per epoch, sensitivity to learning rate."),
    ("bph", "[TBD: metrics, threshold policy, calibration, open decision O-005; threshold chosen on validation data only]"),
], size=20)

# 6 Literature
s = new_slide("Literature: what is known and open",
    "Literature. Only papers that were read from the primary source are used, and the numbers inside them are their own, under their own protocols. "
    "Datasets: FaceForensics++, Celeb-DF and DFDC are the standard visual data (Rossler et al., 2019; Li et al., 2020; Dolhansky et al., 2020). "
    "Visual detectors: Face X-ray, DeepfakeBench and the universal fake-image detector suggest that a plain backbone or a frozen pretrained feature space can be a strong baseline under a shared recipe (Li et al., 2020; Yan et al., 2023; Ojha et al., 2024). "
    "Pulse evidence: older work reports high in-dataset accuracy, later work is mixed on modern fakes, which is why rPPG is a hypothesis to test (Ciftci et al., 2020; Hernandez-Ortega et al., 2020; Seibold et al., 2025; Harraq and Aldwairi, 2026; Lin et al., 2023). "
    "Audio and audio-visual: ASVspoof 2019, AASIST, FakeAVCeleb and audio-visual dissonance (Wang et al., 2020; Jung et al., 2021; Khalid et al., 2022; Chugh et al., 2021). Source: docs/RESEARCH.md.")
add_table(s, ["Area", "Papers read", "What it suggests", "Left open for us"], [
    ["Face datasets", "Rossler 2019; Li 2020; Dolhansky 2020", "Standard data; accuracy depends on compression and on unseen content", "Which sets and splits (O-003, O-004)"],
    ["Visual detectors", "Li 2020; Yan 2023; Ojha 2024", "Plain backbones can be close under one recipe; frozen features can generalise", "Backbone, fine-tuning (O-007, O-008)"],
    ["Pulse (rPPG)", "Ciftci 2020; Hernandez-Ortega 2020; Seibold 2025; Harraq 2026; Lin 2023", "Mixed evidence on modern fakes; pulse can be forged", "Feasibility test first (O-010)"],
    ["Audio, audio-visual", "Wang 2020; Jung 2021; Khalid 2022; Chugh 2021", "Seed spread is real; naive fusion can do poorly", "Fusion scheme (O-009)"],
], widths=[1900000, 3000000, 3516308, 2300000], size=14, row_h=800000)
tb = add_body(s, [("p", "Numbers inside these papers are theirs, under their own protocols. None is our result.")], y=5150000, h=500000, size=13)

# 7 MVP
s = new_slide("MVP (I-0): the simplest detector",
    "MVP. The criterion for the MVP is written before any model is chosen: it runs from the raw input file to a score and a metric under the Stage 2 protocol, needs no tuning to produce a number, trains within minutes on the measured machine, and is simple enough to explain (docs/DECISIONS.md, O-006). "
    "Options compared for the MVP: a small network from scratch, a frozen pretrained feature extractor with a linear classifier, a fine-tuned pretrained backbone, and for audio a small spectrogram network. We have not chosen. "
    "Fill this slide from the experiment record of I-0: the hypothesis, what was built, and what happened, including whether it met the failure criterion.")
add_body(s, [
    ("h", "What it is"),
    ("ph", "[TBD: baseline design, open decision O-006]"),
    ("h", "Selection criterion (written before choosing)"),
    ("p", "Runs from raw file to score and metric under the Stage 2 protocol; needs no tuning to give a number; trains within minutes on this machine; simple to explain."),
    ("h", "What happened"),
    ("ph", "[to be filled from E-xxx, I-0]"),
], size=20)
add_tag(s)

# 8 to 14 iterations
for k in range(1, 8):
    s = new_slide("Iteration %d: tried, why, what happened" % k,
        "Iteration %d (slot I-%d). Say what the hypothesis was and that it was written before the run, what we changed relative to the previous best, why we changed it (tie it to the previous iteration or to a paper from docs/RESEARCH.md), "
        "and what happened, including if it failed the failure criterion. Compare with the previous iteration on the same data, split, budget and metric, with at least three seeds and an interval of the paired difference. "
        "If the interval includes zero, say that the evidence does not separate the two and that the simpler or cheaper one is kept. Fill every field from the experiment record E-xxx in docs/EXPERIMENTS.md. Nothing is pre-written here." % (k, k))
    add_body(s, [
        ("h", "Hypothesis, success criterion, failure criterion (written before the run)"),
        ("ph", "[to be filled from docs/EXPERIMENTS.md, E-xxx]"),
        ("h", "What we tried"),
        ("ph", "[to be filled from E-xxx]"),
        ("h", "Why"),
        ("ph", "[to be filled: tied to the previous iteration or to a paper in docs/RESEARCH.md]"),
        ("h", "What happened"),
        ("ph", "[to be filled from E-xxx, including failed runs]"),
    ], size=20)
    add_tag(s)

# 15 rPPG
s = new_slide("Heart-rate evidence: test first",
    "Heart-rate evidence. The user decided that rPPG is a candidate improvement and not an adopted component (D-006). The literature is mixed: older work reports high in-dataset accuracy (Ciftci et al., 2020; Hernandez-Ortega et al., 2020), "
    "a 2025 study found a plausible pulse in high-quality face swaps (Seibold et al., 2025), a 2026 preprint reported a reproduction that fell much lower on talking-face fakes under an identity-disjoint split (Harraq and Aldwairi, 2026), and pulse signals can be forged on purpose (Lin et al., 2023). "
    "So we test it in four steps, with success and failure thresholds written before the run (O-010). If the test fails, we report a negative result and do not include rPPG. Fill the result from the Stage 6 experiments.")
add_body(s, [
    ("h", "Four-step feasibility test, thresholds written before the run (O-010)"),
    ("b", "(a) Signal sanity: can a heart rate be recovered from our data at all?"),
    ("b", "(b) Separation: do pulse features separate real from fake under an identity-disjoint split, for several generator families?"),
    ("b", "(c) Added value: does adding pulse evidence raise the held-out metric beyond the interval of the paired difference?"),
    ("b", "(d) Negative controls: time-shuffled or mismatched signals."),
    ("h", "Result"),
    ("ph", "[to be filled from E-xxx, Stage 6; adopt only if (c) is met, otherwise report as a negative result]"),
], size=20)
add_tag(s)

# 16 final comparison
s = new_slide("Final comparison under one protocol",
    "Final comparison. All candidates are compared on the same data, split, budget and metric, with at least three seeds. Show mean, standard deviation over seeds and the bootstrap interval, and the paired difference to the best simple candidate. "
    "Show the cost axes in the same table: parameters, FLOPs, peak VRAM, seconds per epoch. State the protocol next to every number: in-dataset and cross-dataset numbers are never mixed in one row. "
    "Only after the table is filled from the experiment record may the speaker use words like 'differs from'. We do not say 'better than' unless the interval of the paired difference excludes zero. Every cell comes from E-xxx.")
add_table(s, ["Candidate", "Protocol", "Metric (mean, sd over seeds)", "95% interval", "Cost axes"], [
    ["[TBD: candidate A]", "[TBD]", TODO, TODO, TODO],
    ["[TBD: candidate B]", "[TBD]", TODO, TODO, TODO],
    ["[TBD: final system]", "[TBD]", TODO, TODO, TODO],
], widths=[2300000, 1700000, 2700000, 1800000, 2216308], size=14, row_h=700000)
add_tag(s)

# 17 limitations
s = new_slide("Limitations: what we cannot claim",
    "Limitations. State what is known before any run: published numbers use different protocols, so we compare with a published detector only by re-running it on our protocol; a detector trained on one kind of manipulation supports no claim about other kinds without a test; "
    "labelled audio-visual datasets are fewer and smaller than visual ones; compute is one 12 GB GPU. Then add the limits that appeared during the experiments, from the experiment log, and name the experiment for each.")
add_body(s, [
    ("b", "Published numbers use different protocols; we compare only by re-running on our protocol."),
    ("b", "A detector trained on one kind of fake supports no claim about other kinds without a test."),
    ("b", "Labelled audio-visual data is scarcer than visual data, which limits the audio branch."),
    ("b", "Compute is one 12 GB GPU, which bounds model size and clip length."),
    ("bph", "[TBD: limits found during the experiments, from the experiment log, each with its E-xxx]"),
], size=22)

# 18 conclusion
s = new_slide("Conclusion and further work",
    "Conclusion. Written after the results exist, from the experiment log only. Say what was tried, what worked, what did not, and what the evidence does not support, each tied to an experiment ID. "
    "Then give the further work in point form. Do not claim anything that is not in a logged experiment.")
add_body(s, [
    ("h", "Conclusion"),
    ("ph", "[to be filled from E-xxx: what was tried, what worked, what did not, what the evidence does not support]"),
    ("h", "Further work"),
    ("ph", "[TBD: after the results]"),
], size=22)
add_tag(s)

# 19 references
refs_left = [
    "Dolhansky et al., The DeepFake Detection Challenge (DFDC) Dataset, arXiv:2006.07397, 2020.",
    "Yan et al., DeepfakeBench, NeurIPS Datasets and Benchmarks, 2023.",
    "Li et al., Celeb-DF, arXiv:1909.12962, 2020.",
    "Li et al., Face X-ray for More General Face Forgery Detection, arXiv:1912.13458, 2020.",
    "Rossler et al., FaceForensics++, arXiv:1901.08971, 2019.",
    "Ojha et al., Towards Universal Fake Image Detectors, arXiv:2302.10174, 2024.",
    "Ciftci et al., FakeCatcher, arXiv:1901.02212, 2020.",
    "Hernandez-Ortega et al., DeepFakesON-Phys, arXiv:2010.00400, 2020.",
]
refs_right = [
    "Seibold et al., High-quality deepfakes have a heart!, Frontiers in Imaging 4, 2025.",
    "Harraq and Aldwairi, Physiological Signals as a Forensic Modality, arXiv:2607.21776, 2026 (preprint).",
    "Lin et al., Detection of Photoplethysmography Manipulation in Video Forgery, EUSIPCO 2023.",
    "Wang et al., ASVspoof 2019, Computer Speech and Language, 2020.",
    "Jung et al., AASIST, arXiv:2110.01200, 2021.",
    "Khalid et al., FakeAVCeleb, arXiv:2108.05080, 2022.",
    "Chugh et al., Not made for each other, arXiv:2005.14405, 2021.",
]
s = new_slide("References",
    "References. Only papers that were read from the primary source and are listed in docs/RESEARCH.md with status READ are cited. The preprint status of the arXiv entries was not verified beyond the title page.")
add_body(s, [("b", r) for r in refs_left], x=737846, y=1000000, w=5300000, h=4300000, size=12)
add_body(s, [("b", r) for r in refs_right], x=6150000, y=1000000, w=5300000, h=4300000, size=12)

prs.core_properties.title = "Multimodal Deepfake Detection (presentation, draft skeleton)"
prs.core_properties.author = "Project group, K. J. Somaiya School of Engineering"
prs.core_properties.last_modified_by = "Project group"
prs.core_properties.comments = "Skeleton built 2026-10-24; no results; see docs/_agents/A.md"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
