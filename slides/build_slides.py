"""Builds slides/DeepFake_Detection_Presentation.pptx from the reference deck's master and layouts.

Source (read only): Document Template/DeepFake_Detection_System_Presentation (2).pptx
Kept: slide master (red bars, Somaiya Vidyavihar University logo, Somaiya Trust logo), layouts, theme, slide size.
Removed: every slide of the reference deck (its project content). Geometry, type scale and measurement: slide_common.py.

Run from the repository root:   python slides/build_slides.py
Rules (CLAUDE.md, docs/DOC_FACTS.md): no result is written unless it is logged; numbers that do not exist are the single
placeholder [pending] (experiment ID in the speaker notes); no IDs (D-, O-, E-, R-) in visible text; no em or en dashes.
"""
from __future__ import annotations

import os
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import slide_common as C  # noqa: E402
from slide_common import (BOT, COLW, DARK, FONT, GREY, H, L, LINE, MID, R, RED, TOP, W, col_x, span_w)  # noqa: E402

SRC = os.path.join(ROOT, "Document Template", "DeepFake_Detection_System_Presentation (2).pptx")
OUT = os.path.join(HERE, "DeepFake_Detection_Presentation.pptx")
FIG = os.path.join(ROOT, "assets", "figures")
SCR = os.path.join(ROOT, "assets", "screens")
BRD = os.path.join(ROOT, "assets", "branding")

SLIDE_W_ = C.SLIDE_W
FOOTER_TEXT = "Electronics and Computer Engineering"
NSMAP = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
         "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}

# diagram palette (same as the figures of the paper and the report)
F_BLUE, F_GREEN, F_ORANGE, F_RED, F_GREY, F_WHITE = "DDE7F4", "D8EBD8", "FBE8C8", "F6D9D7", "ECEFF3", "FFFFFF"
EDGE = "8496B0"

prs = Presentation(SRC)
LAYOUT = {l.name: l for l in prs.slide_layouts}
_sld = prs.slides._sldIdLst
for sldId in list(_sld):                      # remove every slide of the reference deck
    prs.part.drop_rel(sldId.rId)
    _sld.remove(sldId)

_count = [0]
WARN: list[str] = []


# ---------------------------------------------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------------------------------------------
def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def style_run(run, size, bold=False, italic=False, color=DARK):
    run.font.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)


TOKEN = re.compile(r"(\*\*.+?\*\*|\[pending\])")


def add_runs(p, text, size, bold=False, italic=False, color=DARK):
    """Adds runs to paragraph p. **x** is bold; [pending] is always the one placeholder style."""
    for part in TOKEN.split(text):
        if not part:
            continue
        if part == "[pending]":
            r = p.add_run(); r.text = part
            style_run(r, size, bold, True, RED)
        elif part.startswith("**"):
            r = p.add_run(); r.text = part[2:-2]
            style_run(r, size, True, italic, color)
        else:
            r = p.add_run(); r.text = part
            style_run(r, size, bold, italic, color)


def set_bullet(p, level=0, size=18):
    ppr = p._p.get_or_add_pPr()
    marl = 257175 + level * 285750
    ppr.set("marL", str(marl))
    ppr.set("indent", "-257175")
    for t in ("a:buNone", "a:buChar", "a:buFont", "a:buAutoNum"):
        for e in ppr.findall(qn(t)):
            ppr.remove(e)
    bf = etree.SubElement(ppr, qn("a:buFont")); bf.set("typeface", "Arial")
    bc = etree.SubElement(ppr, qn("a:buChar")); bc.set("char", "•" if level == 0 else "-")


def set_spacing(p, before=0, after=0):
    ppr = p._p.get_or_add_pPr()
    for t in ("a:spcBef", "a:spcAft"):
        for e in ppr.findall(qn(t)):
            ppr.remove(e)
    # schema order: lnSpc, spcBef, spcAft, bu*
    sb = etree.Element(qn("a:spcBef")); etree.SubElement(sb, qn("a:spcPts")).set("val", str(int(before * 100)))
    sa = etree.Element(qn("a:spcAft")); etree.SubElement(sa, qn("a:spcPts")).set("val", str(int(after * 100)))
    ppr.insert(0, sa)
    ppr.insert(0, sb)


def fill_text(tf, paras, anchor=MSO_ANCHOR.TOP, inset=0):
    """paras: list of (kind, text). kinds: h (block heading), b (bullet), b2 (sub bullet), p (plain), s (secondary plain),
    eq (equation line), cap (caption, 12 pt), n (numbered line given with its own number)."""
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "rIns", "tIns", "bIns"):
        bp.set(k, str(inset))
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, qn("a:noAutofit"))
    first = True
    for item in paras:
        kind, text = item[0], item[1]
        over = item[2] if len(item) > 2 else None
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        size = over or {"h": C.SZ_LEAD, "b": C.SZ_BODY, "b2": C.SZ_BODY_S, "p": C.SZ_BODY, "s": C.SZ_BODY_S, "eq": C.SZ_BODY,
                        "cap": C.SZ_TABLE_S, "n": C.SZ_BODY, "bs": C.SZ_BODY_S, "eq20": 20}[kind]
        if kind == "h":
            add_runs(p, text, size, bold=True, color=RED)
            set_spacing(p, 0 if first else 8, 4)
        elif kind in ("b", "b2", "bs"):
            add_runs(p, text, size)
            set_bullet(p, 1 if kind == "b2" else 0)
            set_spacing(p, 0, 6)
        elif kind == "cap":
            add_runs(p, text, size, italic=True, color=MID)
            set_spacing(p, 0, 0)
        elif kind == "n":
            add_runs(p, text, size)
            ppr = p._p.get_or_add_pPr(); ppr.set("marL", "342900"); ppr.set("indent", "-342900")
            set_spacing(p, 0, 6)
        elif kind in ("eq", "eq20"):
            add_runs(p, text, size)
            set_spacing(p, 0, 10)
        else:
            add_runs(p, text, size)
            set_spacing(p, 0, 6)
        first = False


_shape_n = [0]


def _name(prefix):
    _shape_n[0] += 1
    return f"{prefix}_{_shape_n[0]}"


def add_text(slide, x, y, w, h, paras, name="text", anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tb.name = _name(name)
    fill_text(tb.text_frame, paras, anchor)
    need = C.estimate_text_height(tb)
    if need > h * 1.0:
        WARN.append(f"slide {_count[0]}: {tb.name} needs {need/914400:.2f} in, has {h/914400:.2f} in")
    return tb


def add_card(slide, x, y, w, h, paras, fill=C.BAND, name="card", anchor=MSO_ANCHOR.TOP, inset=110000):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    sh.name = _name(name)
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(LINE); sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    fill_text(sh.text_frame, paras, anchor, inset)
    need = C.estimate_text_height(sh)
    if need > h:
        WARN.append(f"slide {_count[0]}: {sh.name} needs {need/914400:.2f} in, has {h/914400:.2f} in")
    return sh


# ---------------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------------
def _cell_border(cell, color=LINE, width=6350):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for e in tcPr.findall(qn(tag)):
            tcPr.remove(e)
    for tag in ("a:lnL", "a:lnR", "a:lnT"):
        ln = etree.Element(qn(tag)); ln.set("w", "0"); etree.SubElement(ln, qn("a:noFill")); tcPr.insert(0, ln)
    lnb = etree.Element(qn("a:lnB")); lnb.set("w", str(width))
    sf = etree.SubElement(lnb, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    # order inside tcPr: lnL lnR lnT lnB then fill
    kids = list(tcPr)
    for k in kids:
        tcPr.remove(k)
    order = {qn("a:lnL"): 0, qn("a:lnR"): 1, qn("a:lnT"): 2, qn("a:lnB"): 3}
    lns = sorted([k for k in kids if k.tag in order] + [lnb], key=lambda e: order[e.tag])
    for k in lns + [k for k in kids if k.tag not in order]:
        tcPr.append(k)


CELL_MX, CELL_MY = 70000, 35000


def _table_rows_h(widths, header, rows, size, bold_first_col):
    hs = []
    for ri, row in enumerate([header] + rows):
        hmax = 0
        for ci, val in enumerate(row):
            lines = val.split(chr(10))
            nl = sum(C.lines_for_runs([(ln, ri == 0 or (bold_first_col and ci == 0), False)], size, widths[ci] - 2 * CELL_MX) for ln in lines)
            hmax = max(hmax, nl * size * C.LINE_FACTOR * 12700 + 2 * CELL_MY)
        hs.append(int(hmax + 12000))
    return hs


def add_table(slide, x, y, widths, header, rows, size=C.SZ_TABLE, bold_first_col=False, name="table", max_h=None, pad=0, fill=None):
    """Native table. size 'auto' picks the largest size from 18 to 12 pt that fits max_h. Row heights are estimated from the
    text so that the frame height is what PowerPoint draws; pad (EMU) is added to every body row for air."""
    if size == "auto":
        for cand in (18, 17, 16, 15, 14, 13, 12):
            hs = _table_rows_h(widths, header, rows, cand, bold_first_col)
            if sum(hs) + pad * len(rows) <= max_h:
                size = cand
                break
        else:
            size = 12
    if fill is not None:     # spread the free height over the body rows (at most 160,000 EMU per row)
        base = sum(_table_rows_h(widths, header, rows, size, bold_first_col))
        pad = max(pad, int(min(160000, max(0, (fill - base) / max(1, len(rows))))))
    nr, nc = len(rows) + 1, len(header)
    gt = slide.shapes.add_table(nr, nc, Emu(x), Emu(y), Emu(sum(widths)), Emu(400000 * nr))
    gt.name = _name(name)
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    for i, wd in enumerate(widths):
        tbl.columns[i].width = Emu(wd)
    hs = _table_rows_h(widths, header, rows, size, bold_first_col)
    total = 0
    for ri, row in enumerate([header] + rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Emu(CELL_MX)
            cell.margin_top = cell.margin_bottom = Emu(CELL_MY)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            for li, line in enumerate(val.split(chr(10))):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                if ri == 0:
                    add_runs(p, line, size, bold=True, color="FFFFFF")
                else:
                    add_runs(p, line, size, bold=(bold_first_col and ci == 0))
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(RED if ri == 0 else (C.BAND if ri % 2 == 0 else "FFFFFF"))
            _cell_border(cell)
        h = hs[ri] + (pad if ri else 0)
        tbl.rows[ri].height = Emu(h)
        total += h
    gt.height = Emu(total)
    if max_h is not None and total > max_h:
        WARN.append(f"slide {_count[0]}: {gt.name} is {total/914400:.2f} in tall, limit {max_h/914400:.2f} in")
    return gt, total


# ---------------------------------------------------------------------------------------------------------------------
# diagram helpers (names start with dg_ or arrow_ so that the checker can treat them as diagram parts)
# ---------------------------------------------------------------------------------------------------------------------
def dbox(slide, x, y, w, h, text, fill=F_BLUE, size=C.SZ_TABLE, bold=False, name="dg_box", align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, dashed=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    sh.name = _name(name)
    sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(EDGE); sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "rIns"):
        bp.set(k, "45000")
    for k in ("tIns", "bIns"):
        bp.set(k, "30000")
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, qn("a:noAutofit"))
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        add_runs(p, line, size, bold=bold or (i == 0 and len(lines) > 1 and line.startswith("**")))
        set_spacing(p, 0, 0)
    need = C.estimate_text_height(sh)
    if need > h:
        WARN.append(f"slide {_count[0]}: {sh.name} '{text[:20]}' needs {need/914400:.2f} in, has {h/914400:.2f} in")
    return sh


def darrow(slide, x1, y1, x2, y2, dashed=False, both=False):
    cx = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(x1), Emu(y1), Emu(x2), Emu(y2))
    cx.name = _name("arrow")
    cx.line.color.rgb = rgb(DARK); cx.line.width = Pt(1.75)
    ln = cx.line._get_or_add_ln()
    if dashed:
        pd = etree.SubElement(ln, qn("a:prstDash")); pd.set("val", "dash")
    if both:
        he = etree.SubElement(ln, qn("a:headEnd")); he.set("type", "triangle"); he.set("w", "lg"); he.set("len", "lg")
    te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "lg"); te.set("len", "lg")
    return cx


def add_pic(slide, path, x, y, w=None, h=None, alt=""):
    pic = slide.shapes.add_picture(path, Emu(x), Emu(y), Emu(w) if w else None, Emu(h) if h else None)
    pic.name = _name("pic")
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic



def fig(slide, name, x, y, w=None, h=None, alt=""):
    """Places an assets/figures PNG at its own aspect ratio (width or height given)."""
    from PIL import Image
    iw, ih = Image.open(os.path.join(FIG, name)).size
    if w is None:
        w = int(h * iw / ih)
    else:
        h = int(w * ih / iw)
    return add_pic(slide, os.path.join(FIG, name), x, y, w=w, h=h, alt=alt), w, h

# ---------------------------------------------------------------------------------------------------------------------
# slide scaffolding
# ---------------------------------------------------------------------------------------------------------------------
def _ph(slide, kind, idx, text_xml, name):
    ids = [int(e.get("id")) for e in slide.shapes._spTree.iter() if e.tag.endswith("}cNvPr")]
    nid = max(ids + [1]) + 1
    xml = (f'<p:sp xmlns:p="{NSMAP["p"]}" xmlns:a="{NSMAP["a"]}"><p:nvSpPr><p:cNvPr id="{nid}" name="{name}"/>'
           f'<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="{kind}" sz="quarter" idx="{idx}"/></p:nvPr></p:nvSpPr>'
           f'<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p>{text_xml}</a:p></p:txBody></p:sp>')
    slide.shapes._spTree.append(etree.fromstring(xml))


def _rpr(extra=""):
    return (f'<a:rPr lang="en-US" sz="1200" b="0" i="0"{extra}><a:solidFill><a:srgbClr val="{GREY}"/></a:solidFill>'
            f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/><a:cs typeface="{FONT}"/></a:rPr>')


def add_footer(slide, number):
    _ph(slide, "ftr", 11, f'<a:r>{_rpr()}<a:t>{FOOTER_TEXT}</a:t></a:r>', "Footer")
    _ph(slide, "sldNum", 12,
        f'<a:fld id="{{00000000-1234-1234-1234-123412341234}}" type="slidenum">{_rpr()}<a:t>{number}</a:t></a:fld>', "Slide number")


def set_title(shape, text, size=C.SZ_TITLE, y=C.TITLE_Y, h=C.TITLE_H, check=True):
    shape.left, shape.top, shape.width, shape.height = Emu(L), Emu(y), Emu(W), Emu(h)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    bp = tf._txBody.find(qn("a:bodyPr"))
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, qn("a:noAutofit"))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    style_run(r, size, True, False, "000000")
    wpt = C.text_width_pt(text, size, True) * 12700
    if check and wpt > 2 * (C.SLIDE_W / 2 - C.LOGO_L_END) - 200000:
        WARN.append(f"slide {_count[0]}: title '{text}' is too wide for the space between the logos")


def new_slide(title, notes, layout="TITLE_ONLY"):
    _count[0] += 1
    s = prs.slides.add_slide(LAYOUT[layout])
    set_title(s.shapes.title, title)
    add_footer(s, _count[0])
    assert len(notes.split(". ")) >= 3, "notes too short on slide %d" % _count[0]
    s.notes_slide.notes_text_frame.text = notes
    return s


# =====================================================================================================================
# SLIDE 1: title
# =====================================================================================================================
_count[0] += 1
s = prs.slides.add_slide(LAYOUT["TITLE"])
set_title(s.shapes.title, "Multimodal Deepfake Detection for Images, Video and Audio", size=24, y=40000, h=820000, check=False)
t1 = s.shapes.title
t1.left, t1.width = Emu(2600000), Emu(SLIDE_W_ - 2 * 2600000)
sub = [p for p in s.placeholders if p.placeholder_format.idx == 1][0]
sub.left, sub.top, sub.width, sub.height = Emu(L), Emu(880000), Emu(W), Emu(640000)
tf = sub.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.TOP
bp = tf._txBody.find(qn("a:bodyPr"))
for e in list(bp):
    bp.remove(e)
etree.SubElement(bp, qn("a:noAutofit"))
for k_ in ("lIns", "rIns", "tIns", "bIns"):
    bp.set(k_, "0")
for i, (txt, sz, it) in enumerate([("A Hypothesis-Driven Study Under a Source-Disjoint Protocol", 20, True),
                                   ("Department of Electronics and Computer Engineering", 18, False)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt
    style_run(r, sz, False, it, GREY)
    set_spacing(p, 0, 2)
# campus photographs, cropped to the same height and aligned to the grid
PH_Y, PH_H = 1600000, 2250000
half = (W - 120000) // 2
for k, (fn, alt) in enumerate([("campus_aryabhat_building.png", "Aryabhat building of the K. J. Somaiya College of Engineering campus"),
                               ("campus_bhaskaracharya_building.png", "Bhaskaracharya building of the K. J. Somaiya College of Engineering campus")]):
    pic = add_pic(s, os.path.join(BRD, fn), L + k * (half + 120000), PH_Y, alt=alt)
    ar_box = half / PH_H
    ar_img = pic.image.size[0] / pic.image.size[1]
    pic.width, pic.height = Emu(half), Emu(PH_H)
    pic.left = Emu(L + k * (half + 120000)); pic.top = Emu(PH_Y)
    if ar_img < ar_box:      # image is taller than the box: crop top and bottom
        keep = ar_img / ar_box
        pic.crop_top = pic.crop_bottom = (1 - keep) / 2
    else:
        keep = ar_box / ar_img
        pic.crop_left = pic.crop_right = (1 - keep) / 2
add_text(s, L, 3930000, W, 400000, [("h", "Group members")], "members_label")
for k, (nm, roll) in enumerate([("Vaibhav Shiroorkar", "16014024805"), ("Shivansh Gupta", "16014023052"), ("Adithya Jithesh", "16014024803")]):
    add_text(s, col_x(4 * k), 4340000, span_w(4), 760000, [("p", f"**{nm}**"), ("s", f"Roll No: {roll}")], "member")
add_text(s, L, 5150000, W, 500000,
         [("p", "**Guide:** Dr. Tejashree Phatak, Assistant Professor")], "guide")
add_footer(s, 1)
s.notes_slide.notes_text_frame.text = (
    "Introduce the project: one system that scores images, videos and the audio inside videos, built from zero and documented as a path from a simple working version to a measured detector. "
    "Introduce the three members in the report order (Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh) and the guide, Dr. Tejashree Phatak, Assistant Professor. "
    "State the shape of the talk: the problem, the literature, the protocol and method, the data and the encoder shortlist, the rule for the working version, the iteration ladder, the rPPG test, the web demo, the timeline to 30 October and the limitations. "
    "Say at once that no final result exists yet: every number that is still missing is shown as [pending] on the slide and is filled only from a logged experiment. "
    "Evidence pointers: docs/IDENTITY.md for the identity fields, docs/DOC_FACTS.md for the wording, docs/PLAN.md for the schedule. "
    "The title and subtitle are the working title of the paper; the wording Group-Disjoint instead of Source-Disjoint in the subtitle is a proposal only, for the authors and the guide to decide.")

# =====================================================================================================================
# SLIDE 2: problem and scope
# =====================================================================================================================
s = new_slide("Problem and scope",
    "The task is to say whether an image, a video with its audio, or a standalone audio clip is real or generated. "
    "The output is a calibrated probability that the content is fake and a statement of which modality carries the evidence; for a video the visual score and the audio score are shown separately, because the rule for combining them is tested only in the late-fusion iteration (I-6, D-018, D-019). "
    "The published literature reports numbers under different protocols, so they cannot be read side by side; DeepfakeBench trained plain detectors on FaceForensics++ c23 and measured a mean AUC of 0.945 within the domain and 0.772 across eight other sets for Xception (Yan et al., 2023, table of R-017, their protocol, not our result). "
    "The aim is a documented path with results comparable to published detectors on a protocol we define and run ourselves (D-005, D-025), not the best model. "
    "Evidence: docs/DOC_FACTS.md section 2, docs/REQUIREMENTS.md, docs/RESEARCH.md R-017.")
x1, w1 = col_x(0), span_w(7)
x2, w2 = col_x(7) + 0, span_w(5)
add_text(s, x1, TOP, w1, 3150000, [
    ("h", "The problem"),
    ("b", "Published detectors are measured under different protocols, so their numbers cannot be read side by side."),
    ("b", "A score on the training data says little about new data: plain Xception trained on FaceForensics++ c23 reached a mean AUC of 0.945 within the domain and 0.772 over eight other datasets (Yan et al., 2023, their protocol)."),
    ("b", "Fake covers a swapped, re-enacted or audio-driven face, a generated image (GAN or diffusion), synthesised or converted speech, and audio and video manipulated separately."),
], "problem")
add_card(s, x2, TOP, w2, 3150000, [
    ("h", "What the system returns"),
    ("b", "A calibrated probability that the content is fake."),
    ("b", "Which modality carries the evidence."),
    ("b", "For a video: a visual score and an audio score, shown separately."),
    ("b", "No temporal localisation."),
], name="card_returns")
add_card(s, L, TOP + 3300000, W, 1450000, [
    ("h", "Aim"),
    ("p", "A documented path: a simple working version, then seven iterations that each follow from a written hypothesis, then integration and freeze. The goal is a result comparable with published detectors on a protocol we define and run ourselves, not the best model."),
], name="card_aim")

# =====================================================================================================================
# SLIDE 3: literature
# =====================================================================================================================
s = new_slide("Literature and what we take from it",
    "Only sources that were opened and read from their primary text are used, and every number inside them is theirs, under their own protocol. "
    "Datasets: FaceForensics++ has four manipulation methods at three compression levels, Celeb-DF is the hard cross-dataset test, and Celeb-DF++ adds 22 newer generators (R-001, R-002, R-053). "
    "Frozen features: a linear probe on frozen CLIP features generalised across generators when trained on ProGAN (Ojha et al., 2023), which motivates the frozen-encoder working version (D-032). "
    "Pulse evidence is mixed, which is why rPPG is a test and not a component (D-006). "
    "Audio: ASVspoof 2019 is the standard benchmark, AASIST reports best-seed and three-seed mean numbers that differ, and audio-visual sets with per-modality labels are FakeAVCeleb and LAV-DF (R-012, R-013, R-014, R-024). "
    "Evidence: docs/RESEARCH.md, rows R-001 to R-053.")
widths = [span_w(2), span_w(4), span_w(4)]
widths.append(W - sum(widths))
gt, th = add_table(s, L, TOP, widths, ["Area", "Sources", "What they show (own protocol)", "What we take"], [
    ["Face datasets", "Rossler et al., 2019; Li et al., 2020a; Dolhansky et al., 2020; Li et al., 2025",
     "FaceForensics++: four methods, three compression levels. Celeb-DF and Celeb-DF++ are hard cross-dataset tests.",
     "FaceForensics++ trains; the Celeb-DF family tests"],
    ["Visual detectors", "Li et al., 2020b; Yan et al., 2023; Dong et al., 2023",
     "One recipe makes detectors comparable; binary detectors can learn identity cues.",
     "One pipeline; identity-disjoint splits"],
    ["Frozen features", "Ojha et al., 2023; Kornblith et al., 2019",
     "A linear probe on frozen CLIP features generalised across generators.",
     "Frozen encoder, linear head"],
    ["Pulse (rPPG)", "Ciftci et al., 2020; Hernandez-Ortega et al., 2021; Seibold et al., 2025; Harraq and Aldwairi, 2026; Lin et al., 2023",
     "High in-dataset accuracy early on; mixed on modern generators; pulse signals can be edited.",
     "Test the signal first"],
    ["Audio", "Wang et al., 2020; Jung et al., 2022; Tak et al., 2022; Muller et al., 2022; Khalid et al., 2021; Chugh et al., 2020; Cai et al., 2022",
     "ASVspoof 2019 is the standard benchmark; scores drop on In-the-Wild; FakeAVCeleb and LAV-DF carry per-modality labels.",
     "ASVspoof trains; In-the-Wild tests; LAV-DF for fusion"],
    ["Calibration", "Guo et al., 2017", "Temperature scaling repairs the confidence of deep networks.", "Temperature scaling, ECE"],
], size="auto", max_h=H - 420000, fill=H - 420000)
add_text(s, L, TOP + th + 90000, W, 300000, [("cap", "Numbers inside the cited papers belong to those papers and their protocols; none is a result of this project.")], "lit_note")

# =====================================================================================================================
# SLIDE 4: protocol (splits)
# =====================================================================================================================
s = new_slide("Protocol: group-disjoint splits",
    "Data is split once, by group: the group is the source video together with the person identity, so no person, video source or recording appears on both sides of a split (D-012, PROTOCOL.md section 1). "
    "Binary detectors can learn identity from the labels alone (Dong et al., 2023), so a split by identity is needed for honest numbers. "
    "The development slice dev_x is carved out of a cross-dataset source, ranks candidates and may be evaluated repeatedly; test_in and test_x are sealed and are touched once per frozen iteration by a guarded command that refuses the run without the final flag and writes a ledger (D-035, D-041). "
    "A script proves group-disjointness, including content-hash de-duplication across splits and across a dataset family (D-026, D-045). "
    "Evidence: docs/PROTOCOL.md sections 1 and 7, tests A6 and A9 in docs/EXPERIMENTS.md V-002.")
DY = TOP
colw_left = span_w(5)
colw_right = span_w(6)
xr = L + W - colw_right
dbox(s, L, DY, colw_left, 3100000, "", fill="E7EDF7", name="dg_panel_left")
dbox(s, xr, DY, colw_right, 3100000, "", fill=F_GREY, name="dg_panel_right")
add_text(s, L + 100000, DY + 60000, colw_left - 200000, 330000, [("s", "**Training dataset**, e.g. FaceForensics++")], "dg_t1")
add_text(s, xr + 100000, DY + 60000, colw_right - 200000, 330000, [("s", "**Cross-dataset source**, e.g. Celeb-DF v2")], "dg_t2")
bw = (colw_left - 3 * 100000) // 2
dbox(s, L + 100000, DY + 450000, bw, 1400000, "**train**\nfits the head", fill=F_GREEN, size=C.SZ_BODY_S)
dbox(s, L + 200000 + bw, DY + 450000, bw, 1400000, "**validation**\nthreshold, temperature, early stopping", fill=F_GREEN, size=C.SZ_BODY_S)
dbox(s, L + 100000, DY + 1990000, colw_left - 200000, 1050000, "**test_in**\nin-dataset, sealed, reported but not the headline", fill=F_RED, size=C.SZ_BODY_S)
bw2 = (colw_right - 3 * 100000) // 2
dbox(s, xr + 100000, DY + 450000, bw2, 1400000, "**dev_x**\ndevelopment slice: ranks candidates, may be reused", fill=F_ORANGE, size=C.SZ_BODY_S)
dbox(s, xr + 200000 + bw2, DY + 450000, bw2, 1400000, "**test_x**\nsealed headline test, touched once per frozen iteration", fill=F_RED, size=C.SZ_BODY_S)
dbox(s, xr + 100000, DY + 1990000, colw_right - 200000, 1050000, "dev_x and test_x share no group, identity or file content", fill=F_WHITE, size=C.SZ_BODY_S)
gap_x0 = L + colw_left
darrow(s, gap_x0 + 30000, DY + 1150000, xr - 30000, DY + 1150000, both=True)
add_text(s, L, TOP + 3250000, W, 1560000, [
    ("b", "Group = source video together with the person identity; a dataset family (Celeb-DF v2 and Celeb-DF++) counts as one source.", 17),
    ("b", "Tuning uses train, validation and dev_x only. The two test sets are touched once per frozen iteration, by a guarded command that writes a ledger.", 17),
    ("b", "A script proves the splits are disjoint, including content-hash de-duplication across splits.", 17),
], "protocol_text")

# =====================================================================================================================
# SLIDE 5: method pipeline
# =====================================================================================================================
s = new_slide("Method: file to calibrated score",
    "The working version uses one frozen pretrained encoder per task and a linear head (D-032). "
    "A video gives 16 uniformly spaced frames, a face detector crops the largest face, the frozen image encoder and the linear head score each frame, the clip score is the mean of the frame probabilities and temperature scaling calibrates it (D-024, D-033, D-034). "
    "Audio uses the first 4 seconds at 16 kHz through a frozen speech encoder, mean and standard deviation over time and a linear head; an image goes through the face path if a face is found and through the whole-image path otherwise. "
    "The threshold and the temperature are fixed on validation data only, and visual and audio scores of a video are shown side by side because no fusion rule is claimed yet (D-040). "
    "Evidence: docs/DOC_FACTS.md sections 4 and 5, figure 1 of the paper (same content), src/dfd/infer.")
n_slots = 7
gap = 140000
bwid = (W - (n_slots - 1) * gap) // n_slots
bh = 830000


def row_boxes(slide, y, labels, fills, label_text, start_slot=0):
    add_text(slide, L, y - 340000, W, 330000, [("s", f"**{label_text}**")], "dg_rowlabel")
    prev = None
    for i, (t, f) in enumerate(zip(labels, fills)):
        x = L + (start_slot + i) * (bwid + gap)
        dbox(slide, x, y, bwid, bh, t, fill=f)
        if prev is not None:
            darrow(slide, prev + bwid, y + bh // 2, x, y + bh // 2)
        prev = x


y0 = TOP + 370000
row_boxes(s, y0, ["16 frames (uniform)", "Face detector, crop 224", "Frozen image encoder", "Linear head", "Mean over frames", "Temperature scaling", "**Visual score**"],
          [F_BLUE, F_BLUE, F_GREEN, F_GREEN, F_BLUE, F_BLUE, F_ORANGE], "Video, visual branch")
y1 = y0 + bh + 560000
row_boxes(s, y1, ["First 4 s at 16 kHz", "Frozen speech encoder", "Mean and std over time", "Linear head", "Temperature scaling", "**Audio score**"],
          [F_BLUE, F_GREEN, F_BLUE, F_GREEN, F_BLUE, F_ORANGE], "Audio branch (an audio file, or the audio track of a video)")
dbox(s, L + 6 * (bwid + gap), y1, bwid, bh, "shown side by side with the visual score", fill=F_GREY)
darrow(s, L + 5 * (bwid + gap) + bwid, y1 + bh // 2, L + 6 * (bwid + gap), y1 + bh // 2)
y2 = y1 + bh + 560000
row_boxes(s, y2, ["Face found?", "Face crop, or whole image", "Frozen image encoder", "Linear head", "Temperature scaling", "**Image score**"],
          [F_BLUE, F_BLUE, F_GREEN, F_GREEN, F_BLUE, F_ORANGE], "Image")
add_text(s, L, y2 + bh + 110000, W, 640000,
         [("s", "Green boxes hold learned or pretrained weights. The threshold and the temperature are fixed on validation data; scores are probabilities that the content is fake.")], "pipeline_note")

# =====================================================================================================================
# SLIDE 6: method and maths
# =====================================================================================================================
s = new_slide("Method and the mathematics",
    "The head is a logistic regression on standardised features, so the loss is convex and the optimum does not depend on the starting point; the seeds change only the batch order and the stopping point (MATH_NOTES section 10). "
    "Class weights n/(2 n1) and n/(2 n0) give the risk under a balanced prior, which matters because FaceForensics++ is 80 percent fake. "
    "Temperature scaling divides the logit by one positive number fitted on validation data, so it changes the probabilities but not the ranking or the AUC (Guo et al., 2017). "
    "Intervals come from resampling groups, not frames, because frames of one video are dependent; two models are compared on the same resampled groups so the paired interval is narrower than two separate ones. "
    "Comparable with a published detector means non-inferior, with delta fixed before the final comparison from the measured seed spread of the baseline (D-025). Evidence: docs/MATH_NOTES.md sections 2, 7, 8, 10; docs/PROTOCOL.md section 6.")
half1, half2 = span_w(6), span_w(6)
add_card(s, col_x(0), TOP, half1, 4800000, [
    ("h", "Scoring"),
    ("eq20", "**Head:** p = σ(w·z + b) on the frozen feature z = f(x), with σ(u) = 1 / (1 + exp(−u))."),
    ("eq20", "**Loss:** class-balanced cross-entropy, weights n / (2 n1) for fake and n / (2 n0) for real; the problem is convex."),
    ("eq20", "**Clip score:** the mean of the frame probabilities."),
    ("eq20", "**Calibration:** q = σ(ℓ / τ), one τ > 0 fitted on validation data; ranking and AUC do not change."),
    ("eq20", "**Threshold:** maximises balanced accuracy on validation data, then fixed for every test set."),
], name="card_maths_l")
add_card(s, col_x(6), TOP, half2, 4800000, [
    ("h", "Evaluation"),
    ("eq20", "**AUC** = P(S1 > S0) + ½ P(S1 = S0): the chance that a random fake outranks a random real example."),
    ("eq20", "**Interval:** resample groups with replacement, 2,000 resamples, 95 percent percentile interval."),
    ("eq20", "**Paired difference** Δ = θ(A) − θ(B) on the same resampled groups."),
    ("eq20", "**Non-inferior** to a re-run baseline: the lower end of the interval of (ours − baseline) is above −δ; δ is fixed beforehand."),
], name="card_maths_r")

# =====================================================================================================================
# SLIDE 7: data
# =====================================================================================================================
s = new_slide("Data and splits",
    "Each task has its own training data and its own development slice, and every split is group-disjoint. "
    "Visual: FaceForensics++ c23 trains and validates (official split: 3,600 training and 700 validation clips), Celeb-DF v2 supplies the development slice (15 of 59 identity groups; 1,728 clips) and the rest of Celeb-DF v2, Celeb-DF++ and DFD are sealed (D-013, D-035, D-045). "
    "Images: the ProGAN validation set is split by class because its images have no identity (D-026); audio: ASVspoof 2019 LA trains and In-the-Wild supplies the development slice (D-035). "
    "LAV-DF has 136,304 videos, all with an audio stream, and per-file labels for video and audio manipulation, so it carries the fusion iteration. "
    "Evidence: docs/DATA_LOG.md, docs/PROTOCOL.md, docs/DOC_FACTS.md section 3, docs/O007_SWEEP.md section 2.")
cw = [span_w(2), span_w(3), span_w(3)]
cw.append(W - sum(cw))
gt, th = add_table(s, L, TOP, cw, ["Task", "Training and validation", "Development slice (dev_x)", "Sealed tests"], [
    ["Face video frames", "FaceForensics++ c23, official train and validation split", "Celeb-DF v2: 15 of 59 identity groups",
     "Rest of Celeb-DF v2 (test_x), Celeb-DF++ (test videos of 21 new generators), DFD; FaceForensics++ test (test_in)"],
    ["Generated images", "ProGAN validation set split by class: 14 classes train, 6 validation", "Two of the seven CNN-synthesis generator units",
     "Other generator units and the diffusion test set (test_x); ProGAN test set (test_in)"],
    ["Speech audio", "ASVspoof 2019 LA train and dev", "In-the-Wild: 12 of 54 speakers",
     "Other In-the-Wild speakers (test_x); ASVspoof 2019 LA eval (test_in)"],
    ["Audio-visual (fusion)", "LAV-DF official train and validation (identity-disjoint)", "n/a: official split used", "LAV-DF official test"],
], size="auto", bold_first_col=True, max_h=3250000, fill=3250000)
add_text(s, L, TOP + th + 130000, W, 1300000, [
    ("b", "All datasets are research-only; Celeb-DF and Celeb-DF++ forbid publishing derived data, so no frame, crop or per-video score of them appears in any document."),
    ("b", "LAV-DF: 136,304 videos, all with audio, labelled per modality (modify_video, modify_audio)."),
], "data_text")

# =====================================================================================================================
# SLIDE 8: shortlist
# =====================================================================================================================
s = new_slide("Encoder shortlist",
    "The shortlist was approved before any feature was extracted: criteria first (licence read from the weights' own repository, offline loading, fit in 12 GB, the families differ), then candidates (D-038). "
    "Images and video frames: two supervised CNNs, two DINOv2 sizes, two CLIP ViT-L/14 models (OpenAI and LAION) and SigLIP base; audio: wav2vec 2.0 base, WavLM base plus, Whisper small encoder and HuBERT base. "
    "A small network from scratch is reported for each task as a reference and is excluded from selection. "
    "DINOv3 was excluded on purpose because its weights sit behind a manual access gate. "
    "The weights of OpenAI CLIP and of WavLM have conditional licences; they are accepted as research-only and stated as a limitation. "
    "Evidence: docs/O007_SWEEP.md section 3.1, docs/DATA_LOG.md encoder weights section; the cost record fills the [pending] cells.")
cw = [span_w(2), span_w(3), span_w(3), span_w(2)]
cw.append(W - sum(cw))
rows = [
    ["Images, video frames", "EfficientNet-B4", "supervised CNN", "17.5 M", "research use (ImageNet)"],
    ["", "Xception", "supervised CNN", "20.8 M", "research use (ImageNet)"],
    ["", "DINOv2 small, base", "self-supervised ViT", "22.1 M, 86.6 M", "Apache-2.0"],
    ["", "CLIP ViT-L/14 (OpenAI)", "image-text contrastive", "304.0 M", "research use (accepted)"],
    ["", "CLIP ViT-L/14 (LAION)", "image-text contrastive", "304.0 M", "MIT tag, research use"],
    ["", "SigLIP base", "sigmoid image-text", "92.9 M", "Apache-2.0"],
    ["Speech audio", "wav2vec 2.0 base", "self-supervised", "94.4 M", "Apache-2.0"],
    ["", "WavLM base plus", "self-supervised", "94.4 M", "research use (accepted)"],
    ["", "Whisper small encoder", "weakly supervised ASR", "[pending]", "Apache-2.0 / MIT"],
    ["", "HuBERT base", "self-supervised", "[pending]", "Apache-2.0"],
    ["Both", "Small network from scratch", "reference, excluded from selection", "[pending]", "own code"],
]
gt, th = add_table(s, L, TOP, cw, ["Task", "Encoder", "Family", "Parameters", "Weights licence"], rows, size="auto", bold_first_col=True, max_h=H - 480000, fill=H - 480000)
add_text(s, L, TOP + th + 90000, W, 330000, [("cap", "Parameters are read from the weight files. DINOv3 is excluded on purpose (manual access gate).")], "short_note")

# =====================================================================================================================
# SLIDE 9: I-0 rule
# =====================================================================================================================
s = new_slide("I-0: the working-version rule",
    "I-0 is the working version and not an improvement: a frozen encoder, a linear head and temperature scaling, chosen by a rule fixed before any feature existed (D-032, D-036, D-040). "
    "The rule ranks the candidates of each task by their mean AUC on the development slice, builds the tie set of every candidate whose paired 95 percent group-bootstrap interval against the leader includes zero, and lets the cheapest candidate in the tie set win; cost is the end-to-end extraction time per 1,000 inputs, then the parameter count. "
    "This prefers the cheaper candidate whenever the evidence does not separate them, which is the project's comparison rule; a permutation test over groups is a sensitivity check and the from-scratch network is reported but excluded. "
    "The working version counts as finished only when the whole chain runs: a file goes in through the web page, the winning encoders score it, the score is calibrated, metrics with intervals exist on validation and development data, and the page shows a real result. "
    "Evidence: docs/O007_SWEEP.md sections 1 to 8, docs/DECISIONS.md D-036 and D-040, experiments E-001 (face frames), E-002 (images), E-003 (speech); the ranking is still being re-run, so the results are [pending].")
xl, wl = col_x(0), span_w(7)
xr2, wr = col_x(7), span_w(5)
add_text(s, xl, TOP, wl, 4800000, [
    ("h", "Rule for choosing the encoder"),
    ("n", "1.  Extract features once per candidate with the frozen encoder."),
    ("n", "2.  Fit a linear head, three seeds."),
    ("n", "3.  Rank by mean development-slice AUC with a paired 95 percent group bootstrap."),
    ("n", "4.  Tie set: every candidate whose interval against the leader includes zero."),
    ("n", "5.  Cheapest wins: lowest extraction time per 1,000 inputs, then fewer parameters."),
    ("p", "The from-scratch reference is reported and excluded from the choice."),
], "rule")
add_card(s, xr2, TOP, wr, 2900000, [
    ("h", "The working version is finished when"),
    ("bs", "a file goes in through the local web page;", 17),
    ("bs", "the winning encoders score it, calibrated, with the threshold fixed on validation data;", 17),
    ("bs", "metrics with intervals exist on validation and development data, three seeds;", 17),
    ("bs", "the page shows a real result, with its warnings.", 17),
], name="card_done")
gt, th = add_table(s, xr2, TOP + 3050000, [span_w(2), span_w(2), wr - 2 * span_w(2)], ["Task", "Winner", "Dev AUC [95% CI]"], [
    ["Face frames", "[pending]", "[pending]"],
    ["Images", "[pending]", "[pending]"],
    ["Speech", "[pending]", "[pending]"],
], size=14, bold_first_col=True, max_h=1750000)

# =====================================================================================================================
# SLIDE 10: ladder
# =====================================================================================================================
s = new_slide("Iteration ladder",
    "After the working version come seven experimental iterations, I-1 to I-7, and a last slot for integration and freeze (D-041, ITERATIONS.md). "
    "Each of I-1 to I-6 takes its lever from one family (fine-tuning depth, frame aggregation, loss and class weighting, augmentation and regularisation, combining encoders, late fusion with calibration); the lever is chosen after the diagnostic battery of the preceding iteration and is logged before the run. "
    "I-7 is the rPPG experiment with success and failure thresholds written beforehand; I-8 runs no experiments, it freezes the code and evaluates every frozen iteration once on the sealed tests. "
    "The battery has eight items and runs after every iteration, including I-0. "
    "No iteration is frozen yet, so there is no per-iteration slide; each frozen iteration will get one slide with its hypothesis, change and logged result. Evidence: docs/ITERATIONS.md, docs/EXPERIMENTS.md section Diagnostic battery.")
labels = [("I-0", "working version", F_GREEN), ("I-1", "fine-tuning depth", F_BLUE), ("I-2", "frame aggregation", F_BLUE),
          ("I-3", "loss weighting", F_BLUE), ("I-4", "augmentation", F_BLUE), ("I-5", "combining encoders", F_BLUE),
          ("I-6", "late fusion", F_BLUE), ("I-7", "rPPG", F_ORANGE), ("I-8", "freeze", F_RED)]
gap = 90000
lw = (W - 8 * gap) // 9
for i, (lab, txt, f) in enumerate(labels):
    x = L + i * (lw + gap)
    dbox(s, x, TOP, lw, 1150000, "**" + lab + "**" + chr(10) + txt, fill=f, size=C.SZ_TABLE)
    if i:
        darrow(s, x - gap, TOP + 575000, x, TOP + 575000)
add_card(s, L, TOP + 1350000, W, 3450000, [
    ("h", "Levers and the diagnostic battery"),
    ("bs", "I-1 to I-6 take their lever from these families, chosen after the battery of the preceding iteration: fine-tuning depth, frame aggregation, loss and class weighting, augmentation and regularisation, combining encoders, late fusion and calibration.", 18),
    ("bs", "I-7 is the rPPG experiment with thresholds written beforehand; I-8 is integration and freeze and runs no experiments.", 18),
    ("bs", "Battery after every iteration: per-dataset and per-generator AUC with intervals; train versus validation and validation versus development gaps; calibration (ECE, Brier); score histograms; splits by compression, face size and number of faces; the 20 worst errors; seed variance.", 18),
    ("bs", "Frozen iterations so far: none. Each gets its own slide once it is frozen.", 18),
], name="card_ladder")

# =====================================================================================================================
# SLIDE 11: rPPG
# =====================================================================================================================
s = new_slide("rPPG: test the signal first",
    "Remote photoplethysmography is a candidate improvement, not an adopted component (D-006, O-010). "
    "The literature is mixed: early work reports high in-dataset accuracy, a 2025 study found a plausible pulse in high-quality face swaps, a 2026 preprint reported a reproduction that fell lower on talking-face fakes under an identity-disjoint split, and pulse signals can be edited on purpose (R-005 to R-010). "
    "So the first step is a signal check on contiguous frames with a time-shuffled control pooled across clips (at least ten shuffles per clip); it passes when at least 60 percent of real clips have a peak SNR above the 95th percentile of the control. "
    "If it fails, the slot reports exactly that and no separation or fusion experiment is run; if it passes, the adopt decision is taken on the development slice only and needs the whole paired interval above zero. "
    "Evidence: docs/ITERATIONS.md section I-7, thresholds written before any run; results are [pending] (experiment I-7 entry, none yet).")
gap = 130000
bw7 = (W - 6 * gap) // 7
bh7 = 900000
ry = TOP
chain = ["Contiguous frames, native rate (about 10 s)", "Face and skin region", "Mean R, G, B per frame", "Window normalisation", "Projection to one pulse signal",
         "Power spectrum 0.7 to 3 Hz", "Peak and SNR"]
for i, t in enumerate(chain):
    x = L + i * (bw7 + gap)
    dbox(s, x, ry, bw7, bh7, t, fill=F_BLUE)
    if i:
        darrow(s, x - gap, ry + bh7 // 2, x, ry + bh7 // 2)
ry2 = ry + bh7 + 280000
ctrl_x = L + 2 * (bw7 + gap)
ctrl_w = 3 * bw7 + 2 * gap
dbox(s, ctrl_x, ry2, ctrl_w, 700000, "Control: the same clips with time-shuffled frames (pooled, at least 10 shuffles per clip)", fill=F_GREY)
res_x = L + 5 * (bw7 + gap)
res_w = 2 * bw7 + gap
dbox(s, res_x, ry2, res_w, 700000, "Pass, or report: pulse not recoverable", fill=F_ORANGE)
darrow(s, L + 3 * (bw7 + gap) + bw7 // 2, ry + bh7, L + 3 * (bw7 + gap) + bw7 // 2, ry2, dashed=True)
darrow(s, ctrl_x + ctrl_w, ry2 + 350000, res_x, ry2 + 350000)
darrow(s, L + 6 * (bw7 + gap) + bw7 // 2, ry + bh7, res_x + res_w - bw7 // 2, ry2)
ty = ry2 + 700000 + 200000
add_text(s, col_x(0), ty, span_w(7), BOT - ty, [
    ("b", "Signal check passes when at least 60 percent of real clips have a peak SNR above the 95th percentile of the shuffled control; if not, the slot reports that the pulse is not recoverable.", 18),
    ("b", "Separation, only if it passes: lower interval bound of the AUC above 0.55 on an identity-disjoint split.", 18),
    ("b", "Adopt only if the whole paired interval (with minus without) is above zero on the development slice.", 18),
], "rppg_text")
add_card(s, col_x(7), ty, span_w(5), BOT - ty, [
    ("h", "Outcome"),
    ("p", "Signal check: [pending]", 20),
    ("p", "Separation: [pending]", 20),
    ("p", "Added value: [pending]", 20),
], name="card_rppg")

# =====================================================================================================================
# SLIDE 12: web demo
# =====================================================================================================================
s = new_slide("Web demo (development stand-in)",
    "The software is a local web page served by Flask on 127.0.0.1 only, with a 512 MB upload limit; uploaded files are deleted after every request (D-043). "
    "It calls an inference library with a fixed response structure: a score per modality, a status, warnings, errors, a model block and a fixed sentence that says what the score means (D-027). "
    "The three screenshots were taken against the development stand-in, which always reports its source as mock; the page then shows the MOCK banner, so the numbers in the screenshots are not measurements. "
    "The real adapter refuses to run without a checkpoint whose metadata and preprocessing identifier match (D-044). "
    "For a video the visual and the audio score are shown separately and the verdict is undetermined, because no combining rule is claimed. Evidence: src/dfd/web, src/dfd/infer, docs/DOC_FACTS.md section 10; screenshots in assets/screens.")
ph = 2650000
crops = [0.12, 0.20, 0.20]          # share of the page cut off at the bottom (the details block)
asp = [1.531, 0.835, 0.813]
wds = [int(ph * a_ / (1 - c)) for a_, c in zip(asp, crops)]
g2 = (W - sum(wds)) // 2
xs = [L, L + wds[0] + g2, L + wds[0] + wds[1] + 2 * g2]
caps = ["(a) start page", "(b) image result", "(c) video result"]
for xx, fn, wd, cr, alt in zip(xs, ["web_start.png", "web_image.png", "web_video.png"], wds, crops,
                               ["Start page of the local web page", "Image result page", "Video result page"]):
    pic = add_pic(s, os.path.join(SCR, fn), xx, TOP, alt=alt + " (development stand-in, MOCK banner)")
    pic.crop_bottom = cr
    pic.left, pic.top, pic.width, pic.height = Emu(xx), Emu(TOP), Emu(wd), Emu(ph)
for xx, wd, cp in zip(xs, wds, caps):
    add_text(s, xx, TOP + ph + 40000, wd, 280000, [("cap", cp)], "pic_caption")
ty = TOP + ph + 400000
add_text(s, L, ty, W, BOT - ty, [
    ("bs", "Screenshots taken against the development stand-in: the MOCK banner is part of the design and the numbers are not measurements.", 17),
    ("bs", "Local Flask page on 127.0.0.1, uploads up to 512 MB, files deleted after each request; it shows a probability per modality, warnings and a fixed sentence on what the score means.", 17),
    ("bs", "With a real checkpoint the page shows real scores: [pending].", 17),
], "web_text")

# =====================================================================================================================
# SLIDE 13: timeline
# =====================================================================================================================
s = new_slide("Timeline to 30 October",
    "The schedule runs from Saturday 24 October to the submission on Friday 30 October 2026 (D-009, PLAN.md). "
    "Saturday was decisions, data and the evaluation harness; Sunday is the encoder sweep and the working-version code path; Monday the working version and its diagnostics; Tuesday I-1 to I-3; Wednesday I-4 to I-7 with the rPPG decision; Thursday integration and freeze with the one-time sealed evaluations; Friday assembly and submission. "
    "Experiments end on Wednesday night, and an iteration that is not finished by then is reported as not run, never as a result. "
    "The scope is not cut: each iteration is complete in itself (hypothesis, run, result, decision), so if work stops after any day the story up to the last finished iteration is still valid. "
    "Evidence: docs/PLAN.md day-by-day schedule and docs/ITERATIONS.md schedule table.")
lab_w = span_w(3)
days = ["Sat 24", "Sun 25", "Mon 26", "Tue 27", "Wed 28", "Thu 29", "Fri 30"]
dw = (W - lab_w) // 7
hdr_y = TOP
for i, d in enumerate(days):
    add_text(s, L + lab_w + i * dw, hdr_y, dw, 330000, [("s", f"**{d}**", 17)], "dg_day")
rows_t = [("Data, harness, decisions", 0, 1, F_GREY), ("Encoder sweep", 1, 2, F_BLUE), ("Working version (I-0)", 1.5, 3, F_GREEN),
          ("I-1 to I-3", 3, 4, F_BLUE), ("I-4 to I-7 (with rPPG)", 4, 5, F_BLUE), ("Integration and freeze", 5, 6, F_RED), ("Assembly, submission", 6, 7, F_ORANGE)]
rh = 500000
ry0 = hdr_y + 420000
for k, (lab, a_, b_, f) in enumerate(rows_t):
    y = ry0 + k * rh
    add_text(s, L, y, lab_w - 60000, rh - 60000, [("s", lab, 17)], "dg_rowlab", anchor=MSO_ANCHOR.MIDDLE)
    dbox(s, int(L + lab_w + a_ * dw + 25000), y + 30000, int((b_ - a_) * dw - 50000), rh - 120000, "", fill=f, name="dg_bar")
for i in range(8):     # day grid lines
    gx = L + lab_w + i * dw
    gl = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(gx), Emu(ry0), Emu(gx), Emu(ry0 + 7 * rh))
    gl.name = _name("arrow_grid")
    gl.line.color.rgb = rgb("D9D9D9"); gl.line.width = Pt(0.75)
add_text(s, L, ry0 + 7 * rh + 150000, W, 700000, [
    ("s", "An iteration not finished by the end of Wednesday is reported as not run. No new experiments on the last day.", 18)], "timeline_note")

# =====================================================================================================================
# SLIDE 14: limitations
# =====================================================================================================================
s = new_slide("Limitations",
    "These limits are known before any final run and are stated now. "
    "The development slices are small (15 identities, 12 speakers, two generator units), so intervals are wide and ties are likely; the Celeb-DF family test has about 44 independent identity groups, so its intervals stay wide whatever the video count (D-045). "
    "Frozen features may be inadequate for face forgery, and the research-only weights of some encoders restrict reuse. "
    "Cross-dataset numbers quoted from papers use other protocols and are context only. "
    "rPPG may not be recoverable from compressed video, and nothing in this project is claimed beyond what its interval supports. Evidence: docs/DOC_FACTS.md section 13, docs/DATA_LOG.md Celeb-DF++ section.")
add_text(s, L, TOP + 50000, W, 4750000, [
    ("b", "Small development slices (15 identities, 12 speakers, two generator units): intervals are wide and ties are likely.", 20),
    ("b", "The Celeb-DF family test has about 44 independent identity groups; intervals stay wide whatever the video count.", 20),
    ("b", "Frozen features may be inadequate for face forgery; the fine-tuning iteration tests this.", 20),
    ("b", "Two encoder weights (OpenAI CLIP, WavLM) are conditionally licensed; they are accepted as research-only, which restricts reuse.", 20),
    ("b", "Cross-dataset numbers from papers use other protocols and are context only, never a comparison.", 20),
    ("b", "rPPG may not be recoverable from compressed video.", 20),
    ("b", "Nothing is claimed beyond what its interval supports; an iteration that does not finish is reported as not run.", 20),
], "limits")

# =====================================================================================================================
# SLIDE 15: conclusion
# =====================================================================================================================
s = new_slide("Conclusion",
    "This slide is a placeholder for the conclusion and is filled only from the logged experiments after the freeze on Thursday 29 October. "
    "What already stands without any result is the design: a group-disjoint protocol with sealed test sets, a rule that prefers the cheaper candidate when evidence ties, a staircase of one sealed evaluation per frozen iteration, and a working-version definition that requires the whole chain to run. "
    "The two confirmatory claims are declared in advance: the final system against the working version, and the final system against a re-run published baseline, the latter judged by non-inferiority with delta fixed beforehand (D-025, D-041). "
    "Every other comparison is labelled exploratory. Evidence: docs/PROTOCOL.md addendum, docs/ITERATIONS.md; the cells below are [pending] until the ledger entries of the freeze exist.")
add_text(s, col_x(0), TOP, span_w(6), 4800000, [
    ("h", "What stands without any result"),
    ("b", "A group-disjoint protocol with sealed test sets and a ledger.", 20),
    ("b", "A rule that prefers the cheaper candidate when the evidence ties.", 20),
    ("b", "One sealed evaluation per frozen iteration (the staircase).", 20),
    ("b", "Two confirmatory claims, declared before any final run.", 20),
], "concl_left")
gt, th = add_table(s, col_x(6), TOP, [span_w(4), span_w(2)], ["Claim", "Outcome"], [
    ["Working version, development AUC", "[pending]"],
    ["Final system versus working version (paired, sealed tests)", "[pending]"],
    ["Final system versus re-run baseline (non-inferiority)", "[pending]"],
    ["rPPG: adopt or not", "[pending]"],
], size="auto", max_h=3600000, pad=60000)
add_text(s, col_x(6), TOP + th + 140000, span_w(6), 900000, [("s", "An iteration that is not finished is reported as not run.")], "concl_note")

# =====================================================================================================================
# SLIDE 16: references
# =====================================================================================================================
REFS = [
    "Cai, Z. et al. (2022). Do you really mean that? Content driven audio-visual deepfake dataset (LAV-DF). DICTA.",
    "Chugh, K. et al. (2020). Not made for each other: audio-visual dissonance-based deepfake detection and localization. ACM Multimedia.",
    "Ciftci, U. A., Demir, I., Yin, L. (2020). FakeCatcher: detection of synthetic portrait videos using biological signals. IEEE TPAMI.",
    "Dolhansky, B. et al. (2020). The DeepFake Detection Challenge (DFDC) dataset. arXiv:2006.07397.",
    "Dong, S. et al. (2023). Implicit identity leakage: the stumbling block to improving deepfake detection generalization. CVPR.",
    "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML, PMLR 70.",
    "Harraq, O., Aldwairi, T. (2026). Physiological signals as a forensic modality for talking-face deepfake detection. arXiv:2607.21776.",
    "Hernandez-Ortega, J. et al. (2021). DeepFakesON-Phys: DeepFakes detection based on heart rate estimation. AAAI Workshops.",
    "Jung, J. et al. (2022). AASIST: audio anti-spoofing using integrated spectro-temporal graph attention networks. ICASSP.",
    "Khalid, H. et al. (2021). FakeAVCeleb: a novel audio-video multimodal deepfake dataset. NeurIPS Datasets and Benchmarks.",
    "Kornblith, S., Shlens, J., Le, Q. V. (2019). Do better ImageNet models transfer better? CVPR.",
    "Li, Y. et al. (2020a). Celeb-DF: a large-scale challenging dataset for DeepFake forensics. CVPR.",
    "Li, L. et al. (2020b). Face X-ray for more general face forgery detection. CVPR.",
    "Li, Y. et al. (2025). Celeb-DF++: a large-scale challenging video DeepFake benchmark for generalizable forensics. arXiv:2507.18015.",
    "Lin, Y. et al. (2023). Detection of photoplethysmography manipulation in video forgery. EUSIPCO.",
    "Muller, N. M. et al. (2022). Does audio deepfake detection generalize? Interspeech.",
    "Ojha, U., Li, Y., Lee, Y. J. (2023). Towards universal fake image detectors that generalize across generative models. CVPR.",
    "Rossler, A. et al. (2019). FaceForensics++: learning to detect manipulated facial images. ICCV.",
    "Seibold, C. et al. (2025). High-quality deepfakes have a heart! Frontiers in Imaging 4. doi:10.3389/fimag.2025.1504551.",
    "Tak, H. et al. (2022). Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation. Odyssey.",
    "Wang, X. et al. (2020). ASVspoof 2019: a large-scale public database of synthesized, converted and replayed speech. Computer Speech and Language. doi:10.1016/j.csl.2020.101114.",
    "Yan, Z. et al. (2023). DeepfakeBench: a comprehensive benchmark of deepfake detection. NeurIPS Datasets and Benchmarks.",
]
s = new_slide("References",
    "Every source cited on a slide is listed here with the author, the year and the venue (or the arXiv identifier where no venue was verified), in the form used by the paper's bibliography. "
    "The entries agree with paper/refs.bib, so the paper and the slides cite the same years and venues. "
    "All entries have status READ or PARTIAL in the project's literature matrix; the parts that were read are recorded there, and numbers quoted on the slides come from those parts. "
    "The in-text form is author and year, with 2020a and 2020b separating the two Li et al. papers of 2020. "
    "Evidence: docs/RESEARCH.md and paper/refs.bib; slides/NEW_REFERENCES.md records how the list was checked.")
half_n = (len(REFS) + 1) // 2
for k, chunk in enumerate([REFS[:half_n], REFS[half_n:]]):
    add_text(s, col_x(6 * k), TOP, span_w(6), 4800000, [("cap", r) for r in chunk], "refs")
# give the reference paragraphs some air
for sh in s.shapes:
    if sh.name.startswith("refs"):
        for p in sh.text_frame.paragraphs:
            set_spacing(p, 0, 3)
            for r in p.runs:
                r.font.italic = False
                r.font.color.rgb = rgb(DARK)
        need = C.estimate_text_height(sh)
        if need > sh.height:
            WARN.append(f"references: {sh.name} needs {need/914400:.2f} in, has {sh.height/914400:.2f} in")

# =====================================================================================================================
# BACKUP
# =====================================================================================================================
s = new_slide("Backup slides",
    "The backup slides hold the details behind the main line: the statistics and decision rules, the staircase of sealed evaluations, the preprocessing, the cost record and the fine-tuning budget, the data licences, the Celeb-DF++ de-duplication and test balance, and the measured environment. "
    "Use them to answer why-this-and-not-that questions: each one names the rule, the reason and the evidence file. "
    "Nothing on these slides is a result unless it is a logged measurement; missing numbers are [pending]. "
    "The order follows the order of the main slides so that a question can be matched to its detail quickly. "
    "Evidence: docs/VIVA_PREP.md lists the likely questions with their answers.")
add_text(s, L, TOP, W, 4800000, [
    ("h", "Contents"),
    ("b", "**Statistics and decision rules:** metrics, bootstrap, seed policy, non-inferiority, tie rule"),
    ("b", "**Staircase and confirmatory claims:** one sealed evaluation per frozen iteration"),
    ("b", "**Preprocessing:** frames, face crops, audio window"),
    ("b", "**Cost record and fine-tuning budget:** cost axes, fit rule for fine-tuning"),
    ("b", "**Data licences and terms:** what may and may not be shown"),
    ("b", "**Celeb-DF++ test balance:** de-duplication and the pooled family test"),
    ("b", "**Environment (measured):** hardware and software"),
], "backup_contents")

s = new_slide("Statistics and decision rules",
    "Every interval resamples groups, not frames, with 2,000 resamples, a 95 percent percentile interval and a fixed base seed, because the units of independence are source videos and identities (D-024, PROTOCOL.md section 6). "
    "Seeds are policy, not decoration: the head of every candidate in the sweep is trained with three seeds, while the fine-tuning budget is one seed per learning rate and two further seeds for the winner only (D-036, D-042). "
    "Comparable means non-inferior: the lower end of the paired interval of (ours minus the re-run baseline) is above minus delta, with delta stated before the final comparison and chosen from the measured seed spread of the baseline; an interval that only contains zero is reported as not distinguished (D-025). "
    "The tie rule for the encoder sweep is the same idea turned around: if the paired interval against the leader includes zero, the cheaper candidate wins. "
    "Evidence: docs/PROTOCOL.md sections 4 to 6, docs/MATH_NOTES.md section 8, docs/DECISIONS.md D-025, D-036, D-042.")
add_text(s, col_x(0), TOP, span_w(6), 4800000, [
    ("h", "Metrics", 19),
    ("b", "Headline: ROC-AUC on a dataset never used in training, with a 95 percent group-bootstrap interval.", 19),
    ("b", "Also EER, balanced accuracy at the validation threshold, log loss, Brier score and ECE (15 equal-width bins) before and after temperature scaling.", 19),
    ("b", "Bootstrap: 2,000 resamples, percentile interval, groups resampled with replacement.", 19),
], "stats_left")
add_text(s, col_x(6), TOP, span_w(6), 4800000, [
    ("h", "Decision rules", 19),
    ("b", "Seeds: three for the head of every candidate in the sweep; in fine-tuning, one seed per learning rate and two further seeds for the winner.", 19),
    ("b", "Comparable with a published detector: non-inferior, lower end of the paired interval above minus delta; delta fixed beforehand.", 19),
    ("b", "Superior only when the whole interval is above zero; an interval that only contains zero is reported as not distinguished.", 19),
    ("b", "Encoder choice: the tie set against the leader, then the cheapest wins.", 19),
], "stats_right")

# staircase
s = new_slide("Staircase and confirmatory claims",
    "Verdicts on the development slice are exploratory: they report sign, size and interval and they choose the next step, never a confirmed performance (D-041). "
    "At the freeze each frozen iteration is evaluated once on the sealed test sets, which gives one test score per iteration, a staircase, and each evaluation is written to the ledger. "
    "The confirmatory claims are exactly two and were declared before any final run: the final system against the working version, and the final system against the re-run published baseline, both as paired group-bootstrap comparisons on sealed data. "
    "Every other comparison is labelled exploratory. "
    "Evidence: docs/PROTOCOL.md addendum, ledger/final_ledger.jsonl, docs/ITERATIONS.md.")
nb = 5
bwid = 1450000
gap = 150000
base_y = TOP + 3900000
heights = [1300000, 1800000, 2300000, 2800000, 3300000]
names = ["I-0", "I-1", "I-2", "...", "final"]
for i in range(nb):
    x = L + i * (bwid + gap)
    f = F_GREEN if names[i] == "final" else F_BLUE
    dbox(s, x, base_y - heights[i], bwid, heights[i], "**" + names[i] + "**" + chr(10) + "frozen, evaluated once", fill=f, anchor=MSO_ANCHOR.TOP, size=C.SZ_BODY_S)
cx = L + nb * (bwid + gap) + 100000
cwid = L + W - cx
dbox(s, cx, base_y - 3300000, cwid, 1000000, "**final versus working version**", fill=F_ORANGE, size=C.SZ_BODY_S)
dbox(s, cx, base_y - 1900000, cwid, 1000000, "**final versus re-run baseline**" + chr(10) + "non-inferiority", fill=F_ORANGE, size=C.SZ_BODY_S)
darrow(s, L + 4 * (bwid + gap) + bwid, base_y - 2900000, cx, base_y - 2800000)
darrow(s, L + 4 * (bwid + gap) + bwid, base_y - 1700000, cx, base_y - 1400000)
add_text(s, L, base_y + 100000, W, 800000, [
    ("s", "Development slice: exploratory verdicts (sign, size, interval). Sealed test sets: one evaluation per frozen iteration, written to a ledger. Two pre-declared confirmatory claims.")], "stair_note")

# preprocessing
s = new_slide("Preprocessing",
    "One shared package prepares inputs for training, evaluation and inference, so that the three cannot drift apart; a configuration hash (the preprocess identifier) is stored in every checkpoint and checked at load (D-028, D-044). "
    "For video: 16 uniformly spaced frames, the YuNet face detector (MIT licence), the largest face enlarged by 1.3, a 224 by 224 crop and JPEG quality 95; frames without a face are left out and reported. "
    "A detector comparison on 300 FaceForensics++ validation frames and 300 Celeb-DF development frames found a face in every frame for three detectors and no geometric reason to re-crop; YuNet takes about 12 ms per frame (M-002). "
    "Images use a face crop if a face is found and otherwise the whole image with the encoder's own rule; audio uses the first 64,600 samples at 16 kHz, repeat-padded, and the Whisper encoder gets its fixed 30 second window with only the frames of the first 4 seconds pooled. "
    "Evidence: docs/DOC_FACTS.md section 4, docs/DECISIONS.md D-028, D-033, D-034, D-044, docs/EXPERIMENTS.md M-002.")
gt, th = add_table(s, L, TOP, [span_w(2), W - span_w(2)], ["Input", "Rule"], [
    ["Video", "16 uniformly spaced frames; YuNet face detector (MIT licence); largest face enlarged by a factor of 1.3; 224 by 224 crops, JPEG quality 95; frames without a face are left out and reported"],
    ["Image", "A face crop if a face is found, otherwise the whole image processed by the encoder's own rule"],
    ["Audio", "First 64,600 samples at 16 kHz, repeat-padded; the Whisper encoder gets its fixed 30 second window with the first 4 seconds first and only those frames pooled"],
    ["Consistency", "One shared package for training, evaluation and inference; a configuration hash is stored in every checkpoint and checked at load"],
], size="auto", bold_first_col=True, max_h=3300000, fill=3300000)
add_text(s, L, TOP + th + 140000, W, 1100000, [
    ("s", "A face-detector comparison on 300 FaceForensics++ validation frames and 300 Celeb-DF development frames found a face in every frame for all three detectors tried.")], "prep_note")

# cost record
s = new_slide("Cost record and fine-tuning budget",
    "Accuracy is never reported alone: every candidate gets a cost record of parameters, FLOPs, peak memory and model-only and end-to-end time per 1,000 inputs, generated from the extraction records (D-036, O007_SWEEP section 7). "
    "The cost values are [pending] here because the clean measurement is still being re-run; parameters are read from the weight headers. "
    "The fine-tuning budget of the video task follows measured step times (M-003, which was partly overlapped with other jobs and is superseded by a clean re-run): three encoders at three learning rates with one seed each under a per-run time cap, then two further seeds for the winner only. "
    "An encoder enters fine-tuning only if its peak memory at batch 32 is at most 10 GiB and one budget takes at most 10 minutes, which excludes the two CLIP ViT-L/14 models on the 12 GB card (D-042). "
    "Evidence: docs/DECISIONS.md D-042, docs/EXPERIMENTS.md M-003 and the model comparison tables.")
gt, th = add_table(s, L, TOP, [span_w(4), span_w(2), span_w(2), W - span_w(4) - 2 * span_w(2)], ["Candidate (images)", "Parameters", "Peak memory", "End-to-end time per 1,000 inputs"], [
    ["EfficientNet-B4", "17.5 M", "[pending]", "[pending]"],
    ["Xception", "20.8 M", "[pending]", "[pending]"],
    ["DINOv2 small", "22.1 M", "[pending]", "[pending]"],
    ["DINOv2 base", "86.6 M", "[pending]", "[pending]"],
    ["CLIP ViT-L/14 (OpenAI, LAION)", "304.0 M", "[pending]", "[pending]"],
], size="auto", bold_first_col=True, max_h=2600000, fill=2600000)
add_text(s, L, TOP + th + 120000, W, BOT - (TOP + th + 120000), [
    ("b", "Fine-tuning of the video task: three encoders, three learning rates, one seed each, a per-run time cap; two further seeds for the winner only.", 18),
    ("b", "An encoder enters if its peak memory at batch 32 is at most 10 GiB and one budget takes at most 10 minutes; the two CLIP ViT-L/14 models do not fit the 12 GB card.", 18),
], "cost_text")

# licences
s = new_slide("Data licences and terms",
    "Every dataset and pretrained weight has its licence, source, version and hash recorded in the data log before use; all datasets are research-only. "
    "Celeb-DF, Celeb-DF++ and FakeAVCeleb forbid publishing derived data, so no frame, crop or per-video score of them appears in any document of this project. "
    "FaceForensics++ is licensed for non-commercial research and education, ASVspoof 2019 LA is under an open data licence (ODC-By as read from its README), and the ProGAN set carries a CC BY-NC-SA 4.0 card tag while the terms of its LSUN images are not established. "
    "The conditional weights licences of OpenAI CLIP and WavLM are accepted as research-only and appear as a limitation. "
    "Evidence: docs/DATA_LOG.md sections Downloaded datasets, Keep drop or defer, Encoder weights.")
gt, th = add_table(s, L, TOP, [span_w(3), span_w(4), W - span_w(3) - span_w(4)], ["Data or weights", "Terms as recorded", "Consequence"], [
    ["FaceForensics++", "Non-commercial research and education only", "Used for training; no frames shown"],
    ["Celeb-DF v2, Celeb-DF++", "Non-commercial research; no publication of derived data", "Sealed tests; no frame, crop or per-video score in any document"],
    ["ASVspoof 2019 LA", "ODC-By (README)", "Audio training and validation"],
    ["ProGAN validation and test sets", "CC BY-NC-SA 4.0 card tag; LSUN image terms not established", "Generated-image training and tests"],
    ["OpenAI CLIP, WavLM weights", "No clean licence statement; conditional", "Accepted as research-only; stated as a limitation"],
], size="auto", bold_first_col=True, max_h=H - 500000, fill=H - 500000)

# celeb-df++
s = new_slide("Celeb-DF++ test balance",
    "Celeb-DF++ is one family with Celeb-DF v2, so no source video or identity of the development slice may be counted twice. "
    "2,079 Celeb-DF++ samples whose target or source identity is one of the 15 development identities are excluded from the sealed test set, and 150 real videos that are byte-identical to Celeb-DF v2 videos are counted once, under their Celeb-DF v2 group id. "
    "The result is a Celeb-DF++ sealed test set of 2,849 fake videos and no real video in 44 identity groups; its AUC is therefore computed only together with the Celeb-DF v2 real videos, and the pooled family test holds 744 real and 6,906 fake videos (89 percent fake). "
    "The group bootstrap resamples identities, so the extra fakes add generator coverage but almost no new independent groups: about 44 identity groups govern the intervals, and balanced accuracy and the threshold carry more noise than the AUC. "
    "Evidence: docs/DECISIONS.md D-045, docs/DATA_LOG.md Celeb-DF++ de-duplication update, split celebdf_pp-3f9df077b9eb.")
steps = [("Celeb-DF++ test videos extracted", "5,078"), ("Excluded: development identities", "2,079"), ("Excluded: duplicate real videos", "150"),
         ("Sealed test, Celeb-DF++ alone", "2,849 fake, 0 real")]
sw = (W - 3 * 150000) // 4
for i, (a, b) in enumerate(steps):
    x = L + i * (sw + 150000)
    f = [F_BLUE, F_RED, F_RED, F_GREEN][i]
    dbox(s, x, TOP, sw, 1250000, f"{a}\n**{b}**", fill=f, size=C.SZ_BODY_S)
    if i:
        darrow(s, x - 150000, TOP + 625000, x, TOP + 625000)
gt, th = add_table(s, L, TOP + 1500000, [span_w(5), span_w(2), span_w(2), W - span_w(5) - 2 * span_w(2)], ["Pooled family test (sealed)", "Real", "Fake", "Groups"], [
    ["Celeb-DF v2", "744", "4,057", "about 44 identities, plus YouTube videos"],
    ["Celeb-DF++ (new generators)", "0", "2,849", "44 identity groups"],
    ["Pooled family test", "744", "6,906", "about 44 identity groups on the fake side"],
], size="auto", bold_first_col=True, max_h=2300000, fill=2300000)
add_text(s, L, TOP + 1500000 + th + 130000, W, 900000, [
    ("s", "Real is 10.8 percent of the pool. The 44 identity groups govern the intervals, and balanced accuracy and the threshold carry more noise than the AUC.")], "pp_note")

# environment
s = new_slide("Environment (measured)",
    "Every experiment names its environment, and this one was measured on the machine itself on 24 October 2026 rather than copied from a description. "
    "The user described the GPU as an RTX 5070 class card with 8 or 12 GB; the measurement shows 12 GB reported by nvidia-smi, about 11.94 GiB as seen by PyTorch, and about 1 GB already used by desktop applications at idle. "
    "The CPU is a Ryzen 9 7900X, not a 7900K as first described. "
    "Plans for fine-tuning are made inside the 11 GiB usable, which is why the two large CLIP models do not fit. "
    "Evidence: docs/EXPERIMENTS.md, section Environment, Hardware and Software tables (environment label ENV-DESKTOP-GPU).")
gt, th = add_table(s, L, TOP, [span_w(3), W - span_w(3)], ["Item", "Measured value"], [
    ["GPU", "NVIDIA GeForce RTX 5070, 12 GB (11.94 GiB seen by PyTorch), compute capability 12.0, driver 595.79"],
    ["CPU and memory", "AMD Ryzen 9 7900X, 12 cores, 24 threads; 31 GB RAM"],
    ["Operating system", "Windows 10 Pro, version 10.0.19045"],
    ["Python and PyTorch", "Python 3.13.15 under uv; PyTorch 2.14.1 with CUDA 13.0 wheels"],
    ["Web page", "Flask 3.1.3, bound to 127.0.0.1"],
], size="auto", bold_first_col=True, max_h=H - 500000, fill=H - 900000)


MIRROR = {4: "fig02_splits", 5: "fig01_pipeline", 8: "fig08_shortlist", 10: "fig03_ladder", 11: "fig05_rppg", 13: "fig06_timeline", 19: "fig04_staircase"}
for _i, _sl in enumerate(prs.slides, 1):
    if _i in MIRROR:
        _tf = _sl.notes_slide.notes_text_frame
        _tf.text = _tf.text + " This slide is drawn natively and mirrors " + MIRROR[_i] + " of the paper and the report, with the same wording, so that the text is large enough to read when projected."
cp = prs.core_properties
cp.title = 'Multimodal Deepfake Detection for Images, Video and Audio: A Hypothesis-Driven Study Under a Source-Disjoint Protocol'
cp.author = 'Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh'
cp.last_modified_by = 'Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh'
cp.subject = 'Final year project presentation'
cp.keywords = ''
cp.comments = ''
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
for w in WARN:
    print("WARN", w)
