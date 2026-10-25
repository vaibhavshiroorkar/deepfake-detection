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



def fit_size(widths, header, rows, sizes=(14, 13, 12), bold_first_col=False):
    """Largest size from sizes at which no cell wraps (so that compact tables stay one line per row)."""
    for sz in sizes:
        ok = True
        for ri, row in enumerate([header] + rows):
            for ci, val in enumerate(row):
                b = ri == 0 or (bold_first_col and ci == 0)
                if C.text_width_pt(val, sz, b) * 12700 > widths[ci] - 2 * CELL_MX - 20000:
                    ok = False
        if ok:
            return sz
    return sizes[-1]


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
for i, (txt, sz, it) in enumerate([("A Hypothesis-Driven Study Under a Group-Disjoint Protocol", 20, True),
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
    "Say at once that no sealed-test result exists yet: the iterations have exploratory results on the development slice, and every sealed-test number that is still missing is shown as [pending] on the slide and is filled only from a logged experiment. "
    "Evidence pointers: docs/IDENTITY.md for the identity fields, docs/DOC_FACTS.md for the wording, docs/PLAN.md for the schedule. "
    "The title and the subtitle (Group-Disjoint Protocol) are the decided title of the project and are the same in the paper and the report.")

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
    "Cost values come from the section 7 cost record of 2026-10-25 (data/_results/cost_record.json): inference in fp16 at batch 64 (speech batch 32) on the RTX 5070, FLOPs per forward pass, end-to-end time per 1,000 inputs on real files (face frames for images, 224 px except EfficientNet-B4 320 and Xception 299, speech 4 s clips). Parameter counts are read from the weight files and the model comparison tables (E-001 to E-003): Whisper small encoder 88,154,112, HuBERT base 94,371,712, the from-scratch references 389,633 (images) and 389,057 (speech). "
    "Evidence: docs/O007_SWEEP.md section 3.1, docs/DATA_LOG.md encoder weights section, data/_results/*/model_comparison.md.")
cw = [span_w(1) + 450000, span_w(3) - 200000, span_w(2) + 150000, span_w(1), span_w(1) + 50000, span_w(1)]
cw.append(W - sum(cw))
hdr_s = ["Task", "Encoder", "Family", "Params", "GFLOPs", "s per 1,000", "Weights licence"]
rows = [
    ["Images, video frames", "EfficientNet-B4", "supervised CNN", "17.5 M", "6.1", "1.73", "research use (ImageNet)"],
    ["", "Xception", "supervised CNN", "20.8 M", "16.7", "3.32", "research use (ImageNet)"],
    ["", "DINOv2 small", "self-supervised ViT", "22.1 M", "12.2", "0.94", "Apache-2.0"],
    ["", "DINOv2 base", "self-supervised ViT", "86.6 M", "46.3", "1.67", "Apache-2.0"],
    ["", "CLIP ViT-L/14 (OpenAI)", "image-text contrastive", "304.0 M", "162.0", "4.45", "research use (accepted)"],
    ["", "CLIP ViT-L/14 (LAION)", "image-text contrastive", "304.0 M", "162.0", "4.03", "MIT tag, research use"],
    ["", "SigLIP base", "sigmoid image-text", "92.9 M", "35.4", "1.44", "Apache-2.0"],
    ["Speech audio", "wav2vec 2.0 base", "self-supervised", "94.4 M", "57.5", "2.04", "Apache-2.0"],
    ["", "WavLM base plus", "self-supervised", "94.4 M", "57.5", "2.53", "research use (accepted)"],
    ["", "Whisper small encoder", "weakly supervised ASR", "88.2 M", "344.3", "11.53", "Apache-2.0 / MIT"],
    ["", "HuBERT base", "self-supervised", "94.4 M", "57.5", "2.04", "Apache-2.0"],
    ["Both", "Small network from scratch", "reference, excluded", "0.39 M", "n/a", "n/a", "own code"],
]
gt, th = add_table(s, L, TOP, cw, hdr_s, rows, size="auto", bold_first_col=True, max_h=H - 800000, fill=H - 800000)
add_text(s, L, TOP + th + 90000, W, 620000, [("cap", "Parameters from the weight files; GFLOPs per forward pass; s per 1,000 inputs end to end (face frames for images). Inference, fp16, batch 64 (speech 32), RTX 5070. DINOv3 is excluded on purpose (manual access gate).")], "short_note")

# =====================================================================================================================
# SLIDE 9a: sweep results
# =====================================================================================================================
s = new_slide("Encoder sweep on the development slice",
    "The sweep ranked every shortlisted encoder per task on the development slice with the rule fixed beforehand; all values are exploratory (D-041) and are the final results of experiments E-001 (face frames), E-002 (images) and E-003 (speech), recorded in D-049 and in data/_results/T-V, T-I and T-A. "
    "Face frames: five of seven candidates form the tie set against the leader, so the cheapest of them, DINOv2 small, wins (0.943 s per 1,000 inputs end to end, 22.1 M parameters); EfficientNet-B4 is outside the tie set by the bootstrap (difference -0.099 [-0.172, -0.023]) but the permutation p-value is 0.110, and the winner would not change. "
    "Images: the tie set has one member, CLIP ViT-L/14 (OpenAI), but the development slice has only two generator units, so its interval is not a population interval; per-generator AUC is in the backup (M-004). "
    "Speech: the tie set has three members and HuBERT base wins on cost, yet the mechanical verdict is inconclusive (best lower bound 0.534, at or below 0.55), the from-scratch reference is below chance (0.246) and the development ECE rose after temperature scaling for every audio encoder. "
    "Evidence: docs/EXPERIMENTS.md E-001 to E-003, docs/DECISIONS.md D-036 and D-049, docs/DOC_FACTS.md section 14.")
add_text(s, L, TOP, W, 300000, [("cap", "Exploratory, development slice. AUC on the development slice: mean of three head seeds with the 95 percent group-bootstrap interval. Tie = tied with the leader.")], "sweep_cap")
HDR = lambda t: [t, "Dev AUC [95% CI]", "Tie"]  # noqa: E731
tv = [["CLIP-L OpenAI", "0.7464 [0.6721, 0.8306]", "yes"], ["SigLIP base", "0.7396 [0.7031, 0.7832]", "yes"], ["CLIP-L LAION", "0.7331 [0.6764, 0.7966]", "yes"],
      ["DINOv2 small", "0.6933 [0.6639, 0.7314]", "yes"], ["DINOv2 base", "0.6745 [0.6323, 0.7239]", "yes"], ["EfficientNet-B4", "0.6471 [0.5967, 0.6987]", "no"],
      ["Xception", "0.6162 [0.5744, 0.6633]", "no"]]
ti = [["CLIP-L OpenAI", "0.9935 [0.9803, 0.9982]", "yes"], ["CLIP-L LAION", "0.9563 [0.7760, 0.9975]", "no"], ["DINOv2 base", "0.9023 [0.8496, 0.9316]", "no"],
      ["SigLIP base", "0.8657 [0.7591, 0.9166]", "no"], ["DINOv2 small", "0.8340 [0.7401, 0.8823]", "no"], ["EfficientNet-B4", "0.6688 [0.6238, 0.7137]", "no"],
      ["Xception", "0.6380 [0.4912, 0.7115]", "no"]]
ta = [["Whisper small", "0.6380 [0.5338, 0.7700]", "yes"], ["WavLM base plus", "0.6334 [0.5291, 0.7714]", "yes"], ["HuBERT base", "0.6100 [0.5434, 0.7256]", "yes"],
      ["wav2vec 2.0 base", "0.4975 [0.3451, 0.7092]", "no"]]
w4 = span_w(4)
cws = [int(w4 * 0.36), int(w4 * 0.52)]
cws.append(w4 - sum(cws))
ty0 = TOP + 330000
szs = fit_size(cws, HDR("Face frames"), tv, bold_first_col=False)
for k_, (nm, rows_) in enumerate([("Face frames", tv), ("Images", ti), ("Speech", ta)]):
    add_table(s, col_x(4 * k_), ty0, cws, HDR(nm), rows_, size=szs, max_h=2600000)
add_text(s, L, ty0 + 2350000, W, BOT - (ty0 + 2350000), [
    ("bs", "Cheapest in the tie set wins: face frames DINOv2 small (tie set of five), images CLIP ViT-L/14 OpenAI (tie set of one), speech HuBERT base (tie set of three).", 16),
    ("bs", "Face frames: EfficientNet-B4 is outside the tie set by the bootstrap, but its permutation p is 0.110; the winner is the same either way.", 16),
    ("bs", "Images: the development slice has 2 generator units, so the interval is not interpretable as a population interval.", 16),
    ("bs", "Speech: verdict inconclusive; the from-scratch reference scores 0.246, below chance; development ECE rose after temperature scaling.", 16),
], "sweep_text")

# =====================================================================================================================
# SLIDE 9b: I-0 rule and results
# =====================================================================================================================
s = new_slide("I-0: the working-version rule",
    "I-0 is the working version and not an improvement: a frozen encoder, a linear head and temperature scaling, chosen by a rule fixed before any feature existed (D-032, D-036, D-040). "
    "The rule ranks the candidates of each task by their mean development-slice AUC, builds the tie set of every candidate whose paired 95 percent group-bootstrap interval against the leader includes zero, and lets the cheapest candidate of the tie set win; the from-scratch network is reported but excluded. "
    "The table is experiment E-004: the heads of the three winners (face frames DINOv2 small, images CLIP ViT-L/14 OpenAI, speech HuBERT base), mean of three head seeds with the 95 percent group-bootstrap interval, exploratory, development slice; the development values reproduce E-001 to E-003. "
    "Train, validation and development AUC are 0.837, 0.799 and 0.693 for face frames, 1.000, 1.000 and 0.994 for images, and 1.000, 0.994 and 0.610 for speech; the shipped head is seed 0 (D-050). "
    "The working version is finished when the chain runs from a file through the web page to a calibrated score with metrics and intervals (D-040); the page check and the parity check (maximum difference 0.0019) passed, details in the backup. "
    "Evidence: docs/EXPERIMENTS.md E-004, docs/DECISIONS.md D-036, D-040, D-049, D-050.")
add_text(s, col_x(0), TOP, span_w(7), 2680000, [
    ("h", "Rule for choosing the encoder"),
    ("n", "1.  Extract features once per candidate with the frozen encoder.", 16),
    ("n", "2.  Fit a linear head, three seeds.", 16),
    ("n", "3.  Rank by mean development-slice AUC, paired 95 percent group bootstrap.", 16),
    ("n", "4.  Tie set: every candidate whose interval against the leader includes zero.", 16),
    ("n", "5.  Cheapest wins: lowest extraction time per 1,000 inputs, then fewer parameters.", 16),
], "rule")
add_card(s, col_x(7), TOP, span_w(5), 2680000, [
    ("h", "The working version is finished when"),
    ("bs", "a file goes in through the local web page;", 16),
    ("bs", "the winning encoders score it, calibrated, threshold fixed on validation data;", 16),
    ("bs", "metrics with intervals exist on validation and development data;", 16),
    ("bs", "the page shows a real result, with its warnings.", 16),
], name="card_done")
cwr = [span_w(2), span_w(3), span_w(2) + span_w(1) // 2]
rest = W - sum(cwr)
cwr = [cwr[0], cwr[1], cwr[2], (rest) // 3 * 1, 0]
cwr = [span_w(2), span_w(3)]
cwr += [(W - sum(cwr)) // 3] * 3
cwr[-1] = W - sum(cwr[:-1])
rows_r = [["Face frames", "DINOv2 small", "0.8367 [0.8267, 0.8475]", "0.7985 [0.7731, 0.8269]", "0.6933 [0.6639, 0.7314]"],
          ["Images", "CLIP ViT-L/14 OpenAI", "1.0000 [1.0000, 1.0000]", "1.0000 [1.0000, 1.0000]", "0.9935 [0.9803, 0.9982]"],
          ["Speech", "HuBERT base", "0.9997 [0.9996, 0.9999]", "0.9942 [0.9910, 0.9967]", "0.6100 [0.5434, 0.7256]"]]
hdr_r = ["Task", "Encoder (winner)", "Train AUC", "Validation AUC", "Dev AUC"]
szr = fit_size(cwr, hdr_r, rows_r, sizes=(14, 13, 12), bold_first_col=True)
gt, th = add_table(s, L, TOP + 2800000, cwr, hdr_r, rows_r, size=szr, bold_first_col=True, max_h=1700000)
add_text(s, L, TOP + 2800000 + th + 50000, W, BOT - (TOP + 2800000 + th + 50000), [
    ("cap", "Exploratory, development slice. Mean of three head seeds with the 95 percent group-bootstrap interval. The from-scratch reference is excluded from the choice. The development interval of the images rests on 2 generator units; speech is inconclusive.")], "rcap")

# =====================================================================================================================
# SLIDE 10: ladder
# =====================================================================================================================
s = new_slide("Iteration ladder",
    "After the working version come seven experimental iterations, I-1 to I-7, and a last slot for integration and freeze (D-041, ITERATIONS.md). "
    "The lever of each slot was chosen after the diagnostic battery of the preceding step and written, with its hypothesis, success line and failure line, before the run (D-051, D-054 to D-058). "
    "I-1 was full fine-tuning of the face encoder (E-005): development AUC 0.7396 to 0.9096 for the same SigLIP encoder, supported. I-2 to I-4 are speech levers (E-007, E-008, E-009): trimming and normalising the clip, augmenting the training clips, and averaging three encoders; the speech development AUC goes from 0.6100 to 0.6723, 0.7455 and 0.8130, each step supported by its mechanical criteria but each on only 12 speakers. "
    "I-5 and I-6 are on LAV-DF (E-010 to E-012): the frozen face head is below chance there, so the fusion was not tested; heads trained on LAV-DF give a fused AUC of 0.9935 that a control shows is driven by a production-pipeline cue of the visual head. I-7 is the rPPG check, which failed (E-006). I-8 runs no experiments; it freezes the code and evaluates every frozen iteration once on the sealed tests, which has not been run. "
    "The figure shows the development-slice results with their 95 percent group-bootstrap intervals; all of them are exploratory (D-041). "
    "Evidence: docs/EXPERIMENTS.md E-004 to E-013, docs/DOC_FACTS.md section 15.")
labels = [("I-0", "working version", F_GREEN), ("I-1", "fine-tuning", F_GREEN), ("I-2", "trim, normalise", F_GREEN),
          ("I-3", "augmentation", F_GREEN), ("I-4", "three encoders", F_GREEN), ("I-5", "LAV-DF check", F_ORANGE),
          ("I-6", "LAV-DF heads", F_ORANGE), ("I-7", "rPPG fails", F_RED), ("I-8", "freeze", F_GREY)]
gap = 90000
lw = (W - 8 * gap) // 9
for i, (lab, txt, f) in enumerate(labels):
    x = L + i * (lw + gap)
    dbox(s, x, TOP, lw, 800000, "**" + lab + "**" + chr(10) + txt, fill=f, size=C.SZ_TABLE)
    if i:
        darrow(s, x - gap, TOP + 400000, x, TOP + 400000)
_f, _fw, _fh = fig(s, "fig08_ladder_results.png", L, TOP + 900000, w=span_w(8), alt="Development-slice AUC with intervals: face video from the working version to fine-tuning and the Xception baseline, speech from the working version to the three-encoder average")
add_card(s, col_x(8), TOP + 900000, span_w(4), 3000000, [
    ("h", "Reading it"),
    ("bs", "Green: supported. Orange: limited. Red: failed.", 16),
    ("bs", "Fine-tuning: +0.1700 over the probe.", 16),
    ("bs", "Speech: 0.6100 to 0.8130.", 16),
    ("bs", "Sealed tests: [pending].", 16),
], name="card_ladder")
add_text(s, L, TOP + 900000 + _fh + 40000, W, BOT - (TOP + 900000 + _fh + 40000), [
    ("cap", "Exploratory, development slice, mean of three seeds, 95 percent group-bootstrap interval. The sealed test sets have not been read.")], "ladder_note")

# =====================================================================================================================
# SLIDE 10b: I-0 diagnosis
# =====================================================================================================================
s = new_slide("I-0 diagnosis",
    "The diagnostic battery was run on the working version (experiment E-004, data/_results/I-0/battery.md); every number is exploratory and from the development slice or validation data, and no lever is chosen on this slide. "
    "Face frames: the frozen DINOv2 small feature separates even the training data only moderately (train AUC 0.837, train minus validation +0.038), the gap from validation to the development slice is +0.105, and temperature scaling fitted on validation does not transfer (ECE 0.205 to 0.222); aggregation is not indicated because the clip mean is above frame level by about 0.03 and the maximum and top-3 mean are not better. "
    "Images: train and validation AUC are 1.000, the development AUC is 0.994 on two generator units (0.999 and 0.983), calibration improves (ECE 0.046 to 0.005), and the errors are border artefacts in some real images and photorealistic StyleGAN faces; with two units a shift to other generators cannot be measured. "
    "Speech: the validation to development gap is +0.384 at an in-domain AUC of 0.994, real In-the-Wild speech scores as fake, the trivial recording features reverse sign between the corpora (M-004), calibration worsens (ECE 0.329 to 0.405) and errors concentrate in short repeat-padded clips with saturated scores. "
    "Evidence: docs/EXPERIMENTS.md E-004 (diagnostic battery items 1 to 8) and M-004; the choice of the first lever is a later decision.")
dgw = [span_w(2), span_w(5)]
dgw.append(W - sum(dgw))
dg_rows = [
    ["Face frames", "Train 0.837, validation 0.799, development 0.693 (gap +0.105). Weakest manipulations on validation: NeuralTextures 0.697, FaceSwap 0.778. Development ECE 0.205 before and 0.222 after temperature scaling.",
     "Bias first: the frozen feature separates even the training data only moderately. Then shift, and calibration under shift. Aggregation is not indicated."],
    ["Images", "Train and validation 1.000, development 0.994 on 2 generator units (0.999 and 0.983). ECE 0.046 before and 0.005 after. Errors: border artefacts in real images, photorealistic StyleGAN faces.",
     "No bias or variance visible. Shift to other generators cannot be measured with 2 units. Calibration is good."],
    ["Speech", "Validation 0.994, development 0.610 (gap +0.384). Real In-the-Wild speech scores as fake. ECE 0.329 before and 0.405 after. Errors are short repeat-padded clips with saturated scores.",
     "Shift is dominant. Missing signal or a shortcut (window or padding effect) is a candidate. Calibration worsens under shift."],
]
dgs = 16
gt, th = add_table(s, L, TOP, dgw, ["Task", "Observations (battery)", "Candidate diagnoses"], dg_rows, size=dgs, bold_first_col=True, max_h=3900000, fill=3900000)
add_text(s, L, TOP + th + 100000, W, BOT - (TOP + th + 100000), [
    ("cap", "Exploratory, development slice. Diagnoses only: no lever is chosen here. The sealed tests are not touched.")], "diag_note")

# =====================================================================================================================
# I-1 and the baseline
# =====================================================================================================================
s = new_slide("I-1: fine-tuning the face encoder",
    "The diagnostic battery of the working version showed bias for the face task: the training AUC of the frozen feature with a linear head was 0.837, so the lever was fine-tuning depth, with the budget fixed beforehand (D-042, D-051). "
    "Nine runs of stage 1 (three encoders, three learning rates, one seed) showed that only the lowest rate, 1e-5, works for every encoder; five of the six runs at the higher rates lost most of the signal. Selection on validation data alone gave SigLIP base as the cheapest member of the tie set. "
    "Three seeds of SigLIP base reach a validation AUC of 0.9899 [0.9809, 0.9967] and a development AUC of 0.9096 [0.8776, 0.9388], against 0.8767 and 0.7396 for the linear probe of the same encoder; the gains are +0.1132 and +0.1700, larger than the predicted +0.02 to +0.05 and at most +0.03, and the three criteria are met (E-005, supported, exploratory). "
    "The baseline is the Xception family re-run with the same recipe and budget (E-013, D-060): development AUC 0.8750 [0.8435, 0.9089]; SigLIP minus Xception is +0.0346 [+0.0136, +0.0550] on the development slice and +0.0027 [-0.0009, +0.0060] on validation. Xception also gains +0.1817 over the working version, so most of the gain comes from fine-tuning and not from the encoder. "
    "Limits: 15 identity groups, one kind of fake on the slice, the baseline learning rate 1e-4 is the highest of the grid, one run was repeated after a data-loader error. The margin of the non-inferiority comparison is 0.0143, twice the larger seed standard deviation of the baseline (D-061). "
    "Evidence: docs/EXPERIMENTS.md E-005 and E-013, docs/DECISIONS.md D-060 and D-061.")
rows_i1 = [["Training subset", "0.9245 [0.9093, 0.9388]", "0.9994 [0.9986, 1.0000]", "not recorded"],
           ["Validation", "0.8767 [0.8582, 0.8966]", "0.9899 [0.9809, 0.9967]", "0.9872 [0.9784, 0.9942]"],
           ["Development slice", "0.7396 [0.7031, 0.7832]", "0.9096 [0.8776, 0.9388]", "0.8750 [0.8435, 0.9089]"]]
hdr_i1 = ["AUC", "Probe, SigLIP base", "Fine-tuned, SigLIP base", "Xception baseline"]
cw_i1 = [span_w(3), span_w(3), span_w(3)]
cw_i1.append(W - sum(cw_i1))
sz_i1 = fit_size(cw_i1, hdr_i1, rows_i1, sizes=(15, 14, 13, 12), bold_first_col=True)
gt, th = add_table(s, L, TOP, cw_i1, hdr_i1, rows_i1, size=sz_i1, bold_first_col=True, max_h=1700000)
add_card(s, col_x(0), TOP + th + 150000, span_w(6), BOT - (TOP + th + 150000), [
    ("h", "What was done"),
    ("bs", "Full fine-tuning, 1,800 steps, learning rate 1e-5, batch 32; three seeds; same recipe for the Xception baseline.", 18),
    ("bs", "Higher learning rates collapsed in five of six runs.", 18),
    ("bs", "Criteria met: validation gain, training AUC at least 0.90, development gain.", 18),
], name="card_i1_done")
add_card(s, col_x(6), TOP + th + 150000, span_w(6), BOT - (TOP + th + 150000), [
    ("h", "What it supports"),
    ("bs", "Gain over the probe: +0.1132 on validation, +0.1700 on the slice.", 18),
    ("bs", "SigLIP minus Xception on the slice: +0.0346 [+0.0136, +0.0550].", 18),
    ("bs", "Exploratory; 15 identities, one kind of fake. Sealed tests: [pending].", 18),
], name="card_i1_says")

# =====================================================================================================================
# I-2 to I-4
# =====================================================================================================================
s = new_slide("I-2 to I-4: three levers for speech",
    "The speech task kept a large shift gap after the working version: validation AUC 0.9942, development AUC 0.6100 on 12 speakers, with recording-condition cues that reverse in sign between ASVspoof 2019 and In-the-Wild (M-004). "
    "I-2 (E-007, D-054): trim leading and trailing silence and normalise the peak before the 4 second window; a manipulation check written beforehand passed; the development AUC rose to 0.6723 [0.6126, 0.7584], +0.0623 [+0.0114, +0.1048], validation fell by 0.0054; supported. "
    "I-3 (E-008, D-055): train the head on the clean clips plus three augmented copies (noise, low-pass, reverberation, MP3 round trip, ranges fixed beforehand); the previous head lost 0.33 AUC on an augmented validation copy and the new head 0.04; development AUC 0.7455 [0.7052, 0.8121], +0.0732 [+0.0122, +0.1225] over the previous head; supported. "
    "I-4 (E-009, D-056): average the logits of HuBERT base, WavLM base plus and the Whisper small encoder; mean pairwise correlation 0.534 on the slice; development AUC 0.8130 [0.7523, 0.8808], +0.0675 [+0.0300, +0.0929] over the HuBERT head, validation 0.9964; supported, at the price of three encoder passes. "
    "Limits: a slice of 12 speakers, so a gain below about +0.09 cannot be shown by one step; the calibration error on the slice stays high after temperature scaling (0.1670 for the average), so the audio score is a ranking score. All verdicts are exploratory. "
    "Evidence: docs/EXPERIMENTS.md E-007, E-008, E-009; docs/DECISIONS.md D-054 to D-056.")
rows_sp = [["I-0: frozen HuBERT head", "0.9942 [0.9910, 0.9967]", "0.6100 [0.5434, 0.7256]", "baseline"],
           ["I-2: trim and normalise", "0.9889 [0.9831, 0.9933]", "0.6723 [0.6126, 0.7584]", "+0.0623 [+0.0114, +0.1048]"],
           ["I-3: plus augmentation", "0.9869 [0.9818, 0.9911]", "0.7455 [0.7052, 0.8121]", "+0.0732 [+0.0122, +0.1225]"],
           ["I-4: three encoders", "0.9964 [0.9943, 0.9982]", "0.8130 [0.7523, 0.8808]", "+0.0675 [+0.0300, +0.0929]"]]
hdr_sp = ["Step", "Validation AUC", "Development slice AUC", "Change on the slice"]
cw_sp = [span_w(3), span_w(3), span_w(3)]
cw_sp.append(W - sum(cw_sp))
sz_sp = fit_size(cw_sp, hdr_sp, rows_sp, sizes=(15, 14, 13, 12), bold_first_col=True)
gt, th = add_table(s, L, TOP, cw_sp, hdr_sp, rows_sp, size=sz_sp, bold_first_col=True, max_h=2300000)
add_card(s, col_x(0), TOP + th + 150000, span_w(6), BOT - (TOP + th + 150000), [
    ("h", "Why each lever"),
    ("bs", "I-2: remove level and silence cues that reverse between corpora.", 18),
    ("bs", "I-3: a head fitted to noisy, filtered, compressed copies cannot rely on those cues.", 18),
    ("bs", "I-4: averaging weakly correlated scores lowers variance.", 18),
], name="card_sp_why")
add_card(s, col_x(6), TOP + th + 150000, span_w(6), BOT - (TOP + th + 150000), [
    ("h", "Limits"),
    ("bs", "12 speakers: wide intervals; the slice is exploratory.", 18),
    ("bs", "Calibration stays poor on the slice: a ranking score.", 18),
    ("bs", "Three encoder passes per clip. Sealed tests: [pending].", 18),
], name="card_sp_limits")

# =====================================================================================================================
# I-5 and I-6: LAV-DF
# =====================================================================================================================
s = new_slide("I-5 and I-6: LAV-DF",
    "LAV-DF is the only data set here in which video and audio are manipulated independently. A fixed subset of 500 videos per cell (real, video fake only, audio fake only, both fake) and split was used, with the source video as the group (D-057). "
    "I-5 (E-010): the pre-check written in advance found the frozen face head below chance on LAV-DF, AUC 0.3838 [0.3599, 0.4079] against the video label; the frozen speech average gave 0.6476 [0.6244, 0.6701] against the audio label; the fusion was not tested. "
    "I-6 (E-011, D-058): heads trained on LAV-DF pass the pre-check (0.7491 and 0.8604) and the logistic-regression fusion reaches 0.9935 [0.9905, 0.9962] on the fused label, +0.0023 [+0.0009, +0.0040] over the better head; mechanically supported. "
    "The post-hoc control (labelled as such, D-059) keeps only the produced videos: the visual head is then at 0.5070 [0.4771, 0.5361] against the video label, while it separates the real originals from audio-only fakes (frames untouched) with 0.9912; the speech head reaches 0.8574 [0.8394, 0.8752] against the audio label within produced videos. The visual head has therefore learned original versus produced, and the fused AUC is an in-data-set result driven by that cue. "
    "A follow-up trained the visual head on produced videos only (E-012): 0.5169 [0.4945, 0.5392], not supported; no visual manipulation signal was recovered by this recipe. No audio-visual detection claim is made; the page shows the two scores side by side with a warning and no fused verdict. "
    "Evidence: docs/EXPERIMENTS.md E-010 to E-012; docs/DECISIONS.md D-057 to D-059.")
_f, _fw, _fh = fig(s, "fig09_lavdf_cells.png", L, TOP, w=span_w(8), alt="Bars of AUC for the visual and the audio head on LAV-DF: against real originals for three fake cells, and within produced videos")
add_card(s, col_x(8), TOP, span_w(4), BOT - TOP, [
    ("h", "The control"),
    ("bs", "Frozen face head: 0.3838, below chance.", 16),
    ("bs", "Visual head: 0.99 against real originals for every fake cell.", 16),
    ("bs", "Within produced videos: 0.5070, chance.", 16),
    ("bs", "Produced-only head: 0.5169.", 16),
    ("bs", "Speech head: 0.8574.", 16),
    ("bs", "No audio-visual claim.", 16),
], name="card_lavdf")
add_text(s, L, TOP + _fh + 40000, span_w(8), BOT - (TOP + _fh + 40000), [
    ("cap", "Exploratory, LAV-DF validation subset of 2,000 videos, 95 percent group-bootstrap interval.")], "lavdf_note")

# =====================================================================================================================
# SLIDE 11: rPPG
# =====================================================================================================================
s = new_slide("rPPG: test the signal first",
    "Remote photoplethysmography is a candidate improvement, not an adopted component (D-006, O-010). "
    "The literature is mixed: early work reports high in-dataset accuracy, a 2025 study found a plausible pulse in high-quality face swaps, a 2026 preprint reported a reproduction that fell lower on talking-face fakes under an identity-disjoint split, and pulse signals can be edited on purpose (R-005 to R-010). "
    "So the first step is a signal check on contiguous frames with a time-shuffled control pooled across clips (at least ten shuffles per clip); it passes when at least 60 percent of real clips have a peak SNR above the 95th percentile of the control. "
    "If it fails, the slot reports exactly that and no separation or fusion experiment is run; if it passes, the adopt decision is taken on the development slice only and needs the whole paired interval above zero. "
    "Evidence: docs/ITERATIONS.md section I-7, thresholds written before any run; the signal check was run and failed (E-006): 130 of 432 real clips, 0.301 [0.260, 0.346], above the 95th percentile of the shuffled control (1.118 dB), against the required 0.6; per data set 0.280 for FaceForensics++, 0.288 for Celeb-DF v2 and 0.333 for LAV-DF; median SNR of the real clips -0.222 dB. The slot reports that the pulse signal is not recoverable from these data, and the separation and fusion tests were not run (D-052). The check has no ground-truth pulse, so it does not show that no pulse exists.")
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
    ("b", "Signal check passes when at least 60 percent of real clips have a peak SNR above the 95th percentile of the shuffled control; if not, the slot reports that the pulse is not recoverable. It did not pass.", 18),
    ("b", "Separation, only if it passes: lower interval bound of the AUC above 0.55 on an identity-disjoint split.", 18),
    ("b", "Adopt only if the whole paired interval (with minus without) is above zero on the development slice.", 18),
], "rppg_text")
add_card(s, col_x(7), ty, span_w(5), BOT - ty, [
    ("h", "Outcome"),
    ("p", "Signal check: fails, 130 of 432 real clips (0.301 [0.260, 0.346]) against 0.6 required", 18),
    ("p", "Separation and added value: not run", 18),
    ("p", "Result: pulse not recoverable from these data", 18),
], name="card_rppg")

# =====================================================================================================================
# SLIDE 12: web demo
# =====================================================================================================================
s = new_slide("Web demo (real checkpoint)",
    "The software is a local web page served by Flask on 127.0.0.1 only, with a 512 MB upload limit; uploaded files are deleted after every request (D-043). "
    "It calls an inference library with a fixed response structure: a score per modality, a status, warnings, errors, a model block and a fixed sentence that says what the score means (D-027). "
    "The three screenshots were taken with the real adapter and the I-0 checkpoint, so there is no MOCK banner: an image, a video with its audio and an audio file, each a training item, so they show that the path works and say nothing about accuracy (E-004, assets/screens/real_image.png, real_video.png, real_audio.png). "
    "For the video the visual and the audio score are shown separately and the verdict is undetermined, because no combining rule is claimed; the audio score carries the warning that its calibration is exploratory (D-050). "
    "Earlier screenshots against the development stand-in remain in the report as a labelled figure; the real adapter refuses to run without a checkpoint whose metadata and preprocessing identifier match (D-044).")
SH = 3050000
spec = [("real_image.png", "(a) image", 0.16, 0.16, 0.08, 0.07), ("real_video.png", "(b) video with audio", 0.16, 0.16, 0.08, 0.07),
        ("real_audio.png", "(c) audio file", 0.16, 0.16, 0.08, 0.07)]
from PIL import Image as _Im
pw = []
for fn, cap, cl, cr_, ct, cb in spec:
    iw, ih = _Im.open(os.path.join(SCR, fn)).size
    pw.append(int(SH * (iw * (1 - cl - cr_)) / (ih * (1 - ct - cb))))
gp = 70000
xs = [L]
for w_ in pw[:-1]:
    xs.append(xs[-1] + w_ + gp)
for (fn, cap, cl, cr_, ct, cb), xx, w_ in zip(spec, xs, pw):
    pic = add_pic(s, os.path.join(SCR, fn), xx, TOP, alt="Web page with the real checkpoint, no MOCK banner: " + cap)
    pic.crop_left, pic.crop_right, pic.crop_top, pic.crop_bottom = cl, cr_, ct, cb
    pic.left, pic.top, pic.width, pic.height = Emu(xx), Emu(TOP), Emu(w_), Emu(SH)
    add_text(s, xx, TOP + SH + 40000, w_, 280000, [("cap", cap)], "pic_caption")
right_x = col_x(8)
assert xs[-1] + pw[-1] < right_x - 100000, (xs[-1] + pw[-1], right_x)
add_card(s, right_x, TOP, span_w(4), 4800000, [
    ("h", "Real checkpoint"),
    ("bs", "Real adapter with the I-0 checkpoint: no MOCK banner.", 16),
    ("bs", "The three files are training items: the path works, accuracy is not claimed.", 16),
    ("bs", "A video shows a visual and an audio score and the verdict Undetermined, as no combining rule is claimed.", 16),
    ("bs", "The audio score carries an exploratory-calibration warning.", 16),
], name="card_web")
ty = TOP + SH + 400000
add_text(s, L, ty, xs[-1] + pw[-1] - L, BOT - ty, [
    ("bs", "Local Flask page on 127.0.0.1; uploads up to 512 MB; files deleted after each request.", 16),
    ("bs", "Sealed-test accuracy of the page: [pending].", 16),
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
    ("s", "An iteration not finished by Wednesday is reported as not run. Status on 25 October: I-1 to I-7 have a logged run; the freeze and the sealed tests are on Thursday 29 October.", 16)], "timeline_note")

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
    ("b", "Every verdict so far is exploratory (development slice or validation); no sealed test set has been read.", 20),
    ("b", "On LAV-DF the fused result is driven by a production-pipeline cue of the visual head; no audio-visual detection is claimed.", 20),
    ("b", "The baseline learning rate is at the edge of its grid; the margin 0.0143 rests on three seeds.", 20),
    ("b", "Two encoder weights (OpenAI CLIP, WavLM) are conditionally licensed and accepted as research-only.", 20),
    ("b", "Cross-dataset numbers from papers use other protocols and are context only; the pulse could not be recovered, and the check has no ground truth.", 20),
], "limits")

# =====================================================================================================================
# SLIDE 15: conclusion
# =====================================================================================================================
s = new_slide("Conclusion",
    "This slide states what the exploratory evidence supports and keeps the sealed-test cells as [pending] until the freeze on Thursday 29 October. "
    "What already stands without any result is the design: a group-disjoint protocol with sealed test sets, a rule that prefers the cheaper candidate when evidence ties, a staircase of one sealed evaluation per frozen iteration, and a working-version definition that requires the whole chain to run. "
    "The two confirmatory claims are declared in advance: the final system against the working version, and the final system against a re-run published baseline, the latter judged by non-inferiority with delta fixed beforehand (D-025, D-041). "
    "Every other comparison is labelled exploratory. Evidence: docs/PROTOCOL.md addendum, docs/ITERATIONS.md; the cells below are [pending] until the ledger entries of the freeze exist.")
add_text(s, col_x(0), TOP, span_w(6), 4800000, [
    ("h", "What stands without any result"),
    ("b", "A group-disjoint protocol with sealed test sets and a ledger.", 18),
    ("b", "A rule that prefers the cheaper candidate when the evidence ties.", 18),
    ("b", "One sealed evaluation per frozen iteration (the staircase).", 18),
    ("b", "Two confirmatory claims, declared before any final run.", 18),
    ("h", "What the exploratory runs show"),
    ("b", "Face development AUC 0.7396 to 0.9096 by fine-tuning.", 18),
    ("b", "Speech 0.6100 to 0.8130 on 12 speakers.", 18),
    ("b", "LAV-DF fusion driven by a pipeline cue.", 18),
], "concl_left")
gt, th = add_table(s, col_x(6), TOP, [span_w(4), span_w(2)], ["Claim", "Outcome"], [
    ["Working version, development AUC (exploratory)", "0.693; 0.994; 0.610"],
    ["Face video, fine-tuned SigLIP minus Xception, development slice (exploratory)", "+0.0346 [+0.0136, +0.0550]"],
    ["Final system versus working version (paired, sealed tests)", "[pending]"],
    ["Final system versus re-run baseline (non-inferiority)", "[pending]"],
    ["rPPG: adopt or not", "not adopted: signal not recoverable"],
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
    ("b", "**Staircase and confirmatory claims:** one sealed evaluation per frozen iteration; the margin 0.0143"),
    ("b", "**Preprocessing:** frames, face crops, audio window"),
    ("b", "**Sweep tables and cost record:** full tables per task, cost and fine-tuning fit"),
    ("b", "**I-0 diagnostic battery:** metrics, subsets, score distributions, worst errors"),
    ("b", "**Parity check, page check and audio diagnosis**"),
    ("b", "**Fine-tuning grid and checkpoint v2;** **LAV-DF control and software status**"),
    ("b", "**Data licences and terms;** **Celeb-DF++ test balance;** **Environment (measured)**"),
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
    "The baseline was re-run (Xception, three seeds, E-013) and the margin is delta = 2 times the larger seed standard deviation (0.0010 on validation, 0.0072 on the development slice), with a floor of 0.01, that is 0.0143 from the unrounded values (D-060, D-061). The final system is the face encoder fine-tuned in I-1, the unchanged image head and the three-encoder speech average; the LAV-DF fusion is not part of it. "
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
    ("s", "Development slice: exploratory verdicts. Sealed test sets: one evaluation per frozen iteration, in a ledger; not yet run (freeze on 29 October). Two pre-declared claims; non-inferiority margin 0.0143 from the baseline seed spread.")], "stair_note")

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
s = new_slide("Sweep tables, face frames",
    "Full table of experiment E-001: frozen image encoders on face video frames, trained on FaceForensics++ c23, validation FaceForensics++ validation, development slice Celeb-DF v2 (1,728 clips, 15 identity groups), exploratory (D-041, D-049). "
    "AUC values are the mean of three head seeds; the interval is the 95 percent group bootstrap with 2,000 resamples; the gap is validation minus development AUC; the tie column is the paired interval against the leader (CLIP ViT-L/14 OpenAI). "
    "Cost is one end-to-end extraction run per candidate and the peak memory of that extraction (data/_results/T-V/model_comparison.md); the cost rule picks DINOv2 small, 0.943 s per 1,000 inputs against 1.732 s for EfficientNet-B4. "
    "The from-scratch reference is reported and excluded; FLOPs and model-only time are in data/_results/cost_record.json. "
    "Evidence: docs/EXPERIMENTS.md E-001, data/_results/T-V/rank.md and model_comparison.md.")
hv = ["Encoder", "Val AUC", "Dev AUC [95% CI]", "Gap", "Tie", "s per 1,000", "Peak GiB"]
rv = [["CLIP ViT-L/14 OpenAI", "0.9046", "0.7464 [0.6721, 0.8306]", "+0.158", "yes", "4.5", "2.85"],
      ["SigLIP base", "0.8767", "0.7396 [0.7031, 0.7832]", "+0.137", "yes", "1.4", "1.30"],
      ["CLIP ViT-L/14 LAION", "0.8635", "0.7331 [0.6764, 0.7966]", "+0.130", "yes", "4.0", "2.72"],
      ["DINOv2 small", "0.7985", "0.6933 [0.6639, 0.7314]", "+0.105", "yes", "0.9", "0.38"],
      ["DINOv2 base", "0.8170", "0.6745 [0.6323, 0.7239]", "+0.143", "yes", "1.7", "0.95"],
      ["EfficientNet-B4", "0.7478", "0.6471 [0.5967, 0.6987]", "+0.101", "no", "1.7", "1.31"],
      ["Xception", "0.7184", "0.6162 [0.5744, 0.6633]", "+0.102", "no", "3.3", "1.96"],
      ["From scratch (excluded)", "n/a", "0.5166 [0.5033, 0.5324]", "n/a", "n/a", "n/a", "n/a"]]
cwv = [span_w(3), span_w(1), span_w(3), span_w(1), span_w(1) // 1, span_w(1), 0]
cwv = [span_w(3), span_w(1), span_w(3), span_w(1)]
cwv += [(W - sum(cwv)) // 3] * 3
cwv[-1] = W - sum(cwv[:-1])
gt, th = add_table(s, L, TOP, cwv, hv, rv, size=fit_size(cwv, hv, rv, sizes=(16, 15, 14, 13, 12), bold_first_col=True), bold_first_col=True, max_h=3600000, fill=3600000)
add_text(s, L, TOP + th + 100000, W, BOT - (TOP + th + 100000), [
    ("cap", "Exploratory, development slice. Gap = validation minus development AUC. Time and peak memory are one end-to-end extraction run; FLOPs and model-only time are on the cost slide.")], "sweep_v_note")

s = new_slide("Sweep tables, images and speech",
    "Full tables of experiments E-002 (generated images) and E-003 (speech), exploratory, development slice (D-041, D-049). "
    "Images: trained on 14 ProGAN classes, validation on the other 6, development slice two CNN-synthesis generator units (5,998 images, 2 groups), so the intervals are not population intervals; per generator, CLIP ViT-L/14 OpenAI scores 0.9988 on StarGAN and 0.9830 on whichfaceisreal, the LAION model 0.9978 and 0.7790 (M-004 c). "
    "Speech: trained on ASVspoof 2019 LA train, validation LA dev, development slice In-the-Wild (6,991 clips, 12 speakers); validation AUC is 0.99 to 1.00 for every encoder but the development AUC is 0.50 to 0.64, the verdict is inconclusive, the from-scratch reference is 0.2458 [0.0706, 0.4404] and the label mapping was checked (M-004 a). "
    "Time is one end-to-end extraction run per 1,000 inputs and the peak memory of the extraction. "
    "Evidence: docs/EXPERIMENTS.md E-002, E-003, M-004; data/_results/T-I and T-A (rank.md, model_comparison.md).")
ri = [["CLIP ViT-L/14 OpenAI", "1.0000", "0.9935 [0.9803, 0.9982]", "+0.007", "yes", "4.9", "2.85"],
      ["CLIP ViT-L/14 LAION", "1.0000", "0.9563 [0.7760, 0.9975]", "+0.044", "no", "4.5", "2.72"],
      ["DINOv2 base", "0.9899", "0.9023 [0.8496, 0.9316]", "+0.088", "no", "2.0", "0.95"],
      ["SigLIP base", "0.9975", "0.8657 [0.7591, 0.9166]", "+0.132", "no", "1.8", "1.30"],
      ["DINOv2 small", "0.9685", "0.8340 [0.7401, 0.8823]", "+0.135", "no", "1.2", "0.38"],
      ["EfficientNet-B4", "0.8990", "0.6688 [0.6238, 0.7137]", "+0.230", "no", "2.5", "1.31"],
      ["Xception", "0.8140", "0.6380 [0.4912, 0.7115]", "+0.176", "no", "2.2", "1.96"],
      ["From scratch (excluded)", "n/a", "0.5500 [0.5080, 0.5735]", "n/a", "n/a", "n/a", "n/a"]]
ra = [["Whisper small encoder", "0.9995", "0.6380 [0.5338, 0.7700]", "+0.362", "yes", "11.5", "1.62"],
      ["WavLM base plus", "0.9924", "0.6334 [0.5291, 0.7714]", "+0.359", "yes", "2.5", "2.38"],
      ["HuBERT base", "0.9942", "0.6100 [0.5434, 0.7256]", "+0.384", "yes", "2.0", "2.40"],
      ["wav2vec 2.0 base", "0.9965", "0.4975 [0.3451, 0.7092]", "+0.499", "no", "2.0", "2.34"],
      ["From scratch (excluded)", "n/a", "0.2458 [0.0706, 0.4404]", "n/a", "n/a", "n/a", "n/a"]]
hi = ["Images", "Val AUC", "Dev AUC [95% CI]", "Gap", "Tie", "s per 1,000", "Peak GiB"]
ha = ["Speech", "Val AUC", "Dev AUC [95% CI]", "Gap", "Tie", "s per 1,000", "Peak GiB"]
szi = fit_size(cwv, hi, ri, sizes=(13, 12), bold_first_col=True)
gt, th1 = add_table(s, L, TOP, cwv, hi, ri, size=szi, bold_first_col=True, max_h=2900000)
gt, th2 = add_table(s, L, TOP + th1 + 110000, cwv, ha, ra, size=szi, bold_first_col=True, max_h=1900000)
assert TOP + th1 + 110000 + th2 <= BOT, (th1, th2)

s = new_slide("Cost record and fine-tuning fit",
    "Accuracy is never reported alone: every candidate has a cost record generated from the extraction records and the clean step-time run (D-036, D-042; data/_results/T-V/model_comparison.md). "
    "The fine-tuning budget of the video task is one pass over 57,600 training frames at batch 32 with a per-run time cap; an encoder enters only if its peak memory at batch 32 is at most 10 GiB and one budget takes at most 10 minutes (D-042). "
    "By the clean step-time run this admits DINOv2 small, SigLIP base, DINOv2 base, EfficientNet-B4 and Xception, and excludes the two CLIP ViT-L/14 models (12.2 and 13.8 GiB, 300 and 457 minutes per budget on the 12 GB card); the first three of the ranking that fit are the I-1 candidates, DINOv2 small, SigLIP base and DINOv2 base (D-049). "
    "FLOPs and model-only time come from the section 7 cost record of 2026-10-25 (data/_results/cost_record.json): inference in fp16 at batch 64 (speech batch 32) on the RTX 5070, FLOPs per forward pass, times per 1,000 inputs. "
    "Evidence: docs/DECISIONS.md D-042, D-049; docs/EXPERIMENTS.md M-003 (superseded by the clean run) and the model comparison tables.")
hc = ["Candidate", "Params", "GFLOPs", "Model s", "End-to-end", "Extr. GiB", "Tune GiB", "Min", "Fits", "I-1"]
rc = [["DINOv2 small", "22.1 M", "12.2", "0.39", "0.94", "0.38", "2.0", "1.4", "yes", "yes"],
      ["SigLIP base", "92.9 M", "35.4", "0.94", "1.44", "1.30", "4.0", "3.4", "yes", "yes"],
      ["DINOv2 base", "86.6 M", "46.3", "1.19", "1.67", "0.95", "4.5", "4.0", "yes", "yes"],
      ["CLIP ViT-L/14 LAION", "304.0 M", "162.0", "3.61", "4.03", "2.72", "12.2", "300.0", "no", "no"],
      ["CLIP ViT-L/14 OpenAI", "304.0 M", "162.0", "4.04", "4.45", "2.85", "13.8", "457.1", "no", "no"],
      ["EfficientNet-B4", "17.5 M", "6.1", "1.24", "1.73", "1.31", "7.3", "4.2", "yes", "no"],
      ["Xception", "20.8 M", "16.7", "0.99", "3.32", "1.96", "4.3", "3.3", "yes", "no"],
      ["HuBERT base (speech)", "94.4 M", "57.5", "1.90", "2.04", "2.40", "n/a", "n/a", "n/a", "n/a"],
      ["Whisper small enc. (speech)", "88.2 M", "344.3", "11.14", "11.53", "1.62", "n/a", "n/a", "n/a", "n/a"]]
cwc = [span_w(3) - 150000, span_w(1), span_w(1) + 50000, span_w(1) - 100000, span_w(1) + 250000, span_w(1) - 100000, span_w(1) - 100000, span_w(1) - 250000, span_w(1) - 300000]
cwc.append(W - sum(cwc))
gt, th = add_table(s, L, TOP, cwc, hc, rc, size=fit_size(cwc, hc, rc, sizes=(14, 13, 12), bold_first_col=True), bold_first_col=True, max_h=3500000, fill=3000000)
add_text(s, L, TOP + th + 100000, W, BOT - (TOP + th + 100000), [
    ("bs", "Inference in fp16 at batch 64 (speech 32) on the RTX 5070; GFLOPs per forward pass; Model s and End-to-end s are seconds per 1,000 inputs, end to end on real files (face frames).", 16),
    ("bs", "Extr. GiB: peak memory of extraction. Tune GiB and Min: fine-tuning peak at batch 32 and minutes per budget. Fits: at most 10 GiB and 10 minutes. I-1: first three of the ranking that fit.", 16),
], "cost_text")

# ---- I-0 battery (1): metrics
s = new_slide("I-0 battery: metrics",
    "Items 2, 3, 4 and 8 of the diagnostic battery for the working version (experiment E-004, data/_results/I-0/battery.md); exploratory, development slice (D-041). "
    "The train minus validation gap is small for all three tasks (0.038, 0.000, 0.006), so there is no sign of variance; the validation minus development gap separates the tasks: 0.105 for face frames, 0.007 for images and 0.384 for speech. "
    "Temperature scaling fitted on validation lowered the validation ECE for images and speech, but on the development slice it did not help face frames or speech (0.205 to 0.222 and 0.329 to 0.405) and helped images (0.046 to 0.005); all three temperatures are below 1, so validation scores are sharpened and development scores are then over-confident. "
    "The seed standard deviation on the development slice (0.0107, 0.0006, 0.0026) is a third or less of the bootstrap half-width (0.0338, 0.0089, 0.0911): the sampling of identities and speakers, not training noise, limits the intervals. "
    "Evidence: docs/EXPERIMENTS.md E-004 and the diagnostic battery definition; D-050 for the shipped seed 0 head.")
hb = ["Task", "Train minus val", "Val minus dev", "ECE val (before / after)", "ECE dev (before / after)", "Brier dev (before / after)"]
rb = [["Face frames", "+0.038", "+0.105", "0.0325 / 0.0321", "0.2048 / 0.2217", "0.2963 / 0.3050"],
      ["Images", "0.000", "+0.007", "0.0050 / 0.0012", "0.0462 / 0.0050", "0.0334 / 0.0299"],
      ["Speech", "+0.006", "+0.384", "0.0883 / 0.0078", "0.3293 / 0.4052", "0.3699 / 0.4247"]]
cwb = [span_w(2), span_w(2), span_w(2)]
cwb += [(W - sum(cwb)) // 3] * 3
cwb[-1] = W - sum(cwb[:-1])
gt, th1 = add_table(s, L, TOP, cwb, hb, rb, size=fit_size(cwb, hb, rb, sizes=(15, 14, 13, 12), bold_first_col=True), bold_first_col=True, max_h=1800000, fill=1500000)
hb2 = ["Task", "Temperature", "Threshold", "Dev EER", "Balanced acc. val / dev", "Seed sd (dev)", "Bootstrap half-width"]
rb2 = [["Face frames", "0.907", "0.4774", "0.3582", "0.750 / 0.617", "0.0107", "0.0338"],
       ["Images", "0.636", "0.2073", "0.0401", "0.998 / 0.955", "0.0006", "0.0089"],
       ["Speech", "0.486", "0.1451", "0.4183", "0.971 / 0.529", "0.0026", "0.0911"]]
cwb2 = [span_w(2), span_w(1) + 300000, span_w(1) + 300000, span_w(1) + 300000, span_w(2)]
cwb2 += [(W - sum(cwb2)) // 2] * 2
cwb2[-1] = W - sum(cwb2[:-1])
gt, th2 = add_table(s, L, TOP + th1 + 150000, cwb2, hb2, rb2, size=fit_size(cwb2, hb2, rb2, sizes=(15, 14, 13, 12), bold_first_col=True), bold_first_col=True, max_h=1800000, fill=1500000)
add_text(s, L, TOP + th1 + th2 + 250000, W, BOT - (TOP + th1 + th2 + 250000), [
    ("cap", "Exploratory, development slice. Mean of three head seeds; the threshold is stored in calibrated space and fixed on validation data. Sealed tests: [pending].")], "batt_note")

# ---- I-0 battery (2): subsets
s = new_slide("I-0 battery: subsets and errors",
    "Items 1, 5, 6 and 7 of the diagnostic battery for the working version (experiment E-004, M-004; data/_results/I-0/battery.md); exploratory, development slice, intervals are 95 percent group bootstrap where stated. "
    "Face frames: validation AUC by manipulation is 0.884 (Deepfakes), 0.834 (Face2Face), 0.778 (FaceSwap) and 0.697 (NeuralTextures); the development slice has one fake label; the clip mean (0.685) is above frame level (0.656) and the median, top-3 mean and maximum are not better; face size shows no trend and no split by compression was possible because FaceForensics++ c23 is one level and the compression of Celeb-DF is not recorded. "
    "Images: all six validation classes are 1.000; per generator CLIP ViT-L/14 OpenAI scores 0.999 on StarGAN and 0.983 on whichfaceisreal. "
    "Speech: validation attacks A01 to A05 score 0.996 to 0.997 and A06 0.984; the development AUC per speaker has minimum 0.503, median 0.692 and maximum 0.990, with the pooled value 0.610. "
    "Worst errors (10 false alarms and 10 misses per task) were classified by eye from local contact sheets, which are not published because of the dataset licences. Evidence: docs/EXPERIMENTS.md E-004 items 1 to 7.")
hs = ["Task", "AUC by method, generator or speaker", "Splits and aggregation", "Worst errors (20 per task)"]
rs = [["Face frames", "Validation: Deepfakes 0.884, Face2Face 0.834, FaceSwap 0.778, NeuralTextures 0.697. Development: one fake label, 0.693.",
       "One face 0.692, two or more 0.762 [0.667, 0.939]; face-height thirds 0.689, 0.696, 0.697. Frame 0.656, clip mean 0.685, median 0.688, top-3 0.663, max 0.662.",
       "False alarms: sharp near-frontal faces, no visible defect. Misses: 6 of 10 from one identity group, 3 from another."],
      ["Images", "Validation: six classes 1.000. Development: StarGAN 0.999, whichfaceisreal 0.983.",
       "The file extension does not separate the classes in any split.",
       "False alarms: real images with stretched border pixels. Misses: 9 of 10 are photorealistic whichfaceisreal fakes."],
      ["Speech", "Validation attacks A01 to A05 0.996 to 0.997, A06 0.984. Development per speaker: minimum 0.503, median 0.692, maximum 0.990; pooled 0.610.",
       "No speaker below 0.5.",
       "Short clips (median 1.9 s and 1.2 s), repeat-padded fraction 0.52 and 0.70, scores saturated at 0 or 1."]]
cws2 = [span_w(1) + 400000, span_w(4), span_w(3)]
cws2.append(W - sum(cws2))
gt, th = add_table(s, L, TOP, cws2, hs, rs, size=14, bold_first_col=True, max_h=4600000)
assert TOP + th <= BOT, th

# ---- parity and page check
s = new_slide("Parity check and page check",
    "The parity check (D-044, experiment E-004) scores seven training items twice: from the stored features and through the real adapter from the raw file, so that any mismatch between training-time and inference-time preprocessing would show. "
    "The maximum absolute difference of the calibrated score is 0.0019 (face frames 0.0002, images 0.0019, speech 0.0005) against a tolerance of 0.02, so the check passes; the two dependences named in advance, the JPEG round trip and the colour order, agree. "
    "The page check started the page with the real adapter and the I-0 checkpoint and uploaded one image, one video with audio and one audio file, all training items, in a real browser: each gave a result with the source real, no MOCK banner and no page error, the video showed a visual and an audio score separately with the verdict Undetermined, the audio score carried the exploratory-calibration warning, and the upload folder was empty afterwards. "
    "These checks show that the path works and say nothing about accuracy. Evidence: docs/EXPERIMENTS.md E-004, data/_results/I-0/parity.json and web_real_check.json, docs/DECISIONS.md D-044 and D-050.")
hp = ["Task", "Items scored", "Max abs. difference (calibrated score)"]
rp = [["Face frames", "2 FaceForensics++ videos (one fake, one real)", "0.0002"],
      ["Images", "2 ProGAN images", "0.0019"],
      ["Speech", "3 ASVspoof utterances", "0.0005"],
      ["All", "7 items, tolerance 0.02", "0.0019 (pass)"]]
cwp = [span_w(2), span_w(6)]
cwp.append(W - sum(cwp))
gt, th = add_table(s, L, TOP, cwp, hp, rp, size=fit_size(cwp, hp, rp, sizes=(16, 15, 14), bold_first_col=True), bold_first_col=True, max_h=2300000, fill=2000000)
add_text(s, L, TOP + th + 120000, W, BOT - (TOP + th + 120000), [
    ("b", "Page check: an image, a video with audio and an audio file (training items) gave results with the source real, no MOCK banner and no page error; the upload folder was empty afterwards.", 17),
    ("b", "The checks show that the path works and say nothing about accuracy.", 17),
], "parity_text")

# ---- audio diagnosis
s = new_slide("Audio diagnosis (measurement)",
    "Measurement M-004 (diagnosis, not an experiment) asked why the speech development AUC is so far below validation; it changes no selection (D-049) and reads only training, validation and development data. "
    "Label check: the In-the-Wild development features agree with the metadata on every label and speaker, and the ASVspoof train and dev features agree with the protocol files (0 mismatches), so no label inversion; the from-scratch reference reaches 0.99996 to 0.99997 on ASVspoof validation and 0.219 to 0.286 on the development slice, so the below-chance value is a property of the data shift. "
    "Trivial features used as the fake score show that the sign of leading silence and of RMS energy reverses between the training corpus and In-the-Wild; with 12 speakers the intervals of these two include 0.5, so the sign is supported but the size is not established, and only duration and the repeat-padded fraction exclude 0.5. "
    "Evidence: docs/EXPERIMENTS.md M-004 (a) and (b), scripts/sweep/m004.py, data/_results/M-004/m004.json.")
ha2 = ["Single feature as fake score", "ASVspoof train", "ASVspoof dev", "In-the-Wild dev"]
ra2 = [["Leading silence", "0.416 [0.350, 0.488]", "0.375 [0.299, 0.453]", "0.719 [0.425, 0.933]"],
       ["Clip duration", "0.479 [0.428, 0.531]", "0.467 [0.393, 0.540]", "0.622 [0.532, 0.670]"],
       ["RMS energy", "0.572 [0.525, 0.618]", "0.594 [0.537, 0.649]", "0.248 [0.073, 0.546]"],
       ["Repeat-padded fraction", "0.530 [0.478, 0.581]", "0.543 [0.471, 0.614]", "0.394 [0.348, 0.481]"]]
cwa = [span_w(3)]
cwa += [(W - sum(cwa)) // 3] * 3
cwa[-1] = W - sum(cwa[:-1])
gt, th = add_table(s, L, TOP, cwa, ha2, ra2, size=fit_size(cwa, ha2, ra2, sizes=(16, 15, 14, 13), bold_first_col=True), bold_first_col=True, max_h=2300000, fill=2000000)
add_text(s, L, TOP + th + 120000, W, BOT - (TOP + th + 120000), [
    ("bs", "Labels agree with the source files (0 mismatches): no label inversion.", 16),
    ("bs", "The from-scratch reference scores 0.99997 in the training domain and 0.219 to 0.286 on In-the-Wild: a data shift, not a label bug.", 16),
    ("bs", "Leading silence and RMS energy reverse sign between the corpora; with 12 speakers their intervals include 0.5.", 16),
    ("bs", "Exploratory, development slice; 0.5 means no information.", 16),
], "audio_text")

# licences# licences
s = new_slide("Fine-tuning grid and checkpoint v2",
    "Stage 1 of the fine-tuning iteration had nine runs, three encoders at three learning rates, one seed each, and the table gives validation and development AUC (E-005). The lowest rate was best for every encoder and five of the six runs at the higher rates lost most of the signal, with no non-finite loss or instability event recorded. The run of DINOv2 base at 1e-4 stopped when a data-loader worker died and was repeated once with the same configuration and seed; the error record is kept. "
    "Selection on validation data alone: leader DINOv2 base 0.9918 [0.9847, 0.9974], SigLIP base in the tie set (difference -0.0025 [-0.0068, +0.0009]), cheapest member SigLIP base (203 against 238 seconds of compute). "
    "The software status: the version-2 checkpoint format of the frozen system records how each score is made, keeps the hash of every file and checks the preprocessing identifier; 304 unit tests pass; the parity check on the graphics processor passed (11 items, largest difference of the calibrated score 0.0019 against a tolerance of 0.02, data/_results/freeze/parity.json). The web page still runs the checkpoint of the working version until the freeze. "
    "Evidence: docs/EXPERIMENTS.md E-005; docs/_agents/INTEGRATION.md; data/_results/freeze/parity.json.")
rows_g = [["DINOv2 small", "0.9869 / 0.9031", "0.7056 / 0.6412", "0.5871 / 0.5679"],
          ["SigLIP base", "0.9893 / 0.9210", "0.9814 / 0.8947", "0.5186 / 0.5432"],
          ["DINOv2 base", "0.9918 / 0.9036", "0.5218 / 0.5379", "0.5583 / 0.5667"]]
hdr_g = ["Encoder", "Rate 1e-5", "Rate 3e-5", "Rate 1e-4"]
cw_g = [span_w(3), span_w(3), span_w(3)]
cw_g.append(W - sum(cw_g))
sz_g = fit_size(cw_g, hdr_g, rows_g, sizes=(16, 15, 14, 13, 12), bold_first_col=True)
gt, th = add_table(s, L, TOP, cw_g, hdr_g, rows_g, size=sz_g, bold_first_col=True, max_h=1700000)
add_text(s, L, TOP + th + 60000, W, 360000, [("cap", "Validation AUC / development slice AUC, seed 0, exploratory.")], "grid_note")
add_card(s, L, TOP + th + 520000, W, BOT - (TOP + th + 520000), [
    ("h", "Checkpoint version 2 and the software status"),
    ("bs", "Per task the checkpoint records how the score is made: linear probe, fine-tuned encoder, or the mean logit of an ensemble of heads.", 16),
    ("bs", "The hash of every file is kept; the preprocessing identifier is checked against the recorded configuration; the audio preprocessing now includes trimming and normalising.", 16),
    ("bs", "304 unit tests pass. Parity check on the graphics processor passed: 11 items, largest difference 0.0019, tolerance 0.02.", 16),
    ("bs", "The web page still runs the checkpoint of the working version until the freeze.", 16),
], name="card_ckpt")

s = new_slide("LAV-DF control and subset",
    "The LAV-DF subset has 500 videos per cell (real, video fake only, audio fake only, both fake) in each of the training, validation and test splits, with the source video as the group and a fixed seed; only the training part, used to fit the fusion layer, and the validation part were scored, and the test part is sealed (D-057). The heads of E-011 were trained on 10,000 further videos of the official training split that share no source video with the scored parts. "
    "The table gives the values of E-011 and the post-hoc control, and the produced-only visual head of E-012. The control keeps only the produced videos, which all went through the data set pipeline, and asks each head for its own label: the visual head 0.5070 [0.4771, 0.5361], the speech head 0.8574 [0.8394, 0.8752]. The visual head separates the real originals from audio-only fakes, whose frames are untouched, with 0.9912 [0.9864, 0.9951]. "
    "The produced-only visual head of E-012 reaches 0.5169 [0.4945, 0.5392] within produced videos and 0.5342 over all videos; the difference to the head of E-011 within produced videos is +0.0100 [-0.0159, +0.0372]. The logistic-regression weights fitted for the fusion are 4.930 for the visual head and 0.813 for the speech head, and the sum of the log-odds reaches 0.9795. "
    "Limits: the identities of LAV-DF are not in its metadata, the early-stopping epoch of the speech heads was chosen on the validation part, and the check is post hoc. Evidence: docs/EXPERIMENTS.md E-010 to E-012; docs/DECISIONS.md D-057 to D-059.")
rows_l = [["Visual head, trained on LAV-DF", "0.7491 [0.7280, 0.7697]", "0.5070 [0.4771, 0.5361]", "0.9912 [0.9864, 0.9951]"],
          ["Speech head, trained on LAV-DF", "0.8604 [0.8435, 0.8769]", "0.8574 [0.8394, 0.8752]", "0.8711 [0.8474, 0.8920]"],
          ["Visual head, produced videos only", "0.5342 [0.5163, 0.5530]", "0.5169 [0.4945, 0.5392]", "0.5347 [0.5088, 0.5598]"],
          ["Frozen face head, trained on FF++", "0.3838 [0.3599, 0.4079]", "not applicable", "0.2876 [0.2568, 0.3174]"]]
hdr_l = ["Head", "Own label, all videos", "Own label, produced videos only", "Real originals against audio-only fakes"]
cw_l = [span_w(3), span_w(3), span_w(3)]
cw_l.append(W - sum(cw_l))
sz_l = 16
gt, th = add_table(s, L, TOP, cw_l, hdr_l, rows_l, size=sz_l, bold_first_col=True, max_h=2900000)
add_text(s, L, TOP + th + 80000, W, BOT - (TOP + th + 80000), [
    ("b", "Subset: 500 videos per cell and split, source video as the group; test part sealed.", 18),
    ("b", "Fusion weights: visual 4.930, audio 0.813; sum of log-odds 0.9795; best rule minus better head +0.0023 [+0.0009, +0.0040].", 18),
    ("cap", "Exploratory; the control is post hoc and labelled as such.")], "lavdf_backup")

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
    "The exclusion rule is by target identity or by celebrity source identity (1,477 by target, 602 only by source), and the source part is required: in a face swap named id_T_id_S the visible face is the source identity, so a sample that shows a development face as the source would otherwise leak it. "
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
], size="auto", bold_first_col=True, max_h=1900000, fill=1900000)
add_text(s, L, TOP + 1500000 + th + 100000, W, BOT - (TOP + 1500000 + th + 100000), [
    ("s", "Real is 9.7 percent of the pool. The 44 identity groups govern the intervals, and balanced accuracy and the threshold carry more noise than the AUC."),
    ("s", "A sample is excluded when its target identity or its celebrity source identity is one of the 15 development identities (1,477 by target, 602 only by source): in a face swap named by two identities the visible face is the source identity.")], "pp_note")

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


MIRROR = {"Protocol: group-disjoint splits": "fig02_splits", "Method: file to calibrated score": "fig01_pipeline", "Encoder shortlist": "fig08_shortlist",
          "rPPG: test the signal first": "fig05_rppg", "Timeline to 30 October": "fig06_timeline",
          "Staircase and confirmatory claims": "fig04_staircase", "Web demo (real checkpoint)": "fig07_web"}
for _sl in prs.slides:
    _t = _sl.shapes.title.text_frame.text if _sl.shapes.title is not None else ""
    if _t in MIRROR:
        _tf = _sl.notes_slide.notes_text_frame
        if _t.startswith("Web demo"):
            _tf.text = _tf.text + " The report carries the labelled figure of the page against the development stand-in (fig07_web); this slide shows the real-checkpoint screenshots instead."
        else:
            _tf.text = _tf.text + " This slide is drawn natively and mirrors " + MIRROR[_t] + " of the paper and the report, with the same wording, so that the text is large enough to read when projected."

cp = prs.core_properties
cp.title = 'Multimodal Deepfake Detection for Images, Video and Audio: A Hypothesis-Driven Study Under a Group-Disjoint Protocol'
cp.author = 'Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh'
cp.last_modified_by = 'Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh'
cp.subject = 'Final year project presentation'
cp.keywords = ''
cp.comments = ''
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
for w in WARN:
    print("WARN", w)
