"""Builds deliverables_v2/presentation/Deepfake_Detection_Presentation_v2.pptx (Phase B deck: 14 main slides, a backup divider and 10 backup slides).

Template (read only): Document Template/DeepFake_Detection_System_Presentation (2).pptx (master, layouts, theme, logos, slide size).
Evidence: deliverables_v2/CONTENT_MAP.md. Helpers are copied from slides/build_slides.py; slide_common.py is a copy of slides/slide_common.py.
Run:   python deliverables_v2/presentation/build_deck_v2.py
Rules (CLAUDE.md): no em or en dashes; every number from CONTENT_MAP.md; no IDs in visible text (IDs only in speaker notes).
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import slide_common as C  # noqa: E402
from slide_common import (BOT, COLW, DARK, FONT, GREY, H, L, LINE, MID, R, RED, TOP, W, col_x, span_w)  # noqa: E402

SRC = os.path.join(ROOT, "Document Template", "DeepFake_Detection_System_Presentation (2).pptx")
OUT = os.path.join(HERE, "Deepfake_Detection_Presentation_v2.pptx")
FIG = os.path.join(ROOT, "deliverables_v2", "figures")
CHARTS = os.path.join(HERE, "_charts")
os.makedirs(CHARTS, exist_ok=True)
# ---- Dashboard screenshots for slide 10 (neither shows dataset media). To show the redesigned v2 website instead, replace these
# ---- two paths (any PNG or JPG; both are drawn at the same width, aspect ratio kept) and rebuild.
DASHBOARD_SCREENSHOTS = [
    os.path.join(ROOT, "deliverables_v2", "website", "test_evidence", "screens", "1_face_swap_video_ffpp_val_1366_report.png"),
    os.path.join(ROOT, "deliverables_v2", "website", "test_evidence", "screens", "1_face_swap_video_ffpp_val_1366_report.png"),
]
# Each entry is a list of crop boxes (left, top, right, bottom, pixels) stacked vertically (the face card crop leaves out the limitations list); None means no crop. Boxes cut away page margins so the text is larger on the slide.
# (older note:) Crop boxes (left, top, right, bottom, pixels) that cut away page margins so the text is larger on the slide; None means no crop.
DASHBOARD_CROPS = [[(137, 478, 1230, 866)], [(137, 944, 671, 1283)]]
DASHBOARD_SCREENSHOT_ALTS = ["Redesigned dashboard for a face-swap video: a donut that counts the checks by outcome (not a probability or a verdict) next to the summary of what the checks found",
                             "Redesigned dashboard card for the face manipulation check: the label Possible manipulation, high uncertainty, and a model-output bar marked not a probability with the decision line"]
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
# CHARTS (matplotlib images, saved in _charts/; every value is copied from deliverables_v2/CONTENT_MAP.md)
# =====================================================================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14.5, "axes.edgecolor": "#595959", "axes.linewidth": 0.8,
                     "xtick.color": "#262626", "ytick.color": "#262626", "axes.labelcolor": "#262626"})
C_I0, C_FIN, C_XC, C_CAND = "#7F7F7F", "#" + RED, "#4F6D95", "#C98A1B"


def forest(fname, panels, size, xlim, xlabel, legend=True, legend_loc="lower left", hspace=0.55):
    """Dot-and-interval chart. panels: list of (panel_title or None, rows); row = (label, [(series, m, lo, hi) ...]).
    A series is a dict with name, color, marker, filled (set once per name; the first occurrence defines the legend)."""
    h_ratios = [max(1, len(r)) for _, r in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=size, dpi=220, gridspec_kw={"height_ratios": h_ratios, "hspace": hspace}, squeeze=False)
    seen = {}
    for ax, (ptitle, rows) in zip(axes[:, 0], panels):
        n = len(rows)
        for ri, (label, pts) in enumerate(rows):
            y = n - 1 - ri
            k = len(pts)
            for pi, (ser, m, lo, hi) in enumerate(pts):
                off = 0 if k == 1 else (pi - (k - 1) / 2) * 0.26
                yy = y - off
                filled = ser.get("filled", True)
                ax.plot([lo, hi], [yy, yy], color=ser["color"], lw=2.2, solid_capstyle="butt", zorder=2)
                h = ax.plot([m], [yy], marker=ser["marker"], ms=8.5, color=ser["color"], mfc=ser["color"] if filled else "white",
                            mew=1.8, ls="none", zorder=3)[0]
                seen.setdefault(ser["name"], h)
        ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=14)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*xlim)
        ax.axvline(0.5, color="#BFBFBF", lw=1, ls="--", zorder=1)
        ax.grid(axis="x", color="#E6E6E6", lw=0.7, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ptitle:
            ax.set_title(ptitle, loc="left", fontsize=15, fontweight="bold", color="#" + RED, pad=4)
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel(xlabel, fontsize=14)
    top = 0.95
    if legend and len(seen) > 1:
        fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=len(seen), fontsize=13.5, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.3)
        top = 0.88
    fig.subplots_adjust(left=0.36, right=0.97, top=top, bottom=0.15)
    path = os.path.join(CHARTS, fname)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


S_I0 = dict(name="First version (I-0)", color=C_I0, marker="o")
S_FIN = dict(name="Final system", color=C_FIN, marker="s")
S_XC = dict(name="Re-run Xception baseline", color=C_XC, marker="^")

CH_VIDEO = forest("chart_video_sealed.png", [(None, [
    ("FF++ c23 test_in\n(700 clips)", [(S_I0, .7946, .7667, .8243), (S_FIN, .9899, .9826, .9959), (S_XC, .9895, .9823, .9952)]),
    ("Celeb-DF v2\n(4,801)", [(S_I0, .6322, .6018, .6658), (S_FIN, .8999, .8759, .9210), (S_XC, .8527, .8237, .8777)]),
    ("Celeb-DF family\n(7,650; confirmatory)", [(S_I0, .6238, .6015, .6486), (S_FIN, .8218, .8020, .8393), (S_XC, .8042, .7841, .8223)]),
    ("Celeb-DF++ fakes vs\nv2 reals (3,593)", [(S_I0, .6119, .5821, .6427), (S_FIN, .7105, .6877, .7331), (S_XC, .7351, .7083, .7599)]),
    ("DFD test_x\n(3,431)", [(S_I0, .5927, .5629, .6258), (S_FIN, .8797, .8566, .9041), (S_XC, .8847, .8655, .9052)]),
])], (7.1, 4.7), (0.5, 1.0), "ROC-AUC, 95% group-bootstrap interval", legend_loc="lower left")

CH_AUDIO = forest("chart_speech_sealed.png", [(None, [
    ("ASVspoof 2019 LA test_in\n(71,237 clips, 67 speakers)", [(S_I0, .9684, .9564, .9777), (S_FIN, .9835, .9798, .9872)]),
    ("In-the-Wild test_x\n(24,788 clips, 42 speakers;\nconfirmatory)", [(S_I0, .7516, .6802, .8035), (S_FIN, .9148, .8896, .9383)]),
])], (7.1, 3.0), (0.6, 1.0), "ROC-AUC, 95% group-bootstrap interval", legend_loc="lower left")

S_T = dict(name="Frozen head, sealed test", color=C_FIN, marker="s")
S_E29 = dict(name="Retrained head (not installed)", color=C_CAND, marker="D")
CH_IMAGE = forest("chart_image_sealed.png", [(None, [
    ("ProGAN test_in\n(8,000 images)", [(S_T, 1.0000, .9999, 1.0000)]),
    ("CNN-synthesis test_x\n(48,805)", [(S_T, .9316, .8933, .9694)]),
    ("Diffusion test_x\n(10,000)", [(S_T, .9354, .8967, .9665)]),
    ("Modern generators,\n21 generators (11,550)", [(S_T, .7039, .6855, .7229), (S_E29, .8839, .8732, .8942)]),
])], (7.1, 4.4), (0.5, 1.025), "ROC-AUC (macro AUC for modern generators), 95% interval", legend_loc="lower left")

S_OPEN = dict(name="Adopted step", color=C_FIN, marker="s")
S_REJ = dict(name="Tried, not adopted", color=C_I0, marker="o", filled=False)
S_BASE = dict(name="Frozen probe", color=C_I0, marker="o")
CH_LADDER = forest("chart_ladder_dev.png", [
    ("Face video, development slice (15 identities)", [
        ("I-0 working version\n(DINOv2 small probe)", [(S_BASE, .6933, .6639, .7314)]),
        ("SigLIP base, frozen probe", [(S_BASE, .7396, .7031, .7832)]),
        ("I-1 SigLIP base,\nfine-tuned", [(S_OPEN, .9096, .8776, .9388)]),
        ("I-1b augmentation, mixup", [(S_REJ, .9315, .9055, .9548)]),
        ("I-1c mean of 3 encoders", [(S_REJ, .9192, .8948, .9457)]),
    ]),
    ("Speech, development slice (12 speakers)", [
        ("I-0 frozen probe", [(S_BASE, .6100, .5434, .7256)]),
        ("I-2 trim silence", [(S_OPEN, .6723, .6126, .7584)]),
        ("I-3 noise, codec, reverb", [(S_OPEN, .7455, .7052, .8121)]),
        ("I-4 three encoders", [(S_OPEN, .8130, .7523, .8808)]),
    ]),
], (6.9, 5.3), (0.5, 1.0), "ROC-AUC, 95% group-bootstrap interval", legend_loc="lower left", hspace=0.5)


def pic_fit(slide, path, x, y, w, h, alt, align="left"):
    """Places a picture inside the box (x, y, w, h) at its own aspect ratio (never stretched)."""
    from PIL import Image
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = int(iw * sc), int(ih * sc)
    px = x if align == "left" else x + (w - pw) // 2
    return add_pic(slide, path, px, y, w=pw, h=ph, alt=alt), pw, ph


def caption(slide, x, y, w, h, text):
    return add_text(slide, x, y, w, h, [("cap", text)], "cap")


# =====================================================================================================================
# SLIDE 1: title
# =====================================================================================================================
_count[0] += 1
s = prs.slides.add_slide(LAYOUT["TITLE"])
set_title(s.shapes.title, "Multimodal Deepfake Detection for Images, Video and Audio", size=24, y=40000, h=820000, check=False)
t1 = s.shapes.title
t1.left, t1.width = Emu(2600000), Emu(C.SLIDE_W - 2 * 2600000)
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
for i, txt in enumerate(["Department of Electronics and Computer Engineering",
                         "K. J. Somaiya School of Engineering, Somaiya Vidyavihar University"]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt
    style_run(r, 18, False, False, GREY)
    set_spacing(p, 0, 2)
PH_Y, PH_H = 1600000, 2250000
half = (W - 120000) // 2
for k, (fn, alt) in enumerate([("campus_aryabhat_building.png", "Aryabhat building of the K. J. Somaiya College of Engineering campus"),
                               ("campus_bhaskaracharya_building.png", "Bhaskaracharya building of the K. J. Somaiya College of Engineering campus")]):
    pic = add_pic(s, os.path.join(BRD, fn), L + k * (half + 120000), PH_Y, alt=alt)
    ar_box = half / PH_H
    ar_img = pic.image.size[0] / pic.image.size[1]
    pic.width, pic.height = Emu(half), Emu(PH_H)
    pic.left = Emu(L + k * (half + 120000)); pic.top = Emu(PH_Y)
    if ar_img < ar_box:
        keep = ar_img / ar_box
        pic.crop_top = pic.crop_bottom = (1 - keep) / 2
    else:
        keep = ar_box / ar_img
        pic.crop_left = pic.crop_right = (1 - keep) / 2
add_text(s, L, 3930000, W, 400000, [("h", "Group members")], "members_label")
for k, (nm, roll) in enumerate([("Shivansh Gupta", "16014023052"), ("Vaibhav Shiroorkar", "16014024805"), ("Adithya Jithesh", "16014024803")]):
    add_text(s, col_x(4 * k), 4340000, span_w(4), 760000, [("p", f"**{nm}**"), ("s", f"Roll No: {roll}")], "member")
add_text(s, L, 5150000, W, 500000, [("p", "**Guide:** Dr. Tejashree Phatak, Assistant Professor")], "guide")
add_footer(s, 1)

s.notes_slide.notes_text_frame.text = (
    "This is a final-year project in the Department of Electronics and Computer Engineering: one local system that checks images, videos and the speech inside videos for manipulation. "
    "Introduce the three members in this order: Shivansh Gupta, Vaibhav Shiroorkar and Adithya Jithesh, and the guide, Dr. Tejashree Phatak, Assistant Professor. "
    "The talk has fourteen main slides: the problem, the protocol, the method, the working version, the iteration ladder, the three sealed tests, the speech result and its shortcut, the image result, the platform and what the tests after the freeze allow it to claim, the dashboard, the Gemini question, the comparison with published numbers and the limits, and the conclusion. Backup slides follow the divider. "
    "State at the start that sealed test sets were read once per frozen version, that only three comparisons were declared confirmatory in advance, and that every other sealed number is exploratory. "
    "Evidence pointers: docs/IDENTITY.md for the identity fields; deliverables_v2/CONTENT_MAP.md and PHASE_B_FIXLIST.md for every number and wording in this deck; decisions D-089 to D-091.")

Y0 = TOP + 450000          # content area below the one-line message
H0 = BOT - Y0


def lead(slide, text):
    """The one message of a main slide, first line under the title."""
    return add_text(slide, L, TOP, W, 400000, [("p", "**" + text + "**", 18)], "text_lead")


# =====================================================================================================================
# SLIDE 2: problem, scope and objectives
# =====================================================================================================================
s = new_slide("Problem, scope, objectives",
    "The problem: a fake can be a swapped or re-enacted face, a generated image or video, or synthetic or converted speech, and a detector that scores well on its own training data can fail on new data. "
    "The aim is a documented path, not the best model in the world: a simple working version first, then improvements that each start from a written hypothesis, judged on data split by source identity. "
    "Context only: DeepfakeBench trained plain detectors on FaceForensics++ c23 and reported a mean frame-level AUC of 0.9450 within the domain and 0.7718 across other datasets for Xception (Yan et al., 2023). That is their protocol and not our result. "
    "The cards on the right separate what we claim, with intervals and only on the data stated, from what we explicitly do not claim: no overall verdict, no audio-visual, pulse or generated-video detection, no detection of modern generated images and no Gemini capability. "
    "Emotion analysis was removed from scope by the user and watermark detection is excluded. "
    "Evidence: CONTENT_MAP sections 3 and 10 (yan2023deepfakebench, R-017, PARTIAL); D-069, D-077.")
lead(s, "One local checker for images, video and speech, built as a documented path.")
for k, (hd, tx, fl) in enumerate([("Image", "generated by GAN or diffusion", F_BLUE), ("Video", "swapped or re-enacted faces", F_GREEN),
                                  ("Audio", "synthetic or converted speech", F_ORANGE)]):
    dbox(s, col_x(4 * k), Y0, span_w(4), 650000, f"**{hd}**" + chr(10) + tx, fill=fl, size=16)
add_card(s, col_x(0), Y0 + 800000, span_w(6), H0 - 800000, [
    ("h", "Objectives"),
    ("n", "1.  Build one local system for images, video and speech.", 16),
    ("n", "2.  Fix the evaluation protocol before any model.", 16),
    ("n", "3.  Improve in steps, each from a written hypothesis; keep failed runs.", 16),
    ("n", "4.  Report one finding per module with its limits, never one overall score.", 16),
    ("p", "Context, their protocol: Xception trained on FaceForensics++ c23 has a mean frame-level AUC of 0.9450 within the domain and 0.7718 across datasets (Yan et al., 2023).", 16),
], name="card_objectives")
add_card(s, col_x(6), Y0 + 800000, span_w(6), 1550000, [
    ("h", "Claimed, with intervals"),
    ("b", "Face-video and speech detectors with sealed results", 16),
    ("b", "An image probe, tested on older and newer generators", 16),
    ("b", "A local platform with eight modules", 16),
], name="card_claimed")
add_card(s, col_x(6), Y0 + 2450000, span_w(6), H0 - 2450000, [
    ("h", "Not claimed"),
    ("b", "An overall verdict or fused score", 16),
    ("b", "Audio-visual, pulse or generated-video detection", 16),
    ("b", "Detection of modern generated images", 16),
    ("b", "Any Gemini capability", 16),
], name="card_notclaimed")

# =====================================================================================================================
# SLIDE 3: protocol and data
# =====================================================================================================================
s = new_slide("Protocol and data",
    "Every dataset is split once, by source identity: no person, speaker or source video appears on both sides of a split. "
    "Validation fits the temperature and the threshold. The development slice ranks candidates and may be looked at repeatedly, so its results are exploratory. The sealed tests are read once per frozen version through a guarded command that refuses a second read; the final reading produced 74 ledger entries and no repeat. "
    "Three comparisons, T1, T2 and T3, were declared before any sealed read (committed 2026-10-25 at 18:44, before the first sealed read); everything else is exploratory. "
    "The Celeb-DF family test has 7,650 videos in 344 groups: 44 identities and 300 single YouTube videos, with 744 real and 6,906 fake videos, so the intervals are governed by the number of groups. "
    "All datasets are research-only licences; no dataset frames or clips appear in this deck. "
    "Evidence: CONTENT_MAP sections 6 and 7; docs/PROTOCOL.md; E-014; D-062, D-066.")
lead(s, "Splits are group-disjoint, sealed sets are read once, three tests were declared first.")
steps = ["**Split once**\nby source identity", "**Train**\nfit the model", "**Validation**\ntemperature, threshold", "**Development slice**\nexploratory only", "**Sealed tests**\nread once, guarded"]
fills = [F_GREY, F_GREEN, F_GREEN, F_ORANGE, F_RED]
gap = 200000
bw = (W - 4 * gap) // 5
for k, tx in enumerate(steps):
    dbox(s, L + k * (bw + gap), Y0, bw, 700000, tx, fill=fills[k], size=C.SZ_TABLE)
    if k < 4:
        darrow(s, L + k * (bw + gap) + bw, Y0 + 350000, L + (k + 1) * (bw + gap), Y0 + 350000)
wd = [1250000, 2450000, W - 3700000]
rows = [["Face video", "FaceForensics++ c23", "FF++ test_in; Celeb-DF family (7,650 videos, 344 groups: 44 identities and 300 single YouTube videos); DFD"],
        ["Speech", "ASVspoof 2019 LA", "ASVspoof test_in; In-the-Wild (42 speakers)"],
        ["Image", "ForenSynths ProGAN", "ProGAN test_in; CNN-synthesis and diffusion sets"]]
gt, th = add_table(s, L, Y0 + 850000, wd, ["Task", "Train and validate", "Sealed tests"], rows, size=14, bold_first_col=True, name="table_data", pad=20000)
yc = Y0 + 850000 + th + 130000
add_card(s, col_x(0), yc, span_w(6), BOT - yc, [
    ("h", "Why this protocol"),
    ("b", "No group on both sides of a split.", 16),
    ("b", "A guard refuses a second read of a sealed set.", 16),
    ("b", "Tuning uses only validation and the development slice.", 16),
], name="card_why")
add_card(s, col_x(6), yc, span_w(6), BOT - yc, [
    ("h", "Declared before any sealed read"),
    ("b", "T1 video: final minus first version.", 16),
    ("b", "T2 speech: final minus first version.", 16),
    ("b", "T3 video: final minus re-run Xception.", 16),
], name="card_tests")

# =====================================================================================================================
# SLIDE 4: method
# =====================================================================================================================
s = new_slide("Method: file to score",
    "Each task uses one frozen pretrained encoder and a linear head, the only trained part. A video gives 16 uniformly spaced frames, the largest face found by YuNet is cropped to 224 by 224 pixels, and the clip score is the mean of the frame probabilities after a temperature. Audio uses 4 seconds at 16 kHz, silence trimmed and peak normalised. "
    "The temperature minimises the negative log-likelihood on the validation split and the threshold maximises balanced accuracy on the same calibrated validation scores; both are then fixed for every other dataset. "
    "The mathematics in one line each: the clip score, the paired group bootstrap with 2,000 resamples where the group is the unit, and the non-inferiority rule with a margin of 0.0100 that was set before the sealed read as twice the larger seed standard deviation of the baseline with a floor of 0.01; the floor decides. Details are in backup slide B1. "
    "Encoders were chosen by a measured sweep: the cheapest candidate whose paired interval against the leader includes zero wins. The result is DINOv2 small for face video, CLIP ViT-L/14 for images and HuBERT base for speech. "
    "Score meaning: temperature scaling makes a score a probability only for content like the validation split, so a score ranks files and is never the probability that a file is fake. "
    "Evidence: CONTENT_MAP sections 5 and 6; E-001 to E-004; D-061, D-066.")
lead(s, "A file becomes a score: frozen encoder, linear head, temperature, threshold.")
steps = ["**Preprocess**\n16 face frames or 4 s audio", "**Frozen encoder**\nchosen by sweep", "**Linear head**\nonly trained part", "**Temperature**\nfitted on validation", "**Threshold**\nfixed on validation"]
fills = [F_GREY, F_BLUE, F_GREEN, F_ORANGE, F_ORANGE]
for k, tx in enumerate(steps):
    dbox(s, L + k * (bw + gap), Y0, bw, 800000, tx, fill=fills[k], size=C.SZ_TABLE)
    if k < 4:
        darrow(s, L + k * (bw + gap) + bw, Y0 + 400000, L + (k + 1) * (bw + gap), Y0 + 400000)
add_card(s, col_x(0), Y0 + 1000000, span_w(7), H0 - 1000000, [
    ("h", "The mathematics, one line each"),
    ("p", "Clip score: s = sigmoid( logit( mean frame probability ) / T )", 20),
    ("p", "Interval: 2,000 paired group-bootstrap resamples, 95%", 20),
    ("p", "T3 rule: non-inferior if the lower bound is above -0.0100", 20),
    ("p", "Encoder rule: ties with the leader, then the cheapest wins", 20),
], name="card_maths")
add_card(s, col_x(7), Y0 + 1000000, span_w(5), H0 - 1000000, [
    ("h", "What a score means"),
    ("p", "A score ranks files. The temperature is fitted on validation, so it is a probability only for content like validation. No output is the probability that a file is fake.", 20),
], name="card_semantics")

# =====================================================================================================================
# SLIDE 5: working version and what we learned
# =====================================================================================================================
s = new_slide("Working version, lessons",
    "The first working version, called I-0, uses one frozen pretrained encoder per task with a linear head, a temperature fitted on validation data and a threshold chosen on validation. "
    "The table shows the lesson: face video was weak everywhere, with a training-set AUC of only 0.8367, which is underfitting, and speech scored 0.9942 on validation but only 0.6100 on the development slice from another corpus, which is a collapse out of corpus. "
    "The diagnostics also showed that leading silence and level reverse sign between the two speech corpora, which prepares the shortcut discussion on slide 8, and that the temperature fitted on validation did not transfer: the calibration error rose on the development slice for both video and speech. "
    "Image detection looked excellent, but only on ProGAN-era generators. "
    "These diagnostics decided the levers on the iteration ladder. All development-slice values are exploratory with wide intervals. "
    "Evidence: E-001 to E-004, E-007, CONTENT_MAP section 8.1; backup slides B3 and B4.")
lead(s, "Face video underfits and speech collapses outside its corpus.")
hdr = ["Task (frozen encoder)", "Validation", "Development slice"]
rows = [["Face video (DINOv2 small)", "0.7985", "0.6933"], ["Image (CLIP ViT-L/14)", "1.0000", "0.9935"], ["Speech (HuBERT base)", "0.9942", "0.6100"]]
wd = [2800000, 1400000, span_w(7) - 4200000]
gt, th = add_table(s, L, Y0, wd, hdr, rows, size=16, bold_first_col=True, name="table_i0", pad=90000)
caption(s, L, Y0 + th + 60000, sum(wd), 520000, "ROC-AUC of the working version I-0. The development slice is exploratory, with wide intervals.")
yc = Y0 + th + 700000
add_card(s, L, yc, span_w(7), 1250000, [
    ("h", "How the encoders were chosen"),
    ("p", "A measured sweep: the cheapest candidate that ties the leader wins.", 18),
], name="card_rule_encoder")
add_card(s, col_x(7), Y0, span_w(5), H0, [
    ("h", "What we learned"),
    ("b", "Frozen face-video features underfit even the training data (training AUC 0.8367).", 18),
    ("b", "Speech: 0.9942 on validation, 0.6100 on the development slice.", 18),
    ("b", "Leading silence and level cues reverse sign between speech corpora.", 18),
    ("b", "The temperature fitted on validation did not transfer.", 18),
], name="card_lessons")

# =====================================================================================================================
# SLIDE 6: iteration ladder
# =====================================================================================================================
s = new_slide("Iteration ladder",
    "Each step on the chart started from a written hypothesis, a success line and a failure line, and is judged on the development slice, so every verdict here is exploratory and the intervals are wide: 15 identities for video and 12 speakers for speech. "
    "For face video, the gain that isolates fine-tuning is the one over the frozen probe of the same encoder: SigLIP base probe 0.7396 [0.7031, 0.7832] to fine-tuned 0.9096 [0.8776, 0.9388], paired +0.1700 [+0.1228, +0.2114]. Do not join the working version at 0.6933, which used a different encoder, with 0.9096 as one gain. The re-run Xception gain over the working version, +0.1783, is consistent with it but changes the encoder as well. "
    "Two further video levers did not enter: augmentation with mixup lost validation accuracy beyond the allowance of 0.0079, a trade-off, and the three-encoder mean gave no gain its interval could show. "
    "For speech, trimming silence, augmentation and averaging three encoders each met their line; the point estimates rise, adjacent intervals overlap. "
    "The audio-visual fusion on LAV-DF and the pulse check were tried and are on the backup slides. "
    "Evidence: E-005 to E-017, E-013, E-016, E-017; CONTENT_MAP section 8; fix list items A3, A10.")
lead(s, "A change entered only when it met its written success line.")
pic, pw, ph = pic_fit(s, CH_LADDER, L, Y0, span_w(6), 3650000, "Dot and interval chart of development-slice AUC along the improvement steps for face video, with the frozen probe of the same encoder, and for speech")
caption(s, L, Y0 + ph + 40000, span_w(6), 650000,
        "Development slice (exploratory): face video, 15 identities; speech, 12 speakers. Hollow marks were tried and not adopted.")
add_card(s, col_x(6), Y0, span_w(6), H0, [
    ("h", "Reading"),
    ("b", "Video: fine-tuning adds +0.1700 [+0.1228, +0.2114] over the frozen probe of the same encoder (exploratory).", 18),
    ("b", "Speech: three steps, each met its line; point estimates rise, adjacent intervals overlap.", 18),
    ("b", "Not adopted: mixup (a trade-off), three-encoder mean (no gain), LAV-DF fusion (a shortcut), pulse check (failed).", 18),
], name="card_reading")

# =====================================================================================================================
# SLIDE 7: sealed tests T1 to T3
# =====================================================================================================================
s = new_slide("Sealed tests T1 to T3",
    "Three tests were declared before any sealed read and each sealed set was read once. T1 is the final video system minus the working version on the Celeb-DF family: AUC 0.6238 to 0.8218, paired difference +0.1980 [+0.1688, +0.2242], improved by the declared rule and also at the 98.33 percent line [+0.1627, +0.2303]. "
    "T2 is the same for speech on In-the-Wild: 0.7516 to 0.9148, +0.1633 [+0.1177, +0.2266], 98.33 percent line [+0.1085, +0.2481]; part of that gain removes a shortcut, which is the next slide. "
    "T3 compares the final video system with a re-run Xception baseline, 0.8042 [0.7841, 0.8223], trained with the same recipe: +0.0176 [+0.0025, +0.0327]. The wording that must be used is: non-inferior at both levels (margin 0.0100); the declared 95 percent superiority rule was met but does not survive the Bonferroni correction (98.33 percent interval [-0.0011, +0.0355]); the baseline ran at 224 px. Mention superiority only together with the Bonferroni qualifier. "
    "The 98.33 percent line was declared as a robustness check beside the 95 percent decision rule; Bonferroni control holds under any dependence between the three tests, so a reader who wants family-wise control should read that column. "
    "Exploratory support: on the Celeb-DF++ fakes with v2 reals the baseline is higher (0.7351 against 0.7105), and on DFD (0.8847 against 0.8797). "
    "Evidence: E-014; D-061, D-062, D-066, D-091; backup slide B10 for the sealed charts.")
lead(s, "T1 and T2 improved by the declared rule; T3 is non-inferior, not superior after Bonferroni.")
wd = [1150000, 2050000, 1600000, 1900000, W - 6700000]
rows = [["T1 video", "Celeb-DF family, 7,650 videos, 344 groups", "0.6238 to 0.8218", "+0.1980 [+0.1688, +0.2242]", "Improved by the declared rule; holds at 98.33%: [+0.1627, +0.2303]"],
        ["T2 speech", "In-the-Wild, 42 speakers", "0.7516 to 0.9148", "+0.1633 [+0.1177, +0.2266]", "Improved by the declared rule; holds at 98.33%: [+0.1085, +0.2481]. Part of the gain removes a shortcut"],
        ["T3 video vs re-run Xception", "Celeb-DF family; Xception 0.8042 [0.7841, 0.8223]", "0.8042 to 0.8218", "+0.0176 [+0.0025, +0.0327]",
         "Non-inferior at both levels (margin 0.0100). The 95% superiority rule was met but does not survive Bonferroni (98.33% interval [-0.0011, +0.0355]). The baseline ran at 224 px"]]
gt, th = add_table(s, L, Y0, wd, ["Test", "Sealed set", "AUC, before to after", "Paired difference [95%]", "Reading"], rows, size=15, bold_first_col=True, name="table_tests", pad=20000)
yc = Y0 + th + 120000
add_card(s, L, yc, W, BOT - yc, [
    ("b", "The 98.33% line was declared beside the 95% rule; a reader who wants family-wise control over the three tests reads the 98.33% column.", 18),
    ("b", "T3 uses one baseline family at one input size; every sealed set was read once.", 18),
], name="card_readtests")

# =====================================================================================================================
# SLIDE 8: speech, the T2 gain and its shortcut
# =====================================================================================================================
s = new_slide("Speech: gain and shortcut",
    "Test 2 shows that the final speech system ranks In-the-Wild clips better than the working version, 0.7516 to 0.9148, +0.1633 [+0.1177, +0.2266]; it does not show how much of that is better detection of synthesis. "
    "Part of the speech gain is the removal of a shortcut: in the training corpus the leading silence and the level of a clip carry a weak class signal that points the other way on In-the-Wild. The single-feature AUC of leading silence is 0.375 on ASVspoof development data against 0.719 on the In-the-Wild development slice, and of RMS energy 0.594 against 0.248; this is consistent with a sign-flipped shortcut, and 12 speakers do not establish it. "
    "On the exploratory sealed staircase the first speech lever, which trims silence and normalises the peak, accounts for 0.061 of the 0.163 gain (0.7516 to 0.8125). A level cue remains on In-the-Wild after normalisation: RMS AUC 0.135 [0.054, 0.382] on the development slice. "
    "The staircase values are point estimates that rise; adjacent intervals overlap. "
    "Evidence: E-007, E-014, M-004, CONTENT_MAP section 9; fix list item A2.")
lead(s, "Part of the speech gain removes a shortcut; not all of it is better detection.")
wd = [2600000, 1500000, 2200000, 1900000, W - 8200000]
gt, th = add_table(s, L, Y0, wd, ["Sealed In-the-Wild AUC", "I-0", "I-2 trim, normalise", "I-3 augment", "I-4 three encoders"],
                   [["Point estimate (exploratory)", "0.7516", "0.8125", "0.8672", "0.9148"]], size=16, bold_first_col=True, name="table_stair", pad=40000)
y2 = Y0 + th + 150000
wd2 = [3700000, 3500000, W - 7200000]
gt2, th2 = add_table(s, L, y2, wd2, ["Single feature as fake score (AUC)", "ASVspoof development", "In-the-Wild development"],
                     [["Leading silence", "0.375 [0.299, 0.453]", "0.719 [0.425, 0.933]"],
                      ["RMS energy", "0.594 [0.537, 0.649]", "0.248 [0.073, 0.546]"]], size=16, bold_first_col=True, name="table_feat", pad=40000)
yc = y2 + th2 + 150000
add_card(s, L, yc, W, BOT - yc, [
    ("b", "Trimming and normalising account for 0.061 of the 0.163 gain; a level cue remains: RMS AUC 0.135 [0.054, 0.382], development slice.", 18),
    ("b", "Consistent with a sign-flipped shortcut; 12 speakers do not establish it.", 18),
    ("b", "Test 2 shows better ranking on In-the-Wild, not how much is better detection of synthesis.", 18),
], name="card_shortcut")

# =====================================================================================================================
# SLIDE 9: image analysis
# =====================================================================================================================
s = new_slide("Image analysis",
    "Still images with a face go to the face-video head; all other images go to a frozen CLIP ViT-L/14 with a linear probe trained on ProGAN images. "
    "The first three rows of the chart are sealed exploratory tests of this frozen head on ProGAN-era generators, AUC 0.9316 to 1.0000. "
    "The last row is the same 11,550 images of the Community Forensics-Eval benchmark, 21 generators including recent commercial ones: the frozen head reaches a macro AUC of 0.7039 and misses 0.7395 of generated images at its threshold while flagging 0.0418 of real ones, which the preregistered line classed as inadequate. A file-format cue alone reaches macro AUC 0.9301, so the benchmark is confounded by format. "
    "A head retrained on 125 recent generators ranks better (macro AUC 0.8839, paired gain +0.1800) but at its validation threshold flags 0.5213 of real images, so it is a research artefact and is not installed. "
    "The dashboard labels a flag 'Possible AI-generation (weak evidence)' and a low score 'Inconclusive'. "
    "Evidence: E-014, E-028, E-029; CONTENT_MAP sections 8.3 and 11.")
lead(s, "The ProGAN-trained probe misses 0.7395 of images from 21 recent generators.")
pic, pw, ph = pic_fit(s, CH_IMAGE, L, Y0, span_w(7), 3500000, "Dot and interval chart of the image head's AUC by test set: ProGAN 1.0000, CNN-synthesis 0.9316, diffusion 0.9354, modern generators 0.7039 frozen and 0.8839 retrained")
caption(s, L, Y0 + ph + 50000, span_w(7), 650000,
        "Image head, sealed test sets, each read once (exploratory). Last row: Community Forensics-Eval, macro AUC over 21 generators, same images for both heads.")
add_text(s, col_x(7), Y0, span_w(5), H0, [
    ("b", "Frozen CLIP ViT-L/14 with a linear probe; faces use the video head.", 16),
    ("b", "Older generators: AUC 0.9316 to 1.0000 (exploratory).", 16),
    ("b", "Modern generators: macro AUC 0.7039; 0.7395 of generated images missed, 0.0418 of real flagged.", 16),
    ("b", "A file-format cue alone reaches macro AUC 0.9301: the benchmark is confounded.", 16),
    ("b", "Retrained head ranks better (0.8839) but flags 0.5213 of real images: not installed.", 16),
], "text_image")

# =====================================================================================================================
# SLIDE 10: platform and the limits set by the tests after the freeze
# =====================================================================================================================
s = new_slide("Platform and its limits",
    "After the freeze the detectors were placed, unchanged, inside a modular platform with eight modules: face video, image detector, speech, generated image, generated video, lip sync, media metadata and content credentials. A coordinator runs one module at a time with a time limit and isolates failures. Green modules are the frozen detectors, orange modules are experimental and never give a finding, grey modules describe the file or its credentials. There is no overall score or verdict. "
    "The table on the right is what the tests after the freeze allow each part to claim. FakeAVCeleb: the pre-stated criteria were met (visual 0.9552, audio 0.9263) but they are not evidence of detection quality, because the video codec coincides with the manipulation method, codec metadata alone separates real from fake video at 0.9985 on the development half, real and fake audio come from different recording channels, and at the frozen thresholds about half of the fake videos are missed (sensitivity 0.480): thresholds do not transfer. "
    "LAV-DF: a fused audio-visual score looked near perfect but its visual part is at chance within the produced videos (0.5032 on the sealed subset, exploratory), so no audio-visual detection is claimed. Generated video: at chance on DF26 (0.4957 [0.4755, 0.5168]), so experimental with no finding. Modern images: the probe misses 0.7395, and the retrained head flags 0.5213 of real images, so it is not installed. "
    "Software verification shows software behaviour, not accuracy: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests), measured on 2026-10-28. "
    "Evidence: E-015, E-021, E-023, E-014, E-019, E-027 to E-029; TESTING_RECORD; fix list items A5, A4, A7, A9.")
lead(s, "Eight modules and no overall score; the tests after the freeze limit what each may claim.")
pic, pw, ph = pic_fit(s, os.path.join(FIG, "fig_platform_v2.png"), L, Y0, span_w(6), 2550000,
                      "Platform schematic: uploaded file, validation, coordinator, eight modules in three sections, report with one entry per module, dashboard")
add_text(s, L, Y0 + ph + 100000, span_w(6), BOT - (Y0 + ph + 100000), [
    ("b", "Modules run one at a time; failures stay inside one module.", 16),
    ("b", "Experimental heads give no finding.", 16),
    ("b", "Software tests: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests).", 16),
], "text_platform")
wd = [1750000, span_w(6) - 1750000]
gt, th = add_table(s, col_x(6), Y0, wd, ["Check", "What it allows"], [
    ["FakeAVCeleb", "Criteria met (visual 0.9552, audio 0.9263), not evidence of detection quality: codec metadata alone gives 0.9985; about half of fake videos missed (sensitivity 0.480); thresholds do not transfer"],
    ["LAV-DF", "Visual part at chance within produced videos (0.5032, exploratory): no audio-visual claim"],
    ["Generated video", "At chance on DF26 (0.4957 [0.4755, 0.5168]): experimental, no finding"],
    ["Modern images", "Probe misses 0.7395; retrained head flags 0.5213 of real images: not installed"],
], size=16, bold_first_col=True, name="table_postfreeze", pad=20000, max_h=H0)

# =====================================================================================================================
# SLIDE 11: dashboard (screenshots of the redesigned website)
# =====================================================================================================================
s = new_slide("Dashboard",
    "This is the redesigned local dashboard, served on 127.0.0.1 only, with uploads deleted after each request. The left screenshot shows the result page for a face-swap test clip: the checks that ran are counted by outcome in a donut that is not a probability or a verdict, next to a summary of what the checks found. The right screenshot is the card of the face-manipulation check: the label reads Possible manipulation, with a high-uncertainty box, and a model-output bar marked not a probability with the decision line. "
    "Every module gives its own result, tested scope and limits. There is no overall score and no overall verdict, experimental modules never give a finding, and no score is presented as a probability. The flag carries high uncertainty because its false-alarm rate differed between test sets, see slide 13. "
    "In the live check on Edge with real detectors, 8 of 8 demonstration inputs behaved as specified, including two inputs that were correctly refused. "
    "The screenshots show no dataset media; the file name is that of a validation test clip. The website runs from deliverables_v2/website on port 8012 over the unchanged backend. "
    "Evidence: docs/TESTING_RECORD.md sections 10 and 12; deliverables_v2/website/README.md; CONTENT_MAP sections 4.3 and 15.")
from PIL import Image as _Im
sw = span_w(6)
_files = []
for k, pth in enumerate(DASHBOARD_SCREENSHOTS):
    if DASHBOARD_CROPS[k] is None:
        _files.append(pth)
    else:
        _cp = os.path.join(CHARTS, "dash_crop_%d.png" % k)
        _src = _Im.open(pth).convert("RGB")
        _parts = [_src.crop(b) for b in DASHBOARD_CROPS[k]]
        _canvas = _Im.new("RGB", (max(q.size[0] for q in _parts), sum(q.size[1] for q in _parts)), "white")
        _yy = 0
        for q in _parts:
            _canvas.paste(q, (0, _yy)); _yy += q.size[1]
        _canvas.save(_cp)
        _files.append(_cp)
SH_ = 2950000
for k, pth in enumerate(_files):
    pic, pw, ph = pic_fit(s, pth, col_x(6 * k), TOP, sw, SH_, DASHBOARD_SCREENSHOT_ALTS[k])
    pic.line.color.rgb = rgb(LINE); pic.line.width = Pt(0.75)
add_text(s, L, TOP + SH_ + 150000, W, BOT - (TOP + SH_ + 150000), [
    ("b", "Each check reports its own result, tested scope and limits; no overall score or verdict.", 20),
    ("b", "The face-video flag reads Possible manipulation, with high uncertainty; no score is a probability.", 20),
    ("b", "Experimental modules give no finding.", 20),
], "text_dashboard")

# =====================================================================================================================
# SLIDE 12: Gemini
# =====================================================================================================================
s = new_slide("Gemini-generated images",
    "No Gemini data, Gemini model or Gemini experiment exists in this project. What exists is a written feasibility assessment of data and provenance and a preregistration proposal that has not been run, so this is an unresolved research direction and not a capability. "
    "The nearest measured generator is Google Imagen 3, one of the 21 generators in the Community Forensics-Eval benchmark, with 203 generated and 688 paired real images. Imagen 3 is not Gemini, and Gemini image models are not one model. "
    "On Imagen 3 the frozen head has an AUC of 0.7691 and misses 0.8719 of generated images; the retrained head has an AUC of 0.8646 but flags 0.5654 of the paired real images. A file-format cue alone separates this group at an AUC of 0.9731, so these numbers are confounded by format. "
    "The study is blocked until the terms of use of Gemini outputs, the provenance of candidate sets, matched real images and the version coverage are resolved. "
    "Evidence: docs/GEMINI_FEASIBILITY.md; E-028, E-029, E-031 (proposed, not run); CONTENT_MAP section 12.")
lead(s, "No Gemini data, model or experiment exists; this is an open question, not a capability.")
add_card(s, col_x(0), Y0, span_w(5), 2000000, [
    ("h", "Status"),
    ("b", "Only a feasibility assessment and a proposal exist.", 16),
    ("b", "Nearest measured generator: Imagen 3, not Gemini.", 16),
    ("b", "Gemini image models are not one model.", 16),
], name="card_status")
add_text(s, col_x(5), Y0, span_w(7), 380000, [("h", "Imagen 3 in the benchmark (203 generated, 688 real)", 16)], "text_imagen")
wd = [3700000, span_w(7) - 3700000]
gt, th = add_table(s, col_x(5), Y0 + 400000, wd, ["Measure", "Value"], [
    ["Frozen head, AUC", "0.7691 [0.7352, 0.8020]"],
    ["Frozen head, generated missed", "0.8719"],
    ["Retrained head, AUC", "0.8646 [0.8360, 0.8917]"],
    ["Retrained head, real flagged", "0.5654"],
    ["File-format cue alone, AUC", "0.9731"],
], size=13, bold_first_col=False, name="table_imagen", pad=0)
add_card(s, L, Y0 + 2250000, W, H0 - 2250000, [
    ("h", "Unresolved before any study"),
    ("b", "Whether Google's terms allow training and testing a detector on Gemini outputs.", 18),
    ("b", "Provenance of candidate sets: prompts, dates and model build are not recorded.", 18),
    ("b", "Matched real images, and coverage of the different Gemini versions.", 18),
], name="card_unresolved")

# =====================================================================================================================
# SLIDE 13: published comparison (context only) and limitations
# =====================================================================================================================
s = new_slide("Comparison and limitations",
    "Published numbers are context. A computed difference is given only when task, data, protocol and metric match, and the only such row in this project is the re-run Xception baseline of test T3. "
    "On ASVspoof 2019 LA our speech equal error rate is 6.94 percent [5.94, 7.79], stored in the freeze report, and AASIST reports 1.13 percent as quoted by its authors; AASIST was not re-run, and whether it transfers better was not measured on a common protocol, so we state both numbers and no difference. "
    "On In-the-Wild ours is 15.78 percent [12.75, 18.70] on 42 of 54 speakers, against 33.9 and 7.2 percent published on the full set: a different test set, no ranking. "
    "On DF26 without DF26 training our head is at chance (0.4957 [0.4755, 0.5168]); the DF26 authors report 44.0 to 69.7 for FF++-trained detectors without intervals; training data and frame sampling differ, so no ordering is drawn. "
    "Limitations: thresholds do not transfer. The false-positive rates of the frozen video threshold are post hoc, a re-analysis of sealed scores already in the ledger, not confirmatory: 0.1098 on 744 real Celeb-DF family videos and 0.1405 on 363 real DFD videos; the speech threshold flags 0.1716 of real In-the-Wild clips. Dataset shortcuts inflate AUC (codec, production cue, file format). The development slices are small, T3 used one baseline family at 224 px, and edited images and social-media recompression were not tested. "
    "Evidence: PERFORMANCE_BENCHMARK; E-014, E-030, E-032; fix list items A6, A7, A8.")
lead(s, "Published numbers are context only; the limits are thresholds and dataset shortcuts.")
add_card(s, col_x(0), Y0, span_w(6), H0, [
    ("h", "Published numbers, context only"),
    ("b", "Only the re-run Xception (T3) shares task, data and protocol.", 18),
    ("b", "Speech EER, ASVspoof 2019 LA: ours 6.94% [5.94, 7.79]; AASIST 1.13% as quoted by its authors. AASIST was not re-run.", 18),
    ("b", "In-the-Wild: ours 15.78% [12.75, 18.70]; full-set published 33.9 and 7.2%. Different test set, no ranking.", 18),
    ("b", "DF26: ours at chance, 0.4957 [0.4755, 0.5168]; authors report 44.0 to 69.7 without intervals. Training data differ, no ordering.", 18),
], name="card_context_pub")
add_card(s, col_x(6), Y0, span_w(6), H0, [
    ("h", "Limitations"),
    ("b", "Thresholds do not transfer. Post hoc false-alarm rate (re-analysis of sealed scores, not confirmatory): 0.1098 on Celeb-DF family reals, 0.1405 on DFD.", 18),
    ("b", "Shortcuts inflate AUC: codec, production cue, file format.", 18),
    ("b", "Small development slices; T3 used one baseline family at 224 px.", 18),
    ("b", "Not tested: edited images, social-media recompression.", 18),
], name="card_limits")

# =====================================================================================================================
# SLIDE 14: conclusion and demo summary
# =====================================================================================================================
s = new_slide("Conclusion and demo",
    "We built one local system that checks images, video and speech with per-module findings and no overall score, and documented each step from a written hypothesis. "
    "On sealed tests, read once, the final video system improved on the first version on the Celeb-DF family (AUC 0.8218 against 0.6238, +0.1980 [+0.1688, +0.2242]). It is non-inferior to a re-run Xception baseline with a margin of 0.0100; the declared 95 percent superiority rule was met (+0.0176 [+0.0025, +0.0327]) but does not survive the Bonferroni correction, and the baseline ran at 224 px. "
    "The speech system improved on In-the-Wild (0.9148 against 0.7516, +0.1633 [+0.1177, +0.2266]); part of that gain removes a silence and level shortcut. "
    "The image probe misses 0.7395 of images from 21 recent generators. On LAV-DF the visual part of a fused score is at chance within produced videos (0.5032), so no audio-visual detection is claimed. On FakeAVCeleb the criteria were met but codec metadata alone gives 0.9985, so they are not evidence of detection quality, and thresholds do not transfer. "
    "For the demonstration, the dashboard runs locally and deletes uploads after analysis; in the live browser check 8 of 8 demonstration inputs behaved as specified, and the software tests give 713 passed, 2 skipped (opt-in GPU tests) and 13 deselected (data-dependent tests). "
    "Remind the audience to read every result as a possible sign within a tested scope, not a verdict. "
    "Evidence: E-014, E-015, E-028, TESTING_RECORD sections 10 and 12; CONTENT_MAP sections 9 and 15; D-091.")
add_card(s, col_x(0), TOP, span_w(8), BOT - TOP, [
    ("h", "Conclusion"),
    ("b", "Group-disjoint protocol; sealed sets read once; three tests declared first.", 18),
    ("b", "Video: Celeb-DF family AUC 0.8218 against 0.6238 (+0.1980 [+0.1688, +0.2242]). Non-inferior to a re-run Xception (margin 0.0100); the 95% superiority rule was met but does not survive Bonferroni; baseline at 224 px.", 18),
    ("b", "Speech: In-the-Wild 0.9148 against 0.7516 (+0.1633 [+0.1177, +0.2266]); part of the gain removes a shortcut.", 18),
    ("b", "Limits: the image probe misses 0.7395 of recent generators; no audio-visual claim (visual part at chance within produced videos, 0.5032); FakeAVCeleb criteria met but not evidence of detection quality (codec 0.9985); thresholds do not transfer.", 18),
], name="card_conclusion")
add_card(s, col_x(8), TOP, span_w(4), BOT - TOP, [
    ("h", "Demo summary"),
    ("b", "Local dashboard on 127.0.0.1; uploads deleted.", 18),
    ("b", "Live check: 8 of 8 demonstration inputs as specified.", 18),
    ("b", "Tests: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests).", 18),
    ("b", "Read each result as a possible sign within its tested scope.", 18),
], name="card_demo")

# =====================================================================================================================
# BACKUP divider
# =====================================================================================================================
BK = ["B1  Statistics and decision rules", "B2  Preprocessing and parity", "B3  Encoder shortlist and sweep", "B4  I-0 diagnosis",
      "B5  I-1b and I-1c", "B6  LAV-DF control and pulse check", "B7  Generated video, lip sync, modern generators",
      "B8  Platform detail", "B9  Licences, data balance, environment", "B10  Sealed AUC by test set"]
s = new_slide("Backup slides",
    "The backup slides hold the material behind the main slides, for questions. B1 gives the statistics and decision rules, B2 the preprocessing and the parity check, B3 the encoder shortlist and sweep, B4 the diagnosis of the working version, B5 the two video levers that did not enter, B6 the LAV-DF control and the pulse check, B7 the generated-video, lip-sync and modern-generator studies, B8 the platform detail, B9 the licences, the data balance and the environment, and B10 the sealed AUC charts. "
    "Every backup slide is exploratory unless it says that a result is one of the three declared tests. "
    "Evidence: deliverables_v2/audit/slides_audit.md section 5; PHASE_B_FIXLIST.md section B3.")
add_text(s, col_x(2), TOP + 200000, span_w(8), 4200000, [("b", t, 20) for t in BK], "text_backup_contents")

# =====================================================================================================================
# B1: statistics and decision rules
# =====================================================================================================================
s = new_slide("B1  Statistics and rules",
    "The headline metric is the ROC-AUC on a dataset never used in training. Intervals are 95 percent percentile intervals from 2,000 bootstrap resamples where the unit is the group, never the clip or frame, and comparisons are paired on the same resampled groups. Also reported: equal error rate, balanced accuracy at the validation threshold and expected calibration error with 15 bins. "
    "Decision rules: three head seeds for every candidate; an encoder is chosen by the tie set against the leader, then the cheapest wins; a difference is non-inferior when the lower end of the paired interval is above minus delta, with delta fixed beforehand at 0.0100 (twice the larger seed standard deviation of the baseline, 0.0003 on validation and 0.0020 on development, with a floor of 0.01; the floor decides). "
    "Superiority needs the whole interval above zero; for T3 this holds at 95 percent but not after Bonferroni. The sealed tests form a staircase: every frozen iteration is read once and recorded in a ledger of 74 entries with no repeat read. "
    "Evidence: docs/PROTOCOL.md; E-013, E-014; D-061, D-062, D-066.")
steps = ["**I-0**\nfrozen, read once", "**I-1**\nfrozen, read once", "**I-2, I-3**\nfrozen, read once", "**Final**\nfrozen, read once"]
bw4 = (W - 3 * gap) // 4
for k, tx in enumerate(steps):
    dbox(s, L + k * (bw4 + gap), TOP, bw4, 700000, tx, fill=F_GREY if k < 3 else F_RED, size=C.SZ_TABLE)
    if k < 3:
        darrow(s, L + k * (bw4 + gap) + bw4, TOP + 350000, L + (k + 1) * (bw4 + gap), TOP + 350000)
add_card(s, col_x(0), TOP + 900000, span_w(6), BOT - TOP - 900000, [
    ("h", "Metrics"),
    ("b", "Headline: ROC-AUC on a dataset never used in training, 95% group-bootstrap interval.", 18),
    ("b", "Also EER, balanced accuracy at the validation threshold, ECE with 15 bins.", 18),
    ("b", "2,000 resamples, percentile interval, groups resampled with replacement, paired.", 18),
], name="card_metrics")
add_card(s, col_x(6), TOP + 900000, span_w(6), BOT - TOP - 900000, [
    ("h", "Decision rules"),
    ("b", "Non-inferior: lower end of the paired interval above -0.0100 (set beforehand; the floor of 0.01 decides).", 18),
    ("b", "Superiority needs the whole interval above zero; for T3 it holds at 95% but not after Bonferroni.", 18),
    ("b", "Encoder: the tie set against the leader, then the cheapest wins.", 18),
    ("b", "Ledger of 74 entries, no repeat read.", 18),
], name="card_rules2")

# =====================================================================================================================
# B2: preprocessing and parity
# =====================================================================================================================
s = new_slide("B2  Preprocessing, parity",
    "One shared package prepares inputs for training, evaluation and inference, and a hash of its configuration is stored in every checkpoint and checked at load. "
    "A video gives 16 uniformly spaced frames; the YuNet detector finds the largest face, which is enlarged by 1.3 and cropped to 224 by 224 with a JPEG round trip at quality 95; frames without a face are left out and reported. An image is face-cropped if a face is found, otherwise processed whole by the encoder's own rule. Audio uses the first 64,600 samples at 16 kHz, repeat-padded. "
    "Scores computed from stored features and by the inference code from raw files agree within 0.0019 against a tolerance of 0.02; the wrapper parity on 11 non-sealed files is also within 0.0019. These checks show that the path works and say nothing about accuracy. "
    "Evidence: CONTENT_MAP section 5; E-004; TESTING_RECORD sections 9 and 10.")
wd = [1500000, W - 1500000]
gt, th = add_table(s, L, TOP, wd, ["Input", "Rule"], [
    ["Video", "16 uniformly spaced frames; YuNet face detector; largest face enlarged by 1.3; 224 by 224 crop, JPEG quality 95; frames without a face are left out and reported"],
    ["Image", "A face crop if a face is found, otherwise the whole image processed by the encoder's own rule"],
    ["Audio", "First 64,600 samples at 16 kHz, repeat-padded; silence trimmed and peak normalised (final system)"],
    ["Consistency", "One shared package for training, evaluation and inference; a configuration hash is stored in every checkpoint and checked at load"],
], size=16, bold_first_col=True, name="table_prep", pad=30000)
yc = TOP + th + 150000
add_card(s, L, yc, W, BOT - yc, [
    ("h", "Parity"),
    ("p", "Scores from stored features and from raw files agree within 0.0019 (tolerance 0.02). The check shows that the path works, not accuracy.", 18),
], name="card_parity")

# =====================================================================================================================
# B3: encoder shortlist and sweep
# =====================================================================================================================
s = new_slide("B3  Encoders and sweep",
    "Seven image encoders and four speech encoders were scored with features extracted once. The tie set holds every candidate whose paired 95 percent interval against the leader includes zero, and the cheapest member wins by end-to-end extraction time per 1,000 inputs, then parameters. A small network from scratch is a reference and excluded. DINOv3 was left out because its access needs a manual gate. "
    "On face video the best non-CNN candidate minus the best supervised CNN is +0.0994 [+0.0228, +0.1723], so frozen self-supervised and vision-language features beat supervised CNNs on this exploratory slice. The speech sweep is inconclusive, best lower bound 0.5338. All three slices are small: 15 identities, 2 generator units and 12 speakers. "
    "The cost table gives the end-to-end time per 1,000 inputs of the selected and neighbouring encoders from the cost record. "
    "Evidence: E-001 to E-003; CONTENT_MAP sections 5.4 and 8.1.")
wd = [1500000, 3500000, W - 7100000, 2100000]
gt, th = add_table(s, L, TOP, wd, ["Task", "Tie set with the leader", "Selected, development AUC", "Leader"], [
    ["Face video, 15 identities", "DINOv2 small and base, SigLIP base, both CLIP ViT-L/14", "DINOv2 small 0.6933 [0.6639, 0.7314]", "CLIP ViT-L/14 OpenAI 0.7464 [0.6721, 0.8306]"],
    ["Image, 2 generator units", "CLIP ViT-L/14 OpenAI only", "CLIP ViT-L/14 OpenAI 0.9935 [0.9803, 0.9982]", "same"],
    ["Speech, 12 speakers", "HuBERT base, WavLM base plus, Whisper small", "HuBERT base 0.6100 [0.5434, 0.7256]; inconclusive", "Whisper small 0.6380 [0.5338, 0.7700]"],
], size=14, bold_first_col=True, name="table_sweep", pad=20000)
y2 = TOP + th + 150000
wd2 = [3300000, 1900000, 2200000, W - 7400000]
gt2, th2 = add_table(s, L, y2, wd2, ["Encoder", "Parameters", "Seconds per 1,000 inputs", "Weights licence"], [
    ["DINOv2 small", "22.1 M", "0.94", "Apache-2.0"],
    ["SigLIP base", "92.9 M", "1.44", "Apache-2.0"],
    ["CLIP ViT-L/14 (OpenAI)", "304.0 M", "4.45", "research use (accepted)"],
    ["HuBERT base", "94.4 M", "2.04", "Apache-2.0"],
    ["WavLM base plus", "94.4 M", "2.53", "research use (accepted)"],
    ["Whisper small encoder", "88.2 M", "11.53", "Apache-2.0 / MIT"],
], size=14, bold_first_col=True, name="table_cost", pad=0)

# =====================================================================================================================
# B4: I-0 diagnosis
# =====================================================================================================================
s = new_slide("B4  I-0 diagnosis",
    "These are the observations of the diagnostic battery on the working version, all exploratory on the development slice, with no sealed set read. They decided which lever to try first; no lever was chosen in this table. "
    "For face frames the training AUC of 0.837 against a validation AUC of 0.799 shows bias first: the frozen feature separates even the training data only moderately. For speech the loss of 0.384 AUC from validation to development, with real In-the-Wild speech scored as fake, points to shift and to a possible recording-cue shortcut, and the calibration error rose after temperature scaling for both video and speech. "
    "Images showed no bias or variance, but shift to other generators cannot be measured with only 2 generator units. "
    "Evidence: E-004, E-007; CONTENT_MAP section 8.1.")
wd = [1300000, W - 1300000 - 3300000, 3300000]
gt, th = add_table(s, L, TOP, wd, ["Task", "Observations (battery)", "Candidate diagnoses"], [
    ["Face frames", "Train 0.837, validation 0.799, development 0.693 (gap +0.105). Weakest on validation: NeuralTextures 0.697, FaceSwap 0.778. Development ECE 0.205 before and 0.222 after temperature scaling.",
     "Bias first: the frozen feature separates even the training data only moderately; then shift."],
    ["Images", "Train and validation 1.000, development 0.994 on 2 generator units. ECE 0.046 before and 0.005 after.",
     "No bias or variance visible; shift to other generators cannot be measured with 2 units."],
    ["Speech", "Validation 0.994, development 0.610 (gap +0.384). Real In-the-Wild speech scored as fake. ECE 0.329 before and 0.405 after.",
     "Shift is dominant; a missing signal or a shortcut is a candidate; calibration worsens under shift."],
], size=16, bold_first_col=True, name="table_diag", pad=60000)
caption(s, L, TOP + th + 80000, W, 450000, "Exploratory, development slice. Diagnoses only: no lever is chosen here. No sealed set was read for them.")

# =====================================================================================================================
# B5: I-1b and I-1c
# =====================================================================================================================
s = new_slide("B5  I-1b and I-1c",
    "Two further video levers were run on the team's request although their triggers were not met to the letter: the gap between validation and development was 0.0803 against about 0.10, and the score correlation was 0.89 to 0.91 against below 0.8. "
    "I-1b adds blur, down-up resizing and JPEG to the training stream, then mixup; nothing was searched. The development slice AUC rises but validation falls by more than the allowed 0.0079, so it is a trade-off and does not enter. "
    "I-1c takes the mean logit of three fine-tuned encoders; its interval on the slice includes zero, so it does not enter. The final system is unchanged and neither lever was tested on the sealed sets. "
    "Evidence: E-016, E-017; CONTENT_MAP section 8.2.")
wd = [2700000, 1900000, 1900000, 2000000, W - 8500000]
gt, th = add_table(s, L, TOP, wd, ["Face video, 3 seeds", "Validation AUC", "Development AUC", "Validation, paired with I-1", "Slice, paired with I-1"], [
    ["I-1, fine-tuned SigLIP base", "0.9899 [0.9809, 0.9967]", "0.9096 [0.8776, 0.9388]", "reference", "reference"],
    ["I-1b, augmentation and mixup", "0.9705 [0.9618, 0.9795]", "0.9315 [0.9055, 0.9548]", "-0.0194 [-0.0251, -0.0129]", "+0.0219 [+0.0090, +0.0382]"],
    ["I-1c, mean of three encoders", "0.9903 [0.9823, 0.9968]", "0.9192 [0.8948, 0.9457]", "+0.0004 [-0.0011, +0.0023]", "+0.0096 [-0.0020, +0.0231]"],
], size=14, bold_first_col=True, name="table_levers", pad=50000)
yc = TOP + th + 150000
add_card(s, col_x(0), yc, span_w(6), BOT - yc, [
    ("h", "I-1b: does not enter"),
    ("p", "Slice up, validation down by more than the allowed 0.0079: a trade-off.", 18),
], name="card_i1b")
add_card(s, col_x(6), yc, span_w(6), BOT - yc, [
    ("h", "I-1c: does not enter"),
    ("p", "The slice interval includes zero; no gain it can show.", 18),
], name="card_i1c")

# =====================================================================================================================
# B6: LAV-DF control and pulse check
# =====================================================================================================================
s = new_slide("B6  LAV-DF and pulse",
    "The frozen face head, trained on FaceForensics++, is below chance on LAV-DF validation (0.3838), so fusion was not tested with it. With heads trained on LAV-DF the fused score looked near perfect, but a control run afterwards, post hoc and labelled so, showed that the visual head separates the data set's produced videos from its originals (0.9912 against audio-only fakes whose frames are untouched) and is at chance within the produced videos (0.5070 on validation; 0.5032 on the sealed subset, exploratory). The audio head is the only one with evidence above chance inside produced videos (0.8574 on validation, 0.8480 sealed). So no audio-visual detection is claimed. "
    "The pulse check fails: 0.301 [0.260, 0.346] above the shuffled control against 0.6 required, on 432 real clips. The positive control recovers an injected 1.2 Hz sinusoid of relative green amplitude 0.008 in 0.873 of the clips, which shows the pipeline's sensitivity and not that a pulse exists in compressed video. "
    "Evidence: E-010, E-011, E-012, E-006, E-014, M-005; CONTENT_MAP sections 8.4 and 8.5.")
wd = [3000000, 1900000, 2100000, W - 7000000]
gt, th = add_table(s, L, TOP, wd, ["LAV-DF validation, exploratory", "Own label, all videos", "Produced videos only", "Originals vs audio-only fakes"], [
    ["Visual head, trained on LAV-DF", "0.7491 [0.7280, 0.7697]", "0.5070 [0.4771, 0.5361]", "0.9912 [0.9864, 0.9951]"],
    ["Audio head, trained on LAV-DF", "0.8604 [0.8435, 0.8769]", "0.8574 [0.8394, 0.8752]", "0.8711 [0.8474, 0.8920]"],
    ["Frozen face head, FF++", "0.3838 [0.3599, 0.4079]", "not applicable", "0.2876 [0.2568, 0.3174]"],
], size=13, bold_first_col=True, name="table_lavdf", pad=20000)
y2 = TOP + th + 130000
wd2 = [3000000, 2600000, W - 5600000]
gt2, th2 = add_table(s, L, y2, wd2, ["LAV-DF sealed subset, exploratory", "Fused label, 2,000 videos", "Produced videos, 1,500"], [
    ["Visual head, trained on LAV-DF", "0.9921 [0.9880, 0.9956]", "0.5032 [0.4746, 0.5320]"],
    ["Audio head, trained on LAV-DF", "n/a", "0.8480 [0.8281, 0.8670]"],
], size=13, bold_first_col=True, name="table_lavdf_sealed", pad=20000)
yc = y2 + th2 + 130000
add_card(s, L, yc, W, BOT - yc, [
    ("b", "No audio-visual detection is claimed: the visual part is at chance within produced videos.", 18),
    ("b", "Pulse check fails: 0.301 [0.260, 0.346] above the shuffled control, 0.6 required; positive control recovers 0.008 amplitude in 0.873 of clips.", 18),
], name="card_lavdf_pulse")

# =====================================================================================================================
# B7: generated video, lip sync, modern generators
# =====================================================================================================================
s = new_slide("B7  Video, lip sync, modern",
    "Generated video: a head chosen on DeepAction reaches 0.8281 there but is at chance on DF26 (0.4957 [0.4755, 0.5168]); the pre-registered source-confound check flags 338 of 440 unmodified real clips from other sources, so the head separates a video source and not generation. It is installed as experimental only. "
    "A head trained on DF26's open generators reached 0.7548 on the commercial generators, a second counted look at that closed subset; the installed head was at chance there, and the two used different real sets. The confound persists: 0.8023 [0.7625, 0.8368] of the 440 real clips are flagged. DF26 was first used zero-shot, one look; its open generators were later used for training. "
    "Lip sync: the band of -6 to +2 frames, fitted on 230 real clips and checked on 230 disjoint ones, flags 1.3 percent of real clips and 0 of 160 Wav2Lip clips, so it is weak evidence and not a detector. "
    "Evidence: E-018 to E-020, E-024 to E-027; CONTENT_MAP section 11; fix list item A7.")
wd = [3200000, 1900000, span_w(12) - 5100000 - 0]
wd = [2900000, 1900000, W - 4800000]
gt, th = add_table(s, L, TOP, wd, ["Generated-video head", "DeepAction development", "DF26 held out"], [
    ["Temporal geometry", "0.5073 [0.4608, 0.5542]", "0.5077 [0.4723, 0.5424]"],
    ["Mean embedding, logistic regression (installed)", "0.8281 [0.7908, 0.8662]", "0.4957 [0.4755, 0.5168]"],
    ["Seven pixel statistics (control)", "0.7613 [0.7182, 0.8033]", "0.5604 [0.5361, 0.5852]"],
], size=13, bold_first_col=True, name="table_genvideo", pad=20000)
y2 = TOP + th + 130000
wd2 = [4000000, 3300000, W - 7300000]
gt2, th2 = add_table(s, L, y2, wd2, ["Head trained on DF26 open generators", "Result [95% interval]", "Reading"], [
    ["AUC, open families held out", "0.7453 [0.7174, 0.7710]", "chosen head"],
    ["AUC, commercial generators", "0.7548 [0.7189, 0.7885]", "one counted look"],
    ["Real FF++ and LAV-DF clips flagged", "0.8023 [0.7625, 0.8368]", "confound persists"],
], size=13, bold_first_col=True, name="table_df26", pad=20000)
y3 = y2 + th2 + 130000
wd3 = [4000000, 3300000, W - 7300000]
gt3, th3 = add_table(s, L, y3, wd3, ["Lip-sync rule", "Real clips flagged", "Wav2Lip clips flagged"], [
    ["Tolerance of 2 frames, fresh clips", "15.0 and 13.3 percent", "85 to 93 percent"],
    ["Band of -6 to +2 frames, disjoint clips", "1.3 percent [0.4, 3.8]", "0 of 160"],
], size=13, bold_first_col=True, name="table_lipsync", pad=20000)
y4 = y3 + th3 + 120000
add_card(s, L, y4, W, BOT - y4, [
    ("p", "A head trained on DF26's open generators reached 0.7548 on the commercial generators (one counted look); the installed head was at chance there; the two used different real sets.", 16),
], name="card_df26_note")

# =====================================================================================================================
# B8: platform detail
# =====================================================================================================================
s = new_slide("B8  Platform detail",
    "The report (schema 2.1) has three sections: authenticity, cross-modal and provenance. Findings are cautious phrases such as indicates manipulation, no indication within the tested scope, inconclusive, consistent, potential mismatch and credentials valid; none means genuine. The contract checker rejects any fused, overall or verdict field. "
    "Each completed module carries an assessment: a label of that module alone, its tier, the raw score with its kind, a calibration statement whose is_probability field is always false, an uncertainty level with its reason, and links to the stored results that justify the wording. The face-video flag reads Possible manipulation with high uncertainty because its false-alarm rate differed between test sets, see the results; the image and speech flags carry weak evidence. "
    "The Evaluation Lab re-analyses recorded score files with the project's metrics and group bootstrap; it runs no model, writes no ledger entry and never touches the approval table or any threshold. A sealed split is accepted only as a re-analysis of an evaluation already in the ledger, with a score file whose SHA-256 equals the ledger entry. A good lab result is not a product approval. "
    "Software verification: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests); the opt-in graphics-card tests gave 5 passed; the live browser check passed 8 of 8 demonstration inputs, and a known limitation is that the speech module has no speech check of its own. "
    "Evidence: D-069, D-076, D-082, D-087, D-088; TESTING_RECORD sections 9, 10, 12, 13.")
add_card(s, col_x(0), TOP, span_w(6), BOT - TOP, [
    ("h", "Report contract and verdicts"),
    ("b", "Sections: authenticity, cross-modal, provenance; each module appears exactly once.", 18),
    ("b", "Each module: label, tier, raw score with its kind, uncertainty, tested scope; is_probability is always false.", 18),
    ("b", "The contract checker rejects any overall, fused or verdict field.", 18),
    ("b", "The face-video flag carries high uncertainty because its false-alarm rate differed between test sets.", 18),
], name="card_contract")
add_card(s, col_x(6), TOP, span_w(6), BOT - TOP, [
    ("h", "Evaluation Lab and tests"),
    ("b", "Re-analyses recorded score files; runs no model, writes no ledger entry.", 18),
    ("b", "A sealed split only as a re-analysis of a ledger entry with a matching SHA-256.", 18),
    ("b", "Tests: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests).", 18),
    ("b", "Live browser check: 8 of 8 demonstration inputs behaved as specified.", 18),
], name="card_lab")

# =====================================================================================================================
# B9: licences, data balance, environment
# =====================================================================================================================
s = new_slide("B9  Licences, balance, setup",
    "All datasets are used under research-only terms. Celeb-DF, Celeb-DF++ and FakeAVCeleb forbid publishing derived data, and DF26 and Community Forensics forbid showing their media, so no frame, crop or per-file score of any dataset appears in this deck. The OpenAI CLIP and WavLM weights are used as research-only, and the SyncNet weights state no licence, which rules out redistribution of checkpoints. These licence questions block distribution, not the use in this project. "
    "Celeb-DF++ balance: a sample is excluded when its target or source identity is one of the 15 development identities (2,079 samples), and 150 byte-identical real videos are counted once. The pooled family test has 744 real and 6,906 fake videos in 344 groups (44 identities and 300 single YouTube videos); real videos are 9.7 percent of the pool, so balanced accuracy and the threshold carry more noise than the AUC. "
    "Environment as measured: RTX 5070 with 12 GB, Ryzen 9 7900X, Windows 10 Pro, Python 3.13.15 and PyTorch 2.14.1 with CUDA 13.0. "
    "Evidence: DATA_LOG; E-014; CONTENT_MAP sections 6 and 14.")
wd = [3000000, 4200000, W - 7200000]
gt, th = add_table(s, L, TOP, wd, ["Data or weights", "Terms as recorded", "Consequence"], [
    ["FaceForensics++", "Non-commercial research and education only", "Training; no frames shown"],
    ["Celeb-DF v2, Celeb-DF++", "Non-commercial research; no publication of derived data", "Sealed tests; no per-video score shown"],
    ["ASVspoof 2019 LA", "ODC-By (README)", "Audio training and validation"],
    ["OpenAI CLIP, WavLM weights", "No clean licence statement; conditional", "Research-only; stated as a limitation"],
], size=14, bold_first_col=True, name="table_licence", pad=15000)
y2 = TOP + th + 130000
wdb = [2300000, 800000, 800000, 2000000]
gtb, thb = add_table(s, L, y2, wdb, ["Family test", "Real", "Fake", "Groups"], [
    ["Celeb-DF v2", "744", "4,057", "344"],
    ["Celeb-DF++ new", "0", "2,849", "44 identities"],
    ["Pooled", "744", "6,906", "344"],
], size=14, bold_first_col=True, name="table_balance", pad=15000)
wde = [1000000, span_w(5) - 1000000]
gte, the = add_table(s, col_x(7), y2, wde, ["Item", "Measured value"], [
    ["GPU", "RTX 5070, 12 GB, driver 595.79"],
    ["CPU, RAM", "Ryzen 9 7900X, 31 GB"],
    ["Software", "Windows 10 Pro; Python 3.13.15; PyTorch 2.14.1, CUDA 13.0"],
], size=14, bold_first_col=True, name="table_env", pad=15000)

# =====================================================================================================================
# B10: sealed AUC by test set (charts)
# =====================================================================================================================
s = new_slide("B10  Sealed AUC by set",
    "These are all sealed AUC values for face video and speech, each set read once, with the 95 percent group-bootstrap interval. Only the Celeb-DF family and In-the-Wild rows feed a confirmatory test; every other row is sealed but exploratory. "
    "On the Celeb-DF family the final system is at 0.8218 and the re-run Xception baseline at 0.8042; on the Celeb-DF v2 part the final system is at 0.8999, on the Celeb-DF++ fakes against v2 reals the baseline is higher by point estimate (0.7351 against 0.7105), and on DFD too (0.8847 against 0.8797). The pooled family value lies between the two parts. "
    "For speech the sealed In-the-Wild AUC rises 0.7516, 0.8125, 0.8672, 0.9148 along the staircase as point estimates; adjacent intervals overlap. On ASVspoof 2019 LA the final system reaches 0.9835. "
    "Evidence: E-014; CONTENT_MAP section 9.")
pic, pw, ph = pic_fit(s, CH_VIDEO, L, TOP, span_w(6), 3550000, "Dot and interval chart of sealed AUC for the first version, the final system and a re-run Xception baseline on five face-video test sets")
caption(s, L, TOP + ph + 40000, span_w(6), 650000, "Face video, sealed, read once. Only the Celeb-DF family is confirmatory.")
pic, pw, ph = pic_fit(s, CH_AUDIO, col_x(6), TOP, span_w(6), 2300000, "Dot and interval chart of sealed AUC for the first version and the final system on ASVspoof 2019 LA and In-the-Wild")
caption(s, col_x(6), TOP + ph + 40000, span_w(6), 650000, "Speech, sealed, read once. In-the-Wild is confirmatory; ASVspoof is exploratory.")

# =====================================================================================================================
# document properties and save
# =====================================================================================================================
cp = prs.core_properties
cp.title = "Multimodal Deepfake Detection for Images, Video and Audio"
cp.author = "Shivansh Gupta, Vaibhav Shiroorkar, Adithya Jithesh"
cp.last_modified_by = "Shivansh Gupta, Vaibhav Shiroorkar, Adithya Jithesh"
cp.subject = "Final year project presentation"
cp.keywords = ""
cp.comments = ""
import datetime as _dt
cp.created = cp.modified = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None)
cp.revision = 1
prs.save(OUT)
import zipfile as _zf, shutil as _sh
_tmp = OUT + ".tmp"
with _zf.ZipFile(OUT) as _zin, _zf.ZipFile(_tmp, "w", _zf.ZIP_DEFLATED) as _zout:
    for _it in _zin.infolist():
        _data = _zin.read(_it.filename)
        if _it.filename == "docProps/app.xml":
            _data = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                     '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                     'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><TotalTime>0</TotalTime><Application>Microsoft Office PowerPoint</Application>'
                     '<Slides>%d</Slides></Properties>' % len(prs.slides)).encode("utf-8")
        _zout.writestr(_it, _data)
_sh.move(_tmp, OUT)
print("saved", OUT, len(prs.slides), "slides")
for w in WARN:
    print("WARN", w)
