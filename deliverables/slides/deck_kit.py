"""Drawing helpers for build_deck.py: text, cards, tables, diagram boxes, pictures and slide scaffolding.

Copied from the October builder (deliverables/presentation/build_deck_v2.py) so the new deck keeps the
same grid, type scale and look on the college master. The only change is that the helpers take the
presentation from `Deck` instead of a module global.
"""
from __future__ import annotations

import re

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

import slide_common as C
from slide_common import BOT, DARK, FONT, GREY, L, LINE, MID, RED, TOP, W

FOOTER_TEXT = "Electronics and Computer Engineering"
NSMAP = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
         "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
# Diagram fills, the same as the figures of the paper and the report.
F_BLUE, F_GREEN, F_ORANGE, F_RED, F_GREY, F_WHITE = "DDE7F4", "D8EBD8", "FBE8C8", "F6D9D7", "ECEFF3", "FFFFFF"
EDGE = "8496B0"
TOKEN = re.compile(r"(\*\*.+?\*\*|\[pending\])")
CELL_MX, CELL_MY = 70000, 35000
Y0 = TOP + 450000          # content area below the one-line message
H0 = BOT - Y0


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, template: str):
        self.prs = Presentation(template)
        self.layout = {lay.name: lay for lay in self.prs.slide_layouts}
        ids = self.prs.slides._sldIdLst
        for sld in list(ids):            # the template should have none; drop any that are there
            self.prs.part.drop_rel(sld.rId)
            ids.remove(sld)
        self.count = 0
        self.shape_n = 0
        self.warnings: list[str] = []

    # text --------------------------------------------------------------------------------------------
    def name(self, prefix):
        self.shape_n += 1
        return f"{prefix}_{self.shape_n}"

    def warn(self, text):
        self.warnings.append(f"slide {self.count}: {text}")

    def add_text(self, slide, x, y, w, h, paras, name="text", anchor=MSO_ANCHOR.TOP):
        tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
        tb.name = self.name(name)
        fill_text(tb.text_frame, paras, anchor)
        need = C.estimate_text_height(tb)
        if need > h:
            self.warn(f"{tb.name} needs {need/914400:.2f} in, has {h/914400:.2f} in")
        return tb

    def add_card(self, slide, x, y, w, h, paras, fill=C.BAND, name="card", anchor=MSO_ANCHOR.TOP, inset=110000):
        sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
        sh.name = self.name(name)
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
        sh.line.color.rgb = rgb(LINE); sh.line.width = Pt(0.75)
        sh.shadow.inherit = False
        fill_text(sh.text_frame, paras, anchor, inset)
        need = C.estimate_text_height(sh)
        if need > h:
            self.warn(f"{sh.name} needs {need/914400:.2f} in, has {h/914400:.2f} in")
        return sh

    def lead(self, slide, text):
        """The one message of a main slide, first line under the title."""
        return self.add_text(slide, L, TOP, W, 400000, [("p", "**" + text + "**", 18)], "text_lead")

    # tables ------------------------------------------------------------------------------------------
    def add_table(self, slide, x, y, widths, header, rows, size=C.SZ_TABLE, bold_first_col=False, name="table",
                  max_h=None, pad=0, fill=None):
        """size 'auto' picks the largest size that fits max_h. fill (EMU) spreads free height over the body
        rows, at most 160,000 EMU each, so a short table does not leave the slide half empty."""
        if size == "auto":
            for cand in (18, 17, 16, 15, 14, 13, 12):
                if sum(_rows_h(widths, header, rows, cand, bold_first_col)) + pad * len(rows) <= max_h:
                    size = cand
                    break
            else:
                size = 12
        if fill is not None:
            base = sum(_rows_h(widths, header, rows, size, bold_first_col))
            pad = max(pad, int(min(160000, max(0, (fill - base) / max(1, len(rows))))))
        gt = slide.shapes.add_table(len(rows) + 1, len(header), Emu(x), Emu(y), Emu(sum(widths)),
                                    Emu(400000 * (len(rows) + 1)))
        gt.name = self.name(name)
        tbl = gt.table
        tbl_pr = tbl._tbl.tblPr
        sid = tbl_pr.find(qn("a:tableStyleId"))
        if sid is None:
            sid = etree.SubElement(tbl_pr, qn("a:tableStyleId"))
        sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
        tbl_pr.set("firstRow", "0"); tbl_pr.set("bandRow", "0")
        for i, wd in enumerate(widths):
            tbl.columns[i].width = Emu(wd)
        hs = _rows_h(widths, header, rows, size, bold_first_col)
        total = 0
        for ri, row in enumerate([header] + rows):
            for ci, val in enumerate(row):
                cell = tbl.cell(ri, ci)
                cell.margin_left = cell.margin_right = Emu(CELL_MX)
                cell.margin_top = cell.margin_bottom = Emu(CELL_MY)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                for li, line in enumerate(val.split("\n")):
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
            self.warn(f"{gt.name} is {total/914400:.2f} in tall, limit {max_h/914400:.2f} in")
        return gt, total

    # diagrams and pictures ---------------------------------------------------------------------------
    def dbox(self, slide, x, y, w, h, text, fill=F_BLUE, size=C.SZ_TABLE, bold=False, align=PP_ALIGN.CENTER):
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
        sh.name = self.name("dg_box")
        sh.adjustments[0] = 0.08
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
        sh.line.color.rgb = rgb(EDGE); sh.line.width = Pt(1.25)
        sh.shadow.inherit = False
        tf = sh.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        bp = tf._txBody.find(qn("a:bodyPr"))
        for k in ("lIns", "rIns"):
            bp.set(k, "45000")
        for k in ("tIns", "bIns"):
            bp.set(k, "30000")
        for e in list(bp):
            bp.remove(e)
        etree.SubElement(bp, qn("a:noAutofit"))
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            add_runs(p, line, size, bold=bold)
            set_spacing(p, 0, 0)
        need = C.estimate_text_height(sh)
        if need > h:
            self.warn(f"{sh.name} '{text[:20]}' needs {need/914400:.2f} in, has {h/914400:.2f} in")
        return sh

    def darrow(self, slide, x1, y1, x2, y2):
        cx = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(x1), Emu(y1), Emu(x2), Emu(y2))
        cx.name = self.name("arrow")
        cx.line.color.rgb = rgb(DARK); cx.line.width = Pt(1.75)
        ln = cx.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "lg"); te.set("len", "lg")
        return cx

    def flow(self, slide, y, h, steps, fills, size=C.SZ_TABLE, gap=200000):
        """A row of boxes joined by arrows, spanning the content width."""
        n = len(steps)
        bw = (W - (n - 1) * gap) // n
        for k, tx in enumerate(steps):
            self.dbox(slide, L + k * (bw + gap), y, bw, h, tx, fill=fills[k], size=size)
            if k < n - 1:
                self.darrow(slide, L + k * (bw + gap) + bw, y + h // 2, L + (k + 1) * (bw + gap), y + h // 2)

    def add_pic(self, slide, path, x, y, w=None, h=None, alt=""):
        pic = slide.shapes.add_picture(str(path), Emu(x), Emu(y), Emu(w) if w else None, Emu(h) if h else None)
        pic.name = self.name("pic")
        pic._element.nvPicPr.cNvPr.set("descr", alt)
        return pic

    def pic_fit(self, slide, path, x, y, w, h, alt, align="center"):
        """Places a picture inside the box (x, y, w, h) at its own aspect ratio, never stretched."""
        from PIL import Image
        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        pw, ph = int(iw * sc), int(ih * sc)
        px = x if align == "left" else x + (w - pw) // 2
        return self.add_pic(slide, path, px, y, w=pw, h=ph, alt=alt), pw, ph

    # scaffolding -------------------------------------------------------------------------------------
    def _ph(self, slide, kind, idx, text_xml, name):
        ids = [int(e.get("id")) for e in slide.shapes._spTree.iter() if e.tag.endswith("}cNvPr")]
        nid = max(ids + [1]) + 1
        xml = (f'<p:sp xmlns:p="{NSMAP["p"]}" xmlns:a="{NSMAP["a"]}"><p:nvSpPr><p:cNvPr id="{nid}" name="{name}"/>'
               f'<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="{kind}" sz="quarter" idx="{idx}"/>'
               f'</p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p>{text_xml}</a:p></p:txBody></p:sp>')
        slide.shapes._spTree.append(etree.fromstring(xml))

    def add_footer(self, slide, number):
        rpr = (f'<a:rPr lang="en-US" sz="1200" b="0" i="0"><a:solidFill><a:srgbClr val="{GREY}"/></a:solidFill>'
               f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/><a:cs typeface="{FONT}"/></a:rPr>')
        self._ph(slide, "ftr", 11, f'<a:r>{rpr}<a:t>{FOOTER_TEXT}</a:t></a:r>', "Footer")
        self._ph(slide, "sldNum", 12, f'<a:fld id="{{00000000-1234-1234-1234-123412341234}}" type="slidenum">'
                 f'{rpr}<a:t>{number}</a:t></a:fld>', "Slide number")

    def set_title(self, shape, text, size=C.SZ_TITLE, y=C.TITLE_Y, h=C.TITLE_H, check=True):
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
        if check and C.text_width_pt(text, size, True) * 12700 > 2 * (C.SLIDE_W / 2 - C.LOGO_L_END) - 200000:
            self.warn(f"title '{text}' is too wide for the space between the logos")

    def new_slide(self, title, notes, layout="TITLE_ONLY"):
        self.count += 1
        s = self.prs.slides.add_slide(self.layout[layout])
        self.set_title(s.shapes.title, title)
        self.add_footer(s, self.count)
        if len(notes.split(". ")) < 3:
            self.warn("speaker notes need at least three sentences")
        s.notes_slide.notes_text_frame.text = notes
        return s


# module-level text helpers ---------------------------------------------------------------------------
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


def add_runs(p, text, size, bold=False, italic=False, color=DARK):
    """**x** is bold; [pending] always gets the one placeholder style (italic, dark red)."""
    for part in TOKEN.split(text):
        if not part:
            continue
        r = p.add_run()
        if part == "[pending]":
            r.text = part
            style_run(r, size, bold, True, RED)
        elif part.startswith("**"):
            r.text = part[2:-2]
            style_run(r, size, True, italic, color)
        else:
            r.text = part
            style_run(r, size, bold, italic, color)


def set_bullet(p, level=0):
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", str(257175 + level * 285750))
    ppr.set("indent", "-257175")
    for t in ("a:buNone", "a:buChar", "a:buFont", "a:buAutoNum"):
        for e in ppr.findall(qn(t)):
            ppr.remove(e)
    etree.SubElement(ppr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(ppr, qn("a:buChar")).set("char", "•" if level == 0 else "-")


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
    """paras: list of (kind, text[, size]). Kinds: h heading, b bullet, b2 sub bullet, p plain, s secondary,
    eq equation line, cap caption, n numbered line that carries its own number."""
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "rIns", "tIns", "bIns"):
        bp.set(k, str(inset))
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, qn("a:noAutofit"))
    for i, item in enumerate(paras):
        kind, text = item[0], item[1]
        size = item[2] if len(item) > 2 else {"h": C.SZ_LEAD, "b": C.SZ_BODY, "b2": C.SZ_BODY_S, "p": C.SZ_BODY,
                                              "s": C.SZ_BODY_S, "eq": C.SZ_BODY, "cap": C.SZ_TABLE_S,
                                              "n": C.SZ_BODY}[kind]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        if kind == "h":
            add_runs(p, text, size, bold=True, color=RED)
            set_spacing(p, 0 if i == 0 else 8, 4)
        elif kind in ("b", "b2"):
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
        elif kind == "eq":
            add_runs(p, text, size)
            set_spacing(p, 0, 10)
        else:
            add_runs(p, text, size)
            set_spacing(p, 0, 6)


def _cell_border(cell, color=LINE, width=6350):
    tc_pr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for e in tc_pr.findall(qn(tag)):
            tc_pr.remove(e)
    lines = []
    for tag in ("a:lnL", "a:lnR", "a:lnT"):
        ln = etree.Element(qn(tag)); ln.set("w", "0"); etree.SubElement(ln, qn("a:noFill")); lines.append(ln)
    lnb = etree.Element(qn("a:lnB")); lnb.set("w", str(width))
    sf = etree.SubElement(lnb, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    lines.append(lnb)
    rest = list(tc_pr)
    for k in rest:
        tc_pr.remove(k)
    for k in lines + rest:      # borders come before the fill inside tcPr
        tc_pr.append(k)


def _rows_h(widths, header, rows, size, bold_first_col):
    hs = []
    for ri, row in enumerate([header] + rows):
        hmax = 0
        for ci, val in enumerate(row):
            nl = sum(C.lines_for_runs([(ln, ri == 0 or (bold_first_col and ci == 0), False)], size,
                                      widths[ci] - 2 * CELL_MX) for ln in val.split("\n"))
            hmax = max(hmax, nl * size * C.LINE_FACTOR * 12700 + 2 * CELL_MY)
        hs.append(int(hmax + 12000))
    return hs
