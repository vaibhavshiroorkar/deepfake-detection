"""Shared geometry, type scale and text measurement for build_slides.py and check_slides.py.

All lengths are EMU (914400 per inch, 12700 per point). The numbers come from the reference deck
"Document Template/DeepFake_Detection_System_Presentation (2).pptx" (measured with python-pptx):
  slide 12,192,000 x 6,858,000 (16:9)
  title placeholder  x 737,846  y 90,000  w 10,716,308  h 700,000  (32 pt bold, centred, Times New Roman)
  header band        logos end at y 850,281 (left logo x 193,862 to 2,510,118; right logo from x 11,124,596)
  footer text        x 4,038,600  y 5,843,197  w 4,114,800  h 514,900  (centred)
  slide number       x 10,169,610 y 5,862,996  (12 pt)
  footer band        red bars from y 6,356,350 (small block from y 6,080,760 at x 0 to 838,200)
"""
from __future__ import annotations

from PIL import ImageFont

EMU_IN = 914400
SLIDE_W, SLIDE_H = 12192000, 6858000

# One grid. Left and right margins are those of the reference title placeholder.
L = 737846
W = 10716308
R = L + W
TOP = 950000          # below the header band (logos end at 850,281, title box ends at 790,000)
BOT = 5780000         # above the footer text (5,843,197)
H = BOT - TOP
TITLE_Y, TITLE_H = 90000, 700000
LOGO_L_END, LOGO_R_START = 2510118, 11124596   # title text must stay between the two logos
FOOTER_TOP = 5843197

# 12-column grid inside [L, R]
GUT = 150000
COLW = (W - 11 * GUT) / 12.0


def col_x(k: int) -> int:
    """Left edge of column k (0 to 11)."""
    return int(round(L + k * (COLW + GUT)))


def span_w(n: int) -> int:
    """Width of n columns including the gutters between them."""
    return int(round(n * COLW + (n - 1) * GUT))


# One type scale (points). Nothing on a slide is smaller than 12.
SZ_TITLE = 32        # slide titles (reference deck)
SZ_TITLE_MAIN = 28   # title slide
SZ_LEAD = 20         # block headings
SZ_BODY = 18         # body text
SZ_BODY_S = 16       # secondary body, equations, diagram row labels
SZ_TABLE = 14        # tables and diagram boxes
SZ_TABLE_S = 12      # dense tables, captions, references, footer
MIN_PT = 12

FONT = "Times New Roman"
RED = "A42530"
DARK = "262626"
GREY = "3F3F3F"
MID = "595959"
BAND = "F5F5F5"
LINE = "BFBFBF"

_fonts = {}


def _font(bold: bool, italic: bool):
    key = (bold, italic)
    if key not in _fonts:
        name = {(False, False): "times.ttf", (True, False): "timesbd.ttf", (False, True): "timesi.ttf", (True, True): "timesbi.ttf"}[key]
        _fonts[key] = ImageFont.truetype("C:/Windows/Fonts/" + name, 200)
    return _fonts[key]


def text_width_pt(text: str, size: float, bold=False, italic=False) -> float:
    return _font(bold, italic).getlength(text) * size / 200.0


def lines_for_runs(runs, size: float, width_emu: float, indent_emu: float = 0) -> int:
    """Greedy word wrap of a paragraph given as [(text, bold, italic), ...]; returns the number of lines."""
    width_pt = (width_emu - indent_emu) / 12700.0
    words = []   # (word, bold, italic, hard_break_before)
    for text, b, it in runs:
        parts = text.split(chr(10))
        for pi, part in enumerate(parts):
            for wi, w in enumerate(part.split(" ")):
                if w == "" and wi > 0:
                    continue
                words.append((w, b, it, pi > 0 and wi == 0))
    lines, cur = 1, 0.0
    for w, b, it, brk in words:
        ww = text_width_pt(w, size, b, it)
        sp = text_width_pt(" ", size, b, it)
        if brk:
            lines += 1
            cur = ww
            continue
        if cur == 0:
            cur = ww
        elif cur + sp + ww <= width_pt:
            cur += sp + ww
        else:
            lines += 1
            cur = ww
    return lines


def wrap_lines(text: str, size: float, width_emu: float, bold=False, italic=False, indent_emu: float = 0) -> int:
    return lines_for_runs([(text, bold, italic)], size, width_emu, indent_emu)


LINE_FACTOR = 1.17   # line pitch of Times New Roman relative to the font size


def para_height_emu(text: str, size: float, width_emu: float, bold=False, italic=False, indent_emu: float = 0,
                    space_before_pt: float = 0, space_after_pt: float = 0) -> float:
    n = wrap_lines(text, size, width_emu, bold, italic, indent_emu)
    return (n * size * LINE_FACTOR + space_before_pt + space_after_pt) * 12700.0


def estimate_text_height(shape, width_emu=None) -> float:
    """Estimated height (EMU) the text of a pptx shape needs, from its runs, insets, bullet indent and spacing."""
    from pptx.oxml.ns import qn
    tf = shape.text_frame
    bp = tf._txBody.find(qn("a:bodyPr"))
    li = int(bp.get("lIns", 91440)); ri = int(bp.get("rIns", 91440))
    ti = int(bp.get("tIns", 45720)); bi = int(bp.get("bIns", 45720))
    w = (width_emu if width_emu is not None else shape.width) - li - ri
    total = ti + bi
    for p in tf.paragraphs:
        ppr = p._p.find(qn("a:pPr"))
        marl = int(ppr.get("marL", 0)) if ppr is not None else 0
        sb = sa = 0.0
        if ppr is not None:
            e = ppr.find(qn("a:spcBef"))
            if e is not None and e.find(qn("a:spcPts")) is not None:
                sb = int(e.find(qn("a:spcPts")).get("val")) / 100.0
            e = ppr.find(qn("a:spcAft"))
            if e is not None and e.find(qn("a:spcPts")) is not None:
                sa = int(e.find(qn("a:spcPts")).get("val")) / 100.0
        runs, size = [], 18.0
        for r in p.runs:
            if r.font.size is not None:
                size = r.font.size.pt
            runs.append((r.text, bool(r.font.bold), bool(r.font.italic)))
        if not runs:
            ep = p._p.find(qn("a:endParaRPr"))
            if ep is not None and ep.get("sz"):
                size = int(ep.get("sz")) / 100.0
            runs = [("", False, False)]
        n = lines_for_runs(runs, size, w, marl)
        total += (n * size * LINE_FACTOR + sb + sa) * 12700.0
    return total
