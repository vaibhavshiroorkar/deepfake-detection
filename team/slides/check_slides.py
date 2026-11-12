"""Layout and content checker for slides/DeepFake_Detection_Presentation.pptx.

    python slides/check_slides.py            prints one table, exit code 1 if any check fails
    from check_slides import run             run() returns a list of (slide_or_item, description, ok) tuples

Checks (python-pptx, no rendering):
  bounds         every body shape lies inside the content area (grid margins L and R, below the header band, above the footer band)
  overlap        no two body shapes overlap (connectors, and a container without text that holds a shape, are allowed)
  grid           left edges lie on the 12-column grid (diagram parts and pictures: the leftmost one on a slide must sit on L)
  text fits      the estimated height of the text of every text shape and table is not larger than its frame
  font size      no run is smaller than 12 pt, tables and references at least 12 pt, body text at least 16 pt
  title          one size per slide type (32 pt; title slide 24 pt), text clear of the two logos
  notes          speaker notes on every slide, at least three sentences
  strings        no IDs, machine label, dashes, reviewer or version phrases, forbidden claims in visible text; no dashes in notes
  placeholders   every [pending] is in the one style (italic, dark red); no other placeholder wording
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slide_common as C  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu  # noqa: E402

PPTX = os.path.join(HERE, "DeepFake_Detection_Presentation.pptx")
TOL = 3000          # EMU, rounding tolerance
ID_RE = re.compile(r"\b(?:D|O|E|R)-\d{3}\b|ENV-DESKTOP-GPU")
FORBIDDEN = ["TBD", "TODO", "FIXME", "Lorem", "LY Project", "RESULT", "reviewer", "previous", "earlier version", "old project",
             "rebuild", "rebuilt", "as before", "venue not verified", "real-time", "state of the art", "state-of-the-art",
             "better than", "TO BE FILLED", "to be filled"]
DASH = re.compile("[—–]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
MIN_BODY = 16
MIN_ANY = 12


def _is_footer(sh) -> bool:
    if sh.is_placeholder:
        t = str(sh.placeholder_format.type)
        return "FOOTER" in t or "SLIDE_NUMBER" in t or "DATE" in t
    return False


def _is_title(sh) -> bool:
    return sh.is_placeholder and str(sh.placeholder_format.type).split(" ")[0] in ("TITLE", "CENTER_TITLE")


def _text_runs(sh):
    """Yield (text, size_pt or None, bold, italic, color) for every run in a shape or table."""
    frames = []
    if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
        frames.append(sh.text_frame)
    if getattr(sh, "has_table", False) and sh.has_table:
        for r in sh.table.rows:
            for c in r.cells:
                frames.append(c.text_frame)
    for tf in frames:
        for p in tf.paragraphs:
            for r in p.runs:
                col = None
                try:
                    col = str(r.font.color.rgb)
                except Exception:
                    pass
                yield r.text, (r.font.size.pt if r.font.size else None), r.font.bold, r.font.italic, col


def _text_of(sh) -> str:
    out = []
    if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
        out.append(sh.text_frame.text)
    if getattr(sh, "has_table", False) and sh.has_table:
        out += [c.text for r in sh.table.rows for c in r.cells]
    return "\n".join(out)


def _inside(a, b, tol=TOL):
    """Shape a (l, t, r, b) lies completely inside b."""
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def _box(sh):
    return (sh.left, sh.top, sh.left + sh.width, sh.top + sh.height)


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > TOL and min(a[3], b[3]) - max(a[1], b[1]) > TOL


def _col_lefts():
    return [C.col_x(k) for k in range(12)]


def run(path: str = PPTX):
    res = []
    add = lambda item, desc, ok: res.append((item, desc, bool(ok)))   # noqa: E731
    prs = Presentation(path)
    add("deck", "slide size 16:9, 12192000 x 6858000", (prs.slide_width, prs.slide_height) == (C.SLIDE_W, C.SLIDE_H))
    add("deck", "slide count at least 14", len(prs.slides) >= 14)
    area = (C.L, C.TOP, C.R, C.BOT)
    lefts = _col_lefts()
    all_visible = []
    for i, s in enumerate(prs.slides, 1):
        body = [sh for sh in s.shapes if not _is_footer(sh) and not _is_title(sh)]
        title = [sh for sh in s.shapes if _is_title(sh)]
        foot = [sh for sh in s.shapes if _is_footer(sh)]
        # ---- bounds
        bad = []
        for sh in body:
            bx = _box(sh)
            lim = area if i > 1 else (C.L, 0, C.R, C.BOT)
            if not _inside(bx, lim):
                bad.append(f"{sh.name} {tuple(round(v / 914400, 2) for v in bx)}")
        add(f"slide {i}", "body shapes inside the content area", not bad if True else True)
        if bad:
            res[-1] = (f"slide {i}", "body shapes inside the content area: " + "; ".join(bad[:3]), False)
        # ---- header/footer bands
        add(f"slide {i}", "no body shape in the footer band", all(sh.top + sh.height <= C.FOOTER_TOP for sh in body))
        for sh in title:
            if i > 1:
                ok = sh.left == C.L and sh.width == C.W and sh.top + sh.height <= C.TOP
                add(f"slide {i}", "title box at the reference position (x 737,846, w 10,716,308) and above the content area", ok)
            else:
                add(f"slide {i}", "title box above the content area", sh.top + sh.height <= C.TOP + 20000)
        # ---- overlaps
        ov = []
        items = [sh for sh in body if not sh.name.startswith("arrow")]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                bx, by = _box(x), _box(y)
                if not _overlap(bx, by):
                    continue
                tx, ty = _text_of(x).strip(), _text_of(y).strip()
                if _inside(by, bx) and not tx:
                    continue
                if _inside(bx, by) and not ty:
                    continue
                ov.append(f"{x.name} x {y.name}")
        add(f"slide {i}", "no overlapping shapes" + (": " + "; ".join(ov[:3]) if ov else ""), not ov)
        # title versus body
        add(f"slide {i}", "title does not overlap the body",
            all(not _overlap(_box(t), _box(b)) for t in title for b in body))
        # ---- grid
        leftmost_dg = None
        gbad = []
        for sh in body:
            if sh.name.startswith("arrow"):
                continue
            exempt = sh.name.startswith("dg_") or sh.name.startswith("pic") or sh.name.startswith("table_") and False
            if sh.name.startswith("dg_") or sh.name.startswith("pic"):
                leftmost_dg = sh.left if leftmost_dg is None else min(leftmost_dg, sh.left)
                continue
            if not any(abs(sh.left - lx) <= TOL for lx in lefts):
                gbad.append(f"{sh.name} x={sh.left}")
        add(f"slide {i}", "left edges on the 12-column grid" + (": " + "; ".join(gbad[:3]) if gbad else ""), not gbad)
        if leftmost_dg is not None:
            add(f"slide {i}", "leftmost diagram or picture starts at the left margin", abs(leftmost_dg - C.L) <= TOL)
        # ---- text fits
        fit_bad = []
        for sh in body:
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame and sh.text_frame.text.strip():
                need = C.estimate_text_height(sh)
                if need > sh.height * 1.02:
                    fit_bad.append(f"{sh.name} needs {need / 914400:.2f} in, has {sh.height / 914400:.2f} in")
            if getattr(sh, "has_table", False) and sh.has_table:
                tot = sum(r.height for r in sh.table.rows)
                if sh.top + tot > C.BOT + TOL:
                    fit_bad.append(f"{sh.name} reaches {(sh.top + tot) / 914400:.2f} in")
        add(f"slide {i}", "text fits its shape; tables end above the footer band" + (": " + "; ".join(fit_bad[:2]) if fit_bad else ""), not fit_bad)
        # ---- fonts
        small, nosize, body_small = [], [], []
        for sh in s.shapes:
            for text, size, b, it, col in _text_runs(sh):
                if not text.strip():
                    continue
                if size is None:
                    nosize.append(text[:20])
                elif size < MIN_ANY:
                    small.append(f"{size}:{text[:20]}")
        add(f"slide {i}", "every run has an explicit size of at least 12 pt" + (": " + "; ".join((small + nosize)[:3]) if (small or nosize) else ""), not small and not nosize)
        # body text (bullets and paragraphs, not tables, captions, diagram boxes) at least 16 pt
        for sh in body:
            if sh.name.startswith(("text_", "card_", "problem", "protocol", "rule", "maths", "data_text", "limits", "web_text",
                                   "concl_left", "rppg_text", "stats_", "backup_contents", "cost_text")):
                for text, size, *_ in _text_runs(sh):
                    if text.strip() and size is not None and size < MIN_BODY and not sh.name.startswith("pic_caption"):
                        body_small.append(f"{sh.name}:{size}")
        add(f"slide {i}", "body text at least 16 pt" + (": " + "; ".join(body_small[:3]) if body_small else ""), not body_small)
        # ---- title size and width
        for sh in title:
            sizes = {r[1] for r in _text_runs(sh)}
            want = {24.0} if i == 1 else {32.0}
            add(f"slide {i}", "title size as in the reference deck (32 pt; 24 pt on the title slide)", sizes == want)
            if i > 1:
                w = C.text_width_pt(sh.text_frame.text, 32, True) * 12700
                add(f"slide {i}", "title text clear of both logos", w <= 2 * (C.SLIDE_W / 2 - C.LOGO_L_END) - 100000)
        # ---- footer
        ftext = " ".join(_text_of(sh) for sh in foot)
        add(f"slide {i}", "footer text is the department line and the slide number is present",
            "Electronics and Computer Engineering" in ftext and str(i) in ftext)
        # ---- notes
        notes = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""
        nsent = len([x for x in re.split(r"(?<=[.!?])\s+", notes.strip()) if x])
        add(f"slide {i}", "speaker notes present (3 or more sentences)", nsent >= 3)
        add(f"slide {i}", "no dashes in the notes", not DASH.search(notes))
        # ---- strings in visible text
        vis = "\n".join(_text_of(sh) for sh in s.shapes)
        all_visible.append(vis)
        bad = [f for f in FORBIDDEN if re.search(re.escape(f), vis, re.I if f.islower() or f in ("reviewer", "previous") else 0)]
        if ID_RE.search(vis):
            bad.append("ID " + ID_RE.search(vis).group(0))
        if DASH.search(vis):
            bad.append("dash")
        if EMAIL.search(vis):
            bad.append("e-mail")
        add(f"slide {i}", "no IDs, dashes, reviewer or forbidden phrases in visible text" + (": " + ", ".join(bad) if bad else ""), not bad)
        # ---- placeholders
        pbad = []
        for sh in s.shapes:
            for text, size, b, it, col in _text_runs(sh):
                if "pending" in text and text != "[pending]" and "[pending]" not in text:
                    pbad.append(text[:20])
                if text == "[pending]" and not (it and col == "A42530"):
                    pbad.append("style")
        add(f"slide {i}", "[pending] only in the one style (italic, dark red)", not pbad)
        # ---- no text inside pictures claims etc: pictures have alt text
        for sh in s.shapes:
            if sh.shape_type == 13:
                add(f"slide {i}", f"picture {sh.name} has alt text", bool(sh._element.nvPicPr.cNvPr.get("descr")))
    text = "\n".join(all_visible)
    add("deck", "identity: names in the report order",
        all(n in text for n in ("Vaibhav Shiroorkar", "Shivansh Gupta", "Adithya Jithesh")) and
        text.find("Vaibhav Shiroorkar") < text.find("Shivansh Gupta") < text.find("Adithya Jithesh"))
    add("deck", "identity: roll numbers, guide and designation", all(x in text for x in ("16014024805", "16014023052", "16014024803", "Dr. Tejashree Phatak", "Assistant Professor")))
    add("deck", "identity: department wording", "Electronics and Computer Engineering" in text and "Telecommunication" not in text)
    add("deck", "placeholder count (informational)", True)
    res[-1] = ("deck", f"placeholder count (informational): {text.count('[pending]')}", True)
    return res


def main() -> int:
    res = run()
    w0 = max(len(str(r[0])) for r in res)
    w1 = max(len(r[1]) for r in res)
    for item, desc, ok in res:
        print(f"{str(item).ljust(w0)}  {'PASS' if ok else 'FAIL'}  {desc}")
    fails = [r for r in res if not r[2]]
    print(f"\n{len(res) - len(fails)} pass, {len(fails)} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
