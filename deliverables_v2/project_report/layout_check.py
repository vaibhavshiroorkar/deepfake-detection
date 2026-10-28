"""Acceptance test of the report layout against the supplied Word template.

Both documents are rendered by Word to PDF and rasterised at the same resolution (pdftoppm, 150 dpi). The pages that
correspond are compared feature by feature, in millimetres, with a tolerance of 2 mm:

  chapter page   template page 2 (the chapter page of the template) against the first page of Chapter 1 of the report:
                 page border, chapter title, chapter box, first text line, text-block left and right edge, footer rule
  front page     template page 3 (a plain justified text page with the footer) against the certificate page of the report:
                 page border, first text line, text-block edges, footer rule

run() returns a list of tuples (item, template_mm, report_mm, delta_mm, ok); running the file prints the table.
Needs: Microsoft Word (for the template PDF, made once), pdftoppm and pdftotext on PATH, PIL and numpy.
"""
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))       # deliverables_v2/project_report (copy of report/layout_check.py)
ROOT = os.path.dirname(os.path.dirname(HERE))             # repository root (the template is only read)
RENDER = os.path.join(HERE, "_render")
TEMPLATE_DOC = os.path.join(ROOT, "Document Template", "67F_B Tech Semester VII Project Report template.doc")
TEMPLATE_PDF = os.path.join(RENDER, "template.pdf")
REPORT_PDF = os.path.join(HERE, "Multimodal_Deepfake_Detection_Project_Report.pdf")
DPI = 150
TOL = 2.0
MM = 25.4 / DPI


def _template_pdf():
    os.makedirs(RENDER, exist_ok=True)
    if not os.path.isfile(TEMPLATE_PDF) or os.path.getmtime(TEMPLATE_PDF) < os.path.getmtime(TEMPLATE_DOC):
        ps = os.path.join(HERE, "word_tools.ps1")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps, "-Mode", "pdf", "-In", TEMPLATE_DOC, "-Pdf", TEMPLATE_PDF],
                       capture_output=True, text=True, check=True)
    return TEMPLATE_PDF


def _page_text(pdf, n):
    r = subprocess.run(["pdftotext", "-f", str(n), "-l", str(n), "-layout", pdf, "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return r.stdout


def _npages(pdf):
    r = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
    return int(re.search(r"Pages:\s+(\d+)", r.stdout).group(1))


def _find_report_pages(pdf):
    chapter = front = None
    for n in range(1, _npages(pdf) + 1):
        lines = [l.strip() for l in _page_text(pdf, n).splitlines() if l.strip()]
        if not lines:
            continue
        if front is None and lines[0] == "Somaiya Vidyavihar University" and any(l == "Certificate" for l in lines[:4]):
            front = n
        if chapter is None and lines[0] == "Chapter 1" and any("Introduction" in l for l in lines[:2]) and not any("....." in l for l in lines[:4]):
            chapter = n
    return chapter, front


def _raster(pdf, n):
    stem = os.path.join(RENDER, "lc_%s_%d" % (os.path.splitext(os.path.basename(pdf))[0], n))
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(n), "-l", str(n), "-png", "-singlefile", pdf, stem], check=True)
    return np.array(Image.open(stem + ".png").convert("RGB")).astype(int)


def _runs(mask_line):
    """Longest contiguous True run of a 1-D boolean array: (length, start, end)."""
    best = (0, 0, 0)
    start = None
    for i, v in enumerate(list(mask_line) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i - 1)
            start = None
    return best


def features(img, kind):
    h, w, _ = img.shape
    dark = img.sum(axis=2) < 300          # black-ish ink
    # footer rule colour: brown 622423 in the template, maroon B7202F in the v2 report; both pass this mask
    red = (img[:, :, 0] > 70) & (img[:, :, 0] < 215) & (img[:, :, 1] < 90) & (img[:, :, 2] < 95) & (img[:, :, 0] - img[:, :, 1] > 40)
    f = {}
    # page border: the long horizontal and vertical black lines
    rows = [(y, _runs(dark[y, :])) for y in range(h)]
    long_rows = [y for y, r in rows if r[0] > 0.75 * w]
    cols = [(x, _runs(dark[:, x])) for x in range(w)]
    long_cols = [x for x, r in cols if r[0] > 0.75 * h]
    f["border top"] = min(long_rows) * MM
    f["border bottom"] = max(long_rows) * MM
    f["border left"] = min(long_cols) * MM
    f["border right"] = max(long_cols) * MM
    top, bottom, left, right = min(long_rows), max(long_rows), min(long_cols), max(long_cols)
    # v2: maroon ink (headings, chapter box, table borders) counts as ink like black does
    maroon = (img[:, :, 0] > 120) & (img[:, :, 1] < 150) & (img[:, :, 2] < 160) & (img[:, :, 0] - img[:, :, 1] > 50)
    inner = dark | maroon
    inner[: top + 4, :] = False
    inner[bottom - 3:, :] = False
    inner[:, : left + 4] = False
    inner[:, right - 3:] = False
    # footer rule: the coloured band (622423 template, B7202F v2) and its extent
    redrows = np.where(red[:, left + 6: right - 6].sum(axis=1) > 0.3 * (right - left))[0]
    redrows = redrows[redrows > h // 2]   # v2: maroon box and table rules sit higher; the footer rule is in the lower half
    fr_top = int(redrows.min())
    f["footer rule top"] = fr_top * MM
    xs = np.where(red[redrows.min(): redrows.max() + 1, :].any(axis=0))[0]
    f["footer rule left"] = xs.min() * MM
    f["footer rule right"] = xs.max() * MM
    body = inner.copy()
    body[fr_top - 2:, :] = False
    ink_rows = np.where(body.any(axis=1))[0]
    if kind == "chapter":
        f["chapter title top"] = ink_rows.min() * MM
        # centre of the first (chapter number) line
        first_end = ink_rows.min()
        while body[first_end].any() or body[first_end + 1].any():
            first_end += 1
        xs = np.where(body[ink_rows.min(): first_end + 1].any(axis=0))[0]
        f["chapter title centre"] = (xs.min() + xs.max()) / 2 * MM
        # chapter box: horizontal rules at least 140 mm long below the title
        box_rows = [y for y in ink_rows if y > first_end and _runs(body[y, :])[0] > 140 / MM]
        f["box top"] = min(box_rows) * MM
        f["box bottom"] = max(box_rows) * MM
        rr = _runs(body[min(box_rows), :])
        f["box left"] = rr[1] * MM
        f["box right"] = rr[2] * MM
        after = [y for y in ink_rows if y > max(box_rows) + 3]
        f["first text line top"] = min(after) * MM
        region = body[min(after):, :]
    else:
        f["first text line top"] = ink_rows.min() * MM
        region = body[ink_rows.min():, :]
    xs = np.where(region.any(axis=0))[0]
    f["text block left"] = xs.min() * MM
    f["text block right"] = xs.max() * MM
    return f


CHAPTER_ITEMS = ["border top", "border bottom", "border left", "border right", "chapter title top", "chapter title centre", "box top", "box bottom",
                 "box left", "box right", "first text line top", "text block left", "text block right", "footer rule top", "footer rule left", "footer rule right"]
FRONT_ITEMS = ["border top", "border bottom", "border left", "border right", "first text line top", "text block left", "text block right",
               "footer rule top", "footer rule left", "footer rule right"]


def run():
    tpl = _template_pdf()
    ch, fr = _find_report_pages(REPORT_PDF)
    if ch is None or fr is None:
        raise RuntimeError("report pages not found: chapter %s, front %s" % (ch, fr))
    out = []
    pairs = [("chapter page", 2, ch, "chapter", CHAPTER_ITEMS), ("front page", 3, fr, "front", FRONT_ITEMS)]
    for name, tp, rp, kind, items in pairs:
        ft = features(_raster(tpl, tp), kind)
        fr_ = features(_raster(REPORT_PDF, rp), kind)
        for it in items:
            a, b = ft[it], fr_[it]
            d = b - a
            out.append(("%s: %s" % (name, it), round(a, 1), round(b, 1), round(d, 1), abs(d) <= TOL))
    return out


if __name__ == "__main__":
    rows = run()
    w = max(len(r[0]) for r in rows)
    print("%s  %10s  %10s  %8s  %s" % ("item".ljust(w), "template", "report", "delta", "ok"))
    for it, a, b, d, ok in rows:
        print("%s  %10.1f  %10.1f  %8.1f  %s" % (it.ljust(w), a, b, d, "PASS" if ok else "FAIL"))
    bad = [r for r in rows if not r[4]]
    print("\n%d of %d rows within %.0f mm" % (len(rows) - len(bad), len(rows), TOL))
    sys.exit(1 if bad else 0)
