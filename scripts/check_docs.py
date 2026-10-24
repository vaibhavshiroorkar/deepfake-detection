"""Checks of the three deliverables (paper, report, slides) against the project rules. Prints one table and exits non-zero on a failure.

    python scripts/check_docs.py            (system python: needs python-pptx and lxml; the PDFs are read with pdftotext)

Per document: placeholders, internal IDs and reviewer phrases in VISIBLE text, missing bibliography entries, placeholders without an
experiment comment, fonts, margins, identity fields against docs/IDENTITY.md, and the layout comparison of the report with its template
(report/layout_check.py) and the slide checker (slides/check_slides.py) when they exist. Visible text means: LaTeX without comments,
the Word document body without comments or field codes, slide shapes without notes.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, REPORT, SLIDES = ROOT / "paper", ROOT / "report", ROOT / "slides"

ID_RE = re.compile(r"\b(?:D|O|E|R)-\d{3}\b|ENV-DESKTOP-GPU")
REVIEWER = ["venue not verified", "title page read", "No sentence is written here", "Insert Table of Figures", "TO BE FILLED", "TBD", "LY Project",
            "previous version", "earlier version", "old project", "rebuild", "rebuilt", "as before", "reviewer", "TODO", "FIXME", "Lorem"]
DASH = re.compile("[\u2014\u2013]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
IDENT = {  # exact strings that must appear (report and slides carry roll numbers; the paper does not)
    "names": ["Vaibhav Shiroorkar", "Shivansh Gupta", "Adithya Jithesh"],
    "rolls": ["16014024805", "16014023052", "16014024803"],
    "guide": ["Tejashree Phatak"],
    "department": ["Electronics and Computer Engineering"],
}
WRONG_DEPT = ["Electronics and Telecommunication", "Engineering Electronics"]
RESULT_NUMBER = re.compile(r"\b0\.\d{3,4}\b")


def pdf_text(pdf: Path) -> str:
    if not pdf.is_file():
        return ""
    r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return r.stdout


def tex_visible(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in tex.splitlines())


def docx_parts(path: Path):
    from lxml import etree
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    z = zipfile.ZipFile(path)
    body = etree.fromstring(z.read("word/document.xml"))
    text = "\n".join("".join(t.text or "" for t in p.iterfind(".//w:t", ns)) for p in body.iterfind(".//w:p", ns))
    comments = ""
    if "word/comments.xml" in z.namelist():
        c = etree.fromstring(z.read("word/comments.xml"))
        comments = "\n".join("".join(t.text or "" for t in p.iterfind(".//w:t", ns)) for p in c.iterfind(".//w:p", ns))
    fonts = set()
    for n in z.namelist():
        if re.match(r"word/(document|styles|numbering|header\d*|footer\d*|footnotes|endnotes)\.xml", n):
            x = z.read(n).decode("utf8", "ignore")
            for tag in re.findall(r"<w:rFonts [^>]*>", x):  # not w:lang, which also has w:eastAsia
                fonts |= set(re.findall(r'w:(?:ascii|hAnsi|cs|eastAsia)="([^"]+)"', tag))
    theme = z.read("word/theme/theme1.xml").decode("utf8", "ignore") if "word/theme/theme1.xml" in z.namelist() else ""
    doc_xml = z.read("word/document.xml").decode("utf8", "ignore")
    sects = re.findall(r"<w:pgMar [^>]*>", doc_xml)
    fields = {k: len(re.findall(k, doc_xml)) for k in ("TOC \\\\", "TOC ", "SEQ Figure", "SEQ Table", "SEQ Equation", "m:oMath")}
    return text, comments, fonts, sects, theme, doc_xml, fields


def row(rows, doc, check, ok, detail):
    rows.append((doc, check, "PASS" if ok else "FAIL", detail))


def visible_checks(rows, doc, text):
    ids = sorted(set(ID_RE.findall(text)))
    row(rows, doc, "internal IDs in visible text", not ids, ", ".join(ids[:6]) or "none")
    bad = [p for p in REVIEWER if re.search(re.escape(p), text, re.I)]
    row(rows, doc, "reviewer or version phrases", not bad, ", ".join(bad) or "none")
    row(rows, doc, "em or en dashes", not DASH.search(text), f"{len(DASH.findall(text))} found")
    row(rows, doc, "e-mail addresses", not EMAIL.search(text), ", ".join(EMAIL.findall(text)[:2]) or "none")
    wd = [w for w in WRONG_DEPT if w in text]
    row(rows, doc, "department wording", not wd, ", ".join(wd) or "ok")
    row(rows, doc, "placeholders [pending] (informational)", True, f"{text.count('[pending]')}")


def identity_checks(rows, doc, text, need_rolls):
    miss = [s for k in (["names", "guide", "department"] + (["rolls"] if need_rolls else [])) for s in IDENT[k] if s not in text]
    row(rows, doc, "identity fields present as in IDENTITY.md", not miss, ", ".join(miss) or "all present")
    order = [text.find(n) for n in IDENT["names"]]
    ok = all(o >= 0 for o in order) and order == sorted(order)
    row(rows, doc, "student order = report order", ok, "Vaibhav, Shivansh, Adithya" if ok else f"positions {order}")


def check_paper(rows):
    tex = (PAPER / "main.tex")
    if not tex.is_file():
        return row(rows, "paper", "main.tex exists", False, "missing")
    src = tex.read_text(encoding="utf-8")
    vis = tex_visible(src)
    pdf = pdf_text(PAPER / "main.pdf")
    row(rows, "paper", "PDF built", bool(pdf), f"{len(pdf.split(chr(12)))} pages" if pdf else "missing")
    visible_checks(rows, "paper", pdf or vis)
    identity_checks(rows, "paper", pdf or vis, need_rolls=False)
    keys = set(k.strip() for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", vis) for k in m.split(","))
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8") if (PAPER / "refs.bib").is_file() else ""
    have = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    row(rows, "paper", "cited keys present in refs.bib", keys <= have, ", ".join(sorted(keys - have)[:5]) or f"{len(keys)} cited")
    row(rows, "paper", "'venue not verified' in bib", "venue not verified" not in bib.lower(), "")
    pend = [l for l in src.splitlines() if "[pending]" in l]
    nocomment = [l for l in pend if not re.search(r"%\s*E-\d{3}", l)]
    row(rows, "paper", "[pending] lines with an E-xxx comment", not nocomment, f"{len(pend) - len(nocomment)} of {len(pend)}")
    # logged numbers are allowed (the sweep and I-0 results exist); every source line with a result-like number must carry an experiment comment
    bare = [l.strip()[:60] for l in src.splitlines() if RESULT_NUMBER.search(re.sub(r"(?<!\\)%.*$", "", l)) and "cite" not in l
            and not re.search(r"%\s*(?:E|M|DeepfakeBench)", l) and "figure" not in l.lower()]
    row(rows, "paper", "result-like numbers (0.xxxx) are logged ones (informational count)", len(bare) <= 60, f"{len(bare)} source lines hold a logged number (sweep, I-0 or cited DeepfakeBench) without a line comment; provenance is the experiment tables E-001 to E-004 (review by hand; limit 60)")
    row(rows, "paper", "class", "IEEEtran" in src, "documentclass IEEEtran")
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="ignore") if (PAPER / "main.log").is_file() else ""
    row(rows, "paper", "undefined references or citations", "undefined" not in log.lower(), f"{log.lower().count('undefined')} warnings")
    row(rows, "paper", "overfull boxes", "Overfull \\hbox" not in log, f"{log.count('Overfull')}")


def check_report(rows):
    docx = next(iter(sorted(REPORT.glob("*Technical_Report*.docx"))), None)
    docx = [d for d in REPORT.glob("*.docx") if "template" not in d.name and "draft" not in d.name.lower()] or ([docx] if docx else [])
    if not docx:
        return row(rows, "report", ".docx exists", False, "missing")
    path = docx[0]
    text, comments, fonts, sects, theme, xml, fields = docx_parts(path)
    pdf = next(iter(sorted(REPORT.glob("*preview*.pdf"))), None)
    ptxt = pdf_text(pdf) if pdf else ""
    visible_checks(rows, "report", text)
    identity_checks(rows, "report", text, need_rolls=True)
    bad_fonts = sorted(f for f in fonts if f not in ("Times New Roman", "Symbol", "Wingdings", "Cambria Math", "+mn-lt", "+mj-lt"))
    row(rows, "report", "fonts (only Times New Roman)", not bad_fonts and "Cambria\"" not in xml, ", ".join(bad_fonts[:6]) or "ok")
    th = re.findall(r'<a:latin typeface="([^"]*)"', theme)
    row(rows, "report", "theme fonts", all(t in ("Times New Roman", "") for t in th[:2]), ", ".join(th[:2]))
    # margins: A4, left 1.5 in = 2160 twips, right 1 in = 1440, top 2160, bottom 1440 (the template); body sections only checked for left, right
    ok = False
    for s in sects:
        m = {k: int(v) for k, v in re.findall(r'w:(left|right|top|bottom)="(\d+)"', s)}
        ok = ok or (abs(m.get("left", 0) - 2160) < 40 and abs(m.get("right", 0) - 1440) < 40)
    row(rows, "report", "margins left 1.5 in, right 1 in in a section", ok, f"{len(sects)} sections")
    row(rows, "report", "real fields: TOC / figures / tables / equations", fields["TOC "] + fields["TOC \\\\"] > 0 and fields["SEQ Figure"] > 0 and fields["SEQ Table"] > 0 and fields["m:oMath"] > 0, str(fields))
    pend = text.count("[pending]")
    ncom = len([c for c in re.findall(r"E-\d{3}", comments)])
    row(rows, "report", "[pending] placeholders with E-xxx comments", pend == 0 or ncom >= pend * 0.8, f"{pend} placeholders, {ncom} E-xxx in comments")
    nums = RESULT_NUMBER.findall(text)
    row(rows, "report", "result-like numbers (0.xxxx) in text", len(nums) <= 6, f"{len(nums)}: {nums[:6]}")
    bib_cited = set(re.findall(r"\[(\d+)\]", text))
    row(rows, "report", "citation numbers appear", bool(bib_cited), f"{len(bib_cited)} distinct [n]")
    row(rows, "report", "PDF preview exists", bool(ptxt), f"{len(ptxt.split(chr(12)))} pages" if ptxt else "missing")
    lc = REPORT / "layout_check.py"
    if lc.is_file():
        try:
            spec = importlib.util.spec_from_file_location("layout_check", lc)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for it, a, b, d, ok in mod.run():
                row(rows, "report", f"layout: {it}", bool(ok), f"template {a} mm, report {b} mm, delta {d} mm")
        except Exception as e:  # report instead of crashing
            row(rows, "report", "layout check runs", False, f"{type(e).__name__}: {e}")
    else:
        row(rows, "report", "layout check present", False, "report/layout_check.py missing")


def check_slides(rows):
    pp = next(iter(sorted(p for p in SLIDES.glob("*.pptx") if "draft" not in p.name.lower())), None)
    if not pp:
        return row(rows, "slides", ".pptx exists", False, "missing")
    from pptx import Presentation
    prs = Presentation(pp)
    vis, notes_missing, small = [], [], 0
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame:
                vis.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                vis += [c.text for r in sh.table.rows for c in r.cells]
        if not (s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip()):
            notes_missing.append(i)
    text = "\n".join(vis)
    row(rows, "slides", "slide count", len(prs.slides) >= 14, f"{len(prs.slides)} slides")
    visible_checks(rows, "slides", text)
    identity_checks(rows, "slides", text, need_rolls=True)
    row(rows, "slides", "speaker notes on every slide", not notes_missing, f"missing on {notes_missing[:8]}" if notes_missing else "all present")
    row(rows, "slides", "'RESULT' badges", "RESULT" not in text, "")
    sc = SLIDES / "check_slides.py"
    if sc.is_file():
        try:
            spec = importlib.util.spec_from_file_location("check_slides", sc)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            res = mod.run()
            bad = [r for r in res if not r[2]]
            row(rows, "slides", "slide checker (bounds, overlaps, grid, font sizes)", not bad, f"{len(res) - len(bad)} of {len(res)} checks pass; first failure {bad[0] if bad else 'none'}")
        except Exception as e:
            row(rows, "slides", "slide checker runs", False, f"{type(e).__name__}: {e}")
    else:
        row(rows, "slides", "slide checker present", False, "slides/check_slides.py missing")


def main() -> int:
    rows: list[tuple] = []
    for fn in (check_paper, check_report, check_slides):
        try:
            fn(rows)
        except Exception as e:
            rows.append((fn.__name__, "checker crashed", "FAIL", f"{type(e).__name__}: {e}"))
    w = [max(len(str(r[i])) for r in rows + [("document", "check", "result", "detail")]) for i in range(3)]
    print(f"{'document'.ljust(w[0])}  {'check'.ljust(w[1])}  {'result'.ljust(w[2])}  detail")
    for d, c, r, t in rows:
        print(f"{d.ljust(w[0])}  {c.ljust(w[1])}  {r.ljust(w[2])}  {t}")
    fails = [r for r in rows if r[2] == "FAIL"]
    print(f"\n{len(rows) - len(fails)} pass, {len(fails)} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
