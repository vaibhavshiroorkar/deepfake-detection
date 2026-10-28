"""Consistency check of the three v2 documents (paper, report, deck) against the Phase B rules.

    .venv/Scripts/python.exe deliverables_v2/check_v2.py [--write deliverables_v2/check_v2_output.txt]

Read-only. Extracts the text of deliverables_v2/research_paper/main.pdf, the report PDF and DOCX XML, and the deck (slide text AND speaker notes),
then (1) greps for the terms of deliverables_v2/PHASE_B_FIXLIST.md section C and prints every hit with its sentence, (2) checks that every
"superior"/"superiority" sentence carries the D-091 qualifier, (3) checks authors, title, no subtitle, test counts, (4) checks that every
decimal with three or more places appears in the project's record (docs/EXPERIMENTS.md, DECISIONS.md, PERFORMANCE_BENCHMARK.md, NUMBERS*.json,
RESEARCH.md, DATA_LOG.md, GEMINI_FEASIBILITY.md, TESTING_RECORD.md); percent values are tried as fractions. It proves that a number is IN the
record, not that it sits next to the same quantity.
"""
import glob
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "deliverables_v2"
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


def pdf_text(p):
    return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def docx_text(p):
    z = zipfile.ZipFile(p)
    parts = [n for n in z.namelist() if n.startswith("word/") and n.endswith(".xml")]
    return "\n".join(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf8", "replace")) for n in parts)


def pptx_text(p):
    from pptx import Presentation
    prs = Presentation(str(p))
    main, notes = [], []
    for i, s in enumerate(prs.slides, 1):
        t = []
        for sh in s.shapes:
            if sh.has_text_frame:
                t.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                for r in sh.table.rows:
                    t.append(" | ".join(c.text for c in r.cells))
        main.append((i, "\n".join(t)))
        if s.has_notes_slide:
            notes.append((i, s.notes_slide.notes_text_frame.text))
    core = prs.core_properties
    return main, notes, f"{core.title} || {core.author} || subject={core.subject!r} keywords={core.keywords!r}"


docs = {}
paper_pdf = V2 / "research_paper" / "main.pdf"
rep_pdf = next(iter(glob.glob(str(V2 / "project_report" / "*.pdf"))), None)
rep_docx = next(iter(glob.glob(str(V2 / "project_report" / "*.docx"))), None)
deck = next(iter(glob.glob(str(V2 / "presentation" / "*.pptx"))), None)
tex = (V2 / "research_paper" / "main.tex").read_text(encoding="utf-8", errors="replace") if (V2 / "research_paper" / "main.tex").exists() else ""
if paper_pdf.exists():
    docs["paper"] = pdf_text(paper_pdf)
if rep_pdf:
    docs["report"] = pdf_text(rep_pdf)
if rep_docx:
    docs["report_docx_xml"] = docx_text(rep_docx)
slide_main, slide_notes, deck_core = ([], [], "")
if deck:
    slide_main, slide_notes, deck_core = pptx_text(deck)
    docs["deck"] = "\n".join(t for _, t in slide_main)
    docs["deck_notes"] = "\n".join(t for _, t in slide_notes)

say("== files: paper=%s report=%s docx=%s deck=%s" % (paper_pdf.exists(), bool(rep_pdf), bool(rep_docx), bool(deck)))
if deck:
    say("== deck: %d slides; core: %s" % (len(slide_main), deck_core))

TERMS = ["superior", "superiority", "robust", "confirmed", "ranks well", r"0\.99(?!\d)", r"\b497\b", r"\b304\b", r"\b710\b", "calibrated probability",
         "Likely", "state of the art", "real-time", "Hypothesis-Driven", "1.732", "0.999983", "—", "–", "622423"]


def sentences_with(text, pat):
    flat = re.sub(r"\s+", " ", text)
    for m in re.finditer(pat, flat, flags=re.I):
        a = max(0, flat.rfind(". ", 0, m.start()) + 2)
        b = flat.find(". ", m.end())
        b = len(flat) if b < 0 else b + 1
        yield flat[a:b].strip()


say("\n== (1) term hits per document (count; sentences listed for the terms that matter)")
for name, text in docs.items():
    say("-- %s" % name)
    for t in TERMS:
        hits = list(sentences_with(text, t))
        if hits:
            say("   %-26s %3d" % (t, len(hits)))
            if t in ("superior", "superiority", "robust", "confirmed", "ranks well", r"0\.99(?!\d)", r"\b497\b", r"\b304\b", r"\b710\b", "Likely", "calibrated probability",
                     "1.732", "0.999983", "Hypothesis-Driven"):
                for h in hits[:12]:
                    say("      > " + h[:260])

say("\n== (2) every sentence with superior/superiority must carry the D-091 qualifier")
QUAL = re.compile(r"bonferroni|98\.33|not claimed|does not survive|not after|no claim|not robust|rule (was )?met|non-inferior|non-inferiority", re.I)
bad = 0
for name in ("paper", "report", "deck", "deck_notes"):
    if name not in docs:
        continue
    for s in sentences_with(docs[name], r"superior"):
        if not QUAL.search(s):
            bad += 1
            say("   NO QUALIFIER in %s: %s" % (name, s[:240]))
say("   sentences without qualifier: %d" % bad)

say("\n== (3) identity, title, test counts")
for name in ("paper", "report", "deck"):
    if name not in docs:
        continue
    t = docs[name]
    i = [t.find(n) for n in ("Shivansh Gupta", "Vaibhav Shiroorkar", "Adithya Jithesh")]
    say("   %-7s author order first positions %s -> %s" % (name, i, "OK" if all(x >= 0 for x in i) and i == sorted(i) else "CHECK"))
    say("   %-7s 713 present: %s; 'Group-Disjoint Protocol': %s; 'Aditya ' (wrong spelling): %s" % (
        name, "713 passed" in re.sub(r"\s+", " ", t) or "713" in t, "Group-Disjoint Protocol" in t, bool(re.search(r"\bAditya\b", t))))

say("\n== (4) decimals with 3 or more places must be in the record")
corpus = ""
for f in ("EXPERIMENTS.md", "DECISIONS.md", "PERFORMANCE_BENCHMARK.md", "NUMBERS.json", "NUMBERS_DERIVED.json", "NUMBERS_LIT.json", "RESEARCH.md",
          "DATA_LOG.md", "GEMINI_FEASIBILITY.md", "TESTING_RECORD.md", "DOC_FACTS.md", "VIVA_PREP.md"):
    p = ROOT / "docs" / f
    if p.exists():
        corpus += p.read_text(encoding="utf-8", errors="replace") + "\n"
corpus += "\n".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in glob.glob(str(ROOT / "docs" / "test_evidence" / "2026-10-28_phase_b" / "*.txt")))
corpus_nospace = corpus.replace("−", "-")
NUM = re.compile(r"(?<![\d.])(\d+\.\d{3,})(?!\d)")
for name in ("paper", "report", "deck", "deck_notes"):
    if name not in docs:
        continue
    text = docs[name].replace("−", "-")
    seen, miss = set(), []
    for m in NUM.finditer(text):
        tok = m.group(1)
        if tok in seen:
            continue
        seen.add(tok)
        if tok in corpus_nospace:
            continue
        try:
            frac = "%.*f" % (len(tok.split(".")[1]) + 2, float(tok) / 100)
        except ValueError:
            frac = None
        if frac and frac in corpus_nospace:
            continue
        miss.append(tok)
    say("   %-11s distinct 3+ decimal tokens %d; not found in the record: %d %s" % (name, len(seen), len(miss), miss[:40]))

if "--write" in sys.argv:
    dest = Path(sys.argv[sys.argv.index("--write") + 1])
    dest.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("written", dest)
