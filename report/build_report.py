"""Builds the technical report skeleton (.docx) from the college template.

Base file: template_base_converted_from_doc.docx, which is the supplied Word
template "67F_B Tech Semester VII Project Report template.doc" converted to
.docx by Microsoft Word itself (SaveAs2, docx format). The original .doc is
not modified. This script removes the template's instruction text, keeps its
page setup features (A4, page border, footer rule colour 622423, Times New
Roman), applies the written formatting rules of the template, and writes the
report skeleton.

Run (from the repository root, any Python with python-docx):
    uv run --no-project --with python-docx python report/build_report.py

Rules in force (CLAUDE.md): no result is written unless it comes from a logged
experiment E-xxx; placeholders say "to be filled from E-xxx"; only papers
marked READ in docs/RESEARCH.md are cited; model, dataset, framework and
metric choices are open (docs/DECISIONS.md O-001 to O-011); no em-dashes or
en-dashes anywhere.
"""
import copy
import os
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(HERE, "template_base_converted_from_doc.docx")
OUT = os.path.join(HERE, "DeepFake_Detection_Technical_Report_draft.docx")
LOGO = os.path.join(ROOT, "assets", "branding", "somaiya_vidyavihar_logo.jpg")

# ---------------------------------------------------------------------------
# Identity details (from the supplied pptx and pdf, as stated by the user).
# Items marked CONFIRM are listed as questions in docs/_agents/A.md.
# ---------------------------------------------------------------------------
UNIVERSITY = "Somaiya Vidyavihar University"
SCHOOL = "K. J. Somaiya School of Engineering"
DEPARTMENT = "Department of Electronics and Computer Engineering"  # CONFIRM
DEGREE = "Bachelor of Technology"
BRANCH = "(Electronics and Computer Engineering)"  # CONFIRM
STUDENTS = [
    ("Vaibhav Shiroorkar", "16014024805"),
    ("Shivansh Gupta", "16014023052"),
    ("Adithya Jithesh", "16014024803"),
]
GUIDE = ("Dr. Tejashree Phatak", "Assistant Professor")
HOD = "Dr. Jagannath Nirmal"
ADDRESS = "Vidyavihar, Mumbai - 400 077"
YEAR = "2026-2027"  # CONFIRM
FOOTER_TEXT = "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch"  # CONFIRM
TITLE = "Multimodal Deepfake Detection for Images, Video and Audio"  # proposal, CONFIRM
SUBTITLE = "A Hypothesis-Driven Study Under a Source-Disjoint Protocol"  # proposal, CONFIRM
RED = RGBColor(0xA4, 0x25, 0x30)  # palette colour of the supplied slide master
FONT = "Times New Roman"

# ---------------------------------------------------------------------------
# References: only papers marked READ in docs/RESEARCH.md. Numbered in order
# of first citation.
# ---------------------------------------------------------------------------
REFS = {
    "dfdc": "B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang and C. Canton Ferrer, “The DeepFake Detection Challenge (DFDC) Dataset,” arXiv:2006.07397v4, 2020.",
    "bench": "Z. Yan, Y. Zhang, X. Yuan, S. Lyu and B. Wu, “DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.",
    "celeb": "Y. Li, X. Yang, P. Sun, H. Qi and S. Lyu, “Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics,” arXiv:1909.12962v4, 2020.",
    "xray": "L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen and B. Guo, “Face X-ray for More General Face Forgery Detection,” arXiv:1912.13458v2, 2020.",
    "ffpp": "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies and M. Nießner, “FaceForensics++: Learning to Detect Manipulated Facial Images,” arXiv:1901.08971v3, 2019.",
    "ojha": "U. Ojha, Y. Li and Y. J. Lee, “Towards Universal Fake Image Detectors that Generalize Across Generative Models,” arXiv:2302.10174v2, 2024.",
    "kornblith": "S. Kornblith, J. Shlens and Q. V. Le, “Do Better ImageNet Models Transfer Better?,” arXiv:1805.08974v3, 2019.",
    "fakecatcher": "U. A. Ciftci, I. Demir and L. Yin, “FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals,” arXiv:1901.02212v3, 2020.",
    "hearts": "U. A. Ciftci, I. Demir and L. Yin, “How Do the Hearts of Deep Fakes Beat? Deep Fake Source Detection via Interpreting Residuals with Biological Signals,” arXiv:2008.11363v1, 2020.",
    "onphys": "J. Hernandez-Ortega, R. Tolosana, J. Fierrez and A. Morales, “DeepFakesON-Phys: DeepFakes Detection based on Heart Rate Estimation,” arXiv:2010.00400v3, 2020.",
    "heart": "C. Seibold, E. L. Wisotzky, A. Beckmann, B. Kossack, A. Hilsmann and P. Eisert, “High-quality deepfakes have a heart!,” Frontiers in Imaging, vol. 4, 2025, doi: 10.3389/fimag.2025.1504551.",
    "harraq": "O. Harraq and T. Aldwairi, “Physiological Signals as a Forensic Modality for Talking-Face Deepfake Detection,” arXiv:2607.21776v1, 2026 (preprint).",
    "lin": "Y. Lin, E. Maiorana, B. Li and P. Campisi, “Detection of Photoplethysmography Manipulation in Video Forgery,” in Proc. European Signal Processing Conference (EUSIPCO), paper 615, 2023 (title page, abstract and introduction read).",
    "asvspoof": "X. Wang, J. Yamagishi, M. Todisco, H. Delgado, A. Nautsch, N. Evans et al., “ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech,” Computer Speech and Language, 2020, doi: 10.1016/j.csl.2020.101114.",
    "aasist": "J. Jung, H. Heo, H. Tak, H. Shim, J. S. Chung, B. Lee, H. Yu and N. Evans, “AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks,” arXiv:2110.01200v1, 2021.",
    "fakeav": "H. Khalid, S. Tariq, M. Kim and S. S. Woo, “FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset,” arXiv:2108.05080v4, 2022.",
    "dissonance": "K. Chugh, P. Gupta, A. Dhall and R. Subramanian, “Not made for each other: Audio-Visual Dissonance-based Deepfake Detection and Localization,” arXiv:2005.14405v3, 2021.",
    "calib": "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. International Conference on Machine Learning (ICML), PMLR 70, 2017.",
}
_order = []


def cite(*keys):
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in _order:
            _order.append(k)
        nums.append(_order.index(k) + 1)
    return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
doc = Document(BASE)
body = doc.element.body

# 1. Remove all template instruction content, keep the final sectPr.
final_sect = body.find(qn("w:sectPr"))
for child in list(body):
    if child is not final_sect:
        body.remove(child)


def set_font(run, size=12, bold=None, italic=None, color=None):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def make_style(name, size, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
               line=1.5, before=0, after=0, outline=None, color=None, keep_next=False):
    from docx.enum.style import WD_STYLE_TYPE
    st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    if color is not None:
        st.font.color.rgb = color
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    pf = st.paragraph_format
    pf.alignment = align
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep_next
    if outline is not None:
        ppr = st.element.get_or_add_pPr()
        ol = OxmlElement("w:outlineLvl")
        ol.set(qn("w:val"), str(outline))
        ppr.append(ol)
    return st


# Styles follow the written rules of the template: Times New Roman, 12 pt
# body at 1.5 spacing justified, 14 pt bold subtopics, chapter title bold.
make_style("Report Body", 12)
make_style("Report Chapter Title", 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, outline=0, keep_next=True)
make_style("Report Subtopic", 14, bold=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, outline=1, before=6, keep_next=True)
make_style("Report Back Heading", 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, outline=0, keep_next=True)
make_style("Report Front Heading", 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True, color=None)
make_style("Report Reference", 12, line=1.5)
make_style("Report Plain", 12, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0)
make_style("TOC 1", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15, before=4)
make_style("TOC 2", 12, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15)
doc.styles["TOC 2"].paragraph_format.left_indent = Inches(0.3)


def para(text="", style="Report Body", bold=None, italic=None, align=None, size=None, color=None,
         page_break_before=False, space_after=None, keep_next=False):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        sz = size if size else {"Report Body": 12, "Report Chapter Title": 16, "Report Subtopic": 14,
                                "Report Front Heading": 16, "Report Back Heading": 16,
                                "Report Reference": 12, "Report Plain": 12}[style]
        set_font(r, sz, bold, italic, color)
    if align is not None:
        p.alignment = align
    if page_break_before:
        p.paragraph_format.page_break_before = True
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    return p


def tbd(text):
    """Placeholder text for anything not yet decided or measured."""
    return "[TBD: " + text + "]"


def frm(text):
    """Placeholder for a number that must come from a logged experiment."""
    return "[to be filled from " + text + "]"


def add_field(paragraph, instr, size=12, bold=False):
    def r_with(child):
        r = paragraph.add_run()
        set_font(r, size, bold)
        r._element.append(child)
        return r
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
    r_with(b)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " " + instr + " "
    r_with(it)
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate")
    r_with(s)
    t = paragraph.add_run("1"); set_font(t, size, bold)
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    r_with(e)


def chapter(num, title, summary, first=False):
    p = doc.add_paragraph(style="Report Chapter Title")
    p.paragraph_format.page_break_before = True
    r = p.add_run("Chapter %d" % num)
    set_font(r, 16, True)
    r.add_break(WD_BREAK.LINE)
    r2 = p.add_run(title)
    set_font(r2, 16, True)
    # Description box (template: boxed italic paragraph under the chapter title).
    t = doc.add_table(rows=1, cols=1)
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    cp = c.paragraphs[0]
    cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cp.paragraph_format.line_spacing = 1.15
    rr = cp.add_run(summary)
    set_font(rr, 12, False, True)
    para("", space_after=0)


def sub(num, title):
    return para("%s %s" % (num, title), "Report Subtopic")


def ph(text, **kw):
    """Placeholder paragraph in the body style, italic."""
    return para(text, italic=True, **kw)


def new_section(fmt, start=1, with_page=True, first=False):
    if first:
        sec = doc.sections[0]
    else:
        sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.left_margin = Inches(1.5)
    sec.right_margin = Inches(1.0)
    sec.top_margin = Inches(1.5)
    sec.bottom_margin = Inches(1.0)
    sec.footer_distance = Inches(0.5)
    sec.header_distance = Inches(0.5)
    sp = sec._sectPr
    tp = sp.find(qn("w:titlePg"))
    if tp is not None:
        sp.remove(tp)
    pg = sp.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        cols = sp.find(qn("w:cols"))
        cols.addprevious(pg)
    for a in list(pg.attrib):
        del pg.attrib[a]
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    sec.footer.is_linked_to_previous = False
    sec.first_page_footer.is_linked_to_previous = False
    ft = sec.footer
    for p in list(ft.paragraphs)[1:]:
        p._element.getparent().remove(p._element)
    fp = ft.paragraphs[0]
    for r in list(fp.runs):
        r._element.getparent().remove(r._element)
    # Footer as in the template: bold Times New Roman 12, rule above in 622423.
    ppr = fp._element.get_or_add_pPr()
    for old in ppr.findall(qn("w:pBdr")):
        ppr.remove(old)
    bdr = OxmlElement("w:pBdr")
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "thinThickSmallGap"); top.set(qn("w:sz"), "24")
    top.set(qn("w:space"), "1"); top.set(qn("w:color"), "622423")
    bdr.append(top)
    ppr.append(bdr)
    fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fp.paragraph_format.left_indent = Inches(-0.4)
    fp.paragraph_format.right_indent = Inches(-0.2)
    r = fp.add_run(FOOTER_TEXT + ("  Page | " if with_page else ""))
    set_font(r, 10, True)
    if with_page:
        add_field(fp, "PAGE \\* MERGEFORMAT", size=10, bold=True)
    return sec


# ---------------------------------------------------------------------------
# SECTION 1: cover, certificate, declaration, abstract, acknowledgement,
# contents. No page numbers (template rule).
# ---------------------------------------------------------------------------
new_section("decimal", 1, with_page=False, first=True)

# Cover page
p = para("", "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER)
p.add_run().add_picture(LOGO, width=Inches(3.0))
para(UNIVERSITY, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=15, space_after=2)
para(SCHOOL, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=2)
para(DEPARTMENT, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=24)
para(TITLE, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=22, color=RED, space_after=6)
para(SUBTITLE, "Report Plain", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=20)
para("Submitted in Partial Fulfillment of Requirements", "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=0)
para("For the Degree Of", "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=8)
para(DEGREE, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=15, space_after=0)
para(BRANCH, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=10)
para("By", "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=4)
for n, roll in STUDENTS:
    para(n, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=0)
    para("Roll No: " + roll, "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=4)
para("Guide", "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=2)
para(GUIDE[0], "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, space_after=0)
para(GUIDE[1], "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=10)
para(UNIVERSITY, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=0)
para(ADDRESS, "Report Plain", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=6)
para(YEAR, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)

# Certificate
para(UNIVERSITY, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, page_break_before=True, space_after=2)
para(SCHOOL, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, space_after=24)
para("Certificate", "Report Front Heading", color=RED, space_after=12)
para("This is to certify that the project report entitled “%s” submitted by the following students at the end of Semester VII of LY B. Tech is a bona fide record for partial fulfillment of requirements for the degree of %s %s of %s." % (TITLE, DEGREE, BRANCH, UNIVERSITY))  # CONFIRM semester wording
para("")
for n, roll in STUDENTS:
    para("%s    (%s)" % (n, roll), "Report Plain", bold=True, space_after=6)
para("", "Report Plain", space_after=30)
t = doc.add_table(rows=2, cols=2)
cells = [("Guide", GUIDE[0]), ("Head of the Department", HOD), ("Examiner", ""), ("Director", "")]
for i, (role, name) in enumerate(cells):
    c = t.rows[i // 2].cells[i % 2]
    c.paragraphs[0].text = ""
    r = c.paragraphs[0].add_run("\n\n_______________________________")
    set_font(r, 11)
    p2 = c.add_paragraph()
    r2 = p2.add_run(role)
    set_font(r2, 12, True)
    p3 = c.add_paragraph()
    r3 = p3.add_run(name if name else " ")
    set_font(r3, 12)
para("", "Report Plain", space_after=12)
para("Date:", "Report Plain", bold=True)
para("Place: Mumbai-77", "Report Plain", bold=True)

# Declaration
para(UNIVERSITY, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, page_break_before=True, space_after=2)
para(SCHOOL, "Report Plain", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, space_after=24)
para("Declaration", "Report Front Heading", color=RED, space_after=12)
para("We declare that this written report submission represents the work done based on our and / or others’ ideas with adequately cited and referenced the original source. We also declare that we have adhered to all principles of intellectual property, academic honesty and integrity as we have not misinterpreted or fabricated or falsified any idea / data / fact / source / original work / matter in our submission.")
para("We understand that any violation of the above will be cause for disciplinary action by the college and may evoke the penal action from the sources which have not been properly cited or from whom proper permission is not sought.")
para("", "Report Plain", space_after=24)
t = doc.add_table(rows=2, cols=2)
for i, (n, roll) in enumerate(STUDENTS):
    c = t.rows[i // 2].cells[i % 2]
    c.paragraphs[0].text = ""
    r = c.paragraphs[0].add_run("\n\n_____________________")
    set_font(r, 11)
    for line, b in (("Signature of the Student", False), (n, True), ("Roll No. " + roll, False)):
        pp = c.add_paragraph()
        rr = pp.add_run(line)
        set_font(rr, 12, b)
para("", "Report Plain", space_after=12)
para("Date:", "Report Plain", bold=True)
para("Place: Mumbai-77", "Report Plain", bold=True)

# Abstract
para("Abstract", "Report Front Heading", page_break_before=True, space_after=12)
para(tbd("abstract, written last from the logged experiments. Structure: the problem (detecting manipulated images, video and audio); why current reported numbers are hard to compare (different protocols); what was built (a minimal baseline and seven documented iterations in one pipeline, with source-disjoint splits and several seeds); the main finding, " + frm("E-xxx, final comparison") + "; what did not work, " + frm("E-xxx, failed iterations") + "; the limits of the evidence. No sentence is written here until the experiment that supports it exists."), italic=True)
para("Keywords: " + tbd("deepfake detection, multimodal, generalisation, evaluation protocol, calibration; confirm after the abstract is written") + ".", italic=True)

# Acknowledgement
para("Acknowledgement", "Report Front Heading", page_break_before=True, space_after=12)  # CONFIRM position
para("We express our gratitude to our guide, %s, %s, Department of Electronics and Computer Engineering, K. J. Somaiya School of Engineering, for her guidance during this project, and to %s, Head of the Department." % (GUIDE[0], GUIDE[1], HOD))
para(tbd("further acknowledgements (faculty, dataset providers, open-source communities actually used); to be written by the students"), italic=True)

# Contents
para("Contents", "Report Front Heading", page_break_before=True, space_after=12)
toc_p = doc.add_paragraph(style="Report Plain")
add_field(toc_p, 'TOC \\h \\z \\t "Report Chapter Title,1,Report Subtopic,2,Report Back Heading,1"', size=12)

# ---------------------------------------------------------------------------
# SECTION 2: roman numerals. List of Figures, List of Tables, Nomenclature.
# ---------------------------------------------------------------------------
new_section("lowerRoman", 1)
para("List of Figures", "Report Front Heading", space_after=12)
para(tbd("table of figures; generated after the figures exist (Word: References, Insert Table of Figures)"), italic=True)
para("List of Tables", "Report Front Heading", page_break_before=True, space_after=12)
para(tbd("table of tables; generated after the tables exist"), italic=True)
para("Nomenclature and Abbreviations", "Report Front Heading", page_break_before=True, space_after=12)
abbr = [
    ("AUC", "Area under the ROC curve (definition to be fixed in Chapter 3, open decision O-005)"),
    ("ECE", "Expected calibration error"),
    ("EER", "Equal error rate"),
    ("MVP", "Minimum viable product, the simplest end-to-end baseline"),
    ("rPPG", "Remote photoplethysmography, a pulse signal estimated from skin colour changes in video"),
    ("VRAM", "Memory on the graphics processor"),
]
t = doc.add_table(rows=0, cols=2)
t.style = doc.styles["Table Grid"]
for a, d in abbr:
    row = t.add_row().cells
    row[0].paragraphs[0].add_run(a); set_font(row[0].paragraphs[0].runs[0], 12, True)
    row[1].paragraphs[0].add_run(d); set_font(row[1].paragraphs[0].runs[0], 12)
para("")
para(tbd("extend the list when the final text exists"), italic=True)

# ---------------------------------------------------------------------------
# SECTION 3: Arabic page numbers from 1. The chapters.
# ---------------------------------------------------------------------------
new_section("decimal", 1)

# ---- Chapter 1 -----------------------------------------------------------
chapter(1, "Introduction",
        "This chapter presents the background of deepfake detection, the motivation for this project, its scope "
        "and objectives, and how the report is organised. " + tbd("sharpen the scope after decisions O-001 and O-002 are made") + ".")
sub("1.1", "Background")
para("Manipulated faces, videos and voices can be produced with consumer hardware and off-the-shelf software " + cite("dfdc") + ". A useful detector has to handle images, video and the audio track of a video, and has to say how sure it is.")
para("The detectors and datasets in the literature are measured under different protocols. In one benchmark that trained all detectors on one dataset and tested them within and across datasets, a plain Xception network reached a mean AUC of 0.9450 on the in-domain tests and 0.7718 on the cross-domain tests " + cite("bench") + ". These are that paper's own numbers under its own protocol, and they are not results of this project. The gap between in-dataset and cross-dataset results recurs across face datasets " + cite("celeb", "xray", "dfdc") + ".")  # source R-017, not ours
sub("1.2", "Motivation")
para("A single in-dataset number says little about behaviour on new material. The motivation of this project is to show a documented path from a simple baseline to a measured detector, where every step has a stated reason and a recorded outcome, including steps that did not help.")
sub("1.3", "Scope of the Project")
para("The scope is images, videos and the audio inside videos (decision D-001 to D-010 in the project record). How each input type is handled is " + tbd("open decision O-002") + ". A detector trained on one kind of manipulation supports no claim about other kinds without a test on them.")
sub("1.4", "Objectives")
para("The objectives are to: (a) build the simplest end-to-end detector and measure it under a protocol defined before any model exists; (b) make at least seven further iterations, each started from a written hypothesis, success criterion and failure criterion; (c) compare candidate models in one pipeline with at least three seeds and interval estimates; (d) test whether heart-rate (rPPG) evidence adds anything beyond the interval of the paired difference. The goal is results comparable with published detectors on our own protocol, not the best model in the world.")
sub("1.5", "Organization of the Report")
para("Chapter 2 surveys the literature. Chapter 3 gives the project design and the mathematical background. Chapter 4 describes the implementation and experiments. Chapter 5 gives the conclusions and further work.")
sub("1.6", "Summary of the Chapter and Outline of the Next")
para("This chapter gave the background, motivation, scope and objectives. The next chapter reviews the related literature and ends with the objective of the work.")

# ---- Chapter 2 -----------------------------------------------------------
chapter(2, "Literature Survey",
        "This chapter presents the literature on face-forgery datasets, visual detectors, physiological-signal "
        "detectors, audio and audio-visual detection, and calibration, and ends with the objective of the work. "
        "Only papers that were read from their primary source are cited.")
sub("2.1", "Datasets for Face Forgery")
para("FaceForensics++ provides real videos manipulated with four methods at several compression levels, with a fixed split and a hidden-test benchmark, and reports that detectors lose accuracy as compression increases " + cite("ffpp") + ". Celeb-DF was built with an improved synthesis pipeline to reduce visible artefacts; its authors ran existing detectors with their released models on it and on earlier sets and found it the hardest of those compared " + cite("celeb") + ". The DFDC dataset offers a large collection of videos from consenting actors with held-out subjects, and its analysis of the challenge entries reports weak performance on test content that came from outside the dataset " + cite("dfdc") + ".")
sub("2.2", "Visual Detectors and Generalisation")
para("Face X-ray predicts the blending boundary between a pasted face and its background and is trained on self-made blended images, so no fakes from a particular generator are required " + cite("xray") + ". DeepfakeBench places many detectors in one codebase with one preprocessing and metric pipeline, and finds that plain strong backbones come close to more elaborate methods when the training recipe is shared " + cite("bench") + ". For still images in general, replacing a trained real-versus-fake network by a linear or nearest-neighbour classifier on the frozen features of a pretrained vision-language model generalised across generators better than the trained baselines in the authors’ tests " + cite("ojha") + ". Work on transfer learning for image classification relates source-task accuracy to accuracy after transfer, with caveats that depend on the regularisers used " + cite("kornblith") + "; whether this holds for forensic detection is open and is tested here.")
sub("2.3", "Physiological Signals")
para("FakeCatcher derives photoplethysmography-like signals from facial regions and classifies them, with high in-dataset accuracy on the generators of its time " + cite("fakecatcher") + ". A follow-up uses the same kind of signal to identify which generator produced a fake " + cite("hearts") + ". DeepFakesON-Phys fine-tunes a heart-rate estimation network as a detector, with one model per database and identity-disjoint development and test sets " + cite("onphys") + ". Later work is more cautious. A study with its own recordings reports that a heart-rate signal can be recovered from high-quality face swaps and can follow the driving video " + cite("heart") + ". An independent reproduction of DeepFakesON-Phys on talking-face fakes, under an identity-disjoint split, reported a much lower AUC than the original in-dataset figure " + cite("harraq") + ". Pulse signals can also be edited on purpose, so a pulse-based check can be attacked " + cite("lin") + ". For these reasons rPPG is treated here as a hypothesis to test and not as a component to adopt.")
sub("2.4", "Audio and Audio-Visual Detection")
para("ASVspoof 2019 is a standard public database of synthesised, converted and replayed speech, with attacks in the evaluation set that are absent from training " + cite("asvspoof") + ". AASIST is a compact model on the raw waveform that applies graph attention over spectral and temporal nodes, and its authors report variation across seeds " + cite("aasist") + ". FakeAVCeleb provides videos with both visual and cloned-voice manipulations, and its baselines show that naive combinations of unimodal detectors can perform poorly " + cite("fakeav") + ". A two-stream audio-visual network that scores the dissonance between the streams over short chunks can also localise manipulated segments " + cite("dissonance") + ".")
sub("2.5", "Calibration")
para("Modern neural networks are often poorly calibrated, and temperature scaling, a single parameter fitted on validation data, corrects this on the tasks the authors studied " + cite("calib") + ".")
sub("2.6", "Gaps and Objective of the Work")
para("Reported numbers rarely share a protocol, and several of the strongest in-dataset results are not matched across datasets or generators. The objective of this work is therefore to define one protocol, run the candidates under it, and report differences with their uncertainty. " + tbd("final objective statement, to be confirmed with the guide after decisions O-001 to O-004"))
sub("2.7", "Summary of the Chapter and Outline of the Next")
para("This chapter reviewed datasets, detectors, physiological signals, audio and audio-visual methods, and calibration. The next chapter gives the project design: the problem statement, the evaluation rules and the mathematical background.")

# ---- Chapter 3 -----------------------------------------------------------
chapter(3, "Project Design",
        "This chapter presents the problem statement, the rules that keep the evaluation honest, and the "
        "mathematical background of the quantities used. Choices of model, dataset and metric are open decisions "
        "and are shown as placeholders.")
sub("3.1", "Problem Statement")
para("Let x denote an input (an image, a video clip, or the audio of a clip) and y in {0, 1} its label, with y = 1 for manipulated content. A detector outputs a score s(x) in [0, 1]. The exact output and the labels are " + tbd("open decision O-001") + ". How image, video and audio inputs are each handled is " + tbd("open decision O-002") + ".")
sub("3.2", "Data, Splits and Leakage Policy")
para("Datasets: " + tbd("open decision O-003") + ". Each dataset’s licence, source, version and hash are recorded before use. Every split is stored as a file with an identifier and a hash. No identity, source video or near-duplicate frame appears in more than one partition (policy " + tbd("open decision O-004") + "). The test partition is touched only for a frozen choice, and the evaluation script refuses test data unless the run is flagged as final.")
sub("3.3", "Comparison Rules")
para("All candidates are compared on the same data, split, compute budget and metric, with at least three random seeds. For each candidate the mean, the standard deviation over seeds and a bootstrap confidence interval are reported. The bootstrap resamples independent units (identities or source videos), not frames, because frames of one video are not independent. Candidates are compared in a paired way: for each resample, the difference of the metric between two candidates on the same units is computed. If the interval of the difference includes zero, the evidence does not separate the candidates, and the simpler or cheaper one is preferred. The number of resamples and the interval level are " + tbd("fixed in Stage 2 before any model exists") + ".")
sub("3.4", "Metrics, Calibration and Thresholds")
para("The metric set is " + tbd("open decision O-005") + ". Whatever is chosen is defined mathematically here before use. The threshold is chosen on validation data only. For calibration, the expected calibration error with B equal-width bins is ECE = sum over m = 1..B of (|B_m| / n) * |acc(B_m) - conf(B_m)|, where B_m is the set of samples whose confidence falls in bin m, acc is the accuracy in the bin and conf is the mean confidence in the bin " + cite("calib") + ". Temperature scaling divides the logit z by a single parameter T > 0 fitted on validation data, so that the probability is sigmoid(z / T) in the binary case " + cite("calib") + ". Whether temperature scaling is used is " + tbd("open decision O-005") + ".")
sub("3.5", "Cost Measurements")
para("For every candidate the following are measured under the same conditions on the project machine and reported next to accuracy: parameters, forward-pass FLOPs for the input size actually used, peak GPU memory in a training step, seconds per epoch, epochs to a validation target fixed in advance, sensitivity to the learning rate, and ease of fine-tuning (number of hyperparameters tuned, stability across seeds).")
sub("3.6", "Candidate Models and Fusion")
para("Baseline: " + tbd("open decision O-006") + ". Candidate backbones per modality: " + tbd("open decision O-007") + ". Fine-tuning strategy: " + tbd("open decision O-008") + ". Fusion across modalities, including inputs with a missing modality (a still image has no audio): " + tbd("open decision O-009") + ".")
sub("3.7", "Heart-Rate (rPPG) Feasibility Test")
para("The feasibility test checks, in order: whether a heart rate can be recovered from the data at all; whether rPPG features separate real from fake under an identity-disjoint split for several generator families; whether adding rPPG evidence raises the held-out metric beyond the interval of the paired difference; and whether negative controls (time-shuffled or mismatched signals) behave as expected. Success and failure thresholds are written before the run " + tbd("open decision O-010") + ". If the test fails, rPPG is reported as a negative result.")
sub("3.8", "Summary of the Chapter and Outline of the Next")
para("This chapter fixed the rules of evaluation and listed the open choices. The next chapter describes the implementation and the experiments, and holds the results once they exist.")

# ---- Chapter 4 -----------------------------------------------------------
chapter(4, "Implementation and Experimentation",
        "This chapter presents the environment, the data, the minimal baseline, the seven iterations, the final "
        "comparison and the heart-rate feasibility test. Every result is copied from a logged experiment; until "
        "an experiment exists the section says so.")
sub("4.1", "Environment")
para("Hardware measured on 2026-10-24 and recorded as ENV-DESKTOP-GPU in the experiment log: one GPU with 12 GB of memory (about 11 GiB usable while the desktop is running), 24 CPU threads and about 31 GB of RAM. Software versions: " + frm("ENV-DESKTOP-GPU in docs/EXPERIMENTS.md") + ".")  # source: docs/PLAN.md fixed constraints
sub("4.2", "Data and Splits")
ph(tbd("datasets, licences, hashes and split files; to be filled from docs/DATA_LOG.md after decisions O-003 and O-004"))
sub("4.3", "Minimal Baseline (I-0)")
para("Hypothesis, success criterion and failure criterion: written in docs/EXPERIMENTS.md before the run. Design: " + tbd("O-006") + ". Result: " + frm("E-xxx, I-0") + ".")
iters = [
    ("4.4", "Iteration 1 (I-1)"), ("4.5", "Iteration 2 (I-2)"), ("4.6", "Iteration 3 (I-3)"),
    ("4.7", "Iteration 4 (I-4)"), ("4.8", "Iteration 5 (I-5)"), ("4.9", "Iteration 6 (I-6)"),
    ("4.10", "Iteration 7 (I-7)"),
]
for n, t_ in iters:
    sub(n, t_)
    para("What was tried: " + tbd("one sentence, from the hypothesis written before the run") + ". Why: " + tbd("the reason, tied to the previous iteration or a paper") + ". What happened: " + frm("E-xxx, " + t_.split("(")[1].rstrip(")")) + ". Failed runs stay in the record.")
sub("4.11", "Final Comparison")
para("Candidates compared under the common protocol of Chapter 3: " + frm("E-xxx, final comparison") + ".")
tbl = doc.add_table(rows=3, cols=4)
tbl.style = doc.styles["Table Grid"]
hdr = ["Candidate", "Metric (mean, sd over seeds)", "95% interval", "Cost axes"]
for i, h in enumerate(hdr):
    c = tbl.rows[0].cells[i]; c.paragraphs[0].add_run(h); set_font(c.paragraphs[0].runs[0], 11, True)
for r_ in (1, 2):
    vals = [tbd("candidate"), frm("E-xxx"), frm("E-xxx"), frm("E-xxx")]
    for i, v in enumerate(vals):
        c = tbl.rows[r_].cells[i]; c.paragraphs[0].add_run(v); set_font(c.paragraphs[0].runs[0], 10, False, True)
para("Table 4.1: Final comparison (structure only). Protocol: " + tbd("split name; in-dataset or cross-dataset") + ".", "Report Plain", italic=True, size=11)
sub("4.12", "Calibration and Cross-Dataset Behaviour")
para(frm("E-xxx") + ".")
sub("4.13", "Heart-Rate Feasibility Test")
para(frm("E-xxx, Stage 6") + ".")
sub("4.14", "Software")
para("The form of the software (command line, local web app, or both) is " + tbd("open decision O-011") + ".")
sub("4.15", "Summary of the Chapter and Outline of the Next")
para("This chapter will summarise what was measured. The next chapter gives the conclusions and further work.")

# ---- Chapter 5 -----------------------------------------------------------
chapter(5, "Conclusions and Further Work",
        "This chapter presents the conclusions drawn from the experiments, with what the evidence does not "
        "support, and the further work. It is written after the results exist.")
sub("5.1", "Conclusions")
para(tbd("8 to 10 lines of conclusion, then the important conclusions in point form, each tied to an experiment ID; written by the students from the results") + ".", italic=True)
sub("5.2", "Limitations")
para("Limits known before any run: published numbers come from different protocols, so comparison with them is made only by re-running a published detector on our protocol; a detector trained on one kind of manipulation supports no claim about other kinds without a test on them; labelled audio-visual datasets are fewer and smaller than visual ones; compute is limited to one 12 GB GPU. " + tbd("limits found during the experiments, from the experiment log"))
sub("5.3", "Further Work")
para(tbd("future scope in point form, after the results"), italic=True)

# ---- Back matter ---------------------------------------------------------
para("Bibliography", "Report Back Heading", page_break_before=True, space_after=12)
BIB_ANCHOR = doc.add_paragraph(style="Report Reference")  # replaced below once all citations are known

para("Publications", "Report Back Heading", page_break_before=True, space_after=12)
para("None yet. " + tbd("list any publication arising from this work, with status; do not list unsubmitted work as published"), italic=True)
para("Appendix A: Experiment Log Index", "Report Back Heading", page_break_before=True, space_after=12)
para(tbd("table of experiment IDs, dates, environments, hypotheses and outcomes, copied from docs/EXPERIMENTS.md"), italic=True)
para("Appendix B: Decision Log Index", "Report Back Heading", space_after=12)
para(tbd("table of decision IDs D-001 to D-xxx and open decisions O-001 to O-011 with their status, copied from docs/DECISIONS.md"), italic=True)

# Fill the bibliography in order of first citation.
anchor = BIB_ANCHOR._element
prev = anchor
for i, k in enumerate(_order, 1):
    np_ = copy.deepcopy(anchor)
    for ch in list(np_):
        if ch.tag != qn("w:pPr"):
            np_.remove(ch)
    prev.addnext(np_)
    from docx.text.paragraph import Paragraph
    pp = Paragraph(np_, BIB_ANCHOR._parent)
    pp.paragraph_format.left_indent = Inches(0.4)
    pp.paragraph_format.first_line_indent = Inches(-0.4)
    r = pp.add_run("[%d]  %s" % (i, REFS[k]))
    set_font(r, 12)
    prev = np_
anchor.getparent().remove(anchor)

# Document properties
cp = doc.core_properties
cp.title = TITLE + " (technical report, draft skeleton)"
cp.author = "Project group, K. J. Somaiya School of Engineering"
cp.last_modified_by = "Project group"
cp.subject = "Semester VII project report"
cp.comments = "Skeleton built 2026-10-24; no results; see docs/_agents/A.md"

doc.save(OUT)
print("saved", OUT, "citations used:", len(_order))
