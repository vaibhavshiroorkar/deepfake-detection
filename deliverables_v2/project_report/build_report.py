"""Builds the v2 project report (Word) from the college template.

Copy of report/build_report.py adapted for deliverables_v2/project_report (2026-10-28). Base: the template
template_base_converted_from_doc.docx copied into this folder (the supplied template "67F_B Tech Semester VII
Project Report template.doc" converted by Microsoft Word). The page geometry (A4, margins, page border), the
footer rule and the chapter box of the template are kept; the brown of the template and of the old builder
(622423) is replaced by the paper's maroon B7202F (MAROON below). Word updates every field and exports the PDF
(word_tools.ps1 in this folder).

Run from the repository root:
    uv run --no-project --with python-docx --with lxml python deliverables_v2/project_report/build_report.py

Reads shared assets from the repository by explicit paths (assets/branding, deliverables_v2/figures,
assets/figures, docs/test_evidence); writes every output (DOCX, PDF, _render) only inside this folder.
Rules in force (CLAUDE.md, deliverables_v2/SPEC.md): no number that is not in deliverables_v2/CONTENT_MAP.md;
identity fields exactly as in docs/IDENTITY.md with the author order of SPEC.md; no em-dashes or en-dashes;
no decision, observation or experiment ids in visible text (they appear only in Word comments).
"""
import copy
import os
import re
import shutil
import subprocess
import sys
import zipfile
from xml.sax.saxutils import escape

from lxml import etree
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.section import Section
from docx.shared import Cm
from docx.text.run import Run
import omml_lite

HERE = os.path.dirname(os.path.abspath(__file__))            # deliverables_v2/project_report
ROOT = os.path.dirname(os.path.dirname(HERE))                  # repository root (read-only use)
sys.path.insert(0, HERE)

BASE = os.path.join(HERE, "template_base_converted_from_doc.docx")
RENDER = os.path.join(HERE, "_render")
STAGE = os.path.join(RENDER, "stage.docx")
OUT = os.path.join(HERE, "Multimodal_Deepfake_Detection_Project_Report.docx")
PDF = os.path.join(HERE, "Multimodal_Deepfake_Detection_Project_Report.pdf")
for _p in (STAGE, OUT, PDF):                                   # outputs never leave this folder
    assert os.path.commonpath([HERE, os.path.abspath(_p)]) == HERE, _p
XSL = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
FIG = os.path.join(ROOT, "assets", "figures")
LOGO = os.path.join(ROOT, "assets", "branding", "somaiya_vidyavihar_logo.jpg")
PLATFORM_FIGURE = os.path.join(ROOT, "deliverables_v2", "figures", "fig_platform_v2.png")   # 8 modules (current)

# DASHBOARD SCREENSHOTS: the one place to swap in screenshots of the redesigned v2 website.
# Each entry: (absolute path, caption, alternative text). Use only screenshots that show no dataset media.
DASHBOARD_SCREENSHOTS = [
    (os.path.join(ROOT, "deliverables_v2", "figures", "dash_v2_overview.png"),
     "Redesigned dashboard after a live analysis of a demonstration video: the donut counts the checks by outcome; it is not a probability, a score or a verdict (no dataset media shown).",
     "Screenshot of the redesigned dashboard: a donut counting eight checks by outcome next to the summary of what the checks found"),
    (os.path.join(ROOT, "deliverables_v2", "figures", "dash_v2_cards.png"),
     "Check cards of the same report: the face-video card reads Possible manipulation with high uncertainty and shows its model output against the decision line, not as a probability.",
     "Screenshot of two check cards: face manipulation analysis with the label Possible manipulation and a model-output bar, and content credentials with no credentials found"),
]
DASHBOARD_WIDTH_CM = 15.0

# identity fields, exactly as in docs/IDENTITY.md
UNIVERSITY = "Somaiya Vidyavihar University"
SCHOOL = "K. J. Somaiya School of Engineering"
DEPARTMENT = "Department of Electronics and Computer Engineering"
BRANCH = "(Electronics and Computer Engineering)"
# author order of deliverables_v2/SPEC.md (user instruction 2026-10-28); names and roll numbers as in docs/IDENTITY.md
STUDENTS = [("Shivansh Gupta", "16014023052"), ("Vaibhav Shiroorkar", "16014024805"), ("Adithya Jithesh", "16014024803")]
GUIDE, GUIDE_DESIG, HOD = "Dr. Tejashree Phatak", "Assistant Professor", "Dr. Jagannath Nirmal"
ADDRESS, YEAR, PLACE = "Vidyavihar, Mumbai - 400 077", "2026-2027", "Mumbai-77"
FOOTER_TEXT = "Department of Electronics and Computer Engineering Semester VII 2026-27 Batch  Page "
TITLE = "Multimodal Deepfake Detection for Images, Video and Audio"   # no subtitle (SPEC.md)
MAROON = "B7202F"   # paper/college-style.tex collegemaroon; replaces the template brown 622423 everywhere
FONT = "Times New Roman"
FOOT_AFTER = 548   # twips of space after the footer text: puts the footer rule where the template has it
FOOT_BEFORE = 180  # twips of space above the footer rule: the template (compatibility mode 11) puts the extra line leading of
                   # 1.5 spacing BELOW the text, so its last body line clears the rule by about 3.6 mm; compatibility mode 15
                   # (kept for the equations and tables) puts it above, so the last line touched the rule. The rule stays
                   # where it is; only the body area ends this much higher.

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
    "w10": "urn:schemas-microsoft-com:office:word",
}
NSDECL = " ".join('xmlns:%s="%s"' % kv for kv in NS.items())
M = "{%s}" % NS["m"]
TEXTW = 9333  # twips, text width of the template (11907 - 1440 - 1134)
SIDE = 1287   # twips, equal left and right margin of the v2 body: (11907 - TEXTW) / 2, so the text block is centred between the page borders (D-092)



def X(s):
    return parse_xml(re.sub(r"^<([\w:]+)", lambda m: "<%s %s" % (m.group(1), NSDECL), s, count=1))


def esc(t):
    return escape(t)


def omml(latex):
    """LaTeX subset -> m:oMath element string (see omml_lite.py)."""
    return omml_lite.omml(latex)


def rpr(bold=False, italic=False, size=None, extra=""):
    s = ""
    if bold:
        s += "<w:b/><w:bCs/>"
    if italic:
        s += "<w:i/><w:iCs/>"
    s += extra
    if size:
        s += '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size * 2, size * 2)
    return "<w:rPr>%s</w:rPr>" % s if s else ""


def run(text, bold=False, italic=False, size=None, extra=""):
    return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr(bold, italic, size, extra), esc(text))


def field(instr, result, bold=False, size=None):
    p = rpr(bold, False, size)
    return ('<w:r>%s<w:fldChar w:fldCharType="begin"/></w:r><w:r>%s<w:instrText xml:space="preserve"> %s </w:instrText></w:r>'
            '<w:r>%s<w:fldChar w:fldCharType="separate"/></w:r><w:r>%s<w:t>%s</w:t></w:r><w:r>%s<w:fldChar w:fldCharType="end"/></w:r>'
            % (p, p, instr, p, p, esc(result), p))


TOKEN = re.compile(r"(<b>.*?</b>|<i>.*?</i>|<c [^>]+>|<pend [^>]+>|<ref [^>]+>|\$[^$]+\$|<br/>)")

# ---------------------------------------------------------------------------------------------------------------
# styles, numbering and fonts


def style_xml():
    tnr = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'

    def st(sid, name, based="Normal", nxt=None, ppr="", rp="", typ="paragraph", q=True, custom=False):
        return ('<w:style w:type="%s" %sw:styleId="%s"><w:name w:val="%s"/>%s%s%s<w:pPr>%s</w:pPr><w:rPr>%s</w:rPr></w:style>'
                % (typ, 'w:customStyle="1" ' if custom else "", sid, name,
                   '<w:basedOn w:val="%s"/>' % based if based else "",
                   '<w:next w:val="%s"/>' % nxt if nxt else "", "<w:qFormat/>" if q else "", ppr, rp))

    sz = lambda n, b=False, i=False: tnr + ("<w:b/><w:bCs/>" if b else "") + ("<w:i/><w:iCs/>" if i else "") + '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (n * 2, n * 2)
    MCOL = '<w:color w:val="%s"/>' % MAROON
    out = []
    out.append(st("BodyText", "Body Text", nxt="BodyText", ppr='<w:spacing w:before="0" w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/>', rp=sz(12)))
    out.append(st("Heading1", "heading 1", nxt="BodyText",
                  ppr='<w:keepNext/><w:keepLines/><w:pageBreakBefore/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="20"/></w:numPr>'
                      '<w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/>', rp=sz(16, True) + MCOL))
    out.append(st("Heading2", "heading 2", nxt="BodyText",
                  ppr='<w:keepNext/><w:keepLines/><w:numPr><w:ilvl w:val="1"/><w:numId w:val="20"/></w:numPr>'
                      '<w:spacing w:before="120" w:after="60" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/><w:outlineLvl w:val="1"/>', rp=sz(14, True) + MCOL))
    out.append(st("Heading3", "heading 3", nxt="BodyText",
                  ppr='<w:keepNext/><w:keepLines/><w:numPr><w:ilvl w:val="2"/><w:numId w:val="20"/></w:numPr>'
                      '<w:spacing w:before="60" w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/><w:outlineLvl w:val="2"/>', rp=sz(12, True)))
    out.append(st("FrontTitle", "Front Title", nxt="BodyText", custom=True,
                  ppr='<w:keepNext/><w:keepLines/><w:spacing w:before="0" w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/>', rp=sz(16, True) + MCOL))
    out.append(st("InstLine", "Institution Line", custom=True,
                  ppr='<w:keepNext/><w:spacing w:before="0" w:after="60" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/>', rp=sz(14, True)))
    out.append(st("Caption", "caption", nxt="BodyText",
                  ppr='<w:keepLines/><w:spacing w:before="60" w:after="160" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>', rp=sz(11, False, False)))
    out.append(st("TableText", "Table Text", custom=True,
                  ppr='<w:spacing w:before="30" w:after="30" w:line="240" w:lineRule="auto"/>', rp=sz(10)))
    out.append(st("ListItem", "List Item", based="BodyText", custom=True,
                  ppr='<w:spacing w:before="0" w:after="60" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/>', rp=sz(12)))
    out.append(st("Code", "Code Line", custom=True,
                  ppr='<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="567"/>',
                  rp='<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="22"/><w:szCs w:val="22"/>'))
    tab = '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="%d"/></w:tabs>' % TEXTW
    out.append(st("TOC1", "toc 1", nxt="BodyText", q=False,
                  ppr=tab + '<w:spacing w:before="40" w:after="0" w:line="320" w:lineRule="auto"/>', rp=sz(12, True)))
    out.append(st("TOC2", "toc 2", nxt="BodyText", q=False,
                  ppr=tab + '<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/><w:ind w:left="397"/>', rp=sz(12)))
    out.append(st("TOC3", "toc 3", nxt="BodyText", q=False,
                  ppr=tab + '<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/><w:ind w:left="964"/>', rp=sz(12)))
    out.append(st("TableofFigures", "table of figures", nxt="BodyText", q=False,
                  ppr=tab + '<w:spacing w:before="0" w:after="60" w:line="264" w:lineRule="auto"/><w:ind w:left="1021" w:right="567" w:hanging="1021"/>', rp=sz(12)))
    out.append('<w:style w:type="character" w:customStyle="1" w:styleId="Placeholder"><w:name w:val="Placeholder Value"/><w:qFormat/><w:rPr><w:b/><w:bCs/></w:rPr></w:style>')
    return out


def numbering_xml():
    def lvl(i, fmt, text, left, hang, pstyle=None, suff=None, rp="", jc="left"):
        return ('<w:lvl w:ilvl="%d"><w:start w:val="1"/><w:numFmt w:val="%s"/>%s%s<w:lvlText w:val="%s"/><w:lvlJc w:val="%s"/>'
                '<w:pPr><w:ind w:left="%d" w:hanging="%d"/></w:pPr>%s</w:lvl>'
                % (i, fmt, '<w:pStyle w:val="%s"/>' % pstyle if pstyle else "", '<w:suff w:val="%s"/>' % suff if suff else "", text, jc, left, hang, rp))
    tnr = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:hint="default"/>'
    head = ('<w:abstractNum w:abstractNumId="20"><w:nsid w:val="5A1B2C3D"/><w:multiLevelType w:val="multilevel"/><w:tmpl w:val="5A1B2C3E"/>'
            + lvl(0, "decimal", "", 0, 0, "Heading1", "nothing", "<w:rPr>%s<w:b/><w:color w:val=\"%s\"/><w:sz w:val=\"32\"/></w:rPr>" % (tnr, MAROON), "center")
            + lvl(1, "decimal", "%1.%2", 680, 680, "Heading2", None, "<w:rPr>%s<w:b/><w:color w:val=\"%s\"/><w:sz w:val=\"28\"/></w:rPr>" % (tnr, MAROON))
            + lvl(2, "decimal", "%1.%2.%3", 850, 850, "Heading3", None, "<w:rPr>%s<w:b/></w:rPr>" % tnr)
            + "".join(lvl(i, "decimal", ".".join("%%%d" % (k + 1) for k in range(i + 1)), 1080, 1080) for i in range(3, 9))
            + "</w:abstractNum>")
    bullet = ('<w:abstractNum w:abstractNumId="21"><w:nsid w:val="5A1B2C40"/><w:multiLevelType w:val="hybridMultilevel"/><w:tmpl w:val="5A1B2C41"/>'
              + lvl(0, "bullet", "\u2022", 624, 340, None, None, "<w:rPr>%s</w:rPr>" % tnr)
              + lvl(1, "bullet", "\u2022", 1021, 340, None, None, "<w:rPr>%s</w:rPr>" % tnr) + "</w:abstractNum>")
    dec = ('<w:abstractNum w:abstractNumId="22"><w:nsid w:val="5A1B2C42"/><w:multiLevelType w:val="hybridMultilevel"/><w:tmpl w:val="5A1B2C43"/>'
           + lvl(0, "decimal", "%1.", 624, 397, None, None, "<w:rPr>%s</w:rPr>" % tnr) + "</w:abstractNum>")
    return [head, bullet, dec]


def fix_fonts(xml_bytes):
    s = xml_bytes.decode("utf8")
    keep = ("Times New Roman", "Symbol", "Wingdings", "Cambria Math")

    def sub(m):
        return m.group(0) if m.group(2) in keep else '%s="Times New Roman"' % m.group(1)
    return re.sub(r'(w:(?:ascii|hAnsi|cs|eastAsia))="([^"]+)"', sub, s).encode("utf8")


def postprocess(path):
    """Times New Roman everywhere (styles, theme, numbering, footers), compatibility mode 15."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            n = item.filename
            if re.match(r"word/(styles|numbering|header\d*|footer\d*|footnotes|endnotes|fontTable)\.xml$", n):
                data = fix_fonts(data)
            if n == "word/theme/theme1.xml":
                data = re.sub(rb'<a:latin typeface="[^"]*"', b'<a:latin typeface="Times New Roman"', data)
                data = re.sub(rb'<a:(ea|cs) typeface="[^"]*"', rb'<a:\1 typeface=""', data)
            if n.endswith(".xml") or n.endswith(".rels"):
                data = data.replace(b"622423", MAROON.encode())
            if n == "word/settings.xml":
                data = data.replace(b'w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="11"',
                                    b'w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"')
            zout.writestr(item, data)
    os.replace(tmp, path)


def outline(inl):
    """Thin maroon border around a figure picture (DrawingML line on the picture shape)."""
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    sppr = next(e for e in inl.iter() if e.tag.endswith("}spPr"))
    ln = parse_xml('<a:ln xmlns:a="%s" w="9525"><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:ln>' % (A, MAROON))
    sppr.append(ln)


# ---------------------------------------------------------------------------------------------------------------


class Builder:
    def __init__(self, reg, cite_order, comment_author="Project team"):
        self.doc = Document(BASE)
        d = self.doc
        self.body = d.element.body
        self.sect = self.body.find(qn("w:sectPr"))
        for ch in list(self.body):
            if ch is not self.sect:
                self.body.remove(ch)
        # styles
        styles = d.styles.element
        for sx in style_xml():
            for old in styles.findall(qn("w:style")):
                if old.get(qn("w:styleId")) == re.search(r'w:styleId="([^"]+)"', sx).group(1):
                    styles.remove(old)
            styles.append(X(sx))
        # numbering
        num = d.part.numbering_part.element
        first_num = num.find(qn("w:num"))
        for ax in numbering_xml():
            first_num.addprevious(X(ax))
        for nid, aid in ((20, 20), (21, 21)):
            num.append(X('<w:num w:numId="%d"><w:abstractNumId w:val="%d"/></w:num>' % (nid, aid)))
        self.next_num = 40
        self.reg = reg                 # label -> number text, from the first pass
        self.new_reg = {}
        self.cite_order = list(cite_order)
        self.new_cites = []
        self.counters = {"fig": 0, "tab": 0, "eq": 0, "ch": 0, "app": 0}
        self.pend_total = 0
        self.comments = []             # (run element, text)
        self.shape_id = 1037
        self.sections = []
        self.author = comment_author
        self.last_el = None
        self.refs = {}

    # ---- low level
    def emit(self, el):
        self.sect.addprevious(el)
        self.last_el = el
        return el

    def cite_no(self, key):
        if key not in self.new_cites:
            self.new_cites.append(key)
        order = self.cite_order if key in self.cite_order else self.new_cites
        return order.index(key) + 1

    def inline(self, text, base_bold=False, base_italic=False, size=None):
        """Mini markup -> run xml. Returns (xml, [pending ids])."""
        out, pend = [], []
        if text.startswith("$") and text.count("$") == 2 and text.endswith("$"):
            out.append(run("\u200b"))
        for part in TOKEN.split(text):
            if not part:
                continue
            if part.startswith("<b>"):
                out.append(run(part[3:-4], True, base_italic, size))
            elif part.startswith("<i>"):
                out.append(run(part[3:-4], base_bold, True, size))
            elif part.startswith("<c "):
                nums = sorted(self.cite_no(k.strip()) for k in part[3:-1].split(","))
                out.append(run("[" + ", ".join(str(n) for n in nums) + "]", base_bold, base_italic, size))
            elif part.startswith("<pend "):
                pend.append(part[6:-1].strip())
                out.append('<w:r><w:rPr><w:rStyle w:val="Placeholder"/>%s</w:rPr><w:t>[pending]</w:t></w:r>' % ("" if not size else '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size * 2, size * 2)))
            elif part.startswith("<ref "):
                lab = part[5:-1].strip()
                num = self.reg.get(lab, "?")
                kind = lab.split(":")[0]
                txt = {"fig": "Figure %s", "tab": "Table %s", "eq": "(%s)"}[kind] % num
                out.append(run(txt, base_bold, base_italic, size))
            elif part == "<br/>":
                out.append("<w:r><w:br/></w:r>")
            elif part.startswith("$"):
                out.append(omml(part[1:-1]))
            else:
                out.append(run(part, base_bold, base_italic, size))
        return "".join(out), pend

    def attach_pending(self, el, pend):
        runs_ = [r for r in el.iter(qn("w:r")) if r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:rStyle")) is not None
                 and r.find(qn("w:rPr")).find(qn("w:rStyle")).get(qn("w:val")) == "Placeholder"]
        for r_, pid in zip(runs_, pend):
            self.comments.append((r_, "Value not yet measured. Experiment %s. The cell is filled from the logged result when the run is frozen." % pid))
            self.pend_total += 1

    def para_xml(self, style, inner, ppr=""):
        return '<w:p><w:pPr><w:pStyle w:val="%s"/>%s</w:pPr>%s</w:p>' % (style, ppr, inner)

    def p(self, text, style="BodyText", ppr="", bold=False, italic=False, size=None):
        inner, pend = self.inline(text, bold, italic, size)
        el = self.emit(X(self.para_xml(style, inner, ppr)))
        self.attach_pending(el, pend)
        return el

    def trace(self, el, text):
        """A Word comment on the first run of a paragraph: where the content comes from (identifiers live only here)."""
        r_ = next((x for x in el.iter(qn("w:r")) if x.find(qn("w:t")) is not None), None)
        if r_ is not None and text:
            self.comments.append((r_, text))

    def bibliography(self, refs):
        for i, key in enumerate(self.new_cites, 1):
            txt = refs[key]
            inner, _ = self.inline(txt, False, False, 11)
            self.emit(X(self.para_xml("BodyText", run("[%d]" % i, False, False, 11) + '<w:r><w:tab/></w:r>' + inner,
                                      '<w:tabs><w:tab w:val="left" w:pos="567"/></w:tabs><w:spacing w:before="0" w:after="100" w:line="276" w:lineRule="auto"/><w:ind w:left="567" w:hanging="567"/><w:jc w:val="left"/>')))

    # ---- headings
    def h1(self, title, kind="chapter", trace=None):
        if kind == "chapter":
            self.counters["ch"] += 1
            self.counters["tab_ch"] = 0
            inner = run("Chapter %d" % self.counters["ch"]) + "<w:r><w:br/></w:r>" + run(title)
            ppr = '<w:ind w:left="130"/>'   # D-092: keeps the chapter title within 2 mm of the template after the margins were centred
        elif kind == "appendix":
            self.counters["app"] += 1
            letter = chr(64 + self.counters["app"])
            inner = run("Appendix %s" % letter) + "<w:r><w:br/></w:r>" + run(title)
            ppr = '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="130"/>'
        else:
            inner = run(title)
            ppr = '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="130"/>'
        el = self.emit(X(self.para_xml("Heading1", inner, ppr)))
        self.trace(el, trace)
        return el

    def h2(self, title, trace=None, plain=None):
        first = getattr(self, "after_box", None) is self.last_el
        ppr = '<w:spacing w:before="0"/>' if first else ""
        if plain is not None:
            ppr = ""
        ppr = ppr if plain is None else '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="0" w:firstLine="0"/>'
        el = self.emit(X(self.para_xml("Heading2", run((plain + "  " if plain else "") + title), ppr)))
        self.trace(el, trace)
        return el

    def h3(self, title, trace=None):
        el = self.emit(X(self.para_xml("Heading3", run(title))))
        self.trace(el, trace)
        return el

    def front_title(self, title):
        return self.emit(X(self.para_xml("FrontTitle", run(title))))

    def box(self, text):
        """The chapter description box of the template: a text box of 454.65 pt by 71.4 pt below the chapter title."""
        self.shape_id += 1
        rp = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>'
        inner, _ = self.inline(text, False, True, 12)
        shape = ('<w:r>%s<w:pict><v:shape id="_x0000_s%d" type="#_x0000_t202" style="position:absolute;left:0;text-align:left;margin-left:5.95pt;'
                 'margin-top:.7pt;width:454.65pt;height:71.4pt;z-index:251655168;mso-width-relative:margin;mso-height-relative:margin" strokecolor="#%s" strokeweight=".75pt">'
                 '<v:textbox><w:txbxContent><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>%s</w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r>'
                 % (rp, self.shape_id, MAROON, inner))
        self.emit(X('<w:p><w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/>%s</w:pPr>%s</w:p>' % (rp, shape)))
        for _ in range(3):
            self.emit(X('<w:p><w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/><w:rPr><w:sz w:val="24"/></w:rPr></w:pPr></w:p>'))
        self.after_box = self.last_el

    # ---- lists
    def bullets(self, items):
        for it in items:
            inner, pend = self.inline(it)
            el = self.emit(X(self.para_xml("ListItem", inner, '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="21"/></w:numPr>')))
            self.attach_pending(el, pend)

    def numbered(self, items):
        nid = self.next_num
        self.next_num += 1
        self.doc.part.numbering_part.element.append(X(
            '<w:num w:numId="%d"><w:abstractNumId w:val="22"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>' % nid))
        for it in items:
            inner, pend = self.inline(it)
            el = self.emit(X(self.para_xml("ListItem", inner, '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="%d"/></w:numPr>' % nid)))
            self.attach_pending(el, pend)

    # ---- figures, tables, equations
    def _seq(self, kind):
        self.counters[kind] += 1
        return self.counters[kind]

    def figure(self, label, fname, caption, width_cm, alt, tag=None, trace=None):
        n = self._seq("fig")
        self.new_reg["fig:" + label] = str(n)
        path = fname if os.path.isabs(fname) else os.path.join(FIG, fname)
        inl = self.doc.part.new_pic_inline(path, Cm(width_cm), None)
        inl.docPr.set("descr", alt)
        inl.docPr.set("name", "Figure %d" % n)
        outline(inl)
        r_ = X("<w:r><w:drawing/></w:r>")
        r_.find(qn("w:drawing")).append(inl)
        pp = X('<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr></w:p>')
        pp.append(r_)
        self.emit(pp)
        cap_inner, pend = self.inline(caption)
        cap = run("Figure ", True) + field("SEQ Figure \\* ARABIC", str(n), True) + run(". ", True) + cap_inner
        if tag:
            cap += run(" (" + tag + ")", False, True)
        el = self.emit(X(self.para_xml("Caption", cap)))
        self.attach_pending(el, pend)
        if trace:
            self.trace(el, trace)

    def figure_pair(self, label, fnames, caption, width_cm, alts, tag=None, trace=None):
        """Two pictures side by side (a borderless table) under one caption."""
        n = self._seq("fig")
        self.new_reg["fig:" + label] = str(n)
        w = int(TEXTW / 2)
        cells = []
        for k, (fn, alt) in enumerate(zip(fnames, alts)):
            path = fn if os.path.isabs(fn) else os.path.join(FIG, fn)
            inl = self.doc.part.new_pic_inline(path, Cm(width_cm), None)
            inl.docPr.set("descr", alt)
            inl.docPr.set("name", "Figure %d%s" % (n, "ab"[k]))
            outline(inl)
            r_ = X("<w:r><w:drawing/></w:r>")
            r_.find(qn("w:drawing")).append(inl)
            cp = X('<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr></w:p>')
            cp.append(r_)
            cell = X('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/></w:tcPr></w:tc>' % w)
            cell.append(cp)
            cells.append(cell)
        tbl = X('<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/>'
                '<w:right w:w="0" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0000"/></w:tblPr><w:tblGrid><w:gridCol w:w="%d"/><w:gridCol w:w="%d"/></w:tblGrid>'
                '<w:tr><w:trPr><w:cantSplit/></w:trPr></w:tr></w:tbl>' % (2 * w, w, w))
        tr = tbl.find(qn("w:tr"))
        for c in cells:
            tr.append(c)
        self.emit(tbl)
        cap_inner, pend = self.inline(caption)
        cap = run("Figure ", True) + field("SEQ Figure \\* ARABIC", str(n), True) + run(". ", True) + cap_inner
        if tag:
            cap += run(" (" + tag + ")", False, True)
        el = self.emit(X(self.para_xml("Caption", cap)))
        self.attach_pending(el, pend)
        if trace:
            self.trace(el, trace)

    def table(self, label, caption, header, rows, widths_cm, align=None, size=10, keep=True, bold_first=False, font=None, trace=None):
        n = self._seq("tab")
        self.new_reg["tab:" + label] = str(n)
        cap_inner, pend = self.inline(caption)
        cap = run("Table ", True) + field("SEQ Table \\* ARABIC", str(n), True) + run(". ", True) + cap_inner
        el = self.emit(X(self.para_xml("Caption", cap, '<w:keepNext/>')))
        self.attach_pending(el, pend)
        if trace:
            self.trace(el, trace)
        self._grid(header, rows, widths_cm, align, size, keep, bold_first, shade=True, borders=True)
        self.emit(X('<w:p><w:pPr><w:spacing w:before="0" w:after="120" w:line="240" w:lineRule="auto"/><w:rPr><w:sz w:val="12"/></w:rPr></w:pPr></w:p>'))

    def _grid(self, header, rows, widths_cm, align, size, keep, bold_first, shade, borders, center=True):
        ws = [int(round(w * 567)) for w in widths_cm]
        total = sum(ws)
        align = align or ["left"] * len(ws)
        jcmap = {"left": "left", "right": "right", "center": "center"}

        def cell(txt, w, i, head, last):
            inner, pend = self.inline(txt, head or (bold_first and i == 0), False, size)
            kn = "<w:keepNext/>" if keep and not last else ""
            pp = '<w:p><w:pPr><w:pStyle w:val="TableText"/>%s<w:jc w:val="%s"/></w:pPr>%s</w:p>' % (kn, jcmap[align[i]] if not head else ("left" if align[i] == "left" else align[i]), inner)
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if (head and shade) else ""
            return '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s<w:vAlign w:val="center"/></w:tcPr>%s</w:tc>' % (w, shd, pp), pend

        rows_xml, pends = [], []
        allrows = ([header] if header else []) + rows
        for ri, row in enumerate(allrows):
            head = bool(header) and ri == 0
            last = ri == len(allrows) - 1
            cells = []
            for i, (c, w) in enumerate(zip(row, ws)):
                cx, pd = cell(c, w, i, head, last)
                cells.append(cx)
                pends.extend(pd)
            trpr = "<w:cantSplit/>" + ("<w:tblHeader/>" if head else "")
            rows_xml.append("<w:tr><w:trPr>%s</w:trPr>%s</w:tr>" % (trpr, "".join(cells)))
        bd = lambda side: '<w:%s w:val="single" w:sz="6" w:space="0" w:color="%s"/>' % (side, MAROON)
        b = (('<w:tblBorders>%s</w:tblBorders>' % "".join(bd(x) for x in ("top", "left", "bottom", "right", "insideH", "insideV"))) if borders else
             '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')
        tbl = ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="%s"/>%s<w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:left w:w="85" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="04A0"/></w:tblPr>'
               '<w:tblGrid>%s</w:tblGrid>%s</w:tbl>' % (total, "center" if center else "left", b, "".join('<w:gridCol w:w="%d"/>' % w for w in ws), "".join(rows_xml)))
        el = self.emit(X(tbl))
        self.attach_pending(el, pends)
        return el

    def simple_table(self, header, rows, widths_cm, align=None, size=11, keep=False, bold_first=False):
        self._grid(header, rows, widths_cm, align, size, keep, bold_first, shade=True, borders=True)
        self.emit(X('<w:p><w:pPr><w:spacing w:before="0" w:after="120" w:line="240" w:lineRule="auto"/><w:rPr><w:sz w:val="12"/></w:rPr></w:pPr></w:p>'))

    def plain_table(self, rows, widths_cm, align=None, size=12, bold_first=False, keep=False, center=True):
        return self._grid(None, rows, widths_cm, align, size, keep, bold_first, shade=False, borders=False, center=center)

    def equation(self, label, latex, trace=None):
        n = self._seq("eq")
        self.new_reg["eq:" + label] = str(n)
        mid = int(TEXTW * 0.84)
        side = (TEXTW - mid) // 2
        math = '<m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>%s</m:oMathPara>' % omml(latex)
        num = run("(") + field("SEQ Equation \\* ARABIC", str(n)) + run(")")
        tbl = ('<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0000"/></w:tblPr>'
               '<w:tblGrid><w:gridCol w:w="%d"/><w:gridCol w:w="%d"/><w:gridCol w:w="%d"/></w:tblGrid>'
               '<w:tr><w:trPr><w:cantSplit/></w:trPr>'
               '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:p></w:tc>'
               '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="80" w:after="80" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>%s</w:p></w:tc>'
               '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="right"/></w:pPr>%s</w:p></w:tc>'
               '</w:tr></w:tbl>' % (TEXTW, side, mid, side, side, mid, math, side, num))
        el = self.emit(X(tbl))
        if trace:
            self.trace(el, trace)

    def picture_par(self, path, width_cm, ppr=""):
        inl = self.doc.part.new_pic_inline(path, Cm(width_cm), None)
        inl.docPr.set("descr", "Somaiya Vidyavihar University, K. J. Somaiya School of Engineering logo")
        r_ = X("<w:r><w:drawing/></w:r>")
        r_.find(qn("w:drawing")).append(inl)
        pp = X('<w:p><w:pPr>%s</w:pPr></w:p>' % ppr)
        pp.append(r_)
        return self.emit(pp)

    def spacer(self, pts=6):
        self.emit(X('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="%d" w:lineRule="exact"/></w:pPr></w:p>' % int(pts * 20)))

    def pagebreak(self):
        self.emit(X('<w:p><w:r><w:br w:type="page"/></w:r></w:p>'))

    # ---- sections
    def sectpr(self, left=SIDE, right=SIDE, num=None, title_pg=False):
        pg = '<w:pgNumType %s/>' % num if num else ""
        return ('<w:sectPr><w:pgSz w:w="11907" w:h="16840" w:code="9"/>'
                '<w:pgMar w:top="1134" w:right="%d" w:bottom="709" w:left="%d" w:header="720" w:footer="720" w:gutter="0"/>'
                '<w:pgBorders w:offsetFrom="page"><w:top w:val="single" w:sz="4" w:space="24" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="24" w:color="auto"/>'
                '<w:bottom w:val="single" w:sz="4" w:space="24" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="24" w:color="auto"/></w:pgBorders>'
                '%s<w:cols w:space="720"/>%s<w:docGrid w:linePitch="360"/></w:sectPr>' % (right, left, pg, "<w:titlePg/>" if title_pg else ""))

    def end_section(self, cfg):
        """Closes the current section (the section properties travel in a one-point paragraph)."""
        sp = self.sectpr(cfg.get("left", SIDE), cfg.get("right", SIDE), cfg.get("num"), cfg.get("title_pg", False))
        p_ = X('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/><w:rPr><w:sz w:val="2"/></w:rPr>%s</w:pPr></w:p>' % sp)
        self.emit(p_)
        self.sections.append(cfg)

    def finish_sections(self, final_cfg):
        """Replaces the body-level section properties by the last section's and creates headers and footers."""
        self.sections.append(final_cfg)
        new = X(self.sectpr(final_cfg.get("left", SIDE), final_cfg.get("right", SIDE), final_cfg.get("num"), final_cfg.get("title_pg", False)))
        self.sect.addprevious(new)
        self.body.remove(self.sect)
        self.sect = new
        sects = [e for e in self.body.iter(qn("w:sectPr"))]
        assert len(sects) == len(self.sections), (len(sects), len(self.sections))

        def ftr_xml(with_text):
            rp = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
            if not with_text:
                return ['<w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr></w:p>']
            para = ('<w:p><w:pPr><w:pStyle w:val="Footer"/><w:pBdr><w:top w:val="thinThickSmallGap" w:sz="24" w:space="1" w:color="%s"/></w:pBdr>' % MAROON + ''
                    '<w:tabs><w:tab w:val="clear" w:pos="4680"/><w:tab w:val="clear" w:pos="9360"/><w:tab w:val="right" w:pos="9078"/></w:tabs>'
                    '<w:spacing w:before="%d" w:after="%d"/><w:ind w:left="51" w:right="-51"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:pPr>'
                    '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>%s</w:p>' % (FOOT_BEFORE, FOOT_AFTER, rp, esc(FOOTER_TEXT), field("PAGE \\* MERGEFORMAT", "1", True, 10.5)))
            return [para, '<w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr></w:p>']

        def fill(part, xmls):
            el = part._element
            for ch in list(el):
                el.remove(ch)
            for x in xmls:
                el.append(X(x))

        def hdr_xml(text):
            if not text:
                return ['<w:p><w:pPr><w:pStyle w:val="Header"/><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:p>']
            return ['<w:p><w:pPr><w:pStyle w:val="Header"/><w:tabs><w:tab w:val="clear" w:pos="4680"/><w:tab w:val="clear" w:pos="9360"/></w:tabs><w:spacing w:before="0" w:after="240" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>%s</w:p>'
                    % run(text, True, False, 16)]

        for sp, cfg in zip(sects, self.sections):
            sec = Section(sp, self.doc.part)
            if cfg.get("footer"):
                sec.footer.is_linked_to_previous = False
                fill(sec.footer, ftr_xml(True))
            if cfg.get("title_pg"):
                sec.different_first_page_header_footer = True
                sec.first_page_footer.is_linked_to_previous = False
                fill(sec.first_page_footer, ftr_xml(True))
                sec.first_page_header.is_linked_to_previous = False
                fill(sec.first_page_header, hdr_xml(None))
            if "header" in cfg:
                sec.header.is_linked_to_previous = False
                fill(sec.header, hdr_xml(cfg["header"]))

    def add_comments(self):
        for r_, text in self.comments:
            self.doc.add_comment(runs=[Run(r_, None)], text=text, author=self.author, initials="PT")


def build(content_fn, reg, cites):
    b = Builder(reg, cites)
    content_fn(b)
    return b


def main():
    import content
    os.makedirs(RENDER, exist_ok=True)
    first = build(content.build, {}, [])               # pass 1: numbers of figures, tables, equations, citations
    second = build(content.build, first.new_reg, first.new_cites)
    assert second.new_reg == first.new_reg, "numbers changed between passes"
    second.add_comments()
    second.doc.core_properties.title = TITLE
    second.doc.core_properties.subject = ""
    second.doc.core_properties.keywords = ""
    second.doc.core_properties.author = ", ".join(s[0] for s in STUDENTS)
    second.doc.core_properties.comments = ""
    second.doc.save(STAGE)
    postprocess(STAGE)
    ps = os.path.join(HERE, "word_tools.ps1")
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps, "-Mode", "update", "-In", STAGE, "-Out", OUT, "-Pdf", PDF]
    print(subprocess.run(cmd, capture_output=True, text=True).stdout)
    print("placeholders:", second.pend_total, "citations:", len(second.new_cites), "figures:", second.counters["fig"],
          "tables:", second.counters["tab"], "equations:", second.counters["eq"])


if __name__ == "__main__":
    main()
