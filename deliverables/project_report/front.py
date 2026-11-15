"""Front matter of the v2 report: cover, certificate, declaration, acknowledgement, abstract, contents, lists (copied from report/front.py and compacted)."""
from build_report import (X, run, field, esc, LOGO, UNIVERSITY, SCHOOL, DEPARTMENT, BRANCH, STUDENTS, GUIDE, GUIDE_DESIG, HOD,
                          ADDRESS, YEAR, PLACE, TITLE, MAROON)


def cp(B, text, size=12, bold=False, italic=False, before=0, after=0, line=276, keep=False, color=None):
    """One centred line of the cover."""
    ppr = '%s<w:spacing w:before="%d" w:after="%d" w:line="%d" w:lineRule="auto"/><w:jc w:val="center"/>' % (
        "<w:keepNext/>" if keep else "", before * 20, after * 20, line)
    return B.emit(X('<w:p><w:pPr>%s</w:pPr>%s</w:p>' % (ppr, run(text, bold, italic, size, extra=('<w:color w:val="%s"/>' % color) if color else ""))))


def cover(B):
    B.picture_par(LOGO, 8.0, '<w:spacing w:before="0" w:after="240"/><w:jc w:val="center"/>')
    cp(B, UNIVERSITY, 16, True, after=3)
    cp(B, SCHOOL, 14, True, after=3)
    cp(B, DEPARTMENT, 14, True, after=34)
    cp(B, TITLE, 24, True, after=44, line=300, color=MAROON)
    cp(B, "Submitted In Partial Fulfillment of Requirements", 12)
    cp(B, "For the Degree Of", 12, after=14)
    cp(B, "Bachelor of Technology", 16, True, after=2)
    cp(B, BRANCH, 14, True, after=22)
    cp(B, "By", 12, after=8)
    B.plain_table([[n, "Roll No: " + r] for n, r in STUDENTS], [6.6, 5.2], ["left", "left"], size=13, bold_first=True)
    cp(B, "Guide", 12, before=22, after=6)
    cp(B, GUIDE, 14, True, after=2)
    cp(B, GUIDE_DESIG, 12, after=28)
    cp(B, UNIVERSITY, 14, True, after=3)
    cp(B, ADDRESS, 12, after=12)
    cp(B, YEAR, 14, True)


def inst_header(B, brk=False):
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="InstLine"/>%s</w:pPr>%s</w:p>' % ("<w:pageBreakBefore/>" if brk else "", run(UNIVERSITY))))
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="InstLine"/><w:spacing w:after="240"/></w:pPr>%s</w:p>' % run(SCHOOL)))


def sign_cells(B, labels, with_names):
    """Signature block: columns of equal width, a rule above the label, no table borders."""
    rows = []
    for lab, name in labels:
        rows.append((lab, name))
    return rows


def signature_table(B, pairs):
    """pairs: list of rows, each a list of (label, name or None); a rule is drawn above every label."""
    w = int(9333 / 2)
    out = []
    for row in pairs:
        cells = []
        for lab, name in row:
            if lab is None:
                cells.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p></w:tc>' % w)
                continue
            body = ('<w:p><w:pPr><w:spacing w:before="720" w:after="0" w:line="240" w:lineRule="auto"/><w:pBdr><w:top w:val="single" w:sz="6" w:space="1" w:color="%s"/></w:pBdr>'
                    '<w:ind w:right="567"/></w:pPr>%s</w:p>' % (MAROON, run(lab, True, False, 12)))
            for extra in (name or []):
                body += '<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="276" w:lineRule="auto"/></w:pPr>%s</w:p>' % extra
            cells.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/></w:tcPr>%s</w:tc>' % (w, body))
        out.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>%s</w:tr>' % "".join(cells))
    tbl = ('<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="left"/><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid><w:gridCol w:w="%d"/><w:gridCol w:w="%d"/></w:tblGrid>%s</w:tbl>' % (2 * w, w, w, "".join(out)))
    B.emit(X(tbl))


def date_place(B):
    for lab, val in (("Date:", ""), ("Place:", " " + PLACE)):
        B.emit(X('<w:p><w:pPr><w:spacing w:before="%d" w:after="0" w:line="360" w:lineRule="auto"/></w:pPr>%s%s</w:p>'
                 % (360 if lab == "Date:" else 0, run(lab, True), run(val))))


def certificate(B):
    inst_header(B)
    B.front_title("Certificate")
    B.p("This is to certify that the project report entitled <b>“%s”</b> submitted by the following students at the end of Semester VII of LY B. Tech "
        "is a bona fide record for partial fulfillment of requirements for the degree of <b>Bachelor of Technology %s</b> of %s." % (TITLE, BRANCH, UNIVERSITY))
    B.plain_table([[n, "(" + r + ")"] for n, r in STUDENTS], [6.6, 4.0], ["left", "left"], size=12, bold_first=True)
    B.spacer(6)
    signature_table(B, [
        [("Guide", [run(GUIDE)]), ("Head of the Department", [run(HOD)])],
        [("Examiner", None), ("Director", None)],
    ])
    date_place(B)


def declaration(B):
    inst_header(B, True)
    B.front_title("Declaration")
    B.p("We declare that this written report submission represents the work done based on our and / or others’ ideas with adequately cited and referenced the original source. "
        "We also declare that we have adhered to all principles of intellectual property, academic honesty and integrity as we have not misinterpreted or fabricated or falsified "
        "any idea / data / fact / source / original work / matter in our submission.")
    B.p("We understand that any violation of the above will be cause for disciplinary action by the college and may evoke the penal action from the sources which have not been "
        "properly cited or from whom proper permission is not sought.")
    def stu(i):
        n, r = STUDENTS[i]
        return ("Signature of the Student", [run(n, True), run("Roll No. ", True) + run(r)])
    signature_table(B, [[stu(0), stu(1)], [stu(2), (None, None)]])
    date_place(B)


def acknowledgement(B):
    B.h1("Acknowledgement", kind="front")
    B.p("We would like to express our sincere gratitude to our project guide, <b>%s</b>, %s, %s, K. J. Somaiya School of Engineering, for her guidance, advice and "
        "encouragement throughout this project." % (GUIDE, GUIDE_DESIG, DEPARTMENT))
    B.p("We thank <b>%s</b>, Head of the %s, for the facilities that made the work possible, and the faculty and staff of K. J. Somaiya School of Engineering for their support." % (HOD, DEPARTMENT))
    B.p("We acknowledge the authors of the datasets used in this work, FaceForensics++, Celeb-DF, Celeb-DF++, DeepFakeDetection, ASVspoof 2019, In-the-Wild, LAV-DF, FakeAVCeleb, "
        "DeepAction, DF26, Community Forensics and the ForenSynths and UnivFD generated-image sets, who made them available for research, and the developers of the open-source "
        "software on which the project is built, among them PyTorch, timm, Hugging Face Transformers, OpenCV and Flask.")
    B.p("Finally we thank our families and friends for their patience and support.")


def abstract(B):
    import chapters
    chapters.abstract(B)


def contents(B):
    B.front_title("Contents")
    toc = ('<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr>%s</w:p>' % field(r'TOC \o "1-2" \h \z \u', "Contents are generated by Word."))
    B.emit(X(toc))


def lists(B):
    """List of Figures and List of Tables, flowing directly after the contents (compact)."""
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="FrontTitle"/><w:spacing w:before="480"/><w:outlineLvl w:val="0"/></w:pPr>%s</w:p>' % run("List of Figures")))
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="TableofFigures"/></w:pPr>%s</w:p>' % field(r'TOC \h \z \c "Figure"', "No table of figures entries found.")))
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="FrontTitle"/><w:spacing w:before="240"/><w:outlineLvl w:val="0"/></w:pPr>%s</w:p>' % run("List of Tables")))
    B.emit(X('<w:p><w:pPr><w:pStyle w:val="TableofFigures"/></w:pPr>%s</w:p>' % field(r'TOC \h \z \c "Table"', "No table of figures entries found.")))
