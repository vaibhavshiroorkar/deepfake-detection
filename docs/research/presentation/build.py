"""Build the research report and presentation from their text sources."""

import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as SlideColor
from pptx.util import Inches as SlideInches
from pptx.util import Pt as SlidePt

HERE = Path(__file__).resolve().parent
AUTHORS = "Vaibhav Shiroorkar; Shivansh Gupta; Adithya Jithesh"


def report():
    document = Document()
    document.core_properties.author = AUTHORS
    document.core_properties.last_modified_by = AUTHORS
    document.core_properties.title = "Deepfake detection research report"
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(.65)
    section.left_margin = section.right_margin = Inches(.75)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.08
    for name, size in [("Title", 25), ("Heading 1", 20)]:
        document.styles[name].font.color.rgb = RGBColor(0, 0, 0)
        document.styles[name].font.size = Pt(size)
    for border in document.styles.element.xpath('.//w:pBdr'):
        border.getparent().remove(border)
    footer = section.footer.paragraphs[0]
    footer.text = "Deepfake detection research | "
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    footer.style = document.styles["Caption"]
    document.styles["Caption"].font.color.rgb = RGBColor(70, 70, 70)
    sections = 0
    for block in (HERE / "research-report.md").read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            document.add_paragraph(block[2:], "Title")
        elif block.startswith("## "):
            if sections in {2, 4, 6, 8}:
                document.add_page_break()
            document.add_heading(block[3:], level=1)
            sections += 1
        elif block.startswith("|"):
            rows = [line.strip("|").split("|") for line in block.splitlines()]
            rows = [row for row in rows if not all(set(cell.strip()) <= {"-", ":"} for cell in row)]
            table = document.add_table(rows=0, cols=len(rows[0]))
            table.style = "Table Grid"
            for row_index, row in enumerate(rows):
                for cell, value in zip(table.add_row().cells, row):
                    cell.text = value.strip()
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        run.font.color.rgb = RGBColor(0, 0, 0)
                        run.bold = row_index == 0
            document.add_paragraph()
        else:
            text = " ".join(block.splitlines())
            paragraph = document.add_paragraph(text)
            if text.startswith("Evidence:"):
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(70, 70, 70)
    document.save(HERE / "Research_Report.docx")


def presentation():
    deck = Presentation()
    deck.slide_width, deck.slide_height = SlideInches(13.333), SlideInches(7.5)
    deck.core_properties.author = AUTHORS
    deck.core_properties.last_modified_by = AUTHORS
    deck.core_properties.title = "Deepfake detection research"
    slides = json.loads((HERE / "slides.json").read_text(encoding="utf-8"))

    def text(slide, value, left, top, width, height, size, color, bold=False):
        box = slide.shapes.add_textbox(*map(SlideInches, (left, top, width, height)))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        paragraph = frame.paragraphs[0]
        paragraph.text = value
        paragraph.font.name = "Calibri"
        paragraph.font.size = SlidePt(size)
        paragraph.font.color.rgb = SlideColor(*color)
        paragraph.font.bold = bold

    for index, item in enumerate(slides, 1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        text(slide, item["title"], .75, .55, 11.8, 1.1, 34, (132, 31, 45), True)
        for row, line in enumerate(item["lines"]):
            text(slide, line, .75, 2.0 + row * .88, 11.8, .82, 23, (25, 25, 25))
        text(slide, "Evidence: " + item["source"], .75, 6.30, 11.8, .48, 11, (70, 70, 70))
        text(slide, f"Equal credit: Vaibhav Shiroorkar, Shivansh Gupta, Adithya Jithesh | {index}/15",
             .75, 7.02, 11.8, .25, 10, (70, 70, 70))
        slide.notes_slide.notes_text_frame.text = item["notes"] + "\n\nEvidence: " + item["source"]
    deck.save(HERE / "Research_Presentation.pptx")


if __name__ == "__main__":
    report()
    presentation()
    print("Wrote Research_Report.docx and Research_Presentation.pptx")
