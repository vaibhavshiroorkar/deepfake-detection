"""Order of the v2 report: front matter, ten chapters, references, one appendix."""
import front
import chapters
from refs import REFS


def build(B):
    # section 1: cover
    front.cover(B)
    B.end_section({"left": 2160, "right": 1440})
    # section 2: certificate to abstract (roman numbers restart)
    front.certificate(B)
    front.declaration(B)
    front.acknowledgement(B)
    front.abstract(B)
    B.end_section({"num": 'w:fmt="lowerRoman" w:start="1"', "footer": True})
    # section 3: contents and the lists of figures and tables
    front.contents(B)
    front.lists(B)
    B.end_section({"num": 'w:fmt="lowerRoman"', "footer": True, "header": ""})
    # section 4: body, arabic numbers from 1
    chapters.ch1(B)
    chapters.ch2(B)
    chapters.ch3(B)
    chapters.ch4(B)
    chapters.ch5(B)
    chapters.ch6(B)
    chapters.ch7(B)
    chapters.ch8(B)
    chapters.ch9(B)
    chapters.ch10(B)
    B.h1("References", kind="front")
    B.bibliography(REFS)
    chapters.appendix_a(B)
    B.finish_sections({"num": 'w:fmt="decimal" w:start="1"', "header": ""})
