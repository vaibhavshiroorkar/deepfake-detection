import front
import ch1_2
import ch3
import ch4
import ch5
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
    # section 3: contents (continuation heading in the header)
    front.contents(B)
    B.end_section({"num": 'w:fmt="lowerRoman"', "title_pg": True, "footer": True, "header": "Contents (continued)"})
    # section 4: lists and nomenclature
    front.lists(B, ch5.nomenclature)
    B.end_section({"num": 'w:fmt="lowerRoman"', "header": ""})
    # section 5: body, arabic numbers from 1
    ch1_2.intro(B)
    ch1_2.survey(B)
    ch3.build(B)
    ch4.build(B)
    ch5.conclusions(B)
    B.h1("References", kind="front")
    B.bibliography(REFS)
    ch5.appendices(B)
    B.finish_sections({"num": 'w:fmt="decimal" w:start="1"', "header": ""})
