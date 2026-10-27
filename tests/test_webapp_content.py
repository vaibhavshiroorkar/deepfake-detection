"""Every research number the web app shows has to be one the record holds.

`webapp/content.py` is written by hand for readers, so nothing generates it from
the result files. Without this check a number can be typed once, drift from the
measurement it came from, and keep being served.
"""

import re
from pathlib import Path

RECORD = (
    Path("docs/research/findings.md"),
    Path("docs/research/result-traceability.md"),
)
# Three or four decimals: ROC-AUC, intervals and deltas. Shorter decimals are
# thresholds and probabilities written for readers, such as 0.5.
NUMBER = re.compile(r"(?<![\d.])[01]\.\d{3,4}(?!\d)")


def test_every_number_on_the_page_is_in_the_findings_or_the_registry() -> None:
    content = Path("src/deepfake_detection/webapp/content.py").read_text(
        encoding="utf-8"
    )
    record = "".join(path.read_text(encoding="utf-8") for path in RECORD)
    missing = sorted(
        {number for number in NUMBER.findall(content) if number not in record}
    )
    assert missing == [], f"Numbers with no recorded source: {missing}"
