"""Number provenance, shared by scripts/build_numbers.py (writes docs/NUMBERS.json) and scripts/check_docs.py (reads it).

Source side: every numeric token of a logged source (an experiment entry of docs/EXPERIMENTS.md, a result file under data/_results,
the dataset log, the hand-kept literature and derived-number files) becomes one row {value, id, kind, src, ctx}.
Document side: a "result-like" number of the paper, the report or the slides is a decimal 0.dd.. (AUC, EER, ECE, Brier, differences),
a count with a unit noun or with thousands separators, a percentage, or a quantity with a unit (s, ms, minutes, bpm, GiB, MB, M).
A document number is accepted when some row has the same value after rounding to the number of decimals the document prints
(and, for the unit conversions listed in CONV, the converted value). Everything else is reported as unmatched.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

DEC = re.compile(r"(?<![\w.,/])[-+−]?0\.\d{2,}(?!\d)")
PCT = re.compile(r"(?<![\w.,/])(\d+(?:\.\d+)?)\s*(?:percent|%)")
COUNT_NOUN = ("clips?|videos?|images?|utterances?|frames?|speakers?|identit(?:y|ies)|groups?|items?|samples?|files|crops|recordings|"
              "attacks|classes|generators|methods|units|parameters|steps|resamples|windows|runs")
COUNT = re.compile(rf"(?<![\w.,/])(\d{{1,3}}(?:,\d{{3}})+|\d+)\s+(?:{COUNT_NOUN})\b")
COMMA = re.compile(r"(?<![\w.,/])\d{1,3}(?:,\d{3})+(?![\d])")
UNIT = re.compile(r"(?<![\w.,/])(\d+(?:\.\d+)?)\s*(ms|s|seconds?|minutes?|min|hours?|bpm|GiB|GB|MB|M)(?![\w])")

CONV = {   # unit conversions allowed between a source value v and a document value (document = v * c), by class of the document number
    "decimal": (1.0,), "count": (1.0,), "percent": (1.0, 100.0),
    "unit": (1.0, 1 / 60.0, 60.0, 1 / 1000.0, 1000.0, 1 / 3600.0, 3600.0, 1e-6, 1e6, 1 / 1024.0, 1024.0),
}


def num(s: str) -> float:
    return float(s.replace(",", "").replace("−", "-").lstrip("+"))


def decimals(s: str) -> int:
    s = s.replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def kind_of(line: str, start: int) -> str:
    back = line[max(0, start - 60):start].lower()
    fwd = line[start:start + 25].lower()
    for k, pat in (("auc", r"auc"), ("eer", r"\beer\b"), ("ece", r"\bece\b"), ("brier", r"brier"), ("bpm", r"bpm|heart rate"),
                   ("snr", r"\bsnr\b")):
        if re.search(pat, back[-40:]):
            return k
    if re.match(r"\s*(percent|%)", fwd):
        return "percent"
    if re.match(r"\s*(ms|s|seconds?|minutes?|min|hours?)\b", fwd):
        return "time"
    if re.match(r"\s*(gib|gb|mb)\b", fwd):
        return "memory"
    if re.match(r"\s+(clips?|videos?|images?|utterances?|frames?|speakers?|identit|groups?|items?|samples?|files|crops|recordings)", fwd):
        return "count"
    return "number"


def source_rows(text: str, id_: str, src: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        for m in TOKEN.finditer(line):
            v = m.group(0).lstrip("+-−")
            rows.append(dict(value=v, id=id_, kind=kind_of(line, m.start()), src=src, ctx=line[max(0, m.start() - 40):m.end() + 20].strip()))
    return rows


def doc_numbers(text: str) -> list[tuple[str, str, float, int]]:
    """Result-like numbers of a document text: (class, printed form, value, decimals printed)."""
    text = text.replace("\\%", "%").replace("~", " ").replace(" ", " ").replace(" ", " ").replace(" ", " ")
    out, seen = [], set()

    def add(cls, form, val, dec, span):
        if span in seen:
            return
        seen.add(span)
        out.append((cls, form, val, dec))

    for m in PCT.finditer(text):
        add("percent", m.group(0), num(m.group(1)), decimals(m.group(1)), m.span(1))
    for m in UNIT.finditer(text):
        add("unit", m.group(0), num(m.group(1)), decimals(m.group(1)), m.span(1))
    for m in COUNT.finditer(text):
        v = num(m.group(1))
        if v >= 10:
            add("count", m.group(0), v, 0, m.span(1))
    for m in COMMA.finditer(text):
        add("count", m.group(0), num(m.group(0)), 0, m.span(0))
    for m in DEC.finditer(text):
        add("decimal", m.group(0), abs(num(m.group(0))), decimals(m.group(0).lstrip("+-−")), m.span(0))
    return out


class Index:
    def __init__(self, rows):
        self.by_val = {}
        for r in rows:
            try:
                v = abs(num(r["value"]))
            except ValueError:
                continue
            self.by_val.setdefault(v, []).append((decimals(r["value"]), r["id"]))
        self.sorted = sorted(self.by_val)

    def match(self, val: float, dec: int, cls: str = "decimal"):
        """IDs of the rows that print `val` at `dec` decimals: within half a unit of the last printed digit, directly or after one
        conversion of CONV[cls]; for a power-of-ten conversion (including 1) the row must have at least as many decimals as the
        document prints, so that a rounded document number is derived from a more precise logged one and never the reverse."""
        import bisect
        import math
        tol = 0.5 * 10 ** (-dec) + 1e-9
        found = set()
        for c in CONV[cls]:
            lo, hi = (val - tol) / c, (val + tol) / c
            p10 = abs(math.log10(c) - round(math.log10(c))) < 1e-12
            shift = round(math.log10(c)) if p10 else 0
            for v in self.sorted[bisect.bisect_left(self.sorted, lo):bisect.bisect_right(self.sorted, hi)]:
                for dv, id_ in self.by_val[v]:
                    if not p10 or dv - shift >= dec:
                        found.add(id_)
        return found
