"""WCAG 2.x contrast check of the colour tokens in static/app.css (light and dark). Read-only; prints a Markdown table.

Text pairs must reach 4.5:1; marks that carry meaning but are not text (control edges, chart slices, bar fill) must reach 3:1.
Exit code 1 if any pair fails.   Run:  .venv/Scripts/python.exe deliverables_v2/website/contrast_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parent / "static" / "app.css"

# (foreground token, background token, kind, what it is)
TEXT = [
    ("text", "bg", "body text on page"), ("text", "card", "body text on card"), ("text", "card-2", "text on beige panel"),
    ("text-2", "bg", "secondary text on page"), ("text-2", "card", "secondary text on card"), ("text-2", "card-2", "secondary text on beige"),
    ("muted", "bg", "muted text on page"), ("muted", "card", "muted text on card"), ("muted", "card-2", "muted text on beige"),
    ("muted", "none-bg", "muted text on neutral chip"),
    ("primary", "card", "brown link/summary on card"), ("primary", "bg", "brown link on page"), ("primary", "primary-soft", "brown on soft brown"),
    ("primary-ink", "primary", "button text on brown"), ("text", "primary-soft", "text on soft brown"),
    ("accent-text", "bg", "kicker on page"), ("accent-text", "card", "kicker on card"), ("text-2", "accent-soft", "limits text on accent-soft"),
    ("text", "accent-soft", "text on accent-soft"), ("muted", "accent-soft", "muted on accent-soft"),
    ("attn-fg", "attn-bg", "merits attention label"), ("warn-fg", "warn-bg", "weak-evidence / warning label"),
    ("none-fg", "none-bg", "nothing-found label (neutral)"), ("inc-fg", "inc-bg", "inconclusive label"),
    ("sys-fg", "sys-bg", "did not complete label"), ("cred-fg", "cred-bg", "credential verified label"),
    ("mock-fg", "mock-bg", "MOCK badge/banner"), ("mock-fg", "card", "MOCK banner stripe base"), ("exp-fg", "exp-bg", "experimental label"),
    ("exp-fg", "card", "experimental dashed text on card"), ("text-2", "card-2", "chip text"),
    ("warn-fg", "card", "uncertainty marker/warn text on card"), ("attn-fg", "card", "error text on card"),
    ("text", "none-bg", "text on neutral"), ("text", "warn-bg", "text on warning panel"), ("text-2", "inc-bg", "n/a"),
    ("muted", "none-bg", "muted on neutral"), ("text-2", "none-bg", "secondary on neutral"),
    ("muted", "card", "caption on card"), ("accent-text", "card-2", "accent on beige"),
]
TEXT = [t for t in TEXT if t[2] != "n/a"]
NONTEXT = [
    ("control-border", "card", "control edge on card"), ("control-border", "bg", "control edge on page"), ("control-border", "card-2", "dashed dropzone edge on beige"),
    ("primary", "card", "score bar fill on card"), ("text", "card", "threshold marker on card"),
    ("d-attention", "card", "chart: merits attention"), ("d-nothing_found", "card", "chart: nothing found"), ("d-credential", "card", "chart: credential"),
    ("d-inconclusive", "card", "chart: inconclusive"), ("d-informational", "card", "chart: file information"),
    ("d-experimental", "card", "chart: experimental"), ("d-not_applicable", "card", "chart: did not apply"), ("d-unsupported", "card", "chart: not available"),
    ("d-did_not_run", "card", "chart: did not complete"), ("accent", "card", "brand dot, non-text mark"),
]


def parse(css: str) -> tuple[dict, dict]:
    light = dict(re.findall(r"--([\w-]+):(#[0-9A-Fa-f]{6})", re.search(r"^:root\{(.*?)^\s*color-scheme:light;\s*\}", css, re.S | re.M).group(1)))
    dark_block = re.search(r':root\[data-theme="dark"\]\{(.*?)color-scheme:dark\}', css, re.S).group(1)
    dark = dict(light)
    dark.update(dict(re.findall(r"--([\w-]+):(#[0-9A-Fa-f]{6})", dark_block)))
    return light, dark


def lum(h: str) -> float:
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a: str, b: str) -> float:
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def main() -> int:
    css = CSS.read_text(encoding="utf-8")
    light, dark = parse(css)
    bad = 0
    seen = set()
    print("| Pair (foreground on background) | Use | Needs | Light | Dark |")
    print("|---|---|---|---|---|")
    for group, need in ((TEXT, 4.5), (NONTEXT, 3.0)):
        for fg, bg, what in group:
            if (fg, bg) in seen:
                continue
            seen.add((fg, bg))
            rl, rd = ratio(light[fg], light[bg]), ratio(dark[fg], dark[bg])
            flag = "" if rl >= need and rd >= need else " FAIL"
            bad += 0 if not flag else 1
            print(f"| {fg} {light[fg]} / {dark[fg]} on {bg} {light[bg]} / {dark[bg]} | {what} | {need} | {rl:.2f} | {rd:.2f}{flag} |")
    print(f"\nfailures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
