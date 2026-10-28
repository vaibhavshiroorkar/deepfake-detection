"""Design-system checks for the dashboard (D1): every text colour pair the page uses meets WCAG 2.x AA (4.5:1) in light and dark mode,
controls meet 3:1, and the dark palette is a designed set (every light token redefined), not an inversion. Computed from dashboard.css."""
import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parents[2] / "src" / "dfd" / "platform" / "dashboard" / "dashboard.css").read_text(encoding="utf-8")

# (foreground token, background token) pairs that carry text on the page
TEXT_PAIRS = [("text", "card"), ("text", "bg"), ("text-2", "card"), ("text-2", "bg"), ("muted", "card"), ("muted", "bg"), ("muted", "card-2"),
              ("muted", "none-bg"), ("primary", "card"), ("primary", "bg"), ("primary", "primary-soft"), ("primary-ink", "primary"),
              ("text", "primary-soft"), ("text-2", "none-bg"), ("text-2", "card-2"), ("accent-text", "card"), ("accent-text", "accent-soft"),
              ("attn-fg", "attn-bg"), ("warn-fg", "warn-bg"), ("none-fg", "none-bg"), ("inc-fg", "inc-bg"), ("sys-fg", "sys-bg"),
              ("cred-fg", "cred-bg"), ("mock-fg", "mock-bg"), ("exp-fg", "exp-bg"), ("mock-fg", "card"), ("text", "warn-bg"),
              ("text", "attn-bg"), ("text", "inc-bg")]
CONTROL_PAIRS = [("control-border", "card"), ("control-border", "bg"), ("primary", "card")]


def _tokens(block: str) -> dict:
    return {k: v.upper() for k, v in re.findall(r"--([a-z0-9-]+):(#[0-9A-Fa-f]{6})", block)}


def _blocks():
    light = _tokens(CSS[CSS.index(":root{"):CSS.index("}", CSS.index(":root{"))])
    dark_media = CSS[CSS.index("@media (prefers-color-scheme:dark)"):CSS.index(":root[data-theme=\"dark\"]")]
    dark_attr = CSS[CSS.index(":root[data-theme=\"dark\"]{"):]
    dark_attr = dark_attr[:dark_attr.index("}")]
    return light, _tokens(dark_media), _tokens(dark_attr)


def _lum(hex_):
    def ch(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(hex_[i:i + 2]) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast_formula_reference_values():
    assert round(contrast("#000000", "#FFFFFF"), 2) == 21.0
    assert round(contrast("#767676", "#FFFFFF"), 2) == 4.54          # the well-known AA boundary grey


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_pairs_meet_aa(theme):
    light, dark, _ = _blocks()
    t = light if theme == "light" else {**light, **dark}
    low = [(f, b, round(contrast(t[f], t[b]), 2)) for f, b in TEXT_PAIRS if contrast(t[f], t[b]) < 4.5]
    assert not low, low


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_control_edges_meet_non_text_contrast(theme):
    light, dark, _ = _blocks()
    t = light if theme == "light" else {**light, **dark}
    low = [(f, b, round(contrast(t[f], t[b]), 2)) for f, b in CONTROL_PAIRS if contrast(t[f], t[b]) < 3.0]
    assert not low, low


def test_dark_palette_is_designed_and_both_dark_switches_agree():
    light, dark_media, dark_attr = _blocks()
    colour_tokens = set(light)
    assert colour_tokens <= set(dark_media), sorted(colour_tokens - set(dark_media))
    assert dark_media == dark_attr
    assert ":root:not([data-theme=\"light\"])" in CSS


def test_contract_palette_is_used():
    """docs/EVAL_LAB_AND_UI_CONTRACT.md section 4: the earthy tokens."""
    light, dark, _ = _blocks()
    assert (light["bg"], light["card"], light["primary"], light["sage"]) == ("#F5F3EA", "#FFFDF6", "#244A38", "#82977A")
    assert (light["accent"], light["text"], light["border"]) == ("#A66A4C", "#282D29", "#D8D7CB")
    assert dark["primary"] != light["primary"] and dark["bg"] != light["bg"]
    # no green for "nothing found" (roadmap 4.1): its colour must be a neutral, hue not in the green range
    r, g, b = (int(light["none-fg"][i:i + 2], 16) for i in (1, 3, 5))
    assert not (g > r + 10 and g > b + 10)


def test_clay_accent_is_not_used_for_text():
    """Clay #A66A4C is 4.31:1 on the card, below 4.5:1, so it may only colour non-text marks; clay text uses accent-text."""
    light, _, _ = _blocks()
    assert contrast(light["accent"], light["card"]) < 4.5
    for m in re.finditer(r"[^-]color:var\(--accent\)", CSS):
        raise AssertionError("text colour uses the clay accent: " + CSS[max(0, m.start() - 40):m.end()])


def test_contrast_ratios_are_recorded_in_the_css_header():
    light, _, _ = _blocks()
    head = CSS[:CSS.index(":root{")]
    for f, b in (("text", "card"), ("primary", "card"), ("exp-fg", "exp-bg"), ("attn-fg", "attn-bg")):
        assert f"{contrast(light[f], light[b]):.2f}" in head, (f, b)


def test_reduced_motion_and_focus_ring_present():
    assert "prefers-reduced-motion:reduce" in CSS and ":focus-visible{outline:3px solid var(--primary)" in CSS
