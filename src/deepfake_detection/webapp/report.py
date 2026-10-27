"""The final report as one print-ready HTML page, from the same data as the app.

Charts are drawn here as static SVG because the PDF has no pointer to hover
with. Equations stay LaTeX and KaTeX renders them before the page is printed.
"""

from __future__ import annotations

import datetime
import subprocess
from html import escape
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from deepfake_detection.paths import PROJECT_ROOT

TEMPLATES = Path(__file__).parent / "templates"
PDF_PATH = PROJECT_ROOT / "dist" / "final-report.pdf"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)


def roc_svg(curves: list[dict]) -> Markup:
    width, height = 340, 300
    left, right, top, bottom = 44, 10, 8, 40
    inner_w, inner_h = width - left - right, height - top - bottom

    def x(v: float) -> float:
        return left + v * inner_w

    def y(v: float) -> float:
        return top + (1 - v) * inner_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="figure" role="img">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{x(v)}" x2="{x(v)}" y1="{top}" y2="{top + inner_h}" stroke="#d3dae1"/>'
            f'<line x1="{left}" x2="{left + inner_w}" y1="{y(v)}" y2="{y(v)}" stroke="#d3dae1"/>'
            f'<text x="{x(v)}" y="{height - 24}" font-size="10" text-anchor="middle" fill="#6b7885">{v}</text>'
            f'<text x="{left - 6}" y="{y(v) + 3}" font-size="10" text-anchor="end" fill="#6b7885">{v}</text>'
        )
    parts.append(
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#9aa5b0" stroke-dasharray="4 4"/>'
    )
    for color, curve in zip(SERIES, curves, strict=False):
        points = " ".join(
            f"{x(f):.1f},{y(t):.1f}"
            for f, t in zip(curve["fpr"], curve["tpr"], strict=True)
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        )
    parts.append(
        f'<text x="{left + inner_w / 2}" y="{height - 6}" font-size="10" text-anchor="middle" fill="#4a5866">False-positive rate</text>'
        f'<text transform="translate(11 {top + inner_h / 2}) rotate(-90)" font-size="10" text-anchor="middle" fill="#4a5866">True-positive rate</text>'
        "</svg>"
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{escape(curve["label"])} '
        f"<b>{curve['auc']:.4f}</b></span>"
        for color, curve in zip(SERIES, curves, strict=False)
    )
    # Every interpolated label goes through html.escape above; the rest is
    # markup built here from numbers.
    return Markup(f'<div class="legend">{legend}</div>{"".join(parts)}')  # noqa: S704


def seed_svg(seeds: dict[str, list[dict]]) -> Markup:
    rows = [(label, values) for label, values in seeds.items() if values]
    width, row_h, left, right, top, bottom = 520, 30, 170, 16, 6, 30
    height = top + len(rows) * row_h + bottom
    lo, hi = 0.6, 0.8
    inner_w = width - left - right

    def x(v: float) -> float:
        return left + (min(max(v, lo), hi) - lo) / (hi - lo) * inner_w

    parts = [f'<svg viewBox="0 0 {width} {height}" class="figure wide" role="img">']
    for tick in (0.6, 0.65, 0.7, 0.75, 0.8):
        parts.append(
            f'<line x1="{x(tick)}" x2="{x(tick)}" y1="{top}" y2="{height - bottom}" stroke="#d3dae1"/>'
            f'<text x="{x(tick)}" y="{height - bottom + 14}" font-size="10" text-anchor="middle" fill="#6b7885">{tick}</text>'
        )
    for index, (label, values) in enumerate(rows):
        cy = top + index * row_h + row_h / 2
        ensemble = values[0]["seed"] == "all"
        weight = 600 if ensemble else 400
        parts.append(
            f'<text x="{left - 8}" y="{cy + 4}" font-size="11" text-anchor="end" '
            f'fill="#16212b" font-weight="{weight}">{escape(label)}</text>'
        )
        if len(values) > 1:
            aucs = [v["roc_auc"] for v in values]
            parts.append(
                f'<line x1="{x(min(aucs))}" x2="{x(max(aucs))}" y1="{cy}" y2="{cy}" stroke="#b5c0ca" stroke-width="2"/>'
            )
        for value in values:
            color = "#2f5bd3" if ensemble else "#4a5866"
            parts.append(
                f'<circle cx="{x(value["roc_auc"])}" cy="{cy}" r="{6 if ensemble else 4.5}" fill="{color}" stroke="#fff" stroke-width="1.5"/>'
            )
            if ensemble:
                parts.append(
                    f'<text x="{x(value["roc_auc"]) + 10}" y="{cy + 4}" font-size="10" fill="#4a5866">{value["roc_auc"]:.4f}</text>'
                )
    parts.append(
        f'<text x="{left + inner_w / 2}" y="{height - 2}" font-size="10" text-anchor="middle" fill="#4a5866">DFDC ROC-AUC per training seed</text></svg>'
    )
    return Markup("".join(parts))  # noqa: S704 (labels escaped above)


def render_report(research: dict, *, for_print: bool = False) -> str:
    """The report page. `for_print` points KaTeX at the files on disk, which a
    browser opening the page from a server would refuse to load."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
    )
    template = environment.get_template("report.html")
    katex_dir = Path(__file__).parent / "static" / "vendor" / "katex"
    if for_print:
        katex_css = (katex_dir / "katex.min.css").as_uri()
        katex_js = (katex_dir / "katex.min.js").as_uri()
    else:
        katex_css = "/static/vendor/katex/katex.min.css"
        katex_js = "/static/vendor/katex/katex.min.js"
    return template.render(
        r=research,
        today=datetime.date.today().isoformat(),
        roc_favc=roc_svg(research["roc"].get("FakeAVCeleb test", [])),
        roc_dfdc=roc_svg(research["roc"].get("DFDC", [])),
        seeds=seed_svg(research["seeds"]),
        katex_css=katex_css,
        katex_js=katex_js,
    )


def find_browser() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Edge or Chrome found to print the PDF. Open /report and print it."
    )


def build_pdf(research: dict, *, output: Path = PDF_PATH) -> Path:
    """Write the HTML next to the PDF and print it with a headless browser.

    The HTML points at the vendored KaTeX by file URL, so the print needs no
    server and no network. The time budget lets KaTeX finish before printing.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    html_path = output.with_suffix(".html")
    html_path.write_text(render_report(research, for_print=True), encoding="utf-8")
    # The executable comes from a fixed list and the only argument derived from
    # data is a path this function wrote.
    subprocess.run(  # noqa: S603
        [
            str(find_browser()),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={output}",
            html_path.as_uri(),
        ],
        check=True,
        timeout=180,
        capture_output=True,
    )
    if not output.is_file():
        raise RuntimeError(f"The browser exited without writing {output}")
    return output
