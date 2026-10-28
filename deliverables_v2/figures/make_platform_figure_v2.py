"""Schematic of the analysis platform for the v2 deliverables (copy of paper/figures/make_platform_figure.py, updated to the code of
2026-10-28: eight modules incl. generated_image (D-083), schema 2.1 with the per-module assessment (D-087)). matplotlib only.

    .venv/Scripts/python.exe deliverables_v2/figures/make_platform_figure_v2.py

Output: deliverables_v2/figures/fig_platform_v2.pdf (vector) and .png (300 dpi). Every label is read from the code: module ids from the module classes, the three sections and the six statuses from
src/dfd/platform/schema.py, the coordinator behaviour from src/dfd/platform/coordinator.py. No measured number appears in the figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, LINE = "#1b1f24", "#5b6573", "#8a94a3"
FC = {"in": "#e8eef7", "proc": "#dfe9f6", "model": "#d6ecd9", "exp": "#fbe9c9", "lim": "#e9e2f3","prov": "#eceff3", "out": "#f6d9d6", "white": "#ffffff"}


def box(ax, x, y, w, h, text, fc, fs=9.0, bold=False, ec=LINE, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=0.9, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=INK, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=INK))


def main():
    W, H = 7.0, 3.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W - 0.05, H - 0.05, "schematic", ha="right", va="top", fontsize=8, color=MUTED, style="italic")

    # left column: input, probe, coordinator
    box(ax, 0.08, 3.05, 1.45, 0.62, "Uploaded file\n(image, video, audio)", FC["in"])
    box(ax, 0.08, 2.05, 1.45, 0.72, "Validate and probe\n(type list, decode\ncheck, ffprobe)", FC["proc"])
    box(ax, 0.08, 0.35, 1.45, 1.42, "Coordinator\n\napplicability first\ntimeout per module\nfailure isolation\none module at a time", FC["proc"], bold=False)
    arrow(ax, 0.80, 3.05, 0.80, 2.78)
    arrow(ax, 0.80, 2.05, 0.80, 1.78)

    # middle: three sections with their modules
    x0, sw = 1.85, 3.05
    ax.add_patch(FancyBboxPatch((x0, 1.62), sw, 2.05, boxstyle="round,pad=0,rounding_size=0.05", fc=FC["white"], ec=LINE, lw=0.9))
    ax.text(x0 + 0.08, 3.52, "authenticity", fontsize=9, fontweight="bold", color=INK, va="center")
    bw, bh = 1.42, 0.46
    box(ax, x0 + 0.08, 2.94, bw, bh, "face_video\nfrozen video head", FC["model"], fs=8.5)
    box(ax, x0 + 1.55, 2.94, bw, bh, "image_detector\nfrozen image path", FC["model"], fs=8.5)
    box(ax, x0 + 0.08, 2.40, bw, bh, "speech\nfrozen speech average", FC["model"], fs=8.5)
    box(ax, x0 + 1.55, 2.40, bw, bh, "generated_video\nexperimental, no finding", FC["exp"], fs=8.0)
    box(ax, x0 + 0.08, 1.86, bw, bh, "generated_image\nexperimental, no finding", FC["exp"], fs=8.0)
    ax.text(x0 + 1.58, 2.09, "the frozen detectors share\none adapter and one\nanalysis per file", fontsize=7.2, color=MUTED, va="center")

    ax.add_patch(FancyBboxPatch((x0, 0.98), sw, 0.52, boxstyle="round,pad=0,rounding_size=0.05", fc=FC["white"], ec=LINE, lw=0.9))
    ax.text(x0 + 0.08, 1.24, "cross_modal", fontsize=9, fontweight="bold", color=INK, va="center")
    box(ax, x0 + 1.10, 1.03, 1.87, 0.42, "lip_sync: SyncNet offset band", FC["lim"], fs=8.5)

    ax.add_patch(FancyBboxPatch((x0, 0.35), sw, 0.52, boxstyle="round,pad=0,rounding_size=0.05", fc=FC["white"], ec=LINE, lw=0.9))
    ax.text(x0 + 0.08, 0.61, "provenance", fontsize=9, fontweight="bold", color=INK, va="center")
    box(ax, x0 + 0.95, 0.40, 0.95, 0.42, "media_metadata", FC["prov"], fs=8.5)
    box(ax, x0 + 1.97, 0.40, 1.00, 0.42, "content_credentials\n(C2PA)", FC["prov"], fs=8.0)
    for y in (2.65, 1.24, 0.61):
        arrow(ax, 1.53, min(y, 1.6), x0, y)

    # right column: report and dashboard
    rx = 5.22
    box(ax, rx, 1.30, 1.70, 2.37, "Report, schema 2.1\n\none entry per module:\nstatus, finding, score\nwith its meaning,\nverdict label and\nuncertainty, model,\ntested scope, limits\n\nno overall score\nor verdict", FC["out"])
    box(ax, rx, 0.52, 1.70, 0.55, "Dashboard, 127.0.0.1\nand Evaluation Lab page", FC["in"])
    arrow(ax, x0 + sw, 2.40, rx, 2.40)
    arrow(ax, rx + 0.85, 1.30, rx + 0.85, 1.07)
    ax.text(rx + 0.85, 0.24, "module statuses: completed, not_applicable,\nunsupported, failed, timed_out, skipped", fontsize=6.8, color=MUTED,
            ha="center", va="center")
    # colour key: what each module is (not its verdict tier; D-087 tiers are in the module table of each document)
    for i, (k, t) in enumerate([("model", "frozen detector model"), ("lim", "timing check, weak evidence"), ("exp", "experimental, no finding"),
                                ("prov", "provenance")]):
        ax.add_patch(FancyBboxPatch((0.08 + i * 1.28, 0.08), 0.14, 0.12, boxstyle="round,pad=0,rounding_size=0.02", fc=FC[k], ec=LINE, lw=0.6))
        ax.text(0.27 + i * 1.28, 0.14, t, fontsize=6.8, color=MUTED, va="center")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_platform_v2.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
