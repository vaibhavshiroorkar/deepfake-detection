"""Generate every figure of the paper, the report and the slides from code (matplotlib only; no Graphviz is installed).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/make_figures.py [--only fig01 fig07]

Output: assets/figures/<name>.png (300 dpi, for Word and slides) and assets/figures/<name>.pdf (vector, for the paper). The same files go
into all three documents. Schematic figures carry the small tag "schematic". The result figures fig08_ladder_results and fig09_lavdf_cells plot
logged measurements only: every number in them is copied from docs/EXPERIMENTS.md or docs/DOC_FACTS.md section 15 (E-004, E-005, E-007 to E-009,
E-011, E-013) and they are exploratory (development slice, validation split). fig03_ladder is a table of the measured ladder (same sources).
Figure 7 (web page) takes screenshots of the local page against the development stand-in (mock adapter) with Playwright and Chromium
(PLAYWRIGHT_BROWSERS_PATH, default D:\\cache\\playwright) and saves them in assets/screens/. Figure 8 reads parameter counts and feature
sizes from the extraction sidecars under data/_features/.
Widths: figures are made 7.0 in wide (two columns of the paper) or 3.4 in (one column); the documents scale them, never stretch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "figures"
SCREENS = ROOT / "assets" / "screens"
DPI = 300

plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, LINE = "#1b1f24", "#5b6573", "#8a94a3"
C = {"in": "#e8eef7", "proc": "#dfe9f6", "model": "#d6ecd9", "out": "#fbe9c9", "warn": "#f6d9d6", "gray": "#eceff3", "dark": "#2f4b7c"}


def canvas(w, h, tag=True):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    if tag:
        ax.text(w - 0.05, 0.05, "schematic", ha="right", va="bottom", fontsize=9, color=MUTED, style="italic")
    return fig, ax


def box(ax, x, y, w, h, text, fc=C["proc"], fs=9.5, bold=False, ec=LINE, lw=0.8, color=INK, round_=0.04, align="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2 if align == "center" else x + 0.06, y + h / 2, text, ha=align, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.15, wrap=True)
    return (x, y, w, h)


def arrow(ax, a, b, color=INK, lw=0.9, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def save(fig, name):
    """Save as 300 dpi PNG and vector PDF, cropped to the drawn content (the canvas is larger than the drawing)."""
    from matplotlib.transforms import Bbox
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    bb = Bbox.union([a.get_window_extent(r) for a in list(ax.texts) + list(ax.patches)])
    crop = bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.06)
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches=crop)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=crop)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1 pipeline
# Design rule (D-053 item 5, reset 2026-10-25): every schematic is drawn 6.1 in wide, the width of the report text block, with all text at 9 pt or more.
# The paper embeds it at the full text width (7.0 in, text 10.3 pt or more) and the report at 6.1 in (no scaling), the slides at 8 in or more.
WD = 6.1
FS = 9.5


def fig01():
    fig, ax = canvas(WD, 4.6)
    ax.text(0.05, 4.25, "threshold fixed on validation data;  encoders frozen;  scores calibrated on that data only", fontsize=9, color=MUTED)
    G, X0, SX, SW = 0.1, 0.05, 5.1, 0.95

    def row(y, label, texts, score_text):
        ax.text(X0, y + 0.74, label, fontsize=FS, fontweight="bold", color=C["dark"])
        n = len(texts)
        W = (SX - X0 - G - (n - 1) * G) / n
        prev, x = None, X0
        for t in texts:
            green = t.startswith(("Frozen", "Linear"))
            b = box(ax, x, y, W, 0.62, t, fc=C["model"] if green else C["proc"], fs=FS)
            if prev:
                arrow(ax, right(prev), left(b))
            prev, x = b, x + W + G
        sc = box(ax, SX, y, SW, 0.62, score_text, fc=C["out"], fs=FS, bold=True)
        arrow(ax, right(prev), left(sc))
        return sc
    sv = row(3.0, "Video, visual branch", ["16 frames", "Face\ncrop 224", "Frozen\nencoder", "Linear\nhead", "Mean over\nframes", "Temp.\nscaling"], "Visual\nscore")
    sa = row(1.75, "Audio branch (audio file, or audio track of a video)", ["First 4 s,\n16 kHz", "Frozen\nencoder", "Mean, std\nover time", "Linear\nhead", "Temp.\nscaling"], "Audio\nscore")
    row(0.4, "Image", ["Face found?", "Crop, or\nwhole image", "Frozen\nencoder", "Linear\nhead", "Temp.\nscaling"], "Image\nscore")
    tg = box(ax, SX, 2.5, SW, 0.4, "side by side", fc=C["gray"], fs=9)
    arrow(ax, bottom(sv), top(tg), ls="--", color=MUTED, lw=0.7)
    arrow(ax, top(sa), bottom(tg), ls="--", color=MUTED, lw=0.7)
    save(fig, "fig01_pipeline")


# ------------------------------------------------------------------ 2 splits
def fig02():
    fig, ax = canvas(WD, 4.2)
    box(ax, 0.05, 0.9, 2.75, 2.85, "", fc=C["in"], round_=0.06)
    ax.text(1.42, 3.5, "Training dataset\n(for example FaceForensics++)", ha="center", va="center", fontsize=FS, fontweight="bold", linespacing=1.15)
    box(ax, 0.13, 1.95, 1.0, 1.0, "train", fc=C["model"], fs=10.5, bold=True)
    box(ax, 1.2, 1.95, 1.52, 1.0, "validation:\nthreshold,\ntemperature,\nearly stopping", fc=C["model"], fs=FS)
    box(ax, 0.13, 1.0, 2.59, 0.85, "test_in (in-dataset,\nsealed, not the headline)", fc=C["warn"], fs=FS)
    box(ax, 3.0, 0.9, 3.05, 2.85, "", fc=C["gray"], round_=0.06)
    ax.text(4.52, 3.5, "Cross-dataset source\n(for example Celeb-DF v2)", ha="center", va="center", fontsize=FS, fontweight="bold", linespacing=1.15)
    box(ax, 3.07, 1.95, 1.45, 1.0, "dev_x: ranks\ncandidates,\nmay be reused", fc=C["out"], fs=FS)
    box(ax, 4.58, 1.95, 1.4, 1.0, "test_x: sealed,\nonce per frozen\niteration", fc=C["warn"], fs=FS)
    box(ax, 3.07, 1.0, 2.91, 0.85, "dev_x and test_x share no group,\nidentity or file content", fc="#ffffff", fs=FS)
    ax.annotate("", xy=(3.0, 2.45), xytext=(2.8, 2.45), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    ax.text(0.05, 0.45, "group = source video together with the person identity;\na family (for example Celeb-DF v2 and Celeb-DF++) counts as one source", fontsize=9, color=MUTED, va="center", linespacing=1.2)
    save(fig, "fig02_splits")


# ------------------------------------------------------------------ 3 ladder
LADDER = [  # step, lever (task), measured result, mechanical verdict; sources: docs/DOC_FACTS.md section 15
    ("I-0", "frozen encoder, linear head", "dev_x AUC 0.693 video, 0.610 speech", "baseline"),
    ("I-1", "full fine-tuning (video); I-1b, I-1c", "dev_x 0.910, gain +0.170; I-1b 0.932 and I-1c 0.919 do not enter", "supported"),
    ("I-2", "trim and normalise (speech)", "dev_x 0.672, gain +0.062", "supported"),
    ("I-3", "noise, codec, reverb copies (speech)", "dev_x 0.746, gain +0.073", "supported"),
    ("I-4", "average of three encoders (speech)", "dev_x 0.813, gain +0.068", "supported"),
    ("I-5", "frozen heads on LAV-DF, then fusion", "video head 0.384, below chance", "not tested"),
    ("I-6", "heads trained on LAV-DF, fusion", "visual head at chance within produced videos", "see text"),
    ("I-7", "rPPG signal check, positive control", "0.301 of real clips, 0.6 needed; control sensitive", "failed"),
    ("I-8", "integration and freeze", "sealed test: T1, T2 improved; T3 superior at 95 percent", "supported"),
]
LAD_COL = {"baseline": C["gray"], "supported": C["model"], "not tested": C["warn"], "see text": C["out"], "failed": C["warn"], "pending": "#ffffff"}


def fig03():
    import textwrap
    n = len(LADDER)
    rh = 0.5
    H = 0.9 + rh * n
    fig, ax = canvas(WD, H, tag=False)
    cols = [("Step", 0.05), ("Lever (task)", 0.5), ("Measured result (exploratory)", 2.85), ("Verdict", 5.15)]
    y0 = H - 0.25
    for tt, x in cols:
        ax.text(x, y0, tt, fontsize=FS, fontweight="bold", va="center", color=C["dark"])
    ax.plot([0.05, 6.05], [y0 - 0.2, y0 - 0.2], color=INK, lw=0.7)
    for k, (s, lev, res, ver) in enumerate(LADDER):
        yy = y0 - 0.2 - rh * (k + 0.55)
        ax.add_patch(plt.Rectangle((5.1, yy - 0.18), 0.95, 0.36, fc=LAD_COL[ver], ec=LINE, lw=0.5))
        ax.text(0.05, yy, s, fontsize=FS, va="center", fontweight="bold")
        ax.text(0.5, yy, textwrap.fill(lev, 31), fontsize=FS, va="center", linespacing=1.1)
        ax.text(2.85, yy, textwrap.fill(res, 27), fontsize=FS, va="center", linespacing=1.1)
        ax.text(5.575, yy, ver, fontsize=9, va="center", ha="center")
    ax.text(0.05, 0.3, "the lever of each step was chosen from the diagnostic battery of the step before;\nverdicts are mechanical and exploratory", fontsize=9, color=MUTED, va="center", linespacing=1.2)
    save(fig, "fig03_ladder")


# ------------------------------------------------------------------ 4 staircase
def fig04():
    fig, ax = canvas(WD, 3.7)
    labels = ["I-0", "I-1", "I-2", "...", "final"]  # schematic: the sealed evaluation is not yet run
    for k, lab in enumerate(labels):
        x0 = 0.1 + k * 0.93
        h = 0.9 + k * 0.25
        box(ax, x0, 0.1, 0.84, h, "", fc=C["proc"] if lab != "final" else C["model"], round_=0.02)
        ax.text(x0 + 0.42, 0.1 + h - 0.12, lab, ha="center", va="top", fontsize=10.5, fontweight="bold")
        if lab != "...":
            ax.text(x0 + 0.42, 0.18, "frozen,\nevaluated\nonce", ha="center", va="bottom", fontsize=9, color=INK, linespacing=1.15)
    ax.text(0.1, 3.45, "development slice: exploratory verdicts (sign, size, interval)", fontsize=9, color=MUTED)
    ax.text(0.1, 3.2, "sealed test sets: one evaluation per frozen iteration, in a ledger", fontsize=9, color=MUTED)
    ax.text(5.4, 3.0, "three pre-declared\nconfirmatory tests", ha="center", va="center", fontsize=9, color=MUTED, linespacing=1.2)
    c1 = box(ax, 4.8, 2.0, 1.2, 0.55, "video: final\nvs I-0", fc=C["out"], fs=10, bold=True)
    c2 = box(ax, 4.8, 1.3, 1.2, 0.55, "speech: final\nvs I-0", fc=C["out"], fs=10, bold=True)
    c3 = box(ax, 4.8, 0.6, 1.2, 0.55, "video: final\nvs baseline", fc=C["out"], fs=10, bold=True)
    arrow(ax, (4.68, 2.0), left(c1), rad=-0.1)
    arrow(ax, (4.68, 1.5), left(c2), rad=0.0)
    arrow(ax, (4.68, 1.0), left(c3), rad=0.1)
    save(fig, "fig04_staircase")


# ------------------------------------------------------------------ 5 rPPG chain
def fig05():
    fig, ax = canvas(WD, 3.2)
    steps = ["10 s of\ncontiguous\nframes", "Face and\nskin region", "Mean\nR, G, B", "Window\nnorm.", "Projection\nto pulse\nsignal",
             "Spectrum\n0.7 to 3 Hz", "Peak\nand SNR"]
    W, G = 0.79, 0.08
    xs, prev = 0.05, None
    bs = []
    for t in steps:
        b = box(ax, xs, 2.2, W, 0.85, t, fc=C["proc"], fs=9)
        if prev:
            arrow(ax, right(prev), left(b))
        prev = b
        bs.append(b)
        xs += W + G
    ctl = box(ax, 0.05, 0.4, 3.4, 0.9, "Control: the same clips with time-shuffled\nframes (pooled, at least 10 shuffles per clip)", fc=C["gray"], fs=FS)
    dec = box(ax, 3.9, 0.4, 2.1, 0.9, "Pass, or report:\npulse not recoverable", fc=C["out"], fs=FS)
    arrow(ax, bottom(bs[-1]), top(dec), rad=0.0)
    arrow(ax, right(ctl), left(dec))
    arrow(ax, (1.75, 2.2), (1.75, 1.3), color=MUTED, ls="--")
    save(fig, "fig05_rppg")


# ------------------------------------------------------------------ 6 timeline
def fig06():
    fig, ax = canvas(WD, 3.7)
    days = ["Sat 24", "Sun 25", "Mon 26", "Tue 27", "Wed 28", "Thu 29", "Fri 30"]
    x0, wd = 1.7, 0.62
    for k, d in enumerate(days):
        ax.text(x0 + k * wd + wd / 2, 3.35, d, ha="center", fontsize=9, fontweight="bold", color=C["dark"])
        ax.plot([x0 + k * wd, x0 + k * wd], [0.55, 3.2], color="#d9dee5", lw=0.5)
    ax.plot([x0 + 7 * wd] * 2, [0.55, 3.2], color="#d9dee5", lw=0.5)
    rows = [("Data, harness, decisions", 0, 1, C["in"]), ("Encoder sweep", 1, 2, C["proc"]), ("Working version", 1.5, 3, C["model"]), ("I-1 to I-3", 3, 4, C["proc"]),
            ("I-4 to I-7 (with rPPG)", 4, 5, C["proc"]), ("Integration and freeze", 5, 6, C["warn"]), ("Assembly, submission", 6, 7, C["out"])]
    for k, (t, a, b, col) in enumerate(rows):
        y = 2.85 - k * 0.34
        ax.text(0.05, y + 0.1, t, fontsize=9, va="center")
        box(ax, x0 + a * wd + 0.02, y, (b - a) * wd - 0.04, 0.2, "", fc=col, round_=0.02)
    ax.text(0.05, 0.3, "an iteration not finished by the end of Wednesday is reported as not run", fontsize=9, color=MUTED)
    save(fig, "fig06_timeline")


# ------------------------------------------------------------------ 7 web screenshots
def fig07():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", r"D:\cache\playwright")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    from dfd.infer.mock_adapter import MockAdapter
    from dfd.web.app import create_app

    tmp = Path(tempfile.mkdtemp())
    import cv2
    img = np.zeros((360, 480, 3), np.uint8)
    for k in range(480):
        img[:, k] = (60 + k // 4, 90, 200 - k // 5)
    cv2.putText(img, "synthetic test image", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(tmp / "test_image.png"), img)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(tmp / "test_clip.mp4")], check=True)
    srv = make_server("127.0.0.1", 0, create_app(MockAdapter(), upload_dir=tmp / "up"), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    SCREENS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 980, "height": 640}, device_scale_factor=2, color_scheme="light")
        pg.goto(url)
        pg.screenshot(path=str(SCREENS / "web_start.png"))
        for name, f in (("web_image", "test_image.png"), ("web_video", "test_clip.mp4")):
            pg.goto(url)
            with pg.expect_file_chooser() as fc:
                pg.click("#choose")
            fc.value.set_files(str(tmp / f))
            pg.click("#go")
            pg.wait_for_selector("#result:not([hidden])")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(SCREENS / f"{name}.png"), full_page=True)
        b.close()
    srv.shutdown()
    # composite: three panels in one figure for the documents
    from PIL import Image
    names = [("web_start.png", "(a) start page"), ("web_image.png", "(b) image result"), ("web_video.png", "(c) video result")]
    ims = [Image.open(SCREENS / n).convert("RGB") for n, _ in names]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7))
    for ax, im, (n, cap) in zip(axs, ims, names):
        ax.imshow(im)
        ax.set_title(cap, fontsize=8)
        ax.axis("off")
    fig.text(0.5, 0.01, "local web page against the development stand-in (mock adapter); the MOCK banner is part of the design", ha="center", fontsize=6.8, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07, wspace=0.04)
    save(fig, "fig07_web")


# ------------------------------------------------------------------ 8 shortlist table graphic
LICENCE = {"IMG-01": "research use (ImageNet)", "IMG-04": "research use (ImageNet)", "IMG-07": "Apache-2.0", "IMG-08": "Apache-2.0",
           "IMG-13": "research use (accepted)", "IMG-15": "MIT tag, research use", "IMG-17": "Apache-2.0", "AU-1": "Apache-2.0", "AU-4": "research use (accepted)",
           "AU-6": "Apache-2.0 / MIT", "AU-8": "Apache-2.0"}
NAMES = {"IMG-01": ("EfficientNet-B4", "supervised CNN"), "IMG-04": ("Xception", "supervised CNN"), "IMG-07": ("DINOv2 small", "self-supervised ViT"),
         "IMG-08": ("DINOv2 base", "self-supervised ViT"), "IMG-13": ("CLIP ViT-L/14 (OpenAI)", "image-text contrastive"), "IMG-15": ("CLIP ViT-L/14 (LAION)", "image-text contrastive"),
         "IMG-17": ("SigLIP base", "sigmoid image-text"), "AU-1": ("wav2vec 2.0 base", "self-supervised"), "AU-4": ("WavLM base plus", "self-supervised"),
         "AU-6": ("Whisper small encoder", "weakly supervised ASR"), "AU-8": ("HuBERT base", "self-supervised")}


def fig08():
    groups = {"Images and video frames": [], "Speech audio": []}
    for cid, (name, fam) in NAMES.items():
        task = "T-A" if cid.startswith("AU") else "T-V"
        fs = sorted((ROOT / "data" / "_features" / cid / task).glob("*.json"))
        if fs:
            m = json.loads(fs[0].read_text())
            params, dim = f"{m['params'] / 1e6:.1f} M", str(m["dim"])
        else:
            params, dim = "[pending]", "[pending]"
        groups["Images and video frames" if cid.startswith("IMG") else "Speech audio"].append((name, fam, params, dim, LICENCE[cid]))
    groups["Images and video frames"].append(("Small CNN from scratch", "reference, not ranked", "0.39 M", "n/a", "own code"))
    groups["Speech audio"].append(("Small log-mel CNN", "reference, not ranked", "0.39 M", "n/a", "own code"))
    rh = 0.27
    n = sum(len(v) for v in groups.values()) + len(groups)
    H = 1.0 + rh * n
    fig, ax = canvas(WD, H, tag=False)
    cols = [("Encoder", 0.05), ("Family", 1.85), ("Params", 3.6), ("Dim", 4.3), ("Weights licence", 4.75)]
    y = H - 0.25
    for tt, x in cols:
        ax.text(x, y, tt, fontsize=FS, fontweight="bold", va="center")
    ax.plot([0.05, 6.05], [y - 0.15, y - 0.15], color=INK, lw=0.7)
    yy = y - 0.15
    for gname, rows in groups.items():
        yy -= rh
        ax.add_patch(plt.Rectangle((0.05, yy - 0.12), 6.0, 0.24, fc="#e3e9f2", ec="none"))
        ax.text(0.1, yy, gname, fontsize=FS, fontweight="bold", va="center", color=C["dark"])
        for r in rows:
            yy -= rh
            for (tt, x), val in zip(cols, r):
                ax.text(x, yy, val, fontsize=9, va="center")
    ax.text(6.05, 0.1, "parameters and feature sizes read from the extraction records", fontsize=9, color=MUTED, ha="right")
    save(fig, "fig08_shortlist")


# ------------------------------------------------------------------ 8b / 9 result figures (measured)
def save_plain(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


# (label, estimate, low, high): docs/DOC_FACTS.md section 15 and the logged blocks of E-004, E-005, E-007, E-008, E-009, E-013b (baseline at 3e-4), E-016, E-017
LAD_V = [("I-0 frozen probe", 0.6933, 0.6639, 0.7314), ("Xception, fine-tuned", 0.8716, 0.8364, 0.9076), ("I-1 SigLIP base, fine-tuned", 0.9096, 0.8776, 0.9388),
         ("I-1b with augmentation, mixup", 0.931514, 0.905484, 0.954778), ("I-1c mean of three encoders", 0.919215, 0.894824, 0.945728)]
LAD_A = [("I-0 frozen probe", 0.6100, 0.5434, 0.7256), ("I-2 trim, normalise", 0.6723, 0.6126, 0.7584), ("I-3 plus augmentation", 0.7455, 0.7052, 0.8121), ("I-4 three encoders", 0.8130, 0.7523, 0.8808)]


def fig08_ladder_results():
    # slide 11 shows this figure at about 1.0 times its size, so every text is made 13 pt here (at least 12 pt on the slide)
    FS = 13
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.1), gridspec_kw={"height_ratios": [len(LAD_V), len(LAD_A)], "hspace": 0.62})
    for ax, rows, title, col in ((axs[0], LAD_V, "Video: Celeb-DF v2 development slice (15 identities)", C["dark"]),
                                 (axs[1], LAD_A, "Speech: In-the-Wild development slice (12 speakers)", C["dark"])):
        ys = list(range(len(rows)))[::-1]
        for y, (lab, est, lo, hi) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], color=col, lw=1.6)
            ax.plot([est], [y], "o", color=col, ms=5)
            ax.text(hi + 0.01, y, f"{est:.3f}", va="center", fontsize=FS, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
        ax.set_xlim(0.5, 1.03)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.axvline(0.5, color=LINE, lw=0.6)
        ax.set_title(title, fontsize=FS, loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=FS)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[1].set_xlabel("AUC with 95 percent group-bootstrap interval\n(exploratory, development slice, mean of 3 seeds)", fontsize=FS)
    save_plain(fig, "fig08_ladder_results")


# LAV-DF validation, heads trained on LAV-DF (E-011 block and its post-hoc control in docs/EXPERIMENTS.md)
CELLS = [("video fake only", (0.9905, 0.9857, 0.9945), (0.5175, 0.4836, 0.5532)), ("audio fake only", (0.9912, 0.9864, 0.9951), (0.8711, 0.8474, 0.8920)),
         ("both fake", (0.9921, 0.9878, 0.9959), (0.8556, 0.8327, 0.8779))]
WITHIN = (("within produced videos", (0.5070, 0.4771, 0.5361), (0.8574, 0.8394, 0.8752)),)


def fig09_lavdf_cells():
    fig, ax = plt.subplots(figsize=(WD, 3.6))
    groups = [(c[0], c[1], c[2]) for c in CELLS] + [(w[0], w[1], w[2]) for w in WITHIN]
    xs = list(range(len(groups)))
    wd = 0.34
    for k, (lab, v, a) in enumerate(groups):
        for off, (est, lo, hi), col, nm in ((-wd / 2, v, "#c8553d", "visual head"), (wd / 2, a, C["dark"], "audio head")):
            ax.bar(k + off, est, wd * 0.92, color=col, label=nm if k == 0 else None, alpha=0.9)
            ax.plot([k + off, k + off], [lo, hi], color=INK, lw=1.0)
            ax.text(k + off, hi + 0.02, f"{est:.2f}", ha="center", fontsize=FS)
    ax.axhline(0.5, color=LINE, lw=0.7, ls="--")
    ax.axvline(len(CELLS) - 0.5, color=LINE, lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["video fake\nonly vs real\noriginals", "audio fake\nonly vs real\noriginals", "both fake\nvs real\noriginals",
                        "within\nproduced videos\n(own label)"], fontsize=FS)
    ax.set_ylim(0.4, 1.12)
    ax.set_ylabel("AUC, 95 percent interval", fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.legend(fontsize=FS, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.08))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_plain(fig, "fig09_lavdf_cells")


def fig07_real():
    """Three real-checkpoint screenshots (assets/screens/real_*.png, E-004 page check) side by side, labelled inside the image, not cropped."""
    from PIL import Image
    names = [("real_image.png", "(a) image"), ("real_video.png", "(b) video with audio"), ("real_audio.png", "(c) audio file")]
    ims = [Image.open(SCREENS / n).convert("RGB") for n, _ in names]
    hmax = max(i.size[1] for i in ims)
    fig, axs = plt.subplots(1, 3, figsize=(WD, 3.45))
    for ax, im, (n, cap) in zip(axs, ims, names):
        canvas_im = Image.new("RGB", (im.size[0], hmax + 110), (255, 255, 255))
        canvas_im.paste(im, (0, 110))
        ax.imshow(canvas_im)
        ax.text(0.02, 0.995, cap, transform=ax.transAxes, fontsize=FS, fontweight="bold", va="top", ha="left", color=INK)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.03)
    save_plain(fig, "fig07_web_real")


FIGS = {"fig07r": fig07_real, "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04, "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08, "fig10": fig08_ladder_results, "fig11": fig09_lavdf_cells}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    for k in ap.parse_args().only:
        FIGS[k]()
