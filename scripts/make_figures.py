"""Draw the deliverable figures from results/numbers.json.

Every value comes from numbers.json, so a figure agrees with the text that cites
it. A figure whose numbers do not exist yet is skipped and named, never drawn
with gaps. Colors: the reference categorical palette (blue, orange, aqua),
checked with the dataviz validator. Every bar carries its value, which also
covers aqua's low contrast on white.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
NUMBERS = json.loads((ROOT / "results" / "numbers.json").read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def value(key: str):
    entry = NUMBERS.get(key)
    return None if entry is None else entry["value"]


def have(*keys: str) -> bool:
    return all(k in NUMBERS for k in keys)


def _bars(ax, positions, heights, color, width, label):
    bars = ax.bar(positions, heights, width=width, color=color, label=label, edgecolor="white", linewidth=1.5)
    for bar, height in zip(bars, heights, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK)


def _finish(ax, ylabel, ylim=(0.4, 1.05)):
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def census() -> str | None:
    corpora = [("FakeAVCeleb", "fakeavceleb"), ("LAV-DF", "lavdf"), ("DFDC", "dfdc"), ("HiFi-AVDF", "hifiavdf")]
    shown = [(name, key) for name, key in corpora if have(f"audit.{key}.raw.clip")]
    if not shown:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.36
    raw = [value(f"audit.{k}.raw.clip") for _, k in shown]
    _bars(ax, [i - width / 2 for i in range(len(shown))], raw, BLUE, width, "Raw files")
    neutral = [(i, value(f"audit.{k}.neutral.clip")) for i, (_, k) in enumerate(shown) if have(f"audit.{k}.neutral.clip")]
    if neutral:
        _bars(ax, [i + width / 2 for i, _ in neutral], [v for _, v in neutral], ORANGE, width, "Neutralized (v4)")
    ax.axhline(0.60, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, label="Preregistered bar (0.60)")
    ax.axhline(0.5, color=MUTED, linewidth=0.6)
    ax.set_xticks(range(len(shown)), [n for n, _ in shown])
    _finish(ax, "Shortcut ceiling (AUC, clip label)", (0.4, 1.1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return _save(fig, "fig_census.png")


def h1() -> str | None:
    stages = [("Raw", "audit.fakeavceleb.raw"), ("Neutralizer v1", "audit.fakeavceleb.v1"),
              ("v4, 600-clip check", "audit.fakeavceleb.v4check"), ("v4, full", "audit.fakeavceleb.neutral")]
    stages = [(n, k) for n, k in stages if have(f"{k}.clip", f"{k}.video", f"{k}.audio")]
    if not stages:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.26
    for offset, (label, color, field) in enumerate([("Real vs fake", BLUE, "clip"), ("Fake video", ORANGE, "video"),
                                                    ("Fake audio", AQUA, "audio")]):
        _bars(ax, [i + (offset - 1) * width for i in range(len(stages))],
              [value(f"{k}.{field}") for _, k in stages], color, width, label)
    ax.axhline(0.60, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xticks(range(len(stages)), [n for n, _ in stages])
    _finish(ax, "Shortcut ceiling on FakeAVCeleb (AUC)", (0.4, 1.1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return _save(fig, "fig_h1.png")


METHOD_NAMES = {"rtvc": "Voice cloning (rtvc)", "wav2lip": "Wav2Lip", "faceswap-wav2lip": "FaceSwap + Wav2Lip",
                "fsgan-wav2lip": "FSGAN + Wav2Lip", "fsgan": "FSGAN (face only)", "faceswap": "FaceSwap (face only)"}


def sync_methods() -> str | None:
    methods = [m for m in METHOD_NAMES if have(*(f"sync.test.seed{s}.method.{m}" for s in (17, 29, 43)))]
    if not methods:
        return None
    means = {m: sum(value(f"sync.test.seed{s}.method.{m}") for s in (17, 29, 43)) / 3 for m in methods}
    methods.sort(key=lambda m: means[m])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, m in enumerate(methods):
        ax.barh(i, means[m], color=BLUE, height=0.55, edgecolor="white", linewidth=1.5)
        seeds = [value(f"sync.test.seed{s}.method.{m}") for s in (17, 29, 43)]
        ax.scatter(seeds, [i] * 3, s=14, color=INK, zorder=3, label="Single seed" if i == 0 else None)
        ax.text(max(seeds) + 0.012, i, f"{means[m]:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=MUTED, linewidth=0.6)
    ax.set_yticks(range(len(methods)), [METHOD_NAMES[m] for m in methods])
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("Sync path ROC-AUC on neutral FakeAVCeleb test (bar: mean of 3 seeds)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _save(fig, "fig_sync_methods.png")


def cross_corpus() -> str | None:
    targets = [("FakeAVCeleb (in-domain)", "test"), ("LAV-DF", "lavdf-test"), ("DFDC", "dfdc"), ("HiFi-AVDF", "hifiavdf")]
    targets = [(n, t) for n, t in targets if have(*(f"sync.{t}.seed{s}.auc" for s in (17, 29, 43)))]
    if len(targets) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    means = [sum(value(f"sync.{t}.seed{s}.auc") for s in (17, 29, 43)) / 3 for _, t in targets]
    _bars(ax, range(len(targets)), means, BLUE, 0.55, "Mean of 3 seeds")
    for i, (_, t) in enumerate(targets):
        ax.scatter([i] * 3, [value(f"sync.{t}.seed{s}.auc") for s in (17, 29, 43)], s=14, color=INK, zorder=3)
    ax.axhline(0.5, color=MUTED, linewidth=0.6)
    ax.set_xticks(range(len(targets)), [n for n, _ in targets])
    _finish(ax, "Sync path ROC-AUC (dots: single seeds)", (0.3, 1.05))
    return _save(fig, "fig_cross_corpus.png")


def h4() -> str | None:
    targets = [("LAV-DF", "lavdf-test"), ("HiFi-AVDF", "hifiavdf")]
    rules = [("Source threshold", "source_validation", BLUE), ("Real-only conformal", "conformal_real_only", ORANGE),
             ("Real-only umbrella", "umbrella_real_only", AQUA)]
    targets = [(n, t) for n, t in targets if have(*(f"h4.{t}-seed{s}.{r}.fpr" for s in (17, 29, 43) for _, r, _ in rules))]
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.26
    for offset, (label, rule, color) in enumerate(rules):
        heights = [sum(value(f"h4.{t}-seed{s}.{rule}.fpr") for s in (17, 29, 43)) / 3 for _, t in targets]
        _bars(ax, [i + (offset - 1) * width for i in range(len(targets))], heights, color, width, label)
    ax.axhline(0.05, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(len(targets) - 0.5, 0.055, "target alpha 0.05", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(targets)), [n for n, _ in targets])
    _finish(ax, "False-positive rate on real clips (mean of 3 seeds)", (0, 1.0))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return _save(fig, "fig_h4.png")


def pipeline() -> str:
    steps = ["Clip", "Neutralize (v4)\none codec, 3 s,\nlevel, dither", "Mouth crops\nand 16 kHz\naudio",
             "Frozen\nAV-HuBERT\nlips / audio", "Alignment\nhead (real\nclips only)", "Threshold:\nvalidation or\nreal-only"]
    fig, ax = plt.subplots(figsize=(8.4, 1.7))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, text in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.15), 0.88, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="#eef4fc", edgecolor=BLUE, linewidth=1.2))
        ax.text(i + 0.5, 0.5, text, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.3)
        if i:
            ax.annotate("", xy=(i + 0.06, 0.5), xytext=(i - 0.06, 0.5), arrowprops={"arrowstyle": "->", "color": MUTED})
    return _save(fig, "fig_pipeline.png")


def _save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    return name


def h2() -> str | None:
    """Fusion minus visual-only AUC per target: the 3-seed mean with its interval, and each seed as a dot."""
    targets = [("FakeAVCeleb, strict", "test-identity-strict"), ("DFDC", "dfdc"), ("LAV-DF", "lavdf-test"),
               ("HiFi-AVDF (reported)", "hifiavdf")]
    targets = [(n, t) for n, t in targets if have(f"h2.{t}.estimate", f"h2.{t}.lower", f"h2.{t}.upper")]
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, (_, t) in enumerate(targets):
        y = len(targets) - 1 - i
        est, lo, hi = value(f"h2.{t}.estimate"), value(f"h2.{t}.lower"), value(f"h2.{t}.upper")
        ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.scatter([est], [y], s=36, color=BLUE, zorder=3, edgecolor="white", linewidth=1.5)
        seeds = [value(f"h2.{t}.seed{s}") for s in (17, 29, 43) if have(f"h2.{t}.seed{s}")]
        ax.scatter(seeds, [y + 0.22] * len(seeds), s=10, color=INK, zorder=3)
        ax.text(hi, y, f"  {est:+.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(targets)), [n for n, _ in reversed(targets)])
    ax.set_xlabel("AUC(fusion) - AUC(visual), mean of 3 seeds with 95% interval (dots: single seeds)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, "fig_h2.png")


def main() -> int:
    for build in (pipeline, census, h1, sync_methods, cross_corpus, h4, h2):
        made = build()
        print(f"{build.__name__:14} {'-> ' + made if made else 'skipped: numbers not produced yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
