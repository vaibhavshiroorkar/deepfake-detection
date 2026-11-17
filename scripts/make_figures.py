"""Draw the deliverable figures from results/numbers.json.

Every value comes from numbers.json, so a figure agrees with the text that cites
it. A figure whose numbers do not exist yet is skipped and named, never drawn
with gaps. One style for all figures: blue, orange and aqua mark the face, voice
and lip-sync checks and nothing else (checked with the dataviz validator); greys mark
everything else. Every bar carries its value, which also covers aqua's low contrast
on white. Labels are plain English.

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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # face, voice and lip-sync checks only
GREY_LIGHT, GREY_MID, GREY_DARK = "#c9c7c1", "#8a8984", "#3d3c39"   # everything that is not a check
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


def _bars(ax, positions, heights, color, width, label, clear_of=None):
    """Bars with their values on top. `clear_of`: a reference line the value labels must not sit on."""
    bars = ax.bar(positions, heights, width=width, color=color, label=label, edgecolor="white", linewidth=1.5)
    for bar, height in zip(bars, heights, strict=True):
        y = height + 0.01
        if clear_of is not None and abs(y - clear_of) < 0.03:
            y = clear_of + 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{height:.3f}", ha="center", va="bottom",
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
    _bars(ax, [i - width / 2 for i in range(len(shown))], raw, GREY_LIGHT, width, "Original files")
    neutral = [(i, value(f"audit.{k}.neutral.clip")) for i, (_, k) in enumerate(shown) if have(f"audit.{k}.neutral.clip")]
    if neutral:
        _bars(ax, [i + width / 2 for i, _ in neutral], [v for _, v in neutral], GREY_DARK, width, "After cleaning")
    ax.axhline(0.60, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, label="Our bar (0.60)")
    ax.axhline(0.5, color=MUTED, linewidth=0.6)
    ax.set_xticks(range(len(shown)), [n for n, _ in shown])
    _finish(ax, "File-clue score (AUC)", (0.4, 1.1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return _save(fig, "fig_census.png")


def h1() -> str | None:
    stages = [("Original files", "audit.fakeavceleb.raw"), ("First cleaning", "audit.fakeavceleb.v1"),
              ("Final cleaning,\n600-clip check", "audit.fakeavceleb.v4check"), ("Final cleaning", "audit.fakeavceleb.neutral")]
    stages = [(n, k) for n, k in stages if have(f"{k}.clip", f"{k}.video", f"{k}.audio")]
    if not stages:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.26
    for offset, (label, color, field) in enumerate([("Real or fake", GREY_DARK, "clip"), ("Fake face", BLUE, "video"),
                                                    ("Fake voice", ORANGE, "audio")]):
        _bars(ax, [i + (offset - 1) * width for i in range(len(stages))],
              [value(f"{k}.{field}") for _, k in stages], color, width, label)
    ax.axhline(0.60, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xticks(range(len(stages)), [n for n, _ in stages])
    _finish(ax, "File-clue score on FakeAVCeleb (AUC)", (0.4, 1.1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return _save(fig, "fig_h1.png")


METHOD_NAMES = {"rtvc": "Voice cloning", "wav2lip": "Wav2Lip", "faceswap-wav2lip": "FaceSwap + Wav2Lip",
                "fsgan-wav2lip": "FSGAN + Wav2Lip", "fsgan": "FSGAN (face only)", "faceswap": "FaceSwap (face only)"}


def sync_methods() -> str | None:
    methods = [m for m in METHOD_NAMES if have(*(f"sync.test.seed{s}.method.{m}" for s in (17, 29, 43)))]
    if not methods:
        return None
    means = {m: sum(value(f"sync.test.seed{s}.method.{m}") for s in (17, 29, 43)) / 3 for m in methods}
    methods.sort(key=lambda m: means[m])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, m in enumerate(methods):
        ax.barh(i, means[m], color=AQUA, height=0.55, edgecolor="white", linewidth=1.5)
        seeds = [value(f"sync.test.seed{s}.method.{m}") for s in (17, 29, 43)]
        ax.scatter(seeds, [i] * 3, s=14, color=INK, zorder=3, label="Single seed" if i == 0 else None)
        ax.text(max(seeds) + 0.012, i, f"{means[m]:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=MUTED, linewidth=0.6)
    ax.set_yticks(range(len(methods)), [METHOD_NAMES[m] for m in methods])
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("Lip-sync check (AUC) on the cleaned FakeAVCeleb test set; bar: mean of 3 seeds")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _save(fig, "fig_sync_methods.png")


def cross_corpus() -> str | None:
    targets = [("FakeAVCeleb\n(trained on)", "test"), ("LAV-DF\n(file label)", "lavdf-test"),
               ("LAV-DF\n(window label)", "lavdf-window"), ("DFDC", "dfdc"), ("HiFi-AVDF", "hifiavdf")]
    targets = [(n, t) for n, t in targets if have(*(f"sync.{t}.seed{s}.auc" for s in (17, 29, 43)))]
    if len(targets) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    means = [sum(value(f"sync.{t}.seed{s}.auc") for s in (17, 29, 43)) / 3 for _, t in targets]
    ax.bar(range(len(targets)), means, width=0.55, color=AQUA, edgecolor="white", linewidth=1.5)
    for i, (_, t) in enumerate(targets):
        seeds = [value(f"sync.{t}.seed{s}.auc") for s in (17, 29, 43)]
        ax.scatter([i] * 3, seeds, s=14, color=INK, zorder=3)
        # The mean sits above the highest seed so the dots never cover it.
        ax.text(i, max(seeds) + 0.02, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=0.6)
    ax.set_xticks(range(len(targets)), [n for n, _ in targets])
    _finish(ax, "Lip-sync check (AUC; dots: single seeds)", (0.3, 1.05))
    return _save(fig, "fig_cross_corpus.png")


def h4() -> str | None:
    targets = [("LAV-DF, file", "lavdf-test"), ("LAV-DF, window", "lavdf-window"), ("HiFi-AVDF", "hifiavdf")]
    rules = [("Threshold from training", "source_validation", GREY_LIGHT),
             ("From 200 real clips", "conformal_real_only", GREY_MID),
             ("Stricter, from 200 real clips", "umbrella_real_only", GREY_DARK)]
    targets = [(n, t) for n, t in targets if have(*(f"h4.{t}-seed{s}.{r}.fpr" for s in (17, 29, 43) for _, r, _ in rules))]
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.26
    for offset, (label, rule, color) in enumerate(rules):
        heights = [sum(value(f"h4.{t}-seed{s}.{rule}.fpr") for s in (17, 29, 43)) / 3 for _, t in targets]
        _bars(ax, [i + (offset - 1) * width for i in range(len(targets))], heights, color, width, label,
              clear_of=0.05)
    ax.axhline(0.05, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, label="Target: 5%")
    ax.set_xticks(range(len(targets)), [n for n, _ in targets])
    _finish(ax, "Real clips wrongly flagged\n(mean of 3 seeds)", (0, 0.6))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8)
    return _save(fig, "fig_h4.png")


def overflowing(fig, pairs) -> list[str]:
    """The labels whose drawn text leaves the box they belong to, as drawn in `fig`."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for text, patch in pairs:
        inner, outer = text.get_window_extent(renderer), patch.get_window_extent(renderer)
        pad = 0.04 * fig.dpi       # 1 mm of breathing room inside the box border
        if inner.x0 < outer.x0 + pad or inner.x1 > outer.x1 - pad or inner.y0 < outer.y0 + pad or inner.y1 > outer.y1 - pad:
            out.append(" ".join(text.get_text().split()))
    return out


def pipeline() -> str:
    fig, pairs = draw_pipeline()
    spilled = overflowing(fig, pairs)
    if spilled:
        raise ValueError(f"method figure: text leaves its box: {spilled}")
    return _save(fig, "fig_pipeline.png")


def draw_pipeline():
    """The method in five stages, with the file audit as the evaluation step above them.

    Drawn to read as a designed figure: one grid, thin strokes, short plain labels (what each
    stage does, the model name small below), colour only where it carries identity (face,
    voice and lip sync keep the colours they have in the result figures).
    """
    raw, clean = value("audit.fakeavceleb.raw.clip"), value("audit.fakeavceleb.neutral.clip")
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 47)
    ax.axis("off")
    edge = "#8a8984"
    pairs = []

    def box(x0, x1, y0, y1, title, detail, face="white", stroke=edge, dashed=False):
        patch = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                               facecolor=face, edgecolor=stroke, linewidth=0.9,
                               linestyle=(0, (3, 2)) if dashed else "solid")
        ax.add_patch(patch)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pairs.append((ax.text(cx, cy + (1.6 if detail else 0), title, ha="center", va="center", fontsize=8.5,
                              fontweight="bold", color=INK), patch))
        if detail:
            pairs.append((ax.text(cx, cy - 2.1, detail, ha="center", va="center", fontsize=6.8, color=MUTED,
                                  linespacing=1.25), patch))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8, "mutation_scale": 7,
                                "shrinkA": 0, "shrinkB": 0})

    cols = {"input": (0, 11), "clean": (14.5, 37), "checks": (41, 66), "compare": (70.5, 84), "answer": (87.5, 100)}
    mid, half = 18, 7
    box(*cols["input"], mid - half, mid + half, "Video", "one person\nspeaking")
    box(*cols["clean"], mid - half, mid + half, "Clean like an upload",
        "re-encode, keep 3 s of\nspeech, level the volume")
    checks = [("Face", "edited faces\nEfficientNet-B0", "#e8f0fb", BLUE, 31),
              ("Voice", "synthetic speech\nWav2Vec2", "#fdeee7", ORANGE, 18),
              ("Lip sync", "do lips match the words?\nAV-HuBERT, real videos only", "#e3f5ee", AQUA, 5)]
    for title, detail, face, stroke, cy in checks:
        box(*cols["checks"], cy - 5, cy + 5, title, detail, face=face, stroke=stroke)
        arrow(cols["clean"][1], mid, cols["checks"][0], cy)
        arrow(cols["checks"][1], cy, cols["compare"][0], mid)
    box(*cols["compare"], mid - half, mid + half, "Compare", "with scores of\nreal videos")
    box(*cols["answer"], mid - half, mid + half, "Answer", "signs, can't tell,\nor no signs")
    arrow(cols["input"][1], mid, cols["clean"][0], mid)
    arrow(cols["compare"][1], mid, cols["answer"][0], mid)

    # The file audit: an evaluation step, run on the raw and on the cleaned files.
    box(0, 66, 37, 46, "File audit (evaluation only)",
        f"what file details alone reveal: {raw:.3f} on the original files, {clean:.3f} after cleaning")
    for x in (5.5, 25.75):
        arrow(x, mid + half, x, 37)

    for (x0, x1), label in zip(cols.values(), ["1  Input", "2  Clean", "3  Check", "4  Compare", "5  Answer"],
                               strict=True):
        ax.text((x0 + x1) / 2, -4.5, label, ha="center", va="center", fontsize=7.5, color=MUTED)
    return fig, pairs


def _save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    return name


def h2() -> str | None:
    """Fusion minus visual-only AUC per target: the 3-seed mean with its interval, and each seed as a dot."""
    targets = [("FakeAVCeleb (unseen people)", "test-identity-strict"), ("DFDC", "dfdc"), ("LAV-DF", "lavdf-window"),
               ("HiFi-AVDF (outside the test)", "hifiavdf")]
    targets = [(n, t) for n, t in targets if have(f"h2.{t}.estimate", f"h2.{t}.lower", f"h2.{t}.upper")]
    if not targets:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, (_, t) in enumerate(targets):
        y = len(targets) - 1 - i
        est, lo, hi = value(f"h2.{t}.estimate"), value(f"h2.{t}.lower"), value(f"h2.{t}.upper")
        ax.plot([lo, hi], [y, y], color=GREY_DARK, linewidth=2, solid_capstyle="round")
        ax.scatter([est], [y], s=36, color=GREY_DARK, zorder=3, edgecolor="white", linewidth=1.5)
        seeds = [value(f"h2.{t}.seed{s}") for s in (17, 29, 43) if have(f"h2.{t}.seed{s}")]
        ax.scatter(seeds, [y + 0.22] * len(seeds), s=10, color=INK, zorder=3)
        ax.text(hi, y, f"  {est:+.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(targets)), [n for n, _ in reversed(targets)])
    ax.set_xlabel("Gain from combining the three checks over the face check alone (AUC)\n"
                  "mean of 3 seeds with its 95% interval; dots: single seeds")
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
