"""Figure of the redesigned dashboard for the report and slides: the top of a real live report (2026-10-27 browser run, demonstration input 3),
cropped from docs/test_evidence/2026-10-27_redesign/screens/. A screenshot of the software, not a result.

    .venv/Scripts/python.exe paper/figures/make_dashboard_figure.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "test_evidence" / "2026-10-27_redesign" / "screens" / "3_talking_head_with_audio_lavdf_train_desktop.png"
OUT = Path(__file__).resolve().parent / "fig11_dashboard.png"


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    w, _ = im.size
    crop = im.crop((90, 360, w - 90, 1325))          # report overview, summary and the first group of checks (pixel rows of the 1280 px wide page)
    crop.save(OUT, optimize=True)
    print(OUT, crop.size)


if __name__ == "__main__":
    main()
