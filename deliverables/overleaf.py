"""Package the report and the paper for Overleaf: one zip each, ready to upload as a new project.

    .venv/Scripts/python.exe deliverables/overleaf.py            stops on a missing number
    .venv/Scripts/python.exe deliverables/overleaf.py --draft    prints a missing number as [pending]

Each zip holds main.tex with every {{key}} filled from results/numbers.json, the figures it uses and
the college logo, all in one folder. Output: deliverables/build/overleaf/{report,paper}.zip.
In Overleaf: New Project, Upload Project, pick the zip, then Recompile (pdfLaTeX).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fill_numbers import PATTERN, render  # noqa: E402

FIG = ROOT / "deliverables" / "figures"
OUT = ROOT / "deliverables" / "build" / "overleaf"
LOGO = "1777459645157_kjsce.png"          # the name the college template uses
LOGO_SOURCE = ROOT / "assets" / "branding" / "somaiya_vidyavihar_logo.jpg"
DOCS = {"report": ROOT / "deliverables" / "report" / "report.tex",
        "paper": ROOT / "deliverables" / "paper" / "paper.tex"}


def fill(text: str, numbers: dict, draft: bool, missing: set[str]) -> str:
    def one(m):
        key = m.group(1)
        if key in numbers:
            return render(numbers[key]["value"], m.group(2), latex=True)
        missing.add(key)
        return r"\textit{[pending]}"
    return PATTERN.sub(one, text)


def package(name: str, source: Path, numbers: dict, draft: bool) -> set[str]:
    missing: set[str] = set()
    text = fill(source.read_text(encoding="utf-8"), numbers, draft, missing)
    text = text.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{./}}")
    folder = OUT / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "main.tex").write_text(text, encoding="utf-8")
    for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))):
        if fig == LOGO:
            from PIL import Image
            Image.open(LOGO_SOURCE).save(folder / LOGO)
        elif (FIG / fig).exists():
            shutil.copy(FIG / fig, folder / fig)
        else:
            # A missing image stops pdfLaTeX, so a draft gets a framed placeholder instead.
            missing.add(f"figure {fig}")
            box = r"\fbox{\parbox{0.8\linewidth}{\centering\vspace{2cm}\textit{[pending] " + \
                  fig.replace("_", r"\_") + r"}\vspace{2cm}}}"
            text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{" + re.escape(fig) + r"\}", lambda _: box, text)
            (folder / "main.tex").write_text(text, encoding="utf-8")
    with zipfile.ZipFile(OUT / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f.name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    draft = parser.parse_args().draft
    numbers = json.loads((ROOT / "results" / "numbers.json").read_text(encoding="utf-8"))
    status = 0
    for name, source in DOCS.items():
        missing = package(name, source, numbers, draft)
        print(f"{name}: {OUT.relative_to(ROOT) / (name + '.zip')}, {len(missing)} missing")
        if missing and not draft:
            print("  missing: " + ", ".join(sorted(missing)))
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
