"""Export the website's pages as a static frontend that calls the detector on the PC.

    .venv/Scripts/python.exe scripts/export_site.py --api https://computer.tailcd6ab1.ts.net --out site_export

Every page is rendered by the site's own renderer (webapp/site.py), so the hosted pages match what the PC
serves. Root-relative links are moved under --base, the GitHub Pages path of this repository, and each page
names the API in a meta tag that site.js reads. The PC must allow the frontend's origin (DDF_CORS_ORIGINS,
default https://vaibhavshiroorkar.github.io). The GitHub Pages workflow runs this on every push to main.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LINK = re.compile(r'(href|src|action)="/(?!/)([^"]*)"')


def _page_url(base: str, target: str) -> str:
    """'/check#x' -> base + '/check/#x'; static files keep their path."""
    path, _, rest = target.partition("#")
    path, _, query = path.partition("?")
    if path.startswith("static/") or "." in path.rsplit("/", 1)[-1]:
        url = f"{base}/{path}"
    else:
        url = f"{base}/{path}/" if path else f"{base}/"
    return url + (f"?{query}" if query else "") + (f"#{rest}" if rest else "")


def rewrite(html: str, api: str, base: str) -> str:
    def one(m):
        attr, target = m.group(1), m.group(2)
        if attr == "action":  # forms post to the detector on the PC
            return f'{attr}="{api}/{target}"'
        return f'{attr}="{_page_url(base, target)}"'

    html = LINK.sub(one, html)
    return html.replace("<head>", f'<head>\n<meta name="ddf-api" content="{api}">', 1)


def export(out: Path, api: str, base: str) -> None:
    from deepfake_detection.webapp import site

    api, base = api.rstrip("/"), base.rstrip("/")
    if out.exists():
        shutil.rmtree(out)
    for route, (template, title) in site.PAGES.items():
        target = (
            out / route.strip("/") / "index.html"
            if route != "/"
            else out / "index.html"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            rewrite(site.render_page(template, title, route), api, base),
            encoding="utf-8",
        )
    shutil.copytree(site.STATIC_SITE, out / "static" / "site")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as they are


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", required=True, help="the detector's public HTTPS address"
    )
    parser.add_argument(
        "--base", default="/deepfake-detection", help="the Pages path of the repository"
    )
    parser.add_argument("--out", default="site_export")
    args = parser.parse_args()
    export(ROOT / args.out, api=args.api, base=args.base)
    print(f"exported -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
