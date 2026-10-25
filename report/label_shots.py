"""Writes labelled copies of the page screenshots (a), (b) into report/_labelled/ (originals in assets/screens are not touched)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
S = Path(__file__).resolve().parent.parent / "assets" / "screens"
O = Path(__file__).resolve().parent / "_labelled"
O.mkdir(exist_ok=True)
for name, lab in (("web_image", "(a)"), ("web_video", "(b)"), ("real_image", "(a)"), ("real_video", "(b)")):
    im = Image.open(S / f"{name}.png").convert("RGB")
    d = ImageDraw.Draw(im)
    sz = max(40, im.width // 14)
    try:
        f = ImageFont.truetype("arialbd.ttf", sz)
    except OSError:
        f = ImageFont.load_default()
    w = d.textlength(lab, font=f)
    d.rectangle([8, 8, 8 + w + 24, 8 + sz + 16], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.text((20, 12), lab, fill=(0, 0, 0), font=f)
    im.save(O / f"{name}.png")
    print(name, im.size)
