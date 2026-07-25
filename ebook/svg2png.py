#!/usr/bin/env python3
# svg2png.py  UNZIP_ROOT
#
# Rasterise every dvisvgm SVG in the EPUB to a high-DPI PNG and rewrite all
# references (XHTML <img> + OPF manifest). Amazon's KFX converter (KDP error
# "We couldn't convert your HTML file to Kindle format") fails on SVG; PNG is
# the reliable image format for Kindle. Display size is preserved (set as px
# width/height from the SVG's pt size) and the PNG is rendered at SCALE× that
# for crispness; CSS (max-width:100%; height:auto) keeps wide figures on-screen.
import sys, os, re, glob, subprocess
from PIL import Image

root  = sys.argv[1]
oebps = os.path.join(root, "OEBPS")
PT2PX   = 96.0 / 72.0    # CSS px per TeX/big point

# Sizing is PER IMAGE. Kindle sizes INLINE images by their intrinsic pixel count
# (it ignores tiny width/height) but honours width/height for block DISPLAY
# equations. So:
#  - inline math/fractions: modest rasterisation zoom, shown at natural px.
#  - display equations: enlarged by the SAME DISPLAY_SCALE for every equation, so
#    the font (px per line) is identical book-wide. A one-line equation and a
#    three-line derivation are scaled by the same factor -- the tall one is
#    simply taller, not larger-typed. Rasterised at ~2x that scale for crispness.
#    (An earlier version boosted short single-line equations to a fixed target
#    height; that gave them a visibly bigger font than multi-line equations.)
DISPLAY_SCALE     = 2.0     # uniform enlargement for ALL display equations
# Kept modest on purpose: at a large scale a typical equation is wider than a
# phone/e-reader column, so CSS max-width:100% shrinks it by a width-dependent
# amount -- a wide equation clamps hard (tiny) while a narrow one escapes and
# looks oversized. A smaller scale lets most equations fit the column unclamped,
# so the on-screen font is uniform; only genuinely very wide formulae still clamp.
MAX_RASTER        = 11.0    # cap on rasterisation zoom (bounds PNG/file size)
INLINE_RASTER     = 2.5     # rasterisation zoom for inline math/fractions

# Which SVGs are used as display equations (class="math-display")?
display_svgs = set()
for _x in glob.glob(os.path.join(oebps, "*.xhtml")):
    _s = open(_x, encoding="utf-8").read()
    for _m in re.finditer(r"<img\b[^>]*?>", _s):
        if "math-display" in _m.group(0):
            _sm = re.search(r"src='([^']+?)\.svg'", _m.group(0))
            if _sm:
                display_svgs.add(os.path.basename(_sm.group(1)) + ".svg")

def intrinsic_pt(svg_path):
    with open(svg_path, encoding="utf-8") as f:
        head = f.read(2000)
    w = re.search(r"width='([\d.]+)pt'", head)
    h = re.search(r"height='([\d.]+)pt'", head)
    if w and h:
        return float(w.group(1)), float(h.group(1))
    vb = re.search(r"viewBox='[\d.\- ]*?([\d.]+) ([\d.]+)'", head)  # last two = w h
    if vb:
        return float(vb.group(1)), float(vb.group(2))
    return None

def blank_png(path):
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(path)

BASE_PT = 10.0   # document base font (\documentclass[10pt]) -> 1em

sizes = {}   # basename.svg -> (display_w_px, display_h_px, w_pt)
for svg in glob.glob(os.path.join(oebps, "*.svg")):
    base = os.path.basename(svg)
    png  = svg[:-4] + ".png"
    pt   = intrinsic_pt(svg)
    wpt, hpt = pt if pt else (0, 0)
    if wpt <= 0 or hpt <= 0:
        # degenerate empty image (e.g. dvisvgm's 0x0 page artifact) -> 1x1 px
        blank_png(png); sizes[base] = (1, 1); os.remove(svg); continue
    dw_nat, dh_nat = max(1, round(wpt * PT2PX)), max(1, round(hpt * PT2PX))
    if base in display_svgs:
        eff  = DISPLAY_SCALE                 # uniform: same font size for every equation
        zoom = min(MAX_RASTER, 2.0 * eff)
        dw, dh = round(dw_nat * eff), round(dh_nat * eff)
        # Bake an opaque white background into display equations. Dark-mode
        # legibility otherwise relies on a CSS `filter: invert(1)`, which some
        # e-readers skip on large images -- leaving tall multi-line equations as
        # black glyphs on a black page. A baked white background is self-contained:
        # the equation reads as black-on-white in any reader, in any colour scheme.
        bg = ["-b", "white"]
    else:
        zoom, dw, dh = INLINE_RASTER, dw_nat, dh_nat
        bg = []                              # inline math stays transparent (small; invert works)
    try:
        subprocess.run(["rsvg-convert", "-z", str(zoom), *bg, svg, "-o", png], check=True)
    except subprocess.CalledProcessError:
        print("  WARN: rsvg failed on", base, "- using blank placeholder")
        blank_png(png)
    sizes[base] = (dw, dh)
    os.remove(svg)

# rewrite XHTML: point <img> at the .png and set the final pixel width/height
# (display equations were already enlarged above; inline math keeps natural px;
# wide display equations are still capped to the column by CSS max-width:100%).
def fix_img(m):
    tag, name = m.group(0), m.group("name")
    base = name.split("/")[-1] + ".svg"
    if base not in sizes:
        return tag
    dw, dh = sizes[base]
    tag = tag.replace(name + ".svg", name + ".png")
    tag = re.sub(r"""\s+(width|height|style)=('[^']*'|"[^"]*")""", "", tag)  # drop existing
    tag = re.sub(r"\s*/?>\s*$", " width='%d' height='%d' />" % (dw, dh), tag)
    return tag

for x in glob.glob(os.path.join(oebps, "*.xhtml")):
    with open(x, encoding="utf-8") as f:
        s = f.read()
    s = re.sub(r"<img\b[^>]*?src='(?P<name>[^']+?)\.svg'[^>]*?>", fix_img, s)
    with open(x, "w", encoding="utf-8") as f:
        f.write(s)

# rewrite OPF manifest: .svg -> .png, image/svg+xml -> image/png
opf = os.path.join(oebps, "content.opf")
with open(opf, encoding="utf-8") as f:
    s = f.read()
s = s.replace(".svg'", ".png'").replace('.svg"', '.png"').replace("image/svg+xml", "image/png")
with open(opf, "w", encoding="utf-8") as f:
    f.write(s)

print("svg2png: rasterised %d images to PNG" % len(sizes))
