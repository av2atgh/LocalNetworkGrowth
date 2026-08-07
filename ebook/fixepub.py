#!/usr/bin/env python3
# fixepub.py  UNZIP_ROOT  CSS_SRC  [BOOK_TITLE]
#
# Post-process the unzipped EPUB tree to (a) merge the starter stylesheet and
# (b) fix the handful of EPUB3 validity issues tex4ebook leaves behind, so the
# package passes EPUBCheck. Called by build.sh between unzip and repack.
import sys, os, re, glob, unicodedata

root  = sys.argv[1]
css   = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "Growing Networks"
oebps = os.path.join(root, "OEBPS")

def clean_alt(text):
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&#38;", "&"), ("&#60;", "<"), ("&#62;", ">")):
        text = text.replace(a, b)
    text = text.replace("&", " and ").replace("<", " lt ").replace(">", " gt ")
    return re.sub(r"\s+", " ", text).strip()

# (1) merge starter CSS into the auto-generated, already-bundled stylesheet(s)
#     (the file is named after the jobname, so locate it by glob)
if os.path.exists(css):
    with open(css, encoding="utf-8") as f:
        extra = f.read()
    for auto in glob.glob(os.path.join(oebps, "*.css")):
        with open(auto, "a", encoding="utf-8") as f:
            f.write("\n/* ---- starter overrides from friends.css ---- */\n" + extra)

# (2) per-XHTML fixes
for path in glob.glob(os.path.join(oebps, "*.xhtml")):
    with open(path, encoding="utf-8") as f:
        s = f.read()

    # empty <title> -> first heading text, else the book title (EPUBCheck RSC-005)
    def fill_title(m):
        if m.group(1).strip():
            return m.group(0)
        h = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", s, re.S)
        t = re.sub(r"<[^>]+>", "", h.group(1)).strip() if h else ""
        return "<title>%s</title>" % (t or title)
    s = re.sub(r"<title>(.*?)</title>", fill_title, s, flags=re.S)

    # deprecated presentational attribute align=... (handled via CSS instead)
    s = re.sub(r"""\s+align=("[^"]*"|'[^']*')""", "", s)

    # \multicolumn spanning a whole row -- used for the class labels in the
    # master tables of chapters 8 and 9 -- comes out of tex4ht as an EMPTY <td>
    # followed by a sibling <div class='multicolumn'>, and a <div> is not
    # allowed between </td> and </tr>. Move the div's content into the td it
    # belongs to, which is both valid and what the print table means.
    s = re.sub(r"></td><div class=(?:\"|')multicolumn(?:\"|')[^>]*>(.*?)</div>",
               lambda m: ">%s</td>" % m.group(1), s, flags=re.S)

    # Sanitise alt text. tex4ht emits multi-line ASCII-art alt for equations,
    # with HTML entities (&gt; &lt; &amp;). Amazon's KFX converter fails to parse
    # both multi-line attribute values AND entities inside alt (error E21018 on
    # <img>), even though both are valid XML and kindlegen tolerates them. The
    # alt is unreadable ASCII-art anyway, so collapse whitespace and turn the
    # XML-significant characters into plain words (no entities, still valid XML).
    s = re.sub(r"alt='([^']*)'", lambda m: "alt='%s'" % clean_alt(m.group(1)), s)
    s = re.sub(r'alt="([^"]*)"', lambda m: 'alt="%s"' % clean_alt(m.group(1)), s)

    # Replace Unicode that Kindle's fonts render as tofu boxes. tex4ht emits
    # Mathematical Alphanumeric Symbols (U+1D400-1D7FF) for some inline math --
    # e.g. U+1D716 (math italic epsilon) for \epsilon -- which Kindle fonts lack.
    # NFKD folds each to its base letter; then map the lunate epsilon (U+03F5) to
    # the standard Greek epsilon (U+03B5), which every font has.
    s = "".join(unicodedata.normalize("NFKD", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF
                else ch for ch in s)
    s = s.replace("ϵ", "ε")

    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

# (3) content.opf: make the generated TOC reachable (drop linear="no") so the
#     non-linear-content rule is satisfied (EPUBCheck OPF-096)
opf = os.path.join(oebps, "content.opf")
if os.path.exists(opf):
    with open(opf, encoding="utf-8") as f:
        s = f.read()
    s = re.sub(r"""\s+linear=("no"|'no')""", "", s)
    # Add a <guide> with a toc reference. Modern KFX uses the EPUB3 nav doc, but
    # Kindle/KDP also look for the legacy guide reference to surface the "Go To >
    # Table of Contents" menu reliably across devices. Point it at the nav doc.
    if "<guide" not in s:
        m = re.search(r"<item\s+href='([^']+)'[^>]*?properties='nav'", s)
        if m:
            guide = ('  <guide>\n'
                     '    <reference type="toc" title="Table of Contents" href="%s"/>\n'
                     '  </guide>\n' % m.group(1))
            s = s.replace("</package>", guide + "</package>")
    with open(opf, "w", encoding="utf-8") as f:
        f.write(s)

print("fixepub: css merged, titles/align/opf fixed")
