#!/bin/sh
# build-book.sh — full-book EPUB3 (preface + 11 chapters) with PNG math/figures.
# Print sources in ../ are never modified.
set -e
cd "$(dirname "$0")"

CHAPTERS="introduction walks friends proteins projects communities graphons motifs bubbles distance outlook"

echo ">> 1/6  preprocessing chapters -> *-body.tex (+ extract figures)"
rm -rf figures
for ch in $CHAPTERS; do
  perl mkbody.pl "$ch" "../$ch.tex" > "$ch-body.tex"
done

echo ">> 2/6  extracting the preface from ../main.tex"
# through mkbody.pl like any chapter: the preface carries inline math too, and
# slicing it straight out of main.tex used to leave it as the one part of the
# book whose formulae were still pictures.
perl -ne 'print if /\\chapter\*\{Preface\}/ .. /\\tableofcontents/ and not /\\tableofcontents/' \
  ../main.tex > _preface-raw.tex
perl mkbody.pl preface _preface-raw.tex > preface-body.tex
rm -f _preface-raw.tex

echo ">> 3/6  rendering TikZ figures -> PNG (standalone)"
sh renderfigs.sh

echo ">> 3b/6  rasterising matplotlib data figures (../figs/*.pdf) -> figures/*.png"
mkdir -p figures
for pdf in ../figs/*.pdf; do
  [ -e "$pdf" ] || continue
  name=$(basename "${pdf%.pdf}")
  pdftoppm -png -r 220 -singlefile "$pdf" "figures/$name" \
    && echo "  rasterised $name.png"
done

echo ">> 4/6  generating .aux/.bbl for citations"
pdflatex -interaction=nonstopmode book-ebook.tex >/dev/null 2>&1 || true
bibtex book-ebook >/dev/null 2>&1 || true

echo ">> 5/6  tex4ebook -> EPUB3 (math/equation SVG via dvisvgm_hashes)"
tex4ebook -f epub3+dvisvgm_hashes -c ebook.cfg book-ebook.tex

echo ">> 6/6  post-process: merge CSS, fix EPUB3 validity, SVG->PNG, repack"
out="$PWD/book-ebook.epub"
tmp="$(mktemp -d)"
unzip -q "$out" -d "$tmp"
python3 fixepub.py "$tmp" friends.css "Local network growth"
python3 svg2png.py "$tmp"
rm -f "$out"
( cd "$tmp" && zip -X -q -0 "$out" mimetype && zip -X -q -9 -r "$out" . -x mimetype )
rm -rf "$tmp"

echo ">> done: $(ls -1 book-ebook.epub 2>/dev/null || echo 'EPUB NOT PRODUCED')"
