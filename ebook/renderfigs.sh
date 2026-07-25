#!/bin/sh
# renderfigs.sh — compile every figures/*.tikz (extracted by mkbody.pl) to a
# tightly-cropped PNG using the real TikZ engine via the standalone class.
# Run from ebook/ after preprocessing, before tex4ebook.
set -e
cd "$(dirname "$0")"

[ -d figures ] || { echo "  (no figures/ dir — nothing to render)"; exit 0; }

for tikz in figures/*.tikz; do
  [ -e "$tikz" ] || continue
  name=$(basename "${tikz%.tikz}")
  cat > _figwrap.tex <<EOF
\documentclass[border=3pt]{standalone}
\input{ebook-preamble}
\begin{document}
\input{figures/$name.tikz}
\end{document}
EOF
  if pdflatex -interaction=nonstopmode -halt-on-error \
       -output-directory=figures -jobname="$name" _figwrap.tex >/dev/null 2>&1; then
    pdftoppm -png -r 200 -singlefile "figures/$name.pdf" "figures/$name"
    echo "  rendered $name.png"
  else
    echo "  WARN: figure $name failed to compile"
  fi
done
rm -f _figwrap.tex figures/*.aux figures/*.log figures/*.pdf
