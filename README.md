# Local network growth
### How simple rules drive network complexity

**Alexei Vazquez**

📖 **Published edition:** https://www.amazon.com/Local-network-growth-simple-complexity/dp/B0H4WSPJ2S

This repository holds the complete LaTeX source of the book — text, figures, the
data behind every plot, and the toolchains that build both the print PDF and the
EPUB.

---

## About the book

The Internet, a living cell, a circle of friends, a billion-dollar construction
project: these systems share almost nothing — yet, drawn as networks, they look
astonishingly alike. Each has a few giant hubs among a multitude of sparsely
connected nodes, short paths between any two parts, dense local clustering,
communities, and many redundant routes.

For two decades such patterns have been credited to *preferential attachment*,
the rich getting richer — a rule that, taken literally, asks every newcomer to
survey the whole network before it links. This book makes a simpler case, and
defends it one mechanism at a time: the global regularities of real networks are
not imposed from above but **emerge from purely local rules**, in which each new
node acts only on a node it has reached and that node's immediate neighbours. A
surfer following links, a friend introducing a friend, a gene copied with its
connections — none consults the network as a whole, yet each builds, in the
aggregate, the full and unmistakable signature of a real complex system.

Written for the curious reader as much as the specialist: the ideas are told in
plain language, and the mathematics is set aside in boxes that can be skipped.

## Contents

| Ch. | Source | Topic |
|-----|--------|-------|
| 1 | `introduction.tex` | Networks everywhere; basic measures; from global to local rules; what the data demand |
| 2 | `walks.tex` | Search and connect — surfing as a random walk, an effective "rich get richer" |
| 3 | `friends.tex` | Friends of friends — triadic closure, and why social networks are assortative |
| 4 | `proteins.tex` | Gene duplication — networks that copy themselves |
| 5 | `projects.tex` | Activity specialization — project schedules, the vanishing critical path, delay risk as percolation |
| 6 | `communities.tex` | Emergence of communities — the Ramsey community number, exactly solvable cases, how many communities real networks have |
| 7 | `distance.tex` | Boosting shortest path multiplicity — mean-field theory, cycles and the boost, real networks |
| 8 | `outlook.tex` | One principle, many faces; what remains to be done |

`main.tex` is the driver (title page, preface, front matter); `references.bib`
holds the bibliography.

## Building

Print edition (5×8 in trim, 141 pages):

```sh
latexmk -pdf main.tex     # -> main.pdf
```

EPUB3 edition:

```sh
cd ebook && sh build-book.sh    # -> ebook/book-ebook.epub
```

See [`ebook/README.md`](ebook/README.md) for the EPUB pipeline, its
dependencies, and why each preprocessing step exists. The current EPUB build
passes EPUBCheck 5.3 with 0 errors and 0 warnings.

## Repository layout

| Path | Contents |
|------|----------|
| `*.tex`, `references.bib` | Book source (second edition) |
| `figs/` | Matplotlib figure scripts, the CSV data they plot, and the generated PDFs |
| `netfig-*.tex`, `svg2tikz.py` | Network diagrams, converted from graph-tool SVGs to TikZ |
| `cover/` | Cover art, back-cover blurbs, author bio and headshot |
| `ebook/` | EPUB3 build toolchain and the generated `book-ebook.epub` |
| `paperback_edition_1/` | **Frozen** snapshot of the first edition — read-only, kept matching the printed book |
| `new_in_the_second_edition.log` | Running changelog of the first → second edition delta |
| `issues.log` | Working notes: verified defects and open questions |

`main.pdf` is tracked, so a rebuild dirties the tree even when nothing changed;
`git checkout main.pdf` after a test-only build.

## About the author

Alexei Vazquez is a physicist, data scientist and network scientist. He has
studied the topology of the Internet, protein interaction networks, metabolic
networks, social and citation graphs, project networks and the emergence of
communities. He works at Nodes & Links Ltd.
