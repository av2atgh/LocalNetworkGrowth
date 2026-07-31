#!/usr/bin/perl
# mkbody.pl  <prefix>  <srcfile>     (body -> STDOUT; figures -> figures/*.tikz)
#
# Transform a camera-ready chapter .tex into an EPUB-friendly body for tex4ht.
# The print sources in ../ are never modified.
#
# tex4ht cannot rasterise TikZ pictures (it drops the graphics and keeps only
# the text nodes), so every tikzpicture is extracted to figures/<prefix>-N.tikz
# and replaced by \includegraphics; renderfigs.sh compiles those to PNG with the
# real TikZ engine via the standalone class.
use strict;
use warnings;

# the square both L=2 variants are drawn on, shared by \sqdiag and \sqstub
use constant SQ => '\draw[line width=0.5pt] (0,0) rectangle (1.8ex,1.8ex); ';

my $prefix = shift // 'fig';
local $/;
my $t = <>;

mkdir 'figures' unless -d 'figures';

# 1) adjustbox, command form (wide equations):  \adjustbox{..}{$\displaystyle X$} -> X
$t =~ s/\\adjustbox\{[^{}]*\}\{\$\\displaystyle\s*(.*?)\s*\$\}/$1/gs;

# 2) adjustbox, environment form (wide figures):  \begin{adjustbox}{..} ... \end{adjustbox}
$t =~ s/\\begin\{adjustbox\}\{[^{}]*\}//g;
$t =~ s/\\end\{adjustbox\}//g;

# 2b) align -> equation + aligned. A top-level align becomes an HTML <table>
# in tex4ht, with each \frac rendered as a separate inline image that Kindle
# scales up to the tall row height (huge fractions). Wrapping the rows in an
# `aligned` inside a single `equation` makes tex4ht render the whole thing as
# one display image instead. \nonumber is dropped (aligned rows aren't numbered;
# the equation carries the single number). Only one align exists book-wide.
$t =~ s/\\begin\{align\}/\\begin{equation}\\begin{aligned}/g;
$t =~ s/\\end\{align\}/\\end{aligned}\\end{equation}/g;
$t =~ s/\\nonumber//g;

# 2b') alignat{N}: render each row as its OWN numbered equation. Collapsing to a
# single aligned would put several \label's in one equation (amsmath drops all
# but the last, breaking \eqref); separate equations keep every label. Alignment
# markers (& and &&) are dropped -- the rows are short and reflow fine.
$t =~ s/\\begin\{alignat\}\{\d+\}(.*?)\\end\{alignat\}/alignat_to_eqs($1)/gse;

# 2b'') split (single numbered equation with internal alignment) -> aligned.
# tex4ht's dvisvgm path overflows its save stack on split; aligned renders it
# fine as one display image. split is always already inside an equation/align.
$t =~ s/\\begin\{split\}/\\begin{aligned}/g;
$t =~ s/\\end\{split\}/\\end{aligned}/g;

# 2c) matplotlib data figures: \includegraphics[..]{figs/NAME.pdf} -> figures/NAME.
# The print build ships these as vector PDFs under ../figs/; build-book.sh
# rasterises each to figures/NAME.png (EPUB readers do not render PDF), and the
# width/options are dropped because CSS (max-width:100%) handles sizing.
$t =~ s!\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+?)(?:\.pdf)?\}!\\includegraphics{figures/$1}!g;

# 2c') pull the shared tables in bodily rather than leaving them as \input.
# Chapters 7 to 9 keep their master tables in ../tab-*.tex, and those tables are
# where the motif glyphs live; leaving the \input for LaTeX to resolve would hide
# them from rule 2d below and from the figure extraction in rule 3.
$t =~ s!\\input\{(tab-[^}]+?)(?:\.tex)?\}!inline_file("../$1.tex")!ge;

# 2d) inline motif glyphs. \ngon{k}, \sqdiag and \sqstub are defined in
# ../main.tex with the \tikz short form rather than as tikzpicture environments,
# so rule 3 below never sees them, and tex4ht would leave the motif column of
# the chapter 8 and 9 tables empty. Rewriting them as tikzpictures hands them to
# the same extraction path as every other figure. The drawing commands are kept
# identical to the definitions in ../main.tex.
$t =~ s!\\ngon\{(\d+)\}!ngon_tikz($1)!ge;
$t =~ s!\\sqdiag\b!tikz_wrap(SQ . '\draw[line width=0.5pt] (0,0) -- (1.8ex,1.8ex);')!ge;
$t =~ s!\\sqstub\b!tikz_wrap(SQ . '\draw[line width=0.5pt] (1.8ex,1.8ex) -- ++(0.9ex,0.55ex);')!ge;

# 2e) inline math that tex4ht can only set as a picture. HTML has no radical
# sign and no fraction bar, so $\sqrt{n}$, $\tfrac12$ and $\binom{n}{2}$ come out
# as images: they do not reflow, do not scale with the reader's font size, cannot
# be selected or searched, and sit badly on the line. Spelling them the way one
# would in an email -- sqrt(n), 1/2, C(n,2) -- lets tex4ht set them as ordinary
# text. Only inline math is touched; a display equation is a long formula and is
# better as a picture than as a line of ASCII.
#
# The three arguments never nest a brace anywhere in this book (checked), so the
# simple [^{}] match is enough and a recursive pattern is not needed.
#
# The span is deliberately unbounded in length. Capping it looks safer and is
# not: a span longer than the cap fails to match, and the scan then pairs that
# span's closing $ with the next opening one, inverting the pairing for the rest
# of the paragraph and silently skipping every formula after it. A blank line
# cannot appear inside the span, which keeps an unmatched $ from running away.
$t =~ s/(?<!\\)\$((?:[^\$\n]|\n(?!\s*\n))*?)(?<!\\)\$/'$' . textify_inline($1) . '$'/ge;

# 3) inline tikzpictures -> figures/<prefix>-N.tikz + \includegraphics
my $n = 0;
$t =~ s/(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})/save_fig($prefix . '-' . (++$n), $1)/gse;

# 4) \input{netfig-*} network figures -> render them as figures too
$t =~ s/\\input\{(netfig-[^}]+)\}/save_input_fig($1)/gse;

# 5) any other bare \input/\include of a sibling file -> resolve from ../
$t =~ s/\\(input|include)\{([^}\/]+)\}/\\$1\{..\/$2\}/g;

print $t;

# An argument is left bare in a quotient if it is a single number, letter or
# command; anything with an operator in it gets brackets, so that \frac{a+b}{c}
# becomes (a+b)/c and not the false a+b/c.
sub atom {
    my ($x) = @_;
    $x =~ s/^\s+|\s+$//g;
    return $x =~ /^(?:\\?[A-Za-z]+|\d+)(?:_\{?[A-Za-z0-9]+\}?)?$/ ? $x : "($x)";
}

sub textify_inline {
    my ($m) = @_;
    # \sqrt{X} -> sqrt(X); brackets always, since sqrt binds tighter than
    # anything that can follow it
    $m =~ s/\\sqrt\s*\{([^{}]*)\}/\\mathrm{sqrt}($1)/g;
    # the brace-less form, \sqrt n, which takes the next single token
    $m =~ s/\\sqrt\s*([A-Za-z0-9])/\\mathrm{sqrt}($1)/g;
    # \binom{A}{B} -> C(A,B), the usual way of writing it on one line
    $m =~ s/\\[td]?binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}/\\mathrm{C}($1,$2)/g;
    # \frac, \tfrac, \dfrac -> a quotient with a slash
    $m =~ s!\\[td]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}!atom($1) . '/' . atom($2)!ge;
    # the two-token short form, \tfrac12
    $m =~ s!\\[td]?frac(\d)(\d)!$1/$2!g;
    # \big( and friends size a delimiter to its contents, which HTML cannot do,
    # so tex4ht sets each one as its own little image. Inline, an ordinary
    # bracket reads the same.
    $m =~ s/\\(?:bigg|Bigg|big|Big)[lrm]?(?=\s*[\(\)\[\]\|\.\\])//g;
    return $m;
}

sub inline_file {
    my ($path) = @_;
    open(my $fh, '<', $path) or die "cannot inline $path: $!";
    local $/;
    my $c = <$fh>;
    close $fh;
    return $c;
}

sub ngon_tikz {
    my ($sides) = @_;
    return tikz_wrap('\node[regular polygon, regular polygon sides=' . $sides
                     . ', draw, minimum size=2.5ex, inner sep=0pt,'
                     . ' line width=0.5pt] {};');
}

sub tikz_wrap {
    my ($body) = @_;
    return '\begin{tikzpicture}' . $body . '\end{tikzpicture}';
}

sub alignat_to_eqs {
    my ($body) = @_;
    my $out = '';
    for my $row (split /\\\\/, $body) {   # split on the LaTeX row break \\
        $row =~ s/&+//g;                  # drop alignment markers (& and &&)
        $row =~ s/^\s+|\s+$//g;
        next if $row eq '';
        $out .= "\\begin{equation}$row\\end{equation}\n";
    }
    return $out;
}

sub save_fig {
    my ($name, $code) = @_;
    open(my $fh, '>', "figures/$name.tikz") or die $!;
    print $fh $code, "\n";
    close $fh;
    return "\\includegraphics{figures/$name}";
}

sub save_input_fig {
    my ($raw) = @_;
    (my $name = $raw) =~ s/\.tex$//;     # netfig-ds.tex -> netfig-ds (single extension,
                                         # else tex4ebook mis-packs the double-dot name)
    open(my $fh, '>', "figures/$name.tikz") or die $!;
    print $fh "\\input{../$raw}\n";
    close $fh;
    return "\\includegraphics{figures/$name}";
}
