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

# 3) inline tikzpictures -> figures/<prefix>-N.tikz + \includegraphics
my $n = 0;
$t =~ s/(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})/save_fig($prefix . '-' . (++$n), $1)/gse;

# 4) \input{netfig-*} network figures -> render them as figures too
$t =~ s/\\input\{(netfig-[^}]+)\}/save_input_fig($1)/gse;

# 5) any other bare \input/\include of a sibling file -> resolve from ../
$t =~ s/\\(input|include)\{([^}\/]+)\}/\\$1\{..\/$2\}/g;

print $t;

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
