#!/usr/bin/env python3
"""Extract node positions/communities and edges from a graph-tool block SVG
and emit a self-contained TikZ picture. Communities are taken from the node fill
colours, most populous first, and rendered as a ramp of greys from black to
white. Up to len(SHADES) groups."""
import re, sys, math

# grey ramp, most populous community first
SHADES = ['black!72', 'white', 'black!30', 'black!52', 'black!12', 'black!88', 'black!40']

def parse(path):
    txt = open(path).read()
    # nodes: <path ... fill="rgb(...)" ... transform="matrix(s,0,0,s,cx,cy)"/>
    node_re = re.compile(
        r'<path[^>]*?fill="(rgb\([^"]*\))"[^>]*?transform="matrix\('
        r'([0-9.eE+-]+),\s*0,\s*0,\s*[0-9.eE+-]+,\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\)"',
        re.S)
    nodes, colors = [], []
    for m in node_re.finditer(txt):
        col = m.group(1)
        cx, cy = float(m.group(3)), float(m.group(4))
        nodes.append((cx, cy)); colors.append(col)
    # map the two colors -> 0 (larger group) and 1 (smaller)
    uniq = {}
    for c in colors: uniq[c] = uniq.get(c, 0) + 1
    order = sorted(uniq, key=lambda c: -uniq[c])      # most common first
    if len(order) > len(SHADES):
        raise SystemExit('%d node colours; only %d styles are defined'
                         % (len(order), len(SHADES)))
    cmap = {c: i for i, c in enumerate(order)}
    comm = [cmap[c] for c in colors]

    # edges: <path fill="none" ... d="M lx ly L lx2 ly2" transform="matrix(s,0,0,s,tx,ty)"/>
    edge_re = re.compile(
        r'<path fill="none"[^>]*?d="M\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+L\s*'
        r'([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*"[^>]*?transform="matrix\('
        r'([0-9.eE+-]+),\s*0,\s*0,\s*[0-9.eE+-]+,\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\)"',
        re.S)
    def nearest(px, py):
        best, bd = -1, 1e9
        for i, (x, y) in enumerate(nodes):
            d = (x-px)**2 + (y-py)**2
            if d < bd: bd, best = d, i
        return best, math.sqrt(bd)
    edges = set()
    for m in edge_re.finditer(txt):
        lx1, ly1, lx2, ly2, s, tx, ty = map(float, m.groups())
        gx1, gy1 = s*lx1+tx, s*ly1+ty
        gx2, gy2 = s*lx2+tx, s*ly2+ty
        a, da = nearest(gx1, gy1)
        b, db = nearest(gx2, gy2)
        if a != b and da < 6 and db < 6:
            edges.add((min(a, b), max(a, b)))
    return nodes, comm, sorted(edges)

def emit(nodes, comm, edges, width=7.0, r=0.10):
    xs = [x for x, y in nodes]; ys = [y for x, y in nodes]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    s = width / (xmax - xmin)
    def XY(p):
        x, y = p
        return (x - xmin) * s, (ymax - y) * s      # flip y for TikZ
    out = []
    out.append(r'\begin{tikzpicture}[')
    out.append(r'  netlink/.style={draw=black!38,line width=0.35pt},')
    for i, fill in enumerate(SHADES):
        end = ',' if i < len(SHADES) - 1 else ']'
        out.append(r'  n%s/.style={circle,draw=black!75,fill=%s,line width=0.3pt,inner sep=0pt,minimum size=%.2fcm}%s'
                   % (chr(65 + i), fill, 2 * r, end))
    coords = [XY(p) for p in nodes]
    for (a, b) in edges:
        out.append(r'  \draw[netlink] (%.3f,%.3f) -- (%.3f,%.3f);' %
                   (coords[a][0], coords[a][1], coords[b][0], coords[b][1]))
    for i, (X, Y) in enumerate(coords):
        st = 'n' + chr(65 + comm[i])
        out.append(r'  \node[%s] at (%.3f,%.3f) {};' % (st, X, Y))
    out.append(r'\end{tikzpicture}')
    return '\n'.join(out)

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    nodes, comm, edges = parse(src)
    sizes = '/'.join(str(comm.count(i)) for i in range(max(comm) + 1))
    sys.stderr.write('%s: %d nodes (%s communities), %d edges\n' %
                     (src, len(nodes), sizes, len(edges)))
    open(dst, 'w').write(emit(nodes, comm, edges) + '\n')
