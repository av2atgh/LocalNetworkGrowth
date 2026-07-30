#!/usr/bin/env python3
"""Community inference from a block model with planted edges, triangles, squares.

The standard degree-corrected SBM describes a network as blocks plus independent
links. Chapter 7 argues that this loses the network's 3- and 4-point structure
entirely, and that the object which would carry it has an order set by the arity
of the growth rule. This module is the affordable version of that idea, fitted
with the same criterion Peixoto uses for the ordinary SBM: minimum description
length.

    THE MODEL

Nodes are partitioned into B blocks. The network is then built from *motif
instances* drawn from a small alphabet,

    edge      2 nodes, 1 link,  |Aut| = 2
    triangle  3 nodes, 3 links, |Aut| = 6
    square    4 nodes, 4 links, |Aut| = 8

Every link belongs to exactly one instance, so a decomposition of the link set
into instances is a latent variable alongside the partition. Motifs of arity 3
and above are planted *inside* a single block -- that is what "each block is
generated from a higher-order object" means -- while edges may run within a block
or between two. Instances are wired by stub matching, so each node carries a
motif-degree per pool, and the result is a Karrer-Newman graph with blocks.

The restriction to within-block motifs is the whole reason this is affordable. A
4-uniform block model needs a symmetric B x B x B x B tensor, about B^4/24 free
entries, and the description length of that tensor caps the resolution at
B_max ~ N^(1/4) against the sqrt(N) of the pair model -- 18 groups against 316 at
N = 10^5. Here the count is M*B + B^2, so the ceiling is the ordinary one. A
triangle spanning two blocks is simply not representable and gets described as
three edges instead; that is a modelling choice, and a real limitation.

    THE DESCRIPTION LENGTH

A two-part code, Sigma = S + L in bits, with everything exact rather than
asymptotic so that it can be checked by brute force (see the self-test).

Instances are grouped into *pools*: one pool per (motif type, block) for
triangles and squares, one per unordered block pair for edges. Given the pool
totals and the per-node slot counts, the number of distinct ways to wire a pool
of m instances of a k-node motif inside one block is

    Omega = (k m)! / ( prod_i d_i! * m! * a^m ),        a = |Aut(motif)|

which for k=2, a=2 is the (2m-1)!! of the configuration model, and for a pool of
edges between two different blocks is the bipartite count

    Omega = e! / ( prod_{i in r} d_i! * prod_{j in s} d_j! ).

S is the sum of log2 Omega over pools -- the cost of saying which wiring was
realised. L is the cost of the model: the partition, the pool totals, and the
per-pool slot counts, each under a uniform prior over its compositions.

With the alphabet reduced to ("edge",) this is an ordinary degree-corrected
block model and the code reduces to its description length exactly; that
reduction is asserted in the self-test, and is the reason alphabets can be
compared. Adding "triangle" and "square" is a strictly nested enlargement, so
Sigma is directly comparable and a negative difference means the motif structure
paid for itself.

MDL can tell a planted triangle from three incidental links because the entropy
term falls faster than the model term rises: m instances of a 3-node motif are
more constrained, hence cheaper to specify, than 3m independent links. Where
triangles are merely incidental the arithmetic goes the other way.

    WHAT THIS IS NOT

The alphabet is a choice, not something inferred. A network built from pentagons
has no representation here beyond "edges", and the honest outcome on such a
network is that neither triangle nor square is selected. motif_sbm_validate.py
uses exactly those cases as negative controls.

Usage:
    from motif_sbm import MotifSBM
    model = MotifSBM(n, edges).fit()
    model.blocks          # inferred partition
    model.description_length()
    model.motif_counts()  # how many of each type were planted

    python3 motif_sbm.py        # runs the self-test
"""
import functools
import itertools
import math
import random
from collections import Counter, defaultdict

# ---------------------------------------------------------------- alphabet

# nodes, links, |Aut| for each motif. |Aut| is what makes the stub count right:
# the number of ways to split k*m labelled slots into m unordered instances is
# (k m)! / (m! * |Aut|^m), which for the square gives the 3 distinct 4-cycles on
# 4 labelled nodes since 4!/8 = 3.
MOTIFS = {
    "edge":     dict(k=2, links=1, aut=2),
    "triangle": dict(k=3, links=3, aut=6),
    "square":   dict(k=4, links=4, aut=8),
    # The cycles above the square. |Aut| of an L-cycle is the dihedral group,
    # 2L: L rotations and the reflection, so that k!/|Aut| = (k-1)!/2 recovers
    # the number of distinct L-cycles on L labelled nodes. They are here so that
    # BB(n,L) for L > 2 can be asked the same question as BB(n,2) -- whether
    # naming the rule's own motif dissolves the communities it appears to have
    # -- which cannot be asked of a rule whose motif is outside the alphabet.
    "pentagon": dict(k=5, links=5, aut=10),
    "hexagon":  dict(k=6, links=6, aut=12),
    "heptagon": dict(k=7, links=7, aut=14),
}

# The biclique family. K_{2,d} is two hubs joined to the same d leaves: d+2
# nodes, 2d links, and |Aut| = 2*d! since the hubs may swap and the leaves
# permute freely. It is what a duplication rule actually builds -- a copy and its
# original sharing d neighbours -- and it contains C(d,2) four-cycles, which is
# why an alphabet of squares sees a pile of overlapping motifs where there is
# really one object. d = 2 is the square itself, so the family starts at 3.
#
# Each d is its own symbol rather than one variable-size symbol, so the pool
# accounting of _pools and _pool_bits carries over untouched, and a fit that
# reaches for many different d pays to announce each of them.
BICLIQUE_MIN_D = 3


def spec(name):
    """Nodes, links and |Aut| for a motif name, including the K2_d family."""
    if name in MOTIFS:
        return MOTIFS[name]
    if name.startswith("K2_"):
        d = int(name[3:])
        return dict(k=d + 2, links=2 * d, aut=2 * math.factorial(d))
    raise KeyError(name)

LOG2 = math.log(2.0)


def _lf(x):
    """log2(x!)"""
    return math.lgamma(x + 1.0) / LOG2


def _lbinom(n, k):
    if k < 0 or k > n:
        return float("inf")
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0)
            - math.lgamma(n - k + 1.0)) / LOG2


def _lmultiset(total, boxes):
    """log2 of the number of ways to put `total` indistinct items in `boxes`."""
    if boxes <= 0:
        return 0.0 if total == 0 else float("inf")
    return _lbinom(total + boxes - 1, boxes - 1)


# ---------------------------------------------------------------- the model

class MotifSBM:
    """Degree-corrected block model whose links are grouped into motifs.

    Parameters
    ----------
    n : int
        number of nodes
    edges : iterable of (u, v)
        undirected, simple, no self-loops, 0-indexed
    alphabet : tuple of str
        subset of ("edge", "triangle", "square"); "edge" must be present, since
        every link has to be describable
    """

    def __init__(self, n, edges, alphabet=("edge", "triangle", "square")):
        assert "edge" in alphabet, "the alphabet must contain 'edge'"
        assert all(m in MOTIFS or m == "biclique" for m in alphabet), \
            f"unknown motif in {alphabet}"
        self.n = n
        self.alphabet = tuple(alphabet)
        # "biclique" is a family, expanded to K2_d as instances are planted
        self.bicliques = "biclique" in alphabet
        self.higher = tuple(m for m in alphabet
                            if m not in ("edge", "biclique"))
        self.edges = sorted({(min(u, v), max(u, v)) for u, v in edges})
        assert all(u != v for u, v in self.edges), "self-loops are not allowed"
        self.adj = defaultdict(set)
        for u, v in self.edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        # state: a partition, and a decomposition of the links into instances
        self.blocks = [0] * n
        self.instances = [("edge", e) for e in self.edges]

    # ------------------------------------------------------------ bookkeeping

    def _pools(self, blocks=None, instances=None):
        """Group instances into pools and accumulate per-node slot counts.

        Returns {pool_key: (motif_name, count, Counter(node -> slots))}, where
        pool_key is (name, r) for a within-block motif and ("edge", r, s) for
        links between blocks r <= s.
        """
        b = self.blocks if blocks is None else blocks
        inst = self.instances if instances is None else instances
        pools = {}
        for name, nodes in inst:
            brs = sorted(b[x] for x in nodes)
            if name == "edge":
                key = ("edge", brs[0], brs[1])
            else:
                # higher-order motifs live inside one block by construction
                assert brs[0] == brs[-1], "higher motif spans blocks"
                key = (name, brs[0])
            if key not in pools:
                pools[key] = [name, 0, Counter()]
            pools[key][1] += 1
            for x in nodes:
                pools[key][2][x] += 1
        return pools

    @staticmethod
    def _pool_bits(key, name, count, slots, block_size):
        """S + L for one pool: the wiring entropy plus the slot description."""
        sp = spec(name)
        k, aut = sp["k"], sp["aut"]
        if name == "edge" and key[1] != key[2]:
            # bipartite pool: e! / (prod_{i in r} d_i! * prod_{j in s} d_j!)
            e = count
            s_bits = _lf(e) - sum(_lf(d) for d in slots.values())
            # slots are described on each side separately
            nr, ns = block_size
            l_bits = _lmultiset(e, nr) + _lmultiset(e, ns)
        else:
            m = count
            s_bits = (_lf(k * m) - sum(_lf(d) for d in slots.values())
                      - _lf(m) - m * math.log2(aut))
            l_bits = _lmultiset(k * m, block_size)
        return max(s_bits, 0.0) + l_bits

    # ------------------------------------------------------------ Sigma

    def description_length(self, blocks=None, instances=None):
        """Sigma = S + L in bits. Lower is better."""
        b = self.blocks if blocks is None else blocks
        inst = self.instances if instances is None else instances
        n = self.n
        sizes = Counter(b)
        B = len(sizes)

        # -- the partition: B, then the group sizes, then the labelling
        bits = math.log2(n)
        bits += _lbinom(n - 1, B - 1)
        bits += _lf(n) - sum(_lf(c) for c in sizes.values())

        pools = self._pools(b, inst)

        # -- the pool totals. Edge counts spread over the B(B+1)/2 block pairs,
        #    each higher motif's total spread over the B blocks.
        n_edge = sum(c for (nm, c, _) in pools.values() if nm == "edge")
        n_pairs = B * (B + 1) // 2
        bits += math.log2(len(self.edges) + 1) + _lmultiset(n_edge, n_pairs)
        used = {nm for (nm, _, _) in pools.values() if nm != "edge"}
        for name in sorted(set(self.higher) | used):
            tot = sum(c for (nm, c, _) in pools.values() if nm == name)
            bits += math.log2(len(self.edges) + 1) + _lmultiset(tot, B)

        # -- per pool: wiring entropy and slot counts
        for key, (name, count, slots) in pools.items():
            if name == "edge" and key[1] != key[2]:
                bsize = (sizes[key[1]], sizes[key[2]])
            else:
                bsize = sizes[key[1]]
            bits += self._pool_bits(key, name, count, slots, bsize)
        return bits

    def motif_counts(self):
        return Counter(name for name, _ in self.instances)

    # ------------------------------------------------------- motif proposals

    def _singleton_edges(self):
        """Links currently described as their own edge-instance."""
        return {nodes for name, nodes in self.instances if name == "edge"}

    def _candidate_triangles(self, free):
        b = self.blocks
        seen = set()
        for (u, v) in free:
            if b[u] != b[v]:
                continue
            for w in self.adj[u] & self.adj[v]:
                if b[w] != b[u]:
                    continue
                tri = tuple(sorted((u, v, w)))
                if tri in seen:
                    continue
                seen.add(tri)
                links = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
                if all(l in free for l in links):
                    yield tri, links

    def _candidate_squares(self, free):
        # 4-cycles u-v-w-x-u, enumerated through pairs of common neighbours
        b = self.blocks
        seen = set()
        for u in range(self.n):
            nb = [x for x in self.adj[u] if b[x] == b[u]]
            for v, x in itertools.combinations(sorted(nb), 2):
                for w in (self.adj[v] & self.adj[x]):
                    if w == u or b[w] != b[u]:
                        continue
                    quad = (u, v, w, x)
                    key = tuple(sorted(quad))
                    cyc = tuple(sorted([tuple(sorted((u, v))),
                                        tuple(sorted((v, w))),
                                        tuple(sorted((w, x))),
                                        tuple(sorted((x, u)))]))
                    if (key, cyc) in seen:
                        continue
                    seen.add((key, cyc))
                    if len(set(quad)) == 4 and all(l in free for l in cyc):
                        yield quad, list(cyc)

    def _candidate_cycles(self, free, L):
        """Simple L-cycles inside one block whose L links are all still loose.

        Used for the pentagon and above. The triangle and the square keep their
        own generators: those are the hot paths, and leaving them untouched means
        no number measured before this family existed can have moved.

        Each cycle is walked from its smallest node, so it is reached once per
        direction and the link set deduplicates the pair.
        """
        b = self.blocks
        seen = set()
        for u in range(self.n):
            stack = [(u, (u,), frozenset((u,)))]
            while stack:
                v, path, used = stack.pop()
                if len(path) == L:
                    if u in self.adj[v]:
                        links = tuple(sorted(
                            tuple(sorted((path[i], path[(i + 1) % L])))
                            for i in range(L)))
                        if links not in seen and all(l in free for l in links):
                            seen.add(links)
                            yield path, list(links)
                    continue
                for w in self.adj[v]:
                    if w <= u or w in used or b[w] != b[u]:
                        continue
                    stack.append((w, path + (w,), used | {w}))

    def _candidate_bicliques(self, free):
        """Maximal K_{2,d} whose 2d links are all still loose, inside a block.

        Two hubs u, v need not be adjacent -- a duplicated pair is joined only if
        the rule allows self-interaction -- so the pair is found by common
        neighbours, and every leaf whose two links are still free is taken. Only
        d >= BICLIQUE_MIN_D is offered, since d = 2 is the square.
        """
        b = self.blocks
        seen = set()
        for u in range(self.n):
            second = set()
            for w in self.adj[u]:
                second |= self.adj[w]
            for v in second:
                if v <= u or b[v] != b[u]:
                    continue
                leaves = [w for w in (self.adj[u] & self.adj[v])
                          if b[w] == b[u]
                          and tuple(sorted((u, w))) in free
                          and tuple(sorted((v, w))) in free]
                if len(leaves) < BICLIQUE_MIN_D:
                    continue
                key = (u, v, tuple(sorted(leaves)))
                if key in seen:
                    continue
                seen.add(key)
                nodes = (u, v) + tuple(sorted(leaves))
                links = [tuple(sorted((h, w))) for w in leaves for h in (u, v)]
                yield f"K2_{len(leaves)}", nodes, links

    def _promote_bicliques(self, verbose=False):
        """Plant bicliques largest-first, keeping Sigma falling."""
        improved = False
        while True:
            base = self.description_length()
            free = self._singleton_edges()
            cands = sorted(self._candidate_bicliques(free),
                           key=lambda c: -len(c[2]))
            used, batch = set(), []
            for name, nodes, links in cands:
                if any(l in used for l in links):
                    continue
                used.update(links)
                batch.append((name, nodes))
            if not batch:
                return improved
            trial = [i for i in self.instances
                     if not (i[0] == "edge" and i[1] in used)] + batch
            if self.description_length(instances=trial) >= base:
                return improved
            self.instances = trial
            improved = True
            if verbose:
                print(f"    + {len(batch)} bicliques, Sigma {base:.1f} -> "
                      f"{self.description_length():.1f}")
            for name in sorted({nm for nm, _ in batch}):
                self._prune_pass(name, verbose)
            return improved

    def _promote_pass(self, verbose=False):
        """Replace groups of singleton links by higher motifs, in bulk.

        One at a time will not work, and the reason is worth stating: the first
        instance of a motif type in a block has to pay for a whole new pool --
        its slot description costs about log2 C(k + n_r - 1, n_r - 1) bits --
        against a saving of only a few links. Promotion is uphill until enough
        instances share the pool. So a maximal edge-disjoint batch is promoted
        together and kept only if Sigma falls, after which a prune pass demotes
        whichever instances do not pay their way individually.
        """
        improved = False
        # largest cycle first, for the reason the biclique pass is run before
        # this one: a smaller symbol offered first consumes the links and the
        # larger object can never be seen
        for name in sorted(self.higher,
                           key=lambda x: -spec(x)["k"]):
            if name == "triangle":
                gen = self._candidate_triangles
            elif name == "square":
                gen = self._candidate_squares
            else:
                gen = functools.partial(self._candidate_cycles,
                                        L=spec(name)["k"])
            while True:
                base = self.description_length()
                free = self._singleton_edges()
                used, batch = set(), []
                for nodes, links in gen(free):
                    if any(l in used for l in links):
                        continue
                    used.update(links)
                    batch.append((name, tuple(nodes)))
                if not batch:
                    break
                trial = [i for i in self.instances
                         if not (i[0] == "edge" and i[1] in used)] + batch
                if self.description_length(instances=trial) >= base:
                    break
                self.instances = trial
                improved = True
                if verbose:
                    print(f"    + {len(batch)} {name}s, Sigma "
                          f"{base:.1f} -> {self.description_length():.1f}")
                self._prune_pass(name, verbose)
                break        # one bulk promotion per type per sweep
        return improved

    def _prune_pass(self, name, verbose=False):
        """Demote instances of `name` whose removal lowers Sigma."""
        while True:
            base = self.description_length()
            best, best_gain = None, 1e-9
            for inst in [i for i in self.instances if i[0] == name]:
                trial = [i for i in self.instances if i is not inst]
                trial += [("edge", l) for l in self._links_of(inst)]
                gain = base - self.description_length(instances=trial)
                if gain > best_gain:
                    best, best_gain = inst, gain
            if best is None:
                return
            self.instances = [i for i in self.instances if i is not best]
            self.instances += [("edge", l) for l in self._links_of(best)]
            if verbose:
                print(f"    - {name} {best[1]}  -{best_gain:.1f} bits")

    @staticmethod
    def _links_of(inst):
        name, nodes = inst
        if name == "edge":
            return [nodes]
        if name == "triangle":
            return [tuple(sorted(p)) for p in itertools.combinations(nodes, 2)]
        if name.startswith("K2_"):
            # nodes = (hub, hub, leaf, leaf, ...)
            u, v = nodes[0], nodes[1]
            return [tuple(sorted((h, w))) for w in nodes[2:] for h in (u, v)]
        # every remaining symbol is a cycle, stored in cyclic order
        L = len(nodes)
        return [tuple(sorted((nodes[i], nodes[(i + 1) % L]))) for i in range(L)]

    # ------------------------------------------------------ partition moves

    def _demote_spanning(self, blocks):
        """Higher motifs that no longer sit inside one block become links."""
        out = []
        for inst in self.instances:
            name, nodes = inst
            if name != "edge" and len({blocks[x] for x in nodes}) > 1:
                out.extend(("edge", l) for l in self._links_of(inst))
            else:
                out.append(inst)
        return out

    def _move_pass(self, verbose=False):
        """One sweep of single-node block moves, greedy on Sigma."""
        improved = False
        for i in range(self.n):
            cur = self.blocks[i]
            cands = {self.blocks[j] for j in self.adj[i]} | {max(self.blocks) + 1}
            cands.discard(cur)
            if not cands:
                continue
            base = self.description_length()
            best, best_gain = None, 1e-9
            for r in cands:
                trial_b = list(self.blocks)
                trial_b[i] = r
                trial_i = self._demote_spanning(trial_b)
                # relabel blocks to be contiguous
                gain = base - self.description_length(trial_b, trial_i)
                if gain > best_gain:
                    best, best_gain = r, gain
            if best is not None:
                self.blocks[i] = best
                self.instances = self._demote_spanning(self.blocks)
                self._relabel()
                improved = True
                if verbose:
                    print(f"    node {i} -> block {best}  -{best_gain:.1f} bits")
        return improved

    def _relabel(self):
        seen = {}
        for i, r in enumerate(self.blocks):
            if r not in seen:
                seen[r] = len(seen)
            self.blocks[i] = seen[r]

    # ------------------------------------------------------------- inference

    def dcsbm_blocks(self, nested=True, seed=0):
        """A degree-corrected SBM partition from graph-tool, as a starting
        point. Single-node greedy moves cannot split one block into two, so
        starting from everything-in-one-block never finds communities at all;
        the block search is not the new part of this model, so it is better to
        borrow a good one than to reimplement agglomeration badly."""
        import graph_tool.all as gt
        g = gt.Graph(directed=False)
        g.add_vertex(self.n)
        g.add_edge_list(self.edges)
        gt.seed_rng(seed)
        if nested:
            st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
            b = list(st.get_bs()[0])[:self.n]
        else:
            st = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
            b = list(st.get_blocks().a)[:self.n]
        return [int(x) for x in b]

    def _fit_from(self, init_blocks, sweeps, verbose):
        self.blocks = ([0] * self.n if init_blocks is None
                       else list(init_blocks))
        self._relabel()
        self.instances = [("edge", e) for e in self.edges]
        for it in range(sweeps):
            if verbose:
                print(f"    sweep {it}: Sigma = {self.description_length():.1f}")
            moved = self._move_pass(verbose)
            # largest structures first: a biclique planted whole is cheaper than
            # the disjoint squares carved out of it, but the squares would eat
            # the links first if promotion ran in the other order
            promoted = self._promote_bicliques(verbose) if self.bicliques else False
            promoted = self._promote_pass(verbose) or promoted
            if not (moved or promoted):
                break
        return self.description_length()

    def fit(self, sweeps=8, init_blocks="dcsbm", verbose=False):
        """Alternate partition sweeps and motif promotion until Sigma stops
        falling, from several starting partitions, keeping the best.

        Both halves are greedy, so each run is a local optimum -- the usual
        caveat for an agglomerative block-model heuristic, and here it bites
        harder than usual, because the two halves interact. A motif has to sit
        inside one block to be representable at all, so a partition that cuts
        every motif in half hides the motif structure completely, and single-node
        moves will not merge two blocks back together to reveal it. The diamond
        lattice is exactly that trap: chapter 6's branch cut puts every one of
        its four-cycles across the cut, and started from that partition the model
        reports no motifs at all, while the one-block start finds all 4^(t-1) of
        them and a description length some 300 bits shorter. Hence more than one
        start, and keep whichever ends lowest.
        """
        starts = []
        if init_blocks == "dcsbm":
            try:
                starts.append(self.dcsbm_blocks())
            except ImportError:
                pass
            starts.append(None)                       # everything in one block
        else:
            starts.append(init_blocks)
        best = None
        for k, init in enumerate(starts):
            if verbose:
                print(f"  start {k} ({'one block' if init is None else 'dcsbm'})")
            sigma = self._fit_from(init, sweeps, verbose)
            if best is None or sigma < best[0]:
                best = (sigma, list(self.blocks), list(self.instances))
        self.blocks, self.instances = list(best[1]), list(best[2])
        return self

    # ------------------------------------------------------------ generative

    @classmethod
    def sample(cls, block_sizes, plan, seed=0):
        """Plant motifs and wire them, for testing the inference.

        `plan` maps block index -> {motif name: how many to plant}. Nodes inside
        a block are chosen uniformly, so the planted degrees are Poisson-ish.
        Returns (n, edges, true_blocks).
        """
        rng = random.Random(seed)
        blocks, start = [], 0
        offsets = []
        for r, sz in enumerate(block_sizes):
            offsets.append(start)
            blocks.extend([r] * sz)
            start += sz
        n = start
        edges = set()
        for r, sz in enumerate(block_sizes):
            members = list(range(offsets[r], offsets[r] + sz))
            for name, count in plan.get(r, {}).items():
                k = MOTIFS[name]["k"]
                for _ in range(count):
                    nodes = rng.sample(members, k)
                    if name == "edge":
                        links = [(nodes[0], nodes[1])]
                    elif name == "triangle":
                        links = list(itertools.combinations(nodes, 2))
                    else:
                        links = [(nodes[i], nodes[(i + 1) % 4]) for i in range(4)]
                    for u, v in links:
                        edges.add((min(u, v), max(u, v)))
        return n, sorted(edges), blocks


# ---------------------------------------------------------------- self-test

def _brute_force_slots(k, m, degrees, aut):
    """Enumerate stub groupings and check both factors of the closed form.

    The closed form is the stub-matching count of the configuration model,
    (k m)! / (prod_i d_i! * m! * aut^m), which is the standard multigraph
    convention -- the same one Peixoto's microcanonical SBM uses. Two things
    have to hold for it, and both are checked here rather than assumed:

      * the number of ways to split k*m labelled stubs into m unordered
        instances is (k m)! / (m! * aut^m);
      * dividing by prod_i d_i! is right because the stub-relabelling group
        prod_i S_{d_i} acts freely on generic configurations, so each appears
        exactly prod_i d_i! times.

    Returns (grouping_count, expected_grouping_count, orbit_sizes_of_generic).
    """
    slots = []
    for node, d in enumerate(degrees):
        slots.extend([node] * d)
    assert len(slots) == k * m
    counts = Counter()
    total = 0
    for perm in itertools.permutations(range(len(slots))):
        groups = tuple(sorted(tuple(sorted(perm[g * k:(g + 1) * k]))
                              for g in range(m)))
        counts[groups] = 1                     # dedupe orderings of the split
    # each distinct split of the labelled stubs, mapped down to nodes
    by_config = Counter()
    for groups in counts:
        cfg = tuple(sorted(tuple(sorted(slots[s] for s in g)) for g in groups))
        by_config[cfg] += 1
    total = len(counts)
    # splitting km labelled stubs into m unordered k-sets
    expected = math.factorial(k * m) // (math.factorial(m)
                                         * math.factorial(k) ** m)
    generic = {cfg: c for cfg, c in by_config.items()
               if all(len(set(g)) == k for g in cfg) and len(set(cfg)) == len(cfg)}
    return total, expected, set(generic.values())


def _selftest():
    ok = True

    print("1. the stub-matching count, all three factors, against enumeration")
    for k, aut, m, degrees in [(2, 2, 2, [1, 1, 1, 1]),
                               (2, 2, 2, [2, 1, 1]),
                               (3, 6, 2, [1, 1, 1, 1, 1, 1]),
                               (3, 6, 2, [2, 2, 1, 1]),
                               (4, 8, 2, [1] * 8)]:
        total, expected, orbits = _brute_force_slots(k, m, degrees, aut)
        want_orbit = math.prod(math.factorial(d) for d in degrees)
        # splits x (distinct motifs per k-set) must equal (km)!/(m! aut^m)
        closed = math.factorial(k * m) / (math.factorial(m) * aut ** m)
        combined = total * (math.factorial(k) / aut) ** m
        good = (total == expected
                and abs(combined - closed) < 1e-9
                and (not orbits or orbits == {want_orbit}))
        ok &= good
        print(f"   {'ok ' if good else 'FAIL'} k={k} m={m} d={degrees}: "
              f"splits {total}={expected}, x{math.factorial(k) // aut} motifs "
              f"per set = {combined:.0f}={closed:.0f}; "
              f"generic orbits {orbits or '{}'}=prod d_i! ({want_orbit})")

    print("\n2. |Aut| conventions give the right number of distinct motifs")
    for name, expect in (("edge", 1), ("triangle", 1), ("square", 3)):
        spec = MOTIFS[name]
        got = math.factorial(spec["k"]) / spec["aut"]
        good = abs(got - expect) < 1e-9
        ok &= good
        print(f"   {'ok ' if good else 'FAIL'} {name}: k!/|Aut| = {got:.0f} "
              f"distinct wirings on k labelled nodes (expect {expect})")

    print("\n3. alphabet ('edge',) reduces to a plain degree-corrected SBM")
    n, edges, truth = MotifSBM.sample([12, 12], {0: {"edge": 30},
                                                 1: {"edge": 30}}, seed=3)
    a = MotifSBM(n, edges, alphabet=("edge",))
    b = MotifSBM(n, edges, alphabet=("edge", "triangle", "square"))
    a.blocks = b.blocks = list(truth)
    # with no higher motifs planted in the decomposition the two must agree
    # except for the cost of announcing the two extra (empty) motif totals
    extra = 2 * math.log2(len(edges) + 1)
    d = abs((b.description_length() - a.description_length()) - extra)
    ok &= d < 1e-9
    print(f"   {'ok ' if d < 1e-9 else 'FAIL'} Sigma differs by exactly the "
          f"two empty-total announcements ({extra:.2f} bits), residual {d:.2e}")

    print("\n4. planted triangles beat loose links, incidental ones do not")
    n, edges, truth = MotifSBM.sample([60], {0: {"triangle": 60}}, seed=5)
    m = MotifSBM(n, edges)
    m.blocks = list(truth)
    before = m.description_length()
    m._promote_pass()
    after, c = m.description_length(), m.motif_counts()
    good = after < before and c["triangle"] > 0
    ok &= good
    print(f"   {'ok ' if good else 'FAIL'} 60 planted in 60 nodes: Sigma "
          f"{before:.1f} -> {after:.1f}, kept {c['triangle']} triangles")

    # A near-complete graph has triangles everywhere and none of them mean
    # anything; the criterion has to refuse those, or it is just counting.
    n, edges, truth = MotifSBM.sample([9], {0: {"triangle": 12}}, seed=5)
    m = MotifSBM(n, edges)
    m.blocks = list(truth)
    before = m.description_length()
    m._promote_pass()
    after, c = m.description_length(), m.motif_counts()
    good = c["triangle"] == 0
    ok &= good
    print(f"   {'ok ' if good else 'FAIL'} 9 nodes, 26 of 36 links: kept "
          f"{c['triangle']} triangles (must be 0 -- incidental, not planted)")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
