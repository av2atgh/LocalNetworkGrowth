/* Triadic-closure growth with node addition, triangle closure and triangle
 * opening decoupled.
 *
 *   cc -O3 -o tc tc.c -lm
 *   ./tc MODE PARAM N SEED OUTPREFIX [PARAM2]
 *
 * Three processes run at total rates alpha (arrival), beta_+ (closure) and
 * beta_- (removal).  An event is an arrival with probability 1-u_+-u_-, a
 * closure with probability u_+ = PARAM, a removal with probability u_- = PARAM2
 * (default 0), u_pm = beta_pm/(alpha+beta_+ + beta_-).  Arrival: a new node
 * links to a uniformly random node.  <k> -> 2(1 + r_+ - r_-), r_pm = beta_pm/alpha,
 * as long as removals find a triangle.
 *
 * MODE (closure rule, and the removal rule that mirrors it)
 *   l    LS(n,1): a new node links to a random node j and to a random
 *        neighbour of j (PARAM ignored).  <k> -> 4.
 *   w    close a uniformly random open triad (path a-c-b, a,b unlinked);
 *        remove the closing link of a uniformly random closed triad, i.e. a
 *        link chosen with probability proportional to its triangles t_e.
 *   p    close a uniformly random potential link (unlinked pair with a common
 *        neighbour), the rule of friends.tex; remove a uniformly random link
 *        that lies in at least one triangle.
 *   c    local: a random node c (k>=2) picks two of its neighbours a, b and
 *        links them if unlinked (closure) or unlinks them if linked (removal);
 *        otherwise retry from a new c.
 *   f    local: a random node a walks a -> j -> l != a and links to l if
 *        unlinked (closure) or unlinks from l if linked (removal); else retry.
 *   r    per-capita rates: every node spawns a newcomer at rate alpha, every
 *        open triad closes at rate beta_+, every closed triad opens at rate
 *        beta_-; PARAM = beta_+/alpha, PARAM2 = beta_-/alpha.  Closure and
 *        removal pick as in w.  The run stops early if E/N exceeds 10.
 *
 * A closure that finds no open triad becomes an arrival; a removal that finds
 * no triangle does nothing (both are counted in .sum).  Removing a link of a
 * triangle never disconnects the network.  With PARAM2 = 0 the random stream,
 * and so every earlier run, is unchanged.
 *
 * Output
 *   OUTPREFIX.ts     checkpoints: N E kmean T Wall transitivity kmax closures removals
 *   OUTPREFIX.nodes  per node (id = birth order): k t knn_sum w_open
 *                    w_open = open triads with the node as an endpoint
 *   OUTPREFIX.edges  int32 pairs (a<b), binary
 *   OUTPREFIX.sum    one line: degree assortativity and totals
 *   With TSONLY set in the environment only OUTPREFIX.ts is written.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------- rng */
static uint64_t s[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t next64(void) {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
}
static void seed_rng(uint64_t x) {
    for (int i = 0; i < 4; i++) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[i] = z ^ (z >> 31);
    }
}
static inline double unif(void) { return (next64() >> 11) * 0x1.0p-53; }
static inline uint64_t randint(uint64_t n) { return next64() % n; }

/* ---------------------------------------------------------------- fenwick */
typedef struct { int64_t *f; int n; } fenwick;
static void fw_init(fenwick *w, int n) { w->n = n; w->f = calloc(n + 1, sizeof(int64_t)); }
static void fw_add(fenwick *w, int i, int64_t v) { for (i++; i <= w->n; i += i & -i) w->f[i] += v; }
static int fw_find(const fenwick *w, int64_t r) {      /* smallest i with prefix > r */
    int pos = 0, step = 1;
    while (step * 2 <= w->n) step *= 2;
    for (; step; step >>= 1)
        if (pos + step <= w->n && w->f[pos + step] <= r) { pos += step; r -= w->f[pos]; }
    return pos;
}

/* ---------------------------------------------------------------- graph */
static int N, NMAX;
static int **adj, *deg, *cap;
static long E;

static uint64_t *H; static int32_t *HV; static uint64_t HMASK;   /* edge -> id, key 0 = empty */
static inline uint64_t ekey(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    return ((uint64_t)(a + 1) << 32) | (uint64_t)(b + 1);
}
static inline uint64_t hmix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; return x; }
static long hslot(int a, int b) {
    uint64_t k = ekey(a, b), i = hmix(k) & HMASK;
    while (H[i]) { if (H[i] == k) return (long)i; i = (i + 1) & HMASK; }
    return -1;
}
static inline int has_edge(int a, int b) { return hslot(a, b) >= 0; }
static inline int edge_id(int a, int b) { return HV[hslot(a, b)]; }
static void hash_insert(int a, int b, int id) {
    uint64_t k = ekey(a, b), i = hmix(k) & HMASK;
    while (H[i]) i = (i + 1) & HMASK;
    H[i] = k; HV[i] = id;
}
static void hash_delete(int a, int b) {                 /* linear probing, backward shift */
    uint64_t i = (uint64_t)hslot(a, b), j = i;
    H[i] = 0;
    for (;;) {
        j = (j + 1) & HMASK;
        if (!H[j]) return;
        uint64_t h = hmix(H[j]) & HMASK;
        if ((j > i && (h <= i || h > j)) || (j < i && h <= i && h > j)) {
            H[i] = H[j]; HV[i] = HV[j]; H[j] = 0; i = j;
        }
    }
}

/* triad centres: Fenwick over all-wedge weights k(k-1)/2 */
static fenwick FW; static int64_t Wall;
/* links: id -> endpoints, triangles t_e; Fenwicks over t_e and over [t_e > 0] */
static int *ea, *eb, *te, *freeid, nfree, nid, TE;
static fenwick FT, FI; static int64_t Tsum3, Nin;       /* sum t_e = 3T, links with t_e > 0 */
static long T;                                          /* triangles */

static void te_add(int id, int d) {
    int old = te[id]; te[id] += d;
    fw_add(&FT, id, d); Tsum3 += d;
    int in = (te[id] > 0) - (old > 0);
    if (in) { fw_add(&FI, id, in); Nin += in; }
}
static int common(int a, int b) {
    if (deg[a] > deg[b]) { int t = a; a = b; b = t; }
    int c = 0;
    for (int i = 0; i < deg[a]; i++) if (has_edge(adj[a][i], b)) c++;
    return c;
}
/* add d to t_e of the two other links of every triangle on (a,b); returns count */
static int touch_triangles(int a, int b, int d) {
    if (deg[a] > deg[b]) { int t = a; a = b; b = t; }
    int c = 0;
    for (int i = 0; i < deg[a]; i++) {
        int x = adj[a][i]; long sl = hslot(x, b);
        if (sl < 0) continue;
        c++;
        if (TE) { te_add(HV[sl], d); te_add(edge_id(a, x), d); }
    }
    return c;
}
static void push(int a, int b) {
    if (deg[a] == cap[a]) { cap[a] = cap[a] ? 2 * cap[a] : 4; adj[a] = realloc(adj[a], cap[a] * sizeof(int)); }
    adj[a][deg[a]++] = b;
}
static void pull(int a, int b) {
    for (int i = 0; i < deg[a]; i++) if (adj[a][i] == b) { adj[a][i] = adj[a][--deg[a]]; return; }
}
static void add_edge(int a, int b) {
    int c = touch_triangles(a, b, +1);
    T += c;
    fw_add(&FW, a, deg[a]); fw_add(&FW, b, deg[b]);     /* k(k-1)/2 grows by k */
    Wall += deg[a] + deg[b];
    push(a, b); push(b, a);
    int id = nfree ? freeid[--nfree] : nid++;
    ea[id] = a; eb[id] = b; te[id] = 0;
    if (TE) te_add(id, c);
    hash_insert(a, b, id); E++;
}
static void remove_edge(int a, int b) {
    int id = edge_id(a, b);
    if (TE) te_add(id, -te[id]);
    hash_delete(a, b);
    pull(a, b); pull(b, a);
    T -= touch_triangles(a, b, -1);
    fw_add(&FW, a, -deg[a]); fw_add(&FW, b, -deg[b]);   /* k(k-1)/2 falls by k-1 (deg already k-1) */
    Wall -= deg[a] + deg[b];
    freeid[nfree++] = id; E--;
}
static int add_node(void) { return N++; }

/* uniformly random open triad; returns 0 if none found */
static int sample_open(int *pa, int *pb) {
    for (int tries = 0; tries < 1000000; tries++) {
        int c = fw_find(&FW, (int64_t)randint((uint64_t)Wall));
        int i = randint(deg[c]), j = randint(deg[c] - 1);
        if (j >= i) j++;
        int a = adj[c][i], b = adj[c][j];
        if (!has_edge(a, b)) { *pa = a; *pb = b; return 1; }
    }
    return 0;
}

/* local rules: random introducer (c) or friend of a friend (f); linked = 0
 * looks for an unlinked pair (closure), linked = 1 for a linked one (removal) */
static int sample_local(char mode, int linked, int *pa, int *pb) {
    if (linked && T == 0) return 0;
    for (int tries = 0; tries < 1000000; tries++) {
        int c = randint(N);
        if (mode == 'c') {
            if (deg[c] < 2) continue;
            int i = randint(deg[c]), j = randint(deg[c] - 1);
            if (j >= i) j++;
            *pa = adj[c][i]; *pb = adj[c][j];
        } else {
            if (deg[c] < 1) continue;
            int j = adj[c][randint(deg[c])];
            if (deg[j] < 2) continue;
            int l = adj[j][randint(deg[j])];
            if (l == c) continue;
            *pa = c; *pb = l;
        }
        if (has_edge(*pa, *pb) == linked) return 1;
    }
    return 0;
}

/* removal under the global rules: link with probability ~ t_e (w, r) or
 * uniform among links with t_e > 0 (p) */
static int sample_closed(char mode, int *pa, int *pb) {
    int id;
    if (mode == 'p') { if (!Nin) return 0; id = fw_find(&FI, (int64_t)randint((uint64_t)Nin)); }
    else { if (!Tsum3) return 0; id = fw_find(&FT, (int64_t)randint((uint64_t)Tsum3)); }
    *pa = ea[id]; *pb = eb[id];
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: tc MODE PARAM N SEED OUTPREFIX [PARAM2]\n"); return 1; }
    char mode = argv[1][0];
    double par = atof(argv[2]), par2 = argc > 6 ? atof(argv[6]) : 0;
    NMAX = atoi(argv[3]);
    seed_rng(strtoull(argv[4], 0, 10));
    const char *pre = argv[5];
    int local = mode == 'c' || mode == 'f';
    TE = par2 > 0 && !local;                     /* t_e bookkeeping only when needed */

    long EMAX = mode == 'r' ? 10L * NMAX + 16
              : mode == 'l' ? 2L * NMAX + 16 : (long)(NMAX * (1 + par / (1 - par - par2))) + 16;
    adj = calloc(NMAX, sizeof(int *)); deg = calloc(NMAX, sizeof(int)); cap = calloc(NMAX, sizeof(int));
    uint64_t hs = 1; while (hs < (uint64_t)(2 * EMAX)) hs <<= 1;
    H = calloc(hs, sizeof(uint64_t)); HV = calloc(hs, sizeof(int32_t)); HMASK = hs - 1;
    fw_init(&FW, NMAX);
    ea = malloc(EMAX * sizeof(int)); eb = malloc(EMAX * sizeof(int)); te = calloc(EMAX, sizeof(int));
    freeid = malloc(EMAX * sizeof(int));
    if (TE) { fw_init(&FT, EMAX); fw_init(&FI, EMAX); }

    char fn[512];
    snprintf(fn, sizeof fn, "%s.ts", pre); FILE *fts = fopen(fn, "w");
    fprintf(fts, "N E kmean T Wall transitivity kmax closures removals\n");

    /* seed: a path of three nodes (one open triad) */
    add_node(); add_node(); add_node(); add_edge(0, 1); add_edge(1, 2);
    if (mode == 'l') { add_edge(0, 2); }       /* LS starts from a triangle */
    long closures = 0, failed = 0, removals = 0, failed_rem = 0;
    double nextck = 10;
    int kmax = 2;

    while (N < NMAX) {
        int close = 0, open = 0;
        if (mode == 'w' || mode == 'p' || local) {
            double x = unif();
            close = x < par; open = !close && x < par + par2;
        } else if (mode == 'r') {
            double Wo = (double)(Wall - 3 * T), x = unif() * (N + par * Wo + par2 * 3.0 * T);
            close = x < par * Wo; open = !close && x < par * Wo + par2 * 3.0 * T;
            if ((double)E / N > 10) break;
        }
        if (mode == 'l') {
            int j = randint(N), l = adj[j][randint(deg[j])], i = add_node();
            add_edge(i, j); add_edge(i, l);
        } else if (close) {
            int a, b, ok;
            for (;;) {
                if (local) { ok = sample_local(mode, 0, &a, &b); break; }
                ok = sample_open(&a, &b);
                if (!ok || mode != 'p') break;
                if (unif() * common(a, b) < 1.0) break;   /* 1/(common neighbours) */
            }
            if (ok) { add_edge(a, b); closures++; }
            else { failed++; int j = randint(N), i = add_node(); add_edge(i, j); }
        } else if (open) {
            int a, b;
            int ok = local ? sample_local(mode, 1, &a, &b) : sample_closed(mode, &a, &b);
            if (ok) { remove_edge(a, b); removals++; } else failed_rem++;
        } else {
            int j = randint(N), i = add_node(); add_edge(i, j);
        }
        if (N >= nextck) {
            kmax = 0;                               /* degrees can now fall */
            for (int i = 0; i < N; i++) if (deg[i] > kmax) kmax = deg[i];
            fprintf(fts, "%d %ld %.6f %ld %lld %.6f %d %ld %ld\n", N, E, 2.0 * E / N, T,
                    (long long)Wall, Wall ? 3.0 * T / Wall : 0, kmax, closures, removals);
            fflush(fts);
            nextck *= pow(10, 0.1);
            if (nextck < N + 1) nextck = N + 1;
        }
    }
    fclose(fts);
    if (getenv("CHECK") && TE) {                /* t_e bookkeeping against a recount */
        long bad = 0, live = 0;
        for (int a = 0; a < N; a++)
            for (int i = 0; i < deg[a]; i++) {
                int b = adj[a][i];
                if (b > a) { live++; if (te[edge_id(a, b)] != common(a, b)) bad++; }
            }
        fprintf(stderr, "check: links %ld (E %ld), wrong t_e %ld, sum t_e %lld vs 3T %ld\n",
                live, E, bad, (long long)Tsum3, 3 * T);
    }
    if (getenv("TSONLY")) return 0;

    /* per-node triangles, neighbour-degree sums, open endpoint triads */
    long *t = calloc(N, sizeof(long));
    for (int a = 0; a < N; a++)
        for (int i = 0; i < deg[a]; i++) {
            int b = adj[a][i];
            if (b > a) { int c = common(a, b); t[a] += c; t[b] += c; }
        }
    double sxy = 0, sx = 0, sx2 = 0;
    snprintf(fn, sizeof fn, "%s.nodes", pre); FILE *fn_ = fopen(fn, "w");
    for (int a = 0; a < N; a++) {
        long knn = 0;
        for (int i = 0; i < deg[a]; i++) {
            int b = adj[a][i]; knn += deg[b];
            if (b > a) { sxy += (double)deg[a] * deg[b]; sx += 0.5 * (deg[a] + deg[b]);
                         sx2 += 0.5 * ((double)deg[a] * deg[a] + (double)deg[b] * deg[b]); }
        }
        t[a] /= 2;                              /* each triangle seen via two edges */
        long w = knn - deg[a] - 2 * t[a];
        fprintf(fn_, "%d %ld %ld %ld\n", deg[a], t[a], knn, w);
    }
    fclose(fn_);
    snprintf(fn, sizeof fn, "%s.edges", pre); FILE *fe = fopen(fn, "wb");
    for (int a = 0; a < N; a++)
        for (int i = 0; i < deg[a]; i++)
            if (adj[a][i] > a) { int32_t e2[2] = {a, adj[a][i]}; fwrite(e2, sizeof e2, 1, fe); }
    fclose(fe);
    double r = (sxy / E - pow(sx / E, 2)) / (sx2 / E - pow(sx / E, 2));
    snprintf(fn, sizeof fn, "%s.sum", pre); FILE *fs = fopen(fn, "w");
    fprintf(fs, "mode %c param %g param2 %g N %d E %ld kmean %.6f T %ld transitivity %.6f assort %.6f "
            "closures %ld failed %ld removals %ld failed_rem %ld\n",
            mode, par, par2, N, E, 2.0 * E / N, T, Wall ? 3.0 * T / Wall : 0, r, closures, failed,
            removals, failed_rem);
    fclose(fs);
    return 0;
}
