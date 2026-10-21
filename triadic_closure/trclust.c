/* Edge-connected triangle clusters (3-clique communities).
 *
 *   cc -O3 -o trclust trclust.c
 *   ./trclust EDGES N
 *
 * Two links are in the same cluster when they lie in a common triangle, and
 * clusters are closed under this relation (union-find over links).  Links in no
 * triangle are left out.  Prints one line:
 *   T clusters T_max N_max E_max
 * T: triangles; clusters: number of clusters; T_max, N_max, E_max: triangles,
 * nodes and links of the cluster with most triangles.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int *par;
static int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
static void unite(int a, int b) { a = find(a); b = find(b); if (a != b) { if (a < b) par[b] = a; else par[a] = b; } }

static long *off;
static int *nb, *eid;                                   /* CSR, neighbours sorted, link id per slot */
static long pos(int a, int b) {                         /* slot of b in a's list, -1 if absent */
    long lo = off[a], hi = off[a + 1] - 1;
    while (lo <= hi) { long m = (lo + hi) / 2; if (nb[m] == b) return m; if (nb[m] < b) lo = m + 1; else hi = m - 1; }
    return -1;
}
static int cmp64(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: trclust EDGES N\n"); return 1; }
    int n = atoi(argv[2]);
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long E = ftell(f) / 8; rewind(f);
    int32_t *e = malloc(E * 8);
    if (fread(e, 8, E, f) != (size_t)E) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    off = calloc(n + 1, sizeof(long));
    for (long i = 0; i < E; i++) { off[e[2 * i] + 1]++; off[e[2 * i + 1] + 1]++; }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    uint64_t *pk = malloc(2 * E * sizeof(uint64_t));    /* neighbour << 32 | link id, sorted per node */
    long *fill = calloc(n, sizeof(long));
    for (long i = 0; i < E; i++) {
        int a = e[2 * i], b = e[2 * i + 1];
        pk[off[a] + fill[a]++] = ((uint64_t)b << 32) | (uint64_t)i;
        pk[off[b] + fill[b]++] = ((uint64_t)a << 32) | (uint64_t)i;
    }
    nb = malloc(2 * E * sizeof(int)); eid = malloc(2 * E * sizeof(int));
    for (int a = 0; a < n; a++) {
        qsort(pk + off[a], off[a + 1] - off[a], sizeof(uint64_t), cmp64);
        for (long s = off[a]; s < off[a + 1]; s++) { nb[s] = (int)(pk[s] >> 32); eid[s] = (int)(pk[s] & 0xffffffffu); }
    }
    free(pk); free(fill);

    /* every triangle seen once, from its link (a,b) with a<b and apex x>b */
    par = malloc(E * sizeof(int)); for (long i = 0; i < E; i++) par[i] = (int)i;
    int *te = calloc(E, sizeof(int));
    long T = 0;
    for (long i = 0; i < E; i++) {
        int a = e[2 * i], b = e[2 * i + 1];
        if (a > b) { int t = a; a = b; b = t; }
        int s = (off[a + 1] - off[a] < off[b + 1] - off[b]) ? a : b, o = s == a ? b : a;
        for (long k = off[s]; k < off[s + 1]; k++) {
            int x = nb[k];
            if (x <= b) continue;
            long q = pos(o, x);
            if (q < 0) continue;
            T++; te[i]++; te[eid[k]]++; te[eid[q]]++;
            unite((int)i, eid[k]); unite((int)i, eid[q]);
        }
    }
    /* per cluster: triangles = sum t_e / 3, links, nodes */
    long *ct = calloc(E, sizeof(long)), *ce = calloc(E, sizeof(long));
    long clusters = 0;
    for (long i = 0; i < E; i++) if (te[i]) { int r = find((int)i); if (!ce[r]) clusters++; ct[r] += te[i]; ce[r]++; }
    long best = -1;
    for (long i = 0; i < E; i++) if (ce[i] && (best < 0 || ct[i] > ct[best])) best = i;
    long nmax = 0, emax = 0, tmax = 0;
    if (best >= 0) {
        char *seen = calloc(n, 1);
        for (long i = 0; i < E; i++)
            if (te[i] && find((int)i) == best) {
                int a = e[2 * i], b = e[2 * i + 1];
                if (!seen[a]) { seen[a] = 1; nmax++; }
                if (!seen[b]) { seen[b] = 1; nmax++; }
            }
        emax = ce[best]; tmax = ct[best] / 3;
    }
    printf("%ld %ld %ld %ld %ld\n", T, clusters, tmax, nmax, emax);
    return 0;
}
