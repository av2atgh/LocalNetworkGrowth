/* cyclecount.c -- exact simple-cycle counts for lengths 3..LMAX.
 *
 * Each cycle is enumerated from its smallest vertex, in both directions, so the
 * tallies are halved at the end.  This is the same convention as `true_cycles`
 * in motif_spectrum.py; that Python version is the reference implementation and
 * this one exists only because the enumeration is exponential in L and the real
 * networks are too big for it.
 *
 * The one addition over the Python is distance pruning: a partial path of
 * length l cannot close into a cycle of length <= LMAX unless the current
 * vertex is within LMAX - l of the start, in the subgraph induced on vertices
 * >= s.  On sparse graphs this is the difference between hours and seconds.
 *
 * Enumeration cost is bounded below by the answer itself, so a graph with
 * 10^12 eight-cycles is out of reach whatever the pruning; -t gives up cleanly
 * instead of running forever.
 *
 *   input   (stdin, text)  "n m" then m lines "u v", 0-indexed, undirected,
 *                          simple, no self-loops
 *   output  (stdout)       one line "L count" for L = 3..LMAX, then "status ok"
 *                          or "status timeout"
 *
 *   usage   cyclecount [-l LMAX] [-t SECONDS] < edges.txt
 *
 * build: cc -O3 -o cyclecount cyclecount.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXL 12

static int n, m;
static int *deg, *off, *adj;      /* CSR adjacency */
static int *dist, *bfsq;          /* BFS scratch, distances from the start */
static char *onpath;              /* vertex is on the current path */
static unsigned long long cnt[MAXL + 1];
static int LMAX = 8;
static double tlimit = 0.0;       /* 0 = no limit */
static clock_t t0;
static int timed_out = 0;
static long long ticks = 0;

/* BFS from s over the induced subgraph on {v : v >= s}, depths up to LMAX. */
static void bfs(int s)
{
    int head = 0, tail = 0, v, i, w;
    /* dist is reset lazily: only vertices touched last time need clearing, but
       an O(n) memset per start is already negligible next to the DFS. */
    for (i = s; i < n; i++) dist[i] = -1;
    dist[s] = 0;
    bfsq[tail++] = s;
    while (head < tail) {
        v = bfsq[head++];
        if (dist[v] >= LMAX) continue;
        for (i = off[v]; i < off[v + 1]; i++) {
            w = adj[i];
            if (w < s || dist[w] >= 0) continue;
            dist[w] = dist[v] + 1;
            bfsq[tail++] = w;
        }
    }
}

/* Extend the path ending at v, which has `len` edges from the start s. */
static void dfs(int s, int v, int len)
{
    int i, w, budget = LMAX - len;

    if (++ticks >= 1000000) {
        ticks = 0;
        if (tlimit > 0.0 &&
            (double)(clock() - t0) / CLOCKS_PER_SEC > tlimit) timed_out = 1;
    }
    if (timed_out) return;

    for (i = off[v]; i < off[v + 1]; i++) {
        w = adj[i];
        if (w == s) {
            if (len >= 2) cnt[len + 1]++;   /* closed a cycle of len+1 edges */
            continue;
        }
        if (w < s || onpath[w]) continue;
        if (budget < 1) continue;
        /* need to get back to s afterwards: 1 edge to w, then dist[w] more */
        if (dist[w] < 0 || dist[w] > budget - 1) continue;
        onpath[w] = 1;
        dfs(s, w, len + 1);
        onpath[w] = 0;
        if (timed_out) return;
    }
}

int main(int argc, char **argv)
{
    int i, a, b, *ea, *eb;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-l") && i + 1 < argc) LMAX = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tlimit = atof(argv[++i]);
        else { fprintf(stderr, "usage: %s [-l LMAX] [-t SECONDS]\n", argv[0]); return 2; }
    }
    if (LMAX < 3 || LMAX > MAXL) { fprintf(stderr, "LMAX out of range\n"); return 2; }

    if (scanf("%d %d", &n, &m) != 2) { fprintf(stderr, "bad header\n"); return 2; }
    ea = malloc(sizeof(int) * m);
    eb = malloc(sizeof(int) * m);
    deg = calloc(n + 1, sizeof(int));
    for (i = 0; i < m; i++) {
        if (scanf("%d %d", &a, &b) != 2) { fprintf(stderr, "bad edge %d\n", i); return 2; }
        ea[i] = a; eb[i] = b; deg[a]++; deg[b]++;
    }
    off = malloc(sizeof(int) * (n + 1));
    off[0] = 0;
    for (i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    adj = malloc(sizeof(int) * 2 * m);
    memset(deg, 0, sizeof(int) * (n + 1));
    for (i = 0; i < m; i++) {
        adj[off[ea[i]] + deg[ea[i]]++] = eb[i];
        adj[off[eb[i]] + deg[eb[i]]++] = ea[i];
    }
    free(ea); free(eb);

    dist = malloc(sizeof(int) * n);
    bfsq = malloc(sizeof(int) * n);
    onpath = calloc(n, sizeof(char));

    t0 = clock();
    for (i = 0; i < n && !timed_out; i++) {
        bfs(i);
        onpath[i] = 1;
        dfs(i, i, 0);
        onpath[i] = 0;
    }

    for (i = 3; i <= LMAX; i++)
        printf("%d %llu\n", i, cnt[i] / 2);
    printf("status %s\n", timed_out ? "timeout" : "ok");
    return 0;
}
