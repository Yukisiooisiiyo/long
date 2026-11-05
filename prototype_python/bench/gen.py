"""Instance generators for benchmarks. All return lp.graph.Graph."""
import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lp.graph import Graph  # noqa: E402


def _graph(edges):
    used = sorted({x for u, v, _ in edges for x in (u, v)})
    idx = {v: i for i, v in enumerate(used)}
    d = {}
    for u, v, w in edges:
        if u != v:
            k = (min(idx[u], idx[v]), max(idx[u], idx[v]))
            d[k] = max(d.get(k, 0), w)
    return Graph(len(used), [(u, v, w) for (u, v), w in d.items()])


def railway(n, seed, extra_ratio=0.25, branch_frac=0.25):
    """Planar-ish rail map: geometric MST + short extra links + pendant branch lines."""
    rng = random.Random(seed)
    core = int(n * (1 - branch_frac))
    pts = [(rng.random(), rng.random()) for _ in range(core)]
    dist = lambda a, b: round(100 * math.dist(pts[a], pts[b]), 2) + 0.01
    pairs = sorted((dist(a, b), a, b) for a in range(core) for b in range(a + 1, core))
    parent = list(range(core))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    rest = []
    for w, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            edges.append((a, b, w))
        else:
            rest.append((w, a, b))
    # extra links: among the shortest remaining pairs (keeps it planar-ish)
    k = int(core * extra_ratio) + 1
    cand = rest[: core * 3]
    rng.shuffle(cand)
    for w, a, b in sorted(cand[: k * 2])[:k]:
        edges.append((a, b, w))
    nid = core
    while nid < n:
        a = rng.randrange(core)
        length = rng.randint(1, 4)
        prev = a
        for _ in range(length):
            if nid >= n:
                break
            edges.append((prev, nid, round(rng.uniform(3, 30), 2)))
            prev = nid
            nid += 1
    return _graph(edges)


def grid(r, c, seed):
    rng = random.Random(seed)
    edges = []
    for i in range(r):
        for j in range(c):
            v = i * c + j
            if j + 1 < c:
                edges.append((v, v + 1, round(rng.uniform(1, 10), 2)))
            if i + 1 < r:
                edges.append((v, v + c, round(rng.uniform(1, 10), 2)))
    return _graph(edges)


def cactus(n, seed):
    """Small cycles (3-7 vertices) glued at vertices: many small blocks."""
    rng = random.Random(seed)
    edges = []
    verts = [0]
    nxt = 1
    while nxt < n:
        a = rng.choice(verts)
        k = rng.randint(2, 6)
        cyc = [a] + list(range(nxt, min(n, nxt + k)))
        nxt += len(cyc) - 1
        for x, y in zip(cyc, cyc[1:]):
            edges.append((x, y, round(rng.uniform(1, 10), 2)))
        if len(cyc) >= 3:
            edges.append((cyc[-1], a, round(rng.uniform(1, 10), 2)))
        # occasional chord makes a block slightly denser
        if len(cyc) >= 5 and rng.random() < 0.5:
            edges.append((cyc[1], cyc[-2], round(rng.uniform(1, 10), 2)))
        verts += cyc[1:]
    return _graph(edges)


def dense(n, seed, p=0.5):
    rng = random.Random(seed)
    edges = [(u, v, round(rng.uniform(1, 10), 2)) for u in range(n) for v in range(u + 1, n)
             if rng.random() < p]
    return _graph(edges)


def to_text(g):
    return "".join(f"{u + 1}, {v + 1}, {w}\r\n" for u, v, w in g.edges)
