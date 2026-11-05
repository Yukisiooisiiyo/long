"""Graph basics: parsing, components, biconnected decomposition, chain compression."""
from __future__ import annotations


class InputError(Exception):
    pass


class Graph:
    """Undirected weighted simple graph. Vertices 0..n-1, edges list of (u, v, w)."""

    def __init__(self, n, edges):
        self.n = n
        self.edges = list(edges)
        self.adj = [[] for _ in range(n)]  # (v, w, eid)
        for i, (u, v, w) in enumerate(self.edges):
            self.adj[u].append((v, w, i))
            self.adj[v].append((u, w, i))
        self.adjmask = [0] * n
        for u in range(n):
            m = 0
            for v, _, _ in self.adj[u]:
                m |= 1 << v
            self.adjmask[u] = m
        self.wmap = {}
        for u, v, w in self.edges:
            self.wmap[(u, v)] = w
            self.wmap[(v, u)] = w

    def weight(self, u, v):
        return self.wmap.get((u, v))

    def induced(self, vertices):
        """Return (subgraph, local->global list) for a vertex subset."""
        vs = sorted(vertices)
        idx = {v: i for i, v in enumerate(vs)}
        es = [(idx[u], idx[v], w) for u, v, w in self.edges if u in idx and v in idx]
        return Graph(len(vs), es), vs


def parse(text):
    """Parse 'u, v, w' lines. Returns (ids, Graph) where ids[i] is the original ID string."""
    ids = []
    index = {}
    best = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().rstrip("\r")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise InputError(f"line {lineno}: expected 3 fields")
        a, b, ws = parts
        try:
            ia, ib = int(a), int(b)
            w = float(ws)
        except ValueError:
            raise InputError(f"line {lineno}: not a number")
        if ia <= 0 or ib <= 0:
            raise InputError(f"line {lineno}: IDs must be positive integers")
        if w < 0 or w != w:
            raise InputError(f"line {lineno}: negative distance")
        if ia == ib:
            continue  # self loop ignored
        for x in (ia, ib):
            if x not in index:
                index[x] = len(ids)
                ids.append(x)
        u, v = index[ia], index[ib]
        key = (min(u, v), max(u, v))
        if key not in best or best[key] < w:
            best[key] = w
    edges = [(u, v, w) for (u, v), w in best.items()]
    return ids, Graph(len(ids), edges)


def components(g):
    seen = [False] * g.n
    comps = []
    for s in range(g.n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [s]
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _, _ in g.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def biconnected(g):
    """Iterative Tarjan. Returns (blocks, articulation_set); block = (vertex_set, edge_id_list)."""
    n = g.n
    disc = [-1] * n
    low = [0] * n
    t = 0
    blocks = []
    art = set()
    estack = []
    for r in range(n):
        if disc[r] != -1:
            continue
        disc[r] = low[r] = t
        t += 1
        if not g.adj[r]:
            blocks.append(({r}, []))
            continue
        root_children = 0
        stack = [(r, -1, iter(g.adj[r]))]
        while stack:
            u, pe, it = stack[-1]
            advanced = False
            for v, _, eid in it:
                if eid == pe:
                    continue
                if disc[v] == -1:
                    estack.append(eid)
                    disc[v] = low[v] = t
                    t += 1
                    stack.append((v, eid, iter(g.adj[v])))
                    advanced = True
                    break
                elif disc[v] < disc[u]:
                    estack.append(eid)
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                if low[u] < low[p]:
                    low[p] = low[u]
                if low[u] >= disc[p]:
                    es = []
                    while True:
                        e = estack.pop()
                        es.append(e)
                        if e == pe:
                            break
                    vs = set()
                    for e in es:
                        a, b, _ = g.edges[e]
                        vs.add(a)
                        vs.add(b)
                    blocks.append((vs, es))
                    if len(stack) > 1:
                        art.add(p)
                    else:
                        root_children += 1
        if root_children >= 2:
            art.add(r)
    return blocks, art


class CGraph:
    """Compressed graph. Kept vertices 0..n-1 (orig[i] = vertex in source graph).
    Edge: (u, v, total, seg_u, seg_v, interior) where interior lists source vertices from u to v.
    Partial value from u (stop at last interior before v) = total - seg_v (only if interior)."""

    def __init__(self, n, orig, edges):
        self.n = n
        self.orig = orig
        self.edges = edges
        self.adj = [[] for _ in range(n)]  # (v, eid)
        for i, (u, v, *_rest) in enumerate(edges):
            self.adj[u].append((v, i))
            self.adj[v].append((u, i))
        self.adjmask = [0] * n
        for u in range(n):
            m = 0
            for v, _ in self.adj[u]:
                m |= 1 << v
            self.adjmask[u] = m
        self.top1 = [0.0] * n
        self.top2 = [0.0] * n
        for u in range(n):
            ws = sorted((edges[e][2] for _, e in self.adj[u]), reverse=True)
            if ws:
                self.top1[u] = ws[0]
            if len(ws) > 1:
                self.top2[u] = ws[1]
        self.max_partial = 0.0
        self.partial_from = [[] for _ in range(n)]  # (eid, other_end, value)
        for i, (u, v, tot, su, sv, inter) in enumerate(edges):
            if inter:
                pu = tot - sv
                pv = tot - su
                self.partial_from[u].append((i, v, pu))
                self.partial_from[v].append((i, u, pv))
                self.max_partial = max(self.max_partial, pu, pv)
        self.wmax = max((e[2] for e in edges), default=0.0)

    def expand_edge(self, eid, frm):
        u, v, tot, su, sv, inter = self.edges[eid]
        if frm == u:
            return [self.orig[u]] + inter + [self.orig[v]]
        return [self.orig[v]] + inter[::-1] + [self.orig[u]]

    def partial_vertices(self, eid, frm):
        """Source vertices visited by a partial move from frm (excluding frm)."""
        full = self.expand_edge(eid, frm)
        return full[1:-1]


def identity_cgraph(g):
    edges = [(u, v, w, w, w, []) for u, v, w in g.edges]
    return CGraph(g.n, list(range(g.n)), edges)


def compress(g, protected=()):
    """Compress maximal chains of degree-2 vertices of a connected graph g.
    Vertices with degree != 2 or in `protected` are kept. Pure cycles are left uncompressed."""
    deg = [len(g.adj[u]) for u in range(g.n)]
    keep = set(u for u in range(g.n) if deg[u] != 2) | set(protected)
    if not keep:
        return identity_cgraph(g)
    while True:
        chains = []
        seen_e = set()
        redo = False
        for u in sorted(keep):
            for v, w, eid in g.adj[u]:
                if eid in seen_e:
                    continue
                seen_e.add(eid)
                inter = []
                segs = [w]
                prev, cur = u, v
                last_e = eid
                while cur not in keep:
                    inter.append(cur)
                    nxt = None
                    for x, wx, ex in g.adj[cur]:
                        if ex != last_e:
                            nxt = (x, wx, ex)
                            break
                    seen_e.add(nxt[2])
                    prev, cur, last_e = cur, nxt[0], nxt[2]
                    segs.append(nxt[1])
                if cur == u and inter:
                    keep.add(inter[len(inter) // 2])
                    redo = True
                    break
                chains.append((u, cur, sum(segs), segs[0], segs[-1], inter))
            if redo:
                break
        if not redo:
            break
    order = sorted(keep)
    idx = {v: i for i, v in enumerate(order)}
    edges = [(idx[a], idx[b], tot, sa, sb, inter) for a, b, tot, sa, sb, inter in chains]
    return CGraph(len(order), order, edges)


def path_length(g, path, cycle=False):
    tot = 0.0
    for a, b in zip(path, path[1:]):
        w = g.weight(a, b)
        if w is None:
            raise ValueError(f"no edge {a}-{b}")
        tot += w
    return tot


def verify(g, path):
    """Validate a path/cycle in g. Cycle = first == last. Returns length."""
    if not path:
        return 0.0
    cyc = len(path) > 2 and path[0] == path[-1]
    body = path[:-1] if cyc else path
    if len(set(body)) != len(body):
        raise ValueError("repeated vertex")
    if cyc and len(body) < 3:
        raise ValueError("cycle too short")
    return path_length(g, path)
