"""Initial-solution heuristics on a plain Graph (one connected component)."""
import random

from .graph import verify


def _better(a, b):
    return a if a[0] >= b[0] else b


def greedy(g):
    best = (-1.0, [])
    for s in range(g.n):
        path = [s]
        vis = {s}
        L = 0.0
        while True:
            u = path[-1]
            cand = [(w, v) for v, w, _ in g.adj[u] if v not in vis]
            if not cand:
                break
            w, v = max(cand)
            path.append(v)
            vis.add(v)
            L += w
        best = _better(best, (L, path))
    return best


def max_spanning_forest(g, weight=None):
    """Kruskal on decreasing weight. weight(eid) -> value; only positive values when given."""
    parent = list(range(g.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if weight is None:
        order = sorted(range(len(g.edges)), key=lambda i: -g.edges[i][2])
    else:
        order = sorted((i for i in range(len(g.edges)) if weight(i) > 0), key=lambda i: -weight(i))
    tree = []
    for i in order:
        u, v, _ = g.edges[i]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append(i)
    return tree


def _tree_adj(g, tree):
    adj = [[] for _ in range(g.n)]
    for i in tree:
        u, v, w = g.edges[i]
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _farthest(adj, s):
    dist = {s: 0.0}
    par = {s: None}
    stack = [s]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w
                par[v] = u
                stack.append(v)
    far = max(dist, key=dist.get)
    path = []
    x = far
    while x is not None:
        path.append(x)
        x = par[x]
    return dist[far], path[::-1]


def mst_diameter(g):
    tree = max_spanning_forest(g)
    adj = _tree_adj(g, tree)
    _, p = _farthest(adj, 0)
    L, path = _farthest(adj, p[-1])
    return L, path


def extend_greedy(g, path):
    path = list(path)
    vis = set(path)
    L = sum(g.weight(a, b) for a, b in zip(path, path[1:]))
    changed = True
    while changed:
        changed = False
        for end in (0, 1):
            u = path[-1] if end else path[0]
            cand = [(w, v) for v, w, _ in g.adj[u] if v not in vis]
            if cand:
                w, v = max(cand)
                if end:
                    path.append(v)
                else:
                    path.insert(0, v)
                vis.add(v)
                L += w
                changed = True
    return L, path


def fundamental_cycles(g, top_m=5):
    """Cycles from max spanning tree + one non-tree edge; then cut each cycle and extend."""
    tree = max_spanning_forest(g)
    tset = set(tree)
    adj = _tree_adj(g, tree)
    par = [-1] * g.n
    depth = [0] * g.n
    pw = [0.0] * g.n
    seen = [False] * g.n
    seen[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if not seen[v]:
                seen[v] = True
                par[v] = u
                depth[v] = depth[u] + 1
                pw[v] = w
                stack.append(v)
    cycles = []
    for i, (u, v, w) in enumerate(g.edges):
        if i in tset:
            continue
        a, b = u, v
        left, right = [a], [b]
        L = w
        while a != b:
            if depth[a] >= depth[b]:
                L += pw[a]
                a = par[a]
                left.append(a)
            else:
                L += pw[b]
                b = par[b]
                right.append(b)
        cyc = left + right[-2::-1]  # u .. lca .. v
        cycles.append((L, cyc))
    cycles.sort(key=lambda x: -x[0])
    best = (-1.0, [])
    for L, cyc in cycles[:top_m]:
        best = _better(best, (L, cyc + [cyc[0]]))
        k = len(cyc)
        for i in range(k):
            opened = cyc[i + 1:] + cyc[:i + 1]  # cut edge (cyc[i], cyc[i+1])
            best = _better(best, extend_greedy(g, opened))
    return best


def beam(g, K=64, alpha=0.3):
    top1 = [max((w for _, w, _ in g.adj[u]), default=0.0) for u in range(g.n)]
    adjm = g.adjmask

    def reach_sum(ends, vis):
        R = 0
        fr = 0
        for e in ends:
            fr |= adjm[e]
        fr &= ~vis
        while fr:
            R |= fr
            nf = 0
            m = fr
            while m:
                b = m & -m
                nf |= adjm[b.bit_length() - 1]
                m ^= b
            fr = nf & ~vis & ~R
        s = 0.0
        m = R
        while m:
            b = m & -m
            s += top1[b.bit_length() - 1]
            m ^= b
        return s

    states = []
    for u, v, w in g.edges:
        vis = (1 << u) | (1 << v)
        states.append((w + alpha * reach_sum((u, v), vis), w, (u, v), vis))
    states.sort(key=lambda s: -s[0])
    beam_ = states[:K]
    best = max(((s[1], list(s[2])) for s in beam_), key=lambda x: x[0], default=(-1.0, []))
    while beam_:
        cand = {}
        for _, L, path, vis in beam_:
            a, b = path[0], path[-1]
            if len(path) >= 3:
                wc = g.weight(a, b)
                if wc is not None and L + wc > best[0]:
                    best = (L + wc, list(path) + [a])
            for end, u in ((0, a), (1, b)):
                for v, w, _ in g.adj[u]:
                    if (vis >> v) & 1:
                        continue
                    np_ = path + (v,) if end else (v,) + path
                    nv = vis | (1 << v)
                    key = (nv, frozenset((np_[0], np_[-1])))
                    nl = L + w
                    if key in cand and cand[key][1] >= nl:
                        continue
                    sc = nl + alpha * reach_sum((np_[0], np_[-1]), nv)
                    cand[key] = (sc, nl, np_, nv)
        if not cand:
            break
        lst = sorted(cand.values(), key=lambda s: -s[0])
        beam_ = lst[:K]
        top = max(beam_, key=lambda s: s[1])
        if top[1] > best[0]:
            best = (top[1], list(top[2]))
    return best


def local_search(g, path):
    """Moves: extend ends, 1/2-vertex detours, Posa rotation (+extend)."""
    L, path = extend_greedy(g, path)
    improved = True
    while improved:
        improved = False
        vis = set(path)
        # detours
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            wuv = g.weight(u, v)
            bestg, ins = 1e-12, None
            for x, wux, _ in g.adj[u]:
                if x in vis:
                    continue
                wxv = g.weight(x, v)
                if wxv is not None and wux + wxv - wuv > bestg:
                    bestg, ins = wux + wxv - wuv, [x]
                for y, wxy, _ in g.adj[x]:
                    if y in vis or y == x:
                        continue
                    wyv = g.weight(y, v)
                    if wyv is not None and wux + wxy + wyv - wuv > bestg:
                        bestg, ins = wux + wxy + wyv - wuv, [x, y]
            if ins:
                path = path[:i + 1] + ins + path[i + 1:]
                L += bestg
                improved = True
                break
        if improved:
            L, path = extend_greedy(g, path)
            continue
        # Posa rotation at both ends
        for rev in (False, True):
            p = path[::-1] if rev else path
            v = p[-1]
            for i in range(len(p) - 2):
                if g.weight(p[i], v) is None:
                    continue
                rot = p[:i + 1] + p[i + 1:][::-1]
                L2, rot2 = extend_greedy(g, rot)
                if L2 > L + 1e-12:
                    L, path = L2, rot2
                    improved = True
                    break
            if improved:
                break
    return L, path


def ils(g, path, iters=50, seed=0):
    rng = random.Random(seed)
    bestL, best = local_search(g, path)
    cur = best
    for _ in range(iters):
        if len(cur) < 3:
            break
        k = rng.randint(1, max(1, len(cur) // 3))
        r = rng.random()
        if r < 0.4:
            p = cur[k:]
        elif r < 0.8:
            p = cur[:-k]
        else:
            i = rng.randint(1, len(cur) - 2)
            a, b = cur[:i], cur[i:]
            p = a if len(a) >= len(b) else b
        L, p = local_search(g, p)
        if L > bestL + 1e-12:
            bestL, best = L, p
            cur = p
        else:
            cur = best
    return bestL, best


def check(g, sol):
    L, p = sol
    if not p:
        return sol
    v = verify(g, p)
    assert abs(v - L) < 1e-6, (v, L, p)
    return sol
