"""Lagrangian relaxation of the degree<=2 constraint: max-weight forest with penalties."""
from .graph import CGraph


def forest(cg, lam):
    parent = list(range(cg.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mods = []
    for i, (u, v, tot, *_r) in enumerate(cg.edges):
        m = tot - lam[u] - lam[v]
        if m > 0:
            mods.append((m, i))
    mods.sort(reverse=True)
    chosen = []
    val = 0.0
    for m, i in mods:
        u, v = cg.edges[i][0], cg.edges[i][1]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            chosen.append(i)
            val += m
    return val, chosen


def bound(cg, lam):
    val, chosen = forest(cg, lam)
    return val + 2 * sum(lam) + cg.max_partial, chosen


def subgradient(cg, lb, iters=100, init="zero"):
    n = cg.n
    if init == "third":
        lam = []
        for u in range(n):
            ws = sorted((cg.edges[e][2] for _, e in cg.adj[u]), reverse=True)
            lam.append(ws[2] if len(ws) > 2 else 0.0)
    else:
        lam = [0.0] * n
    best_ub, chosen = bound(cg, lam)
    best_lam = lam[:]
    best_chosen = chosen
    history = [best_ub]
    theta = 2.0
    stall = 0
    for _ in range(iters):
        ub, chosen = bound(cg, lam)
        deg = [0] * n
        for i in chosen:
            deg[cg.edges[i][0]] += 1
            deg[cg.edges[i][1]] += 1
        g = [d - 2 for d in deg]
        # projected subgradient: components that cannot move (lam=0 and g<0) are dropped
        g = [gi if (gi > 0 or lam[i] > 0) else 0 for i, gi in enumerate(g)]
        norm = sum(x * x for x in g)
        if ub < best_ub - 1e-12:
            best_ub, best_lam, best_chosen = ub, lam[:], chosen
            stall = 0
        else:
            stall += 1
            if stall >= 5:
                theta /= 2
                stall = 0
        history.append(best_ub)
        if norm == 0 or ub <= lb + 1e-9 or theta < 1e-4:
            break
        step = theta * (ub - max(lb, 0.0)) / norm
        lam = [max(0.0, lam[i] + step * g[i]) for i in range(n)]
    return best_lam, best_ub, best_chosen, history


def forest_paths(cg, chosen):
    """Repair forest to degree<=2 (drop lightest edges at high-degree vertices) and return
    compressed-vertex paths with their edge ids, longest first."""
    adj = {}
    for i in chosen:
        u, v = cg.edges[i][0], cg.edges[i][1]
        adj.setdefault(u, []).append(i)
        adj.setdefault(v, []).append(i)
    keep = set(chosen)
    for u, es in adj.items():
        es = [e for e in es if e in keep]
        if len(es) > 2:
            es.sort(key=lambda e: -cg.edges[e][2])
            for e in es[2:]:
                keep.discard(e)
    nbr = {}
    for i in keep:
        u, v = cg.edges[i][0], cg.edges[i][1]
        nbr.setdefault(u, []).append((v, i))
        nbr.setdefault(v, []).append((u, i))
    seen = set()
    paths = []
    for s in nbr:
        if s in seen or len(nbr[s]) != 1:
            continue
        seq = [s]
        es = []
        seen.add(s)
        prev_e = None
        cur = s
        while True:
            nxt = [(v, e) for v, e in nbr[cur] if e != prev_e]
            if not nxt:
                break
            v, e = nxt[0]
            seq.append(v)
            es.append(e)
            seen.add(v)
            prev_e = e
            cur = v
        L = sum(cg.edges[e][2] for e in es)
        paths.append((L, seq, es))
    paths.sort(key=lambda x: -x[0])
    return paths


def max_mod(cg, lam):
    return max([0.0] + [e[2] - lam[e[0]] - lam[e[1]] for e in cg.edges])


def cycle_bound(cg, lam):
    """Cycle = forest + one extra edge (Lagrangian 1-forest)."""
    val, chosen = forest(cg, lam)
    cs = set(chosen)
    extra = max([0.0] + [cg.edges[i][2] - lam[cg.edges[i][0]] - lam[cg.edges[i][1]]
                         for i in range(len(cg.edges)) if i not in cs])
    return val + extra + 2 * sum(lam)


def eliminate_edges(cg, lam, chosen, ub, incumbent, cycle_extra=None):
    """Return set of edge ids provably unable to be on a path/cycle longer than incumbent.
    Only plain edges (no interior) are eliminated so partial-chain moves stay intact."""
    n = cg.n
    mod = [cg.edges[i][2] - lam[cg.edges[i][0]] - lam[cg.edges[i][1]] for i in range(len(cg.edges))]
    tadj = [[] for _ in range(n)]
    for i in chosen:
        u, v = cg.edges[i][0], cg.edges[i][1]
        tadj[u].append((v, i))
        tadj[v].append((u, i))
    comp = [-1] * n
    par = [-1] * n
    pe = [-1] * n
    depth = [0] * n
    for r in range(n):
        if comp[r] != -1:
            continue
        comp[r] = r
        stack = [r]
        while stack:
            u = stack.pop()
            for v, i in tadj[u]:
                if comp[v] == -1:
                    comp[v] = r
                    par[v] = u
                    pe[v] = i
                    depth[v] = depth[u] + 1
                    stack.append(v)
    cset = set(chosen)
    dead = set()
    for i, (u, v, tot, su, sv, inter) in enumerate(cg.edges):
        if i in cset or inter:
            continue
        if comp[u] == comp[v]:
            a, b = u, v
            mn = float("inf")
            while a != b:
                if depth[a] >= depth[b]:
                    mn = min(mn, mod[pe[a]])
                    a = par[a]
                else:
                    mn = min(mn, mod[pe[b]])
                    b = par[b]
            be = ub + mod[i] - mn
        else:
            be = ub + min(mod[i], 0.0)
        # any path using e has length <= be; any cycle using e <= be + wmax
        extra = cg.wmax if cycle_extra is None else cycle_extra
        if be + extra < incumbent - 1e-9:
            dead.add(i)
    return dead


def remove_edges(cg, dead):
    edges = [e for i, e in enumerate(cg.edges) if i not in dead]
    return CGraph(cg.n, cg.orig, edges)
