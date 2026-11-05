"""Exact bitmask DP for small blocks: dp[mask][v] = longest path covering exactly mask ending at v."""
NEG = float("-inf")


def _run(g, init_vertices, allowed):
    n = g.n
    size = 1 << n
    dp = [None] * size
    for s in init_vertices:
        m = 1 << s
        if dp[m] is None:
            dp[m] = [NEG] * n
        dp[m][s] = 0.0
    adj = g.adj
    for mask in range(1, size):
        row = dp[mask]
        if row is None or mask & ~allowed:
            continue
        for v in range(n):
            val = row[v]
            if val == NEG:
                continue
            for u, w, _ in adj[v]:
                bit = 1 << u
                if mask & bit or not (allowed & bit):
                    continue
                nm = mask | bit
                nr = dp[nm]
                if nr is None:
                    nr = dp[nm] = [NEG] * n
                if val + w > nr[u]:
                    nr[u] = val + w
    return dp


def _backtrack(g, dp, mask, v):
    path = [v]
    while mask & (mask - 1):
        cur = dp[mask][v]
        pm = mask ^ (1 << v)
        row = dp[pm]
        found = None
        for u, w, _ in g.adj[v]:
            if (pm >> u) & 1 and row is not None and row[u] != NEG and abs(row[u] + w - cur) < 1e-7:
                found = u
                break
        mask, v = pm, found
        path.append(v)
    return path[::-1]


def tables(g, terminals):
    """Returns dict: free=(L,path), frm={a:(L,path)}, pair={(a,b):(L,path)}, cycle=(L,cyclepath)|None.
    Paths are local vertex lists; frm/pair paths start at a."""
    n = g.n
    full = (1 << n) - 1
    res = {"frm": {}, "pair": {}}
    dp = _run(g, range(n), full)
    best = (0.0, [0])
    for mask in range(1, 1 << n):
        row = dp[mask]
        if row is None:
            continue
        for v in range(n):
            if row[v] > best[0] + 1e-12:
                best = (row[v], (mask, v))
    if isinstance(best[1], tuple):
        best = (best[0], _backtrack(g, dp, *best[1]))
    res["free"] = best
    for a in terminals:
        dpa = _run(g, [a], full)
        bf = (0.0, (1 << a, a))
        bp = {}
        for mask in range(1, 1 << n):
            row = dpa[mask]
            if row is None:
                continue
            for v in range(n):
                val = row[v]
                if val == NEG:
                    continue
                if val > bf[0] + 1e-12:
                    bf = (val, (mask, v))
                if v in terminals and v != a and (v not in bp or val > bp[v][0] + 1e-12):
                    bp[v] = (val, (mask, v))
        res["frm"][a] = (bf[0], _backtrack(g, dpa, *bf[1]))
        for b, (val, st) in bp.items():
            res["pair"][(a, b)] = (val, _backtrack(g, dpa, *st))
    # cycles: smallest vertex s is the start
    bc = None
    for s in range(n):
        allowed = full & ~((1 << s) - 1)
        dps = _run(g, [s], allowed)
        for mask in range(1, 1 << n):
            row = dps[mask]
            if row is None or bin(mask).count("1") < 3:
                continue
            for v, w, _ in g.adj[s]:
                if (mask >> v) & 1 and row[v] != NEG:
                    val = row[v] + w
                    if bc is None or val > bc[0] + 1e-12:
                        bc = (val, (mask, v))
        if bc is not None and isinstance(bc[1], tuple):
            mask, v = bc[1]
            p = _backtrack(g, dps, mask, v)
            bc = (bc[0], p + [p[0]])
    res["cycle"] = bc
    return res
