"""Exhaustive reference solver (no pruning). Only for small graphs."""


def brute(g):
    best = [-1.0, []]

    def dfs(start, u, visited, length, path):
        if length > best[0] + 1e-9:
            best[0] = length
            best[1] = path[:]
        for v, w, _ in g.adj[u]:
            if v == start and len(path) >= 3:
                if length + w > best[0] + 1e-9:
                    best[0] = length + w
                    best[1] = path + [start]
            if not (visited >> v) & 1:
                path.append(v)
                dfs(start, v, visited | (1 << v), length + w, path)
                path.pop()

    for s in range(g.n):
        dfs(s, s, 1 << s, 0.0, [s])
    return best[0], best[1]
