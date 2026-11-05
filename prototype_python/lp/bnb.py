"""Branch-and-bound DFS on a (possibly compressed) CGraph with toggleable components."""
import sys
import time
from dataclasses import dataclass

sys.setrecursionlimit(100000)
EPS = 1e-9
INF = float("inf")


@dataclass
class BnBOpts:
    reach: bool = True        # restrict bounds/moves to vertices reachable through unvisited ones
    b_top1: bool = False      # bound: sum of max incident edge (dominated by top2; off by default)
    b_top2: bool = True       # bound: (top1+top2)/2
    b_bcc: bool = True        # bound: decompose remaining reachable map at articulation points
    b_lagr: bool = True       # bound: Lagrangian forest with fixed multipliers
    lagr_depth: int = 10**9   # use Lagrangian bound at all depths (benchmarks: shallow-only was wrong)
    dominance: bool = True    # (vertex, reachable set) memo
    memo_cap: int = 300000
    order: str = "lagr"       # 'none' | 'weight' | 'bound' | 'lagr' (falls back to weight w/o lambda)
    partial_bound: str = "global"  # 'global': +max partial anywhere | 'local': only chains touching R
    cycle_1forest: bool = True     # Lagrangian 1-forest bound at the root of each cycle search
    time_limit: float = INF


class Timeout(Exception):
    pass


def bits(m):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


class Search:
    def __init__(self, cg, opts, best=-1.0, best_path=None, lam=None, shared=None):
        self.cg = cg
        self.o = opts
        self.best = best
        self.best_path = best_path or []
        self.nodes = 0
        self.memo = {}
        self.lam = lam
        self.shared = shared  # multiprocessing Value for parallel incumbent
        self.t0 = time.perf_counter()
        n = cg.n
        self.t12 = [cg.top1[u] + cg.top2[u] for u in range(n)]
        self.full = (1 << n) - 1
        self.maxpart = [max((val for _, _, val in cg.partial_from[u]), default=0.0) for u in range(n)]
        if lam is not None:
            self.mod_sorted = sorted(
                ((tot - lam[u] - lam[v], i) for i, (u, v, tot, *_r) in enumerate(cg.edges)
                 if tot - lam[u] - lam[v] > 0), reverse=True)

    # ---------- incumbent ----------
    def _cur_best(self):
        if self.shared is not None:
            v = self.shared.value
            if v > self.best:
                return v
        return self.best

    def _update(self, length, tail=None):
        if length > self._cur_best() + EPS:
            self.best = length
            self.best_path = self._expand(tail)
            if self.shared is not None:
                with self.shared.get_lock():
                    if length > self.shared.value:
                        self.shared.value = length

    def _expand(self, tail):
        cg = self.cg
        if self.pstart is not None:
            e, far = self.pstart
            path = cg.expand_edge(e, far)[1:]
        else:
            path = [cg.orig[self.start_v]]
        for e, frm in self.moves:
            path += cg.expand_edge(e, frm)[1:]
        if tail is not None:
            kind, e, frm = tail
            if kind == "partial":
                path += cg.partial_vertices(e, frm)
            else:  # closing edge of a cycle
                path += cg.expand_edge(e, frm)[1:]
        return path

    # ---------- helpers ----------
    def _reach(self, c, visited):
        adjm = self.cg.adjmask
        if not self.o.reach:
            return self.allowed & ~visited
        R = 0
        frontier = adjm[c] & ~visited
        while frontier:
            R |= frontier
            nf = 0
            for v in bits(frontier):
                nf |= adjm[v]
            frontier = nf & ~visited & ~R
        return R

    def _reach_bcc(self, c, visited):
        """One Tarjan DFS from c over unvisited vertices: returns (reachable set, BCC bound)."""
        cg = self.cg
        top1 = cg.top1
        disc = {c: 0}
        low = {c: 0}
        t = 1
        node_ub = {}
        vstack = []
        R = 0
        stack = [(c, -1, iter(cg.adj[c]))]
        while stack:
            u, pe, it = stack[-1]
            adv = False
            for v, e in it:
                if e == pe or ((visited >> v) & 1 and v != c):  # back edges to c count
                    continue
                if v not in disc:
                    disc[v] = low[v] = t
                    t += 1
                    R |= 1 << v
                    vstack.append(v)
                    stack.append((v, e, iter(cg.adj[v])))
                    adv = True
                    break
                elif disc[v] < low[u]:
                    low[u] = disc[v]
            if adv:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                if low[u] < low[p]:
                    low[p] = low[u]
                if low[u] >= disc[p]:
                    s1 = 0.0
                    bc = 0.0
                    while True:
                        x = vstack.pop()
                        s1 += top1[x]
                        nb = node_ub.get(x, 0.0)
                        if nb > bc:
                            bc = nb
                        if x == u:
                            break
                    val = s1 + bc
                    if val > node_ub.get(p, 0.0):
                        node_ub[p] = val
        return R, node_ub.get(c, 0.0)

    def _bound(self, c, R, length, depth, bcc_val=None):
        cg = self.cg
        extra_p = 0.0
        if self.partial_ok:
            if self.o.partial_bound == "local":
                mp_ = self.maxpart
                extra_p = mp_[c]
                for u in bits(R):
                    if mp_[u] > extra_p:
                        extra_p = mp_[u]
            else:
                extra_p = cg.max_partial
        extra_c = cg.top1[self.s] if self.cycle else 0.0
        ub = INF
        if self.o.b_top1:
            s1 = 0.0
            top1 = cg.top1
            for u in bits(R):
                s1 += top1[u]
            ub = min(ub, length + s1 + extra_p + extra_c)
        if self.o.b_top2:
            s2 = 0.0
            t12 = self.t12
            for u in bits(R):
                s2 += t12[u]
            ub = min(ub, length + (cg.top1[c] + s2 + extra_p + extra_c) / 2.0)
        if self.o.b_bcc:
            bv = bcc_val if bcc_val is not None else self._bcc_bound(c, R)
            ub = min(ub, length + bv + extra_p + extra_c)
        # at the root of a cycle search (c == s) the remainder is a cycle, not a forest
        if self.lam is not None and self.cycle and c == self.s:
            if self.o.cycle_1forest:
                ub = min(ub, self._lagr_1forest(R | (1 << c)))
        elif self.o.b_lagr and self.lam is not None and depth <= self.o.lagr_depth:
            ub = min(ub, length + self._lagr_bound(c, R) + extra_p)
        return ub

    def _bcc_bound(self, c, R):
        cg = self.cg
        top1 = cg.top1
        H = R | (1 << c)
        disc = {c: 0}
        low = {c: 0}
        t = 1
        node_ub = {}
        vstack = []
        stack = [(c, -1, iter(cg.adj[c]))]
        while stack:
            u, pe, it = stack[-1]
            adv = False
            for v, e in it:
                if e == pe or not (H >> v) & 1:
                    continue
                if v not in disc:
                    disc[v] = low[v] = t
                    t += 1
                    vstack.append(v)
                    stack.append((v, e, iter(cg.adj[v])))
                    adv = True
                    break
                elif disc[v] < low[u]:
                    low[u] = disc[v]
            if adv:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                if low[u] < low[p]:
                    low[p] = low[u]
                if low[u] >= disc[p]:
                    s1 = 0.0
                    bc = 0.0
                    while True:
                        x = vstack.pop()
                        s1 += top1[x]
                        nb = node_ub.get(x, 0.0)
                        if nb > bc:
                            bc = nb
                        if x == u:
                            break
                    val = s1 + bc
                    if val > node_ub.get(p, 0.0):
                        node_ub[p] = val
        return node_ub.get(c, 0.0)

    def _lagr_bound(self, c, R):
        cg = self.cg
        lam = self.lam
        H = R | (1 << c)
        if self.cycle:
            H |= 1 << self.s
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                px = parent[x]
                parent[x] = parent.get(px, px)
                x = px
            return x

        tot = 0.0
        edges = cg.edges
        for m, i in self.mod_sorted:
            u, v = edges[i][0], edges[i][1]
            if (H >> u) & 1 and (H >> v) & 1:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
                    tot += m
        s = lam[c]
        for u in bits(R):
            s += 2 * lam[u]
        if self.cycle:
            s += lam[self.s]
        return tot + s

    def _lagr_1forest(self, H):
        """Any cycle inside H is a forest plus one edge with degrees <= 2."""
        cg = self.cg
        lam = self.lam
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                px = parent[x]
                parent[x] = parent.get(px, px)
                x = px
            return x

        tot = 0.0
        extra = 0.0
        edges = cg.edges
        for m, i in self.mod_sorted:
            u, v = edges[i][0], edges[i][1]
            if (H >> u) & 1 and (H >> v) & 1:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
                    tot += m
                elif m > extra:
                    extra = m
        s = 0.0
        for u in bits(H):
            s += 2 * lam[u]
        return tot + extra + s

    def _check_time(self):
        if self.nodes & 1023 == 0 and time.perf_counter() - self.t0 > self.o.time_limit:
            raise Timeout()

    # ---------- DFS ----------
    def _dfs(self, c, visited, length, depth, arr_e, ocount):
        self.nodes += 1
        self._check_time()
        cg = self.cg
        # terminal candidates
        if self.cycle:
            if depth >= 1:
                for v, e in cg.adj[c]:
                    if v == self.s and e != self.first_e:
                        if ocount + len(cg.edges[e][5]) >= 3:
                            self._update(length + cg.edges[e][2], ("close", e, c))
        else:
            if self.target is None or c == self.target:
                self._update(length)
            if self.partial_ok:
                for e, other, val in cg.partial_from[c]:
                    if e != arr_e and e != self.pstart_e and (visited >> other) & 1:
                        self._update(length + val, ("partial", e, c))
            if self.target is not None and c == self.target:
                return
        bcc_val = None
        if self.o.b_bcc and self.o.reach:
            R, bcc_val = self._reach_bcc(c, visited)
        else:
            R = self._reach(c, visited)
        if not R:
            return
        if self.target is not None and not (R >> self.target) & 1:
            return
        ub = self._bound(c, R, length, depth, bcc_val)
        if ub <= self._cur_best() + EPS:
            return
        if self.o.dominance:
            # c's own arrival edge only matters for c's terminal moves (already handled above)
            key = (c, R, self.pstart_e, self.s)
            old = self.memo.get(key)
            if old is not None and old >= length - 1e-12:
                return
            if old is not None or len(self.memo) < self.o.memo_cap:
                self.memo[key] = length
        moves = [(v, e) for v, e in cg.adj[c] if (R >> v) & 1 and e != self.pstart_e]
        order = self.o.order
        if order == "weight":
            moves.sort(key=lambda x: -cg.edges[x[1]][2])
        elif order == "bound":
            s2 = 0.0
            for u in bits(R):
                s2 += self.t12[u]
            moves.sort(key=lambda x: -(cg.edges[x[1]][2] + (cg.top1[x[0]] + s2 - self.t12[x[0]]) / 2))
        elif order == "lagr":
            lam = self.lam
            if lam is None:
                moves.sort(key=lambda x: -cg.edges[x[1]][2])
            else:
                moves.sort(key=lambda x: -(cg.edges[x[1]][2] - lam[c] - lam[x[0]]))
        for v, e in moves:
            if (visited >> v) & 1:
                continue
            if depth == 0:
                self.first_e = e
            self.moves.append((e, c))
            self._dfs(v, visited | (1 << v), length + cg.edges[e][2], depth + 1, e,
                      ocount + 1 + len(cg.edges[e][5]))
            self.moves.pop()

    # ---------- entry points ----------
    def _setup(self, cycle, target, partial_ok, keyarr):
        self.cycle = cycle
        self.target = target
        self.partial_ok = partial_ok
        self.keyarr = keyarr
        self.moves = []
        self.pstart = None
        self.pstart_e = -1
        self.s = -1
        self.first_e = -1
        self.allowed = self.full
        self.memo = {}

    def start_descriptors(self, start=None, target=None):
        cg = self.cg
        if start is not None:
            return [("v", start)]
        ds = [("v", s) for s in range(cg.n)]
        if target is None:
            for i, (u, v, tot, su, sv, inter) in enumerate(cg.edges):
                if inter:
                    ds.append(("p", i, u, v, tot - su))  # starts next to u, runs to v
                    ds.append(("p", i, v, u, tot - sv))
        return ds

    def run_path(self, start=None, target=None, descriptors=None):
        cg = self.cg
        self._setup(False, target, target is None, cg.max_partial > 0)
        ds = descriptors if descriptors is not None else self.start_descriptors(start, target)
        for d in ds:
            if d[0] == "v":
                s = d[1]
                self.start_v = s
                self.pstart = None
                self.pstart_e = -1
                self._dfs(s, 1 << s, 0.0, 0, -1, 1)
            else:
                _, e, far, w0, init = d
                self.start_v = w0
                self.pstart = (e, far)
                self.pstart_e = e
                self._dfs(w0, 1 << w0, init, 0, e, 1 + len(cg.edges[e][5]))
        self.pstart = None
        self.pstart_e = -1
        return self.best, self.best_path

    def run_cycle(self, starts=None):
        cg = self.cg
        self._setup(True, None, False, True)
        for s in (starts if starts is not None else range(cg.n)):
            self.s = s
            self.start_v = s
            excluded = (1 << s) - 1
            self.allowed = self.full & ~excluded
            self._dfs(s, (1 << s) | excluded, 0.0, 0, -1, 1)
        return self.best, self.best_path
