"""Orchestration: heuristics -> (Lagrangian) -> branch and bound, optionally via block decomposition."""
import multiprocessing as mp
import time
from dataclasses import dataclass, field

from . import bitdp, heuristics as H, lagrange as LG
from .bnb import BnBOpts, Search, Timeout
from .graph import biconnected, components, compress, identity_cgraph, verify

EPS = 1e-9
ALL_INIT = ("greedy", "mst", "fcycle", "beam", "lagr", "ils")


@dataclass
class Config:
    mode: str = "decomp"          # 'decomp' | 'whole'
    compress: bool = False        # benchmarks: chain compression loosens bounds (off by default)
    bnb: BnBOpts = field(default_factory=BnBOpts)
    init: tuple = ALL_INIT
    lagr_iters: int = 100
    lagr_init: str = "zero"       # 'zero' | 'third'
    edge_del: bool = True
    lagr_cycle: bool = True       # 1-forest cycle bound for the root proof and edge deletion
    lagr_auto: bool = True        # disable the Lagrangian bound when the root bound is far from the
    lagr_auto_ratio: float = 1.3  # incumbent (ub / incumbent above this ratio)
    workers: int = 1
    t_dp: int = 12
    k_max: int = 4                # legacy switch (used only when q_max is None)
    q_max: int = 40               # decompose a big block if its table count k(k-1)/2 + k + 2 <= q_max
    beam_K: int = 32
    ils_iters: int = 30
    seed: int = 0


class Stats(dict):
    def add(self, k, v):
        self[k] = self.get(k, 0) + v


def _better(a, b):
    return a if a[0] >= b[0] else b


# ---------------------------------------------------------------- initial solutions
def _cg_path_to_orig(cg, seq, es):
    path = [cg.orig[seq[0]]]
    for v, e in zip(seq, es):
        path += cg.expand_edge(e, v)[1:]
    return path


def initial_solution(g, cg, cfg, st):
    t0 = time.perf_counter()
    best = (0.0, [0]) if g.n else (-1.0, [])
    init = cfg.init
    if "greedy" in init:
        best = _better(best, H.greedy(g))
    if "mst" in init:
        best = _better(best, H.mst_diameter(g))
    if "fcycle" in init and len(g.edges) >= g.n:
        best = _better(best, H.fundamental_cycles(g))
    if "beam" in init:
        best = _better(best, H.beam(g, K=cfg.beam_K))
    lag = None
    o = cfg.bnb
    if o.b_lagr or cfg.edge_del or "lagr" in init or o.order == "lagr":
        t1 = time.perf_counter()
        lam, ub, chosen, hist = LG.subgradient(cg, best[0], cfg.lagr_iters, cfg.lagr_init)
        st.add("t_lagr", time.perf_counter() - t1)
        st["lagr_ub"] = ub
        st["lagr_ub0"] = hist[0]
        st["lagr_iters_used"] = len(hist)
        lag = (lam, ub, chosen)
        if "lagr" in init:
            for L, seq, es in LG.forest_paths(cg, chosen)[:3]:
                p = _cg_path_to_orig(cg, seq, es)
                best = _better(best, H.extend_greedy(g, p))
    if "ils" in init and best[1]:
        p = best[1][:-1] if len(best[1]) > 2 and best[1][0] == best[1][-1] else best[1]
        best = _better(best, H.ils(g, p, iters=cfg.ils_iters, seed=cfg.seed))
    st.add("t_init", time.perf_counter() - t0)
    st["init_value"] = max(st.get("init_value", 0.0), best[0])
    return best, lag


# ---------------------------------------------------------------- parallel B&B
_G = {}


def _worker(chunk):
    s = Search(_G["cg"], _G["opts"], _G["best"], _G["path"], _G["lam"], shared=_G["shared"])
    try:
        s.run_path(descriptors=chunk)
    except Timeout:
        return s.best, s.best_path, s.nodes, True
    return s.best, s.best_path, s.nodes, False


def _worker_cycle(starts):
    s = Search(_G["cg"], _G["opts"], _G["best"], _G["path"], _G["lam"], shared=_G["shared"])
    try:
        s.run_cycle(starts=starts)
    except Timeout:
        return s.best, s.best_path, s.nodes, True
    return s.best, s.best_path, s.nodes, False


def _parallel_cycle(cg, opts, best, lam, workers, st):
    starts = list(range(cg.n))
    k = max(1, len(starts) // (workers * 4))
    chunks = [starts[i:i + k] for i in range(0, len(starts), k)]
    shared = mp.Value("d", best[0])
    _G.update(cg=cg, opts=opts, best=best[0], path=best[1], lam=lam, shared=shared)
    timed_out = False
    with mp.get_context("fork").Pool(workers) as pool:
        for b, p, nodes, to in pool.imap_unordered(_worker_cycle, chunks):
            st.add("nodes", nodes)
            timed_out |= to
            if b > best[0] + EPS:
                best = (b, p)
    if timed_out:
        raise Timeout()
    return best


def _parallel_path(cg, opts, best, lam, workers, st):
    probe = Search(cg, opts)
    ds = probe.start_descriptors()
    k = max(1, len(ds) // (workers * 4))
    chunks = [ds[i:i + k] for i in range(0, len(ds), k)]
    shared = mp.Value("d", best[0])
    _G.update(cg=cg, opts=opts, best=best[0], path=best[1], lam=lam, shared=shared)
    ctx = mp.get_context("fork")
    timed_out = False
    with ctx.Pool(workers) as pool:
        for b, p, nodes, to in pool.imap_unordered(_worker, chunks):
            st.add("nodes", nodes)
            timed_out |= to
            if b > best[0] + EPS:
                best = (b, p)
    if timed_out:
        raise Timeout()
    return best


# ---------------------------------------------------------------- whole-component B&B
def solve_whole(g, cfg, st, deadline):
    if g.n == 1:
        return (0.0, [0])
    cg = compress(g) if cfg.compress else identity_cgraph(g)
    st.add("cg_vertices", cg.n)
    best, lag = initial_solution(g, cg, cfg, st)
    lam = lag[0] if lag else None
    # a cycle is a path plus one edge, so cycles are bounded by ub + wmax (or the 1-forest bound)
    cyc_ub = (LG.cycle_bound(cg, lag[0]) if cfg.lagr_cycle else lag[1] + cg.wmax) if lag else None
    if lag:
        st["cycle_ub"] = cyc_ub
    if lag and lag[1] <= best[0] + EPS and cyc_ub <= best[0] + EPS:
        st["proved_by_lagr"] = True
        return best
    if cfg.edge_del and lag:
        extra = LG.max_mod(cg, lag[0]) if cfg.lagr_cycle else None
        dead = LG.eliminate_edges(cg, lag[0], lag[2], lag[1], best[0], extra)
        st.add("edges_deleted", len(dead))
        st.add("edges_total", len(cg.edges))
        if dead:
            cg = LG.remove_edges(cg, dead)
    opts = _auto_lagr(cfg, _with_deadline(cfg.bnb, deadline), lag, best, st)
    t0 = time.perf_counter()
    if cfg.workers > 1:
        best = _parallel_path(cg, opts, best, lam, cfg.workers, st)
        best = _parallel_cycle(cg, opts, best, lam, cfg.workers, st)
        st.add("t_bnb", time.perf_counter() - t0)
        return best
    s = Search(cg, opts, best[0], best[1], lam)
    try:
        s.run_path()
    finally:
        st.add("nodes", s.nodes)
    best = (s.best, s.best_path)
    s = Search(cg, opts, best[0], best[1], lam)
    try:
        s.run_cycle()
    finally:
        st.add("nodes", s.nodes)
    st.add("t_bnb", time.perf_counter() - t0)
    return (s.best, s.best_path)


def _auto_lagr(cfg, opts, lag, best, st):
    """Turn the Lagrangian bound off when the root bound is far above the incumbent."""
    if not (cfg.lagr_auto and lag and opts.b_lagr):
        return opts
    from dataclasses import replace
    ratio = lag[1] / max(best[0], 1e-9)
    st["lagr_ratio"] = max(st.get("lagr_ratio", 0.0), ratio)
    if ratio > cfg.lagr_auto_ratio:
        st["lagr_disabled"] = True
        return replace(opts, b_lagr=False, cycle_1forest=False)
    return opts


def _with_deadline(o, deadline):
    from dataclasses import replace
    return replace(o, time_limit=max(0.0, deadline - time.perf_counter()))


# ---------------------------------------------------------------- block tables
def bnb_tables(lg, terms, cfg, st, deadline):
    cg = compress(lg, protected=terms) if cfg.compress else identity_cgraph(lg)
    st.add("cg_vertices", cg.n)
    pos = {v: i for i, v in enumerate(cg.orig)}
    res = {"frm": {}, "pair": {}}
    best, lag = initial_solution(lg, cg, cfg, st)
    lam = lag[0] if lag else None
    opts = _auto_lagr(cfg, _with_deadline(cfg.bnb, deadline), lag, best, st)

    def run(fn, b0=(-1.0, [])):
        s = Search(cg, opts, b0[0], b0[1], lam)
        try:
            out = fn(s)
        finally:
            st.add("nodes", s.nodes)
        return out

    t0 = time.perf_counter()
    if not (lag and lag[1] <= best[0] + EPS):  # path query only: path bound suffices
        best = run(lambda s: s.run_path(), best)
    res["free"] = best
    cyc = run(lambda s: s.run_cycle(), best)
    res["cycle"] = cyc if (cyc[1] and cyc[1][0] == cyc[1][-1] and len(cyc[1]) > 3) else None
    for a in terms:
        res["frm"][a] = run(lambda s: s.run_path(start=pos[a]))
    for i, a in enumerate(terms):
        for b in terms[i + 1:]:
            L, p = run(lambda s: s.run_path(start=pos[a], target=pos[b]))
            res["pair"][(a, b)] = (L, p)
            res["pair"][(b, a)] = (L, p[::-1])
    st.add("t_bnb", time.perf_counter() - t0)
    return res


# ---------------------------------------------------------------- decomposition
def solve_decomp(g, cfg, st, deadline):
    blocks, art = biconnected(g)
    st.add("blocks", len(blocks))
    st["max_block"] = max(st.get("max_block", 0), max(len(b[0]) for b in blocks))
    if len(blocks) == 1:
        return solve_whole(g, cfg, st, deadline)
    tabs = []
    for vs, _ in blocks:
        lg, verts = g.induced(vs)
        terms_g = sorted(set(vs) & art)
        loc = {v: i for i, v in enumerate(verts)}
        terms = [loc[v] for v in terms_g]
        if lg.n <= cfg.t_dp:
            t0 = time.perf_counter()
            tab = bitdp.tables(lg, terms)
            st.add("t_bitdp", time.perf_counter() - t0)
            st.add("bitdp_blocks", 1)
        elif (len(terms) <= cfg.k_max if cfg.q_max is None
              else len(terms) * (len(terms) - 1) // 2 + len(terms) + 2 <= cfg.q_max):
            tab = bnb_tables(lg, terms, cfg, st, deadline)
            st.add("bnb_blocks", 1)
        else:
            st["fallback_whole"] = True
            return solve_whole(g, cfg, st, deadline)
        tabs.append(_globalize(tab, verts))
    return _combine(blocks, art, tabs)


def _globalize(tab, verts):
    m = lambda p: [verts[x] for x in p]
    out = {"free": (tab["free"][0], m(tab["free"][1])),
           "frm": {verts[a]: (L, m(p)) for a, (L, p) in tab["frm"].items()},
           "pair": {(verts[a], verts[b]): (L, m(p)) for (a, b), (L, p) in tab["pair"].items()},
           "cycle": None if tab["cycle"] is None else (tab["cycle"][0], m(tab["cycle"][1]))}
    return out


def _combine(blocks, art, tabs):
    nb = len(blocks)
    vsets = [b[0] for b in blocks]
    blocks_of = {a: [i for i in range(nb) if a in vsets[i]] for a in art}
    root = max(range(nb), key=lambda i: len(vsets[i]))
    parent_art = {root: None}
    child_arts = {}
    child_blocks = {}
    order = []
    stack = [root]
    while stack:
        B = stack.pop()
        order.append(B)
        cs = [a for a in vsets[B] if a in art and a != parent_art[B]]
        child_arts[B] = cs
        for a in cs:
            child_blocks[a] = [C for C in blocks_of[a] if C != B]
            for C in child_blocks[a]:
                parent_art[C] = a
                stack.append(C)
    down = {}      # a -> (val, ('via', B))
    vias = {}      # (B, a) -> (val, choice)
    best = (-1.0, None)

    def cand(val, desc):
        nonlocal best
        if val > best[0] + EPS:
            best = (val, desc)

    for B in reversed(order):
        tab = tabs[B]
        for a in child_arts[B]:
            pass  # downs of child arts computed when their child blocks were processed
        p = parent_art[B]
        cs = child_arts[B]
        for a in cs:
            vl = sorted(((vias[(C, a)][0], C) for C in child_blocks[a]), reverse=True)
            down[a] = (vl[0][0], vl[0][1]) if vl else (0.0, None)
            if len(vl) >= 2:
                cand(vl[0][0] + vl[1][0], ("art2", a, vl[0][1], vl[1][1]))
            elif vl:
                cand(vl[0][0], ("art1", a, vl[0][1]))
        # candidates topped at B
        cand(tab["free"][0], ("free", B))
        if tab["cycle"] is not None:
            cand(tab["cycle"][0], ("cycle", B))
        for a in cs:
            cand(tab["frm"][a][0] + down[a][0], ("tail", B, a))
            for b in cs:
                if a < b:
                    cand(down[a][0] + tab["pair"][(a, b)][0] + down[b][0], ("pair", B, a, b))
        if p is not None:
            bv = (tab["frm"][p][0], ("frm",))
            for c in cs:
                v = tab["pair"][(p, c)][0] + down[c][0]
                if v > bv[0] + EPS:
                    bv = (v, ("pair", c))
            vias[(B, p)] = bv

    def exp_down(a):
        val, B = down[a]
        if B is None:
            return [a]
        return exp_via(B, a)

    def exp_via(B, p):
        _, ch = vias[(B, p)]
        if ch[0] == "frm":
            return list(tabs[B]["frm"][p][1])
        c = ch[1]
        return list(tabs[B]["pair"][(p, c)][1]) + exp_down(c)[1:]

    val, d = best
    k = d[0]
    if k == "free":
        path = tabs[d[1]]["free"][1]
    elif k == "cycle":
        path = tabs[d[1]]["cycle"][1]
    elif k == "art1":
        path = exp_via(d[2], d[1])
    elif k == "art2":
        path = exp_via(d[2], d[1])[::-1] + exp_via(d[3], d[1])[1:]
    elif k == "tail":
        _, B, a = d
        path = tabs[B]["frm"][a][1][::-1] + exp_down(a)[1:]
    else:
        _, B, a, b = d
        path = exp_down(a)[::-1] + tabs[B]["pair"][(a, b)][1][1:] + exp_down(b)[1:]
    return (val, path)


# ---------------------------------------------------------------- top level
def solve(g, cfg=None, time_limit=float("inf")):
    cfg = cfg or Config()
    st = Stats()
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    best = (-1.0, [])
    timed_out = False
    for comp in components(g):
        sub, verts = g.induced(comp)
        try:
            if cfg.mode == "whole":
                L, p = solve_whole(sub, cfg, st, deadline)
            else:
                L, p = solve_decomp(sub, cfg, st, deadline)
        except Timeout:
            timed_out = True
            break
        if L > best[0] + EPS:
            best = (L, [verts[x] for x in p])
    st["time"] = time.perf_counter() - t0
    st["timeout"] = timed_out
    if best[1] and not timed_out:
        v = verify(g, best[1])
        assert abs(v - best[0]) < 1e-6, (v, best)
    return best, st
