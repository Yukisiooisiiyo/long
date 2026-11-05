"""Ablation benchmarks. Usage: python bench/run.py EXP OUTFILE [shard nshards]"""
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gen import cactus, dense, grid, railway  # noqa: E402
from lp import lagrange as LG  # noqa: E402
from lp.bnb import BnBOpts  # noqa: E402
from lp.graph import biconnected, compress  # noqa: E402
from lp.solver import ALL_INIT, Config, solve  # noqa: E402

LIMIT = float(os.environ.get("LIMIT", "40"))

INSTANCES = {
    "rail80": lambda: railway(80, 1),
    "rail120": lambda: railway(120, 2),
    "grid5x5": lambda: grid(5, 5, 1),
    "grid5x6": lambda: grid(5, 6, 3),
    "dense16": lambda: dense(16, 2),
    "dense18": lambda: dense(18, 1),
    "cactus300": lambda: cactus(300, 1),
    "rail100": lambda: railway(100, 1),
    "grid6x6": lambda: grid(6, 6, 1),
    "dense22": lambda: dense(22, 1),
    "rail160": lambda: railway(160, 3),
}
CORE = ["rail80", "rail120", "grid5x5", "grid5x6", "dense16", "dense18"]

FULL_O = BnBOpts(reach=True, b_top1=True, b_top2=True, b_bcc=True, b_lagr=True, dominance=True,
                 order="bound", lagr_depth=3, cycle_1forest=False)
OLD = dict(lagr_cycle=False, lagr_auto=False, q_max=None, k_max=4)
FULL = Config(mode="whole", compress=True, bnb=FULL_O, init=ALL_INIT, edge_del=True, **OLD)
BASE_O = BnBOpts(reach=True, b_top1=True, b_top2=False, b_bcc=False, b_lagr=False, dominance=False,
                 order="weight", lagr_depth=3, cycle_1forest=False)
BASE = Config(mode="whole", compress=False, bnb=BASE_O, init=("greedy",), edge_del=False, **OLD)


BEST = Config(mode="whole", compress=False, bnb=replace(FULL_O, lagr_depth=10**9, cycle_1forest=True),
              init=ALL_INIT, edge_del=True, lagr_cycle=True, lagr_auto=False, q_max=None, k_max=4)


def o(cfg, **kw):
    return replace(cfg, bnb=replace(cfg.bnb, **kw))


def exp_minus():
    cs = {"FULL": FULL}
    for f in ["reach", "b_top1", "b_top2", "b_bcc", "b_lagr", "dominance"]:
        cs["-" + f] = o(FULL, **{f: False})
    cs["order=weight"] = o(FULL, order="weight")
    cs["order=none"] = o(FULL, order="none")
    cs["order=lagr"] = o(FULL, order="lagr")
    cs["-compress"] = replace(FULL, compress=False)
    cs["-edge_del"] = replace(FULL, edge_del=False)
    cs["init=none"] = replace(FULL, init=())
    cs["init=greedy"] = replace(FULL, init=("greedy",))
    for x in ALL_INIT:
        cs["-init_" + x] = replace(FULL, init=tuple(i for i in ALL_INIT if i != x))
    return cs, CORE


def exp_plus():
    cs = {"BASE": BASE}
    for f in ["b_top2", "b_bcc", "b_lagr", "dominance"]:
        cs["+" + f] = o(BASE, **{f: True})
    cs["+order=bound"] = o(BASE, order="bound")
    cs["+compress"] = replace(BASE, compress=True)
    cs["+lagr+edge_del"] = replace(o(BASE, b_lagr=True), edge_del=True)
    cs["+init_all"] = replace(BASE, init=ALL_INIT)
    cs["-reach"] = o(BASE, reach=False)
    return cs, CORE


def exp_decomp():
    cs = {"FULL-whole": FULL, "FULL-decomp": replace(FULL, mode="decomp"),
          "decomp,t_dp=0": replace(FULL, mode="decomp", t_dp=0),
          "decomp,k_max=99": replace(FULL, mode="decomp", k_max=99),
          "decomp,t_dp=0,k_max=99": replace(FULL, mode="decomp", t_dp=0, k_max=99)}
    return cs, ["cactus300", "rail100", "rail80", "rail120", "grid5x5"]


def exp_best():
    cs = {"BEST": BEST}
    for f in ["reach", "b_top1", "b_top2", "b_bcc", "b_lagr", "dominance", "cycle_1forest"]:
        cs["-" + f] = o(BEST, **{f: False})
    cs["order=weight"] = o(BEST, order="weight")
    cs["order=lagr"] = o(BEST, order="lagr")
    cs["+compress"] = replace(BEST, compress=True)
    cs["-edge_del"] = replace(BEST, edge_del=False)
    cs["init=none"] = replace(BEST, init=())
    cs["init=greedy"] = replace(BEST, init=("greedy",))
    cs["decomp,k_max=99"] = replace(BEST, mode="decomp", k_max=99)
    cs["decomp,k_max=4"] = replace(BEST, mode="decomp")
    return cs, CORE + ["grid6x6", "dense22", "rail160"]


def exp_new():
    new = Config()  # defaults after applying the recommendations (decomposition mode)
    cs = {"FULL": FULL, "BEST": BEST, "NEW": new, "NEW-whole": replace(new, mode="whole"),
          "NEW-no-auto": replace(new, lagr_auto=False), "NEW,workers=2": replace(new, workers=2)}
    return cs, ["rail80", "rail100", "rail120", "rail160", "cactus300", "grid5x5", "grid5x6",
                "grid6x6", "dense16", "dense18", "dense22"]


def exp_parallel():
    cs = {"workers=1": BEST, "workers=2": replace(BEST, workers=2),
          "FULL,workers=1": FULL, "FULL,workers=2": replace(FULL, workers=2)}
    return cs, ["dense18", "grid5x6", "rail120", "rail160"]


def run(exp, out, shard=0, nshards=1):
    cs, insts = {"minus": exp_minus, "plus": exp_plus, "decomp": exp_decomp,
                 "parallel": exp_parallel, "best": exp_best,
                 "new": exp_new}[exp]()
    jobs = [(i, c) for i in insts for c in cs]
    with open(out, "a") as f:
        for k, (iname, cname) in enumerate(jobs):
            if k % nshards != shard:
                continue
            g = INSTANCES[iname]()
            t0 = time.perf_counter()
            (L, p), st = solve(g, cs[cname], time_limit=LIMIT)
            rec = {"exp": exp, "inst": iname, "cfg": cname, "len": L,
                   "wall": time.perf_counter() - t0, **{k2: v for k2, v in st.items()}}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(iname, cname, round(L, 2), st.get("nodes"), round(rec["wall"], 2),
                  "TO" if st["timeout"] else "", flush=True)


def lagr_study(out):
    """Root Lagrangian: bound quality and multiplier initialisation; edge deletion rate."""
    with open(out, "a") as f:
        for iname in CORE + ["cactus300", "rail100"]:
            g = INSTANCES[iname]()
            cg = compress(g)
            (opt, _), _ = solve(g, FULL, time_limit=120)
            for init in ["zero", "third"]:
                t0 = time.perf_counter()
                lam, ub, chosen, hist = LG.subgradient(cg, opt, 300, init)
                t = time.perf_counter() - t0
                dead = LG.eliminate_edges(cg, lam, chosen, ub, opt)
                rec = {"exp": "lagr", "inst": iname, "init": init, "opt": opt, "ub0": hist[0],
                       "ub": ub, "iters": len(hist), "time": t, "wmax": cg.wmax,
                       "edges": len(cg.edges), "dead_at_opt": len(dead),
                       "hist": [hist[i] for i in (1, 5, 10, 25, 50, 100, 200) if i < len(hist)]}
                f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)


def structure(out):
    with open(out, "a") as f:
        for iname, mk in INSTANCES.items():
            g = mk()
            bl, art = biconnected(g)
            big = max(bl, key=lambda b: len(b[0]))
            deg2 = sum(1 for u in range(g.n) if len(g.adj[u]) == 2)
            cg = compress(g)
            rec = {"exp": "struct", "inst": iname, "n": g.n, "m": len(g.edges), "blocks": len(bl),
                   "max_block": len(big[0]), "arts_in_max_block": len(big[0] & art),
                   "deg2": deg2, "compressed_n": cg.n, "compressed_m": len(cg.edges)}
            f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    exp, out = sys.argv[1], sys.argv[2]
    if exp == "lagr":
        lagr_study(out)
    elif exp == "struct":
        structure(out)
    else:
        sh = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        ns = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        run(exp, out, sh, ns)
