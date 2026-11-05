"""Compare every configuration against the exhaustive solver on many small random graphs."""
import itertools
import random
import sys
from dataclasses import replace

sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
from lp.brute import brute  # noqa: E402
from lp.bnb import BnBOpts  # noqa: E402
from lp.graph import Graph  # noqa: E402
from lp.solver import ALL_INIT, Config, solve  # noqa: E402


def rand_graph(rng):
    kind = rng.choice(["sparse", "dense", "chain", "cactus", "tree", "disc"])
    n = rng.randint(2, 10)
    edges = {}

    def add(u, v):
        if u != v:
            edges[(min(u, v), max(u, v))] = round(rng.uniform(0.1, 10), 2) if rng.random() > 0.1 else float(rng.randint(1, 3))

    if kind in ("sparse", "chain", "tree", "disc"):
        for v in range(1, n):
            if kind == "disc" and rng.random() < 0.25:
                continue
            add(v, rng.randrange(v) if kind != "chain" else v - 1)
        extra = {"sparse": n // 3, "chain": 2, "tree": 0, "disc": 1}[kind]
        for _ in range(extra):
            add(rng.randrange(n), rng.randrange(n))
    elif kind == "dense":
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.5:
                    add(u, v)
    else:  # cactus: small cycles glued at vertices
        verts = [0]
        nxt = 1
        while nxt < n:
            a = rng.choice(verts)
            k = rng.randint(1, 4)
            cyc = [a] + list(range(nxt, min(n, nxt + k)))
            nxt += len(cyc) - 1
            for x, y in zip(cyc, cyc[1:]):
                add(x, y)
            if len(cyc) >= 3:
                add(cyc[-1], a)
            verts += cyc[1:]
    es = [(u, v, w) for (u, v), w in edges.items()]
    used = sorted({x for u, v, _ in es for x in (u, v)})
    if not used:
        return None
    idx = {v: i for i, v in enumerate(used)}
    return Graph(len(used), [(idx[u], idx[v], w) for u, v, w in es])


def configs():
    flags = ["reach", "b_top1", "b_top2", "b_bcc", "b_lagr", "dominance"]
    out = []
    rng = random.Random(1)
    for _ in range(40):
        o = BnBOpts(**{f: rng.random() < 0.5 for f in flags + ["cycle_1forest"]},
                    order=rng.choice(["none", "weight", "bound", "lagr"]),
                    partial_bound=rng.choice(["global", "local"]),
                    lagr_depth=rng.choice([0, 3, 10**9]))
        init = tuple(x for x in ALL_INIT if rng.random() < 0.5)
        out.append(Config(mode=rng.choice(["decomp", "whole"]), compress=rng.random() < 0.6, bnb=o,
                          init=init, edge_del=rng.random() < 0.4, t_dp=rng.choice([0, 3, 6, 12]),
                          k_max=rng.choice([1, 2, 6]), lagr_init=rng.choice(["zero", "third"]),
                          lagr_cycle=rng.random() < 0.5, lagr_auto=rng.random() < 0.5,
                          lagr_auto_ratio=rng.choice([1.0, 1.3, 2.0]),
                          q_max=rng.choice([None, 0, 5, 40])))
    out.append(Config())
    out.append(Config(mode="whole"))
    out.append(Config(mode="whole", workers=2))
    out.append(Config(workers=2, q_max=99))
    return out


def main(trials=400):
    rng = random.Random(int(__import__("os").environ.get("SEED", "0")))
    cfgs = configs()
    bad = 0
    count = 0
    for t in range(trials):
        g = rand_graph(rng)
        if g is None:
            continue
        ref, _ = brute(g)
        for cfg in cfgs:
            if cfg.workers > 1 and t % 20:
                continue
            (L, p), st = solve(g, cfg)
            count += 1
            if abs(L - ref) > 1e-6:
                bad += 1
                print("MISMATCH", t, ref, L, p, cfg, g.edges)
                if bad > 5:
                    return 1
    print(f"checked {count} solves on {trials} graphs, mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 400))
