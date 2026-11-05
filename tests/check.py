"""Cross-check the C++ binary against the Python brute-force solver (small graphs) and the
Python prototype (medium graphs). Usage: python3 tests/check.py BINARY PROTOTYPE_DIR TRIALS SEED"""
import random
import subprocess
import sys

BIN, PROTO, TRIALS, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, PROTO)
sys.path.insert(0, PROTO + "/bench")
sys.path.insert(0, PROTO + "/tests")
from lp.brute import brute  # noqa: E402
from lp.solver import Config, solve  # noqa: E402
from test_correctness import rand_graph  # noqa: E402
from gen import cactus, dense, grid, railway  # noqa: E402

FLAGS = [[], ["--threads", "1"], ["--threads", "2"], ["--t-dp", "0"], ["--t-dp", "0", "--threads", "1"],
         ["--t-dp", "20"], ["--lagr-ratio", "0"], ["--lagr-ratio", "100"], ["--no-edge-deletion"],
         ["--t-dp", "3", "--threads", "2"],
         ["--x", "fdp=0"], ["--x", "lp_bnc=0"], ["--x", "path_task=0"], ["--x", "enum_n=0"],
         ["--x", "fdp=0", "--x", "lp_bnc=0", "--x", "path_task=0", "--t-dp", "0", "--threads", "2"],
         ["--x", "task_budget=1", "--x", "task_depth=30", "--x", "lp_bnc=0", "--t-dp", "0"]]


def to_text(g, rng):
    lines = []
    ids = list(range(1, g.n + 1))
    rng.shuffle(ids)  # IDs are not 1..n in order
    for u, v, w in g.edges:
        a, b = (ids[u], ids[v]) if rng.random() < 0.5 else (ids[v], ids[u])
        sp = lambda: " " * rng.randint(0, 2)
        lines.append(f"{sp()}{a}{sp()},{sp()}{b}{sp()},{sp()}{w}{sp()}")
    nl = rng.choice(["\r\n", "\n"])
    return nl.join(lines) + rng.choice([nl, ""]), ids


def run(text, flags):
    r = subprocess.run([BIN] + flags, input=text.encode(), capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {r.stderr.decode()}")
    out = r.stdout.decode()
    assert out == "" or out.endswith("\r\n"), repr(out[-5:])
    return [int(x) for x in out.split("\r\n") if x]


def length_of(g, ids, out):
    inv = {v: i for i, v in enumerate(ids)}
    p = [inv[x] for x in out]
    cyc = len(p) > 2 and p[0] == p[-1]
    body = p[:-1] if cyc else p
    assert len(set(body)) == len(body), "repeated vertex"
    L = 0.0
    for a, b in zip(p, p[1:]):
        w = g.weight(a, b)
        assert w is not None, "missing edge"
        L += w
    return L


def main():
    rng = random.Random(SEED)
    bad = checks = 0
    for t in range(TRIALS):
        if t % 5 == 4:  # medium instance compared with the Python prototype
            kind = rng.choice(["rail", "grid", "cactus", "dense"])
            g = {"rail": lambda: railway(rng.randint(30, 70), rng.randrange(10**6)),
                 "grid": lambda: grid(rng.randint(3, 5), rng.randint(3, 5), rng.randrange(10**6)),
                 "cactus": lambda: cactus(rng.randint(30, 120), rng.randrange(10**6)),
                 "dense": lambda: dense(rng.randint(8, 14), rng.randrange(10**6))}[kind]()
            (ref, _), _ = solve(g, Config())
            flags_list = [rng.choice(FLAGS), []]
        else:
            g = rand_graph(rng)
            if g is None:
                continue
            ref, _ = brute(g)
            flags_list = FLAGS
        text, ids = to_text(g, rng)
        for fl in flags_list:
            out = run(text, fl)
            L = length_of(g, ids, out) if out else 0.0
            checks += 1
            if abs(L - ref) > 1e-6:
                bad += 1
                print("MISMATCH", t, fl, ref, L, out, g.edges)
                if bad > 5:
                    return 1
    print(f"checked {checks} runs, mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
