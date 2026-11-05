"""Cross-check the frontier DP against branch and bound (and brute force on small graphs).
Usage: python3 tests/check_fdp.py BINARY PROTOTYPE_DIR TRIALS SEED"""
import random, subprocess, sys
BIN, PROTO, TRIALS, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, PROTO); sys.path.insert(0, PROTO + "/bench"); sys.path.insert(0, PROTO + "/tests")
sys.path.insert(0, __import__("os").path.dirname(__file__))
from lp.brute import brute  # noqa: E402
from test_correctness import rand_graph  # noqa: E402
from gen import cactus, dense, grid, railway  # noqa: E402
from check import to_text, run, length_of  # noqa: E402

ON = ["--x", "fdp_min_n=1", "--x", "fdp_width=24"]
VARIANTS = [[], ON, ON + ["--t-dp", "0"], ON + ["--t-dp", "3"], ON + ["--t-dp", "0", "--threads", "1"],
            ["--x", "fdp_budget=50"] + ON + ["--t-dp", "0"],  # forces the give-up path
            ON + ["--x", "fdp_width=5", "--t-dp", "0"],
            # pruning and the beam pass on every DP run (tiny beam: the exact pass must still be exact)
            ON + ["--x", "fdp_prune_width=0", "--x", "fdp_beam=3"],
            ON + ["--x", "fdp_prune_width=0", "--x", "fdp_beam=50", "--t-dp", "0"],
            # rank-based reduction on every step
            ON + ["--x", "fdp_rank=1", "--x", "fdp_beam=0"], ON + ["--x", "fdp_rank=1", "--t-dp", "0"],
            # Lagrangian branching split as deep as possible
            ["--x", "fdp=0", "--x", "lp_bnc=0", "--x", "task_budget=1", "--x", "task_depth=30"],
            ["--x", "fdp=0", "--x", "task_budget=1", "--x", "task_depth=30", "--t-dp", "0", "--threads", "2"],
            ["--x", "fdp=0", "--x", "path_task=0", "--x", "lp_bnc=0"],
            # LP branch and cut forced on everything dense enough, and with a tiny node cap (fallback)
            ["--x", "fdp=0", "--x", "lp_max_n=100000"], ["--x", "lp_nodes=2", "--x", "fdp=0", "--x", "task_budget=1"],
            # small blocks: bitmask DP only / enumeration that gives up at once
            ["--x", "enum_n=0"], ["--x", "enum_budget=5"]]
rng = random.Random(SEED)
bad = checks = 0
for t in range(TRIALS):
    kind = rng.choice(["rand", "rand", "rail", "grid", "strip", "cactus", "dense"])
    g = {"rand": lambda: rand_graph(rng),
         "rail": lambda: railway(rng.randint(15, 160), rng.randrange(10**6)),
         "grid": lambda: grid(rng.randint(2, 6), rng.randint(2, 6), rng.randrange(10**6)),
         "strip": lambda: grid(rng.randint(2, 3), rng.randint(3, 25), rng.randrange(10**6)),
         "cactus": lambda: cactus(rng.randint(10, 150), rng.randrange(10**6)),
         "dense": lambda: dense(rng.randint(5, 14), rng.randrange(10**6), rng.choice([0.3, 0.5, 0.9]))}[kind]()
    if g is None or not g.edges:
        continue
    text, ids = to_text(g, rng)
    if g.n <= 11:
        ref = brute(g)[0]
    else:  # an independent reference binary (an earlier verified build) when given, else B&B only
        import os
        rb = os.environ.get("REF_BIN")
        if rb:
            r = subprocess.run([rb, "--x", "fdp=0"], input=text.encode(), capture_output=True, timeout=300)
            ref = length_of(g, ids, [int(x) for x in r.stdout.decode().split("\r\n") if x])
        else:
            ref = length_of(g, ids, run(text, ["--x", "fdp=0", "--x", "lp_bnc=0", "--x", "path_task=0"]))
    for fl in VARIANTS:
        L = length_of(g, ids, run(text, fl))
        checks += 1
        if abs(L - ref) > 1e-6 * (1 + abs(ref)):
            bad += 1
            print("MISMATCH", kind, g.n, fl, L, ref, flush=True)
            open(f"/tmp/fdp_bad_{bad}.txt", "w").write(text)
print(f"checked {checks} runs, mismatches: {bad}")
