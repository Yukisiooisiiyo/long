"""Scaling sweep: same generators as the Python prototype (longest_path/bench/gen.py), small to large.
Usage: python3 bench/sweep.py BINARY OUT.jsonl FAMILY [LIMIT_SEC]"""
import json, os, re, subprocess, sys, time
import importlib.util
PROTO = os.environ.get("PROTO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototype_python"))
sys.path.insert(0, PROTO)
_spec = importlib.util.spec_from_file_location("gen", os.path.join(PROTO, "bench", "gen.py"))
gen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen)
railway, grid, cactus, dense, to_text = gen.railway, gen.grid, gen.cactus, gen.dense, gen.to_text

BIN, OUT, FAM = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = float(sys.argv[4]) if len(sys.argv) > 4 else 30
SEEDS = [1, 2, 3]
FAMS = {
    "rail": [(f"rail{n}", lambda s, n=n: railway(n, s)) for n in
             [20, 40, 60, 80, 100, 120, 160, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]],
    "cactus": [(f"cactus{n}", lambda s, n=n: cactus(n, s)) for n in
               [30, 100, 300, 1000, 3000, 10000, 30000, 100000]],
    "grid": [(f"grid{r}x{c}", lambda s, r=r, c=c: grid(r, c, s)) for r, c in
             [(3, 3), (4, 4), (5, 5), (5, 6), (6, 6), (6, 7), (7, 7), (7, 8), (8, 8), (8, 9), (9, 9), (9, 10), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]],
    "dense": [(f"dense{n}", lambda s, n=n: dense(n, s)) for n in
              [10, 14, 18, 22, 26, 30, 34, 36, 38, 40, 44, 48, 52, 56, 60, 70, 80]],
    # same 30 stations, from sparse to complete
    "dens30": [(f"p{p:.2f}_n30", lambda s, p=p: dense(30, s, p)) for p in
               [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
    # long thin grids: few rows, many columns
    "strip": [(f"grid{r}x{c}", lambda s, r=r, c=c: grid(r, c, s)) for r, c in
              [(3, 10), (3, 30), (3, 100), (4, 30), (4, 100), (5, 30), (5, 60), (6, 100), (8, 100)]],
}
with open(OUT, "a") as f:
    fails = 0
    for name, mk in FAMS[FAM]:
        if fails >= 2:
            print(name, "skipped (two smaller sizes already timed out)", flush=True)
            continue
        timed = 0
        for s in SEEDS:
            g = mk(s)
            text = to_text(g).encode()
            t0 = time.perf_counter()
            r = subprocess.run([BIN, "--verbose", "--time-limit", "true", "--time-limit-sec", str(LIMIT)],
                               input=text, capture_output=True)
            wall = time.perf_counter() - t0
            m = dict(re.findall(r"(\w+)=([\w.\-]+)", r.stderr.decode()))
            rec = {"fam": FAM, "inst": name, "seed": s, "n": g.n, "m": len(g.edges), "exit": r.returncode,
                   "wall": wall, "stats": m}
            f.write(json.dumps(rec) + "\n"); f.flush()
            timed += r.returncode == 2
            print(name, s, f"n={g.n} m={len(g.edges)}", "exit", r.returncode, f"{wall:.3f}s",
                  "nodes", m.get("nodes"), "maxblk", m.get("max_block"), "thr", m.get("threads"), flush=True)
        fails = fails + 1 if timed == len(SEEDS) else 0
