"""Benchmark matrix: scale x density x structure.

  structure: loops (small cycles glued in a tree), rail (points in the plane:
             tree of nearest links + short extra links), grid (square lattice)
  density:   edges per station m/n = 1.2, 1.5, 2.0, 3.0
  scale:     n = 30, 100, 300, 1000, 3000

Usage: python3 bench/matrix.py BINARY OUT.jsonl [LIMIT_SEC] [STRUCT,...]
A size is skipped once every seed of a smaller size of the same row timed out.
"""
import json, math, random, re, subprocess, sys, time

SCALES = [30, 100, 300, 1000, 3000]
DENS = [1.2, 1.5, 2.0, 3.0]
SEEDS = [1, 2]


def text(edges):
    return "".join(f"{u + 1}, {v + 1}, {w}\r\n" for u, v, w in edges)


def loops(n, d, rng):
    """Blocks of 4-8 stations glued at one station; chords inside blocks raise density."""
    edges, blocks, have = [], [], set()
    nxt = 1
    def add(a, b):
        k = (min(a, b), max(a, b))
        if a != b and k not in have:
            have.add(k); edges.append((a, b, round(rng.uniform(1, 10), 2))); return True
        return False
    while nxt < n:
        a = rng.randrange(nxt)
        k = min(rng.randint(4, 8), n - nxt + 1)
        cyc = [a] + list(range(nxt, nxt + k - 1))
        nxt += k - 1
        for x, y in zip(cyc, cyc[1:]):
            add(x, y)
        if len(cyc) >= 3:
            add(cyc[-1], cyc[0])
        blocks.append(cyc)
    target = int(d * n)
    tries = 0
    while len(edges) < target and tries < 50 * n:
        tries += 1
        b = rng.choice(blocks)
        if len(b) >= 4:
            add(*rng.sample(b, 2))
    return edges


def rail(n, d, rng):
    """Random points; the tree of nearest links, plus the extra links picked among the shortest."""
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    dist = lambda a, b: round(100 * math.dist(pts[a], pts[b]), 2) + 0.01
    K = 12
    cell = {}
    g = max(1, int(math.sqrt(n / 4)))
    for i, (x, y) in enumerate(pts):
        cell.setdefault((int(x * g), int(y * g)), []).append(i)
    pairs = set()
    for i, (x, y) in enumerate(pts):
        cx, cy = int(x * g), int(y * g)
        near = []
        r = 1
        while True:
            near = [j for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                    for j in cell.get((cx + dx, cy + dy), []) if j != i]
            if len(near) >= K or r > g:
                break
            r += 1
        near.sort(key=lambda j: math.dist(pts[i], pts[j]))
        for j in near[:K]:
            pairs.add((min(i, j), max(i, j)))
    pairs = sorted((dist(a, b), a, b) for a, b in pairs)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edges, rest = [], []
    for w, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; edges.append((a, b, w))
        else:
            rest.append((w, a, b))
    need = max(0, int(d * n) - len(edges))
    cand = rest[: need * 3 // 2 + 1]
    rng.shuffle(cand)
    edges += [(a, b, w) for w, a, b in sorted(cand[:need])]
    return edges


def grid(n, d, rng):
    """Square lattice; below 2.0 a random spanning tree plus random lattice links, at 3.0 one diagonal per square."""
    r = int(round(math.sqrt(n))); c = max(2, round(n / r))
    lat = []
    for i in range(r):
        for j in range(c):
            v = i * c + j
            if j + 1 < c: lat.append((v, v + 1))
            if i + 1 < r: lat.append((v, v + c))
    w = lambda: round(rng.uniform(1, 10), 2)
    N = r * c
    if d >= 2.0:
        edges = [(a, b, w()) for a, b in lat]
        if d > 2.0:
            for i in range(r - 1):
                for j in range(c - 1):
                    v = i * c + j
                    edges.append((v, v + c + 1, w()) if rng.random() < 0.5 else (v + 1, v + c, w()))
        return edges
    rng.shuffle(lat)
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edges, rest = [], []
    for a, b in lat:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; edges.append((a, b, w()))
        else:
            rest.append((a, b))
    need = max(0, int(d * N) - len(edges))
    edges += [(a, b, w()) for a, b in rest[:need]]
    return edges


GENS = {"loops": loops, "rail": rail, "grid": grid}


def main():
    BIN, OUT = sys.argv[1], sys.argv[2]
    LIMIT = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    structs = sys.argv[4].split(",") if len(sys.argv) > 4 else list(GENS)
    with open(OUT, "a") as f:
        for st in structs:
            for d in DENS:
                dead = False
                for n in SCALES:
                    for s in SEEDS:
                        rec = {"struct": st, "dens": d, "n": n, "seed": s}
                        if dead:
                            rec["status"] = "skip"
                        else:
                            e = GENS[st](n, d, random.Random(s * 1000 + n))
                            rec["stations"] = len({x for u, v, _ in e for x in (u, v)})
                            rec["edges"] = len(e)
                            t0 = time.perf_counter()
                            p = subprocess.run([BIN, "--verbose", "--time-limit", "true",
                                                "--time-limit-sec", str(LIMIT)],
                                               input=text(e).encode(), capture_output=True)
                            rec["sec"] = round(time.perf_counter() - t0, 3)
                            rec["status"] = "ok" if p.returncode == 0 else ("timeout" if p.returncode == 2 else f"rc{p.returncode}")
                            err = p.stderr.decode()
                            for k in ("length", "max_block", "blocks", "fdp_blocks", "bitdp_blocks", "bnb_blocks", "lp_runs", "lp_nodes", "nodes"):
                                m = re.search(rf"\b{k}=([\d.]+)", err)
                                if m: rec[k] = float(m.group(1))
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(st, d, n, s, rec["status"], rec.get("sec", ""), "maxblk", rec.get("max_block", ""), flush=True)
                    row = [json.loads(l) for l in open(OUT)]
                    if all(r["status"] in ("timeout", "skip") for r in row
                           if r["struct"] == st and r["dens"] == d and r["n"] == n):
                        dead = True


if __name__ == "__main__":
    main()
