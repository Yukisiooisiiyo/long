"""Exactness with awkward distances: huge values that differ only in the last decimal, mixed
numbers of decimals, and forms that fall back to floating point (exponents, >6 decimals).
The reference is a brute force in exact rational arithmetic.
Usage: python3 tests/check_precision.py BINARY TRIALS SEED"""
import itertools, random, subprocess, sys
from fractions import Fraction
BIN, TRIALS, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(SEED)

def brute(n, E):
    adj = {}
    for u, v, w in E:
        adj.setdefault(u, {})[v] = max(adj.get(u, {}).get(v, Fraction(0)), w) if v in adj.get(u, {}) else w
        adj.setdefault(v, {})[u] = adj[u][v]
    best = Fraction(0)
    def dfs(s, u, seen, L):
        nonlocal best
        best = max(best, L)
        for v, w in adj.get(u, {}).items():
            if v == s and len(seen) >= 3: best = max(best, L + w)
            if v not in seen:
                seen.add(v); dfs(s, v, seen, L + w); seen.remove(v)
    for s in adj: dfs(s, s, {s}, Fraction(0))
    return best, adj

def fmt(kind):
    if kind == "big":  # ~1e9 with 2 decimals: differences of 0.01 must count
        x = rng.randint(10**11, 10**11 + 50); return f"{x // 100}.{x % 100:02d}"
    if kind == "mixed":
        d = rng.randint(0, 6); x = rng.randint(1, 10**(d + 2)); s = str(x).rjust(d + 1, "0")
        return s if d == 0 else s[:-d] + "." + s[-d:]
    if kind == "exp":
        return f"{rng.randint(1, 99)}e{rng.randint(0, 2)}"
    return f"{rng.randint(1, 10**8)}.{rng.randint(0, 10**7 - 1):07d}"  # 7 decimals: floating point

bad = {k: 0 for k in ("big", "mixed", "exp", "long")}
cnt = dict(bad)
for t in range(TRIALS):
    kind = rng.choice(list(bad))
    n = rng.randint(2, 8)
    E = []
    for u, v in itertools.combinations(range(1, n + 1), 2):
        if rng.random() < 0.5: E.append((u, v, fmt(kind)))
    if not E: continue
    opt, adj = brute(n, [(u, v, Fraction(w) if "e" not in w else Fraction(float(w))) for u, v, w in E])
    text = "".join(f"{u}, {v}, {w}\r\n" for u, v, w in E)
    out = subprocess.run([BIN], input=text.encode(), capture_output=True).stdout.decode().split("\r\n")
    p = [int(x) for x in out if x]
    L = sum((adj[a][b] for a, b in zip(p, p[1:])), Fraction(0))
    cnt[kind] += 1
    if L != opt:
        bad[kind] += 1
        if bad[kind] <= 3: print("MISMATCH", kind, float(L), float(opt), "diff", float(opt - L))
print("runs", cnt, "mismatches", bad)
