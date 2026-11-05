"""Exactness on bigger instances: the new defaults against an independent route.
Grids: the frontier DP without pruning or beam (generous caps). Dense: an earlier verified build
(REF_BIN) or this build without the LP branch and cut.
Usage: python3 tests/check_big.py BINARY [REF_BIN]"""
import glob, re, subprocess, sys
BIN = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else None
def L(cmd, f):
    r = subprocess.run(cmd, input=open(f, 'rb').read(), capture_output=True, timeout=900)
    m = re.search(r"length=([\d.]+)", r.stderr.decode())
    return float(m.group(1)) if m else None
bad = 0
cases = []
for f in sorted(glob.glob("bench/data/tune/grid1[01]x1[01]_s*.txt")) + sorted(glob.glob("bench/data/tune/grid9x*_s*.txt")):
    cases.append((f, [BIN, "--verbose", "--x", "fdp_prune_width=99", "--x", "fdp_layer=100000000", "--x", "fdp_budget=2000000000", "--x", "fdp_width_sparse=16", "--x", "fdp_rank=0"]))
for f in sorted(glob.glob("bench/data/tune/dense[456]*_s*.txt")):
    cases.append((f, [REF, "--verbose"] if REF else [BIN, "--verbose", "--x", "lp_bnc=0"]))
for f, ref in cases:
    a = L([BIN, "--verbose"], f)
    b = L(ref, f)
    ok = a is not None and b is not None and abs(a - b) < 1e-6
    bad += not ok
    print(f.split('/')[-1], a, b, "ok" if ok else "MISMATCH", flush=True)
print("mismatches:", bad)
