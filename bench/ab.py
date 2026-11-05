"""A/B timing of builds and switches. Runs every configuration on every input, interleaved, and
prints the median wall time (ms) per input plus per-set totals.
Usage: python3 bench/ab.py SET[,SET..] NAME=BINARY[|flag flag..] ...
Sets: example, normal, std, hard."""
import glob, os, re, statistics, subprocess, sys, time

SETS = {
    "example": ["sample_input.txt"],
    "normal": sorted(glob.glob("bench/data/normal/*.txt")),
    "std": sorted(glob.glob("bench/instances/*.txt")),
    "hard": [f"bench/data/tune/{n}.txt" for n in ("grid10x10_s1", "grid11x11_s1", "grid12x12_s1", "grid13x13_s1",
                                                  "dense48_s1", "dense52_s2", "dense60_s2")]
            + ["bench/data/hard/rail2000_s3.txt", "bench/data/hard/grid4x50_s1.txt"],
}
REPS = {"example": 41, "normal": 7, "std": 5, "hard": 3}
sets = sys.argv[1].split(",")
cfgs = []
for a in sys.argv[2:]:
    name, rest = a.split("=", 1)
    parts = rest.split("|")
    cfgs.append((name, parts[0], parts[1].split() if len(parts) > 1 and parts[1] else []))
print("input", *[c[0] for c in cfgs], sep="\t")
for st in sets:
    tot = {c[0]: 0.0 for c in cfgs}
    for f in SETS[st]:
        data = open(f, "rb").read()
        ts = {c[0]: [] for c in cfgs}
        lens = set()
        for _ in range(REPS[st]):
            for name, b, fl in cfgs:
                t0 = time.perf_counter()
                r = subprocess.run([b, "--verbose"] + fl, input=data, capture_output=True)
                ts[name].append(time.perf_counter() - t0)
                m = re.search(r"length=([\d.e+]+)", r.stderr.decode())
                lens.add((r.returncode, r.stdout))
        row = [os.path.basename(f)[:-4]]
        for name, _, _ in cfgs:
            med = statistics.median(ts[name])
            tot[name] += med
            row.append(f"{med*1000:.2f}")
        if len(lens) > 1:
            row.append("OUTPUT DIFFERS")
        print(*row, sep="\t", flush=True)
    print(f"[{st} total]", *[f"{tot[c[0]]*1000:.1f}" for c in cfgs], sep="\t", flush=True)
