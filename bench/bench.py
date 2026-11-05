"""Time the C++ binary on bench/instances. Usage: python3 bench/bench.py BINARY bench/results/OUT.jsonl [names...]"""
import json
import os
import re
import subprocess
import sys
import time

BIN, OUT = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or sorted(f[:-4] for f in os.listdir("bench/instances"))
LIMIT = os.environ.get("LIMIT", "120")
CONFIGS = {"threads=1": ["--threads", "1"], "threads=2": ["--threads", "2"]}
extra = os.environ.get("EXTRA", "")
if extra:
    CONFIGS = {extra: extra.split()}

with open(OUT, "a") as f:
    for name in names:
        text = open(f"bench/instances/{name}.txt", "rb").read()
        for cname, fl in CONFIGS.items():
            t0 = time.perf_counter()
            r = subprocess.run([BIN, "--verbose", "--time-limit", "true", "--time-limit-sec", LIMIT] + fl,
                               input=text, capture_output=True)
            wall = time.perf_counter() - t0
            err = r.stderr.decode()
            m = dict(re.findall(r"(\w+)=([\w.\-]+)", err))
            rec = {"inst": name, "cfg": cname, "exit": r.returncode, "wall": wall,
                   "len": float(m["length"]) if "length" in m else None,
                   "nodes": int(m["nodes"]) if "nodes" in m else None, "stats": m}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(name, cname, "exit", r.returncode, f"{wall:.3f}s", m.get("length"), m.get("nodes"),
                  "fallback" if m.get("fallback", "0") != "0" else "", flush=True)
