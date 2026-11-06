"""Same search on every platform: runs fixed inputs single-threaded and records the search
statistics (answer length, nodes of each method, LP nodes, frontier DP states). Two builds that
take the same path give identical records, whatever the compiler, standard library or CPU.

  record:  python3 bench/determinism.py BINARY OUT.json
  compare: python3 bench/determinism.py BINARY OUT.json --ref bench/results/determinism_ref.json

Exit code 1 when --ref is given and any record differs."""
import glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(HERE, "data", "hard", "*.txt")))
FILES += [os.path.join(HERE, "data", "hardset", f + ".txt") for f in (
    "dense100_s1", "dense150_s2", "dense200_s1", "er5_100_s2", "er5_200_s1", "reg3_100_s2",
    "reg3_200_s2", "reg4_200_s1", "grid14x14_s2", "grid14x14_s3", "grid16x16_s2")]
KEYS = ("length", "nodes", "blocks", "max_block", "bitdp_blocks", "bnb_blocks", "dead_edges", "lagr_off",
        "proved_at_root", "fdp_blocks", "fdp_width", "fdp_states", "fdp_fallback", "tasks_skipped",
        "lp_runs", "lp_nodes", "lp_fallback")


def run(binary, path):
    with open(path, "rb") as f:
        p = subprocess.run([binary, "--verbose", "--threads", "1", "--time-limit", "true", "--time-limit-sec", "120"],
                           stdin=f, capture_output=True)
    rec = {"status": p.returncode}
    err = p.stderr.decode(errors="replace")
    for k in KEYS:
        m = re.search(rf"\b{k}=([^\s]+)", err)
        if m:
            rec[k] = m.group(1)
    return rec


def main():
    binary, out = sys.argv[1], sys.argv[2]
    ref = None
    if "--ref" in sys.argv:
        with open(sys.argv[sys.argv.index("--ref") + 1]) as f:
            ref = json.load(f)
    res = {}
    diff = 0
    for path in FILES:
        name = os.path.basename(path)
        res[name] = run(binary, path)
        mark = ""
        if ref is not None:
            if ref.get(name) == res[name]:
                mark = "same"
            else:
                diff += 1
                bad = [k for k in set(res[name]) | set(ref.get(name, {})) if res[name].get(k) != ref.get(name, {}).get(k)]
                mark = "DIFF " + ", ".join(f"{k}: {ref.get(name, {}).get(k)} -> {res[name].get(k)}" for k in sorted(bad))
        print(name, res[name].get("length"), "lp_nodes=" + str(res[name].get("lp_nodes")), mark, flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    if ref is not None:
        print(f"{len(FILES)} inputs, {diff} differ from the reference")
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
