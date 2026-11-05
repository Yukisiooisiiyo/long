"""CLI: python main.py [--mode decomp|whole] [--workers N] [--time-limit true --time-limit-sec S
                      --on-timeout error|approx] [--verbose] < input"""
import argparse
import sys

from lp.graph import InputError, parse
from lp.solver import Config, solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="decomp", choices=["decomp", "whole"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", default="false", choices=["true", "false"])
    ap.add_argument("--time-limit-sec", type=float, default=10.0)
    ap.add_argument("--on-timeout", default="error", choices=["error", "approx"])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    data = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    try:
        ids, g = parse(data)
    except InputError as e:
        sys.stderr.write(f"input error: {e}\n")
        return 1
    if g.n == 0:
        return 0
    cfg = Config(mode=a.mode, workers=a.workers)
    tl = a.time_limit_sec if a.time_limit == "true" else float("inf")
    (L, path), st = solve(g, cfg, time_limit=tl)
    if st["timeout"]:
        if a.on_timeout == "error":
            sys.stderr.write("time limit reached before an exact solution was found\n")
            return 2
        if a.verbose:
            sys.stderr.write("time limit reached; output is approximate\n")
    if a.verbose:
        sys.stderr.write(f"length={L:.6f} stats={dict(st)}\n")
    out = "".join(f"{ids[v]}\r\n" for v in path)
    sys.stdout.buffer.write(out.encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
