"""Hard instances for the next round: large square grids, dense graphs above the LP size limit, and
medium-density random graphs (3- and 4-regular, average degree 5), which are neither thin enough
for the frontier DP nor dense enough for the current LP rule. Writes bench/data/hardset/*.txt."""
import os, random, sys
import networkx as nx
import importlib.util
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototype_python")
sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location("gen", os.path.join(_p, "bench", "gen.py"))
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
grid, dense = _gen.grid, _gen.dense

out = os.path.join(os.path.dirname(__file__), "data", "hardset")
os.makedirs(out, exist_ok=True)
def save(name, edges):
    with open(os.path.join(out, name + ".txt"), "w") as f:
        f.write("".join(f"{u + 1}, {v + 1}, {w}\r\n" for u, v, w in edges))
for s in (1, 2, 3):
    for k in (14, 16, 20):
        save(f"grid{k}x{k}_s{s}", grid(k, k, s).edges)
    for n in (100, 150, 200):
        save(f"dense{n}_s{s}", dense(n, s).edges)
    rng = random.Random(s)
    for d in (3, 4):
        for n in (50, 100, 200):
            G = nx.random_regular_graph(d, n, seed=s * 1000 + n + d)
            save(f"reg{d}_{n}_s{s}", [(u, v, round(rng.uniform(1, 10), 2)) for u, v in G.edges()])
    for n in (50, 100, 200):  # Erdos-Renyi, average degree 5, largest component
        G = nx.gnm_random_graph(n, 5 * n // 2, seed=s * 7 + n)
        G = G.subgraph(max(nx.connected_components(G), key=len))
        save(f"er5_{n}_s{s}", [(u, v, round(rng.uniform(1, 10), 2)) for u, v in G.edges()])
