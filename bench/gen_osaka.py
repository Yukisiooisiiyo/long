"""Build bench/data/real/osaka.txt: every station in Osaka prefecture (ekidata.jp CSV, 2018 free
edition, via github.com/piuccio/open-data-jp-railway-stations and -lines). Stations joined by a
transfer (same station_g_cd) are one node; consecutive stations of a line (e_sort order) are an edge
with the great-circle distance in km (2 decimals); the Osaka Loop line is closed.
Usage: python3 bench/gen_osaka.py STATIONS_CSV LINES_CSV OUT.txt NAMES.json"""
import collections, csv, json, math, sys

st_csv, ln_csv, out, names_out = sys.argv[1:5]
rows = [r for r in csv.DictReader(open(st_csv, encoding="utf-8")) if r["pref_cd"] == "27" and r["e_status"] == "0"]
by_line = collections.defaultdict(list)
for r in rows:
    by_line[r["line_cd"]].append(r)


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"])))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


E, name = {}, {}
for lc, st in by_line.items():
    st.sort(key=lambda r: int(r["e_sort"]))
    for r in st:
        name[r["station_g_cd"]] = r["station_name"]
    pairs = list(zip(st, st[1:]))
    if lc == "11623":  # 大阪環状線
        pairs.append((st[-1], st[0]))
    for a, b in pairs:
        u, v = a["station_g_cd"], b["station_g_cd"]
        if u != v:
            k = (min(u, v), max(u, v))
            E[k] = max(E.get(k, 0), round(hav(a, b), 2))
ids = sorted(name)
idx = {g: i + 1 for i, g in enumerate(ids)}
with open(out, "w", newline="") as f:
    f.write("".join(f"{idx[u]}, {idx[v]}, {d:.2f}\r\n" for (u, v), d in sorted(E.items())))
json.dump({idx[g]: name[g] for g in ids}, open(names_out, "w"), ensure_ascii=False)
print(len(ids), "stations", len(E), "edges")
