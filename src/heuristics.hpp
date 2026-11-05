// Initial solutions (portfolio): greedy, max spanning tree diameter, fundamental cycles,
// beam search, Lagrangian forest repair, iterated local search. All work on one connected graph.
#pragma once
#include <functional>
#include <numeric>
#include <random>

#include "graph.hpp"

namespace lp {

struct Sol {
    double len = -1.0;
    std::vector<int> path;  // a cycle repeats its first vertex at the end
};
inline void take(Sol& a, const Sol& b) {
    if (gt(b.len, a.len)) a = b;
}

struct DSU {
    std::vector<int> p;
    explicit DSU(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
    int f(int x) {
        while (p[x] != x) x = p[x] = p[p[x]];
        return x;
    }
    bool unite(int a, int b) {
        a = f(a), b = f(b);
        if (a == b) return false;
        p[a] = b;
        return true;
    }
};

inline double plen(const Graph& g, const std::vector<int>& p) {
    double L = 0;
    for (size_t i = 0; i + 1 < p.size(); ++i) L += g.weight(p[i], p[i + 1]);
    return L;
}

inline Sol greedy(const Graph& g) {
    Sol best;
    std::vector<char> vis(g.n);
    for (int s = 0; s < g.n; ++s) {
        std::fill(vis.begin(), vis.end(), 0);
        std::vector<int> p{s};
        vis[s] = 1;
        double L = 0;
        while (true) {
            int u = p.back(), bv = -1;
            double bw = -1;
            for (auto [v, e] : g.adj[u])
                if (!vis[v] && g.edges[e].w > bw) bw = g.edges[e].w, bv = v;
            if (bv < 0) break;
            vis[bv] = 1;
            p.push_back(bv);
            L += bw;
        }
        take(best, {L, p});
    }
    return best;
}

inline std::vector<int> max_spanning_tree(const Graph& g) {
    std::vector<int> order(g.edges.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return g.edges[a].w > g.edges[b].w; });
    DSU d(g.n);
    std::vector<int> tree;
    for (int i : order)
        if (d.unite(g.edges[i].u, g.edges[i].v)) tree.push_back(i);
    return tree;
}

inline Sol mst_diameter(const Graph& g) {
    auto tree = max_spanning_tree(g);
    std::vector<std::vector<std::pair<int, double>>> adj(g.n);
    for (int i : tree) {
        adj[g.edges[i].u].push_back({g.edges[i].v, g.edges[i].w});
        adj[g.edges[i].v].push_back({g.edges[i].u, g.edges[i].w});
    }
    auto far = [&](int s) {
        std::vector<double> d(g.n, -1);
        std::vector<int> par(g.n, -1), st{s};
        d[s] = 0;
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            for (auto [v, w] : adj[u])
                if (d[v] < 0) d[v] = d[u] + w, par[v] = u, st.push_back(v);
        }
        int b = (int)(std::max_element(d.begin(), d.end()) - d.begin());
        std::vector<int> p;
        for (int x = b; x != -1; x = par[x]) p.push_back(x);
        std::reverse(p.begin(), p.end());
        return Sol{d[b], p};
    };
    return far(far(0).path.back());
}

inline Sol extend_greedy(const Graph& g, std::vector<int> p) {
    std::vector<char> vis(g.n, 0);
    for (int x : p) vis[x] = 1;
    double L = plen(g, p);
    bool changed = true;
    while (changed) {
        changed = false;
        for (int end = 0; end < 2; ++end) {
            int u = end ? p.back() : p.front(), bv = -1;
            double bw = -1;
            for (auto [v, e] : g.adj[u])
                if (!vis[v] && g.edges[e].w > bw) bw = g.edges[e].w, bv = v;
            if (bv >= 0) {
                if (end) p.push_back(bv);
                else p.insert(p.begin(), bv);
                vis[bv] = 1;
                L += bw;
                changed = true;
            }
        }
    }
    return {L, p};
}

inline Sol fundamental_cycles(const Graph& g, int top_m = 5) {
    auto tree = max_spanning_tree(g);
    std::vector<char> in(g.edges.size(), 0);
    for (int i : tree) in[i] = 1;
    std::vector<std::vector<std::pair<int, double>>> adj(g.n);
    for (int i : tree) {
        adj[g.edges[i].u].push_back({g.edges[i].v, g.edges[i].w});
        adj[g.edges[i].v].push_back({g.edges[i].u, g.edges[i].w});
    }
    std::vector<int> par(g.n, -1), dep(g.n, 0);
    std::vector<double> pw(g.n, 0);
    std::vector<char> seen(g.n, 0);
    std::vector<int> st{0};
    seen[0] = 1;
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        for (auto [v, w] : adj[u])
            if (!seen[v]) seen[v] = 1, par[v] = u, dep[v] = dep[u] + 1, pw[v] = w, st.push_back(v);
    }
    std::vector<std::pair<double, int>> cyc;
    for (int i = 0; i < (int)g.edges.size(); ++i) {
        if (in[i]) continue;
        int a = g.edges[i].u, b = g.edges[i].v;
        double L = g.edges[i].w;
        while (a != b) {
            if (dep[a] >= dep[b]) L += pw[a], a = par[a];
            else L += pw[b], b = par[b];
        }
        cyc.push_back({L, i});
    }
    std::sort(cyc.begin(), cyc.end(), [](auto& x, auto& y) { return x.first > y.first; });
    Sol best;
    for (int k = 0; k < std::min<int>(top_m, (int)cyc.size()); ++k) {
        int i = cyc[k].second;
        int a = g.edges[i].u, b = g.edges[i].v;
        std::vector<int> left{a}, right{b};
        while (a != b) {
            if (dep[a] >= dep[b]) a = par[a], left.push_back(a);
            else b = par[b], right.push_back(b);
        }
        std::vector<int> c = left;
        for (int j = (int)right.size() - 2; j >= 0; --j) c.push_back(right[j]);
        std::vector<int> closed = c;
        closed.push_back(c[0]);
        take(best, {cyc[k].first, closed});
        int m = (int)c.size();
        for (int s = 0; s < m; ++s) {
            std::vector<int> opened;
            for (int j = 1; j <= m; ++j) opened.push_back(c[(s + j) % m]);
            take(best, extend_greedy(g, opened));
        }
    }
    return best;
}

// With `bonus`, a path x..y (at least one edge) is worth len + bonus[x] + bonus[y].
inline Sol beam_search(const Graph& g, int K = 32, double alpha = 0.3,
                       const std::vector<double>* bonus = nullptr) {
    auto B = [&](int v) { return bonus ? (*bonus)[v] : 0.0; };
    int n = g.n, W = (n + 63) / 64;
    std::vector<double> top1(n, 0);
    for (int u = 0; u < n; ++u)
        for (auto [v, e] : g.adj[u]) top1[u] = std::max(top1[u], g.edges[e].w);
    struct St {
        double score, len;
        std::vector<int> path;
        std::vector<uint64_t> vis;
    };
    std::vector<int> mark(n, 0), q;
    int stampv = 0;
    auto reach_sum = [&](int a, int b, const std::vector<uint64_t>& vis) {
        ++stampv;
        q.clear();
        double s = 0;
        for (int e : {a, b})
            for (auto [v, ed] : g.adj[e])
                if (!((vis[v >> 6] >> (v & 63)) & 1) && mark[v] != stampv) mark[v] = stampv, q.push_back(v);
        for (size_t i = 0; i < q.size(); ++i) {
            int u = q[i];
            s += top1[u];
            for (auto [v, ed] : g.adj[u])
                if (!((vis[v >> 6] >> (v & 63)) & 1) && mark[v] != stampv) mark[v] = stampv, q.push_back(v);
        }
        return s;
    };
    std::vector<St> beam;
    for (auto& e : g.edges) {
        std::vector<uint64_t> vis(W, 0);
        vis[e.u >> 6] |= 1ULL << (e.u & 63);
        vis[e.v >> 6] |= 1ULL << (e.v & 63);
        beam.push_back({e.w + B(e.u) + B(e.v) + alpha * reach_sum(e.u, e.v, vis), e.w, {e.u, e.v}, vis});
    }
    auto bycore = [](const St& a, const St& b) { return a.score > b.score; };
    std::sort(beam.begin(), beam.end(), bycore);
    if ((int)beam.size() > K) beam.resize(K);
    Sol best;
    auto val = [&](const St& s) { return s.len + B(s.path.front()) + B(s.path.back()); };
    for (auto& s : beam) take(best, {val(s), s.path});
    while (!beam.empty()) {
        struct KeyH {
            size_t operator()(const std::pair<std::vector<uint64_t>, std::pair<int, int>>& k) const {
                size_t h = std::hash<long long>()(((long long)k.second.first << 32) ^ k.second.second);
                for (auto w : k.first) h ^= std::hash<uint64_t>()(w) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        std::unordered_map<std::pair<std::vector<uint64_t>, std::pair<int, int>>, size_t, KeyH> idx;
        std::vector<St> cand;
        for (auto& s : beam) {
            int a = s.path.front(), b = s.path.back();
            if (s.path.size() >= 3 && !bonus) {
                double wc = g.weight(a, b);
                if (!std::isnan(wc)) {
                    std::vector<int> cp = s.path;
                    cp.push_back(a);
                    take(best, {s.len + wc, cp});
                }
            }
            for (int end = 0; end < 2; ++end) {
                int u = end ? b : a;
                for (auto [v, e] : g.adj[u]) {
                    if ((s.vis[v >> 6] >> (v & 63)) & 1) continue;
                    std::vector<int> np = s.path;
                    if (end) np.push_back(v);
                    else np.insert(np.begin(), v);
                    auto nv = s.vis;
                    nv[v >> 6] |= 1ULL << (v & 63);
                    double nl = s.len + g.edges[e].w;
                    int x = np.front(), y = np.back();
                    auto key = std::make_pair(nv, std::make_pair(std::min(x, y), std::max(x, y)));
                    auto it = idx.find(key);
                    if (it != idx.end() && cand[it->second].len >= nl) continue;
                    double sc = nl + B(x) + B(y) + alpha * reach_sum(x, y, nv);
                    if (it != idx.end()) cand[it->second] = {sc, nl, np, nv};
                    else {
                        idx.emplace(std::move(key), cand.size());
                        cand.push_back({sc, nl, np, nv});
                    }
                }
            }
        }
        if (cand.empty()) break;
        std::sort(cand.begin(), cand.end(), bycore);
        if ((int)cand.size() > K) cand.resize(K);
        beam = std::move(cand);
        for (auto& s : beam) take(best, {val(s), s.path});
    }
    return best;
}

// local search: extend ends, 1/2-vertex detours, Posa rotations (+extend)
inline Sol local_search(const Graph& g, std::vector<int> p) {
    Sol cur = extend_greedy(g, p);
    bool improved = true;
    std::vector<char> vis(g.n);
    while (improved) {
        improved = false;
        std::fill(vis.begin(), vis.end(), 0);
        for (int x : cur.path) vis[x] = 1;
        auto& P = cur.path;
        for (size_t i = 0; i + 1 < P.size() && !improved; ++i) {
            int u = P[i], v = P[i + 1];
            double wuv = g.weight(u, v), bg = 1e-12;
            std::vector<int> ins;
            for (auto [x, e1] : g.adj[u]) {
                if (vis[x]) continue;
                double wux = g.edges[e1].w, wxv = g.weight(x, v);
                if (!std::isnan(wxv) && wux + wxv - wuv > bg) bg = wux + wxv - wuv, ins = {x};
                for (auto [y, e2] : g.adj[x]) {
                    if (vis[y] || y == x) continue;
                    double wyv = g.weight(y, v);
                    if (!std::isnan(wyv) && wux + g.edges[e2].w + wyv - wuv > bg)
                        bg = wux + g.edges[e2].w + wyv - wuv, ins = {x, y};
                }
            }
            if (!ins.empty()) {
                P.insert(P.begin() + i + 1, ins.begin(), ins.end());
                cur = extend_greedy(g, P);
                improved = true;
            }
        }
        if (improved) continue;
        for (int rev = 0; rev < 2 && !improved; ++rev) {
            std::vector<int> q = cur.path;
            if (rev) std::reverse(q.begin(), q.end());
            int v = q.back();
            for (size_t i = 0; i + 2 < q.size(); ++i) {
                if (!g.has(q[i], v)) continue;
                std::vector<int> rot(q.begin(), q.begin() + i + 1);
                rot.insert(rot.end(), q.rbegin(), q.rend() - (i + 1));
                Sol r = extend_greedy(g, rot);
                if (r.len > cur.len + 1e-12) {
                    cur = r;
                    improved = true;
                    break;
                }
            }
        }
    }
    return cur;
}

inline Sol ils(const Graph& g, std::vector<int> p, int iters = 30, unsigned seed = 0) {
    std::mt19937 rng(seed);  // fixed seed: reproducible output
    Sol best = local_search(g, p), curr = best;
    for (int it = 0; it < iters; ++it) {
        if (curr.path.size() < 3) break;
        int m = (int)curr.path.size();
        int k = std::uniform_int_distribution<int>(1, std::max(1, m / 3))(rng);
        double r = std::uniform_real_distribution<double>(0, 1)(rng);
        std::vector<int> q;
        if (r < 0.4) q.assign(curr.path.begin() + k, curr.path.end());
        else if (r < 0.8) q.assign(curr.path.begin(), curr.path.end() - k);
        else {
            int i = std::uniform_int_distribution<int>(1, m - 2)(rng);
            if (i >= m - i) q.assign(curr.path.begin(), curr.path.begin() + i);
            else q.assign(curr.path.begin() + i, curr.path.end());
        }
        Sol s = local_search(g, q);
        if (s.len > best.len + 1e-12) best = curr = s;
        else curr = best;
    }
    return best;
}

// Beam search that keeps the start fixed at x and grows the other end; value = len + b[x] + b[end].
// Used from the vertices with the largest bonuses: the ends of a good bonus path sit there.
inline Sol beam_from(const Graph& g, int x, const std::vector<double>& bonus, int K = 64) {
    int W = (g.n + 63) / 64;
    struct St {
        double score, len;
        std::vector<int> path;
        std::vector<uint64_t> vis;
    };
    std::vector<uint64_t> v0(W, 0);
    v0[x >> 6] |= 1ULL << (x & 63);
    std::vector<St> beam{{0, 0, {x}, v0}};
    Sol best;
    while (!beam.empty()) {
        std::unordered_map<long long, size_t> seen;  // (end, hash of visited) -> index
        std::vector<St> cand;
        for (auto& s : beam) {
            int u = s.path.back();
            for (auto [v, e] : g.adj[u]) {
                if ((s.vis[v >> 6] >> (v & 63)) & 1) continue;
                St t{0, s.len + g.edges[e].w, s.path, s.vis};
                t.path.push_back(v);
                t.vis[v >> 6] |= 1ULL << (v & 63);
                t.score = t.len + bonus[x] + bonus[v];
                take(best, {t.score, t.path});
                size_t h = (size_t)v * 1000003u;
                for (auto w : t.vis) h ^= std::hash<uint64_t>()(w) + 0x9e3779b9 + (h << 6) + (h >> 2);
                auto it = seen.find((long long)h);
                if (it != seen.end() && cand[it->second].vis == t.vis && cand[it->second].path.back() == v) {
                    if (cand[it->second].len >= t.len) continue;
                    cand[it->second] = std::move(t);
                    continue;
                }
                seen[(long long)h] = cand.size();
                cand.push_back(std::move(t));
            }
        }
        // keep the longest prefixes (the end bonus is only collected when the path stops)
        std::sort(cand.begin(), cand.end(), [](const St& a, const St& b) { return a.len > b.len; });
        if ((int)cand.size() > K) cand.resize(K);
        beam = std::move(cand);
    }
    return best;
}

// Long x..y path with both ends fixed: start from a hop-shortest path, then repeatedly insert
// 1-3 vertex detours between consecutive vertices (the ends never move).
inline Sol grow_between(const Graph& g, int x, int y) {
    std::vector<int> par(g.n, -1), q{x};
    par[x] = x;
    for (size_t i = 0; i < q.size() && par[y] < 0; ++i)
        for (auto [v, e] : g.adj[q[i]])
            if (par[v] < 0) par[v] = q[i], q.push_back(v);
    if (par[y] < 0) return {};
    std::vector<int> p;
    for (int v = y; v != x; v = par[v]) p.push_back(v);
    p.push_back(x);
    std::reverse(p.begin(), p.end());
    std::vector<char> vis(g.n, 0);
    for (int v : p) vis[v] = 1;
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            int u = p[i], v = p[i + 1];
            double wuv = g.weight(u, v), bg = 1e-12;
            std::vector<int> ins;
            for (auto [a, e1] : g.adj[u]) {
                if (vis[a]) continue;
                double w1 = g.edges[e1].w, wav = g.weight(a, v);
                if (!std::isnan(wav) && w1 + wav - wuv > bg) bg = w1 + wav - wuv, ins = {a};
                for (auto [b, e2] : g.adj[a]) {
                    if (vis[b] || b == a) continue;
                    double w2 = g.edges[e2].w, wbv = g.weight(b, v);
                    if (!std::isnan(wbv) && w1 + w2 + wbv - wuv > bg) bg = w1 + w2 + wbv - wuv, ins = {a, b};
                    for (auto [c, e3] : g.adj[b]) {
                        if (vis[c] || c == a || c == b) continue;
                        double wcv = g.weight(c, v);
                        if (!std::isnan(wcv) && w1 + w2 + g.edges[e3].w + wcv - wuv > bg)
                            bg = w1 + w2 + g.edges[e3].w + wcv - wuv, ins = {a, b, c};
                    }
                }
            }
            if (!ins.empty()) {
                for (int z : ins) vis[z] = 1;
                p.insert(p.begin() + i + 1, ins.begin(), ins.end());
                improved = true;
            }
        }
        if (improved) continue;
        // reroute: replace u-v by any detour u -> (unvisited vertices) -> v that is longer
        std::vector<int> from(g.n, -2), bq;
        for (size_t i = 0; i + 1 < p.size() && !improved; ++i) {
            int u = p[i], v = p[i + 1];
            std::fill(from.begin(), from.end(), -2);
            bq.clear();
            for (auto [a, e] : g.adj[u])
                if (!vis[a]) from[a] = u, bq.push_back(a);
            int hit = -1;
            for (size_t k = 0; k < bq.size() && hit < 0; ++k) {
                int a = bq[k];
                if (g.has(a, v)) {
                    hit = a;
                    break;
                }
                for (auto [b, e] : g.adj[a])
                    if (!vis[b] && from[b] == -2) from[b] = a, bq.push_back(b);
            }
            if (hit < 0) continue;
            std::vector<int> det;
            for (int z = hit; z != u; z = from[z]) det.push_back(z);
            std::reverse(det.begin(), det.end());
            std::vector<int> seg{u};
            seg.insert(seg.end(), det.begin(), det.end());
            seg.push_back(v);
            if (plen(g, seg) > g.weight(u, v) + 1e-12) {
                for (int z : det) vis[z] = 1;
                p.insert(p.begin() + i + 1, det.begin(), det.end());
                improved = true;
            }
        }
    }
    return {plen(g, p), p};
}

// Local search for the bonus objective len + bonus[front] + bonus[back] (path of >= 2 vertices):
// end extension, 1/2-vertex detours, Posa rotations, then perturb-and-repeat (fixed seed).
inline Sol improve_bonus(const Graph& g, std::vector<int> p, const std::vector<double>& bonus, int iters = 60,
                         unsigned seed = 0) {
    auto value = [&](const std::vector<int>& q) { return plen(g, q) + bonus[q.front()] + bonus[q.back()]; };
    std::vector<char> vis(g.n);
    auto ls = [&](std::vector<int> q) {
        bool improved = true;
        while (improved) {
            improved = false;
            std::fill(vis.begin(), vis.end(), 0);
            for (int x : q) vis[x] = 1;
            // extend an end: v -> x adds w(v,x) + b(x) - b(v)
            for (int end = 0; end < 2 && !improved; ++end) {
                int v = end ? q.back() : q.front();
                double bg = 1e-12;
                int bx = -1;
                for (auto [x, e] : g.adj[v])
                    if (!vis[x] && g.edges[e].w + bonus[x] - bonus[v] > bg) bg = g.edges[e].w + bonus[x] - bonus[v], bx = x;
                if (bx >= 0) {
                    if (end) q.push_back(bx);
                    else q.insert(q.begin(), bx);
                    improved = true;
                }
            }
            if (improved) continue;
            // detours (ends unchanged)
            for (size_t i = 0; i + 1 < q.size() && !improved; ++i) {
                int u = q[i], v = q[i + 1];
                double wuv = g.weight(u, v), bg = 1e-12;
                std::vector<int> ins;
                for (auto [x, e1] : g.adj[u]) {
                    if (vis[x]) continue;
                    double wxv = g.weight(x, v);
                    if (!std::isnan(wxv) && g.edges[e1].w + wxv - wuv > bg) bg = g.edges[e1].w + wxv - wuv, ins = {x};
                    for (auto [y, e2] : g.adj[x]) {
                        if (vis[y] || y == x) continue;
                        double wyv = g.weight(y, v);
                        if (!std::isnan(wyv) && g.edges[e1].w + g.edges[e2].w + wyv - wuv > bg)
                            bg = g.edges[e1].w + g.edges[e2].w + wyv - wuv, ins = {x, y};
                    }
                }
                if (!ins.empty()) {
                    q.insert(q.begin() + i + 1, ins.begin(), ins.end());
                    improved = true;
                }
            }
            if (improved) continue;
            // Posa rotation at either end: new end p[i+1]
            double cur = value(q);
            for (int rev = 0; rev < 2 && !improved; ++rev) {
                std::vector<int> r = q;
                if (rev) std::reverse(r.begin(), r.end());
                int v = r.back();
                for (size_t i = 0; i + 2 < r.size(); ++i) {
                    if (!g.has(r[i], v)) continue;
                    std::vector<int> rot(r.begin(), r.begin() + i + 1);
                    rot.insert(rot.end(), r.rbegin(), r.rend() - (i + 1));
                    if (value(rot) > cur + 1e-12) {
                        q = rot;
                        improved = true;
                        break;
                    }
                }
            }
        }
        return q;
    };
    std::mt19937 rng(seed);
    std::vector<int> best = ls(p), curp = best;
    double bv = value(best);
    for (int it = 0; it < iters; ++it) {
        int m = (int)curp.size();
        if (m < 3) break;
        int k = std::uniform_int_distribution<int>(1, std::max(1, m / 3))(rng);
        double r = std::uniform_real_distribution<double>(0, 1)(rng);
        std::vector<int> q;
        if (r < 0.5) q.assign(curp.begin() + k, curp.end());
        else q.assign(curp.begin(), curp.end() - k);
        if (q.size() < 2) continue;
        q = ls(q);
        double v = value(q);
        if (v > bv + 1e-12) bv = v, best = curp = q;
        else curp = best;
    }
    return {bv, best};
}

}  // namespace lp
