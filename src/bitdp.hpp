// Exact tables for small blocks: best path, best path from each articulation, best path between
// articulation pairs, best cycle. Three ways: closed form for a single cycle, path enumeration for
// sparse blocks, bitmask DP (dp[mask][v] = longest path covering exactly mask, ending at v) otherwise.
#pragma once
#include <functional>
#include <map>

#include "heuristics.hpp"

namespace lp {

struct Tables {
    Sol free;                               // best path (or cycle) inside the block
    std::map<int, Sol> frm;                 // a -> best path starting at a
    std::map<std::pair<int, int>, Sol> pr;  // (a,b) -> best path a..b
    Sol cycle;                              // len < 0: none worth having
};

class BitDP {
   public:
    explicit BitDP(const Graph& g) : g(g), n(g.n), dp((size_t(1) << g.n) * g.n) {}

    Tables tables(const std::vector<int>& terms) {
        Tables T;
        uint32_t full = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
        run(all(), full);
        T.free = {0.0, {0}};
        double bv = 0;
        uint32_t bm = 1;
        int bx = 0;
        for (uint32_t m = 1; m <= full; ++m)
            for (int v = 0; v < n; ++v)
                if (D(m, v) > bv + 1e-12) bv = D(m, v), bm = m, bx = v;
        T.free = {bv, back(bm, bx)};
        std::vector<char> isterm(n, 0);
        for (int a : terms) isterm[a] = 1;
        for (int a : terms) {
            run({a}, full);
            double fb = 0;
            uint32_t fm = 1u << a;
            int fx = a;
            std::map<int, std::pair<double, std::pair<uint32_t, int>>> bp;
            for (uint32_t m = 1; m <= full; ++m)
                for (int v = 0; v < n; ++v) {
                    double val = D(m, v);
                    if (val == kNeg) continue;
                    if (val > fb + 1e-12) fb = val, fm = m, fx = v;
                    if (isterm[v] && v != a) {
                        auto it = bp.find(v);
                        if (it == bp.end() || val > it->second.first + 1e-12) bp[v] = {val, {m, v}};
                    }
                }
            T.frm[a] = {fb, back(fm, fx)};
            for (auto& [b, x] : bp) T.pr[{a, b}] = {x.first, back(x.second.first, x.second.second)};
        }
        // cycles: the smallest vertex s is the start
        T.cycle = Sol{};
        for (int s = 0; s < n; ++s) {
            uint32_t allowed = full & ~((1u << s) - 1);
            run({s}, allowed);
            double cb = -1;
            uint32_t cm = 0;
            int cx = -1;
            for (uint32_t m = 1; m <= full; ++m) {
                if (std::popcount(m) < 3 || !(m >> s & 1)) continue;
                for (auto [v, e] : g.adj[s]) {
                    if (!(m >> v & 1) || D(m, v) == kNeg) continue;
                    double val = D(m, v) + g.edges[e].w;
                    if (val > cb + 1e-12) cb = val, cm = m, cx = v;
                }
            }
            if (cx >= 0 && gt(cb, T.cycle.len)) {
                auto p = back(cm, cx);
                p.push_back(p[0]);
                T.cycle = {cb, p};
            }
        }
        return T;
    }

   private:
    static constexpr double kNeg = -std::numeric_limits<double>::infinity();
    const Graph& g;
    int n;
    std::vector<double> dp;
    double& D(uint32_t m, int v) { return dp[(size_t)m * n + v]; }
    std::vector<int> all() const {
        std::vector<int> a(n);
        std::iota(a.begin(), a.end(), 0);
        return a;
    }
    void run(const std::vector<int>& starts, uint32_t allowed) {
        std::fill(dp.begin(), dp.end(), kNeg);
        for (int s : starts) D(1u << s, s) = 0.0;
        uint32_t full = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
        for (uint32_t m = 1; m <= full; ++m) {
            if (m & ~allowed) continue;
            for (int v = 0; v < n; ++v) {
                double val = D(m, v);
                if (val == kNeg) continue;
                for (auto [u, e] : g.adj[v]) {
                    uint32_t b = 1u << u;
                    if ((m & b) || !(allowed & b)) continue;
                    double& t = D(m | b, u);
                    double nv = val + g.edges[e].w;
                    if (nv > t) t = nv;
                }
            }
        }
    }
    std::vector<int> back(uint32_t m, int v) {
        std::vector<int> p{v};
        while (m & (m - 1)) {
            double cur = D(m, v);
            uint32_t pm = m ^ (1u << v);
            int found = -1;
            for (auto [u, e] : g.adj[v])
                if ((pm >> u & 1) && D(pm, u) != kNeg && std::fabs(D(pm, u) + g.edges[e].w - cur) < 1e-7) {
                    found = u;
                    break;
                }
            m = pm, v = found;
            p.push_back(v);
        }
        std::reverse(p.begin(), p.end());
        return p;
    }
};

// A block that is a single cycle (as many edges as vertices) needs no DP: every path between
// two of its vertices is one of the two arcs. Same tables as BitDP::tables, in O(n^2) for the pairs.
inline Tables cycle_tables(const Graph& g, const std::vector<int>& terms) {
    int n = g.n;
    std::vector<int> c{0}, pos(n, -1);
    pos[0] = 0;
    for (int prev = -1, u = 0; (int)c.size() < n;) {
        int nx = -1;
        for (auto [v, e] : g.adj[u])
            if (v != prev && pos[v] < 0) {
                nx = v;
                break;
            }
        prev = u, u = nx;
        pos[u] = (int)c.size();
        c.push_back(u);
    }
    std::vector<double> ew(n), pre(n + 1, 0);
    for (int i = 0; i < n; ++i) ew[i] = g.weight(c[i], c[(i + 1) % n]), pre[i + 1] = pre[i] + ew[i];
    double W = pre[n];
    auto arc = [&](int i, int j, bool fwd) {  // vertices from position i to j, stepping +1 or -1
        std::vector<int> p;
        for (int k = i;; k = fwd ? (k + 1) % n : (k + n - 1) % n) {
            p.push_back(c[k]);
            if (k == j) break;
        }
        return p;
    };
    Tables T;
    int km = 0;
    for (int i = 1; i < n; ++i)
        if (ew[i] < ew[km]) km = i;
    T.free = {W - ew[km], arc((km + 1) % n, km, true)};
    for (int a : terms) {
        int i = pos[a];
        double f1 = W - ew[(i + n - 1) % n], f2 = W - ew[i];  // forward to i-1, backward to i+1
        T.frm[a] = f1 >= f2 ? Sol{f1, arc(i, (i + n - 1) % n, true)} : Sol{f2, arc(i, (i + 1) % n, false)};
        for (int b : terms) {
            if (b == a) continue;
            int j = pos[b];
            double fw = j > i ? pre[j] - pre[i] : W - (pre[i] - pre[j]);  // forward arc i -> j
            double bw = W - fw;
            T.pr[{a, b}] = fw >= bw ? Sol{fw, arc(i, j, true)} : Sol{bw, arc(i, j, false)};
        }
    }
    T.cycle = {W, c};
    T.cycle.path.push_back(c[0]);
    return T;
}

// Tiny sparse blocks: enumerate every simple path by DFS instead of running the bitmask DP once per
// terminal and once per cycle start (each run clears 2^n * n entries). Same tables.
// Gives up (returns false) after `budget` DFS steps: dense blocks have too many paths.
inline bool enum_tables(const Graph& g, const std::vector<int>& terms, Tables& T, long long budget) {
    int n = g.n;
    std::vector<char> isterm(n, 0);
    for (int a : terms) isterm[a] = 1;
    T = Tables{};
    T.free = {0.0, {0}};
    long long steps = 0;
    bool over = false;
    T.cycle = Sol{};
    std::vector<int> stk;
    std::vector<char> used(n, 0);
    for (int s = 0; s < n; ++s) {
        // frm / pr only needed from terminals; free and cycles from every start
        Sol* fr = nullptr;
        if (isterm[s]) {
            T.frm[s] = {0.0, {s}};
            fr = &T.frm[s];
        }
        stk.assign(1, s);
        used.assign(n, 0);
        used[s] = 1;
        std::function<void(int, double)> dfs = [&](int u, double L) {
            if (over || ++steps > budget) {
                over = true;
                return;
            }
            if (L > T.free.len + 1e-12) T.free = {L, stk};
            if (fr && L > fr->len + 1e-12) *fr = {L, stk};
            if (isterm[s] && isterm[u] && u != s) {
                auto it = T.pr.find({s, u});
                if (it == T.pr.end() || L > it->second.len + 1e-12) T.pr[{s, u}] = {L, stk};
            }
            for (auto [v, e] : g.adj[u]) {
                double w = g.edges[e].w;
                if (v == s) {  // close a cycle: counted from its smallest vertex, in one direction
                    bool smallest = stk.size() >= 3 && stk[1] < u;
                    for (size_t i = 1; smallest && i < stk.size(); ++i) smallest = stk[i] > s;
                    if (smallest && gt(L + w, T.cycle.len)) {
                        auto c = stk;
                        c.push_back(s);
                        T.cycle = {L + w, c};
                    }
                    continue;
                }
                if (used[v]) continue;
                used[v] = 1;
                stk.push_back(v);
                dfs(v, L + w);
                stk.pop_back();
                used[v] = 0;
            }
        };
        dfs(s, 0.0);
        if (over) return false;
    }
    return true;
}

}  // namespace lp
