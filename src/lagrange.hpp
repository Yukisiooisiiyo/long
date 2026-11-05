// Lagrangian relaxation of "degree <= 2": max-weight forest with penalised weights.
#pragma once
#include <limits>
#include "heuristics.hpp"

namespace lp {

struct LagrResult {
    std::vector<double> lam;
    double ub = kInfL();  // path upper bound
    std::vector<int> forest;
    static double kInfL() { return std::numeric_limits<double>::infinity(); }
};

inline std::pair<double, std::vector<int>> lagr_forest(const Graph& g, const std::vector<double>& lam) {
    std::vector<std::pair<double, int>> mods;
    for (int i = 0; i < (int)g.edges.size(); ++i) {
        double m = g.edges[i].w - lam[g.edges[i].u] - lam[g.edges[i].v];
        if (m > 0) mods.push_back({m, i});
    }
    std::sort(mods.begin(), mods.end(), [](auto& a, auto& b) { return a.first > b.first; });
    DSU d(g.n);
    double val = 0;
    std::vector<int> ch;
    for (auto [m, i] : mods)
        if (d.unite(g.edges[i].u, g.edges[i].v)) val += m, ch.push_back(i);
    return {val, ch};
}

inline LagrResult subgradient(const Graph& g, double lb, int iters = 100) {
    int n = g.n;
    std::vector<double> lam(n, 0.0);
    LagrResult best;
    double theta = 2.0;
    int stall = 0;
    for (int it = 0; it <= iters; ++it) {
        auto [val, ch] = lagr_forest(g, lam);
        double ub = val;
        for (double x : lam) ub += 2 * x;
        if (ub < best.ub - 1e-12) {
            best.ub = ub, best.lam = lam, best.forest = ch;
            stall = 0;
        } else if (++stall >= 5) {
            theta /= 2;
            stall = 0;
        }
        std::vector<int> deg(n, 0);
        for (int i : ch) ++deg[g.edges[i].u], ++deg[g.edges[i].v];
        double norm = 0;
        std::vector<double> gr(n);
        for (int v = 0; v < n; ++v) {
            double gv = deg[v] - 2;
            if (!(gv > 0 || lam[v] > 0)) gv = 0;  // projected: cannot go below zero
            gr[v] = gv;
            norm += gv * gv;
        }
        if (norm == 0 || ub <= lb + 1e-9 || theta < 1e-4) break;
        double step = theta * (ub - std::max(lb, 0.0)) / norm;
        for (int v = 0; v < n; ++v) lam[v] = std::max(0.0, lam[v] + step * gr[v]);
    }
    return best;
}

// Path whose two ends a, b are fixed (degree 1 there, at most 2 elsewhere): forest relaxation with
// those capacities. Used to bound "cycle through s whose first step is s -> v": w(s, v) plus a v..s
// path that avoids the edge s-v.
inline LagrResult subgradient_ends(const Graph& g, double lb, int iters, int a, int b,
                                   const std::vector<double>* warm = nullptr) {
    int n = g.n;
    std::vector<double> lam = warm && (int)warm->size() == n ? *warm : std::vector<double>(n, 0.0);
    std::vector<int> cap(n, 2);
    cap[a] = cap[b] = 1;
    LagrResult best;
    double theta = 2.0;
    int stall = 0;
    for (int it = 0; it <= iters; ++it) {
        auto [val, ch] = lagr_forest(g, lam);
        double ub = val;
        for (int v = 0; v < n; ++v) ub += cap[v] * lam[v];
        if (ub < best.ub - 1e-12) {
            best.ub = ub, best.lam = lam, best.forest = ch;
            stall = 0;
        } else if (++stall >= 5) {
            theta /= 2;
            stall = 0;
        }
        std::vector<int> deg(n, 0);
        for (int i : ch) ++deg[g.edges[i].u], ++deg[g.edges[i].v];
        double norm = 0;
        std::vector<double> gr(n);
        for (int v = 0; v < n; ++v) {
            double gv = deg[v] - cap[v];
            if (!(gv > 0 || lam[v] > 0)) gv = 0;
            gr[v] = gv;
            norm += gv * gv;
        }
        if (norm == 0 || ub <= lb + 1e-9 || theta < 1e-4) break;
        double step = theta * (ub - std::max(lb, 0.0)) / norm;
        for (int v = 0; v < n; ++v) lam[v] = std::max(0.0, lam[v] + step * gr[v]);
    }
    return best;
}

// Multipliers tuned for cycles. Relaxation: forest plus one extra edge (1-forest), degree <= 2.
inline LagrResult subgradient_cycle(const Graph& g, double lb, int iters = 100,
                                    const std::vector<double>* warm = nullptr) {
    int n = g.n;
    std::vector<double> lam = warm && (int)warm->size() == n ? *warm : std::vector<double>(n, 0.0);
    LagrResult best;
    double theta = 2.0;
    int stall = 0;
    for (int it = 0; it <= iters; ++it) {
        auto [val, ch] = lagr_forest(g, lam);
        std::vector<char> in(g.edges.size(), 0);
        for (int i : ch) in[i] = 1;
        int ex = -1;
        double extra = 0;
        for (int i = 0; i < (int)g.edges.size(); ++i) {
            double m = g.edges[i].w - lam[g.edges[i].u] - lam[g.edges[i].v];
            if (!in[i] && m > extra) extra = m, ex = i;
        }
        double ub = val + extra;
        for (double x : lam) ub += 2 * x;
        if (ub < best.ub - 1e-12) {
            best.ub = ub, best.lam = lam, best.forest = ch;
            stall = 0;
        } else if (++stall >= 5) {
            theta /= 2;
            stall = 0;
        }
        std::vector<int> deg(n, 0);
        for (int i : ch) ++deg[g.edges[i].u], ++deg[g.edges[i].v];
        if (ex >= 0) ++deg[g.edges[ex].u], ++deg[g.edges[ex].v];
        double norm = 0;
        std::vector<double> gr(n);
        for (int v = 0; v < n; ++v) {
            double gv = deg[v] - 2;
            if (!(gv > 0 || lam[v] > 0)) gv = 0;
            gr[v] = gv;
            norm += gv * gv;
        }
        if (norm == 0 || ub <= lb + 1e-9 || theta < 1e-4) break;
        double step = theta * (ub - std::max(lb, 0.0)) / norm;
        for (int v = 0; v < n; ++v) lam[v] = std::max(0.0, lam[v] + step * gr[v]);
    }
    return best;
}

// cycle bound: forest plus one extra edge
inline double cycle_bound(const Graph& g, const std::vector<double>& lam) {
    auto [val, ch] = lagr_forest(g, lam);
    std::vector<char> in(g.edges.size(), 0);
    for (int i : ch) in[i] = 1;
    double extra = 0, s = 0;
    for (int i = 0; i < (int)g.edges.size(); ++i)
        if (!in[i]) extra = std::max(extra, g.edges[i].w - lam[g.edges[i].u] - lam[g.edges[i].v]);
    for (double x : lam) s += 2 * x;
    return val + extra + s;
}

// paths of the repaired forest (degree <= 2), longest first
inline std::vector<std::vector<int>> forest_paths(const Graph& g, const std::vector<int>& forest) {
    std::vector<std::vector<int>> inc(g.n);
    for (int i : forest) inc[g.edges[i].u].push_back(i), inc[g.edges[i].v].push_back(i);
    std::vector<char> keep(g.edges.size(), 0);
    for (int i : forest) keep[i] = 1;
    for (int u = 0; u < g.n; ++u) {
        std::vector<int> es;
        for (int e : inc[u])
            if (keep[e]) es.push_back(e);
        if (es.size() > 2) {
            std::sort(es.begin(), es.end(), [&](int a, int b) { return g.edges[a].w > g.edges[b].w; });
            for (size_t k = 2; k < es.size(); ++k) keep[es[k]] = 0;
        }
    }
    std::vector<std::vector<std::pair<int, int>>> nb(g.n);
    for (int i = 0; i < (int)g.edges.size(); ++i)
        if (keep[i]) nb[g.edges[i].u].push_back({g.edges[i].v, i}), nb[g.edges[i].v].push_back({g.edges[i].u, i});
    std::vector<char> seen(g.n, 0);
    std::vector<std::pair<double, std::vector<int>>> out;
    for (int s = 0; s < g.n; ++s) {
        if (seen[s] || nb[s].size() != 1) continue;
        std::vector<int> p{s};
        seen[s] = 1;
        int prev_e = -1, cur = s;
        double L = 0;
        while (true) {
            int nx = -1, ne = -1;
            for (auto [v, e] : nb[cur])
                if (e != prev_e) nx = v, ne = e;
            if (nx < 0) break;
            p.push_back(nx);
            seen[nx] = 1;
            L += g.edges[ne].w;
            prev_e = ne;
            cur = nx;
        }
        out.push_back({L, p});
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<std::vector<int>> res;
    for (auto& [L, p] : out) res.push_back(p);
    return res;
}

// Edges that cannot lie on any path/cycle longer than `incumbent` (1-forest criterion for cycles).
inline std::vector<char> dead_edges(const Graph& g, const LagrResult& lr, double incumbent) {
    int n = g.n;
    auto mod = [&](int i) { return g.edges[i].w - lr.lam[g.edges[i].u] - lr.lam[g.edges[i].v]; };
    double maxmod = 0;
    for (int i = 0; i < (int)g.edges.size(); ++i) maxmod = std::max(maxmod, mod(i));
    std::vector<std::vector<std::pair<int, int>>> tadj(n);
    std::vector<char> in(g.edges.size(), 0), dead(g.edges.size(), 0);
    for (int i : lr.forest) {
        in[i] = 1;
        tadj[g.edges[i].u].push_back({g.edges[i].v, i});
        tadj[g.edges[i].v].push_back({g.edges[i].u, i});
    }
    std::vector<int> comp(n, -1), par(n, -1), pe(n, -1), dep(n, 0);
    for (int r = 0; r < n; ++r) {
        if (comp[r] != -1) continue;
        comp[r] = r;
        std::vector<int> st{r};
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            for (auto [v, i] : tadj[u])
                if (comp[v] == -1) comp[v] = r, par[v] = u, pe[v] = i, dep[v] = dep[u] + 1, st.push_back(v);
        }
    }
    for (int i = 0; i < (int)g.edges.size(); ++i) {
        if (in[i]) continue;
        int a = g.edges[i].u, b = g.edges[i].v;
        double be;
        if (comp[a] == comp[b]) {
            double mn = std::numeric_limits<double>::infinity();
            while (a != b) {
                if (dep[a] >= dep[b]) mn = std::min(mn, mod(pe[a])), a = par[a];
                else mn = std::min(mn, mod(pe[b])), b = par[b];
            }
            be = lr.ub + mod(i) - mn;
        } else {
            be = lr.ub + std::min(mod(i), 0.0);
        }
        // a path using e is <= be; a cycle using e is a path plus one edge: <= be + maxmod
        if (be + maxmod < incumbent - kEps * (1 + std::fabs(incumbent))) dead[i] = 1;
    }
    return dead;
}

}  // namespace lp
