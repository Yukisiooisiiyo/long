// Graph basics: input parsing, connected components, biconnected components.
#pragma once
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lp {

constexpr double kEps = 1e-9;
// Exact mode: when every distance in the input is a plain decimal with at most 6 digits after the
// point, all distances are scaled to integers (e.g. 8.54 -> 854). Sums of integers below 2^53 are
// exact in double, and any answer that beats another does so by at least 1, so "better" is
// "larger by more than 0.5". Otherwise a relative tolerance absorbs rounding.
inline bool g_exact = false;
inline bool gt(double a, double b) { return g_exact ? a > b + 0.5 : a > b + kEps * (1.0 + std::fabs(b)); }

struct Edge {
    int u, v;
    double w;
};

struct Graph {
    int n = 0;
    std::vector<Edge> edges;
    std::vector<std::vector<std::pair<int, int>>> adj;  // (neighbor, edge id)

    Graph() = default;
    Graph(int n_, std::vector<Edge> es) : n(n_), edges(std::move(es)) { build(); }

    void build() {
        adj.assign(n, {});
        for (int i = 0; i < (int)edges.size(); ++i) {
            adj[edges[i].u].push_back({edges[i].v, i});
            adj[edges[i].v].push_back({edges[i].u, i});
        }
    }
    // weight of edge u-v, or NaN when absent (degrees are small, so a scan is cheap)
    double weight(int u, int v) const {
        for (auto [x, e] : adj[u])
            if (x == v) return edges[e].w;
        return std::nan("");
    }
    bool has(int u, int v) const { return !std::isnan(weight(u, v)); }

    // induced subgraph on `verts` (sorted); returns local graph, local->global in verts
    Graph induced(const std::vector<int>& verts) const {
        std::unordered_map<int, int> idx;
        idx.reserve(verts.size() * 2);
        for (int i = 0; i < (int)verts.size(); ++i) idx[verts[i]] = i;
        std::vector<Edge> es;
        for (int a : verts)
            for (auto [b, e] : adj[a])
                if (a < b) {
                    auto it = idx.find(b);
                    if (it != idx.end()) es.push_back({idx[a], it->second, edges[e].w});
                }
        return Graph((int)verts.size(), std::move(es));
    }
};

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Returns original IDs (index -> ID) and the graph. Input: one route per line, "start ID, end ID, distance". Spaces / tabs around fields, CRLF or LF,
// blank lines and a missing final newline are accepted. One pass over the buffer, no substrings.
// Numbers are read with strtoll / strtod and must fill their trimmed field. Self loops are ignored;
// for a duplicated route the longest distance is kept. Errors name the line.
// scale (out): 10^d when every distance is a plain decimal with at most d <= 6 decimals and the
// scaled total stays below 2^50; the returned weights are then the scaled integers. 0 otherwise.
inline std::pair<std::vector<long long>, Graph> parse(const std::string& text, double* scale = nullptr,
                                                      bool allow_exact = true) {
    std::vector<long long> ids;
    bool exact = allow_exact && scale;
    int maxdec = 0;
    std::vector<std::pair<long long, int>> digits;  // per raw edge: (digits as integer, decimals)
    std::unordered_map<long long, int> index;
    index.reserve(1024);
    struct E3 {
        int u, v;
        double w;
    };
    std::vector<E3> raw;
    std::pair<long long, int> cur_digits{0, 0};
    const char* base = text.c_str();
    size_t n = text.size(), pos = 0;
    int lineno = 0;
    auto isws = [](char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
    while (pos <= n) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = n;
        size_t a0 = pos, a1 = nl;
        while (a0 < a1 && isws(base[a0])) ++a0;
        while (a1 > a0 && isws(base[a1 - 1])) --a1;
        pos = nl + 1;
        ++lineno;
        if (a0 == a1) {
            if (nl >= n) break;
            continue;
        }
        auto fail = [&](const std::string& m) {
            throw InputError("line " + std::to_string(lineno) + ": " + m);
        };
        size_t fb[3], fe[3];
        int nf = 0;
        for (size_t s0 = a0;;) {
            size_t c = s0;
            while (c < a1 && base[c] != ',') ++c;
            if (nf < 3) {
                size_t x = s0, y = c;
                while (x < y && isws(base[x])) ++x;
                while (y > x && isws(base[y - 1])) --y;
                fb[nf] = x, fe[nf] = y;
            }
            ++nf;
            if (c >= a1) break;
            s0 = c + 1;
        }
        if (nf != 3) fail("expected 3 comma-separated fields");
        long long a, b;
        double w;
        {
            char* end;
            errno = 0;
            a = std::strtoll(base + fb[0], &end, 10);
            if (fb[0] == fe[0] || end != base + fe[0] || errno == ERANGE) fail("not a number");
            errno = 0;
            b = std::strtoll(base + fb[1], &end, 10);
            if (fb[1] == fe[1] || end != base + fe[1] || errno == ERANGE) fail("not a number");
            errno = 0;
            w = std::strtod(base + fb[2], &end);  // the program never changes LC_NUMERIC
            if (fb[2] == fe[2] || end != base + fe[2] || (errno == ERANGE && std::isinf(w))) fail("not a number");
            if (exact) {  // plain decimal? read it digit by digit, exactly
                long long iv = 0;
                int dec = -1, nd = 0;
                size_t k = fb[2];
                if (base[k] == '+') ++k;
                for (; k < fe[2] && exact; ++k) {
                    char ch = base[k];
                    if (ch == '.' && dec < 0) dec = 0;
                    else if (ch >= '0' && ch <= '9') {
                        if (++nd > 15) exact = false;
                        iv = iv * 10 + (ch - '0');
                        if (dec >= 0) ++dec;
                    } else
                        exact = false;
                }
                if (nd == 0 || dec > 6) exact = false;
                if (exact) maxdec = std::max(maxdec, std::max(dec, 0)), cur_digits = {iv, std::max(dec, 0)};
            }
        }
        if (a <= 0 || b <= 0) fail("IDs must be positive integers");
        if (!(w >= 0) || std::isinf(w)) fail("distance must be a non-negative number");
        if (a != b) {
            int uv[2];
            int k = 0;
            for (long long x : {a, b}) {
                auto it = index.find(x);
                if (it == index.end()) {
                    it = index.emplace(x, (int)ids.size()).first;
                    ids.push_back(x);
                }
                uv[k++] = it->second;
            }
            raw.push_back({std::min(uv[0], uv[1]), std::max(uv[0], uv[1]), w});
            if (exact) digits.push_back(cur_digits);
        }
        if (nl >= n) break;
    }
    if (exact) {  // scale to integers if the total stays well inside the exact range of double
        double p10 = 1;
        for (int i = 0; i < maxdec; ++i) p10 *= 10;
        double total = 0;
        for (size_t i = 0; i < raw.size(); ++i) {
            double f = 1;
            for (int k = digits[i].second; k < maxdec; ++k) f *= 10;
            raw[i].w = (double)digits[i].first * f;
            total += raw[i].w;
        }
        if (total >= 1125899906842624.0) {  // 2^50: fall back to the decimal values
            exact = false;
            for (auto& e : raw) e.w /= p10;
        } else if (scale)
            *scale = p10;
    }
    if (!exact && scale) *scale = 0;
    std::sort(raw.begin(), raw.end(), [](const E3& x, const E3& y) { return x.u != y.u ? x.u < y.u : x.v < y.v; });
    std::vector<Edge> es;
    for (size_t i = 0; i < raw.size();) {
        size_t j = i;
        double w = raw[i].w;
        while (j < raw.size() && raw[j].u == raw[i].u && raw[j].v == raw[i].v) w = std::max(w, raw[j].w), ++j;
        es.push_back({raw[i].u, raw[i].v, w});
        i = j;
    }
    return {ids, Graph((int)ids.size(), std::move(es))};
}

inline std::vector<std::vector<int>> components(const Graph& g) {
    std::vector<int> comp(g.n, -1);
    std::vector<std::vector<int>> out;
    for (int s = 0; s < g.n; ++s) {
        if (comp[s] != -1) continue;
        std::vector<int> st{s}, cur;
        comp[s] = (int)out.size();
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            cur.push_back(u);
            for (auto [v, e] : g.adj[u])
                if (comp[v] == -1) {
                    comp[v] = (int)out.size();
                    st.push_back(v);
                }
        }
        std::sort(cur.begin(), cur.end());
        out.push_back(cur);
    }
    return out;
}

struct Blocks {
    std::vector<std::vector<int>> verts;  // sorted vertex lists
    std::vector<char> is_art;
};

// Iterative Tarjan for biconnected components (blocks) of a connected graph.
inline Blocks biconnected(const Graph& g) {
    int n = g.n;
    Blocks res;
    res.is_art.assign(n, 0);
    std::vector<int> disc(n, -1), low(n, 0);
    std::vector<int> estack;
    int t = 0;
    struct Fr {
        int u, pe;
        size_t it;
    };
    for (int r = 0; r < n; ++r) {
        if (disc[r] != -1) continue;
        disc[r] = low[r] = t++;
        if (g.adj[r].empty()) {
            res.verts.push_back({r});
            continue;
        }
        int root_children = 0;
        std::vector<Fr> st{{r, -1, 0}};
        while (!st.empty()) {
            Fr& f = st.back();
            int u = f.u;
            bool adv = false;
            while (f.it < g.adj[u].size()) {
                auto [v, e] = g.adj[u][f.it++];
                if (e == f.pe) continue;
                if (disc[v] == -1) {
                    estack.push_back(e);
                    disc[v] = low[v] = t++;
                    st.push_back({v, e, 0});
                    adv = true;
                    break;
                } else if (disc[v] < disc[u]) {
                    estack.push_back(e);
                    low[u] = std::min(low[u], disc[v]);
                }
            }
            if (adv) continue;
            int pe = f.pe;
            st.pop_back();
            if (!st.empty()) {
                int p = st.back().u;
                low[p] = std::min(low[p], low[u]);
                if (low[u] >= disc[p]) {
                    std::set<int> vs;
                    while (true) {
                        int e = estack.back();
                        estack.pop_back();
                        vs.insert(g.edges[e].u);
                        vs.insert(g.edges[e].v);
                        if (e == pe) break;
                    }
                    res.verts.emplace_back(vs.begin(), vs.end());
                    if (st.size() > 1) res.is_art[p] = 1;
                    else ++root_children;
                }
            }
        }
        if (root_children >= 2) res.is_art[r] = 1;
    }
    return res;
}

// Length of a path (or a cycle when first == last). Throws when invalid.
inline double verify(const Graph& g, const std::vector<int>& p) {
    if (p.empty()) return 0.0;
    bool cyc = p.size() > 2 && p.front() == p.back();
    std::vector<int> body(p.begin(), cyc ? p.end() - 1 : p.end());
    std::vector<int> s = body;
    std::sort(s.begin(), s.end());
    if (std::adjacent_find(s.begin(), s.end()) != s.end()) throw std::runtime_error("repeated vertex");
    if (cyc && body.size() < 3) throw std::runtime_error("cycle too short");
    double L = 0;
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        double w = g.weight(p[i], p[i + 1]);
        if (std::isnan(w)) throw std::runtime_error("missing edge");
        L += w;
    }
    return L;
}

}  // namespace lp
