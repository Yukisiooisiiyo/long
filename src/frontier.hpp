// Frontier DP (path-decomposition DP) for thin blocks.
//
// Edges are processed one by one in an order where few vertices are "open" at a time (the frontier:
// vertices seen already that still have unprocessed edges). A state records, for each frontier
// vertex, how the partial solution touches it: unused, interior (degree 2), or an end of a partial
// path, together with the partner end of that partial path. The number of states depends on the
// frontier width, not on the size of the graph, so long thin graphs (strip grids, railway-like
// blocks with treewidth around 4) are solved in time roughly proportional to their length, where
// branch and bound blows up. The DP is exact. When the frontier is too wide or the state count
// exceeds a budget it gives up and the caller falls back to branch and bound.
#pragma once
#include <algorithm>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <type_traits>
#include <vector>

#include "graph.hpp"
#include "heuristics.hpp"
#include "search.hpp"

namespace lp {

struct FrontierOrder {
    std::vector<int> edges;  // processing order (edge ids)
    int width = 0;           // largest frontier
    long long cost = 0;      // sum of frontier sizes (tie-break)
};

// frontier width of a vertex order (edges sorted by their later endpoint)
inline FrontierOrder frontier_from_vertex_order(const Graph& g, const std::vector<int>& vord) {
    std::vector<int> pos(g.n);
    for (int i = 0; i < g.n; ++i) pos[vord[i]] = i;
    FrontierOrder o;
    o.edges.resize(g.edges.size());
    std::iota(o.edges.begin(), o.edges.end(), 0);
    std::stable_sort(o.edges.begin(), o.edges.end(), [&](int a, int b) {
        const Edge &x = g.edges[a], &y = g.edges[b];
        int xa = std::max(pos[x.u], pos[x.v]), ya = std::max(pos[y.u], pos[y.v]);
        if (xa != ya) return xa < ya;
        return std::min(pos[x.u], pos[x.v]) < std::min(pos[y.u], pos[y.v]);
    });
    int m = (int)o.edges.size();
    std::vector<int> first(g.n, -1), last(g.n, -1);
    for (int i = 0; i < m; ++i)
        for (int x : {g.edges[o.edges[i]].u, g.edges[o.edges[i]].v}) {
            if (first[x] < 0) first[x] = i;
            last[x] = i;
        }
    std::vector<int> diff(m + 1, 0);
    for (int v = 0; v < g.n; ++v)
        if (first[v] >= 0) ++diff[first[v]], --diff[last[v] + 1];  // present while first <= i <= last
    for (int i = 0, cur = 0; i < m; ++i) {
        cur += diff[i];
        o.width = std::max(o.width, cur);
        o.cost += cur;
    }
    return o;
}

// try a few vertex orders (BFS and greedy min-frontier from peripheral vertices), keep the thinnest
inline FrontierOrder best_frontier_order(const Graph& g) {
    auto bfs = [&](int s, std::vector<int>* dist) {
        std::vector<int> d(g.n, -1), ord;
        std::queue<int> q;
        q.push(s);
        d[s] = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            ord.push_back(u);
            for (auto [v, e] : g.adj[u])
                if (d[v] < 0) d[v] = d[u] + 1, q.push(v);
        }
        if (dist) *dist = d;
        return ord;
    };
    std::vector<int> starts;
    {
        std::vector<int> d;
        auto o1 = bfs(0, &d);
        int a = o1.back();
        auto o2 = bfs(a, &d);
        int b = o2.back();
        starts = {a, b};
        // a few more spread-out starts
        for (int k = 0; k < 2; ++k) {
            std::vector<int> dd;
            bfs(starts.back(), &dd);
            int far = 0;
            for (int v = 0; v < g.n; ++v)
                if (dd[v] > dd[far]) far = v;
            if (std::find(starts.begin(), starts.end(), far) == starts.end()) starts.push_back(far);
        }
    }
    FrontierOrder best;
    best.width = 1 << 30;
    auto consider = [&](const std::vector<int>& vord) {
        if ((int)vord.size() != g.n) return;
        FrontierOrder o = frontier_from_vertex_order(g, vord);
        if (o.width < best.width || (o.width == best.width && o.cost < best.cost)) best = std::move(o);
    };
    for (int s : starts) {
        consider(bfs(s, nullptr));
        if (g.n > 3000) continue;  // the greedy order is quadratic
        // greedy: next vertex = the candidate that closes the most open vertices / opens the fewest
        std::vector<char> in(g.n, 0);
        std::vector<int> rem(g.n);  // neighbours not yet placed
        for (int v = 0; v < g.n; ++v) rem[v] = (int)g.adj[v].size();
        std::vector<int> ord{s};
        in[s] = 1;
        for (auto [v, e] : g.adj[s]) --rem[v];
        std::vector<char> cand(g.n, 0);
        for (auto [v, e] : g.adj[s]) cand[v] = 1;
        std::vector<int> bfsd;
        bfs(s, &bfsd);
        while ((int)ord.size() < g.n) {
            int pick = -1;
            long long bestscore = 0;
            for (int v = 0; v < g.n; ++v) {
                if (!cand[v] || in[v]) continue;
                // placing v: v opens if it still has unplaced neighbours; placed neighbours whose
                // last unplaced neighbour is v close
                int closes = 0;
                for (auto [x, e] : g.adj[v])
                    if (in[x] && rem[x] == 1) ++closes;
                int opens = rem[v] > 0 ? 1 : 0;
                // new neighbours that become candidates do not open until placed
                long long score = (long long)(closes - opens) * 1000000 - bfsd[v] * 1000 - (long long)rem[v];
                if (pick < 0 || score > bestscore) pick = v, bestscore = score;
            }
            if (pick < 0) break;
            in[pick] = 1;
            ord.push_back(pick);
            for (auto [x, e] : g.adj[pick]) {
                --rem[x];
                if (!in[x]) cand[x] = 1;
            }
        }
        consider(ord);
    }
    return best;
}

// KW = words per state key: 2 words hold up to 24 frontier slots; 1 word holds up to 12 (5 bits per
// slot + 3 flag bits), which makes states smaller and hashing / comparing / unpacking cheaper.
template <int KW>
class FrontierDPT {
   public:
    static constexpr int kMaxWidth = KW == 1 ? 12 : 24;

    struct Result {
        bool ok = false;  // false: gave up (too wide / too many states / no answer)
        Sol sol;          // path (open) or cycle (closed: front == back)
        long long states = 0;
    };

    // mode 0: best path, value = length + bonus at both ends (ends distinct, at least one edge).
    //         If forced >= 0 that vertex must be one end (it gets no bonus unless bonus says so).
    // mode 1: best cycle.
    FrontierDPT(const Graph& g, const FrontierOrder& ord, Control& ctl, long long budget, long long layer_cap)
        : g(g), ord(ord), ctl(ctl), budget(budget), layer_cap(layer_cap) {}

    // lb >= 0: only answers longer than lb matter (a solution of length lb is known to the caller).
    // States whose upper bound does not beat it are dropped. If nothing beats lb, ok is true and
    // sol is empty.
    Result run(int mode, const std::vector<double>* bonus, int forced, double lb = -1,
               const std::vector<double>* lam = nullptr, long long beam = 0) {
        Result R;
        if (ord.width > kMaxWidth || g.edges.empty()) return R;
        mode_ = mode;
        bonus_ = bonus;
        forced_ = forced;
        int m = (int)ord.edges.size();
        std::vector<int> first(g.n, -1), last(g.n, -1);
        for (int i = 0; i < m; ++i)
            for (int x : {g.edges[ord.edges[i]].u, g.edges[ord.edges[i]].v}) {
                if (first[x] < 0) first[x] = i;
                last[x] = i;
            }
        // frontier slots: vertices in order of entry
        std::vector<int> F;
        std::vector<int> slot(g.n, -1);
        std::vector<State> cur{State{Key{}, 0.0}};
        back.clear();
        back.reserve(m);
        best_val = lb;
        best_layer = -1;
        prune_ = lb >= 0 || beam > 0;  // compute the bound data
        // upper-bound data: each unprocessed edge is worth half its weight at each end, and a vertex
        // takes at most its two heaviest unprocessed edges (one if it is already a path end)
        std::vector<std::vector<double>> st1(g.n), st2(g.n);  // suffix top1 / top2 per vertex
        std::vector<int> ptr(g.n, 0);
        std::vector<double> future(m + 1, 0.0);
        if (prune_) {
            std::vector<std::vector<double>> ws(g.n);
            for (int i = 0; i < m; ++i) {
                const Edge& E = g.edges[ord.edges[i]];
                ws[E.u].push_back(E.w), ws[E.v].push_back(E.w);
            }
            for (int v = 0; v < g.n; ++v) {
                int k = (int)ws[v].size();
                st1[v].assign(k + 1, 0), st2[v].assign(k + 1, 0);
                for (int j = k - 1; j >= 0; --j) {
                    double a = st1[v][j + 1], b = st2[v][j + 1], w = ws[v][j];
                    if (w > a) b = a, a = w;
                    else if (w > b) b = w;
                    st1[v][j] = a, st2[v][j] = b;
                }
            }
            // future[i] = sum over vertices first seen after step i
            std::vector<double> byfirst(m + 1, 0.0);
            for (int v = 0; v < g.n; ++v)
                if (first[v] >= 0) byfirst[first[v]] += (st1[v][0] + st2[v][0]) / 2;
            future.assign(m + 1, 0.0);
            for (int i = m - 1; i >= 0; --i) future[i] = future[i + 1] + (i + 1 < m ? byfirst[i + 1] : 0.0);
            // Lagrangian bound for the unprocessed part: max forest of the positive reduced weights
            // w - lam_u - lam_v over edges after step i, plus lam_v times the capacity each vertex
            // has left (2 unused or unseen, 1 path end, 0 interior). In cycle mode one extra edge.
            use_lam_ = lam && (int)lam->size() == g.n;
            if (use_lam_) {
                lam_ = *lam;
                lforest_.assign(m + 1, 0.0);
                lfut_.assign(m + 1, 0.0);
                std::vector<std::pair<double, int>> red;
                for (int i = 0; i < m; ++i) {
                    const Edge& E = g.edges[ord.edges[i]];
                    double r = E.w - lam_[E.u] - lam_[E.v];
                    if (r > 0) red.push_back({r, i});
                }
                std::stable_sort(red.rbegin(), red.rend());
                DSU d(g.n);
                for (int i = 0; i < m; ++i) {  // forest over edges with index > i
                    std::iota(d.p.begin(), d.p.end(), 0);
                    double val = 0, extra = 0;
                    for (auto [r, j] : red) {
                        if (j <= i) continue;
                        const Edge& E = g.edges[ord.edges[j]];
                        if (d.unite(E.u, E.v)) val += r;
                        else extra = std::max(extra, r);
                    }
                    lforest_[i] = val + (mode_ == 1 ? extra : 0.0);
                }
                std::vector<double> lam_first(m + 1, 0.0);
                for (int v = 0; v < g.n; ++v)
                    if (first[v] >= 0) lam_first[first[v]] += 2 * lam_[v];
                for (int i = m - 1; i >= 0; --i) lfut_[i] = lfut_[i + 1] + (i + 1 < m ? lam_first[i + 1] : 0.0);
            }
            maxb_ = 0;
            if (mode_ == 0 && bonus_)
                for (double b : *bonus_) maxb_ = std::max(maxb_, b);
        }
        long long stored = 0;
        for (int i = 0; i < m; ++i) {
            ctl.check();
            const Edge& E = g.edges[ord.edges[i]];
            if (prune_) ++ptr[E.u], ++ptr[E.v];
            for (int x : {E.u, E.v})
                if (first[x] == i && slot[x] < 0) slot[x] = (int)F.size(), F.push_back(x);
            int a = slot[E.u], b = slot[E.v];
            std::vector<int> leave;
            for (int x : {E.u, E.v})
                if (last[x] == i && std::find(leave.begin(), leave.end(), slot[x]) == leave.end())
                    leave.push_back(slot[x]);
            std::stable_sort(leave.rbegin(), leave.rend());
            int W = (int)F.size();
            // new slot numbering after the leaving vertices are removed
            std::vector<int> remap(W);
            for (int s = 0, k = 0; s < W; ++s)
                remap[s] = std::find(leave.begin(), leave.end(), s) != leave.end() ? -1 : k++;
            if (prune_) {  // potentials of the frontier after this step, by new slot
                pot0_.clear(), pot1_.clear(), lpot_.clear();
                for (int s2 = 0; s2 < W; ++s2)
                    if (remap[s2] >= 0) {
                        int x = F[s2];
                        pot0_.push_back((st1[x][ptr[x]] + st2[x][ptr[x]]) / 2);
                        pot1_.push_back(st1[x][ptr[x]] / 2);
                        lpot_.push_back(use_lam_ ? lam_[x] : 0.0);
                    }
                fut_ = future[i];
                if (use_lam_) lbase_ = lforest_[i] + lfut_[i];
            }
            table.clear((size_t)cur.size() * 2 + 16);
            nxt.clear();
            back.emplace_back();
            auto& bk = back.back();
            for (int si = 0; si < (int)cur.size(); ++si) {
                if ((si & 32767) == 32767) ctl.check();
                const State& S = cur[si];
                uint8_t c[kMaxWidth];
                int t, f;
                decode(S.k, W, c, t, f);
                // skip the edge
                finish(c, W, t, f, S.v, leave, remap, F, si, false, i, bk);
                // take the edge
                uint8_t ca = c[a], cb = c[b];
                if (ca == 1 || cb == 1) continue;
                double v2 = S.v + E.w;
                uint8_t d[kMaxWidth];
                std::memcpy(d, c, W);
                if (ca == 0 && cb == 0) {
                    d[a] = (uint8_t)(3 + b), d[b] = (uint8_t)(3 + a);
                } else if (ca == 0 || cb == 0) {
                    int z = ca == 0 ? a : b, o = ca == 0 ? b : a;  // z: unused, o: an end
                    uint8_t co = d[o];
                    d[z] = co;  // z inherits o's partner (or the terminal mark)
                    if (co >= 3) d[co - 3] = (uint8_t)(3 + z);
                    d[o] = 1;
                } else if (ca == 2 && cb == 2) {  // two parts whose far ends are final: path done
                    if (mode_ != 0) continue;
                    d[a] = d[b] = 1;
                    if (forced_ >= 0 && !f) continue;
                    if (only_closed(d, W)) record(v2, i, si, true);
                    continue;
                } else if (ca >= 3 && cb >= 3 && ca - 3 == b) {  // closes a cycle
                    if (mode_ != 1) continue;
                    d[a] = d[b] = 1;
                    if (only_closed(d, W)) record(v2, i, si, true);
                    continue;
                } else {  // join two different parts
                    if (ca == 2 || cb == 2) {
                        int o = ca == 2 ? b : a;  // the end whose far side is still open
                        uint8_t co = d[o];
                        d[co - 3] = 2;
                    } else {
                        d[ca - 3] = cb, d[cb - 3] = ca;
                    }
                    d[a] = d[b] = 1;
                }
                finish(d, W, t, f, v2, leave, remap, F, si, true, i, bk);
                if ((long long)nxt.size() > layer_cap) return R;
            }
            if (rank_ > 0 && (long long)nxt.size() > rank_) rank_reduce(W - (int)leave.size(), bk);
            if (beam > 0 && (long long)nxt.size() > beam) {  // keep the most promising states only
                std::vector<int> id(nxt.size());
                std::iota(id.begin(), id.end(), 0);
                // total order (ties by index) and a sorted result: the same states on every platform
                std::nth_element(id.begin(), id.begin() + beam, id.end(), [&](int x, int y) {
                    return nxt[x].pr != nxt[y].pr ? nxt[x].pr > nxt[y].pr : x < y;
                });
                id.resize(beam);
                std::sort(id.begin(), id.end());
                std::vector<State> n2;
                std::vector<uint32_t> b2;
                n2.reserve(beam), b2.reserve(beam);
                for (int j : id) n2.push_back(nxt[j]), b2.push_back(bk[j]);
                nxt.swap(n2);
                bk.swap(b2);
            }
            stored += (long long)nxt.size();
            R.states = std::max<long long>(R.states, (long long)nxt.size());
            if (stored > budget) return R;  // too many states: give up
            for (int s : leave) slot[F[s]] = -1;
            std::vector<int> F2;
            for (int s = 0; s < W; ++s)
                if (remap[s] >= 0) F2.push_back(F[s]);
            F.swap(F2);
            for (int s = 0; s < (int)F.size(); ++s) slot[F[s]] = s;
            cur.swap(nxt);
        }
        if (best_layer < 0) {
            R.ok = lb >= 0;  // nothing beats lb
            return R;
        }
        // rebuild the edge set
        std::vector<int> taken;
        {
            int i = best_layer, idx = best_src;
            if (best_took) taken.push_back(ord.edges[i]);
            for (int j = i - 1; j >= 0; --j) {
                uint32_t e = back[j][idx];
                if (e >> 31) taken.push_back(ord.edges[j]);
                idx = (int)(e & 0x7fffffffu);
            }
        }
        R.sol = walk(taken);
        R.sol.len = best_val;
        R.ok = !R.sol.path.empty();
        return R;
    }

    long long rank_ = 0;  // rank-based reduction when a step has more states than this (0 = off)
    int rank_max_points = 16;  // groups whose matchings span more points are left alone (vectors too long)

   private:
    struct Key1 {
        uint64_t lo = 0;
        bool operator==(const Key1& o) const { return lo == o.lo; }
    };
    struct Key2 {
        uint64_t lo = 0, hi = 0;
        bool operator==(const Key2& o) const { return lo == o.lo && hi == o.hi; }
    };
    using Key = std::conditional_t<KW == 1, Key1, Key2>;
    struct State {
        Key k;
        double v;
        double pr = 0;  // upper bound of the best completion (beam ranking)
    };
    // open-addressing table: key -> index in nxt
    struct Table {
        std::vector<int> idx;
        size_t mask = 0;
        void clear(size_t want) {
            size_t cap = 16;
            while (cap < want * 2) cap <<= 1;
            if (idx.size() < cap) idx.assign(cap, -1);
            else std::fill(idx.begin(), idx.end(), -1);
            mask = idx.size() - 1;
        }
    };

    const Graph& g;
    const FrontierOrder& ord;
    Control& ctl;
    long long budget, layer_cap;
    int mode_ = 0, forced_ = -1;
    const std::vector<double>* bonus_ = nullptr;
    std::vector<State> nxt;
    std::vector<std::vector<uint32_t>> back;  // per layer: parent index | took << 31
    Table table;
    double best_val = -1;
    bool prune_ = false;
    double maxb_ = 0, fut_ = 0;
    std::vector<double> pot0_, pot1_, lpot_, lam_, lforest_, lfut_;
    bool use_lam_ = false;
    double lbase_ = 0;
    int best_layer = -1, best_src = -1;
    bool best_took = false;

    static void decode(const Key& k, int W, uint8_t* c, int& t, int& f) {
        if constexpr (KW == 1) {
            uint64_t x = k.lo;
            for (int s = 0; s < W; ++s, x >>= 5) c[s] = (uint8_t)(x & 31);
            t = (int)((k.lo >> 60) & 3);
            f = (int)((k.lo >> 62) & 1);
            return;
        } else {
        for (int s = 0; s < W; ++s) {
            int bit = s * 5;
            uint64_t x = bit < 64 ? (k.lo >> bit) | (bit > 59 ? (k.hi << (64 - bit)) : 0) : (k.hi >> (bit - 64));
            c[s] = (uint8_t)(x & 31);
        }
        t = (int)((k.hi >> 60) & 3);
        f = (int)((k.hi >> 62) & 1);
        }
    }
    static Key encode(const uint8_t* c, int W, int t, int f) {
        Key k;
        if constexpr (KW == 1) {
            uint64_t x = 0;
            for (int s = W; s-- > 0;) x = (x << 5) | c[s];
            k.lo = x | (uint64_t)t << 60 | (uint64_t)f << 62;
            return k;
        } else {
        for (int s = 0; s < W; ++s) {
            int bit = s * 5;
            uint64_t x = c[s];
            if (bit < 64) {
                k.lo |= x << bit;
                if (bit > 59) k.hi |= x >> (64 - bit);
            } else
                k.hi |= x << (bit - 64);
        }
        k.hi |= (uint64_t)t << 60 | (uint64_t)f << 62;
        return k;
        }
    }
    static size_t hash(const Key& k) {
        if constexpr (KW == 1) {
            uint64_t h = k.lo * 0x9E3779B97F4A7C15ull;
            return (size_t)(h ^ (h >> 29));
        } else {
            uint64_t h = k.lo * 0x9E3779B97F4A7C15ull ^ (k.hi + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full;
            return (size_t)(h ^ (h >> 29));
        }
    }
    static bool only_closed(const uint8_t* d, int W) {  // no other open part anywhere on the frontier
        for (int s = 0; s < W; ++s)
            if (d[s] >= 2) return false;
        return true;
    }
    void record(double v, int layer, int src, bool took) {
        if (best_layer < 0 ? (best_val < 0 || gt(v, best_val)) : gt(v, best_val)) best_val = v, best_layer = layer, best_src = src, best_took = took;
    }

    // remove the vertices whose last edge was this one, then store the state
    void finish(const uint8_t* c0, int W, int t, int f, double v, const std::vector<int>& leave,
                const std::vector<int>& remap, const std::vector<int>& F, int src, bool took, int layer,
                std::vector<uint32_t>& bk) {
        uint8_t c[kMaxWidth];
        std::memcpy(c, c0, W);
        for (int s : leave) {
            uint8_t x = c[s];
            int vx = F[s];
            bool is_forced = vx == forced_;
            if (x <= 1) {
                if (is_forced) return;  // the forced vertex must be an end
            } else {
                if (mode_ != 0) return;  // a cycle has no ends
                double bon = bonus_ ? (*bonus_)[vx] : 0.0;
                if (x == 2) {  // the far end is already final: the path is complete
                    if (t != 1) return;
                    if (is_forced) f = 1;
                    if (forced_ >= 0 && !f) return;
                    c[s] = 0;
                    // other leaving slots must not hold open parts either (checked by only_closed)
                    if (only_closed(c, W)) record(v + bon, layer, src, took);
                    return;
                }
                if (t >= 2) return;
                ++t;
                if (is_forced) f = 1;
                if (forced_ >= 0 && !f && t == 2) return;
                c[x - 3] = 2;
                v += bon;
            }
            c[s] = 0;
        }
        // compact
        uint8_t d[kMaxWidth];
        int W2 = 0;
        for (int s = 0; s < W; ++s)
            if (remap[s] >= 0) {
                uint8_t x = c[s];
                d[W2++] = x >= 3 ? (uint8_t)(3 + remap[x - 3]) : x;
            }
        double pr = v;
        if (prune_) {
            double extra = mode_ == 0 ? maxb_ * (2 - t) : 0.0;
            double ub = v + fut_ + extra;
            for (int s2 = 0; s2 < W2; ++s2) ub += d[s2] == 0 ? pot0_[s2] : d[s2] == 1 ? 0.0 : pot1_[s2];
            if (best_val >= 0 && !gt(ub, best_val)) return;
            pr = ub;
            if (use_lam_) {
                double ul = v + lbase_ + extra;
                for (int s2 = 0; s2 < W2; ++s2) ul += d[s2] == 0 ? 2 * lpot_[s2] : d[s2] == 1 ? 0.0 : lpot_[s2];
                if (best_val >= 0 && !gt(ul, best_val)) return;
                pr = std::min(pr, ul);
            }
        }
        Key k = encode(d, W2, t, f);
        size_t h = hash(k) & table.mask;
        while (true) {
            int j = table.idx[h];
            if (j < 0) {
                table.idx[h] = (int)nxt.size();
                nxt.push_back({k, v, pr});
                bk.push_back((uint32_t)src | (took ? 0x80000000u : 0u));
                if (nxt.size() * 2 > table.idx.size()) grow();
                return;
            }
            if (nxt[j].k == k) {
                if (v > nxt[j].v) nxt[j].v = v, nxt[j].pr = pr, bk[j] = (uint32_t)src | (took ? 0x80000000u : 0u);
                return;
            }
            h = (h + 1) & table.mask;
        }
    }
    // Rank-based reduction (Bodlaender, Cygan, Kratsch, Nederlof 2013). States with the same
    // degree pattern (unused / interior / end for each frontier vertex, plus t and f) differ only in
    // how their path ends are paired. Whether a state and a future completion form one path (one
    // cycle through a virtual vertex z joining the two final ends) is decided over GF(2) by the
    // "cut" vectors of the pairings, so within a group it suffices to keep, heaviest first, the
    // states whose vectors are linearly independent: any completion that works with a dropped
    // state works with a kept state that is at least as heavy. Exact.
    void rank_reduce(int W2, std::vector<uint32_t>& bk) {
        struct G {
            std::vector<int> ids;
        };
        std::unordered_map<uint64_t, G> groups;
        std::vector<uint64_t> pat(nxt.size());
        for (int j = 0; j < (int)nxt.size(); ++j) {
            uint8_t c[kMaxWidth];
            int t, f;
            decode(nxt[j].k, W2, c, t, f);
            uint64_t key = (uint64_t)t | (uint64_t)f << 2;
            for (int s2 = 0; s2 < W2; ++s2) key |= (uint64_t)(c[s2] == 0 ? 0 : c[s2] == 1 ? 1 : 2) << (3 + 2 * s2);
            groups[key].ids.push_back(j);
        }
        std::vector<char> keep(nxt.size(), 1);
        bool any = false;
        for (auto& [key, gr] : groups) {
            if (gr.ids.size() < 2) continue;
            // points: end slots in slot order, then z when it is an end (t == 1)
            uint8_t c0[kMaxWidth];
            int t, f;
            decode(nxt[gr.ids[0]].k, W2, c0, t, f);
            std::vector<int> pts;
            for (int s2 = 0; s2 < W2; ++s2)
                if (c0[s2] >= 2) pts.push_back(s2);
            bool zend = mode_ == 0 && t == 1;
            int P = (int)pts.size() + (zend ? 1 : 0);
            if (P < 4 || P > rank_max_points) continue;  // P = 2: a single pairing
            int r = P / 2;
            size_t rankmax = (size_t)1 << (r - 1);
            if (gr.ids.size() <= rankmax) continue;
            std::vector<int> pidx(kMaxWidth + 1, -1);
            for (int i = 0; i < (int)pts.size(); ++i) pidx[pts[i]] = i;
            int zi = zend ? P - 1 : -1;
            size_t words = (((size_t)1 << (P - 1)) + 63) / 64;
            std::sort(gr.ids.begin(), gr.ids.end(), [&](int a, int b) { return nxt[a].v != nxt[b].v ? nxt[a].v > nxt[b].v : a < b; });
            std::vector<std::vector<uint64_t>> basis;  // rows, each with its pivot = highest set bit
            std::vector<size_t> piv;
            std::vector<uint64_t> vec(words);
            for (int j : gr.ids) {
                if (basis.size() >= rankmax) {
                    keep[j] = 0;
                    any = true;
                    continue;
                }
                uint8_t c[kMaxWidth];
                int tt, ff;
                decode(nxt[j].k, W2, c, tt, ff);
                // pairs over point indices
                std::vector<std::pair<int, int>> prs;
                int two[2], nt = 0;
                for (int s2 = 0; s2 < W2; ++s2) {
                    if (c[s2] >= 3 && (int)(c[s2] - 3) > s2) prs.push_back({pidx[s2], pidx[c[s2] - 3]});
                    else if (c[s2] == 2 && nt < 2) two[nt++] = pidx[s2];
                }
                if (mode_ == 0) {
                    if (tt == 1 && nt == 1) prs.push_back({two[0], zi});
                    else if (tt == 2 && nt == 2) prs.push_back({two[0], two[1]});
                    else if (nt != 0) continue;  // unexpected shape: keep the state
                }
                if ((int)prs.size() != r) continue;  // keep
                // consistent cuts: point 0 on the X side, every pair on one side
                int fixed = -1;
                std::vector<uint32_t> pm;
                for (int i = 0; i < r; ++i) {
                    uint32_t m = 0;
                    for (int x : {prs[i].first, prs[i].second})
                        if (x > 0) m |= 1u << (x - 1);
                    if (prs[i].first == 0 || prs[i].second == 0) fixed = i;
                    pm.push_back(m);
                }
                std::fill(vec.begin(), vec.end(), 0);
                std::vector<int> free_;
                uint32_t base = pm[fixed];
                for (int i = 0; i < r; ++i)
                    if (i != fixed) free_.push_back(i);
                for (uint32_t sub = 0; sub < (1u << (r - 1)); ++sub) {
                    uint32_t idx = base;
                    for (int b = 0; b < r - 1; ++b)
                        if (sub >> b & 1) idx |= pm[free_[b]];
                    vec[idx >> 6] |= 1ull << (idx & 63);
                }
                // reduce against the basis
                for (size_t q = 0; q < basis.size(); ++q)
                    if (vec[piv[q] >> 6] >> (piv[q] & 63) & 1)
                        for (size_t w = 0; w < words; ++w) vec[w] ^= basis[q][w];
                size_t hb = SIZE_MAX;
                for (size_t w = words; w-- > 0;)
                    if (vec[w]) {
                        hb = w * 64 + 63 - static_cast<size_t>(std::countl_zero(vec[w]));
                        break;
                    }
                if (hb == SIZE_MAX) {
                    keep[j] = 0;
                    any = true;
                    continue;
                }
                // keep rows reduced: clear the new pivot from older rows
                for (size_t q = 0; q < basis.size(); ++q)
                    if (basis[q][hb >> 6] >> (hb & 63) & 1)
                        for (size_t w = 0; w < words; ++w) basis[q][w] ^= vec[w];
                basis.push_back(vec);
                piv.push_back(hb);
            }
        }
        if (!any) return;
        std::vector<State> n2;
        std::vector<uint32_t> b2;
        for (int j = 0; j < (int)nxt.size(); ++j)
            if (keep[j]) n2.push_back(nxt[j]), b2.push_back(bk[j]);
        nxt.swap(n2);
        bk.swap(b2);
        rank_dropped_ += (long long)(n2.size() - nxt.size());
    }
  public:
    long long rank_dropped_ = 0;
  private:

    void grow() {
        size_t cap = table.idx.size() * 2;
        table.idx.assign(cap, -1);
        table.mask = cap - 1;
        for (int j = 0; j < (int)nxt.size(); ++j) {
            size_t h = hash(nxt[j].k) & table.mask;
            while (table.idx[h] >= 0) h = (h + 1) & table.mask;
            table.idx[h] = j;
        }
    }

    // order the chosen edges into a path (from the forced end if any) or a closed cycle
    Sol walk(const std::vector<int>& taken) const {
        Sol s;
        if (taken.empty()) return s;
        std::vector<std::vector<int>> nb(g.n);
        for (int e : taken) nb[g.edges[e].u].push_back(g.edges[e].v), nb[g.edges[e].v].push_back(g.edges[e].u);
        int start = -1;
        if (forced_ >= 0) start = forced_;
        else
            for (int v = 0; v < g.n; ++v)
                if (nb[v].size() == 1) {
                    start = v;
                    break;
                }
        if (start < 0) start = g.edges[taken[0]].u;  // cycle
        int prev = -1, u = start;
        s.path.push_back(u);
        while (true) {
            int nx = -1;
            for (int w : nb[u])
                if (w != prev) {
                    nx = w;
                    break;
                }
            if (nx < 0) break;
            if (nx == start) {
                s.path.push_back(start);
                break;
            }
            s.path.push_back(nx);
            prev = u, u = nx;
            if (s.path.size() > (size_t)g.n + 1) break;
        }
        return s;
    }
};

using FrontierDP = FrontierDPT<2>;

}  // namespace lp
