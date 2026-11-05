// Exact branch-and-bound DFS (iterative) with the bounds and pruning chosen by the benchmarks:
//   - reachability and articulation points in one Tarjan scan ("remaining map" bound)
//   - (top1 + top2) / 2 bound
//   - Lagrangian forest bound at every depth (auto-disabled when loose), 1-forest at cycle roots
//   - dominance memo keyed by (vertex, reachable set, cycle start)
//   - children ordered by Lagrangian-penalised weight
#pragma once
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>

#include "graph.hpp"

namespace lp {

using Clock = std::chrono::steady_clock;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Timeout {};
struct Budget {};  // node budget of a trial search exhausted (Lagrangian branching)

// ---------------------------------------------------------------- incumbent shared by threads
struct Incumbent {
    std::atomic<double> best{-1.0};
    std::mutex mu;
    std::vector<int> path;

    double get() const { return best.load(std::memory_order_relaxed); }
    void offer(double L, const std::vector<int>& p) {
        if (!gt(L, get())) return;
        std::lock_guard<std::mutex> lk(mu);
        if (gt(L, best.load())) {
            path = p;
            best.store(L);
        }
    }
    template <class F>
    void offer_lazy(double L, F make_path) {  // builds the path only when it improves
        if (!gt(L, get())) return;
        std::lock_guard<std::mutex> lk(mu);
        if (gt(L, best.load())) {
            path = make_path();
            best.store(L);
        }
    }
};

// set once the time limit is hit: big tables are then left to the OS instead of being freed one
// node at a time (freeing a memo of millions of entries took about 1 s after the deadline)
inline std::atomic<bool> g_abort{false};

struct Control {
    bool has_deadline = false;
    Clock::time_point deadline;
    std::atomic<bool> stop{false};
    std::atomic<long long> nodes{0};
    void check() {
        if (has_deadline && Clock::now() > deadline) stop = true;
        if (stop) {
            g_abort = true;
            throw Timeout{};
        }
    }
};

// ---------------------------------------------------------------- dominance memo (sharded)
struct Memo {
    struct Key {
        int c;
        long long s;
        std::vector<uint64_t> R;
        bool operator==(const Key& o) const { return c == o.c && s == o.s && R == o.R; }
    };
    struct H {
        size_t operator()(const Key& k) const {
            uint64_t h = 1469598103934665603ULL ^ (uint64_t)(k.c * 1000003LL + k.s * 7919LL);
            for (uint64_t w : k.R) {
                h ^= w + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            return (size_t)h;
        }
    };
    static constexpr int kShards = 64;
    std::mutex mu[kShards];
    std::unordered_map<Key, double, H> mp[kShards];
    std::atomic<size_t> size{0};
    size_t cap;
    explicit Memo(size_t cap_) : cap(cap_) {}
    ~Memo() {
        if (g_abort)
            for (auto& m : mp) new std::unordered_map<Key, double, H>(std::move(m));  // leaked on purpose
    }

    // true when a recorded prefix with the same future was at least as long (prune)
    bool dominated(Key&& k, double len) {
        size_t h = H{}(k);
        int sh = (int)(h % kShards);
        std::lock_guard<std::mutex> lk(mu[sh]);
        auto it = mp[sh].find(k);
        if (it != mp[sh].end()) {
            if (it->second >= len - 1e-12) return true;
            it->second = len;
            return false;
        }
        if (size.load(std::memory_order_relaxed) < cap) {
            mp[sh].emplace(std::move(k), len);
            size.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
};

// ---------------------------------------------------------------- per-graph data shared by searchers
struct SearchCtx {
    const Graph* g = nullptr;
    int W = 1;
    std::vector<double> top1, t12;
    // Lagrangian
    bool use_lagr = false;      // bound enabled
    bool use_1forest = false;   // 1-forest bound at cycle roots
    std::vector<double> lam;    // empty = no multipliers (ordering falls back to weight)
    std::vector<std::pair<double, int>> mods;  // (w - lam_u - lam_v, eid), positive only, descending

    void init(const Graph& gr) {
        g = &gr;
        W = std::max(1, (gr.n + 63) / 64);
        top1.assign(gr.n, 0.0);
        t12.assign(gr.n, 0.0);
        for (int u = 0; u < gr.n; ++u) {
            double a = 0, b = 0;
            for (auto [v, e] : gr.adj[u]) {
                double w = gr.edges[e].w;
                if (w > a) {
                    b = a;
                    a = w;
                } else if (w > b) b = w;
            }
            top1[u] = a;
            t12[u] = a + b;
        }
    }
    void set_lambda(const std::vector<double>& l) {
        lam = l;
        mods.clear();
        for (int i = 0; i < (int)g->edges.size(); ++i) {
            auto& e = g->edges[i];
            double m = e.w - lam[e.u] - lam[e.v];
            if (m > 0) mods.push_back({m, i});
        }
        std::sort(mods.begin(), mods.end(), [](auto& a, auto& b) { return a.first > b.first; });
    }
};

// ---------------------------------------------------------------- one searcher per thread
class Searcher {
   public:
    Searcher(const SearchCtx& ctx, Incumbent& inc, Control& ctl, Memo* memo)
        : C(ctx), g(*ctx.g), inc(inc), ctl(ctl), memo(memo), W(ctx.W), n(ctx.g->n) {
        vis.assign((size_t)(n + 2) * W, 0);
        Rbuf.assign((size_t)(n + 2) * W, 0);
        path.assign(n + 2, -1);
        disc.assign(n, 0);
        low.assign(n, 0);
        stamp.assign(n, 0);
        nub.assign(n, 0.0);
        uf.assign(n, 0);
        ufst.assign(n, 0);
    }

    long long nodes = 0;
    // optional end bonuses (block-cut tree "down" values): a path x..y is worth
    // len + bonus[x] + bonus[y] (a single vertex counts its bonus once)
    const std::vector<double>* bonus = nullptr;
    long long budget = -1;  // trial search: stop after this many nodes (throws Budget)
    int first = -1;         // force the first move of a cycle search

    // simple paths starting at `start`; target = -1 for "end anywhere"
    void run_path(int start, int target) {
        first = -1;
        cycle = false;
        s = -1;
        tgt = target;
        sb = bonus ? (*bonus)[start] : 0.0;
        std::fill(vis.begin(), vis.begin() + W, 0);
        setb(V(0), start);
        go(start);
    }
    // simple cycles whose smallest vertex is s
    void run_cycle(int s0, int first_v = -1) {
        first = first_v;
        sb = 0.0;
        cycle = true;
        s = s0;
        tgt = -1;
        uint64_t* v0 = V(0);
        std::fill(v0, v0 + W, 0);
        for (int x = 0; x <= s0; ++x) setb(v0, x);
        go(s0);
    }

   private:
    const SearchCtx& C;
    const Graph& g;
    Incumbent& inc;
    Control& ctl;
    Memo* memo;
    int W, n;
    bool cycle = false;
    int s = -1, tgt = -1;
    double sb = 0.0;  // bonus of the start vertex
    long long run_nodes = 0;
    std::vector<uint64_t> vis, Rbuf;
    std::vector<int> path;
    struct Frame {
        double len;
        int mb, me, next;
    };
    std::vector<Frame> st;
    std::vector<std::pair<int, double>> mv;  // (vertex, weight)
    std::vector<std::pair<double, int>> tmp;
    // Tarjan / union-find scratch (stamped)
    std::vector<int> disc, low, stamp, uf, ufst;
    std::vector<double> nub;
    int cur = 0, ufcur = 0;
    struct TF {
        int u, pe;
        size_t it;
    };
    std::vector<TF> tst;
    std::vector<int> vst;

    uint64_t* V(int d) { return vis.data() + (size_t)d * W; }
    uint64_t* RB(int d) { return Rbuf.data() + (size_t)d * W; }
    static bool tb(const uint64_t* b, int i) { return (b[i >> 6] >> (i & 63)) & 1; }
    static void setb(uint64_t* b, int i) { b[i >> 6] |= 1ULL << (i & 63); }

    void go(int start) {
        run_nodes = 0;
        st.clear();
        mv.clear();
        path[0] = start;
        enter(start, 0.0, 0);
        while (!st.empty()) {
            int d = (int)st.size() - 1;
            Frame& f = st.back();
            if (f.next == f.me) {
                mv.resize(f.mb);
                st.pop_back();
                continue;
            }
            auto [v, w] = mv[f.next++];
            double L = f.len + w;
            const uint64_t* pv = V(d);
            uint64_t* cv = V(d + 1);
            if (tb(pv, v)) continue;
            std::copy(pv, pv + W, cv);
            setb(cv, v);
            path[d + 1] = v;
            enter(v, L, d + 1);
        }
    }

    void enter(int c, double len, int depth) {
        if ((++nodes & 1023) == 0) {
            ctl.nodes.fetch_add(1024, std::memory_order_relaxed);
            ctl.check();
        }
        if (budget >= 0 && ++run_nodes > budget) throw Budget{};
        // terminal candidates
        if (cycle) {
            if (depth >= 2)
                for (auto [v, e] : g.adj[c])
                    if (v == s) {
                        double L = len + g.edges[e].w;
                        inc.offer_lazy(L, [&] {
                            std::vector<int> p(path.begin(), path.begin() + depth + 1);
                            p.push_back(s);
                            return p;
                        });
                    }
        } else if (tgt < 0 || c == tgt) {
            double eb = (bonus && depth > 0) ? (*bonus)[c] : 0.0;
            inc.offer_lazy(len + sb + eb, [&] { return std::vector<int>(path.begin(), path.begin() + depth + 1); });
            if (c == tgt) return;
        }
        // reachable set + remaining-map (BCC) bound in one Tarjan scan
        uint64_t* R = RB(depth);
        double bcc = scan(c, V(depth), R);
        bool any = false;
        for (int i = 0; i < W; ++i)
            if (R[i]) {
                any = true;
                break;
            }
        if (!any) return;
        if (tgt >= 0 && !tb(R, tgt)) return;
        // bounds
        // A future path ends at some u in R. That end uses one block edge (not two) and earns its
        // end bonus; so for the top-2 bound the end adds (2*bonus(u) - top2(u)) to the incidence
        // sum, and for the Lagrangian bound (bonus(u) - lambda(u)). Take the best u for each.
        double s12 = 0, slam = 0, mxb = 0;
        double end2 = -kInf, endL = -kInf;
        bool lam_ok = !C.lam.empty();
        for (int i = 0; i < W; ++i)
            for (uint64_t x = R[i]; x; x &= x - 1) {
                int u = i * 64 + std::countr_zero(x);
                double bu = bonus ? (*bonus)[u] : 0.0;
                s12 += C.t12[u];
                if (bu > mxb) mxb = bu;
                if (2 * bu - (C.t12[u] - C.top1[u]) > end2) end2 = 2 * bu - (C.t12[u] - C.top1[u]);
                if (lam_ok) {
                    slam += C.lam[u];
                    if (bu - C.lam[u] > endL) endL = bu - C.lam[u];
                }
            }
        double ub;
        if (cycle) {
            double extra_c = C.top1[s];  // closing edge (incidence at s)
            ub = len + (C.top1[c] + s12 + extra_c) / 2.0;
            ub = std::min(ub, len + bcc + extra_c);
        } else {
            ub = len + sb + (C.top1[c] + s12 + end2) / 2.0;
            ub = std::min(ub, len + bcc + sb);  // end bonuses are inside the scan's branch values
        }
        if (cycle && c == s) {
            if (C.use_1forest) ub = std::min(ub, one_forest(c, R, slam));
        } else if (C.use_lagr) {
            double f = forest(c, R);
            ub = std::min(ub, len + f + C.lam[c] + 2 * slam + (cycle ? C.lam[s] : sb + endL));
        }
        if (!gt(ub, inc.get())) return;
        // the root of a forced-first search shares its state with sibling searches but explores only
        // one child, so it must not be recorded or pruned by the memo
        if (memo && !(depth == 0 && first >= 0)) {
            long long ks = !cycle ? -1 : s;
            Memo::Key k{c, ks, std::vector<uint64_t>(R, R + W)};
            if (memo->dominated(std::move(k), len + sb)) return;
        }
        // children: reachable neighbours, best penalised weight first
        int mb = (int)mv.size();
        tmp.clear();
        for (auto [v, e] : g.adj[c])
            if (tb(R, v) && !(depth == 0 && first >= 0 && v != first)) {
                double w = g.edges[e].w;
                double key = lam_ok ? w - C.lam[c] - C.lam[v] : w;
                tmp.push_back({key, (int)mv.size()});
                mv.push_back({v, w});
            }
        // sort the appended slice by key (descending)
        std::vector<std::pair<int, double>> slice(mv.begin() + mb, mv.end());
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < tmp.size(); ++i) mv[mb + i] = slice[tmp[i].second - mb];
        st.push_back({len, mb, (int)mv.size(), mb});
    }

    // Tarjan DFS from c over unvisited vertices (back edges to c count). Fills R (reachable,
    // excluding c) and returns the remaining-map bound: along the block-cut tree rooted at c a path
    // continues into one branch only, so branches combine by max, block contents by sum of top1.
    double scan(int c, const uint64_t* visited, uint64_t* R) {
        std::fill(R, R + W, 0);
        ++cur;
        int t = 0;
        stamp[c] = cur;
        disc[c] = low[c] = t++;
        nub[c] = 0;
        tst.clear();
        vst.clear();
        tst.push_back({c, -1, 0});
        while (!tst.empty()) {
            TF& f = tst.back();
            int u = f.u;
            bool adv = false;
            while (f.it < g.adj[u].size()) {
                auto [v, e] = g.adj[u][f.it++];
                if (e == f.pe) continue;
                if (v != c && tb(visited, v)) continue;
                if (stamp[v] != cur) {
                    stamp[v] = cur;
                    disc[v] = low[v] = t++;
                    nub[v] = 0;
                    setb(R, v);
                    vst.push_back(v);
                    tst.push_back({v, e, 0});
                    adv = true;
                    break;
                } else if (disc[v] < low[u]) {
                    low[u] = disc[v];
                }
            }
            if (adv) continue;
            tst.pop_back();
            if (!tst.empty()) {
                int p = tst.back().u;
                if (low[u] < low[p]) low[p] = low[u];
                if (low[u] >= disc[p]) {
                    // the path either ends inside this block (end bonus <= mb) or continues
                    // into one child branch (bc already includes that branch's end bonus)
                    double s1 = 0, bc = 0, mb = 0;
                    while (true) {
                        int x = vst.back();
                        vst.pop_back();
                        s1 += C.top1[x];
                        if (nub[x] > bc) bc = nub[x];
                        if (bonus && (*bonus)[x] > mb) mb = (*bonus)[x];
                        if (x == u) break;
                    }
                    double val = s1 + std::max(bc, mb);
                    if (val > nub[p]) nub[p] = val;
                }
            }
        }
        return nub[c];
    }

    int find(int x) {
        while (true) {
            if (ufst[x] != ufcur) {
                ufst[x] = ufcur;
                uf[x] = x;
            }
            if (uf[x] == x) return x;
            int p = uf[x];
            if (ufst[p] != ufcur) {
                ufst[p] = ufcur;
                uf[p] = p;
            }
            uf[x] = uf[p];
            x = p;
        }
    }
    bool inH(int v, const uint64_t* R, int c) const {
        return v == c || tb(R, v) || (cycle && v == s);
    }
    // max-weight forest on H = R + c (+ s) with penalised weights
    double forest(int c, const uint64_t* R) {
        ++ufcur;
        double tot = 0;
        for (auto& [m, i] : C.mods) {
            int u = g.edges[i].u, v = g.edges[i].v;
            if (!inH(u, R, c) || !inH(v, R, c)) continue;
            int a = find(u), b = find(v);
            if (a != b) {
                uf[a] = b;
                tot += m;
            }
        }
        return tot;
    }
    // cycle root: any cycle inside H is a forest plus one edge
    double one_forest(int c, const uint64_t* R, double slam) {
        ++ufcur;
        double tot = 0, extra = 0;
        for (auto& [m, i] : C.mods) {
            int u = g.edges[i].u, v = g.edges[i].v;
            if (!inH(u, R, c) || !inH(v, R, c)) continue;
            int a = find(u), b = find(v);
            if (a != b) {
                uf[a] = b;
                tot += m;
            } else if (m > extra) extra = m;
        }
        return tot + extra + 2 * (slam + C.lam[c]);
    }
};

}  // namespace lp
