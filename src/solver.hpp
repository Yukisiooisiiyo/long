// Solver: connected components -> blocks (2-connected parts) -> an exact method per block -> the
// block-cut tree combines the blocks' answers. Per block, by measured cost:
//   single cycle         closed form (two arcs)
//   up to 12 vertices    path enumeration (sparse) or bitmask DP
//   thin (frontier <=16) frontier DP (frontier.hpp)
//   dense                LP branch and cut (lp_bnc.hpp)
//   otherwise            branch and bound with Lagrangian bounds (search.hpp)
// A component that is one block is solved whole. Design decisions: docs/HISTORY.md.
#pragma once
#include <functional>
#include <thread>

#include "bitdp.hpp"
#include "lagrange.hpp"
#include "search.hpp"
#include "frontier.hpp"
#include "lp_bnc.hpp"

namespace lp {

struct Config {
    int threads = 1;
    // Starting threads costs ~0.3 ms and only pays off on big blocks (2 threads helped only graphs
    // whose largest block has 40+ vertices). Without an explicit --threads, a component whose
    // largest block is smaller runs on one thread.
    bool threads_auto = false;
    int par_min_block = 40;
    int t_dp = 12;                   // small-block methods up to this many vertices
    double lagr_ratio = 1.3;         // Lagrangian bound off when root ub / incumbent exceeds this
    int lagr_iters = 100;
    bool edge_deletion = true;
    size_t memo_bytes = size_t(1) << 30;  // dominance memo budget
    // small blocks: enumerate the paths when the block is sparse (dense ones: bitmask DP)
    int enum_n = 12;                 // 0 = always bitmask DP
    long long enum_budget = 100000;  // DFS steps before falling back to the bitmask DP
    int enum_extra = 8;              // at most n + this many edges
    // frontier DP for thin blocks (exact; falls back to branch and bound when it gives up).
    // State counts depend on the frontier width only (about 100 at width 5, 1.5k at 7, 25k at 9).
    bool fdp = true;
    int fdp_width = 8;               // always used up to this frontier width
    int fdp_width_sparse = 16;       // up to this width when the block is sparse (m <= 2n)
    int fdp_min_n = 13;              // and the block has at least this many vertices
    long long fdp_budget = 50000000; // total stored states before giving up (4 bytes each)
    long long fdp_layer = 1 << 20;   // states in one step before giving up
    long long fdp_beam = 1000;       // beam pass before the exact pass (wide frontiers)
    int fdp_prune_width = 8;         // from this width on, drop states that cannot beat a known answer
    long long fdp_rank = 500000;     // rank-based reduction above this many states per step (0 = off)
    int fdp_rank_points = 16;
    // LP branch and cut for dense blocks
    bool fdp_narrow = true;      // one-word state keys for frontiers up to 12 wide
    bool lp_bnc = true;
    int lp_max_n = 400;
    long long lp_max_m = 20000;  // and edges (the simplex keeps a dense table)
    int fdp_lp_width = 14;       // with the revised simplex, blocks wider than this go to the LP first
    bool lp_blossom = true;      // blossom cuts in the LP
    int lp_order = 1;            // LP node order: 0 depth-first, 1 best-first with plunging
    bool lp_race = true;         // depth-first and best-first in two threads, the first proof wins
    bool lp_contract = true;     // contract degree-2 chains inside the LP
    double lp_plunge = 5e-4;     // plunge while the bound is within this fraction of the best open bound
    bool lp_revised = true;      // revised simplex (sparse, product-form inverse) instead of the dense table
    int lp_cuts_round = 20;      // most violated cuts added per separation round
    bool lp_light_init = true;   // no beam search before the LP
    double lp_min_ratio = 1.0;   // every block the frontier DP does not take (was 2.0: dense only)
    long long lp_nodes = 3000;
    // Lagrangian branching: per-task multipliers, a budgeted search, then split by the next step
    bool path_task = true;           // also in the path phase (the cycle phase always uses it)
    int task_lagr_iters = 50;
    int task_depth = 8;              // prefix length up to which a hard task is split further
    long long task_budget = 2000;    // nodes of the trial search before splitting
};

struct Stats {
    long long nodes = 0;
    int blocks = 0, max_block = 0, bitdp_blocks = 0, bnb_blocks = 0, dead = 0;
    int lagr_off = 0, proved_root = 0;
    int threads_used = 1;
    int fdp_blocks = 0, fdp_width = 0, fdp_fallback = 0;
    long long tasks_skipped = 0;
    long long lp_nodes = 0, lp_runs = 0, lp_fallback = 0, lp_chains = 0;
    long long fdp_states = 0;
};

class Solver {
   public:
    Solver(Config cfg, Control& ctl) : cfg(cfg), ctl(ctl), user_threads(cfg.threads) {}
    Stats st;
    Incumbent glob;  // best known in input-graph indices (used for --on-timeout approx)

    Sol solve(const Graph& g) {
        Sol best;
        auto comps = components(g);
        // with a time limit, a cheap answer first so that --on-timeout approx has something to print
        std::vector<Graph> subs;
        bool whole = comps.size() == 1;  // the whole graph: no copy
        for (auto& c : comps) {
            if (!whole) subs.push_back(g.induced(c));
            if (c.size() == 1) continue;
            if (!ctl.has_deadline) continue;
            Sol h = mst_diameter(whole ? g : subs.back());
            glob.offer(h.len, mapv(h.path, c));
        }
        for (size_t i = 0; i < comps.size(); ++i) {
            if (comps[i].size() == 1) continue;
            Sol s = solve_component(whole ? g : subs[i], comps[i]);
            if (gt(s.len, best.len)) best = {s.len, mapv(s.path, comps[i])};
        }
        return best;
    }

   private:
    Config cfg;
    Control& ctl;
    int user_threads;

    static std::vector<int> mapv(const std::vector<int>& p, const std::vector<int>& m) {
        std::vector<int> r;
        r.reserve(p.size());
        for (int x : p) r.push_back(m[x]);
        return r;
    }

    void offer_global(const Sol& s, const std::vector<int>& m) {
        if (s.len >= 0) glob.offer(s.len, mapv(s.path, m));
    }

    // ------------------------------------------------------------ parallel helper
    void parallel_for(int count, const std::function<void(int, int)>& body) {  // body(item, thread)
        int T = std::max(1, std::min(cfg.threads, count));
        std::atomic<int> next{0};
        auto worker = [&](int tid) {
            try {
                for (int i; (i = next.fetch_add(1)) < count;) {
                    if (ctl.stop) break;
                    body(i, tid);
                }
            } catch (const Timeout&) {
                ctl.stop = true;
            }
        };
        if (T == 1) worker(0);
        else {
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) th.emplace_back(worker, t);
            for (auto& x : th) x.join();
        }
    }

    LPOptions lp_opts() const {
        LPOptions o;
        o.node_cap = cfg.lp_nodes, o.cuts_round = cfg.lp_cuts_round, o.blossoms = cfg.lp_blossom;
        o.race = cfg.lp_race && cfg.threads >= 2, o.order = cfg.lp_order, o.plunge = cfg.lp_plunge, o.contract = cfg.lp_contract;
        return o;
    }

    // ------------------------------------------------------------ initial solution portfolio
    // light: skip the beam search (the LP finds good solutions itself; on dense graphs the beam
    // search was 96% of the time)
    Sol initial(const Graph& g, bool light = false) {
        Sol best{0.0, {0}};
        if (g.n <= 5000) take(best, greedy(g));
        take(best, mst_diameter(g));
        if (g.edges.size() >= (size_t)g.n) take(best, fundamental_cycles(g));
        if (g.n <= 800 && !light) take(best, beam_search(g));
        return best;
    }
    Sol polish(const Graph& g, Sol best, const LagrResult* lr) {
        if (lr && !lr->forest.empty()) {
            auto ps = forest_paths(g, lr->forest);
            for (size_t i = 0; i < ps.size() && i < 3; ++i) take(best, extend_greedy(g, ps[i]));
        }
        std::vector<int> p = best.path;
        if (p.size() > 2 && p.front() == p.back()) p.pop_back();
        if (!p.empty()) take(best, ils(g, p));
        return best;
    }

    // ------------------------------------------------------------ frontier DP
    // Runs on one block: path with end bonuses (optionally from a forced end) or cycle. Independent
    // runs go to separate threads. Returns false when any run gives up (caller falls back).
    struct FdpTask {
        int mode;
        const std::vector<double>* bonus;
        int forced;
        double lb = -1;  // only answers above this matter (empty out: nothing beat it)
        const std::vector<double>* lam = nullptr;  // multipliers for the DP's upper bound
        long long beam = 0;
        Sol out;
        bool ok = false;
        long long states = 0;
    };
    bool run_fdp(const Graph& g, const FrontierOrder& o, std::vector<FdpTask>& tasks) {
        std::atomic<bool> timeout{false};
        auto run_with = [&](auto tag, FdpTask& t) {
            using DP = typename decltype(tag)::type;
            DP dp(g, o, ctl, cfg.fdp_budget, cfg.fdp_layer);
            dp.rank_ = cfg.fdp_rank;
            dp.rank_max_points = cfg.fdp_rank_points;
            double lb = t.lb;
            if (t.beam > 0) {  // a beam-limited pass first: a strong answer makes the exact pass prune
                DP bd(g, o, ctl, cfg.fdp_budget, cfg.fdp_layer);
                bd.rank_ = cfg.fdp_rank;
                bd.rank_max_points = cfg.fdp_rank_points;
                auto rb = bd.run(t.mode, t.bonus, t.forced, lb, t.lam, t.beam);
                if (rb.ok && !rb.sol.path.empty()) t.out = rb.sol, lb = rb.sol.len;
            }
            auto r = dp.run(t.mode, t.bonus, t.forced, lb, t.lam);
            if (r.ok && r.sol.path.empty() && !t.out.path.empty()) r.sol = t.out;  // the beam answer was optimal
            t.ok = r.ok, t.out = r.sol, t.states = r.states;
        };
        auto one = [&](FdpTask& t) {
            try {
                // one-word state keys when the frontier is at most 12 wide
                if (cfg.fdp_narrow && o.width <= FrontierDPT<1>::kMaxWidth)
                    run_with(std::type_identity<FrontierDPT<1>>{}, t);
                else
                    run_with(std::type_identity<FrontierDPT<2>>{}, t);
            } catch (const Timeout&) {
                timeout = true;
                ctl.stop = true;
            }
        };
        if (cfg.threads >= 2 && tasks.size() >= 2) {
            std::vector<std::thread> th;
            for (size_t i = 1; i < tasks.size(); ++i) th.emplace_back(one, std::ref(tasks[i]));
            one(tasks[0]);
            for (auto& x : th) x.join();
        } else
            for (auto& t : tasks) {
                one(t);
                if (!t.ok) break;
            }
        if (timeout) throw Timeout{};
        bool ok = true;
        for (auto& t : tasks) st.fdp_states = std::max(st.fdp_states, t.states), ok = ok && t.ok;
        return ok;
    }
    // LP branch and cut for blocks the frontier DP does not take: up to lp_max_n vertices and at
    // least lp_min_ratio edges per vertex
    // size of the LP after its chain contraction: the vertices of degree other than 2
    int lp_size(const Graph& g) const {
        if (!cfg.lp_contract) return g.n;
        int k = 0;
        for (int v = 0; v < g.n; ++v) k += g.adj[v].size() != 2;
        return std::max(k, 1);
    }
    bool use_lp(const Graph& g) const {
        return cfg.lp_bnc && lp_size(g) <= cfg.lp_max_n && (long long)g.edges.size() <= cfg.lp_max_m &&
               (double)g.edges.size() >= cfg.lp_min_ratio * g.n;
    }
    bool use_fdp(const Graph& g, FrontierOrder& o) {
        if (!cfg.fdp || g.n < cfg.fdp_min_n) return false;
        o = best_frontier_order(g);
        int lim = (long long)g.edges.size() <= 2LL * g.n ? std::max(cfg.fdp_width, cfg.fdp_width_sparse)
                                                          : cfg.fdp_width;
        // wider than fdp_lp_width: the LP (revised simplex) was faster in the measurements
        if (cfg.lp_revised && o.width > cfg.fdp_lp_width && use_lp(g)) return false;
        return o.width <= std::min(lim, FrontierDP::kMaxWidth);
    }

    size_t memo_cap(int W) const { return std::max<size_t>(100000, cfg.memo_bytes / (W * 8 + 96)); }

    // ------------------------------------------------------------ whole-component branch and bound
    Sol solve_whole(const Graph& g0, const std::vector<int>& to_global) {
        {
            FrontierOrder o;
            if (use_fdp(g0, o)) {
                // wide frontier: start from a heuristic answer so the DP can drop hopeless states
                Sol h;
                if (o.width >= cfg.fdp_prune_width) h = polish(g0, initial(g0), nullptr);
                double lb = h.path.empty() ? -1.0 : h.len;
                LagrResult lp_, lc_;
                if (lb >= 0) lp_ = subgradient(g0, lb, cfg.lagr_iters), lc_ = subgradient_cycle(g0, lb, cfg.lagr_iters);
                long long beam = lb >= 0 ? cfg.fdp_beam : 0;
                std::vector<FdpTask> T{{0, nullptr, -1, lb, lb >= 0 ? &lp_.lam : nullptr, beam, {}},
                                       {1, nullptr, -1, lb, lb >= 0 ? &lc_.lam : nullptr, beam, {}}};
                if (run_fdp(g0, o, T)) {
                    Sol p = T[0].out, c = T[1].out;
                    if (p.path.empty()) p = h;
                    ++st.fdp_blocks;
                    st.fdp_width = std::max(st.fdp_width, o.width);
                    Sol b = p;
                    if (!c.path.empty() && gt(c.len, b.len)) b = c;
                    offer_global(b, to_global);
                    return b;
                }
                ++st.fdp_fallback;
            }
        }
        if (g0.n <= 2) return solve_whole_impl(g0, to_global);
        // Renumber vertices light first (by top1 + top2). Cycles are enumerated from their smallest
        // vertex, so light vertices become starts whose tasks are pruned early (measured best).
        std::vector<double> t12(g0.n, 0);
        for (int u = 0; u < g0.n; ++u) {
            double a = 0, b = 0;
            for (auto [v, e] : g0.adj[u]) {
                double w = g0.edges[e].w;
                if (w > a) b = a, a = w;
                else if (w > b) b = w;
            }
            t12[u] = a + b;
        }
        std::vector<int> ord(g0.n), pos(g0.n);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return t12[a] < t12[b]; });
        for (int i = 0; i < g0.n; ++i) pos[ord[i]] = i;
        std::vector<Edge> es;
        for (auto& e : g0.edges) es.push_back({pos[e.u], pos[e.v], e.w});
        Graph gp(g0.n, std::move(es));
        std::vector<int> tg(g0.n);
        for (int i = 0; i < g0.n; ++i) tg[i] = to_global[ord[i]];
        Sol r = solve_whole_impl(gp, tg);
        for (int& v : r.path) v = ord[v];
        return r;
    }

    Sol solve_whole_impl(const Graph& g0, const std::vector<int>& to_global) {
        if (g0.n == 1) return {0.0, {0}};
        Sol best = initial(g0, cfg.lp_light_init && use_lp(g0));
        LagrResult lr = subgradient(g0, best.len, cfg.lagr_iters);
        best = polish(g0, best, &lr);
        offer_global(best, to_global);
        double cyc_ub = cycle_bound(g0, lr.lam);
        if (!gt(lr.ub, best.len) && !gt(cyc_ub, best.len)) {
            ++st.proved_root;
            return best;
        }
        Graph g = g0;
        if (cfg.edge_deletion) {
            auto dead = dead_edges(g0, lr, best.len);
            std::vector<Edge> es;
            for (size_t i = 0; i < g0.edges.size(); ++i)
                if (!dead[i]) es.push_back(g0.edges[i]);
                else ++st.dead;
            g = Graph(g0.n, std::move(es));
        }
        SearchCtx ctx;
        ctx.init(g);
        ctx.set_lambda(lr.lam);
        bool lagr = lr.ub / std::max(best.len, 1e-12) <= cfg.lagr_ratio;
        if (!lagr) ++st.lagr_off;
        ctx.use_lagr = ctx.use_1forest = lagr;
        Incumbent inc;
        inc.offer(best.len, best.path);
        // dense graphs: LP branch and cut first; if it proves the answer, done
        if (use_lp(g)) {
            auto r = run_lp_bnc(cfg.lp_revised, g, nullptr, -1, inc, ctl, lp_opts());
            st.lp_nodes += r.nodes, st.lp_runs += 1, st.lp_chains += r.chains;
            if (ctl.stop) throw Timeout{};
            if (r.proved) {
                offer_global({inc.get(), inc.path}, to_global);
                return {inc.get(), inc.path};
            }
            ++st.lp_fallback;
        }
        // path phase; promising starts first: the incumbent's ends, then heavy vertices
        std::vector<int> starts(g.n);
        std::iota(starts.begin(), starts.end(), 0);
        {
            std::vector<double> key(g.n);
            for (int v = 0; v < g.n; ++v) key[v] = ctx.t12[v];
            if (!best.path.empty()) key[best.path.front()] = key[best.path.back()] = kInf;
            std::stable_sort(starts.begin(), starts.end(), [&](int a, int b) { return key[a] > key[b]; });
        }
        if (cfg.path_task) {
            run_paths_task(g, ctx, inc, starts);
        } else {
            Memo memo(memo_cap(ctx.W));
            std::vector<std::unique_ptr<Searcher>> S;
            for (int t = 0; t < std::max(1, cfg.threads); ++t)
                S.emplace_back(std::make_unique<Searcher>(ctx, inc, ctl, &memo));
            parallel_for(g.n, [&](int i, int t) { S[t]->run_path(starts[i], -1); });
            for (auto& x : S) st.nodes += x->nodes;
        }
        offer_global({inc.get(), inc.path}, to_global);
        if (ctl.stop) throw Timeout{};
        // cycle phase, with multipliers tuned for cycles
        SearchCtx cctx = ctx;
        {
            LagrResult lc = subgradient_cycle(g, inc.get(), cfg.lagr_iters);
            cctx.set_lambda(lc.lam);
            bool on = lc.ub / std::max(inc.get(), 1e-12) <= cfg.lagr_ratio;
            cctx.use_lagr = cctx.use_1forest = on;
        }
        run_cycles_task(g, cctx, inc);
        offer_global({inc.get(), inc.path}, to_global);
        if (ctl.stop) throw Timeout{};
        return {inc.get(), inc.path};
    }

    // ------------------------------------------------------------ Lagrangian branching
    // Cycle phase. Task s looks for cycles through s that use only vertices >= s. Multipliers
    // re-tuned for that subgraph give a much tighter bound than one shared set; a task whose bound
    // does not beat the incumbent is skipped. The rest are split by their first step.
    void run_cycles_task(const Graph& g, const SearchCtx& base, Incumbent& inc) {
        std::atomic<long long> nodes{0};
        parallel_for(g.n, [&](int s, int) {
            std::vector<Edge> es;
            for (auto& e : g.edges)
                if (e.u >= s && e.v >= s) es.push_back(e);
            if (es.size() < 3) return;
            bool touches = false;
            for (auto& e : es)
                if (e.u == s || e.v == s) touches = true;
            if (!touches) return;
            Graph gs(g.n, std::move(es));
            if (!base.lam.empty() && !gt(cycle_bound(gs, base.lam), inc.get())) {
                ++st_tasks_skipped;
                return;
            }
            LagrResult lr = subgradient_cycle(gs, inc.get(), cfg.task_lagr_iters, base.lam.empty() ? nullptr : &base.lam);
            if (!gt(lr.ub, inc.get())) {
                ++st_tasks_skipped;
                return;
            }
            std::vector<std::pair<double, int>> firsts;
            for (auto [v, e] : gs.adj[s]) firsts.push_back({gs.edges[e].w - lr.lam[v], v});
            std::stable_sort(firsts.rbegin(), firsts.rend());
            for (auto [key, v] : firsts) {
                if (ctl.stop) return;
                explore_cycle(g, s, {s, v}, g.weight(s, v), lr.lam, 1, inc, nodes);
            }
        });
        st.nodes += nodes.load();
        st.tasks_skipped += st_tasks_skipped.load();
        st_tasks_skipped = 0;
    }
    std::atomic<int> st_tasks_skipped{0};

    // Path phase with the same branching. Start s: bound with
    // multipliers re-tuned for "a path with one end at s" (capacity 1 at s); skipped if it cannot
    // beat the incumbent; else a budgeted search, and if that does not finish, one subtask per next
    // step, each on the graph without the fixed prefix and with its own multipliers.
    void run_paths_task(const Graph& g, const SearchCtx& base, Incumbent& inc, const std::vector<int>& starts) {
        std::atomic<long long> nodes{0};
        parallel_for(g.n, [&](int i, int) {
            int s0 = starts[i];
            explore_path(g, {s0}, 0.0, base.lam, 0, inc, nodes);
        });
        st.nodes += nodes.load();
        st.tasks_skipped += st_tasks_skipped.load();
        st_tasks_skipped = 0;
    }
    void explore_path(const Graph& g, std::vector<int> P, double L, const std::vector<double>& warm, int depth,
                      Incumbent& inc, std::atomic<long long>& nodes) {
        if (ctl.stop) return;
        int vk = P.back();
        if (P.size() >= 2) inc.offer_lazy(L, [&] { return P; });
        std::vector<char> gone(g.n, 0);
        for (size_t i = 0; i + 1 < P.size(); ++i) gone[P[i]] = 1;
        std::vector<Edge> es;
        for (auto& e : g.edges)
            if (!gone[e.u] && !gone[e.v]) es.push_back(e);
        Graph gp(g.n, std::move(es));
        if (gp.adj[vk].empty()) return;
        LagrResult lv = subgradient_ends(gp, inc.get() - L, cfg.task_lagr_iters, vk, vk, warm.empty() ? nullptr : &warm);
        if (!gt(lv.ub + L, inc.get())) {
            ++st_tasks_skipped;
            return;
        }
        std::vector<double> lam = lv.lam;
        for (int x = 0; x < g.n; ++x)
            if (gone[x]) lam[x] = 0;
        SearchCtx cs;
        cs.init(gp);
        cs.set_lambda(lam);
        bool on = (lv.ub + L) / std::max(inc.get(), 1e-12) <= cfg.lagr_ratio;
        cs.use_lagr = cs.use_1forest = on;
        Incumbent loc;
        loc.best.store(inc.get() - L);
        auto publish = [&] {
            if (loc.path.empty() || !gt(loc.get() + L, inc.get())) return;
            auto c = loc.path;
            if (c.front() != vk) std::reverse(c.begin(), c.end());
            std::vector<int> full(P.begin(), P.end() - 1);
            full.insert(full.end(), c.begin(), c.end());
            inc.offer(loc.get() + L, full);
        };
        if (depth < cfg.task_depth) {
            Searcher S(cs, loc, ctl, nullptr);
            S.budget = cfg.task_budget;
            bool done = true;
            try {
                S.run_path(vk, -1);
            } catch (const Budget&) {
                done = false;
            }
            nodes += S.nodes;
            publish();
            if (done) return;
            std::vector<std::pair<double, int>> kids;
            for (auto [x, e] : gp.adj[vk]) kids.push_back({gp.edges[e].w - lv.lam[x], x});
            std::stable_sort(kids.rbegin(), kids.rend());
            for (auto [key, x] : kids) {
                auto Q = P;
                Q.push_back(x);
                explore_path(g, Q, L + g.weight(vk, x), lv.lam, depth + 1, inc, nodes);
            }
            return;
        }
        Memo memo(memo_cap(cs.W));
        Searcher S(cs, loc, ctl, &memo);
        S.run_path(vk, -1);
        nodes += S.nodes;
        publish();
    }

    // Cycles through s (vertices >= s) that start with the fixed prefix P = s, v1, .., vk.
    // Bound: len(P) + a vk..s path in what is left, with multipliers re-tuned for exactly that.
    // A short search with a node budget runs first; if it does not finish, the prefix is extended
    // by one step and each child gets its own multipliers (Lagrangian branching).
    void explore_cycle(const Graph& g, int s, std::vector<int> P, double L, const std::vector<double>& warm,
                       int depth, Incumbent& inc, std::atomic<long long>& nodes) {
        if (ctl.stop) return;
        int vk = P.back();
        std::vector<char> interior(g.n, 0);
        for (size_t i = 1; i + 1 < P.size(); ++i) interior[P[i]] = 1;
        if (P.size() >= 3) {  // close the prefix itself
            double w = g.weight(vk, s);
            if (!std::isnan(w))
                inc.offer_lazy(L + w, [&] {
                    auto c = P;
                    c.push_back(s);
                    return c;
                });
        }
        std::vector<Edge> es;
        for (auto& e : g.edges) {
            if (e.u < s || e.v < s || interior[e.u] || interior[e.v]) continue;
            if ((e.u == vk && e.v == s) || (e.u == s && e.v == vk)) continue;
            es.push_back(e);
        }
        Graph gp(g.n, es);
        LagrResult lv = subgradient_ends(gp, inc.get() - L, cfg.task_lagr_iters, vk, s, &warm);
        if (!gt(lv.ub + L, inc.get())) {
            ++st_tasks_skipped;
            return;
        }
        // the search graph: what is left plus one virtual edge s - vk worth the whole prefix
        es.push_back({s, vk, L});
        Graph gv(g.n, std::move(es));
        std::vector<double> lam = lv.lam;
        for (int x = 0; x < g.n; ++x)
            if (x < s || interior[x]) lam[x] = 0;
        SearchCtx cs;
        cs.init(gv);
        cs.set_lambda(lam);
        bool on = (lv.ub + L) / std::max(inc.get(), 1e-12) <= cfg.lagr_ratio;
        cs.use_lagr = cs.use_1forest = on;
        Incumbent loc;
        loc.best.store(inc.get());
        auto publish = [&] {
            if (loc.path.empty() || !gt(loc.get(), inc.get())) return;
            auto c = loc.path;  // s, vk, ..., s (the first step is the virtual edge)
            if (c.size() >= 2 && c[1] != vk) std::reverse(c.begin(), c.end());
            std::vector<int> full = P;
            full.insert(full.end(), c.begin() + 2, c.end());
            inc.offer(loc.get(), full);
        };
        if (depth < cfg.task_depth) {
            Searcher S(cs, loc, ctl, nullptr);  // budgeted run: no memo (a cut run would leave bad entries)
            S.budget = cfg.task_budget;
            bool done = true;
            try {
                S.run_cycle(s, vk);
            } catch (const Budget&) {
                done = false;
            }
            nodes += S.nodes;
            publish();
            if (done) return;
            std::vector<std::pair<double, int>> kids;
            for (auto [x, e] : gp.adj[vk])
                if (x != s) kids.push_back({gp.edges[e].w - lv.lam[x], x});
            std::stable_sort(kids.rbegin(), kids.rend());
            for (auto [key, x] : kids) {
                auto Q = P;
                Q.push_back(x);
                explore_cycle(g, s, Q, L + g.weight(vk, x), lv.lam, depth + 1, inc, nodes);
            }
            return;
        }
        Memo memo(memo_cap(cs.W));
        Searcher S(cs, loc, ctl, &memo);
        S.run_cycle(s, vk);
        nodes += S.nodes;
        publish();
    }

    // ------------------------------------------------------------ decomposition
    // Blocks are processed leaves-first on the block-cut tree. For an articulation a, down(a) is the
    // best path from a into a's subtree. A small block uses exact bitDP tables. A big block needs only
    // three branch-and-bound runs whatever its number of articulations, because the subtrees enter as
    // end bonuses: (1) the best path topped at the block, worth len + down(x) + down(y), pruned
    // against the best answer so far; (2) the best cycle; (3) via(B,p), the best path from the parent
    // articulation p into the block (exact).
    struct Via {
        double v = 0;
        std::vector<int> path;  // component coordinates, starts at p
        int ext = -1;           // end vertex whose subtree extends the path (-1: none)
    };

    // Bookkeeping: arrays indexed by vertex / block; the answer path is built once at the end (a
    // closure per improvement, not a path); the whole-component heuristic runs only when a block
    // needs branch and bound (or a time limit asks for an early answer).
    Sol solve_component(const Graph& g, const std::vector<int>& to_global) {
        Blocks B = biconnected(g);
        st.blocks += (int)B.verts.size();
        int mb = 0;
        for (auto& b : B.verts) mb = std::max(mb, (int)b.size());
        st.max_block = std::max(st.max_block, mb);
        if (cfg.threads_auto) cfg.threads = mb >= cfg.par_min_block ? user_threads : 1;
        st.threads_used = std::max(st.threads_used, cfg.threads);
        if (B.verts.size() == 1) return solve_whole(g, to_global);

        double bestv = -1;
        std::function<std::vector<int>()> bestmk;
        auto better = [&](double v) { return bestv < 0 || gt(v, bestv); };
        bool heur_done = false;
        auto ensure_heur = [&] {
            if (heur_done) return;
            heur_done = true;
            Sol h = polish(g, initial(g), nullptr);
            if (better(h.len)) bestv = h.len, bestmk = [hp = h.path] { return hp; };
            offer_global(h, to_global);
        };
        if (ctl.has_deadline) ensure_heur();
        int nb = (int)B.verts.size();
        std::vector<std::vector<int>> blocks_of(g.n);
        for (int i = 0; i < nb; ++i)
            for (int v : B.verts[i])
                if (B.is_art[v]) blocks_of[v].push_back(i);
        int root = 0;
        for (int i = 1; i < nb; ++i)
            if (B.verts[i].size() > B.verts[root].size()) root = i;
        std::vector<int> parent_art(nb, -2), order;
        std::vector<std::vector<int>> child_blocks(g.n);
        std::vector<std::vector<int>> child_arts(nb);
        parent_art[root] = -1;
        std::vector<int> stck{root};
        while (!stck.empty()) {
            int b = stck.back();
            stck.pop_back();
            order.push_back(b);
            for (int a : B.verts[b])
                if (B.is_art[a] && a != parent_art[b]) {
                    child_arts[b].push_back(a);
                    for (int c : blocks_of[a])
                        if (c != b) {
                            child_blocks[a].push_back(c);
                            parent_art[c] = a;
                            stck.push_back(c);
                        }
                }
        }
        std::vector<std::pair<double, int>> down(g.n, {0.0, -1});  // art -> (value, child block or -1)
        std::vector<Via> via(nb);                                   // block -> path from its parent art
        // expansions, iterative: down(a) follows via paths block by block
        auto exp_down = [&](int a) {
            std::vector<int> r{a};
            for (int cur = a;;) {
                int cb = down[cur].second;
                if (cb < 0) break;
                const Via& x = via[cb];
                r.insert(r.end(), x.path.begin() + 1, x.path.end());
                if (x.ext < 0) break;
                cur = x.ext;
            }
            return r;
        };
        auto exp_via = [&](int cb) {
            const Via& x = via[cb];
            std::vector<int> r = x.path;
            if (x.ext >= 0) {
                auto d = exp_down(x.ext);
                r.insert(r.end(), d.begin() + 1, d.end());
            }
            return r;
        };
        auto rev = [](std::vector<int> x) {
            std::reverse(x.begin(), x.end());
            return x;
        };
        auto cat = [](std::vector<int> x, const std::vector<int>& y) {
            x.insert(x.end(), y.begin() + 1, y.end());
            return x;
        };
        std::vector<int> loc(g.n, -1);

        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if (ctl.stop) throw Timeout{};
            int b = *it;
            const auto& vs = B.verts[b];
            int p = parent_art[b];
            // down values of this block's child articulations (their child blocks are done)
            for (int a : child_arts[b]) {
                std::vector<std::pair<double, int>> vl;
                for (int c : child_blocks[a]) vl.push_back({via[c].v, c});
                std::stable_sort(vl.begin(), vl.end(), [](auto& x, auto& y) { return x.first > y.first; });
                down[a] = vl[0];
                if (vl.size() >= 2) {
                    int b1 = vl[0].second, b2 = vl[1].second;
                    if (better(vl[0].first + vl[1].first))
                        bestv = vl[0].first + vl[1].first,
                        bestmk = [=, &exp_via] { return cat(rev(exp_via(b1)), exp_via(b2)); };
                } else {
                    int b1 = vl[0].second;
                    if (better(vl[0].first)) bestv = vl[0].first, bestmk = [=, &exp_via] { return exp_via(b1); };
                }
            }
            Graph lg = g.induced(vs);
            auto G = [&](const std::vector<int>& lp_) { return mapv(lp_, vs); };
            for (int i = 0; i < (int)vs.size(); ++i) loc[vs[i]] = i;

            if (lg.n <= std::min(cfg.t_dp, 20)) {
                ++st.bitdp_blocks;
                std::vector<int> terms;
                for (int i = 0; i < (int)vs.size(); ++i)
                    if (B.is_art[vs[i]]) terms.push_back(i);
                Tables T;
                if (lg.edges.size() == (size_t)lg.n) T = cycle_tables(lg, terms);  // a single cycle
                else if (!(lg.n <= cfg.enum_n && (int)lg.edges.size() <= lg.n + cfg.enum_extra &&
                           enum_tables(lg, terms, T, cfg.enum_budget)))
                    T = BitDP(lg).tables(terms);
                if (better(T.free.len)) bestv = T.free.len, bestmk = [q = G(T.free.path)] { return q; };
                if (T.cycle.len >= 0 && better(T.cycle.len))
                    bestv = T.cycle.len, bestmk = [q = G(T.cycle.path)] { return q; };
                for (int a : child_arts[b]) {
                    const Sol& fa = T.frm.at(loc[a]);
                    if (better(fa.len + down[a].first))
                        bestv = fa.len + down[a].first,
                        bestmk = [=, q = G(fa.path), &exp_down] { return cat(rev(q), exp_down(a)); };
                    for (int c : child_arts[b])
                        if (a < c) {
                            const Sol& pc = T.pr.at({loc[a], loc[c]});
                            double v = down[a].first + pc.len + down[c].first;
                            if (better(v))
                                bestv = v, bestmk = [=, q = G(pc.path), &exp_down] {
                                    return cat(cat(rev(exp_down(a)), q), exp_down(c));
                                };
                        }
                }
                if (p >= 0) {
                    Via v{T.frm.at(loc[p]).len, G(T.frm.at(loc[p]).path), -1};
                    for (int c : child_arts[b]) {
                        double x = T.pr.at({loc[p], loc[c]}).len + down[c].first;
                        if (gt(x, v.v)) v = {x, G(T.pr.at({loc[p], loc[c]}).path), c};
                    }
                    via[b] = v;
                }
                for (int v : vs) loc[v] = -1;
                continue;
            }

            std::vector<double> bonus(lg.n, 0.0);
            for (int a : child_arts[b]) bonus[loc[a]] = down[a].first;
            {  // thin big block: frontier DP for the same three answers
                FrontierOrder o;
                if (use_fdp(lg, o)) {
                    double lb = o.width >= cfg.fdp_prune_width ? bestv : -1.0;
                    LagrResult lc_;
                    if (lb >= 0) lc_ = subgradient_cycle(lg, lb, cfg.lagr_iters);
                    // (path multipliers ignore the end bonuses, so the path task uses the plain bound)
                    long long beam = lb >= 0 ? cfg.fdp_beam : 0;
                    std::vector<FdpTask> T{{0, &bonus, -1, lb, nullptr, beam, {}},
                                           {1, nullptr, -1, lb, lb >= 0 ? &lc_.lam : nullptr, beam, {}}};
                    if (p >= 0) T.push_back({0, &bonus, loc[p], -1.0, nullptr, 0, {}});
                    if (run_fdp(lg, o, T)) {
                        Sol tp = T[0].out, tc = T[1].out, tv = p >= 0 ? T[2].out : Sol{};
                        ++st.fdp_blocks;
                        st.fdp_width = std::max(st.fdp_width, o.width);
                        if (!tp.path.empty()) {
                            auto lp_ = tp.path;
                            int x = vs[lp_.front()], y = vs[lp_.back()];
                            bool ex = bonus[lp_.front()] > 0, ey = bonus[lp_.back()] > 0;
                            if (better(tp.len))
                                bestv = tp.len, bestmk = [=, q = G(lp_), &exp_down] {
                                    std::vector<int> r = ex ? rev(exp_down(x)) : std::vector<int>{x};
                                    r = cat(r, q);
                                    if (ey) r = cat(r, exp_down(y));
                                    return r;
                                };
                        }
                        if (!tc.path.empty() && better(tc.len)) bestv = tc.len, bestmk = [q = G(tc.path)] { return q; };
                        if (p >= 0) {
                            auto lp_ = tv.path;
                            bool ext = lp_.size() > 1 && bonus[lp_.back()] > 0;
                            via[b] = Via{tv.len, G(lp_), ext ? vs[lp_.back()] : -1};
                        }
                        for (int v : vs) loc[v] = -1;
                        continue;
                    }
                    ++st.fdp_fallback;
                }
            }
            // LP branch and cut for the same answers; whatever it proves is not searched again
            bool done_top = false, done_via = p < 0;
            if (use_lp(lg)) {
                {  // (1) + (2): best path with end bonuses or cycle, only if better than the answer so far
                    Incumbent inc;
                    inc.best.store(bestv);
                    auto r = run_lp_bnc(cfg.lp_revised, lg, &bonus, -1, inc, ctl, lp_opts());
                    st.lp_nodes += r.nodes, ++st.lp_runs, st.lp_chains += r.chains;
                    if (ctl.stop) throw Timeout{};
                    if (r.proved) {
                        done_top = true;
                        if (!inc.path.empty() && better(inc.get())) {
                            auto lp_ = inc.path;
                            if (lp_.size() > 2 && lp_.front() == lp_.back()) {
                                bestv = inc.get(), bestmk = [q = G(lp_)] { return q; };
                            } else {
                                int x = vs[lp_.front()], y = vs[lp_.back()];
                                bool ex = bonus[lp_.front()] > 0, ey = bonus[lp_.back()] > 0;
                                bestv = inc.get(), bestmk = [=, q = G(lp_), &exp_down] {
                                    std::vector<int> r2 = ex ? rev(exp_down(x)) : std::vector<int>{x};
                                    r2 = cat(r2, q);
                                    if (ey) r2 = cat(r2, exp_down(y));
                                    return r2;
                                };
                            }
                        }
                    } else
                        ++st.lp_fallback;
                }
                if (p >= 0) {  // (3): exact best path from the parent articulation
                    Incumbent inc;
                    int pl = loc[p];
                    for (auto [v, e] : lg.adj[pl]) {  // seed: the heaviest single step (any valid path will do)
                        double val = lg.edges[e].w + bonus[v];
                        inc.offer(val, std::vector<int>{pl, v});
                    }
                    auto r = run_lp_bnc(cfg.lp_revised, lg, &bonus, pl, inc, ctl, lp_opts());
                    st.lp_nodes += r.nodes, ++st.lp_runs, st.lp_chains += r.chains;
                    if (ctl.stop) throw Timeout{};
                    if (r.proved && !inc.path.empty()) {
                        done_via = true;
                        auto lp_ = inc.path;
                        if (lp_.front() != pl) std::reverse(lp_.begin(), lp_.end());
                        bool ext = lp_.size() > 1 && bonus[lp_.back()] > 0;
                        via[b] = Via{inc.get(), G(lp_), ext ? vs[lp_.back()] : -1};
                    } else
                        ++st.lp_fallback;
                }
                if (done_top && done_via) {
                    for (int v : vs) loc[v] = -1;
                    continue;
                }
            }
            // big block: three searches with end bonuses
            ++st.bnb_blocks;
            ensure_heur();
            Sol hb = initial(lg);
            LagrResult lr = subgradient(lg, hb.len, cfg.lagr_iters);
            SearchCtx ctx;
            ctx.init(lg);
            ctx.set_lambda(lr.lam);
            bool lagr = lr.ub / std::max(hb.len, 1e-12) <= cfg.lagr_ratio;
            if (!lagr) ++st.lagr_off;
            ctx.use_lagr = ctx.use_1forest = lagr;
            auto search = [&](Incumbent& inc, int mode, int start) {  // 0 path, 1 cycle, 2 from start
                if (mode == 1) {
                    run_cycles_task(lg, ctx, inc);
                    if (ctl.stop) throw Timeout{};
                    return;
                }
                Memo memo(memo_cap(ctx.W));
                int T = (mode == 2) ? 1 : std::max(1, cfg.threads);
                std::vector<std::unique_ptr<Searcher>> S;
                for (int t = 0; t < T; ++t) {
                    S.emplace_back(std::make_unique<Searcher>(ctx, inc, ctl, &memo));
                    S.back()->bonus = &bonus;
                }
                if (mode == 2) {
                    try {
                        S[0]->run_path(start, -1);
                    } catch (const Timeout&) {
                        ctl.stop = true;
                    }
                } else {
                    parallel_for(lg.n, [&](int v, int t) { S[t]->run_path(v, -1); });
                }
                for (auto& x : S) st.nodes += x->nodes;
                if (ctl.stop) throw Timeout{};
            };
            if (!done_top) {  // (1) best path topped at this block, against the best answer so far
                Incumbent inc;
                inc.best.store(bestv);
                // bonus-aware seed: without it the search starts far below the answer and spends
                // most of its time finding a good path rather than proving it
                Sol hs = beam_search(lg, 64, 0.0, &bonus);  // look-ahead off: it pulls ends away from bonuses
                {  // fixed-start beams from the largest bonuses
                    std::vector<int> ord(lg.n);
                    std::iota(ord.begin(), ord.end(), 0);
                    std::stable_sort(ord.begin(), ord.end(), [&](int a, int c) { return bonus[a] > bonus[c]; });
                    for (int i = 0; i < std::min(lg.n, 8) && bonus[ord[i]] > 0; ++i) {
                        Sol bf = beam_from(lg, ord[i], bonus);
                        if (gt(bf.len, hs.len)) hs = bf;
                    }
                    // pairs of large bonuses joined by a long path (both ends fixed)
                    int M = std::min(lg.n, 6);
                    for (int i = 0; i < M; ++i)
                        for (int j = i + 1; j < M; ++j) {
                            int x = ord[i], y = ord[j];
                            if (bonus[x] <= 0) continue;
                            Sol gb = grow_between(lg, x, y);
                            if (gb.path.size() < 2) continue;
                            gb.len += bonus[x] + bonus[y];
                            if (gt(gb.len, hs.len)) hs = gb;
                        }
                }
                if (hs.path.size() >= 2) {
                    inc.offer(hs.len, hs.path);
                    Sol hi = improve_bonus(lg, hs.path, bonus);
                    inc.offer(hi.len, hi.path);
                }
                search(inc, 0, -1);
                if (!inc.path.empty()) {
                    auto lp_ = inc.path;
                    int x = vs[lp_.front()], y = vs[lp_.back()];
                    bool ex = bonus[lp_.front()] > 0, ey = lp_.size() > 1 && bonus[lp_.back()] > 0;
                    if (better(inc.get()))
                        bestv = inc.get(), bestmk = [=, q = G(lp_), &exp_down] {
                            // only child articulations carry a bonus, so only they are extended
                            std::vector<int> r = ex ? rev(exp_down(x)) : std::vector<int>{x};
                            r = cat(r, q);
                            if (ey) r = cat(r, exp_down(y));
                            return r;
                        };
                }
            }
            if (!done_top) {  // (2) best cycle inside the block
                Incumbent inc;
                inc.best.store(bestv);
                search(inc, 1, -1);
                if (!inc.path.empty() && better(inc.get())) bestv = inc.get(), bestmk = [q = G(inc.path)] { return q; };
            }
            if (!done_via) {  // (3) via: from the parent articulation, exact
                Incumbent inc;
                search(inc, 2, loc[p]);
                auto lp_ = inc.path;
                int y = vs[lp_.back()];
                bool ext = lp_.size() > 1 && bonus[lp_.back()] > 0;
                via[b] = Via{inc.get(), G(lp_), ext ? y : -1};
            }
            for (int v : vs) loc[v] = -1;
        }
        Sol res{bestv, bestmk ? bestmk() : std::vector<int>{0}};
        offer_global(res, to_global);
        return res;
    }
};

}  // namespace lp
