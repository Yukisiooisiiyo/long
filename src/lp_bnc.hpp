// LP-based branch and cut for dense blocks.
//
// The longest path or cycle is written as one cycle problem on V + {z}: z is joined to every vertex
// by an edge worth that vertex's end bonus (0 when there is none). A cycle through z is a path whose
// ends are z's two neighbours; a cycle avoiding z is a cycle of the graph. Variables x_e, y_v in
// [0, 1]; degree rows x(delta(v)) = 2 y_v; generalized subtour cuts x(delta(S)) >= 2 (y_i + y_j - 1)
// for i in S, j outside, separated with Gomory-Hu (Gusfield) min cuts; blossom cuts. Branching on
// fractional variables. Nodes are taken best bound first, diving into the 1-branch while its bound
// stays within a small fraction of the best open bound: depth first spent most of its time in a
// wrong first branch when the incumbent was weak, and its node count swung with tie-breaking.
//
// Exactness does not depend on the simplex being right. A node is pruned only by a bound computed
// from the final duals in the safe form  pi^T b + sum_j max over [l_j, u_j] of (c_j - pi^T a_j) x_j,
// which is an upper bound for ANY pi with the right signs; infeasibility needs a checked Farkas row.
// Solutions are rebuilt from the graph and measured there. Anything unexpected (iteration limit,
// node budget, unproved infeasibility) returns "not proved" and the caller falls back to branch and
// bound with the improved incumbent.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "graph.hpp"
#include "heuristics.hpp"
#include "revised.hpp"
#include "search.hpp"

namespace lp {

class BoundedSimplex {
   public:
    // maximize c^T x, rows: sum a_rj x_j (= or <=) b_r, l <= x <= u. Columns get added up front;
    // rows can be appended later (cuts).
    struct SparseCol {
        std::vector<std::pair<int, double>> e;  // (row, coefficient)
    };
    int R = 0, C = 0;      // rows, columns (structural + one identity column per row)
    int S = 0;             // structural columns
    std::vector<double> c, l, u, b;
    std::vector<char> is_eq;
    std::vector<std::vector<double>> T;  // tableau rows: B^-1 [A | b]; last entry = rhs
    std::vector<double> d;               // reduced costs (c_j - pi^T a_j); d[C] = objective
    std::vector<int> basic;              // row -> column
    std::vector<int> pos;                // column -> row or -1
    std::vector<double> xn;              // value of nonbasic columns (at a bound)
    std::vector<int> idcol;              // row -> its identity column (slack / artificial)
    std::vector<std::vector<std::pair<int, double>>> rowsA;  // original rows over structural columns
    long long iters = 0, iter_cap = 200000;
    Control* ctl = nullptr;  // time limit: checked every 64 pivots

    void init(int nstruct, const std::vector<double>& cost) {
        S = C = nstruct;
        c = cost;
        l.assign(C, 0.0), u.assign(C, 1.0);
        xn.assign(C, 0.0);
        pos.assign(C, -1);
        d = c;
        d.push_back(0.0);
    }
    // append a row over structural columns; its identity column (slack for <=, fixed artificial for =)
    // becomes basic. Valid at any time (the row is expressed in the current basis).
    void add_row(const std::vector<std::pair<int, double>>& a, double rhs, bool eq) {
        int col = C++;
        for (auto& row : T) row.insert(row.end() - 1, 0.0);
        c.push_back(0.0), l.push_back(0.0), u.push_back(eq ? 0.0 : kBig), xn.push_back(0.0), pos.push_back(R);
        d.insert(d.end() - 1, 0.0);
        std::vector<double> row(C + 1, 0.0);
        for (auto [j, v] : a) row[j] += v;
        row[col] = 1.0;
        row[C] = rhs;
        // eliminate basic columns
        for (int r = 0; r < R; ++r) {
            double f = row[basic[r]];
            if (f != 0.0)
                for (int j = 0; j <= C; ++j) row[j] -= f * T[r][j];
        }
        T.push_back(std::move(row));
        basic.push_back(col);
        idcol.push_back(col);
        b.push_back(rhs);
        is_eq.push_back(eq);
        rowsA.push_back(a);
        ++R;
    }
    // Remove rows whose own slack is basic (inactive cuts). Their slack column is a unit vector in
    // that row only, so the rest of the tableau is unchanged. Returns old row -> new row (-1: gone).
    std::vector<int> drop_rows(const std::vector<char>& drop) {
        std::vector<char> dropcol(C, 0);
        for (int r = 0; r < R; ++r)
            if (drop[r]) dropcol[idcol[r]] = 1;
        std::vector<int> colmap(C, -1), rowmap(R, -1);
        int nc = 0, nr = 0;
        for (int j = 0; j < C; ++j)
            if (!dropcol[j]) colmap[j] = nc++;
        for (int r = 0; r < R; ++r)
            if (!drop[r]) rowmap[r] = nr++;
        auto squeeze = [&](const std::vector<double>& v) {  // length C (+1 tail kept)
            std::vector<double> o;
            o.reserve(nc + 1);
            for (int j = 0; j < C; ++j)
                if (!dropcol[j]) o.push_back(v[j]);
            if ((int)v.size() > C) o.push_back(v[C]);
            return o;
        };
        std::vector<std::vector<double>> T2;
        T2.reserve(nr);
        std::vector<int> basic2, idcol2;
        std::vector<double> b2;
        std::vector<char> eq2;
        std::vector<std::vector<std::pair<int, double>>> rowsA2;
        for (int r = 0; r < R; ++r) {
            if (drop[r]) continue;
            T2.push_back(squeeze(T[r]));
            basic2.push_back(colmap[basic[r]]);
            idcol2.push_back(colmap[idcol[r]]);
            b2.push_back(b[r]);
            eq2.push_back(is_eq[r]);
            rowsA2.push_back(std::move(rowsA[r]));
        }
        d = squeeze(d);
        c = squeeze(c), l = squeeze(l), u = squeeze(u), xn = squeeze(xn);
        T.swap(T2), basic.swap(basic2), idcol.swap(idcol2), b.swap(b2), is_eq.swap(eq2), rowsA.swap(rowsA2);
        R = nr, C = nc;
        pos.assign(C, -1);
        for (int r = 0; r < R; ++r) pos[basic[r]] = r;
        return rowmap;
    }
    double xval(int j) const { return pos[j] >= 0 ? bval(pos[j]) : xn[j]; }
    // values of all basic variables: rhs minus the nonbasic columns that sit at a nonzero bound
    void basic_values(std::vector<double>& xb) const {
        std::vector<int> nz;
        for (int j = 0; j < C; ++j)
            if (pos[j] < 0 && xn[j] != 0.0) nz.push_back(j);
        xb.resize(R);
        for (int r = 0; r < R; ++r) {
            const std::vector<double>& tr = T[r];
            double v = tr[C];
            for (int j : nz) v -= tr[j] * xn[j];
            xb[r] = v;
        }
    }
    double bval(int r) const {
        double v = T[r][C];
        for (int j = 0; j < C; ++j)
            if (pos[j] < 0 && xn[j] != 0.0) v -= T[r][j] * xn[j];
        return v;
    }
    void set_bounds(int j, double lo, double hi) {
        l[j] = lo, u[j] = hi;
        if (pos[j] < 0) xn[j] = d[j] > 0 ? hi : lo;  // the bound that keeps the basis dual feasible
    }

    // returns 0 optimal, 1 infeasible (candidate; check with farkas_ok), 2 iteration limit
    int solve() {
        // dual phase first if some basic variable violates its bounds, then primal
        for (;;) {
            int st = dual();
            if (st != 0) return st;
            int ps = primal();
            if (ps == 3) continue;  // primal made a basic infeasible (numerics): redo dual
            return ps;
        }
    }
    int infeasible_row = -1;

    // multipliers pi_r = -d[idcol r]; signs fixed for <= rows (pi >= 0)
    std::vector<double> duals() const {
        std::vector<double> pi(R);
        for (int r = 0; r < R; ++r) {
            pi[r] = -d[idcol[r]];
            if (!is_eq[r]) pi[r] = std::max(0.0, pi[r]);
        }
        return pi;
    }
    // safe upper bound for the current bounds, valid for any pi with these signs
    double safe_bound(const std::vector<std::vector<std::pair<int, double>>>& cols) const {
        auto pi = duals();
        double v = 0;
        for (int r = 0; r < R; ++r) v += pi[r] * b[r];
        for (int j = 0; j < S; ++j) {
            double rc = c[j];
            for (auto [r, a] : cols[j])
                if (r < R) rc -= pi[r] * a;
            v += std::max(rc * l[j], rc * u[j]);
        }
        return v;
    }
    // Farkas check of the infeasible row: the combination y = row of B^-1 must have no solution in
    // the bounds (slacks >= 0, artificials = 0)
    bool farkas_ok(const std::vector<std::vector<std::pair<int, double>>>& cols) const {
        if (infeasible_row < 0) return false;
        int rr = infeasible_row;
        std::vector<double> y(R);
        for (int r = 0; r < R; ++r) y[r] = T[rr][idcol[r]];
        double lo = 0, hi = 0;
        for (int j = 0; j < S; ++j) {
            double a = 0;
            for (auto [r, v] : cols[j])
                if (r < R) a += y[r] * v;
            lo += std::min(a * l[j], a * u[j]);
            hi += std::max(a * l[j], a * u[j]);
        }
        for (int r = 0; r < R; ++r) {  // identity columns: slack in [0, inf), artificial fixed 0
            if (is_eq[r]) continue;
            if (y[r] > 0) hi = INFINITY;
            else if (y[r] < 0) lo = -INFINITY;
        }
        double rhs = 0;
        for (int r = 0; r < R; ++r) rhs += y[r] * b[r];
        double tol = 1e-7 * (1 + std::fabs(rhs));
        return rhs > hi + tol || rhs < lo - tol;
    }

   private:
    static constexpr double kBig = 1e30, kTol = 1e-9, kPiv = 1e-9;
    std::vector<int> nzj;
    std::vector<double> xb_;

    void pivot(int r, int q) {
        std::vector<double>& pr = T[r];
        double p = pr[q];
        nzj.clear();
        for (int j = 0; j <= C; ++j)
            if (pr[j] != 0.0) pr[j] /= p, nzj.push_back(j);  // the pivot row is often sparse
        for (int i = 0; i < R; ++i) {
            if (i == r) continue;
            double f = T[i][q];
            if (f == 0.0) continue;
            std::vector<double>& ti = T[i];
            for (int j : nzj) ti[j] -= f * pr[j];
            ti[q] = 0.0;
        }
        double f = d[q];
        if (f != 0.0) {
            for (int j : nzj) d[j] -= f * pr[j];
            d[q] = 0.0;
        }
        int out = basic[r];
        pos[out] = -1;
        basic[r] = q;
        pos[q] = r;
        if ((++iters & 63) == 0 && ctl) ctl->check();
    }

    int dual() {
        for (;;) {
            if (iters > iter_cap) return 2;
            int r = -1;
            double worst = kTol, val = 0;
            basic_values(xb_);
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                double x = xb_[i];
                double viol = std::max(l[j] - x, x - u[j]);
                if (viol > worst) worst = viol, r = i, val = x;
            }
            if (r < 0) return 0;
            int jb = basic[r];
            bool low = val < l[jb];
            int q = -1;
            double best = INFINITY;
            for (int j = 0; j < C; ++j) {
                if (pos[j] >= 0 || l[j] == u[j]) continue;
                double a = T[r][j];
                if (std::fabs(a) < kPiv) continue;
                bool atlow = xn[j] == l[j];
                // need x_B to move up (low) or down; x_B changes by -a * dx_j
                bool ok = low ? ((atlow && a < 0) || (!atlow && a > 0)) : ((atlow && a > 0) || (!atlow && a < 0));
                if (!ok) continue;
                double ratio = std::fabs(d[j] / a);
                if (ratio < best - 1e-12 || (ratio < best + 1e-12 && q >= 0 && std::fabs(a) > std::fabs(T[r][q])))
                    best = ratio, q = j;
            }
            if (q < 0) {
                infeasible_row = r;
                return 1;
            }
            // the leaving variable goes to the violated bound
            double target = low ? l[jb] : u[jb];
            pivot(r, q);
            xn[jb] = target;
        }
    }

    int primal() {
        int degenerate = 0;
        for (;;) {
            if (iters > iter_cap) return 2;
            bool bland = degenerate > 50;
            int q = -1;
            double bestd = kTol;
            for (int j = 0; j < C; ++j) {
                if (pos[j] >= 0 || l[j] == u[j]) continue;
                bool atlow = xn[j] == l[j];
                double gain = atlow ? d[j] : -d[j];
                if (gain > bestd) {
                    q = j;
                    if (bland) break;
                    bestd = gain;
                }
            }
            if (q < 0) return 0;
            double dir = xn[q] == l[q] ? 1.0 : -1.0;
            double theta = u[q] - l[q];  // bound flip
            int r = -1;
            double rtarget = 0;
            basic_values(xb_);
            for (int i = 0; i < R; ++i) {
                double a = T[i][q] * dir;  // x_Bi changes by -a * theta
                if (std::fabs(a) < kPiv) continue;
                int j = basic[i];
                double x = xb_[i];
                double lim;
                double tgt;
                if (a > 0) lim = (x - l[j]) / a, tgt = l[j];
                else lim = (u[j] >= kBig / 2) ? INFINITY : (x - u[j]) / a, tgt = u[j];
                if (lim < 0) lim = 0;
                if (lim < theta - 1e-12 || (r >= 0 && lim < theta + 1e-12 && bland && basic[i] < basic[r])) {
                    theta = lim, r = i, rtarget = tgt;
                }
            }
            if (theta >= kBig / 2) return 2;  // unbounded: cannot happen with bounded columns
            degenerate = theta < 1e-12 ? degenerate + 1 : 0;
            if (r < 0) {  // bound flip
                xn[q] = dir > 0 ? u[q] : l[q];
                continue;
            }
            int jb = basic[r];
            pivot(r, q);
            xn[jb] = rtarget;
            // check: a basic variable outside its bounds after the step means numerics: go dual
            basic_values(xb_);
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                double x = xb_[i];
                if (x < l[j] - 1e-7 || x > u[j] + 1e-7) return 3;
            }
        }
    }
};

template <class LPT>
class LPBranchCutT {
   public:
    struct Result {
        bool proved = false;  // true: no answer above `inc` exists beyond what was offered
        long long nodes = 0, lps = 0, cuts = 0;
    };

    // bonus may be null; forced >= 0: the path must have that vertex as an end (no cycles then)
    LPBranchCutT(const Graph& g, const std::vector<double>* bonus, int forced, Incumbent& inc, Control& ctl)
        : g(g), bonus(bonus), forced(forced), inc(inc), ctl(ctl) {}

    long long node_cap = 3000;
    int order = 1;         // node order: 0 depth first, 1 best first with plunging
    double plunge = 5e-4;  // dive while the bound is within this fraction of the best open bound
    std::atomic<bool>* race_stop = nullptr;  // set by the other racer when it has finished
    int cuts_per_round = 0;   // 0: all violated cuts found
    int cut_cap = 20000;  // live cuts; inactive ones are purged

    Result run() {
        Result res;
        n = g.n;
        N = n + 1;  // z = n
        // columns: graph edges, z edges, y_v (v in 0..n)
        ne = (int)g.edges.size();
        int m = ne + n;
        S = m + N;
        std::vector<double> cost(S, 0.0);
        eu.resize(m), ev.resize(m);
        for (int i = 0; i < ne; ++i) eu[i] = g.edges[i].u, ev[i] = g.edges[i].v, cost[i] = g.edges[i].w;
        for (int v = 0; v < n; ++v)
            eu[ne + v] = v, ev[ne + v] = n, cost[ne + v] = (bonus && v != forced) ? (*bonus)[v] : 0.0;
        lp.init(S, cost);
        lp.ctl = &ctl;
        cols.assign(S, {});
        for (int v = 0; v < N; ++v) {  // degree rows
            std::vector<std::pair<int, double>> a;
            for (int i = 0; i < m; ++i)
                if (eu[i] == v || ev[i] == v) a.push_back({i, 1.0});
            a.push_back({m + v, -2.0});
            add_row(a, 0.0, true);
        }
        if (forced >= 0) {  // a path from `forced`: z used, edge forced-z taken
            lp.set_bounds(m + n, 1.0, 1.0);
            lp.set_bounds(ne + forced, 1.0, 1.0);
        }
        // (a one-vertex "path" z-v-z would need edge v-z twice, which its bound forbids)
        if (order == 1) best_first(res);
        else dfs(res);
        res.lps = lps;
        res.cuts = lp.R - N;
        res.proved = proved;
        return res;
    }

   private:
    const Graph& g;
    const std::vector<double>* bonus;
    int forced;
    Incumbent& inc;
    Control& ctl;
    int n = 0, N = 0, ne = 0, S = 0;
    std::vector<int> eu, ev;
    LPT lp;
    std::vector<std::vector<std::pair<int, double>>> cols;
    long long nodes = 0, lps = 0;
    double last_ub = 0;  // bound of the node just solved
    bool proved = true;

    // drop inactive cuts (slack basic and positive) once there are many, to keep the tableau small
    void purge() {
        if (lp.R - N <= std::max(100, N)) return;
        std::vector<char> drop(lp.R, 0);
        int k = 0;
        for (int r = N; r < lp.R; ++r)
            if (lp.basic[r] == lp.idcol[r] && lp.xval(lp.idcol[r]) > 1e-6) drop[r] = 1, ++k;
        if (k == 0) return;
        auto rowmap = lp.drop_rows(drop);
        for (auto& col : cols) {
            std::vector<std::pair<int, double>> c2;
            for (auto [r, v] : col)
                if (rowmap[r] >= 0) c2.push_back({rowmap[r], v});
            col.swap(c2);
        }
        purged += k;
    }
    long long purged = 0;

    void add_row(const std::vector<std::pair<int, double>>& a, double rhs, bool eq) {
        int r = lp.R;
        lp.add_row(a, rhs, eq);
        for (auto [j, v] : a) cols[j].push_back({r, v});
    }

    // generalized subtour cuts violated by the current point
    // Blossom cuts. For any vertex set H and any odd set T of edges leaving H:
    //   x(E(H)) + x(T) - sum_{v in H} y_v <= (|T| - 1) / 2.
    // It is half the degree rows of H plus half the bounds x_e <= 1 of T, rounded down
    // (a Chvatal-Gomory cut), so it is valid whatever H and T are. Candidates: H = a connected
    // piece of the fractional edges, T = the edges leaving H with x > 1/2 (made odd if needed).
    int separate_blossoms() {
        int m = ne + n;
        std::vector<double> x(m), y(N);
        for (int i = 0; i < m; ++i) x[i] = lp.xval(i);
        for (int v = 0; v < N; ++v) y[v] = lp.xval(m + v);
        std::vector<int> comp(N, -1), par(N);
        std::iota(par.begin(), par.end(), 0);
        std::function<int(int)> fnd = [&](int a) { return par[a] == a ? a : par[a] = fnd(par[a]); };
        std::vector<char> touched(N, 0);
        for (int i = 0; i < m; ++i)
            if (x[i] > 1e-6 && x[i] < 1 - 1e-6) par[fnd(eu[i])] = fnd(ev[i]), touched[eu[i]] = touched[ev[i]] = 1;
        std::vector<std::vector<int>> groups(N);
        for (int v = 0; v < N; ++v)
            if (touched[v]) groups[fnd(v)].push_back(v);
        int added = 0;
        for (auto& H : groups) {
            if (H.size() < 3) continue;
            std::vector<char> in(N, 0);
            for (int v : H) in[v] = 1;
            double inner = 0, ysum = 0;
            std::vector<std::pair<double, int>> out;
            for (int i = 0; i < m; ++i) {
                bool a = in[eu[i]], b2 = in[ev[i]];
                if (a && b2) inner += x[i];
                else if (a != b2 && x[i] > 1e-9) out.push_back({x[i], i});
            }
            for (int v : H) ysum += y[v];
            std::stable_sort(out.rbegin(), out.rend());
            int k = 0;
            while (k < (int)out.size() && out[k].first > 0.5) ++k;
            if (k % 2 == 0) {  // make it odd: drop the weakest tooth or add the next edge, whichever is better
                if (k + 1 <= (int)out.size() && (k == 0 || out[k].first > 1 - out[k - 1].first)) ++k;
                else --k;
            }
            if (k < 1) continue;
            double tx = 0;
            for (int t = 0; t < k; ++t) tx += out[t].first;
            double viol = inner + tx - ysum - (k - 1) / 2.0;
            if (viol <= 1e-6) continue;
            std::vector<std::pair<int, double>> a;
            for (int i = 0; i < m; ++i)
                if (in[eu[i]] && in[ev[i]]) a.push_back({i, 1.0});
            for (int t = 0; t < k; ++t) a.push_back({out[t].second, 1.0});
            for (int v : H) a.push_back({m + v, -1.0});
            add_row(a, (k - 1) / 2.0, false);
            ++added, ++blossom_cuts;
        }
        return added;
    }
   public:
    bool blossoms = false;
    long long blossom_cuts = 0;
   private:

    int separate() {
        int m = ne + n;
        std::vector<double> x(m), y(N);
        for (int i = 0; i < m; ++i) x[i] = lp.xval(i);
        for (int v = 0; v < N; ++v) y[v] = lp.xval(m + v);
        // support graph (edges with x > 0) as arcs with reverse arcs, for max flows
        arc_to.clear(), arc_cap.clear(), head.assign(N, -1), nxt_arc.clear();
        auto add_arc = [&](int a, int b2, double c) {
            arc_to.push_back(b2), arc_cap.push_back(c), nxt_arc.push_back(head[a]), head[a] = (int)arc_to.size() - 1;
        };
        for (int i = 0; i < m; ++i)
            if (x[i] > 1e-9) add_arc(eu[i], ev[i], x[i]), add_arc(ev[i], eu[i], x[i]);
        std::vector<std::vector<char>> sides;
        // Gusfield: N-1 max flows give a tree of min cuts between all pairs
        std::vector<int> par(N, 0);
        std::vector<double> fl(N, 0);
        for (int s = 1; s < N; ++s) {
            int t = par[s];
            std::vector<char> side;
            if ((s & 15) == 0) ctl.check();
            double f = maxflow(s, t, side);
            fl[s] = f;
            for (int v = s + 1; v < N; ++v)
                if (side[v] && par[v] == t) par[v] = s;
            sides.push_back(side);
        }
        int added = 0;
        // candidates, most violated first; at most cuts_per_round of them
        std::vector<std::pair<double, int>> cand;
        std::vector<std::pair<int, int>> ij(N, {-1, -1});
        for (int s = 1; s < N; ++s) {
            const auto& side = sides[s - 1];  // side[v] = 1: v with s
            int bi = -1, bj = -1;
            for (int v = 0; v < N; ++v) {
                if (side[v]) {
                    if (bi < 0 || y[v] > y[bi]) bi = v;
                } else if (bj < 0 || y[v] > y[bj])
                    bj = v;
            }
            if (bi < 0 || bj < 0) continue;
            double need = 2 * (y[bi] + y[bj] - 1);
            if (fl[s] < need - 1e-6) cand.push_back({need - fl[s], s}), ij[s] = {bi, bj};
        }
        std::stable_sort(cand.rbegin(), cand.rend());
        std::vector<std::vector<char>> seen;
        for (auto [viol, s] : cand) {
            if (lp.R - N >= cut_cap || (cuts_per_round > 0 && added >= cuts_per_round)) break;
            const auto& side = sides[s - 1];
            if (std::find(seen.begin(), seen.end(), side) != seen.end()) continue;
            seen.push_back(side);
            auto [bi, bj] = ij[s];
            // x(delta(S)) - 2 y_i - 2 y_j >= -2   <=>   -x(delta(S)) + 2 y_i + 2 y_j <= 2
            std::vector<std::pair<int, double>> a;
            for (int i = 0; i < m; ++i)
                if (side[eu[i]] != side[ev[i]]) a.push_back({i, -1.0});
            a.push_back({m + bi, 2.0});
            a.push_back({m + bj, 2.0});
            add_row(a, 2.0, false);
            ++added;
        }
        return added;
    }

    // Edmonds-Karp on the sparse support graph; side = vertices reachable from s in the residual
    std::vector<int> arc_to, head, nxt_arc;
    std::vector<double> arc_cap;
    double maxflow(int s, int t, std::vector<char>& side) {
        std::vector<double> f(arc_to.size(), 0.0);  // arc 2k and 2k+1 are each other's reverse
        std::vector<int> pa(N);                     // arc used to reach a vertex
        double total = 0;
        for (;;) {
            std::fill(pa.begin(), pa.end(), -1);
            pa[s] = -2;
            std::vector<int> q{s};
            for (size_t h = 0; h < q.size() && pa[t] == -1; ++h) {
                int a = q[h];
                for (int e = head[a]; e >= 0; e = nxt_arc[e]) {
                    int b2 = arc_to[e];
                    if (pa[b2] == -1 && arc_cap[e] - f[e] > 1e-12) pa[b2] = e, q.push_back(b2);
                }
            }
            if (pa[t] == -1) break;
            double aug = INFINITY;
            for (int v = t; v != s; v = arc_to[pa[v] ^ 1]) aug = std::min(aug, arc_cap[pa[v]] - f[pa[v]]);
            for (int v = t; v != s; v = arc_to[pa[v] ^ 1]) f[pa[v]] += aug, f[pa[v] ^ 1] -= aug;
            total += aug;
        }
        side.assign(N, 0);
        for (int v = 0; v < N; ++v) side[v] = pa[v] != -1;
        return total;
    }

    // integral point with no violated cut: rebuild the cycle / path and offer it
    void take_integral() {
        int m = ne + n;
        std::vector<std::vector<int>> nb(N);
        for (int i = 0; i < m; ++i)
            if (lp.xval(i) > 0.5) nb[eu[i]].push_back(ev[i]), nb[ev[i]].push_back(eu[i]);
        int start = -1;
        for (int v = 0; v < N; ++v)
            if (!nb[v].empty()) {
                start = v;
                break;
            }
        if (start < 0) return;
        std::vector<int> cyc{start};
        for (int prev = -1, cur = start;;) {
            if (nb[cur].size() != 2) return;
            int nx = nb[cur][0] == prev ? nb[cur][1] : nb[cur][0];
            if (nx == start) break;
            if ((int)cyc.size() > N) return;
            cyc.push_back(nx);
            prev = cur, cur = nx;
        }
        // rotate so that z (if present) is first, then drop it: a path
        auto it = std::find(cyc.begin(), cyc.end(), n);
        std::vector<int> path;
        if (it != cyc.end()) {
            std::rotate(cyc.begin(), it, cyc.end());
            path.assign(cyc.begin() + 1, cyc.end());
            if (forced >= 0 && path.back() == forced) std::reverse(path.begin(), path.end());
            if (forced >= 0 && path.front() != forced) return;
        } else {
            if (forced >= 0) return;
            path = cyc;
            path.push_back(cyc[0]);
        }
        if (path.size() < 2) return;
        double L = 0;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            double w = g.weight(path[i], path[i + 1]);
            if (std::isnan(w)) return;
            L += w;
        }
        bool cycle = path.size() > 2 && path.front() == path.back();
        if (!cycle && bonus) L += (*bonus)[path.front()] + (*bonus)[path.back()];
        if (!cycle && forced >= 0 && bonus) L -= (*bonus)[forced];  // the forced end has no bonus
        inc.offer(L, path);
    }

    // Solves the current node (cut rounds). Returns the branching variable, or -1 when the node
    // is closed (bound not above the incumbent, infeasible, integral) or proving failed.
    // the branching variable among [lo, hi): the first fractional y, or the most fractional x
    int pick(int lo, int hi) const {
        int q = -1;
        double best = 1e-6;
        for (int j = lo; j < hi; ++j) {
            double x = lp.xval(j), f = std::min(x, 1 - x);
            if (f <= 1e-6) continue;
            if (lo == ne + n) return j;
            if (f > best) best = f, q = j;
        }
        return q;
    }

    int process_node() {
        ctl.check();  // throws Timeout when the time limit is reached
        if (race_stop && race_stop->load(std::memory_order_relaxed)) {
            proved = false;
            return -1;
        }
        if (++nodes > node_cap) {
            proved = false;
            return -1;
        }
        for (int round = 0;; ++round) {
            ++lps;
            int st = lp.solve();
            if (st == 2) {
                proved = false;
                return -1;
            }
            if (st == 1) {
                if (lp.farkas_ok(cols)) return -1;  // certified infeasible
                proved = false;
                return -1;
            }
            double ub = lp.safe_bound(cols);
            if (!gt(ub, inc.get())) return -1;
            last_ub = ub;
            purge();
            int added = (round < 50) ? separate() : 0;
            if (!added && blossoms && round < 50) added = separate_blossoms();
            if (added) continue;
            break;
        }
        // branching variable (y first: whether a vertex is used)
        int m = ne + n;
        int q = pick(m, S);
        if (q < 0) q = pick(0, m);
        if (q < 0) {
            take_integral();
            // the bound above was not beaten by more than tolerance only if this point is the answer;
            // re-check: if the integral point is worse than the bound (numerics), keep searching
            double ub = lp.safe_bound(cols);
            if (gt(ub, inc.get())) {
                // pick any fractional-looking variable is impossible: give up proving
                proved = false;
            }
            return -1;
        }
        return q;
    }

    void dfs(Result& res) {
        if (!proved) return;
        int q = process_node();
        res.nodes = nodes;
        if (q < 0) return;
        double lo = lp.l[q], hi = lp.u[q];
        for (double v : {1.0, 0.0}) {
            if (v < lo || v > hi) continue;
            lp.set_bounds(q, v, v);
            dfs(res);
            lp.set_bounds(q, lo, hi);
            if (!proved) return;
        }
    }

    // Best-first: always the open node with the highest bound (ties: the newest, as in a dive).
    // Each node keeps its fixings; switching nodes resets the previous fixings to the root bounds.
    void best_first(Result& res) {
        struct Node {
            double ub;
            long long seq;
            std::vector<std::pair<int, double>> fix;
        };
        auto worse = [](const Node& a, const Node& b) { return a.ub != b.ub ? a.ub < b.ub : a.seq < b.seq; };
        std::vector<double> l0(S), u0(S);
        for (int j = 0; j < S; ++j) l0[j] = lp.l[j], u0[j] = lp.u[j];
        std::vector<Node> heap;
        heap.push_back({std::numeric_limits<double>::infinity(), 0, {}});
        long long seq = 0;
        std::vector<int> cur;  // variables fixed for the node being solved
        while (!heap.empty() && proved) {
            std::pop_heap(heap.begin(), heap.end(), worse);
            Node nd = std::move(heap.back());
            heap.pop_back();
            if (!gt(nd.ub, inc.get())) continue;
            for (int j : cur) lp.set_bounds(j, l0[j], u0[j]);
            cur.clear();
            for (auto [j, v] : nd.fix) lp.set_bounds(j, v, v), cur.push_back(j);
            // plunge: keep diving into the 1-branch while its bound stays close to the best open bound
            for (;;) {
                int q = process_node();
                res.nodes = nodes;
                if (!proved || q < 0) break;
                double ub = last_ub;
                bool dive = false;
                if (plunge > 0) {
                    double top = heap.empty() ? -std::numeric_limits<double>::infinity() : heap.front().ub;
                    dive = ub >= top - plunge * std::fabs(top);
                }
                for (double v : {0.0, 1.0}) {  // pushed 0 then 1: on equal bounds the 1-branch comes first
                    if (v < lp.l[q] || v > lp.u[q]) continue;
                    if (dive && v == 1.0) continue;
                    Node ch{ub, ++seq, nd.fix};
                    ch.fix.push_back({q, v});
                    heap.push_back(std::move(ch));
                    std::push_heap(heap.begin(), heap.end(), worse);
                }
                if (!dive || 1.0 < lp.l[q] || 1.0 > lp.u[q]) break;
                lp.set_bounds(q, 1.0, 1.0), cur.push_back(q);
                nd.fix.push_back({q, 1.0});
            }
        }
        for (int j : cur) lp.set_bounds(j, l0[j], u0[j]);
    }
};

using LPResult = LPBranchCutT<BoundedSimplex>::Result;

struct LPOptions {
    long long node_cap = 3000;
    int cuts_round = 20;
    bool blossoms = true, race = false;
    int order = 1;
    double plunge = 5e-4;
};

inline std::atomic<long long> g_race_dfs_won{0};  // statistics: races won by the depth-first side

// run the branch and cut with the dense-table simplex or the revised simplex; with `race`, depth
// first and best first run in two threads on the same incumbent and the first proof wins
inline LPResult run_lp_bnc(bool revised, const Graph& g, const std::vector<double>* bonus, int forced, Incumbent& inc,
                           Control& ctl, const LPOptions& o) {
    auto setup = [&](auto& bc, int order) {
        bc.node_cap = o.node_cap;
        bc.cuts_per_round = o.cuts_round;
        bc.blossoms = o.blossoms;
        bc.order = order;
        bc.plunge = o.plunge;
    };
    auto finish = [](auto&, const auto& r) { return LPResult{r.proved, r.nodes, r.lps, r.cuts}; };
    if (o.race && revised) {
        LPBranchCutT<RevisedSimplex> a(g, bonus, forced, inc, ctl), b(g, bonus, forced, inc, ctl);
        setup(a, 0), setup(b, 1);
        std::atomic<bool> stop_a{false}, stop_b{false};
        a.race_stop = &stop_a, b.race_stop = &stop_b;
        typename LPBranchCutT<RevisedSimplex>::Result ra, rb;
        bool to_a = false, to_b = false;
        auto runner = [&](auto& bc, auto& r, std::atomic<bool>& other, bool& to) {
            try {
                r = bc.run();
            } catch (Timeout&) {
                to = true;
            }
            if (r.proved) other.store(true);
        };
        std::thread ta([&] { runner(a, ra, stop_b, to_a); });
        runner(b, rb, stop_a, to_b);
        ta.join();
        if (to_a || to_b || ctl.stop) throw Timeout{};
        if (ra.proved) {
            ++g_race_dfs_won;
            return finish(a, ra);
        }
        return finish(b, rb);
    }
    if (revised) {
        LPBranchCutT<RevisedSimplex> bc(g, bonus, forced, inc, ctl);
        setup(bc, o.order);
        auto r = bc.run();
        return finish(bc, r);
    }
    LPBranchCutT<BoundedSimplex> bc(g, bonus, forced, inc, ctl);
    setup(bc, o.order);
    auto r = bc.run();
    return finish(bc, r);
}

}  // namespace lp
