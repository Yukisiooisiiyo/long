// LP-based branch and cut for dense blocks.
//
// The longest path or cycle is written as one cycle problem on V + {z}: z is joined to every vertex
// by an edge worth that vertex's end bonus (0 when there is none). A cycle through z is a path whose
// ends are z's two neighbours; a cycle avoiding z is a cycle of the graph. Variables x_e, y_v in
// [0, 1]; degree rows x(delta(v)) = 2 y_v; generalized subtour cuts x(delta(S)) >= 2 (y_i + y_j - 1)
// for i in S, j outside, separated with Gomory-Hu (Gusfield) min cuts. Branching on fractional
// variables, depth first.
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
#include <vector>

#include "graph.hpp"
#include "heuristics.hpp"
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
    double xval(int j) const { return pos[j] >= 0 ? bval(pos[j]) : xn[j]; }
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

    void pivot(int r, int q) {
        std::vector<double>& pr = T[r];
        double p = pr[q];
        for (double& v : pr) v /= p;
        for (int i = 0; i < R; ++i) {
            if (i == r) continue;
            double f = T[i][q];
            if (f == 0.0) continue;
            std::vector<double>& ti = T[i];
            for (int j = 0; j <= C; ++j) ti[j] -= f * pr[j];
        }
        double f = d[q];
        if (f != 0.0)
            for (int j = 0; j <= C; ++j) d[j] -= f * pr[j];
        int out = basic[r];
        pos[out] = -1;
        basic[r] = q;
        pos[q] = r;
        ++iters;
    }

    int dual() {
        for (;;) {
            if (iters > iter_cap) return 2;
            int r = -1;
            double worst = kTol, val = 0;
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                double x = bval(i);
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
            for (int i = 0; i < R; ++i) {
                double a = T[i][q] * dir;  // x_Bi changes by -a * theta
                if (std::fabs(a) < kPiv) continue;
                int j = basic[i];
                double x = bval(i);
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
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                double x = bval(i);
                if (x < l[j] - 1e-7 || x > u[j] + 1e-7) return 3;
            }
        }
    }
};

class LPBranchCut {
   public:
    struct Result {
        bool proved = false;  // true: no answer above `inc` exists beyond what was offered
        long long nodes = 0, lps = 0, cuts = 0;
    };

    // bonus may be null; forced >= 0: the path must have that vertex as an end (no cycles then)
    LPBranchCut(const Graph& g, const std::vector<double>* bonus, int forced, Incumbent& inc, Control& ctl)
        : g(g), bonus(bonus), forced(forced), inc(inc), ctl(ctl) {}

    long long node_cap = 3000;
    int cut_cap = 600;

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
        dfs(res, 0);
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
    BoundedSimplex lp;
    std::vector<std::vector<std::pair<int, double>>> cols;
    long long nodes = 0, lps = 0;
    bool proved = true;

    void add_row(const std::vector<std::pair<int, double>>& a, double rhs, bool eq) {
        int r = lp.R;
        lp.add_row(a, rhs, eq);
        for (auto [j, v] : a) cols[j].push_back({r, v});
    }

    // generalized subtour cuts violated by the current point
    int separate() {
        int m = ne + n;
        std::vector<double> x(m), y(N);
        for (int i = 0; i < m; ++i) x[i] = lp.xval(i);
        for (int v = 0; v < N; ++v) y[v] = lp.xval(m + v);
        std::vector<std::vector<double>> cap(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < m; ++i)
            if (x[i] > 1e-9) cap[eu[i]][ev[i]] += x[i], cap[ev[i]][eu[i]] += x[i];
        std::vector<std::vector<char>> sides;
        // Gusfield: N-1 max flows give a tree of min cuts between all pairs
        std::vector<int> par(N, 0);
        std::vector<double> fl(N, 0);
        for (int s = 1; s < N; ++s) {
            int t = par[s];
            std::vector<char> side;
            double f = maxflow(cap, s, t, side);
            fl[s] = f;
            for (int v = s + 1; v < N; ++v)
                if (side[v] && par[v] == t) par[v] = s;
            sides.push_back(side);
        }
        int added = 0;
        std::vector<std::vector<char>> seen;
        for (int s = 1; s < N && lp.R - N < cut_cap; ++s) {
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
            if (fl[s] < need - 1e-6) {
                if (std::find(seen.begin(), seen.end(), side) != seen.end()) continue;
                seen.push_back(side);
                // x(delta(S)) - 2 y_i - 2 y_j >= -2   <=>   -x(delta(S)) + 2 y_i + 2 y_j <= 2
                std::vector<std::pair<int, double>> a;
                for (int i = 0; i < m; ++i)
                    if (side[eu[i]] != side[ev[i]]) a.push_back({i, -1.0});
                a.push_back({m + bi, 2.0});
                a.push_back({m + bj, 2.0});
                add_row(a, 2.0, false);
                ++added;
            }
        }
        return added;
    }

    // Edmonds-Karp on a dense capacity matrix; side = vertices reachable from s in the residual
    double maxflow(const std::vector<std::vector<double>>& cap, int s, int t, std::vector<char>& side) {
        int k = (int)cap.size();
        std::vector<std::vector<double>> f(k, std::vector<double>(k, 0.0));
        double total = 0;
        std::vector<int> prv(k);
        for (;;) {
            std::fill(prv.begin(), prv.end(), -1);
            prv[s] = s;
            std::vector<int> q{s};
            for (size_t h = 0; h < q.size() && prv[t] < 0; ++h) {
                int a = q[h];
                for (int b2 = 0; b2 < k; ++b2)
                    if (prv[b2] < 0 && cap[a][b2] - f[a][b2] > 1e-12) prv[b2] = a, q.push_back(b2);
            }
            if (prv[t] < 0) break;
            double aug = INFINITY;
            for (int v = t; v != s; v = prv[v]) aug = std::min(aug, cap[prv[v]][v] - f[prv[v]][v]);
            for (int v = t; v != s; v = prv[v]) f[prv[v]][v] += aug, f[v][prv[v]] -= aug;
            total += aug;
        }
        side.assign(k, 0);
        for (int v = 0; v < k; ++v) side[v] = prv[v] >= 0;
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

    void dfs(Result& res, int depth) {
        if (!proved) return;
        if (ctl.stop) {
            proved = false;
            return;
        }
        if (++nodes > node_cap) {
            proved = false;
            return;
        }
        res.nodes = nodes;
        for (int round = 0;; ++round) {
            ++lps;
            int st = lp.solve();
            if (st == 2) {
                proved = false;
                return;
            }
            if (st == 1) {
                if (lp.farkas_ok(cols)) return;  // certified infeasible
                proved = false;
                return;
            }
            double ub = lp.safe_bound(cols);
            if (!gt(ub, inc.get())) return;
            int added = (round < 50) ? separate() : 0;
            if (added) continue;
            break;
        }
        // branching variable: the most fractional (y first: whether a vertex is used)
        int m = ne + n, q = -1;
        double bestf = 1e-6;
        for (int j = m; j < S && q < 0; ++j) {
            double x = lp.xval(j), f = std::min(x, 1 - x);
            if (f > bestf) bestf = f, q = j;
        }
        if (q < 0) {
            bestf = 1e-6;
            for (int j = 0; j < m; ++j) {
                double x = lp.xval(j), f = std::min(x, 1 - x);
                if (f > bestf) bestf = f, q = j;
            }
        }
        if (q < 0) {
            take_integral();
            // the bound above was not beaten by more than tolerance only if this point is the answer;
            // re-check: if the integral point is worse than the bound (numerics), keep searching
            double ub = lp.safe_bound(cols);
            if (gt(ub, inc.get())) {
                // pick any fractional-looking variable is impossible: give up proving
                proved = false;
            }
            return;
        }
        double lo = lp.l[q], hi = lp.u[q];
        for (double v : {1.0, 0.0}) {
            if (v < lo || v > hi) continue;
            lp.set_bounds(q, v, v);
            dfs(res, depth + 1);
            lp.set_bounds(q, lo, hi);
            if (!proved) return;
        }
    }
};

}  // namespace lp
