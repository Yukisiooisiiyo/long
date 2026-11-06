// Revised simplex for the LP branch and cut: same interface as BoundedSimplex, but it never keeps
// the full table B^-1 A. The constraint matrix stays sparse (an edge column has 2 degree entries
// plus the cuts that cross it); the basis inverse is kept in product form (one "eta" column per
// basis change on top of a factorization rebuilt every few dozen changes). Each iteration solves
// two systems with B (FTRAN / BTRAN) and prices columns with sparse dot products, instead of
// updating the whole dense table (which was 70% of the time on the hard inputs).
//
// As with BoundedSimplex, exactness never depends on this code: the caller prunes only with the
// safe dual bound and accepts infeasibility only with a checked Farkas row.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "search.hpp"

namespace lp {

class RevisedSimplex {
   public:
    int R = 0, C = 0, S = 0;
    std::vector<double> c, l, u, b;
    std::vector<char> is_eq;
    std::vector<int> basic, pos, idcol;
    std::vector<double> xn;
    std::vector<std::vector<std::pair<int, double>>> colA;  // every column, identity columns too
    long long iters = 0, iter_cap = 200000;
    Control* ctl = nullptr;
    int infeasible_row = -1;
    int refactor_every = 64;
    long long inconsistent = 0;

    void init(int nstruct, const std::vector<double>& cost) {
        S = C = nstruct;
        c = cost;
        l.assign(C, 0.0), u.assign(C, 1.0), xn.assign(C, 0.0), pos.assign(C, -1);
        colA.assign(C, {});
        d = c;
        dirty = true;
    }
    void add_row(const std::vector<std::pair<int, double>>& a, double rhs, bool eq) {
        int col = C++;
        c.push_back(0.0), l.push_back(0.0), u.push_back(eq ? 0.0 : kBig), xn.push_back(0.0), pos.push_back(R);
        d.push_back(0.0);
        colA.push_back({{R, 1.0}});
        for (auto [j, v] : a) colA[j].push_back({R, v});
        basic.push_back(col);
        idcol.push_back(col);
        b.push_back(rhs);
        is_eq.push_back(eq);
        ++R;
        dirty = true;  // the new identity column extends the basis; refactor before the next solve
    }
    void set_bounds(int j, double lo, double hi) {
        l[j] = lo, u[j] = hi;
        if (pos[j] < 0) {
            xn[j] = (j < (int)d.size() && d[j] > 0) ? hi : lo;
            xdirty = true;
        }
    }
    double xval(int j) const { return pos[j] >= 0 ? xB[pos[j]] : xn[j]; }

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
        auto sq = [&](auto& v) {
            using V = std::decay_t<decltype(v)>;
            V o;
            o.reserve(nc);
            for (int j = 0; j < C; ++j)
                if (!dropcol[j]) o.push_back(std::move(v[j]));
            v.swap(o);
        };
        for (auto& col : colA) {  // remap rows inside every column
            std::vector<std::pair<int, double>> c2;
            for (auto [r, v] : col)
                if (rowmap[r] >= 0) c2.push_back({rowmap[r], v});
            col.swap(c2);
        }
        sq(colA), sq(c), sq(l), sq(u), sq(xn), sq(d);
        std::vector<int> basic2, idcol2;
        std::vector<double> b2, xB2;
        std::vector<char> eq2;
        for (int r = 0; r < R; ++r) {
            if (drop[r]) continue;
            if (r < (int)xB.size()) xB2.push_back(xB[r]);  // the remaining basic values do not change
            basic2.push_back(colmap[basic[r]]);
            idcol2.push_back(colmap[idcol[r]]);
            b2.push_back(b[r]);
            eq2.push_back(is_eq[r]);
        }
        basic.swap(basic2), idcol.swap(idcol2), b.swap(b2), is_eq.swap(eq2), xB.swap(xB2);
        R = nr, C = nc;
        pos.assign(C, -1);
        for (int r = 0; r < R; ++r) pos[basic[r]] = r;
        refactor();  // the product-form inverse refers to the old rows: rebuild it now
        return rowmap;
    }

    int solve() {
        for (;;) {
            if (dirty && !refactor()) return 2;
            if (xdirty) recompute_x();
            int st = dual();
            if (st == 4) continue;  // refactor requested mid-way
            if (st != 0) return st;
            int ps = primal();
            if (ps == 3 || ps == 4) continue;
            return ps;
        }
    }

    std::vector<double> duals() const {
        std::vector<double> w(R);
        for (int r = 0; r < R; ++r) w[r] = c[basic[r]];
        btran(w);
        for (int r = 0; r < R; ++r)
            if (!is_eq[r]) w[r] = std::max(0.0, w[r]);
        return w;
    }
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
    bool farkas_ok(const std::vector<std::vector<std::pair<int, double>>>& cols) const {
        if (infeasible_row < 0) return false;
        std::vector<double> y(R, 0.0);
        y[infeasible_row] = 1.0;
        btran(y);
        double lo = 0, hi = 0;
        for (int j = 0; j < S; ++j) {
            double a = 0;
            for (auto [r, v] : cols[j])
                if (r < R) a += y[r] * v;
            lo += std::min(a * l[j], a * u[j]);
            hi += std::max(a * l[j], a * u[j]);
        }
        for (int r = 0; r < R; ++r) {
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
    struct Eta {
        int p;
        double piv;
        std::vector<std::pair<int, double>> e;  // entries i != p
    };
    std::vector<Eta> etas;
    std::vector<double> xB, d;
    bool dirty = true, xdirty = true;
    int since = 0;

    void ftran(std::vector<double>& v) const {
        for (const Eta& E : etas) {
            double t = v[E.p];
            if (t == 0.0) continue;
            v[E.p] = t * E.piv;
            for (auto [i, x] : E.e) v[i] += x * t;
        }
    }
    void btran(std::vector<double>& w) const {
        for (auto it = etas.rbegin(); it != etas.rend(); ++it) {
            double s = it->piv * w[it->p];
            for (auto [i, x] : it->e) s += x * w[i];
            w[it->p] = s;
        }
    }
    void push_eta(const std::vector<double>& alpha, int p) {
        Eta E;
        E.p = p;
        double ap = alpha[p];
        E.piv = 1.0 / ap;
        for (int i = 0; i < R; ++i)
            if (i != p && std::fabs(alpha[i]) > 1e-13) E.e.push_back({i, -alpha[i] / ap});
        etas.push_back(std::move(E));
    }
    void column(int j, std::vector<double>& v) const {
        v.assign(R, 0.0);
        for (auto [r, a] : colA[j]) v[r] += a;
    }

    // rebuild B^-1 from the identity: identity columns go back to their own row, the other basic
    // columns are pivoted in (sparsest first, largest pivot among the rows still free)
    bool refactor() {
        etas.clear();
        since = 0;
        std::vector<int> nb(R, -1);
        std::vector<char> rowfree(R, 1);
        std::vector<int> structs;
        for (int r = 0; r < R; ++r) {
            int j = basic[r];
            if (j >= S) {  // an identity column: back to its own row (its only entry)
                int own = colA[j][0].first;
                nb[own] = j, rowfree[own] = 0;
            } else
                structs.push_back(j);
        }
        std::stable_sort(structs.begin(), structs.end(), [&](int a, int b2) { return colA[a].size() < colA[b2].size(); });
        std::vector<double> v;
        for (int j : structs) {
            column(j, v);
            ftran(v);
            int p = -1;
            double best = 1e-9;
            for (int i = 0; i < R; ++i)
                if (rowfree[i] && std::fabs(v[i]) > best) best = std::fabs(v[i]), p = i;
            if (p < 0) {  // singular: this column leaves; it becomes nonbasic at a bound
                pos[j] = -1;
                xn[j] = l[j];
                continue;
            }
            push_eta(v, p);
            nb[p] = j, rowfree[p] = 0;
        }
        for (int i = 0; i < R; ++i)
            if (nb[i] < 0) {  // rows left uncovered (after a singular column): their identity column
                int j = idcol[i];
                nb[i] = j;
            }
        basic = nb;
        std::fill(pos.begin(), pos.end(), -1);
        for (int r = 0; r < R; ++r) pos[basic[r]] = r;
        dirty = false;
        recompute_x();
        recompute_d();
        return true;
    }
    void recompute_x() {
        std::vector<double> rhs(b.begin(), b.end());
        for (int j = 0; j < C; ++j)
            if (pos[j] < 0 && xn[j] != 0.0)
                for (auto [r, a] : colA[j]) rhs[r] -= a * xn[j];
        ftran(rhs);
        xB.swap(rhs);
        xdirty = false;
    }
    void recompute_d() {
        std::vector<double> w(R);
        for (int r = 0; r < R; ++r) w[r] = c[basic[r]];
        btran(w);
        d.assign(C, 0.0);
        for (int j = 0; j < C; ++j) {
            if (pos[j] >= 0) continue;
            double s = c[j];
            for (auto [r, a] : colA[j]) s -= w[r] * a;
            d[j] = s;
        }
    }
    // row r of B^-1 A over the nonbasic columns
    void row_alpha(int r, std::vector<double>& ar) const {
        std::vector<double> rho(R, 0.0);
        rho[r] = 1.0;
        btran(rho);
        ar.assign(C, 0.0);
        for (int j = 0; j < C; ++j) {
            if (pos[j] >= 0) continue;
            double s = 0;
            for (auto [i, a] : colA[j]) s += rho[i] * a;
            ar[j] = s;
        }
    }
    // basis change: q enters by delta, the variable at row r leaves at `target`
    void change(int q, int r, double delta, double target, const std::vector<double>& aq, const std::vector<double>& ar) {
        for (int i = 0; i < R; ++i) xB[i] -= aq[i] * delta;
        double enter_val = xn[q] + delta;
        double ratio = d[q] / aq[r];
        for (int j = 0; j < C; ++j)
            if (pos[j] < 0 && ar[j] != 0.0) d[j] -= ratio * ar[j];
        int jb = basic[r];
        d[q] = 0.0;
        d[jb] = -ratio;
        pos[jb] = -1;
        xn[jb] = target;
        basic[r] = q;
        pos[q] = r;
        xB[r] = enter_val;
        push_eta(aq, r);
        if ((++iters & 63) == 0 && ctl) ctl->check();
        if (++since >= refactor_every) dirty = true;
    }

    int dual() {
        std::vector<double> ar, aq;
        for (;;) {
            if (iters > iter_cap) return 2;
            if (dirty) return 4;
            int r = -1;
            double worst = kTol, val = 0;
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                double x = xB[i];
                double viol = std::max(l[j] - x, x - u[j]);
                if (viol > worst) worst = viol, r = i, val = x;
            }
            if (r < 0) return 0;
            int jb = basic[r];
            bool low = val < l[jb];
            row_alpha(r, ar);
            int q = -1;
            double best = INFINITY;
            for (int j = 0; j < C; ++j) {
                if (pos[j] >= 0 || l[j] == u[j]) continue;
                double a = ar[j];
                if (std::fabs(a) < kPiv) continue;
                bool atlow = xn[j] == l[j];
                bool ok = low ? ((atlow && a < 0) || (!atlow && a > 0)) : ((atlow && a > 0) || (!atlow && a < 0));
                if (!ok) continue;
                double ratio = std::fabs(d[j] / a);
                if (ratio < best - 1e-12 || (ratio < best + 1e-12 && q >= 0 && std::fabs(a) > std::fabs(ar[q])))
                    best = ratio, q = j;
            }
            if (q < 0) {
                infeasible_row = r;
                return 1;
            }
            column(q, aq);
            ftran(aq);
            if (std::fabs(aq[r]) < kPiv || std::fabs(aq[r] - ar[q]) > 1e-6 * (1 + std::fabs(ar[q]))) {
                ++inconsistent;
                if (since == 0) return 2;  // even right after a rebuild: give up (the caller falls back)
                dirty = true;  // numerically inconsistent: rebuild and retry
                return 4;
            }
            double target = low ? l[jb] : u[jb];
            double delta = (val - target) / aq[r];
            change(q, r, delta, target, aq, ar);
        }
    }

    int primal() {
        std::vector<double> ar, aq;
        int degenerate = 0;
        for (;;) {
            if (iters > iter_cap) return 2;
            if (dirty) return 4;
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
            column(q, aq);
            ftran(aq);
            double dir = xn[q] == l[q] ? 1.0 : -1.0;
            double theta = u[q] - l[q];
            int r = -1;
            double rtarget = 0;
            for (int i = 0; i < R; ++i) {
                double a = aq[i] * dir;
                if (std::fabs(a) < kPiv) continue;
                int j = basic[i];
                double x = xB[i];
                double lim, tgt;
                if (a > 0) lim = (x - l[j]) / a, tgt = l[j];
                else lim = (u[j] >= kBig / 2) ? INFINITY : (x - u[j]) / a, tgt = u[j];
                if (lim < 0) lim = 0;
                if (lim < theta - 1e-12 || (r >= 0 && lim < theta + 1e-12 && bland && basic[i] < basic[r]))
                    theta = lim, r = i, rtarget = tgt;
            }
            if (theta >= kBig / 2) return 2;
            degenerate = theta < 1e-12 ? degenerate + 1 : 0;
            if (r < 0) {  // bound flip
                double nv = dir > 0 ? u[q] : l[q];
                double delta = nv - xn[q];
                for (int i = 0; i < R; ++i) xB[i] -= aq[i] * delta;
                xn[q] = nv;
                continue;
            }
            row_alpha(r, ar);
            change(q, r, theta * dir, rtarget, aq, ar);
            for (int i = 0; i < R; ++i) {
                int j = basic[i];
                if (xB[i] < l[j] - 1e-7 || xB[i] > u[j] + 1e-7) return 3;
            }
        }
    }
};

}  // namespace lp
