// Longest one-way ticket: exact longest simple path / cycle in an undirected weighted graph.
//
// usage: longest [--threads N] [--time-limit true|false] [--time-limit-sec S]
//                [--on-timeout error|approx] [--verbose]
//                [--t-dp N] [--lagr-ratio R] [--no-edge-deletion] [--x name=value ...] < input
//   --x tuning knobs (integers): fdp, fdp_width, fdp_width_sparse, fdp_min_n, fdp_budget,
//       fdp_layer, fdp_beam, fdp_prune_width, fdp_rank, lp_bnc, lp_max_n, lp_nodes, path_task,
//       task_budget, task_depth, enum_n, enum_budget, par_min_block.
//       Defaults are the measured best; the tests use the knobs to force every code path.
// stdout: station IDs, one per line, CRLF (also after the last line). Nothing else.
// exit codes: 0 ok, 1 input/usage error, 2 time limit reached (error mode), 3 internal error
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "solver.hpp"

using namespace lp;

namespace {

// canonical orientation so that the same path/cycle always prints the same way
std::vector<int> normalize(const std::vector<int>& p, const std::vector<long long>& ids) {
    if (p.size() > 2 && p.front() == p.back()) {
        std::vector<int> body(p.begin(), p.end() - 1);
        size_t m = body.size(), k = 0;
        for (size_t i = 1; i < m; ++i)
            if (ids[body[i]] < ids[body[k]]) k = i;
        std::vector<int> fwd, bwd;
        for (size_t i = 0; i < m; ++i) fwd.push_back(body[(k + i) % m]), bwd.push_back(body[(k + m - i) % m]);
        auto& c = (ids[fwd[1]] <= ids[bwd[1]]) ? fwd : bwd;
        c.push_back(c[0]);
        return c;
    }
    if (!p.empty() && ids[p.front()] > ids[p.back()]) return {p.rbegin(), p.rend()};
    return p;
}

int usage(const std::string& m) {
    std::cerr << "usage error: " << m << "\n";
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);  // keep "\r\n" as is (no "\r\r\n")
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::ios::sync_with_stdio(false);
    Config cfg;
    unsigned hw = std::thread::hardware_concurrency();
    cfg.threads = hw ? (int)hw : 1;
    cfg.threads_auto = true;  // cleared by --threads
    bool time_limit = false, approx = false, verbose = false;
    double limit_sec = 10.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        try {
            if (a == "--threads") cfg.threads = std::max(1, std::stoi(val())), cfg.threads_auto = false;
            else if (a == "--time-limit") {
                std::string v = val();
                if (v != "true" && v != "false") return usage("--time-limit takes true or false");
                time_limit = (v == "true");
            } else if (a == "--time-limit-sec") limit_sec = std::stod(val());
            else if (a == "--on-timeout") {
                std::string v = val();
                if (v != "error" && v != "approx") return usage("--on-timeout takes error or approx");
                approx = (v == "approx");
            } else if (a == "--verbose") verbose = true;
            else if (a == "--t-dp") cfg.t_dp = std::stoi(val());
            else if (a == "--lagr-ratio") cfg.lagr_ratio = std::stod(val());
            else if (a == "--no-edge-deletion") cfg.edge_deletion = false;
            else if (a == "--x") {  // tuning knobs: name=value (see Config in solver.hpp)
                std::string kv = val();
                auto eq = kv.find('=');
                if (eq == std::string::npos) return usage("--x takes name=value");
                std::string k = kv.substr(0, eq);
                long long v = std::stoll(kv.substr(eq + 1));
                if (k == "fdp") cfg.fdp = v;
                else if (k == "fdp_width") cfg.fdp_width = (int)v;
                else if (k == "fdp_width_sparse") cfg.fdp_width_sparse = (int)v;
                else if (k == "fdp_min_n") cfg.fdp_min_n = (int)v;
                else if (k == "fdp_budget") cfg.fdp_budget = v;
                else if (k == "fdp_layer") cfg.fdp_layer = v;
                else if (k == "fdp_beam") cfg.fdp_beam = v;
                else if (k == "fdp_prune_width") cfg.fdp_prune_width = (int)v;
                else if (k == "fdp_rank") cfg.fdp_rank = v;
                else if (k == "lp_bnc") cfg.lp_bnc = v;
                else if (k == "lp_max_n") cfg.lp_max_n = (int)v;
                else if (k == "lp_nodes") cfg.lp_nodes = v;
                else if (k == "path_task") cfg.path_task = v;
                else if (k == "task_budget") cfg.task_budget = v;
                else if (k == "task_depth") cfg.task_depth = (int)v;
                else if (k == "enum_n") cfg.enum_n = (int)v;
                else if (k == "enum_budget") cfg.enum_budget = v;
                else if (k == "par_min_block") cfg.par_min_block = (int)v;
                else return usage("unknown --x switch " + k);
            }
            else return usage("unknown option " + a);
        } catch (const std::exception& e) {
            return usage(e.what());
        }
    }

    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<long long> ids;
    Graph g;
    try {
        std::tie(ids, g) = parse(text);
    } catch (const InputError& e) {
        std::cerr << "input error: " << e.what() << "\n";
        return 1;
    }
    if (g.n == 0) return 0;

    auto t0 = Clock::now();
    Control ctl;
    if (time_limit) {
        ctl.has_deadline = true;
        ctl.deadline = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(limit_sec));
    }
    Solver solver(cfg, ctl);
    Sol best;
    bool timed_out = false;
    try {
        best = solver.solve(g);
    } catch (const Timeout&) {
        timed_out = true;
    }
    if (timed_out) {
        if (!approx) {
            std::cerr << "time limit reached before an exact solution was found\n";
            return 2;
        }
        best = {solver.glob.get(), solver.glob.path};
        if (verbose) std::cerr << "time limit reached; the output is the best path found so far\n";
    }
    try {
        double L = verify(g, best.path);
        if (std::fabs(L - best.len) > 1e-6 * (1 + std::fabs(L))) throw std::runtime_error("length mismatch");
    } catch (const std::exception& e) {
        std::cerr << "internal error: " << e.what() << "\n";
        return 3;
    }
    std::vector<int> out = normalize(best.path, ids);
    std::string buf;
    for (int v : out) buf += std::to_string(ids[v]) + "\r\n";
    std::fwrite(buf.data(), 1, buf.size(), stdout);
    std::fflush(stdout);
    if (verbose) {
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        auto& s = solver.st;
        std::ostringstream os;
        os.precision(10);
        os << "length=" << best.len << " time=" << sec << "s nodes=" << s.nodes << " blocks=" << s.blocks
           << " max_block=" << s.max_block << " bitdp_blocks=" << s.bitdp_blocks
           << " bnb_blocks=" << s.bnb_blocks << " dead_edges=" << s.dead
           << " lagr_off=" << s.lagr_off << " proved_at_root=" << s.proved_root
           << " threads=" << s.threads_used << " fdp_blocks=" << s.fdp_blocks << " fdp_width=" << s.fdp_width
           << " fdp_states=" << s.fdp_states << " fdp_fallback=" << s.fdp_fallback << " tasks_skipped=" << s.tasks_skipped << " lp_runs=" << s.lp_runs << " lp_nodes=" << s.lp_nodes
           << " lp_fallback=" << s.lp_fallback << "\n";
        std::cerr << os.str();
    }
    return 0;
}
