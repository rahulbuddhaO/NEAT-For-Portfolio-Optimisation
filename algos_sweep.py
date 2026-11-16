"""
algos_sweep.py -- the nine comparison algorithms (PSO, CMA-ES, DE, GA, ABC, GWO, Firefly, NSGA-II operators with a
single Sharpe objective, GP-tree) run on exactly the same stock draws, windows, seeds, features, costs and fitness
as the NEAT sweep, with ONE flat job queue (every finished job is saved; a re-run resumes).

Needs neat_sweep.py and neat_sharpe_parallel_aws_32.py in the same folder.

Shared network for the eight parameter-based algorithms (same starting point as NEAT):
    inputs -> 8 tanh hidden units  AND  direct input -> output links, 15 outputs -> softmax weights.
    node function tanh(2.5 z) exactly as neat-python's tanh. Structure fixed; the algorithm searches the weights only.
    parameters are bounded to [-30, 30] (same as NEAT); initial population drawn from N(0, 1) (CMA-ES starts at 0, sigma0 = 0.5).
Every algorithm gets population 150 and 100 generations (about 15,000 fitness evaluations), the same as NEAT.

Environment switches: NEAT_MASTER, NEAT_N, NEAT_DRAWS, NEAT_CONFIGS, NEAT_WINDOWS, NEAT_SEEDS, NEAT_JOBS,
NEAT_SWEEP_DIR (run folder), ALGO_LIST (e.g. PSO,DE), ALGO_POP, ALGO_ITERS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker; the workers themselves run in parallel
import json, multiprocessing, pickle, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import neat_sharpe_parallel_aws_32 as base
import neat_sweep as ns

ALGOS = ["PSO", "CMAES", "DE", "GA", "ABC", "GWO", "FIREFLY", "NSGA2", "GPTREE"]
LABEL = {"PSO": "PSO", "CMAES": "sep-CMA-ES", "DE": "DE", "GA": "GA", "ABC": "ABC", "GWO": "GWO",
         "FIREFLY": "Firefly", "NSGA2": "NSGA-II (single objective)", "GPTREE": "GP-tree"}
POP = int(os.environ.get("ALGO_POP", 150))
ITERS = int(os.environ.get("ALGO_ITERS", 100))
_sel = base._env_list("ALGO_LIST")
RUN_ALGOS = [a for a in ALGOS if (not _sel or a in _sel)]
H, LO, HI = 8, -30.0, 30.0        # same weight/bias limits as NEAT (weight/bias min/max = -30/30 in its config)
STEP = 10.0                       # scale of the search steps (PSO max velocity, NSGA-II mutation); unchanged from before
COST = base.COST_BPS / 1e4
_MASTER = None


# ---------------------------------------------------------------- shared network and fitness
class Policy:
    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        a = n_in * (H + n_out)
        self.sl = (slice(0, a), slice(a, a + H), slice(a + H, a + H + H * n_out),
                   slice(a + H + H * n_out, a + H + H * n_out + n_out))
        self.D = a + H + H * n_out + n_out

    def weights(self, theta, X):
        Wc = theta[self.sl[0]].reshape(self.n_in, H + self.n_out)
        b1, W2, b2 = theta[self.sl[1]], theta[self.sl[2]].reshape(H, self.n_out), theta[self.sl[3]]
        Z = X @ Wc                                    # input->hidden and the direct input->output links in one product
        h = np.tanh(2.5 * (Z[:, :H] + b1))
        o = h @ W2 + Z[:, H:] + b2
        o = o if base.OUTPUT_MAP == "free" else np.tanh(2.5 * o)    # native and bounded: outputs in [-1, 1]
        return _softmax(o)


def _softmax(o):
    z = o - o.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def stats(Wt, R):
    """Net daily returns and daily turnover; identical rule to the NEAT simulation (day 1 trades from 1/N)."""
    n = Wt.shape[1]
    prev = np.vstack([np.full((1, n), 1.0 / n), Wt[:-1]])
    to = np.abs(Wt - prev).sum(axis=1)
    return (Wt * R).sum(axis=1) - to * COST, to


def fit_from(net):
    """Same as the NEAT fitness: annualised Sharpe clipped to [-3, 5]."""
    net = net[np.isfinite(net)]                  # the first day has no return (NaN) and is dropped, as in the NEAT fitness
    if len(net) < 20 or net.std(ddof=1) < 1e-10:
        return -10.0
    sd = net.std()
    if sd < 1e-10:
        return -10.0
    return float(np.clip(np.sqrt(252) * net.mean() / sd, -3.0, 5.0))


def _evalpop(fit, P):
    return np.array([fit(p) for p in P])


def _init(rng, P, D):
    return np.clip(rng.normal(0.0, 1.0, (P, D)), LO, HI)


class _Best:
    def __init__(self):
        self.f, self.x = -np.inf, None

    def update(self, X, f):
        i = int(np.argmax(f))
        if f[i] > self.f:
            self.f, self.x = float(f[i]), np.array(X[i], copy=True)


# ---------------------------------------------------------------- the algorithms (all maximise `fit`)
def run_pso(fit, D, rng, P, iters, seed):
    W_, C1, C2, VMAX = 0.7, 1.5, 1.5, 0.2 * STEP
    x = _init(rng, P, D); v = np.zeros((P, D)); f = _evalpop(fit, x)
    pb, pf = x.copy(), f.copy(); best = _Best(); best.update(x, f)
    for _ in range(iters - 1):
        g = best.x
        v = W_ * v + C1 * rng.random((P, D)) * (pb - x) + C2 * rng.random((P, D)) * (g - x)
        v = np.clip(v, -VMAX, VMAX); x = np.clip(x + v, LO, HI)
        f = _evalpop(fit, x); imp = f > pf
        pb[imp], pf[imp] = x[imp], f[imp]; best.update(x, f)
    return best.x, best.f


def run_cmaes(fit, D, rng, P, iters, seed):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(D), 0.5, {"popsize": P, "maxiter": iters, "CMA_diagonal": True,
                                                      "bounds": [LO, HI], "seed": int(seed) % (2 ** 31 - 2) + 1,
                                                      "verbose": -9})
    best = _Best()
    for _ in range(iters):
        sols = es.ask(); X = np.array(sols); f = _evalpop(fit, X)
        es.tell(sols, list(-f)); best.update(X, f)
    return best.x, best.f


def run_de(fit, D, rng, P, iters, seed):
    F_, CR = 0.5, 0.9
    x = _init(rng, P, D); f = _evalpop(fit, x); best = _Best(); best.update(x, f)
    ar = np.arange(P)
    for _ in range(iters - 1):
        idx = np.array([rng.choice(P - 1, 3, replace=False) for _ in range(P)])
        idx = idx + (idx >= ar[:, None])                       # three distinct individuals, none equal to i
        mutant = x[idx[:, 0]] + F_ * (x[idx[:, 1]] - x[idx[:, 2]])
        cross = rng.random((P, D)) < CR; cross[ar, rng.integers(D, size=P)] = True
        trial = np.clip(np.where(cross, mutant, x), LO, HI)
        ft = _evalpop(fit, trial); better = ft >= f
        x[better], f[better] = trial[better], ft[better]; best.update(x, f)
    return best.x, best.f


def run_ga(fit, D, rng, P, iters, seed):
    x = _init(rng, P, D); f = _evalpop(fit, x); best = _Best(); best.update(x, f)
    for _ in range(iters - 1):
        elite, ef = x[int(np.argmax(f))].copy(), f.max()

        def select():
            t = rng.integers(P, size=(P, 3))
            return x[t[np.arange(P), np.argmax(f[t], axis=1)]]
        p1, p2 = select(), select()
        do = rng.random((P, 1)) < 0.9
        child = np.where(do & (rng.random((P, D)) < 0.5), p2, p1)          # uniform crossover
        child = child + (rng.random((P, D)) < 0.1) * rng.normal(0.0, 0.3, (P, D))   # Gaussian mutation
        child = np.clip(child, LO, HI); child[0] = elite                   # elitism 1
        fn = np.empty(P); fn[0] = ef; fn[1:] = _evalpop(fit, child[1:])
        x, f = child, fn; best.update(x, f)
    return best.x, best.f


def run_abc(fit, D, rng, P, iters, seed):
    SN, LIMIT, MR = max(P // 2, 2), 100, 0.1          # MR: share of dimensions changed per trial (modified ABC)
    x = _init(rng, SN, D); f = _evalpop(fit, x); trials = np.zeros(SN, int); best = _Best(); best.update(x, f)

    def neighbour(i):
        k = int(rng.integers(SN - 1)); k += k >= i
        m = rng.random(D) < MR; m[int(rng.integers(D))] = True
        v = x[i].copy(); v[m] = x[i][m] + rng.uniform(-1, 1, D)[m] * (x[i][m] - x[k][m])
        return np.clip(v, LO, HI)

    def greedy(i, v):
        fv = fit(v)
        if fv > f[i]:
            x[i], f[i], trials[i] = v, fv, 0
        else:
            trials[i] += 1
        return fv, v
    for _ in range(iters - 1):
        for i in range(SN):                                       # employed bees
            fv, v = greedy(i, neighbour(i))
        pr = f - f.min() + 1e-9; pr = pr / pr.sum()
        for i in rng.choice(SN, size=SN, p=pr):                   # onlooker bees
            greedy(int(i), neighbour(int(i)))
        j = int(np.argmax(trials))
        if trials[j] > LIMIT:                                     # scout bee
            x[j] = _init(rng, 1, D)[0]; f[j] = fit(x[j]); trials[j] = 0
        best.update(x, f)
    return best.x, best.f


def run_gwo(fit, D, rng, P, iters, seed):
    x = _init(rng, P, D); f = _evalpop(fit, x); best = _Best(); best.update(x, f)
    for t in range(iters - 1):
        lead = x[np.argsort(-f)[:3]]; a = 2.0 * (1.0 - t / max(iters - 1, 1))
        new = np.zeros_like(x)
        for L in lead:
            A = 2 * a * rng.random((P, D)) - a; C = 2 * rng.random((P, D))
            new += L - A * np.abs(C * L - x)
        x = np.clip(new / 3.0, LO, HI); f = _evalpop(fit, x); best.update(x, f)
    return best.x, best.f


def run_firefly(fit, D, rng, P, iters, seed):
    B0, GAMMA, ALPHA = 1.0, 0.5, 0.2          # squared distance is divided by D so that gamma is dimension-free
    x = _init(rng, P, D); f = _evalpop(fit, x); best = _Best(); best.update(x, f)
    for _ in range(iters - 1):
        I = f.copy()
        for i in range(P):
            for j in np.nonzero(I > I[i])[0]:
                r2 = float(np.mean((x[i] - x[j]) ** 2))
                x[i] = x[i] + B0 * np.exp(-GAMMA * r2) * (x[j] - x[i]) + ALPHA * (2.0 * rng.random(D) - 1.0)
            x[i] = np.clip(x[i], LO, HI)
        f = _evalpop(fit, x); best.update(x, f)
    return best.x, best.f


def run_nsga2(fit, D, rng, P, iters, seed):
    """NSGA-II variation operators (SBX, polynomial mutation, binary tournament, mu+lambda survival) with ONE
    objective (Sharpe). With a single objective, non-dominated sorting reduces to ranking by fitness."""
    ETA_C, ETA_M, PC, PM = 20.0, 20.0, 0.9, 1.0 / D
    x = _init(rng, P, D); f = _evalpop(fit, x); best = _Best(); best.update(x, f)
    for _ in range(iters - 1):
        a = rng.integers(P, size=(P, 2)); par = x[np.where(f[a[:, 0]] > f[a[:, 1]], a[:, 0], a[:, 1])]
        p1, p2 = par[0::2], par[1::2]; m = len(p1)
        u = rng.random((m, D))
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (ETA_C + 1)), (1 / (2 * (1 - u) + 1e-12)) ** (1 / (ETA_C + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2); c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        do = (rng.random((m, 1)) < PC) & (rng.random((m, D)) < 0.5)
        c1, c2 = np.where(do, c1, p1), np.where(do, c2, p2)
        child = np.empty((2 * m, D)); child[0::2], child[1::2] = c1, c2
        mu = rng.random(child.shape) < PM; r = rng.random(child.shape)
        delta = np.where(r < 0.5, (2 * r) ** (1 / (ETA_M + 1)) - 1, 1 - (2 * (1 - r)) ** (1 / (ETA_M + 1)))
        child = np.clip(child + mu * delta * STEP, LO, HI)
        fc = _evalpop(fit, child)
        allx, allf = np.vstack([x, child]), np.concatenate([f, fc]); keep = np.argsort(-allf)[:P]
        x, f = allx[keep], allf[keep]; best.update(x, f)
    return best.x, best.f


# ---------------------------------------------------------------- GP-tree: one expression tree per asset
_PRIMS = (("add", 2), ("sub", 2), ("mul", 2), ("pdiv", 2), ("neg", 1), ("tanh", 1))


def _rand_tree(rng, depth, n_in, grow):
    if depth == 0 or (grow and rng.random() < 0.3):
        return ("var", int(rng.integers(n_in))) if rng.random() < 0.85 else ("const", float(rng.uniform(-1, 1)))
    name, ar = _PRIMS[int(rng.integers(len(_PRIMS)))]
    return (name,) + tuple(_rand_tree(rng, depth - 1, n_in, grow) for _ in range(ar))


def _eval_tree(t, X):
    k = t[0]
    if k == "var":
        return X[:, t[1]]
    if k == "const":
        return np.full(X.shape[0], t[1])
    a = _eval_tree(t[1], X)
    if k == "neg":
        return -a
    if k == "tanh":
        return np.tanh(a)
    b = _eval_tree(t[2], X)
    if k == "add":
        return a + b
    if k == "sub":
        return a - b
    if k == "mul":
        return a * b
    return np.where(np.abs(b) < 1e-3, 1.0, a / np.where(np.abs(b) < 1e-3, 1.0, b))          # protected division


def _paths(t, p=()):
    out = [p]
    if t[0] not in ("var", "const"):
        for i in range(1, len(t)):
            out += _paths(t[i], p + (i,))
    return out


def _get(t, p):
    for i in p:
        t = t[i]
    return t


def _put(t, p, sub):
    if not p:
        return sub
    lst = list(t); lst[p[0]] = _put(t[p[0]], p[1:], sub)
    return tuple(lst)


def _depth(t):
    return 0 if t[0] in ("var", "const") else 1 + max(_depth(c) for c in t[1:])


def _tree_out(trees, X):
    O = np.column_stack([np.clip(np.nan_to_num(_eval_tree(t, X), nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)
                         for t in trees])
    return _softmax(base.map_outputs(O))


def run_gptree(Xtr, Rtr, rng, P, iters):
    """15 co-evolved trees per individual. Ramped half-and-half start (depth 2-4), tournament 3, subtree crossover
    (individual p=0.9, each tree p=0.4), subtree mutation (each tree p=0.15), max depth 8, elitism 1."""
    n_in, n_out = Xtr.shape[1], Rtr.shape[1]
    X32 = Xtr.astype(np.float64)

    def fit(ind):
        return fit_from(stats(_tree_out(ind, X32), Rtr)[0])

    def rand_ind():
        return [_rand_tree(rng, int(rng.integers(2, 5)), n_in, bool(rng.random() < 0.5)) for _ in range(n_out)]

    pop = [rand_ind() for _ in range(P)]; f = np.array([fit(i) for i in pop]); best_i = int(np.argmax(f))
    best = (float(f[best_i]), pop[best_i])
    for _ in range(iters - 1):
        elite = pop[int(np.argmax(f))]; ef = f.max(); new = [elite]
        while len(new) < P:
            def sel():
                t = rng.integers(P, size=3); return pop[int(t[np.argmax(f[t])])]
            c = list(sel()); other = sel()
            if rng.random() < 0.9:
                for i in range(n_out):
                    if rng.random() < 0.4:
                        pa, pb = _paths(c[i]), _paths(other[i])
                        sub = _get(other[i], pb[int(rng.integers(len(pb)))])
                        cand = _put(c[i], pa[int(rng.integers(len(pa)))], sub)
                        if _depth(cand) <= 8:
                            c[i] = cand
            for i in range(n_out):
                if rng.random() < 0.15:
                    pa = _paths(c[i])
                    cand = _put(c[i], pa[int(rng.integers(len(pa)))], _rand_tree(rng, int(rng.integers(0, 3)), n_in, True))
                    if _depth(cand) <= 8:
                        c[i] = cand
            new.append(c)
        fn = np.empty(P); fn[0] = ef; fn[1:] = [fit(i) for i in new[1:]]
        pop, f = new, fn
        i = int(np.argmax(f))
        if f[i] > best[0]:
            best = (float(f[i]), pop[i])
    return best[1], best[0]


FUNCS = {"PSO": run_pso, "CMAES": run_cmaes, "DE": run_de, "GA": run_ga, "ABC": run_abc, "GWO": run_gwo,
         "FIREFLY": run_firefly, "NSGA2": run_nsga2}
PRIORITY = {"GPTREE": 0, "FIREFLY": 1}


# ---------------------------------------------------------------- jobs
def _init_worker(pickle_path):
    global _MASTER
    _MASTER = pd.read_pickle(pickle_path)


def _result_path(res_dir, draw, algo, fc, w, seed):
    return Path(res_dir) / f"d{draw}_{algo}__{fc}_w{w}_s{seed}.pkl"


def _job(spec):
    draw, tickers, algo, fc, w, seed, out_dir = spec
    t0 = time.time()
    try:
        master = _MASTER
        if master is None or len(master) < 1000:
            raise RuntimeError(f"worker has a master dataset with only {0 if master is None else len(master)} rows")
        index = master.index
        blocks = np.array_split(np.arange(len(index)), base.N_WINDOWS + 1)
        train_pos, test_pos = np.concatenate(blocks[:w]), blocks[w]
        feat, ret = base.build_features(master, list(tickers), base.FEATURE_CONFIGS[fc])
        Xtr, Rtr = feat.values[train_pos], ret.values[train_pos]
        Xte, Rte = feat.values[test_pos], ret.values[test_pos]
        n_in, n_out = Xtr.shape[1], Rtr.shape[1]
        rng = np.random.default_rng([seed, w, draw, ALGOS.index(algo), list(base.FEATURE_CONFIGS_ALL).index(fc)])
        if algo == "GPTREE":
            trees, bf = run_gptree(Xtr, Rtr, rng, POP, ITERS)
            Wte = _tree_out(trees, Xte.astype(np.float64)); n_par = int(sum(len(_paths(t)) for t in trees))
        else:
            pol = Policy(n_in, n_out)
            fit = lambda th: fit_from(stats(pol.weights(th, Xtr), Rtr)[0])
            theta, bf = FUNCS[algo](fit, pol.D, rng, POP, ITERS, seed)
            Wte = pol.weights(theta, Xte); n_par = pol.D
        net, to = stats(Wte, Rte)
        oos = pd.Series(net, index=feat.index[test_pos])
        oos_d, to_d = base.net_of_costs(Wte, Rte, feat.index[test_pos])     # same cost rule as NEAT and the benchmarks
        return {"ok": True, "draw": draw, "algo": algo, "fc": fc, "w": w, "seed": seed, "perf": base.performance_summary(oos), "perf_drift": base.performance_summary(oos_d),
                "turnovers_drift": to_d.astype(np.float32), "output_map": base.OUTPUT_MAP,
                "oos_turnover": float(to.mean()), "oos_ret": oos, "weights": Wte.astype(np.float32),
                "turnovers": to.astype(np.float32), "train_fitness": bf, "n_params": n_par, "secs": time.time() - t0}
    except Exception as e:
        import traceback
        return {"ok": False, "draw": draw, "algo": algo, "fc": fc, "w": w, "seed": seed,
                "error": str(e), "traceback": traceback.format_exc(), "secs": time.time() - t0}


def summarise(out_dir, draws, algos, fcs, windows, seeds):
    res_dir = Path(out_dir) / "results"; rows = []
    for d in draws:
        for a in algos:
            for fc in fcs:
                for w in windows:
                    for s in seeds:
                        f = _result_path(res_dir, d, a, fc, w, s)
                        if f.exists():
                            r = pickle.load(open(f, "rb")); p = r["perf"]
                            pd_ = r.get("perf_drift", p)
                            rows.append({"Draw": d, "Algo": a, "Config": fc, "Window": w, "Seed": s, "Sharpe": pd_["Sharpe"],
                                         "Sharpe_simcost": p["Sharpe"],
                                         "AnnReturn": pd_["AnnReturn"], "AnnVol": pd_["AnnVol"], "MaxDD": pd_["MaxDrawdown"],
                                         "Turnover": r["oos_turnover"], "TrainFitness": r["train_fitness"],
                                         "Params": r["n_params"], "Secs": r["secs"]})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_dir) / "algos_runs.csv", index=False)
        piv = df.groupby(["Algo", "Config"]).Sharpe.mean().unstack("Config")
        piv = piv.reindex([a for a in algos if a in piv.index])
        piv.to_csv(Path(out_dir) / "algos_mean_sharpe.csv")
        print("\nMean Sharpe per run (5 bp, drift-turnover rule = same as NEAT and the benchmarks), by algorithm and feature set:\n",
              piv.round(3).to_string())
    return df


def main():
    multiprocessing.freeze_support()
    master = ns.load_master()
    print(f"[master] {base.MASTER_CSV_PATH}: {len(master)} days"
          + (f", {master.index.min().date()} -> {master.index.max().date()}" if len(master) else "")
          + f", {master.shape[1]} columns", flush=True)
    if len(master) < 1000:
        sys.exit("STOP: the master dataset is empty after the date filter. Check NEAT_MASTER.")
    order = [c[:-len("_Price")] for c in master.columns if c.endswith("_Price")]
    for d, tk in ns.DRAWS.items():
        assert [t for t in order if t in tk] == tk, f"draw {d}: tickers must follow master order"
    out_dir = Path(os.environ.get("NEAT_SWEEP_DIR") or
                   base.LOCAL_OUTPUT_ROOT / f"ALGOS_N{ns.N_STOCKS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    res_dir = out_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {out_dir}")
    windows = sorted(base.WINDOWS_TO_RUN) if base.WINDOWS_TO_RUN else list(range(1, base.N_WINDOWS + 1))
    fcs = list(base.FEATURE_CONFIGS)
    if os.environ.get("NEAT_AGG_ONLY") == "1":
        summarise(out_dir, list(ns.DRAWS), RUN_ALGOS, fcs, windows, base.SEEDS); return
    pkl = out_dir / "master_cached.pkl"; master.to_pickle(pkl)
    specs, skipped = [], 0
    for d, tk in ns.DRAWS.items():
        for a in RUN_ALGOS:
            for fc in fcs:
                for w in windows:
                    for s in base.SEEDS:
                        if _result_path(res_dir, d, a, fc, w, s).exists():
                            skipped += 1; continue
                        specs.append((d, tuple(tk), a, fc, w, s, str(out_dir)))
    specs.sort(key=lambda x: (PRIORITY.get(x[2], 2), x[0], -x[4], x[5]))
    total = len(specs)
    json.dump({"n_stocks": ns.N_STOCKS, "draws": ns.DRAWS, "algos": RUN_ALGOS, "configs": fcs, "seeds": base.SEEDS,
               "windows": windows, "population": POP, "iterations": ITERS, "cost_bps": base.COST_BPS, "output_map": base.OUTPUT_MAP, "param_bounds": [LO, HI],
               "master": base.MASTER_CSV_PATH, "workers": base.N_CONCURRENT_JOBS, "script": "algos_sweep.py"},
              open(out_dir / "algos_manifest.json", "w"), indent=2)
    print(f"{total} jobs to run ({skipped} already done), {base.N_CONCURRENT_JOBS} workers, population {POP}, "
          f"{ITERS} generations, {len(RUN_ALGOS)} algorithm(s), {len(fcs)} feature set(s)", flush=True)
    t0, ndone, nerr = time.time(), 0, 0
    if total:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=base.N_CONCURRENT_JOBS, maxtasksperchild=1, initializer=_init_worker,
                      initargs=(str(pkl),)) as pool:
            for r in pool.imap_unordered(_job, specs, chunksize=1):
                ndone += 1
                if r["ok"]:
                    with open(_result_path(res_dir, r["draw"], r["algo"], r["fc"], r["w"], r["seed"]), "wb") as f:
                        pickle.dump(r, f)
                    msg = f"Sharpe={r['perf']['Sharpe']:>6.2f} params={r['n_params']:<6} job={r['secs']/60:5.1f}min"
                else:
                    nerr += 1
                    with open(out_dir / "errors.log", "a") as f:
                        f.write(f"draw{r['draw']} {r['algo']} {r['fc']} w{r['w']} s{r['seed']}: {r['error']}\n{r['traceback']}\n")
                    msg = f"FAILED: {r['error'][:80]}"
                el = time.time() - t0
                print(f"[{ndone:>5}/{total}] draw{r['draw']} {r['algo']:<8} {r['fc']:<15} w{r['w']:<2} s{r['seed']:<3} {msg} | "
                      f"elapsed {el/3600:.2f}h, rough ETA {el/ndone*(total-ndone)/3600:.1f}h", flush=True)
    print(f"\nFinished: {ndone} jobs run, {nerr} failed. Building summary ...", flush=True)
    summarise(out_dir, list(ns.DRAWS), RUN_ALGOS, fcs, windows, base.SEEDS)
    if nerr:
        print(f"{nerr} jobs failed -- see {out_dir/'errors.log'}; fix and re-run with NEAT_SWEEP_DIR={out_dir} to retry only those.")
    print("DONE.")


if __name__ == "__main__":
    main()
