"""
neat_sharpe_parallel_aws_32.py  (32-stock version of neat_sharpe_parallel_aws.py)
CHANGES vs the 6-stock script: master CSV path, TICKERS, a vectorised but numerically identical
simulate(), optional environment-variable overrides (NEAT_CONFIGS, NEAT_WINDOWS, NEAT_SEEDS,
NEAT_GENERATIONS, NEAT_TICKERS, NEAT_TAG, NEAT_VIZ, NEAT_JOBS, NEAT_JOB_TIMEOUT), and per-job timing.
Everything else (fitness, NEAT settings, windows, benchmarks) is unchanged.

ORIGINAL DOCSTRING FOLLOWS
==========================
neat_sharpe_parallel_aws.py
============================
Job-level-parallel rewrite of neat_FINAL_v4_pure_sharpe.py (v5), built to
actually use a many-core box like your c7i.16xlarge (64 vCPUs).

WHY THIS EXISTS
----------------
The v5 script parallelizes at the POPULATION level: one (config, window,
seed) job runs at a time, and THAT job spreads its 150-genome population
across all N_CORES workers via an inner multiprocessing.Pool. On a 16-core
desktop this is fine. On a 64-vCPU box it isn't: 150 genomes / 64 workers
is only ~2.3 genomes per worker, so each generation's parallel burst lasts
a couple of seconds before every worker goes idle waiting for the next
generation's serial reproduction/speciation step (which only ever runs on
one core, regardless of how many you have). That's the mechanical reason
CloudWatch showed CPU utilization oscillating in the 30-48% band instead
of sitting near 100%.

This version flips the parallelism to the JOB level instead: it runs many
(config, window, seed) jobs CONCURRENTLY, each single-threaded (no inner
pool at all). With 495 total jobs and N_CONCURRENT_JOBS close to the full
core count, the box now has ~60 independent NEAT trainings running at
once instead of one job monopolizing all 64 cores inefficiently. This is
the exact same restructuring already applied to ppo_sharpe_comparison.py
locally, using the same Pool(maxtasksperchild=1) pattern for per-job
process isolation (each pool worker is torn down after exactly one job,
preserving the v5 memory-leak fix's guarantee that every job gets a fully
fresh process -- maxtasksperchild=1 gives you that for free, no separate
ctx.Process/Queue machinery needed).

WHAT THIS COSTS YOU vs. v5's within-job parallelism
-----------------------------------------------------
Going fully single-threaded per job means an INDIVIDUAL job with no
competition for cores would take longer wall-clock than it did with 64
(inefficient) workers behind it, because population evaluation within
that one job is no longer parallelized at all. The bet is that running
~60 of these concurrently more than makes up for that. Based on the
observed ~30-48% utilization band, expect something in the neighborhood
of 2-3x faster overall completion -- NOT a clean 64/16=4x or more, because
you were already extracting real (if inefficient) value from the 64 cores
before. Time the first window after switching and compare against the
old pace before trusting a longer projection -- exactly the same caution
the original script's own startup banner already gives you.

WHAT'S UNCHANGED
------------------
Fitness function, NEAT hyperparameters, feature configs, walk-forward
windows, benchmarks, resume contract (whole-window-only trust -- a
partially-done window is always fully redone, same as v5), network
diagram / generation-stats scope (seed=NETWORK_VIZ_SEED across all
windows). All of Section 3's methodology is identical; only HOW the
compute is parallelized changed, which affects wall-clock time only, not
results (same seeds, same NEAT config, same everything -- see the
CHANGED comments below for the two places that had to adapt mechanically
to concurrent jobs).

TWO THINGS YOU MUST FILL IN BEFORE RUNNING (search for "FILL IN"):
  1. MASTER_CSV_PATH -- the Linux path to master_dataset.csv on this box
  2. RESUME_FROM_DIR -- the Linux path to your currently-running job's
     OUTPUT_DIR, so this picks up from your already-completed windows
     instead of starting over. Get both from the currently-running job's
     run_manifest.json (run: find / -maxdepth 6 -iname "run_manifest.json"
     2>/dev/null    then cat whatever it finds) or from its startup
     banner further up your terminal's scrollback.

Usage
-----
    python3 neat_sharpe_parallel_aws.py
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# CHANGED: must be set before numpy/scipy import. With up to ~60 jobs now
# running concurrently instead of 1, any BLAS/OMP internal threading would
# oversubscribe the box (60 processes x N internal threads each, fighting
# over 64 real cores). Same fix already applied in ppo_sharpe_comparison.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import neat

warnings.filterwarnings("ignore")


# ======================================================================
# FAST CROSSOVER (exact speed-up for neat-python 2.x)
# ======================================================================
# neat-python 2.0's DefaultGenome.configure_crossover runs a cycle check (which also copies the whole
# connection list) for EVERY inherited connection gene, i.e. quadratic work per child. In this study that
# check can never fire: a child receives exactly the connection keys of its fitter parent, and that parent is
# already acyclic. Removing it gives identical genomes and an identical random stream, only much faster.
# Disable with NEAT_FAST_CROSSOVER=0. selftest_fast_crossover() verifies equality before a long run.
_ORIGINAL_XOVER = None
FAST_XOVER_STATUS = "not applied"


def _apply_fast_crossover():
    global _ORIGINAL_XOVER, FAST_XOVER_STATUS
    if os.environ.get("NEAT_FAST_CROSSOVER", "1") != "1":
        FAST_XOVER_STATUS = "disabled by NEAT_FAST_CROSSOVER=0"
        return
    try:
        import inspect
        import neat.genome as ng
        _ORIGINAL_XOVER = ng.DefaultGenome.configure_crossover
        src = inspect.getsource(_ORIGINAL_XOVER)
    except Exception as e:
        FAST_XOVER_STATUS = f"not applied (cannot inspect neat-python: {e})"
        return
    if "creates_cycle" not in src:
        FAST_XOVER_STATUS = "not needed (this neat-python version has no cycle check in crossover)"
        return

    def fast_configure_crossover(self, genome1, genome2, config):
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        p1 = {cg.innovation: cg for cg in parent1.connections.values()}
        p2 = {cg.innovation: cg for cg in parent2.connections.values()}
        for innovation_num in set(p1.keys()) | set(p2.keys()):
            cg1 = p1.get(innovation_num)
            cg2 = p2.get(innovation_num)
            if cg1 is not None and cg2 is not None:
                if cg1.key != cg2.key:
                    new_gene = cg1.copy()
                else:
                    new_gene = cg1.crossover(cg2)
                self.connections[new_gene.key] = new_gene
            elif cg1 is not None:
                new_gene = cg1.copy()
                self.connections[new_gene.key] = new_gene
        for key, ng1 in parent1.nodes.items():
            ng2 = parent2.nodes.get(key)
            assert key not in self.nodes
            if ng2 is None:
                self.nodes[key] = ng1.copy()
            else:
                self.nodes[key] = ng1.crossover(ng2)

    ng.DefaultGenome.configure_crossover = fast_configure_crossover
    FAST_XOVER_STATUS = "patched"


_apply_fast_crossover()


def _env_list(name, cast=str):
    v = os.environ.get(name)
    return [cast(x.strip()) for x in v.split(",") if x.strip()] if v else None


# ======================================================================
# CONFIG
# ======================================================================

# FILL IN #1 -- Linux path to master_dataset.csv on this EC2 instance.
MASTER_CSV_PATH = os.environ.get("NEAT_MASTER", r"/home/ubuntu/master_dataset_32.csv")

LOCAL_OUTPUT_ROOT = Path(r"/home/ubuntu/NEAT_Local_Runs")

# FILL IN #2 -- your currently-running job's OUTPUT_DIR (from its
# run_manifest.json "output_dir" field, or its startup banner). This is
# what lets the new run skip your already-completed windows instead of
# redoing them. Leave as None only if you genuinely want a fresh run.
RESUME_FROM_DIR = os.environ.get("NEAT_RESUME") or None      # fresh run unless NEAT_RESUME=<output dir> is set

_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
if RESUME_FROM_DIR is not None:
    OUTPUT_DIR = Path(RESUME_FROM_DIR)
else:
    OUTPUT_DIR = LOCAL_OUTPUT_ROOT / f"NEAT32{os.environ.get('NEAT_TAG', '')}_sharpe_{_RUN_STAMP}"
CHARTS_DIR = OUTPUT_DIR / "charts"
NETVIZ_DIR = OUTPUT_DIR / "network_diagrams"
GENSTATS_DIR = OUTPUT_DIR / "generation_stats"

TICKERS    = _env_list("NEAT_TICKERS") or [
    "GILD", "NVDA", "WFC", "MRK", "KO", "CMCSA", "MS", "COP", "PEP", "QCOM", "FDX", "V", "ORCL", "COST",
    "OXY", "NKE", "NEE", "EBAY", "CAT", "TXN", "SLB", "DUK", "BX", "MMM", "CRM", "AMT", "TJX", "HAL",
    "BIIB", "ADBE", "NEM", "CI"]   # order must match the master dataset (pair columns follow this order)
START_DATE = "2009-01-01"
END_DATE   = "2023-12-31"

N_WINDOWS            = 11
NEAT_POPULATION_SIZE = 150
NEAT_GENERATIONS     = int(os.environ.get("NEAT_GENERATIONS", 100))
SEED                 = 42
SEEDS = _env_list("NEAT_SEEDS", int) or [42, 7, 123, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
COST_BPS             = 5
# Comparability switches (defaults reproduce the earlier runs exactly):
#   OUTPUT_MAP=native    each method as before (NEAT/GP-tree outputs unbounded, the 8 algorithms tanh-bounded)
#   OUTPUT_MAP=bounded   every method's outputs limited to [-1, 1] before the softmax (max 34.5% per stock with 15 stocks)
#   OUTPUT_MAP=free      no bound for any method: NEAT output nodes and the 8 algorithms' outputs are linear
#                        (their hidden nodes keep their activations); GP-tree outputs only clipped at +-30 for safety
# Inputs are used exactly as built by build_features (no normalisation of any kind).
OUTPUT_MAP = os.environ.get("OUTPUT_MAP", "free").lower()      # default: no cap for any method
if OUTPUT_MAP not in ("native", "bounded", "free"):
    raise SystemExit(f"OUTPUT_MAP must be native, bounded or free (got {OUTPUT_MAP!r})")
RISK_FREE_ANNUAL     = 0.0
N_CORES              = multiprocessing.cpu_count()

# CHANGED: this is now JOB-level concurrency, not the inner population-eval
# pool size (there is no inner pool anymore -- see train_neat_single_threaded
# below). Each job is single-threaded, so this should be close to N_CORES,
# not divided by anything. Leaving 2 cores of headroom for the OS/scheduler.
N_CONCURRENT_JOBS = int(os.environ.get("NEAT_JOBS", max(1, N_CORES - 2)))   # lower NEAT_JOBS if memory runs short

# Network-diagram scope -- unchanged from v5.
NETWORK_VIZ_SEED    = SEEDS[0]
NETWORK_VIZ_WINDOWS = set(range(1, N_WINDOWS + 1)) if os.environ.get("NEAT_VIZ", "0") == "1" else set()   # diagrams off by default (too large with 32 stocks)
WINDOWS_TO_RUN = set(_env_list("NEAT_WINDOWS", int) or [])  or None
JOB_TIMEOUT = int(os.environ.get("NEAT_JOB_TIMEOUT", 14 * 24 * 3600))

# ======================================================================
# FEATURE CONFIGURATIONS
# ======================================================================

# The three feature sets of the original ablation (names kept so earlier results stay valid):
#   PRICE_ONLY = lagged returns          CORR_ONLY = returns + pairwise correlations
#   FULL       = returns + correlations + SMA/EMA pair-agreement signals + sentiment
# Five more cells complete the price / correlation / sentiment ablation (run on request via NEAT_CONFIGS):
#   CORR_ALONE = correlations only       SENT_ONLY = sentiment only      PRICE_SENT = returns + sentiment
#   CORR_SENT  = correlations + sentiment            PRICE_CORR_SENT = returns + correlations + sentiment
_ORIGINAL_CONFIGS = {
    "PRICE_ONLY": {"price": True, "sentiment": False, "correlation": False, "sma_ema": False},
    "CORR_ONLY":  {"price": True, "sentiment": False, "correlation": True,  "sma_ema": False},
    "FULL":       {"price": True, "sentiment": True,  "correlation": True,  "sma_ema": True},
}
_EXTRA_CONFIGS = {
    "CORR_ALONE":      {"price": False, "sentiment": False, "correlation": True,  "sma_ema": False},
    "SENT_ONLY":       {"price": False, "sentiment": True,  "correlation": False, "sma_ema": False},
    "PRICE_SENT":      {"price": True,  "sentiment": True,  "correlation": False, "sma_ema": False},
    "CORR_SENT":       {"price": False, "sentiment": True,  "correlation": True,  "sma_ema": False},
    "PRICE_CORR_SENT": {"price": True,  "sentiment": True,  "correlation": True,  "sma_ema": False},
}
FEATURE_CONFIGS_ALL = {**_ORIGINAL_CONFIGS, **_EXTRA_CONFIGS}
FEATURE_CONFIGS = dict(_ORIGINAL_CONFIGS)               # default run = the original three
_cfgs = _env_list("NEAT_CONFIGS")
if _cfgs:
    _unknown = [c for c in _cfgs if c not in FEATURE_CONFIGS_ALL]
    if _unknown:
        raise SystemExit(f"Unknown NEAT_CONFIGS {_unknown}. Valid: {list(FEATURE_CONFIGS_ALL)}")
    FEATURE_CONFIGS = {k: v for k, v in FEATURE_CONFIGS_ALL.items() if k in _cfgs}

# ======================================================================
# FITNESS FUNCTION  (unchanged from v5)
# ======================================================================

def fitness_pure_sharpe(daily_returns: pd.Series, avg_turnover: float,
                         min_days: int = 20) -> float:
    r = daily_returns.dropna().values
    if len(r) < min_days:
        return -10.0
    if r.std(ddof=1) < 1e-10:
        return -10.0
    return float(np.clip(_sharpe(r), -3.0, 5.0))


def _sharpe(r: np.ndarray) -> float:
    std = r.std()
    if std < 1e-10:
        return -10.0
    return float(np.sqrt(252) * r.mean() / std)


def _sortino(r: np.ndarray) -> float:
    downside = r[r < 0].std()
    if downside < 1e-6:
        return 0.0
    return float((r.mean() * 252) / (downside * np.sqrt(252)))


def _ann_return(equity: np.ndarray, n: int) -> float:
    return float(equity[-1] ** (252 / n)) - 1.0


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(abs(dd.min()))


def sharpe_ratio(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna().values
    if len(r) < 2 or r.std() < 1e-10:
        return -10.0
    return _sharpe(r)


def performance_summary(daily_returns: pd.Series) -> dict:
    r = daily_returns.dropna()
    if r.empty:
        return {k: np.nan for k in
                ["Sharpe", "Sortino", "Calmar", "CumReturn",
                 "AnnReturn", "AnnVol", "MaxDrawdown"]}
    rv = r.values
    equity = np.cumprod(1 + rv)
    cum = equity[-1] - 1
    ann = _ann_return(equity, len(rv))
    vol = rv.std() * np.sqrt(252)
    mdd = _max_drawdown(equity)
    sh = _sharpe(rv)
    so = _sortino(rv)
    cal = ann / (mdd + 1e-12) if mdd > 1e-6 else np.nan
    return {
        "Sharpe": round(sh, 4), "Sortino": round(so, 4), "Calmar": round(cal, 4),
        "CumReturn": round(cum, 4), "AnnReturn": round(ann, 4),
        "AnnVol": round(vol, 4), "MaxDrawdown": round(-mdd, 4),
    }


# ======================================================================
# NEAT HELPERS  (unchanged from v5)
# ======================================================================

def map_outputs(out):
    """Output mapping applied before the softmax (see OUTPUT_MAP). 'bounded' clips to [-1, 1]: tanh output nodes are
    unchanged, relu/sigmoid/GP outputs get the same range as the tanh-bounded comparison algorithms."""
    if OUTPUT_MAP == "bounded":
        return np.clip(out, -1.0, 1.0)
    return out


def weights_from_outputs(raw_outputs: np.ndarray) -> np.ndarray:
    x = map_outputs(np.array(raw_outputs, dtype=float))
    x -= x.max()
    w = np.exp(x)
    w /= w.sum()
    return w


def simulate_slow(net, feat_values: np.ndarray, ret_values: np.ndarray,
             index: pd.DatetimeIndex):
    n_assets = ret_values.shape[1]
    prev_w = np.full(n_assets, 1.0 / n_assets)
    out_rets = np.zeros(len(feat_values))
    turnovers = np.zeros(len(feat_values))
    free = OUTPUT_MAP == "free"
    outs = set(net.output_nodes)
    for i in range(len(feat_values)):
        if free:      # per-day reference evaluation with LINEAR output nodes (same rule as simulate)
            vals = dict(zip(net.input_nodes, feat_values[i]))
            vals.update({k: 0.0 for k in net.output_nodes})
            raw = {k: 0.0 for k in net.output_nodes}
            for node, act, agg, bias, resp, links in net.node_evals:
                pre = bias + resp * agg([vals[j] * wt for j, wt in links])
                vals[node] = act(pre)
                if node in outs:
                    raw[node] = pre
            o = [raw[k] for k in net.output_nodes]
        else:
            o = net.activate(feat_values[i])
        w = weights_from_outputs(o)
        turnover = np.abs(w - prev_w).sum()
        turnovers[i] = turnover
        out_rets[i] = float(w @ ret_values[i]) - turnover * COST_BPS / 10000.0
        prev_w = w
    return pd.Series(out_rets, index=index), float(turnovers.mean())




def _act_vec(fn):
    name = getattr(fn, "__name__", "")
    if name == "tanh_activation":
        return lambda z: np.tanh(np.clip(2.5 * z, -60.0, 60.0))
    if name == "sigmoid_activation":
        return lambda z: 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60.0, 60.0)))
    if name == "relu_activation":
        return lambda z: np.where(z > 0.0, z, 0.0)
    raise RuntimeError(f"unsupported activation function for vectorised simulate: {name}")


def simulate(net, feat_values: np.ndarray, ret_values: np.ndarray,
             index: pd.DatetimeIndex, details: bool = False):
    """Vectorised over days; numerically identical to simulate_slow (same maths, no per-day Python loop)."""
    X = np.ascontiguousarray(feat_values, dtype=np.float64)
    T = X.shape[0]
    n_assets = ret_values.shape[1]
    evals = net.node_evals
    col = {ne[0]: j for j, ne in enumerate(evals)}
    in_col = {k: j for j, k in enumerate(net.input_nodes)}
    W = np.zeros((X.shape[1], len(evals)))
    hidden_links = []
    for j, (node, act, agg, bias, resp, links) in enumerate(evals):
        if getattr(agg, "__name__", "") != "sum_aggregation":
            raise RuntimeError("vectorised simulate supports sum aggregation only")
        hl = []
        for i, w in links:
            if i in in_col:
                W[in_col[i], j] += w
            elif i in col:
                hl.append((col[i], w))
        hidden_links.append(hl)
    Z = X @ W
    V = np.zeros((T, len(evals)))
    PRE = np.zeros((T, len(evals)))
    for j, (node, act, agg, bias, resp, links) in enumerate(evals):
        z = Z[:, j]
        if hidden_links[j]:
            cs = np.fromiter((c for c, _ in hidden_links[j]), dtype=int)
            ws = np.fromiter((w for _, w in hidden_links[j]), dtype=float)
            z = z + V[:, cs] @ ws
        PRE[:, j] = bias + resp * z
        V[:, j] = _act_vec(act)(PRE[:, j])
    out = np.zeros((T, n_assets))
    src = PRE if OUTPUT_MAP == "free" else V          # free: linear output nodes (no cap on any position)
    for a, k in enumerate(net.output_nodes):
        if k in col:
            out[:, a] = src[:, col[k]]
    out = map_outputs(out)
    x = out - out.max(axis=1, keepdims=True)
    w = np.exp(x)
    w /= w.sum(axis=1, keepdims=True)
    prev = np.vstack([np.full((1, n_assets), 1.0 / n_assets), w[:-1]])
    turnovers = np.abs(w - prev).sum(axis=1)
    out_rets = (w * ret_values).sum(axis=1) - turnovers * COST_BPS / 10000.0
    if details:
        return pd.Series(out_rets, index=index), float(turnovers.mean()), w, turnovers
    return pd.Series(out_rets, index=index), float(turnovers.mean())


def _check_fast_sim(net, feat_values, ret_values, index, n=40):
    """Safety net: the vectorised simulate must agree with the original per-day loop on the first n days."""
    if len(feat_values) == 0:
        raise RuntimeError("test window has 0 days -- the master dataset was not loaded correctly (check its dates)")
    a, ta = simulate(net, feat_values[:n], ret_values[:n], index[:n])
    b, tb = simulate_slow(net, feat_values[:n], ret_values[:n], index[:n])
    d = float(np.nanmax(np.abs(a.values - b.values)))
    if not (d < 1e-9 and abs(ta - tb) < 1e-9):
        raise RuntimeError(f"vectorised simulate disagrees with the original: max diff {d:.3e}, turnover {ta} vs {tb}")


def write_neat_config(n_inputs: int, n_outputs: int, config_path: str) -> None:
    # CHANGED: takes config_path explicitly now instead of writing to one
    # shared global NEAT_CONFIG_PATH. With jobs running concurrently, a
    # shared file would race -- job A (6 inputs) could get overwritten by
    # job B (57 inputs) before A ever reads it back. v5's version was safe
    # ONLY because jobs ran strictly one at a time; that assumption no
    # longer holds, so every job now gets its own private .ini file (see
    # run_ablation's window_jobs construction below).
    cfg = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 1000.0
no_fitness_termination = True
pop_size               = {NEAT_POPULATION_SIZE}
reset_on_extinction    = True

[DefaultGenome]
num_inputs             = {n_inputs}
num_hidden             = 8
num_outputs            = {n_outputs}
initial_connection     = full_direct
feed_forward           = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob          = 0.5
conn_delete_prob       = 0.3
node_add_prob          = 0.2
node_delete_prob       = 0.2
activation_default     = tanh
activation_options     = tanh sigmoid relu
activation_mutate_rate = 0.1
aggregation_default    = sum
aggregation_options    = sum
aggregation_mutate_rate = 0.0
bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_max_value         = 30.0
bias_min_value         = -30.0
bias_mutate_power      = 0.5
bias_mutate_rate       = 0.7
bias_replace_rate      = 0.1
response_init_mean     = 1.0
response_init_stdev    = 0.0
response_max_value     = 30.0
response_min_value     = -30.0
response_mutate_power  = 0.0
response_mutate_rate   = 0.0
response_replace_rate  = 0.0
weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_max_value       = 30
weight_min_value       = -30
weight_mutate_power    = 0.5
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.1
enabled_default        = True
enabled_mutate_rate    = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func   = max
max_stagnation         = 15
species_elitism        = 2

[DefaultReproduction]
elitism                = 2
survival_threshold     = 0.2
"""
    with open(config_path, "w") as f:
        f.write(cfg)


# ======================================================================
# NEAT NETWORK TOPOLOGY DIAGRAM  (unchanged from v5)
# ======================================================================

def draw_net(config, genome, filepath: Path, input_names, output_names):
    try:
        import graphviz
    except ImportError:
        print(f"    [viz] 'graphviz' python package not installed -- "
              f"skipping {filepath.name} (pip install graphviz)")
        return
    try:
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="LR")

        input_keys = list(config.genome_config.input_keys)
        output_keys = list(config.genome_config.output_keys)
        name_map = {}
        for k, name in zip(input_keys, input_names):
            name_map[k] = name
            dot.node(name, shape="box", style="filled", fillcolor="lightblue")
        for k, name in zip(output_keys, output_names):
            name_map[k] = name
            dot.node(name, shape="box", style="filled", fillcolor="lightgreen")
        for k in genome.nodes:
            if k in name_map:
                continue
            name_map[k] = f"h{k}"
            dot.node(f"h{k}", shape="circle", style="filled", fillcolor="white")

        for cg in genome.connections.values():
            if not cg.enabled:
                continue
            a, b = cg.key
            color = "darkgreen" if cg.weight > 0 else "firebrick"
            width = str(0.3 + min(abs(cg.weight), 5.0) * 0.4)
            dot.edge(name_map[a], name_map[b], color=color, penwidth=width)

        dot.render(str(filepath.with_suffix("")), cleanup=True)
        print(f"    [viz] wrote {filepath.name}")
    except Exception as e:
        print(f"    [viz] network diagram failed ({e}) -- continuing run")


# ======================================================================
# GENERATION-LEVEL EVOLUTION TRACKING  (unchanged from v5)
# ======================================================================

class GenerationTracker(neat.reporting.BaseReporter):
    def __init__(self):
        self.records = []
        self._gen = 0

    def start_generation(self, generation):
        self._gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        self.records.append({
            "generation": self._gen,
            "best_fitness": best_genome.fitness,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else np.nan,
            "num_species": len(species.species),
        })


def plot_convergence(records: list[dict], filepath: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    [gen-stats] matplotlib not installed -- skipping {filepath.name}")
        return
    df = pd.DataFrame(records)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["generation"], df["best_fitness"], label="Best fitness", color="darkgreen")
    ax1.plot(df["generation"], df["mean_fitness"], label="Mean fitness", color="steelblue")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["num_species"], label="Num species",
              color="darkorange", linestyle=":")
    ax2.set_ylabel("Number of species")
    ax2.legend(loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ======================================================================
# SINGLE-THREADED NEAT TRAINING  (CHANGED: replaces the inner Pool)
# ======================================================================
# v5 divided the 150-genome population across an inner multiprocessing.Pool
# of N_CORES workers. That's the piece removed here: genome evaluation is
# now a plain Python loop, running entirely within whichever single worker
# process this job was dispatched to. The parallelism moved to the OUTER
# level instead -- see run_jobs_parallel below, which runs ~N_CONCURRENT_JOBS
# of these single-threaded trainings at once.

def train_neat_single_threaded(train_features: pd.DataFrame, train_returns: pd.DataFrame,
                                config_path: str, track_stats: bool = False):
    feat_v = train_features.values
    ret_v = train_returns.values
    index = train_features.index

    n_inputs = train_features.shape[1]
    n_outputs = train_returns.shape[1]
    write_neat_config(n_inputs, n_outputs, config_path)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path,
    )

    tracker = GenerationTracker() if track_stats else None

    def eval_genomes(genomes, cfg):
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            rets, avg_turnover = simulate(net, feat_v, ret_v, index)
            genome.fitness = fitness_pure_sharpe(rets, avg_turnover)

    pop = neat.Population(config)
    if tracker is not None:
        pop.add_reporter(tracker)
    best = pop.run(eval_genomes, NEAT_GENERATIONS)

    n_nodes = len(best.nodes)
    n_conns = len([c for c in best.connections.values() if c.enabled])
    gen_records = tracker.records if tracker is not None else []
    return neat.nn.FeedForwardNetwork.create(best, config), n_nodes, n_conns, best, config, gen_records


# ======================================================================
# PER-JOB EXECUTION + JOB-LEVEL PARALLEL SCHEDULER
# ======================================================================
# CHANGED: replaces v5's run_isolated_job (ctx.Process + Queue, one job at
# a time) with a Pool(maxtasksperchild=1)-based scheduler -- the same
# pattern already proven in ppo_sharpe_comparison.py. maxtasksperchild=1
# means every pool worker is torn down and replaced with a fresh process
# after exactly one job, which is exactly v5's original per-job process
# isolation guarantee (the memory-leak fix), just achieved for free by the
# Pool itself instead of manual ctx.Process/Queue bookkeeping -- and it
# gives you job-LEVEL concurrency in the same step.

def _run_single_neat_job(job_args: dict) -> dict:
    fc_name = job_args["fc_name"]
    seed = job_args["seed"]
    w_idx = job_args["w_idx"]
    t_job = time.time()
    try:
        train_feat = job_args["train_feat"]
        test_feat = job_args["test_feat"]
        train_ret = job_args["train_ret"]
        test_ret = job_args["test_ret"]
        viz_this_run = job_args["viz_this_run"]
        netviz_dir = job_args["netviz_dir"]
        genstats_dir = job_args["genstats_dir"]
        config_path = job_args["config_path"]

        random.seed(seed)
        np.random.seed(seed)

        best_net, n_nodes, n_conns, best_genome, neat_config, gen_records = \
            train_neat_single_threaded(train_feat, train_ret, config_path,
                                        track_stats=viz_this_run)
        oos_ret, oos_turnover = simulate(best_net, test_feat.values,
                                          test_ret.values, test_feat.index)
        _check_fast_sim(best_net, test_feat.values, test_ret.values, test_feat.index)
        perf = performance_summary(oos_ret)

        if viz_this_run:
            input_names = list(train_feat.columns)
            diagram_path = netviz_dir / f"{fc_name}_window{w_idx}_seed{seed}.png"
            draw_net(neat_config, best_genome, diagram_path, input_names, TICKERS)

            tag = f"{fc_name}_window{w_idx}_seed{seed}"
            pd.DataFrame(gen_records).to_csv(
                genstats_dir / f"{tag}_generations.csv", index=False)
            plot_convergence(gen_records, genstats_dir / f"{tag}_convergence.png",
                              title=f"{fc_name} | Window {w_idx} | Seed {seed}")

        try:
            Path(config_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "ok": True, "fc_name": fc_name, "seed": seed, "w_idx": w_idx,
            "perf": perf, "n_nodes": n_nodes, "n_conns": n_conns,
            "oos_turnover": oos_turnover, "oos_ret": oos_ret,
            "secs": time.time() - t_job,
        }
    except Exception as e:
        import traceback
        return {
            "ok": False, "fc_name": fc_name, "seed": seed, "w_idx": w_idx,
            "error": str(e), "traceback": traceback.format_exc(),
        }


def run_jobs_parallel(job_args_list: list[dict], max_attempts: int = 2,
                       timeout_seconds: int = JOB_TIMEOUT) -> dict:
    """Runs every job in job_args_list, up to N_CONCURRENT_JOBS at a time,
    each in its own fresh OS process (maxtasksperchild=1). Retries a job
    with a fresh process if it produces no result within timeout_seconds
    (dead/stuck child) -- same reasoning as v5's run_isolated_job. A job
    that DID run and reported a genuine failure (ok=False) raises
    immediately rather than retrying, since retrying won't fix a real bug.

    timeout_seconds defaults to 90 minutes, not v5's 30 -- a single-threaded
    job on a big window (late windows have far more training days) can
    genuinely take much longer than a job that used to have 64 (if
    inefficient) workers behind it. Raise this further if early jobs on
    window 4+ are timing out legitimately rather than actually hanging.
    """
    ctx = multiprocessing.get_context("spawn")
    pending = list(job_args_list)
    results = {}
    attempt_count = {}
    while pending:
        pool = ctx.Pool(processes=N_CONCURRENT_JOBS, maxtasksperchild=1)
        async_map = {(ja["fc_name"], ja["seed"]): (ja, pool.apply_async(_run_single_neat_job, (ja,)))
                     for ja in pending}
        still_pending = []
        for key, (ja, ar) in async_map.items():
            try:
                result = ar.get(timeout=timeout_seconds)
            except Exception as e:
                attempt_count[key] = attempt_count.get(key, 0) + 1
                if attempt_count[key] >= max_attempts:
                    pool.terminate(); pool.join()
                    raise RuntimeError(
                        f"NEAT job (config={key[0]}, seed={key[1]}) failed to produce "
                        f"a result after {max_attempts} attempts: {e}")
                print(f"    [retry] job (config={key[0]}, seed={key[1]}) produced no "
                      f"result -- will respawn.")
                still_pending.append(ja)
                continue
            if not result["ok"]:
                pool.terminate(); pool.join()
                raise RuntimeError(
                    f"NEAT job failed (seed={result['seed']}, config={result['fc_name']}, "
                    f"window={result['w_idx']}): {result['error']}\n{result['traceback']}")
            results[key] = result
            perf = result["perf"]
            print(f"    [done] config={key[0]:<11} seed={key[1]:<3} Sharpe={perf['Sharpe']:>6.2f}  "
                  f"CumRet={perf['CumReturn']:>8.2%}  MaxDD={perf['MaxDrawdown']:>8.2%}  "
                  f"AvgTurnover={result['oos_turnover']:.3f}  Nodes={result['n_nodes']}  "
                  f"Conns={result['n_conns']}  Secs={result['secs']:.0f}")
        pool.terminate(); pool.join()
        pending = still_pending
    return results


# ======================================================================
# BENCHMARKS  (unchanged from v5)
# ======================================================================

def drift_turnover(W, R) -> np.ndarray:
    """Daily turnover actually traded: sum|w_t - (w_(t-1) drifted with day t-1 returns)|; day 1 starts from 1/N."""
    W = np.asarray(W, dtype=float)
    R = np.nan_to_num(np.asarray(R, dtype=float))
    n = W.shape[1]
    pre = np.empty_like(W)
    pre[0] = 1.0 / n
    if len(W) > 1:
        g = W[:-1] * (1.0 + R[:-1])
        pre[1:] = g / g.sum(axis=1, keepdims=True)
    return np.abs(W - pre).sum(axis=1)


def net_of_costs(W, R, index, bps=None):
    """ONE cost rule for every strategy (NEAT, the 9 algorithms, Equal-Weight, Markowitz):
    bps per unit of drift turnover. Returns (net daily returns, daily turnover)."""
    bps = COST_BPS if bps is None else bps
    W = np.asarray(W, dtype=float)
    R = np.asarray(R, dtype=float)
    to = drift_turnover(W, R)
    return pd.Series((W * R).sum(axis=1) - to * bps / 10000.0, index=index), to


def equal_weight_returns(test_returns: pd.DataFrame) -> pd.Series:
    """1/N rebalanced daily, paying COST_BPS on the daily rebalancing trades."""
    n = test_returns.shape[1]
    W = np.tile(np.full(n, 1.0 / n), (len(test_returns), 1))
    return net_of_costs(W, test_returns.values, test_returns.index)[0]


def markowitz_weights(train_returns: pd.DataFrame) -> np.ndarray:
    """Long-only maximum-Sharpe weights fitted on the training window (same optimiser as before)."""
    mean_ret = train_returns.mean().values
    cov = train_returns.cov().values
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        port_ret = w @ mean_ret
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 1e6
        return -(port_ret - RISK_FREE_ANNUAL / 252) / port_vol

    res = minimize(neg_sharpe, w0, bounds=[(0.0, 1.0)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                    method="SLSQP")
    return res.x if res.success else w0


def markowitz_returns(train_returns: pd.DataFrame, test_returns: pd.DataFrame) -> pd.Series:
    """Max-Sharpe weights held as a daily-rebalanced target, paying COST_BPS on every rebalancing trade."""
    w = markowitz_weights(train_returns)
    return net_of_costs(np.tile(w, (len(test_returns), 1)), test_returns.values, test_returns.index)[0]


# ======================================================================
# FEATURE BUILDER  (unchanged from v5)
# ======================================================================

def polarity(label) -> int:
    return {"positive": 1, "negative": -1, "neutral": 0}.get(str(label).lower(), 0)


def pair_list(tickers):
    return list(itertools.combinations(tickers, 2))


def build_features(master: pd.DataFrame, tickers: list[str], cfg: dict):
    pairs = pair_list(tickers)
    ret_cols = [f"{t}_Return" for t in tickers]
    asset_ret = master[ret_cols].copy()
    asset_ret.columns = tickers

    feat = pd.DataFrame(index=master.index)
    if cfg.get("price", True):
        for t in tickers:
            feat[f"lag_ret_{t}"] = asset_ret[t]
    if cfg["correlation"]:
        for a, b in pairs:
            feat[f"Corr_{a}_vs_{b}"] = master[f"Corr_{a}_vs_{b}"]
    if cfg["sma_ema"]:
        for a, b in pairs:
            feat[f"SMA_{a}_vs_{b}"] = master[f"SMA_{a}_vs_{b}"]
            feat[f"EMA_{a}_vs_{b}"] = master[f"EMA_{a}_vs_{b}"]
    if cfg["sentiment"]:
        for t in tickers:
            label = master[f"{t}_Sentiment_Label"]
            score = master[f"{t}_Sentiment_Score"]
            signed = score * label.map(polarity)
            filled = signed.ffill().fillna(0.0)
            is_new = signed.notna()
            days_since = (~is_new).groupby(is_new.cumsum()).cumcount()
            filled = filled * (0.92 ** days_since)
            feat[f"lag_sent_{t}"] = filled

    feat = feat.shift(1).fillna(0.0)
    return feat, asset_ret


# ======================================================================
# CHARTS  (unchanged from v5)
# ======================================================================

def make_charts(returns_df: pd.DataFrame, summary_df: pd.DataFrame,
                 final_df: pd.DataFrame, neat_keys: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[charts] matplotlib not installed -- skipping charts (pip install matplotlib)")
        return

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for fc_name in neat_keys:
        col = f"{fc_name}_seed{NETWORK_VIZ_SEED}"
        if col in returns_df.columns:
            equity = (1 + returns_df[col].dropna()).cumprod()
            plt.plot(equity.index, equity.values, label=f"{fc_name} (seed {NETWORK_VIZ_SEED})")
    for bm in ["EqualWeight", "Markowitz"]:
        if bm in returns_df.columns:
            equity = (1 + returns_df[bm].dropna()).cumprod()
            plt.plot(equity.index, equity.values, label=bm, linestyle="--")
    plt.title("Equity Curves (growth of $1, stitched OOS)")
    plt.xlabel("Date"); plt.ylabel("Portfolio Value")
    plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS_DIR / "equity_curves.png", dpi=150)
    plt.close()

    sharpe_cols = [f"{fc}_Sharpe" for fc in neat_keys] + \
                  ["EqualWeight_Sharpe", "Markowitz_Sharpe"]
    window_means = summary_df.groupby("Window")[sharpe_cols].mean()
    ax = window_means.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Mean Sharpe Ratio by Window")
    ax.set_xlabel("Window"); ax.set_ylabel("Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "sharpe_by_window.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(final_df["Strategy"], final_df["Sharpe_mean"])
    plt.title("Final Aggregate Sharpe Ratio by Strategy")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "final_sharpe_bar.png", dpi=150)
    plt.close()

    print(f"[charts] wrote 3 PNGs -> {CHARTS_DIR}")


# ======================================================================
# MAIN ABLATION RUNNER
# ======================================================================

def _load_resume_state(resume_dir: Path, neat_keys: list[str]):
    """Unchanged from v5: trusts only WHOLE completed windows (all 15
    seeds present); a partially-done window is dropped and fully redone."""
    summary_path = resume_dir / "ablation_window_summary.csv"
    returns_path = resume_dir / "ablation_oos_returns.csv"
    empty = ([], {k: {s: [] for s in SEEDS} for k in neat_keys},
             {"EqualWeight": [], "Markowitz": []}, set())
    if not summary_path.exists() or not returns_path.exists():
        print(f"[resume] {resume_dir} has no usable checkpoint yet -- starting fresh.")
        return empty

    old_summary = pd.read_csv(summary_path, parse_dates=["TestStart", "TestEnd"])
    old_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    all_pairs = {(int(r["Seed"]), int(r["Window"])) for _, r in old_summary.iterrows()}
    done_windows = {w for w in range(1, N_WINDOWS + 1)
                    if all((s, w) in all_pairs for s in SEEDS)}
    partial_windows = {w for (_, w) in all_pairs} - done_windows
    if partial_windows:
        print(f"[resume] windows {sorted(partial_windows)} were only partially "
              f"completed in {resume_dir} -- redoing them fully rather than "
              f"trusting partial data (resume only trusts whole windows).")

    old_summary = old_summary[old_summary["Window"].isin(done_windows)]
    window_rows = old_summary.to_dict("records")

    all_daily_seeded = {k: {s: [] for s in SEEDS} for k in neat_keys}
    all_daily_bm = {"EqualWeight": [], "Markowitz": []}
    seen_windows = set()
    for _, r in old_summary.sort_values(["Window", "Seed"]).iterrows():
        s, w = int(r["Seed"]), int(r["Window"])
        start, end = r["TestStart"], r["TestEnd"]
        for fc in neat_keys:
            col = f"{fc}_seed{s}"
            if col in old_returns.columns:
                all_daily_seeded[fc][s].append(old_returns.loc[start:end, col])
        if w not in seen_windows:
            seen_windows.add(w)
            for bm in ["EqualWeight", "Markowitz"]:
                if bm in old_returns.columns:
                    all_daily_bm[bm].append(old_returns.loc[start:end, bm])

    print(f"[resume] loaded {len(done_windows)} fully completed window(s) "
          f"({sorted(done_windows)}) from {resume_dir} -- these will be skipped.")
    return window_rows, all_daily_seeded, all_daily_bm, done_windows


def run_ablation(master: pd.DataFrame) -> None:
    index = master.index
    blocks = np.array_split(np.arange(len(index)), N_WINDOWS + 1)

    neat_keys = list(FEATURE_CONFIGS.keys())

    if RESUME_FROM_DIR is not None:
        window_rows, all_daily_seeded, all_daily_bm, done_windows = \
            _load_resume_state(Path(RESUME_FROM_DIR), neat_keys)
    else:
        all_daily_seeded = {k: {s: [] for s in SEEDS} for k in neat_keys}
        all_daily_bm = {"EqualWeight": [], "Markowitz": []}
        window_rows = []
        done_windows = set()

    total_runs = len(neat_keys) * N_WINDOWS * len(SEEDS)

    summary_path = OUTPUT_DIR / "ablation_window_summary.csv"
    returns_path = OUTPUT_DIR / "ablation_oos_returns.csv"
    final_path = OUTPUT_DIR / "ablation_final_summary.csv"
    NETVIZ_DIR.mkdir(parents=True, exist_ok=True)
    GENSTATS_DIR.mkdir(parents=True, exist_ok=True)
    returns_cols = {}

    for w_idx in range(1, N_WINDOWS + 1):
        if w_idx in done_windows:
            print(f"\n[resume] window {w_idx}/{N_WINDOWS} already completed -- skipping")
            continue
        if WINDOWS_TO_RUN is not None and w_idx not in WINDOWS_TO_RUN:
            continue

        train_pos = np.concatenate(blocks[:w_idx])
        test_pos = blocks[w_idx]
        if len(test_pos) == 0:
            continue

        t_start = index[test_pos[0]].date()
        t_end = index[test_pos[-1]].date()

        print(f"\n{'=' * 65}")
        print(f"WINDOW {w_idx}/{N_WINDOWS} | train {len(train_pos)}d | "
              f"test {t_start} -> {t_end} ({len(test_pos)}d)")
        print(f"{'=' * 65}")

        _, asset_ret_bm = build_features(master, TICKERS, FEATURE_CONFIGS_ALL["FULL"])
        train_ret_bm = asset_ret_bm.iloc[train_pos]
        test_ret_bm = asset_ret_bm.iloc[test_pos]

        ew_ret = equal_weight_returns(test_ret_bm)
        mv_ret = markowitz_returns(train_ret_bm, test_ret_bm)
        ew_perf = performance_summary(ew_ret)
        mv_perf = performance_summary(mv_ret)
        all_daily_bm["EqualWeight"].append(ew_ret)
        all_daily_bm["Markowitz"].append(mv_ret)
        print(f"  [Benchmarks] EqualWeight Sharpe={ew_perf['Sharpe']:.2f} "
              f"(CumRet={ew_perf['CumReturn']:.2%})   "
              f"Markowitz Sharpe={mv_perf['Sharpe']:.2f} "
              f"(CumRet={mv_perf['CumReturn']:.2%})")

        # CHANGED: build ALL 45 jobs for this window up front, then dispatch
        # them together via run_jobs_parallel so up to N_CONCURRENT_JOBS run
        # at once -- v5 built and ran one job at a time inside this loop.
        window_jobs = []
        for seed in SEEDS:
            for fc_name, fc_flags in FEATURE_CONFIGS.items():
                feat, asset_ret = build_features(master, TICKERS, fc_flags)
                train_feat = feat.iloc[train_pos]
                test_feat = feat.iloc[test_pos]
                train_ret = asset_ret.iloc[train_pos]
                test_ret = asset_ret.iloc[test_pos]
                viz_this_run = (seed == NETWORK_VIZ_SEED and w_idx in NETWORK_VIZ_WINDOWS)
                config_path = str(OUTPUT_DIR / f"neat_config_{fc_name}_w{w_idx}_s{seed}.ini")
                window_jobs.append({
                    "fc_name": fc_name, "seed": seed, "w_idx": w_idx,
                    "train_feat": train_feat, "test_feat": test_feat,
                    "train_ret": train_ret, "test_ret": test_ret,
                    "viz_this_run": viz_this_run,
                    "netviz_dir": NETVIZ_DIR, "genstats_dir": GENSTATS_DIR,
                    "config_path": config_path,
                })

        print(f"  [launching] {len(window_jobs)} jobs for window {w_idx} across up to "
              f"{N_CONCURRENT_JOBS} concurrent processes ({len(train_pos)}d train) ...")

        window_results = run_jobs_parallel(window_jobs)

        for seed in SEEDS:
            row = {"Seed": seed, "Window": w_idx, "TestStart": t_start, "TestEnd": t_end,
                   "EqualWeight_CumReturn": ew_perf["CumReturn"],
                   "EqualWeight_Sharpe": ew_perf["Sharpe"],
                   "Markowitz_CumReturn": mv_perf["CumReturn"],
                   "Markowitz_Sharpe": mv_perf["Sharpe"]}
            for fc_name in neat_keys:
                result = window_results[(fc_name, seed)]
                perf = result["perf"]
                row[f"{fc_name}_CumReturn"] = perf["CumReturn"]
                row[f"{fc_name}_AnnReturn"] = perf["AnnReturn"]
                row[f"{fc_name}_Sharpe"] = perf["Sharpe"]
                row[f"{fc_name}_Sortino"] = perf["Sortino"]
                row[f"{fc_name}_Calmar"] = perf["Calmar"]
                row[f"{fc_name}_AnnVol"] = perf["AnnVol"]
                row[f"{fc_name}_MaxDD"] = perf["MaxDrawdown"]
                row[f"{fc_name}_AvgTurnover"] = round(result["oos_turnover"], 4)
                row[f"{fc_name}_NumNodes"] = result["n_nodes"]
                row[f"{fc_name}_NumConns"] = result["n_conns"]
                all_daily_seeded[fc_name][seed].append(result["oos_ret"])
            window_rows.append(row)

        # CHANGED: checkpoint once per WINDOW, not once per seed. Jobs now
        # finish out of order (concurrently), so a mid-window checkpoint
        # isn't a coherent concept the way it was when jobs ran strictly
        # sequentially. This loses nothing resume-wise: _load_resume_state
        # already only ever trusted a WHOLE completed window -- a
        # partially-done window was always fully redone regardless of how
        # many per-seed checkpoints existed inside it.
        pd.DataFrame(window_rows).to_csv(summary_path, index=False)
        returns_cols = {}
        for fc_name in neat_keys:
            for s in SEEDS:
                if all_daily_seeded[fc_name][s]:
                    returns_cols[f"{fc_name}_seed{s}"] = pd.concat(all_daily_seeded[fc_name][s])
        for bm_name in ["EqualWeight", "Markowitz"]:
            if all_daily_bm[bm_name]:
                returns_cols[bm_name] = pd.concat(all_daily_bm[bm_name])
        pd.DataFrame(returns_cols).to_csv(returns_path)
        print(f"  [checkpoint] window {w_idx} saved ({len(window_jobs)} jobs)")

    if not returns_cols:
        for fc_name in neat_keys:
            for s in SEEDS:
                if all_daily_seeded[fc_name][s]:
                    returns_cols[f"{fc_name}_seed{s}"] = pd.concat(all_daily_seeded[fc_name][s])
        for bm_name in ["EqualWeight", "Markowitz"]:
            if all_daily_bm[bm_name]:
                returns_cols[bm_name] = pd.concat(all_daily_bm[bm_name])

    summary_df = pd.DataFrame(window_rows)
    returns_df = pd.DataFrame(returns_cols)

    print("\n" + "=" * 75)
    print("FULL OOS AGGREGATE  (all windows stitched, mean +/- std across seeds)")
    print("=" * 75)
    final_rows = []
    for fc_name in neat_keys:
        seed_summaries = [performance_summary(returns_df[f"{fc_name}_seed{s}"]) for s in SEEDS]
        sharpes = [p["Sharpe"] for p in seed_summaries]
        mean_p = {k: float(np.mean([p[k] for p in seed_summaries])) for k in seed_summaries[0]}
        mean_p["CumReturn"] = float(np.median([p["CumReturn"] for p in seed_summaries]))
        final_rows.append({"Strategy": fc_name, "Sharpe_mean": round(np.mean(sharpes), 4),
                            "Sharpe_std": round(np.std(sharpes), 4),
                            **{k: mean_p[k] for k in mean_p if k != "Sharpe"}})
        print(f"  {fc_name:<12} Sharpe={np.mean(sharpes):>6.2f}+/-{np.std(sharpes):<5.2f}  "
              f"CumRet={mean_p['CumReturn']:>9.2%}  AnnRet={mean_p['AnnReturn']:>8.2%}  "
              f"Sortino={mean_p['Sortino']:>6.2f}  Calmar={mean_p['Calmar']:>6.2f}  "
              f"AnnVol={mean_p['AnnVol']:>7.2%}  MaxDD={mean_p['MaxDrawdown']:>8.2%}")
    for bm_name in ["EqualWeight", "Markowitz"]:
        p = performance_summary(returns_df[bm_name])
        final_rows.append({"Strategy": bm_name, "Sharpe_mean": p["Sharpe"], "Sharpe_std": 0.0,
                            **{k: p[k] for k in p if k != "Sharpe"}})
        print(f"  {bm_name:<12} Sharpe={p['Sharpe']:>6.2f}       "
              f"CumRet={p['CumReturn']:>9.2%}  AnnRet={p['AnnReturn']:>8.2%}  "
              f"Sortino={p['Sortino']:>6.2f}  Calmar={p['Calmar']:>6.2f}  "
              f"AnnVol={p['AnnVol']:>7.2%}  MaxDD={p['MaxDrawdown']:>8.2%}")

    final_df = pd.DataFrame(final_rows)

    summary_df.to_csv(summary_path, index=False)
    returns_df.to_csv(returns_path)
    final_df.to_csv(final_path, index=False)
    print(f"\n[ablation] wrote window summary    -> {summary_path}")
    print(f"[ablation] wrote OOS daily returns -> {returns_path}")
    print(f"[ablation] wrote aggregate summary -> {final_path}")

    xlsx_path = OUTPUT_DIR / "ablation_dissertation_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Aggregate Summary", index=False)
        summary_df.to_excel(writer, sheet_name="Window by Window", index=False)
        returns_df.to_excel(writer, sheet_name="Daily Returns", index=False)
    print(f"[ablation] wrote dissertation Excel -> {xlsx_path}")

    make_charts(returns_df, summary_df, final_df, neat_keys)
    print("\n[ablation] ALL DONE.")


def selftest_fast_crossover(n_in=40, n_out=10, pop=150, gens=6):
    """Runs the same small evolution with the original and the fast crossover; they must match exactly."""
    if FAST_XOVER_STATUS != "patched":
        return FAST_XOVER_STATUS
    import neat.genome as ng
    import tempfile
    fast = ng.DefaultGenome.configure_crossover
    old_pop = globals()["NEAT_POPULATION_SIZE"]

    def run(fn):
        ng.DefaultGenome.configure_crossover = fn
        globals()["NEAT_POPULATION_SIZE"] = pop
        path = os.path.join(tempfile.mkdtemp(), "selftest.ini")
        write_neat_config(n_in, n_out, path)
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, path)
        random.seed(123); np.random.seed(123)
        p = neat.Population(cfg)

        def ev(genomes, c):
            for _, g in genomes:
                g.fitness = math.tanh(0.02 * sum(cg.weight * (((k[0] * 31 + k[1] * 17) % 7) - 3)
                                                 for k, cg in g.connections.items() if cg.enabled))
        p.run(ev, gens)
        return sorted((gid, tuple(sorted((k, round(cg.weight, 12), cg.enabled)
                                          for k, cg in g.connections.items())))
                      for gid, g in p.population.items())

    try:
        a = run(_ORIGINAL_XOVER)
        b = run(fast)
    finally:
        ng.DefaultGenome.configure_crossover = fast
        globals()["NEAT_POPULATION_SIZE"] = old_pop
    return "patched and verified identical to the original" if a == b else "MISMATCH"


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(SEED)
    np.random.seed(SEED)

    if not Path(MASTER_CSV_PATH).exists():
        print(f"[FATAL] MASTER_CSV_PATH does not exist: {MASTER_CSV_PATH}")
        print(f"[FATAL] Fill in MASTER_CSV_PATH near the top of this file with the "
              f"real Linux path to master_dataset.csv on this box.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_timestamp": _RUN_STAMP,
        "fitness_mode": "pure_sharpe",
        "feature_configs": list(FEATURE_CONFIGS.keys()),
        "seeds": SEEDS,
        "n_windows": N_WINDOWS,
        "population_size": NEAT_POPULATION_SIZE,
        "generations": NEAT_GENERATIONS,
        "cost_bps": COST_BPS,
        "network_viz_seed": NETWORK_VIZ_SEED,
        "network_viz_windows": sorted(NETWORK_VIZ_WINDOWS),
        "generation_stats_scope": "same as network_viz (seed + windows above)",
        "master_csv_path": MASTER_CSV_PATH,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "job_isolation": "each (config,window,seed) job runs in its own spawned process "
                          "(Pool maxtasksperchild=1)",
        "parallelism": f"JOB-level: up to {N_CONCURRENT_JOBS} concurrent single-threaded "
                        f"jobs (was population-level in v5: 1 job at a time x {N_CORES} "
                        f"inner workers)",
        "resume_from_dir": RESUME_FROM_DIR,
        "tickers": TICKERS,
        "n_tickers": len(TICKERS),
        "windows_run": sorted(WINDOWS_TO_RUN) if WINDOWS_TO_RUN else "all",
        "script": "neat_sharpe_parallel_aws_32.py",
    }
    with open(OUTPUT_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 65)
    print("NEAT PORTFOLIO ABLATION STUDY -- JOB-LEVEL PARALLEL (pure Sharpe)")
    print("=" * 65)
    print(f"  Master CSV     : {MASTER_CSV_PATH}")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    print(f"  Resuming from  : {RESUME_FROM_DIR if RESUME_FROM_DIR else '(fresh run)'}")
    print(f"  Feature configs: {list(FEATURE_CONFIGS.keys())}")
    print(f"  Fitness        : PURE SHARPE (annualized Sharpe ratio, clipped [-3,5])")
    print(f"  Population     : {NEAT_POPULATION_SIZE}")
    print(f"  Generations    : {NEAT_GENERATIONS}")
    print(f"  Seeds          : {SEEDS}  ({len(SEEDS)} total)")
    print(f"  CPU cores      : {N_CORES}")
    print(f"  Concurrent jobs: {N_CONCURRENT_JOBS}  (CHANGED: job-level parallelism, "
          f"single-threaded per job -- was 1 job x {N_CORES} inner workers)")
    print(f"  Windows        : {N_WINDOWS}")
    print(f"  Total NEAT runs: {len(FEATURE_CONFIGS) * N_WINDOWS * len(SEEDS)}")
    print(f"  Job isolation  : Pool(maxtasksperchild=1) -- fresh process per job")
    print(f"  Checkpoints    : after every WINDOW (all jobs finish concurrently, "
          f"out of order)")
    print(f"  Network diagrams: seed={NETWORK_VIZ_SEED}, windows={sorted(NETWORK_VIZ_WINDOWS)}")
    print("=" * 65)

    print(f"\n[ablation] loading master dataset ...")
    master = pd.read_csv(MASTER_CSV_PATH, index_col=0, parse_dates=True)
    master = master.loc[(master.index >= START_DATE) & (master.index <= END_DATE)]
    print(f"[ablation] {len(master)} trading days loaded\n")

    run_ablation(master)
