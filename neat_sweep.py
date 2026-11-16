"""
neat_sweep.py -- runs NEAT (pure Sharpe) on several random stock draws with ONE flat job queue.
Needs neat_sharpe_parallel_aws_32.py in the same folder.
Environment switches (all optional): NEAT_MASTER, NEAT_N, NEAT_DRAWS, NEAT_CONFIGS, NEAT_WINDOWS,
NEAT_SEEDS, NEAT_GENERATIONS, NEAT_JOBS, NEAT_SWEEP_DIR, NEAT_AGG_ONLY=1
"""
import json, multiprocessing, os, pickle, random, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import neat_sharpe_parallel_aws_32 as base

# random stock draws (fixed seeds, always in master order). Choose the size with NEAT_N = 10 | 15 | 20 | 32
ALL32 = ["GILD", "NVDA", "WFC", "MRK", "KO", "CMCSA", "MS", "COP", "PEP", "QCOM", "FDX", "V", "ORCL", "COST", "OXY", "NKE", "NEE", "EBAY", "CAT", "TXN", "SLB", "DUK", "BX", "MMM", "CRM", "AMT", "TJX", "HAL", "BIIB", "ADBE", "NEM", "CI"]
DRAW_SETS = {
    10: {
        1: ["GILD", "WFC", "KO", "QCOM", "V", "COST", "NEE", "TXN", "NEM", "CI"],
        2: ["MRK", "MS", "COP", "CAT", "MMM", "CRM", "TJX", "BIIB", "ADBE", "CI"],
        3: ["GILD", "WFC", "MRK", "CMCSA", "COP", "ORCL", "COST", "TXN", "DUK", "AMT"],
        4: ["CMCSA", "COP", "FDX", "NKE", "NEE", "CRM", "AMT", "TJX", "BIIB", "ADBE"],
        5: ["GILD", "WFC", "PEP", "QCOM", "V", "TXN", "DUK", "BIIB", "ADBE", "CI"],
        6: ["NVDA", "MS", "COP", "PEP", "QCOM", "ORCL", "COST", "NEE", "EBAY", "BX"],
        7: ["NVDA", "PEP", "V", "NKE", "NEE", "TXN", "CRM", "TJX", "HAL", "CI"],
        8: ["WFC", "KO", "MS", "COST", "OXY", "TXN", "SLB", "CRM", "AMT", "HAL"],
        9: ["KO", "OXY", "NEE", "TXN", "DUK", "MMM", "AMT", "TJX", "ADBE", "CI"],
        10: ["CMCSA", "MS", "FDX", "V", "ORCL", "OXY", "CAT", "BX", "TJX", "ADBE"],
    },
    15: {
        1: ["WFC", "CMCSA", "QCOM", "FDX", "V", "ORCL", "NKE", "NEE", "EBAY", "CAT", "BX", "MMM", "CRM", "AMT", "BIIB"],
        2: ["GILD", "NVDA", "WFC", "KO", "COP", "PEP", "FDX", "COST", "NKE", "BX", "TJX", "HAL", "BIIB", "NEM", "CI"],
        3: ["GILD", "NVDA", "KO", "FDX", "ORCL", "NEE", "EBAY", "SLB", "BX", "MMM", "CRM", "TJX", "HAL", "ADBE", "NEM"],
        4: ["WFC", "MRK", "CMCSA", "COP", "V", "COST", "OXY", "NEE", "SLB", "BX", "MMM", "CRM", "TJX", "NEM", "CI"],
        5: ["MRK", "KO", "CMCSA", "COP", "PEP", "FDX", "V", "ORCL", "NKE", "CAT", "SLB", "DUK", "MMM", "BIIB", "ADBE"],
    },
    20: {
        1: ["GILD", "WFC", "COP", "PEP", "QCOM", "FDX", "V", "ORCL", "COST", "NKE", "EBAY", "TXN", "DUK", "BX", "MMM", "AMT", "TJX", "HAL", "NEM", "CI"],
        2: ["MRK", "KO", "CMCSA", "COP", "QCOM", "FDX", "V", "ORCL", "COST", "EBAY", "CAT", "TXN", "SLB", "CRM", "AMT", "HAL", "BIIB", "ADBE", "NEM", "CI"],
        3: ["WFC", "MRK", "KO", "MS", "COP", "QCOM", "FDX", "NKE", "NEE", "CAT", "TXN", "SLB", "DUK", "MMM", "CRM", "AMT", "HAL", "BIIB", "NEM", "CI"],
        4: ["GILD", "NVDA", "MRK", "MS", "PEP", "FDX", "V", "COST", "OXY", "EBAY", "TXN", "DUK", "BX", "MMM", "CRM", "AMT", "TJX", "HAL", "BIIB", "ADBE"],
        5: ["NVDA", "KO", "MS", "PEP", "QCOM", "V", "COST", "OXY", "EBAY", "TXN", "SLB", "DUK", "BX", "MMM", "CRM", "AMT", "HAL", "BIIB", "ADBE", "NEM"],
    },
    32: {1: ALL32},
}
N_STOCKS = int(os.environ.get("NEAT_N", 10))
DRAWS = DRAW_SETS[N_STOCKS]
_sel = base._env_list("NEAT_DRAWS", int)
if _sel:
    DRAWS = {k: v for k, v in DRAWS.items() if k in _sel}
FC_PRIORITY = {"FULL": 0, "PRICE_CORR_SENT": 0, "CORR_ONLY": 1, "CORR_ALONE": 1, "CORR_SENT": 1,
               "PRICE_SENT": 2, "SENT_ONLY": 2, "PRICE_ONLY": 2}      # longest jobs first
_MASTER = None


def _init_worker(pickle_path):
    global _MASTER
    _MASTER = pd.read_pickle(pickle_path)


def _genome_to_dict(g, cfg):
    """Plain-Python copy of the winning network so it can be re-evaluated later without neat-python objects."""
    gc = cfg.genome_config
    return {"input_keys": list(gc.input_keys), "output_keys": list(gc.output_keys),
            "nodes": {int(k): {"bias": float(n.bias), "response": float(n.response),
                               "activation": str(n.activation), "aggregation": str(n.aggregation)}
                      for k, n in g.nodes.items()},
            "connections": [(int(k[0]), int(k[1]), float(c.weight), bool(c.enabled))
                            for k, c in g.connections.items()]}


def _job(spec):
    draw, tickers, fc, w, seed, out_dir = spec
    t0 = time.time()
    try:
        master = _MASTER
        if master is None or len(master) < 1000:
            raise RuntimeError(f"worker has a master dataset with only {0 if master is None else len(master)} rows")
        index = master.index
        blocks = np.array_split(np.arange(len(index)), base.N_WINDOWS + 1)
        train_pos = np.concatenate(blocks[:w])
        test_pos = blocks[w]
        feat, ret = base.build_features(master, list(tickers), base.FEATURE_CONFIGS[fc])
        random.seed(seed); np.random.seed(seed)
        cfg_path = str(Path(out_dir) / f"tmp_d{draw}_{fc}_w{w}_s{seed}.ini")
        best_net, n_nodes, n_conns, best_genome, neat_cfg, _r = base.train_neat_single_threaded(
            feat.iloc[train_pos], ret.iloc[train_pos], cfg_path, track_stats=False)
        test_feat, test_ret = feat.iloc[test_pos], ret.iloc[test_pos]
        if len(test_pos) == 0 or len(train_pos) == 0:
            raise RuntimeError(f"empty window (train {len(train_pos)}d, test {len(test_pos)}d)")
        base._check_fast_sim(best_net, test_feat.values, test_ret.values, test_feat.index)
        oos_ret, oos_turn, W_oos, TO_oos = base.simulate(best_net, test_feat.values, test_ret.values,
                                                           test_feat.index, details=True)
        gdict = _genome_to_dict(best_genome, neat_cfg)
        try:
            Path(cfg_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": True, "draw": draw, "fc": fc, "w": w, "seed": seed,
                "perf": base.performance_summary(oos_ret), "n_nodes": n_nodes, "n_conns": n_conns,
                "oos_turnover": oos_turn, "oos_ret": oos_ret, "secs": time.time() - t0,
                "weights": W_oos.astype(np.float32), "turnovers": TO_oos.astype(np.float32),
                "turnovers_drift": base.drift_turnover(W_oos, test_ret.values).astype(np.float32),
                "output_map": base.OUTPUT_MAP, "genome": gdict}
    except Exception as e:
        import traceback
        return {"ok": False, "draw": draw, "fc": fc, "w": w, "seed": seed,
                "error": str(e), "traceback": traceback.format_exc(), "secs": time.time() - t0}


def _result_path(res_dir, draw, fc, w, seed):
    return Path(res_dir) / f"d{draw}_{fc}_w{w}_s{seed}.pkl"


def _load_master_csv(path):
    """Reads the master dataset; accepts ISO dates and day-first dates (what Excel writes after a re-save)."""
    m = pd.read_csv(path, index_col=0, low_memory=False)
    raw = pd.Series(m.index.astype(str), index=m.index).str.strip()
    idx = pd.to_datetime(raw.str[:10], format="%Y-%m-%d", errors="coerce")
    if idx.isna().mean() > 0.5:
        idx = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        print(f"[master] WARNING: dates in {path} are not ISO (e.g. '{raw.iloc[0]}'); read as day-first.", flush=True)
    idx = pd.DatetimeIndex(idx.values)
    keep = ~pd.isna(idx)
    m = m.loc[keep]; m.index = idx[keep]; m.index.name = "Date"
    return m.sort_index()


def load_master():
    master = _load_master_csv(base.MASTER_CSV_PATH)
    return master.loc[(master.index >= base.START_DATE) & (master.index <= base.END_DATE)]


def aggregate(out_dir, master):
    out_dir = Path(out_dir); res_dir = out_dir / "results"
    index = master.index
    blocks = np.array_split(np.arange(len(index)), base.N_WINDOWS + 1)
    windows = sorted(base.WINDOWS_TO_RUN) if base.WINDOWS_TO_RUN else list(range(1, base.N_WINDOWS + 1))
    fcs, seeds = list(base.FEATURE_CONFIGS.keys()), base.SEEDS
    sweep_rows = []
    for d, tickers in DRAWS.items():
        ddir = out_dir / f"draw{d}"; ddir.mkdir(exist_ok=True)
        asset_ret = master[[f"{t}_Return" for t in tickers]].copy(); asset_ret.columns = tickers
        rows, daily, bm = [], {fc: {s: [] for s in seeds} for fc in fcs}, {"EqualWeight": [], "Markowitz": []}
        done_windows = []
        for w in windows:
            paths = {(fc, s): _result_path(res_dir, d, fc, w, s) for fc in fcs for s in seeds}
            if not all(p.exists() for p in paths.values()):
                continue
            done_windows.append(w)
            train_pos, test_pos = np.concatenate(blocks[:w]), blocks[w]
            R_test = asset_ret.iloc[test_pos]
            ew = base.equal_weight_returns(R_test)                          # pays daily rebalancing costs
            mv = base.markowitz_returns(asset_ret.iloc[train_pos], R_test)  # pays daily rebalancing costs
            ewp, mvp = base.performance_summary(ew), base.performance_summary(mv)
            bm["EqualWeight"].append(ew); bm["Markowitz"].append(mv)
            for s in seeds:
                row = {"Draw": d, "Seed": s, "Window": w, "TestStart": index[test_pos[0]].date(),
                       "TestEnd": index[test_pos[-1]].date(),
                       "EqualWeight_CumReturn": ewp["CumReturn"], "EqualWeight_Sharpe": ewp["Sharpe"],
                       "Markowitz_CumReturn": mvp["CumReturn"], "Markowitz_Sharpe": mvp["Sharpe"]}
                for fc in fcs:
                    r = pickle.load(open(paths[(fc, s)], "rb"))
                    if "weights" in r:   # same cost rule as the benchmarks (drift turnover)
                        nr, _ = base.net_of_costs(r["weights"], R_test.values, R_test.index)
                    else:
                        nr = r["oos_ret"]
                    p = base.performance_summary(nr)
                    row[f"{fc}_Sharpe_simcost"] = r["perf"]["Sharpe"]
                    row.update({f"{fc}_CumReturn": p["CumReturn"], f"{fc}_AnnReturn": p["AnnReturn"],
                                f"{fc}_Sharpe": p["Sharpe"], f"{fc}_Sortino": p["Sortino"],
                                f"{fc}_Calmar": p["Calmar"], f"{fc}_AnnVol": p["AnnVol"],
                                f"{fc}_MaxDD": p["MaxDrawdown"], f"{fc}_AvgTurnover": round(r["oos_turnover"], 4),
                                f"{fc}_NumNodes": r["n_nodes"], f"{fc}_NumConns": r["n_conns"]})
                    daily[fc][s].append(nr)
                rows.append(row)
        if not rows:
            print(f"[aggregate] draw {d}: no complete windows yet"); continue
        cols = {}
        for fc in fcs:
            for s in seeds:
                if daily[fc][s]:
                    cols[f"{fc}_seed{s}"] = pd.concat(daily[fc][s])
        for k in bm:
            cols[k] = pd.concat(bm[k])
        returns_df, summary_df = pd.DataFrame(cols), pd.DataFrame(rows)
        final = []
        for fc in fcs:
            sp = [base.performance_summary(returns_df[f"{fc}_seed{s}"]) for s in seeds]
            sh = [p["Sharpe"] for p in sp]
            mp = {k: float(np.mean([p[k] for p in sp])) for k in sp[0]}
            mp["CumReturn"] = float(np.median([p["CumReturn"] for p in sp]))
            final.append({"Strategy": fc, "Sharpe_mean": round(np.mean(sh), 4), "Sharpe_std": round(np.std(sh), 4),
                          **{k: v for k, v in mp.items() if k != "Sharpe"}})
        for k in ("EqualWeight", "Markowitz"):
            p = base.performance_summary(returns_df[k])
            final.append({"Strategy": k, "Sharpe_mean": p["Sharpe"], "Sharpe_std": 0.0,
                          **{kk: v for kk, v in p.items() if kk != "Sharpe"}})
        final_df = pd.DataFrame(final)
        summary_df.to_csv(ddir / "ablation_window_summary.csv", index=False)
        returns_df.to_csv(ddir / "ablation_oos_returns.csv")
        final_df.to_csv(ddir / "ablation_final_summary.csv", index=False)
        for _, r in final_df.iterrows():
            sweep_rows.append({"Draw": d, "Strategy": r["Strategy"], "Sharpe_mean": r["Sharpe_mean"],
                               "Sharpe_std_over_seeds": r["Sharpe_std"], "AnnReturn": r["AnnReturn"],
                               "AnnVol": r["AnnVol"], "MaxDrawdown": r["MaxDrawdown"],
                               "windows_complete": f"{len(done_windows)}/{len(windows)}"})
        print(f"[aggregate] draw {d}: {len(done_windows)}/{len(windows)} windows complete -> {ddir}")
    if sweep_rows:
        sw = pd.DataFrame(sweep_rows)
        sw.to_csv(out_dir / "sweep_by_draw.csv", index=False)
        across = sw.groupby("Strategy")["Sharpe_mean"].agg(["mean", "std", "min", "max", "count"]).round(4)
        across.to_csv(out_dir / "sweep_across_draws.csv")
        print("\nSharpe across draws (mean of per-draw means):\n", across.to_string())


def main():
    multiprocessing.freeze_support()
    master = load_master()
    print(f"[master] {base.MASTER_CSV_PATH}: {len(master)} days"
          + (f", {master.index.min().date()} -> {master.index.max().date()}" if len(master) else "")
          + f", {master.shape[1]} columns", flush=True)
    if len(master) < 1000:
        sys.exit(f"STOP: the master dataset has only {len(master)} rows after the date filter "
                 f"({base.START_DATE} to {base.END_DATE}). Check the file at NEAT_MASTER ({base.MASTER_CSV_PATH}).")
    order = [c[:-len("_Price")] for c in master.columns if c.endswith("_Price")]
    for d, tk in DRAWS.items():
        miss = [t for t in tk if t not in order]
        assert not miss, f"draw {d}: tickers missing from master: {miss}"
        assert [t for t in order if t in tk] == tk, f"draw {d}: tickers must follow master order"
    out_dir = Path(os.environ.get("NEAT_SWEEP_DIR") or
                   base.LOCAL_OUTPUT_ROOT / f"NEAT_sweep_N{N_STOCKS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    res_dir = out_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {out_dir}")
    if os.environ.get("NEAT_AGG_ONLY") == "1":
        aggregate(out_dir, master); return
    try:
        xs = base.selftest_fast_crossover()
    except Exception as e:
        xs = f"self-test could not run ({str(e)[:80]})"
    print(f"Fast crossover: {xs}", flush=True)
    if xs == "MISMATCH":
        sys.exit("Fast crossover does not match the original -- set NEAT_FAST_CROSSOVER=0 and tell Claude.")
    pkl = out_dir / "master_cached.pkl"; master.to_pickle(pkl)
    windows = sorted(base.WINDOWS_TO_RUN) if base.WINDOWS_TO_RUN else list(range(1, base.N_WINDOWS + 1))
    specs, skipped = [], 0
    for d, tk in DRAWS.items():
        for fc in base.FEATURE_CONFIGS:
            for w in windows:
                for s in base.SEEDS:
                    if _result_path(res_dir, d, fc, w, s).exists():
                        skipped += 1; continue
                    specs.append((d, tuple(tk), fc, w, s, str(out_dir)))
    specs.sort(key=lambda x: (FC_PRIORITY.get(x[2], 1), x[0], -x[3], x[4]))
    total = len(specs)
    json.dump({"n_stocks": N_STOCKS, "draws": DRAWS, "configs": list(base.FEATURE_CONFIGS), "seeds": base.SEEDS, "windows": windows,
               "generations": base.NEAT_GENERATIONS, "population": base.NEAT_POPULATION_SIZE,
               "cost_bps": base.COST_BPS, "output_map": base.OUTPUT_MAP, "fast_crossover": xs, "master": base.MASTER_CSV_PATH, "workers": base.N_CONCURRENT_JOBS,
               "script": "neat_sweep.py"}, open(out_dir / "sweep_manifest.json", "w"), indent=2)
    print(f"{total} jobs to run ({skipped} already done), {base.N_CONCURRENT_JOBS} workers, "
          f"{base.NEAT_GENERATIONS} generations, {len(DRAWS)} draw(s)", flush=True)
    t0, ndone, nerr = time.time(), 0, 0
    if total:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=base.N_CONCURRENT_JOBS, maxtasksperchild=1,
                      initializer=_init_worker, initargs=(str(pkl),)) as pool:
            for r in pool.imap_unordered(_job, specs, chunksize=1):
                ndone += 1
                if r["ok"]:
                    with open(_result_path(res_dir, r["draw"], r["fc"], r["w"], r["seed"]), "wb") as f:
                        pickle.dump(r, f)
                    msg = f"Sharpe={r['perf']['Sharpe']:>6.2f} conns={r['n_conns']:<5} job={r['secs']/60:5.1f}min"
                else:
                    nerr += 1
                    with open(out_dir / "errors.log", "a") as f:
                        f.write(f"draw{r['draw']} {r['fc']} w{r['w']} s{r['seed']}: {r['error']}\n{r['traceback']}\n")
                    msg = f"FAILED: {r['error'][:80]}"
                el = time.time() - t0
                eta = el / ndone * (total - ndone)
                print(f"[{ndone:>5}/{total}] draw{r['draw']} {r['fc']:<10} w{r['w']:<2} s{r['seed']:<3} {msg} | "
                      f"elapsed {el/3600:.2f}h, rough ETA {eta/3600:.1f}h", flush=True)
    print(f"\nFinished: {ndone} jobs run, {nerr} failed. Building summaries ...", flush=True)
    aggregate(out_dir, master)
    if nerr:
        print(f"{nerr} jobs failed -- see {out_dir/'errors.log'}; fix and re-run with NEAT_SWEEP_DIR={out_dir} to retry only those.")
    print("DONE.")


if __name__ == "__main__":
    main()
