"""
export_daily.py -- builds ONE compact file of daily out-of-sample returns for the model-confidence-set / SPA tests.

For every strategy (NEAT x 8 feature sets, 9 algorithms x 8 feature sets, Equal-Weight, Markowitz) and every draw,
it rebuilds the daily net returns of each run from the saved weights with the SAME cost rule for everyone
(5 bp per unit of drift turnover), averages the 5 seeds, and stitches the 11 test windows together.

Output: <NEAT runs root>\\daily_returns_by_draw.csv.gz   (columns  "<strategy>|d<draw>",  rows = test days)

Run from the files-v4 folder:   python export_daily.py
"""
import os
os.environ.setdefault("NEAT_N", "15")
os.environ.setdefault("NEAT_MASTER", r"C:\Users\rahul\Downloads\master_dataset_32_clean.csv")
os.environ.setdefault("OUTPUT_MAP", "free")
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import neat_sharpe_parallel_aws_32 as base
import neat_sweep as ns

ROOT = Path(os.environ.get("RUNS_ROOT", r"C:\Users\rahul\neat_runs"))
NEAT_DIR = Path(os.environ.get("NEAT_DIR", ROOT / "A5_n15"))
ALGO_DIR = Path(os.environ.get("ALGO_DIR", ROOT / "B3_algos_n15"))
OUT = Path(os.environ.get("OUT_FILE", ROOT / "daily_returns_by_draw.csv.gz"))
CFGS = ["PRICE_ONLY", "CORR_ONLY", "FULL", "CORR_ALONE", "SENT_ONLY", "PRICE_SENT", "CORR_SENT", "PRICE_CORR_SENT"]
ALGOS = ["PSO", "CMAES", "DE", "GA", "ABC", "GWO", "FIREFLY", "NSGA2", "GPTREE"]

man = json.load(open(ALGO_DIR / "algos_manifest.json"))
draws = {int(k): v for k, v in man["draws"].items()}
seeds, windows = man["seeds"], man["windows"]
master = ns.load_master()
blocks = np.array_split(np.arange(len(master)), base.N_WINDOWS + 1)
print(f"[export] master {len(master)} days; draws {list(draws)}; seeds {seeds}; windows {windows[0]}..{windows[-1]}", flush=True)

cols, missing = {}, 0
for d, tk in draws.items():
    R_all = master[[f"{t}_Return" for t in tk]]
    R_all.columns = tk
    strat_paths = {f"NEAT_{c}": (NEAT_DIR / "results", "d{d}_" + c + "_w{w}_s{s}.pkl") for c in CFGS}
    strat_paths.update({f"{a}_{c}": (ALGO_DIR / "results", "d{d}_" + a + "__" + c + "_w{w}_s{s}.pkl")
                        for a in ALGOS for c in CFGS})
    pieces = {k: [] for k in list(strat_paths) + ["EqualWeight", "Markowitz"]}
    for w in windows:
        tr, te = np.concatenate(blocks[:w]), blocks[w]
        Rte = R_all.iloc[te]
        pieces["EqualWeight"].append(base.equal_weight_returns(Rte))
        pieces["Markowitz"].append(base.markowitz_returns(R_all.iloc[tr], Rte))
        for name, (folder, pat) in strat_paths.items():
            runs = []
            for s in seeds:
                f = folder / pat.format(d=d, w=w, s=s)
                if not f.exists():
                    missing += 1
                    continue
                W = pickle.load(open(f, "rb"))["weights"]
                runs.append(base.net_of_costs(W, Rte.values, Rte.index)[0])
            if runs:
                pieces[name].append(pd.concat(runs, axis=1).mean(axis=1))
    for name, p in pieces.items():
        if len(p) == len(windows):
            cols[f"{name}|d{d}"] = pd.concat(p)
    print(f"[export] draw {d} done", flush=True)

df = pd.DataFrame(cols)
df.index.name = "Date"
df.to_csv(OUT, float_format="%.8g", compression="gzip")
print(f"[export] wrote {OUT}  ({df.shape[0]} days x {df.shape[1]} columns); missing result files: {missing}")
if missing:
    print("WARNING: some result files were missing -- strategies with incomplete windows were left out.")
