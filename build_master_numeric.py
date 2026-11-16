"""Builds the numeric part of the 33-stock master dataset: prices, returns, and pairwise
Corr / SMA / EMA columns, using the same definitions as the existing 6-ticker master.

Usage (on the server):  python3 build_master_numeric.py
Optional arguments:     python3 build_master_numeric.py PRICES_DIR EXISTING_MASTER OUT_CSV
"""
import itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------ CONFIG ------------------------------
PRICES_DIR      = "/home/ubuntu/fnspid/full_history"
EXISTING_MASTER = "/home/ubuntu/master_dataset.xlsx"      # used only to validate; may be missing
OUT_CSV         = "/home/ubuntu/master_dataset_33_numeric.csv"
START, END = "2009-01-01", "2023-12-31"
ORIGINAL = ["GILD", "NVDA", "WFC", "MRK", "KO"]           # QQQ removed
NEW = ["CMCSA", "MS", "COP", "PEP", "QCOM", "FDX", "V", "ORCL", "COST", "OXY", "NKE", "EA", "NEE", "EBAY",
       "CAT", "TXN", "SLB", "DUK", "BX", "MMM", "CRM", "AMT", "TJX", "HAL", "BIIB", "ADBE", "NEM", "CI"]
TICKERS = ORIGINAL + NEW
WIN_CORR, WIN_MA = 30, 20
if len(sys.argv) > 1: PRICES_DIR = sys.argv[1]
if len(sys.argv) > 2: EXISTING_MASTER = sys.argv[2]
if len(sys.argv) > 3: OUT_CSV = sys.argv[3]


# ------------------------------ PRICES ------------------------------
def load_prices(prices_dir, tickers):
    files = {p.stem.lower(): p for p in Path(prices_dir).glob("*.csv")}     # case-insensitive (nvda.csv)
    if not files:
        sys.exit(f"No CSV files found in {prices_dir}")
    series, dropped = {}, {}
    for t in tickers:
        p = files.get(t.lower())
        if p is None:
            dropped[t] = "file not found"; continue
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        col = "adj close" if "adj close" in df.columns else ("close" if "close" in df.columns else None)
        if "date" not in df.columns or col is None:
            dropped[t] = f"unexpected columns {list(df.columns)}"; continue
        d = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce")
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=pd.DatetimeIndex(d)).dropna()
        s = s[s.index.notna()]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            dropped[t] = "no usable rows"; continue
        series[t] = s
        if col != "adj close":
            print(f"  WARNING {t}: no 'adj close' column, using '{col}' (not dividend-adjusted)")
    if dropped:
        print("DROPPED:", dropped)
    P = pd.DataFrame(series)
    P = P.sort_index().loc[START:END].dropna(how="all").ffill(limit=3)
    P = P[[t for t in tickers if t in P.columns]]
    return P


# ------------------------------ FEATURES ------------------------------
def numeric_block(P):
    T = list(P.columns)
    R = P.pct_change()
    out = {}
    for t in T: out[f"{t}_Price"] = P[t]
    for t in T: out[f"{t}_Return"] = R[t]
    sma = {t: P[t].rolling(WIN_MA).mean() for t in T}
    ema = {t: P[t].ewm(span=WIN_MA, adjust=False).mean() for t in T}
    sg_sma = {t: np.sign(P[t] - sma[t]) for t in T}        # +1 above, -1 below, 0 tie, NaN until defined
    sg_ema = {t: np.sign(P[t] - ema[t]) for t in T}
    for a, b in itertools.combinations(T, 2):
        out[f"Corr_{a}_vs_{b}"] = R[a].rolling(WIN_CORR).corr(R[b])
        out[f"SMA_{a}_vs_{b}"] = sg_sma[a] * sg_sma[b]      # +1 same side of own average, -1 opposite, 0 tie
        out[f"EMA_{a}_vs_{b}"] = sg_ema[a] * sg_ema[b]
    return pd.DataFrame(out, index=P.index)


# ------------------------------ VALIDATION ------------------------------
def validate(M):
    p = Path(EXISTING_MASTER)
    if not p.exists():
        print(f"\nValidation skipped: {EXISTING_MASTER} not found"); return
    try:
        old = pd.read_excel(p, index_col=0, header=1)
        if "GILD_Return" not in old.columns:
            old = pd.read_excel(p, index_col=0, header=0)
        old.index = pd.to_datetime(old.index, errors="coerce")
        old = old[old.index.notna()]
    except Exception as e:
        print(f"\nValidation skipped: could not read existing master ({e})"); return
    common = M.index.intersection(old.index)
    print(f"\nVALIDATION vs existing master ({len(common)} shared dates, {len(old)} rows in old, {len(M)} rows in new)")
    cols = [f"{t}_Price" for t in ORIGINAL] + [f"{t}_Return" for t in ORIGINAL]
    cols += [f"{k}_{a}_vs_{b}" for a, b in itertools.combinations(ORIGINAL, 2) for k in ("Corr", "SMA", "EMA")]
    bad = []
    for c in cols:
        if c not in old.columns:
            bad.append((c, "missing in old")); continue
        a, b = M[c].loc[common], old[c].astype(float).loc[common]
        d = (a - b).abs().max()
        nan_ok = bool((a.isna() == b.isna()).all())
        if not nan_ok or (pd.notna(d) and d > 1e-6):
            bad.append((c, f"maxdiff={d:.3g}, nan-pattern-ok={nan_ok}"))
    if not bad:
        print(f"  PASS: all {len(cols)} original-ticker columns match the existing master")
    else:
        print(f"  CHECK: {len(bad)} of {len(cols)} columns differ. First few:")
        for x in bad[:8]: print("   ", x)


# ------------------------------ MAIN ------------------------------
if __name__ == "__main__":
    print(f"Reading prices from {PRICES_DIR}")
    P = load_prices(PRICES_DIR, TICKERS)
    print(f"\nprices: {P.shape[0]} days x {P.shape[1]} tickers, {P.index[0].date()} -> {P.index[-1].date()}")
    late = {t: str(P[t].first_valid_index().date()) for t in P.columns if P[t].first_valid_index() > P.index[5]}
    if late: print("Tickers that start after the first trading days:", late)
    print("Missing values per ticker (after forward-fill of up to 3 days):",
          {t: int(P[t].isna().sum()) for t in P.columns if P[t].isna().sum() > 0} or "none")
    M = numeric_block(P)
    M.index.name = "Date"
    validate(M)
    M.to_csv(OUT_CSV)
    n = len(P.columns)
    print(f"\nWrote {OUT_CSV}: {M.shape[0]} rows x {M.shape[1]} columns "
          f"({n} prices, {n} returns, {n*(n-1)//2} each of Corr/SMA/EMA)")
    print(f"Model inputs for this universe: PRICE_ONLY={n}, CORR_ONLY={n + n*(n-1)//2}, "
          f"FULL (before sentiment)={n + n*(n-1)//2 + 2*(n*(n-1)//2)}")
