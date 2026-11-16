"""
mcs_spa.py -- multiple-testing-robust tests on daily out-of-sample returns (Sharpe ratio as the criterion).

1. Model Confidence Set (Hansen, Lunde & Nason, 2011), T_R (range) elimination rule, loss = -Sharpe ratio.
   Returns the set of strategies that cannot be statistically separated from the best one (alpha = 10% and 25%).
2. Hansen (2005) SPA-type test (studentised, max statistic): "does ANY strategy have a higher Sharpe than the
   benchmark?", run separately against Equal-Weight and Markowitz; plus Romano-Wolf StepM to name which ones.
Both use the stationary bootstrap (Politis & Romano, 1994), mean block length 10 days, 2,000 resamples, the same
resampled days for every strategy (keeps the cross-strategy dependence).

Main analysis: one daily series per strategy = average of the 5 draws (each draw already seed-averaged).
Robustness: the MCS is repeated inside each draw separately and inclusion counts are reported.

Usage:  python mcs_spa.py daily_returns_by_draw.csv.gz  [out_dir]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B, BLOCK, SEED = 2000, 10, 12345


def stationary_bootstrap_idx(T, B, block, rng):
    p = 1.0 / block
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    newb = rng.random((B, T)) < p
    jump = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(newb[:, t], jump[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def sharpe(X):
    return np.sqrt(252) * X.mean(axis=0) / X.std(axis=0, ddof=1)


def boot_sharpes(X, idx):
    out = np.empty((idx.shape[0], X.shape[1]))
    for b in range(idx.shape[0]):
        out[b] = sharpe(X[idx[b]])
    return out


def mcs(names, sr, sr_b, alpha):
    """Hansen-Lunde-Nason MCS with the T_R (range) rule, loss = -Sharpe: at each step test whether any pair of the
    remaining strategies differs; if so, eliminate the strategy with the largest standardised deficit to any other.
    Returns (included names, elimination list with running MCS p-values)."""
    keep = list(range(len(names))); elim = []; pmax = 0.0
    while len(keep) > 1:
        s, sb = sr[keep], sr_b[:, keep]
        D = s[None, :] - s[:, None]                       # D[i, j] = SR_j - SR_i  (= loss_i - loss_j)
        Db = sb[:, None, :] - sb[:, :, None]
        se = (Db - D).std(axis=0, ddof=1)
        np.fill_diagonal(se, 1.0); se[se < 1e-12] = 1e-12
        t = D / se
        T = np.abs(t).max()
        Tb = (np.abs(Db - D) / se).max(axis=(1, 2))
        p = float((Tb >= T).mean())
        pmax = max(pmax, p)                                # MCS p-values are the running maximum
        if pmax >= alpha:
            break
        worst = keep[int(np.argmax(t.max(axis=1)))]
        elim.append((names[worst], round(pmax, 4)))
        keep.remove(worst)
    return [names[i] for i in keep], elim


def spa_stepm(names, sr, sr_b, bench_i, n_days, alpha=0.05):
    """Studentised max test vs one benchmark (Hansen SPA, consistent re-centring) + Romano-Wolf StepM."""
    others = [i for i in range(len(names)) if i != bench_i]
    d = sr[others] - sr[bench_i]
    db = sr_b[:, others] - sr_b[:, [bench_i]]
    se = (db - d).std(axis=0, ddof=1); se[se < 1e-12] = 1e-12
    t = d / se
    # Hansen (2005) consistent re-centring: strategies that are not clearly worse than the benchmark are centred at
    # zero (the null boundary); clearly worse ones (t below -sqrt(2 log log n)) keep their negative mean.
    thr = -np.sqrt(2 * np.log(np.log(max(n_days, 16))))
    mu = np.where(t >= thr, 0.0, d)
    zb = ((db - d + mu) / se)
    p_spa = float((np.maximum(zb.max(axis=1), 0) >= max(t.max(), 0)).mean())
    # StepM: which strategies are significantly better than the benchmark (familywise error controlled)
    active = list(range(len(others))); better = []
    while active:
        cz = ((db[:, active] - d[active]) / se[active]).max(axis=1)
        crit = np.quantile(cz, 1 - alpha)
        rej = [j for j in active if t[j] > crit]
        if not rej:
            break
        better += [names[others[j]] for j in rej]
        active = [j for j in active if j not in rej]
    best = int(np.argmax(t))
    return p_spa, better, names[others[best]], float(d[best]), float(t[best])


def main():
    src = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
    raw = pd.read_csv(src, index_col=0, parse_dates=True)
    strategies = sorted({c.split("|")[0] for c in raw.columns})
    draws = sorted({c.split("|")[1] for c in raw.columns})
    complete = [s for s in strategies if all(f"{s}|{d}" in raw.columns for d in draws)]
    ens = pd.DataFrame({s: raw[[f"{s}|{d}" for d in draws]].mean(axis=1) for s in complete}).dropna()
    names = list(ens.columns); X = ens.values; T = len(X)
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_idx(T, B, BLOCK, rng)
    sr, sr_b = sharpe(X), boot_sharpes(X, idx)
    print(f"[mcs] {len(names)} strategies, {T} days, {len(draws)} draws averaged, {B} stationary-bootstrap resamples")

    rows = []
    for a in (0.10, 0.25):
        inc, elim = mcs(names, sr, sr_b, a)
        rows.append({"alpha": a, "in_MCS": ", ".join(sorted(inc, key=lambda n: -sr[names.index(n)])), "size": len(inc)})
        print(f"\nMCS at {int(a*100)}%: {len(inc)} strategies cannot be separated from the best:")
        for n in sorted(inc, key=lambda n: -sr[names.index(n)]):
            print(f"   {n:<24} Sharpe {sr[names.index(n)]:.3f}")
    pd.DataFrame(rows).to_csv(out / "mcs_result.csv", index=False)

    spa_rows = []
    for bench in ("EqualWeight", "Markowitz"):
        if bench not in names:
            continue
        p, better, best, gap, tbest = spa_stepm(names, sr, sr_b, names.index(bench), T)
        spa_rows.append({"benchmark": bench, "benchmark_sharpe": sr[names.index(bench)], "SPA_p_value": p,
                         "best_strategy": best, "best_gap": gap, "significantly_better_StepM": ", ".join(better) or "none"})
        print(f"\nSPA vs {bench} (Sharpe {sr[names.index(bench)]:.3f}): p = {p:.3f}; best = {best} (gap {gap:+.3f}); "
              f"significantly better (StepM, 5%): {', '.join(better) or 'none'}")
    pd.DataFrame(spa_rows).to_csv(out / "spa_result.csv", index=False)

    # robustness: MCS inside each draw
    inc_count = {n: 0 for n in names}
    for d in draws:
        Xd = pd.DataFrame({s: raw[f"{s}|{d}"] for s in names}).dropna().values
        idxd = stationary_bootstrap_idx(len(Xd), B, BLOCK, np.random.default_rng(SEED))
        incd, _ = mcs(names, sharpe(Xd), boot_sharpes(Xd, idxd), 0.10)
        for n in incd:
            inc_count[n] += 1
    tab = pd.DataFrame({"Strategy": names, "Sharpe_ensemble": sr,
                        "in_MCS10_per_draw": [inc_count[n] for n in names]}).sort_values("Sharpe_ensemble", ascending=False)
    tab.to_csv(out / "mcs_by_draw.csv", index=False)
    print("\nTop 15 by ensemble Sharpe, with number of draws (of %d) in which each is in the 10%% MCS:" % len(draws))
    print(tab.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
