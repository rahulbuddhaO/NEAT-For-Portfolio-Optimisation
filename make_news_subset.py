"""Streams the 23 GB FNSPID news file and keeps only the 33 study tickers from 2009 onward.
Run in the background:  nohup python3 -u make_news_subset.py > subset.log 2>&1 &
"""
import sys, time
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/fnspid/nasdaq_exteral_data.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/fnspid/news_subset.csv"
TICKERS = {"GILD", "NVDA", "WFC", "MRK", "KO",
           "CMCSA", "MS", "COP", "PEP", "QCOM", "FDX", "V", "ORCL", "COST", "OXY", "NKE", "EA", "NEE", "EBAY",
           "CAT", "TXN", "SLB", "DUK", "BX", "MMM", "CRM", "AMT", "TJX", "HAL", "BIIB", "ADBE", "NEM", "CI"}
KEEP = ["Date", "Stock_symbol", "Article_title", "Article",
        "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]

first, n_in, n_out, t0 = True, 0, 0, time.time()
reader = pd.read_csv(SRC, usecols=KEEP, chunksize=200_000, dtype=str, on_bad_lines="skip")
for i, ch in enumerate(reader):
    n_in += len(ch)
    ch = ch[ch["Stock_symbol"].isin(TICKERS) & (ch["Date"].fillna("").str[:10] >= "2009-01-01")]
    n_out += len(ch)
    ch.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    if i % 10 == 0:
        print(f"{n_in:,} rows read, {n_out:,} kept, {time.time()-t0:.0f}s", flush=True)
print(f"DONE: {n_in:,} rows read, {n_out:,} kept -> {OUT}  ({time.time()-t0:.0f}s)", flush=True)
