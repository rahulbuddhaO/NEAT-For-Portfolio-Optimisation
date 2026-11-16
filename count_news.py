import sys
import pandas as pd
from collections import Counter

CSV = r"C:\Users\rahul\Downloads\nasdaq_exteral_data.csv"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
SIX = ["GILD", "NVDA", "QQQ", "WFC", "MRK", "KO"]

art = Counter()
pairs = None
n_chunks = 0
for chunk in pd.read_csv(CSV, usecols=["Date", "Stock_symbol"], chunksize=CHUNK,
                         dtype=str, on_bad_lines="skip"):
    chunk = chunk.dropna()
    art.update(chunk["Stock_symbol"].value_counts().to_dict())
    chunk["day"] = chunk["Date"].str[:10]
    p = chunk[["Stock_symbol", "day"]].drop_duplicates()
    pairs = p if pairs is None else pd.concat([pairs, p]).drop_duplicates()
    n_chunks += 1
    if n_chunks % 5 == 0:
        print(f"  ...{n_chunks * CHUNK:,} rows read, {len(art):,} tickers so far", flush=True)

days = pairs.groupby("Stock_symbol").size()
out = pd.DataFrame({"articles": pd.Series(art), "days_with_news": days}).fillna(0).astype(int)
out = out.sort_values("articles", ascending=False)

print(f"Tickers in file: {len(out)}   Total articles: {int(out.articles.sum()):,}\n")
print("TOP 50 BY ARTICLE COUNT")
print(out.head(50).to_string())
out.head(50).to_csv("top50_news_coverage.csv")
print("\nYOUR SIX TICKERS")
print(out.reindex(SIX).fillna(0).astype(int).to_string())
