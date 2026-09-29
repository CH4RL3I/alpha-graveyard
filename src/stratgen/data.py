"""Free daily OHLCV from Yahoo Finance's public chart endpoint (no key, no yfinance).

Stooq was tried first and now sits behind a JavaScript challenge, so plain CSV downloads fail.
Yahoo's v8 chart endpoint returns JSON with split-adjusted OHLC and a dividend-adjusted close.
We rescale open by adjclose/close so open and close are both total-return adjusted.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import UNIVERSE

URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
    "?period1=946684800&period2={p2}&interval=1d&events=div,split"
)


def root() -> Path:
    return Path.cwd()


def fetch_one(ticker: str) -> pd.DataFrame:
    url = URL.format(t=ticker, p2=int(time.time()))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "adjclose": res["indicators"]["adjclose"][0]["adjclose"],
            "volume": q["volume"],
        },
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None),
    )
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df[df.index < pd.Timestamp.now().normalize()]  # drop a possibly incomplete bar today


def adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Adjusted open and close (both total-return adjusted)."""
    f = raw["adjclose"] / raw["close"]
    return pd.DataFrame({"open": raw["open"] * f, "close": raw["adjclose"]})


def build_panel(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join adjusted open/close on the dates all tickers share. Columns: (ticker, field)."""
    parts = {t: adjust(r) for t, r in raws.items()}
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    return panel[[(t, f) for t in raws for f in ("open", "close")]]


def fetch(data_dir: Path | None = None, sample_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or root() / "data"
    sample_dir = sample_dir or root() / "sample"
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    raws = {}
    for t in UNIVERSE:
        df = fetch_one(t)
        df.to_csv(data_dir / "raw" / f"{t}.csv")
        raws[t] = df
        print(f"{t}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        time.sleep(0.5)
    panel = build_panel(raws)
    write_panel(panel, data_dir / "prices.csv.gz")
    sample_dir.mkdir(exist_ok=True)
    write_panel(panel, sample_dir / "prices.csv.gz")
    print(
        f"panel: {len(panel)} common days, {panel.index[0].date()} to {panel.index[-1].date()} "
        f"(fetched {datetime.now(UTC):%Y-%m-%d})"
    )
    return panel


def write_panel(panel: pd.DataFrame, path: Path) -> None:
    flat = panel.copy()
    flat.columns = [f"{t}.{f}" for t, f in panel.columns]
    flat.round(6).to_csv(path)


def read_panel(path: Path) -> pd.DataFrame:
    flat = pd.read_csv(path, index_col=0, parse_dates=True)
    flat.columns = pd.MultiIndex.from_tuples([tuple(c.split(".")) for c in flat.columns])
    return flat


def load_prices() -> pd.DataFrame:
    """Prefer the local cache in data/, fall back to the committed sample."""
    for p in (root() / "data" / "prices.csv.gz", root() / "sample" / "prices.csv.gz"):
        if p.exists():
            return read_panel(p)
    raise FileNotFoundError("no price data: run `stratgen fetch` or restore sample/prices.csv.gz")
