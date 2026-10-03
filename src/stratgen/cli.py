"""stratgen fetch | search | report"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import COST_BPS, SPLIT_DATE, UNIVERSE


def _cmd_fetch(_: argparse.Namespace) -> None:
    from .data import fetch

    fetch()


def _cmd_search(a: argparse.Namespace) -> None:
    from .data import load_prices
    from .report import save_search
    from .search import run_search

    prices = load_prices()
    discovery = prices[prices.index < pd.Timestamp(SPLIT_DATE)]  # validation is never passed on
    assert discovery.index.max() < pd.Timestamp(SPLIT_DATE)
    t0 = time.time()
    res = run_search(discovery, trials=a.trials, seed=a.seed, cost_bps=a.cost_bps)
    meta = {
        "cost_bps": a.cost_bps,
        "split_date": SPLIT_DATE,
        "universe": UNIVERSE,
        "discovery_end": str(discovery.index.max().date()),
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(a.out)
    save_search(res, out, meta)
    best = res.fitness.max()
    print(f"{res.trials} unique candidates in {meta['seconds']}s, best discovery Sharpe {best:.2f}")
    print(f"saved {out}")


def _cmd_report(a: argparse.Namespace) -> None:
    from .data import load_prices
    from .report import analyse, load_search, write_outputs

    meta, cands, _ = load_search(Path(a.search))
    result = analyse(load_prices(), cands, meta)
    write_outputs(result, Path(a.results), Path(a.figures))
    s = result["summary"]
    print(
        f"trials {s['trials_unique_evaluated']}, PBO {s['pbo']:.3f}, "
        f"survivors {s['n_survivors_top10']}/10"
    )
    print(f"wrote {a.results} and {a.figures}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stratgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download daily OHLCV into data/").set_defaults(fn=_cmd_fetch)
    s = sub.add_parser("search", help="random search + GA on the discovery period")
    s.add_argument("--trials", type=int, default=15000, help="unique candidates to evaluate")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cost-bps", type=float, default=COST_BPS)
    s.add_argument("--out", default="results/search.json.gz")
    s.set_defaults(fn=_cmd_search)
    r = sub.add_parser("report", help="held-out evaluation, PBO/DSR, figures")
    r.add_argument("--search", default="results/search.json.gz")
    r.add_argument("--results", default="results")
    r.add_argument("--figures", default="docs/figures")
    r.set_defaults(fn=_cmd_report)
    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
