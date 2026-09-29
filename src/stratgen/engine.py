"""Vectorised backtest with strict next-day execution.

Timing convention (all indices are trading-day positions in the price panel):

* A signal at index t is computed from closes at indices <= t only.
* The order is sent after the close of t and filled at the OPEN of t+1.
* The position therefore earns open-to-open returns starting at t+1. Define
  r[i] = open[i] / open[i-1] - 1 (realised at the open of day i). The position that earns r[i]
  was filled at open[i-1], so it came from the signal at t = i-2.
* Cost is charged on r[i] as cost_rate * |pos[i] - pos[i-1]| (the fill happened at open[i-1]).

Every return r[i] therefore uses opens <= i and signals built from closes <= i-2, so a series
computed on a truncated panel equals the same prefix of a series computed on the full panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .grammar import Candidate, Leaf, Node, Rule

LAG = 2  # pos[i] = signal[i - LAG]; see module docstring


class Evaluator:
    def __init__(self, prices: pd.DataFrame, cost_bps: float):
        self.dates = prices.index
        self.cost = cost_bps / 1e4
        self.assets = list(dict.fromkeys(prices.columns.get_level_values(0)))
        self._close = {a: prices[(a, "close")].to_numpy(float) for a in self.assets}
        self._ret = {}
        for a in self.assets:
            o = prices[(a, "open")].to_numpy(float)
            r = np.zeros(len(o))
            r[1:] = o[1:] / o[:-1] - 1.0
            self._ret[a] = r
        self._ind: dict[tuple, np.ndarray] = {}
        self._leaf: dict[tuple, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.dates)

    # ------------------------------------------------------------ indicators
    def _series(self, a: str) -> pd.Series:
        return pd.Series(self._close[a])

    def _indicator(self, a: str, key: tuple) -> np.ndarray:
        k = (a, *key)
        if k in self._ind:
            return self._ind[k]
        s = self._series(a)
        kind = key[0]
        if kind == "sma":
            out = s.rolling(key[1]).mean()
        elif kind == "mom":
            out = s / s.shift(key[1]) - 1.0
        elif kind == "rsi":
            n = key[1]
            d = s.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            rs = up / dn.replace(0.0, np.nan)
            out = 100 - 100 / (1 + rs)
            out = out.where(~((dn == 0) & up.notna()), 100.0)
        elif kind == "z":
            n = key[1]
            sd = s.rolling(n).std().replace(0.0, np.nan)
            out = (s - s.rolling(n).mean()) / sd
        elif kind == "std":
            out = s.pct_change().rolling(key[1]).std()
        else:
            raise ValueError(kind)
        arr = out.to_numpy(float)
        self._ind[k] = arr
        return arr

    def _leaf_signal(self, a: str, leaf: Leaf) -> np.ndarray:
        k = (a, leaf.kind, leaf.params, leaf.inv)
        if k in self._leaf:
            return self._leaf[k]
        p = leaf.params
        if leaf.kind == "ma":
            fast, slow = self._indicator(a, ("sma", p[0])), self._indicator(a, ("sma", p[1]))
            valid = ~np.isnan(fast) & ~np.isnan(slow)
            with np.errstate(invalid="ignore"):
                cond = fast > slow
        elif leaf.kind == "mom":
            x = self._indicator(a, ("mom", p[0]))
            valid = ~np.isnan(x)
            with np.errstate(invalid="ignore"):
                cond = x > p[1]
        elif leaf.kind == "rsi":
            x = self._indicator(a, ("rsi", p[0]))
            valid = ~np.isnan(x)
            with np.errstate(invalid="ignore"):
                cond = x > p[1] if p[2] else x < p[1]
        elif leaf.kind == "z":
            x = self._indicator(a, ("z", p[0]))
            valid = ~np.isnan(x)
            with np.errstate(invalid="ignore"):
                cond = x > p[1] if p[2] else x < -p[1]
        elif leaf.kind == "vol":
            sh, lg = self._indicator(a, ("std", p[0])), self._indicator(a, ("std", p[1]))
            with np.errstate(invalid="ignore", divide="ignore"):
                x = sh / lg
            valid = ~np.isnan(x) & np.isfinite(x)
            with np.errstate(invalid="ignore"):
                cond = x > p[2] if p[3] else x < p[2]
        else:
            raise ValueError(leaf.kind)
        sig = (~cond if leaf.inv else cond) & valid  # warm-up is always False, even when inverted
        self._leaf[k] = sig
        return sig

    # ------------------------------------------------------------ signals and returns
    def signal(self, a: str, rule: Rule) -> np.ndarray:
        """Boolean signal at the close of each day, using closes <= that day."""
        if isinstance(rule, Leaf):
            return self._leaf_signal(a, rule)
        assert isinstance(rule, Node)
        left, right = self.signal(a, rule.left), self.signal(a, rule.right)
        return (left & right) if rule.op == "and" else (left | right)

    def position(self, sig: np.ndarray) -> np.ndarray:
        pos = np.zeros(len(sig))
        pos[LAG:] = sig[:-LAG]
        return pos

    def returns_from_position(self, a: str, pos: np.ndarray) -> np.ndarray:
        turn = np.abs(np.diff(pos, prepend=0.0))
        return pos * self._ret[a] - self.cost * turn

    def returns(self, cand: Candidate) -> np.ndarray:
        pos = self.position(self.signal(cand.asset, cand.rule).astype(float))
        return self.returns_from_position(cand.asset, pos)

    def n_changes(self, cand: Candidate, upto: int | None = None) -> int:
        pos = self.position(self.signal(cand.asset, cand.rule).astype(float))[:upto]
        return int(np.count_nonzero(np.diff(pos, prepend=0.0)))

    def asset_returns(self, a: str) -> np.ndarray:
        return self._ret[a]


# ---------------------------------------------------------------- metrics


def ann_sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd < 1e-14:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS))


def perf(x: np.ndarray) -> dict[str, float]:
    """Annualised Sharpe, geometric return, volatility and max drawdown of a daily return array."""
    eq = np.cumprod(1.0 + x)
    years = len(x) / TRADING_DAYS
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {
        "sharpe": ann_sharpe(x),
        "cagr": float(eq[-1] ** (1 / years) - 1),
        "vol": float(x.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_dd": float(dd.min()),
    }
