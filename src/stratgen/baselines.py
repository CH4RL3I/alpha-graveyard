"""Reference strategies under the same timing and cost model as the candidates.

* buy_hold: 100% SPY. Cost is not charged (a single entry).
* sixty_forty: constant 60/40 SPY/IEF mix, rebalanced daily, no rebalancing cost charged.
* tsmom_spy: 12-month time-series momentum on SPY, long when the 252-day return is positive,
  flat otherwise, with the same next-open execution and costs as the candidates.
* tsmom_multi: the same rule applied to all twelve ETFs, equal weight 1/12 each.

Flat positions earn zero (no cash yield), for baselines and candidates alike.
"""

from __future__ import annotations

import numpy as np

from .engine import Evaluator
from .grammar import Candidate, Leaf

TSMOM = Leaf("mom", (252, 0.0), False)


def baselines(ev: Evaluator) -> dict[str, np.ndarray]:
    out = {
        "Buy and hold SPY": ev.asset_returns("SPY").copy(),
        "60/40 SPY/IEF": 0.6 * ev.asset_returns("SPY") + 0.4 * ev.asset_returns("IEF"),
        "12m TSMOM (SPY)": ev.returns(Candidate("SPY", TSMOM)),
    }
    multi = [ev.returns(Candidate(a, TSMOM)) for a in ev.assets]
    out["12m TSMOM (12 ETFs)"] = np.mean(multi, axis=0)
    out["Buy and hold SPY"][0] = 0.0
    return out
