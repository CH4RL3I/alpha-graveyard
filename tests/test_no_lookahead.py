import numpy as np
import pandas as pd

from stratgen.engine import Evaluator
from stratgen.grammar import Candidate, Leaf, Node, random_candidate

from .conftest import make_prices

RULES = [
    Leaf("ma", (10, 50)),
    Leaf("mom", (5, 0.0)),
    Leaf("rsi", (3, 30, False)),
    Leaf("z", (20, 1.0, False), inv=True),
    Leaf("vol", (10, 60, 1.0, True)),
    Node("and", Leaf("ma", (5, 50)), Leaf("rsi", (7, 60, True), inv=True)),
]


def _returns(prices, cand):
    return Evaluator(prices, 5.0).returns(cand)


def test_prefix_invariance_all_rule_types():
    """Returns on a truncated panel equal the same prefix of returns on the full panel."""
    p = make_prices(500)
    for k in (150, 300, 420):
        for rule in RULES:
            c = Candidate("AAA", rule)
            np.testing.assert_allclose(_returns(p.iloc[:k], c), _returns(p, c)[:k], atol=1e-12)


def test_prefix_invariance_random_candidates():
    p = make_prices(500, seed=3)
    rng = np.random.default_rng(1)
    for _ in range(60):
        c = random_candidate(rng, ["AAA", "BBB", "SPY", "IEF"])
        np.testing.assert_allclose(_returns(p.iloc[:333], c), _returns(p, c)[:333], atol=1e-12)


def test_future_data_cannot_change_past_returns():
    p = make_prices(400)
    q = p.copy()
    q.iloc[250:] = q.iloc[250:] * 3.0  # rewrite the future
    for rule in RULES:
        c = Candidate("BBB", rule)
        np.testing.assert_allclose(_returns(p, c)[:250], _returns(q, c)[:250], atol=1e-12)


def test_signal_on_t_is_executed_at_t_plus_1_open():
    """A jump into the close of day k makes a momentum signal fire at close k. The order fills at
    open k+1 and the position first earns the open[k+1] -> open[k+2] return (r[k+2])."""
    n, k = 60, 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.full(n, 100.0)
    close[k:] = 110.0  # jump between close k-1 and close k
    open_ = close.copy()
    open_[k] = 100.0  # day k opens before the jump: it happens intraday
    df = pd.DataFrame({("X", "open"): open_, ("X", "close"): close}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # noise-free growth after the jump so a filled position has a visible P&L
    ev = Evaluator(df, 0.0)
    c = Candidate("X", Leaf("mom", (1, 0.01)))
    sig = ev.signal("X", c.rule)
    assert sig[k] and not sig[k - 1] and not sig[k + 1]  # signal known at close k only
    pos = ev.position(sig.astype(float))
    assert pos[k + 1] == 0 and pos[k + 2] == 1  # held over r[k+2], not over r[k+1]
    r = ev.returns(c)
    assert np.all(r[: k + 2] == 0)  # the jump return r[k]=open[k]/open[k-1]-1 = 0, r[k+1] = +10%
    assert ev.asset_returns("X")[k + 1] > 0.09  # the move the strategy must NOT capture
