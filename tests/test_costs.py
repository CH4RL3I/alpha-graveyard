import numpy as np

from stratgen.engine import Evaluator
from stratgen.grammar import Candidate, Leaf

from .conftest import make_prices


def test_always_long_pays_one_entry_cost():
    p = make_prices(200)
    ev = Evaluator(p, 5.0)
    pos = np.ones(len(p))
    pos[:2] = 0.0
    r = ev.returns_from_position("AAA", pos)
    gross = pos * ev.asset_returns("AAA")
    diff = gross - r
    assert np.count_nonzero(diff) == 1
    assert np.isclose(diff.sum(), 5.0 / 1e4)
    assert np.isclose(diff[2], 5.0 / 1e4)  # charged on the first return earned after the fill


def test_round_trip_costs_two_units():
    p = make_prices(50)
    ev = Evaluator(p, 10.0)
    pos = np.zeros(len(p))
    pos[10:20] = 1.0
    net = ev.returns_from_position("BBB", pos)
    gross = pos * ev.asset_returns("BBB")
    assert np.isclose((gross - net).sum(), 2 * 10.0 / 1e4)


def test_cost_scales_with_turnover_and_zero_cost_matches_gross():
    p = make_prices(400, seed=5)
    c = Candidate("SPY", Leaf("rsi", (2, 50, False)))  # trades a lot
    hi, lo, zero = (Evaluator(p, b).returns(c) for b in (20.0, 5.0, 0.0))
    ev0 = Evaluator(p, 0.0)
    pos = ev0.position(ev0.signal("SPY", c.rule).astype(float))
    assert np.allclose(zero, pos * ev0.asset_returns("SPY"))
    turnover = np.abs(np.diff(pos, prepend=0.0)).sum()
    assert turnover > 20
    assert np.isclose((zero - lo).sum(), 5.0 / 1e4 * turnover)
    assert np.isclose((zero - hi).sum(), 20.0 / 1e4 * turnover)
