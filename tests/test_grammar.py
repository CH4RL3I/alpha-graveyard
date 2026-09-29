import numpy as np
import pandas as pd

from stratgen.engine import Evaluator, ann_sharpe
from stratgen.grammar import (
    Leaf,
    Node,
    candidate_from_json,
    candidate_to_json,
    crossover,
    depth,
    mutate,
    random_candidate,
)

from .conftest import make_prices


def _frame(close):
    idx = pd.bdate_range("2015-01-01", periods=len(close))
    df = pd.DataFrame({("X", "open"): close, ("X", "close"): close}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_ma_cross_on_ramp_true_after_warmup_only():
    ev = Evaluator(_frame(np.linspace(100, 200, 120)), 0.0)
    sig = ev.signal("X", Leaf("ma", (5, 50)))
    assert not sig[:49].any() and sig[49:].all()


def test_inverted_leaf_is_false_during_warmup():
    ev = Evaluator(_frame(np.linspace(100, 200, 120)), 0.0)
    sig = ev.signal("X", Leaf("ma", (5, 50), inv=True))
    assert not sig.any()  # ramp: fast > slow after warm-up, so NOT is false; warm-up never fires


def test_and_or_logic():
    ev = Evaluator(_frame(np.linspace(100, 200, 120)), 0.0)
    a, b = Leaf("ma", (5, 50)), Leaf("mom", (100, 0.0))
    sa, sb = ev.signal("X", a), ev.signal("X", b)
    assert np.array_equal(ev.signal("X", Node("and", a, b)), sa & sb)
    assert np.array_equal(ev.signal("X", Node("or", a, b)), sa | sb)
    assert sb.sum() < sa.sum()


def test_rsi_extremes_and_zscore():
    up = Evaluator(_frame(np.linspace(100, 200, 80)), 0.0)
    assert up.signal("X", Leaf("rsi", (14, 90, True)))[20:].all()  # monotone up: RSI 100
    down = Evaluator(_frame(np.linspace(200, 100, 80)), 0.0)
    assert down.signal("X", Leaf("rsi", (14, 10, False)))[20:].all()  # monotone down: RSI 0
    c = np.full(60, 100.0)
    c[-1] = 130.0
    z = Evaluator(_frame(c), 0.0)
    assert z.signal("X", Leaf("z", (20, 2.0, True)))[-1]
    assert not z.signal("X", Leaf("z", (20, 2.0, True)))[:-1].any()  # zero std: undefined, False


def test_vol_ratio():
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0, 0.002, 200), rng.normal(0, 0.03, 12)])
    ev = Evaluator(_frame(100 * np.exp(np.cumsum(r))), 0.0)
    sig = ev.signal("X", Leaf("vol", (10, 60, 1.5, True)))
    assert sig[-3:].all() and not sig[100:200].any()


def test_random_genomes_are_valid_and_json_roundtrips():
    rng = np.random.default_rng(7)
    assets = ["AAA", "BBB"]
    p = make_prices(300)
    ev = Evaluator(p, 5.0)
    cands = [random_candidate(rng, assets) for _ in range(80)]
    cands += [mutate(c, rng, assets) for c in cands[:40]]
    cands += [crossover(cands[i], cands[i + 1], rng) for i in range(40)]
    for c in cands:
        assert depth(c.rule) <= 3
        r = ev.returns(c)
        assert r.shape == (300,) and np.isfinite(r).all()
        assert candidate_from_json(candidate_to_json(c)) == c


def test_ann_sharpe_matches_overfit_library():
    from overfit import annualize_sharpe, sharpe_ratio

    x = np.random.default_rng(2).normal(0.0005, 0.01, 500)
    assert np.isclose(ann_sharpe(x), annualize_sharpe(sharpe_ratio(x), 252))
