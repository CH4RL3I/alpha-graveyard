from pathlib import Path

import numpy as np
import pandas as pd

from stratgen.baselines import baselines
from stratgen.config import SPLIT_DATE
from stratgen.data import read_panel
from stratgen.engine import Evaluator
from stratgen.report import _distinct_top
from stratgen.search import run_search

from .conftest import make_prices


def _run(p, seed=0, trials=400):
    return run_search(p, trials=trials, seed=seed, pop_size=40)


def test_search_is_deterministic_for_fixed_seed():
    p = make_prices(700)
    a, b = _run(p), _run(p)
    assert a.candidates == b.candidates
    assert np.array_equal(a.fitness, b.fitness)
    assert a.trials == 400 == len(set(a.candidates))


def test_different_seeds_differ():
    p = make_prices(700)
    assert _run(p, 0).candidates != _run(p, 1).candidates


def test_ga_stage_runs_and_does_not_hurt_best():
    p = make_prices(700)
    r = _run(p)
    assert "ga" in r.stage and "random" in r.stage
    n_rand = r.stage.count("random")
    assert r.fitness.max() >= r.fitness[:n_rand].max()


def test_split_dates_are_disjoint_and_ordered():
    prices = read_panel(Path(__file__).parents[1] / "sample" / "prices.csv.gz")
    cut = pd.Timestamp(SPLIT_DATE)
    is_idx, oos_idx = prices.index[prices.index < cut], prices.index[prices.index >= cut]
    assert len(is_idx) > 1000 and len(oos_idx) > 1000
    assert is_idx.max() < cut <= oos_idx.min()
    assert len(is_idx) + len(oos_idx) == len(prices)


def test_search_result_is_independent_of_validation_data():
    """The search only sees the discovery slice; scrambling everything after it changes nothing."""
    p = make_prices(800)
    cut = p.index[500]
    a = _run(p[p.index < cut])
    q = p.copy()
    q[q.index >= cut] = np.random.default_rng(9).uniform(1, 500, q[q.index >= cut].shape)
    b = _run(q[q.index < cut])
    assert a.candidates == b.candidates and np.array_equal(a.fitness, b.fitness)


def test_baselines_use_same_pipeline():
    p = make_prices(600)
    bl = baselines(Evaluator(p, 5.0))
    assert len(bl) == 4 and "12m TSMOM (SPY)" in bl
    assert all(v.shape == (600,) for v in bl.values())
    assert np.all(bl["12m TSMOM (SPY)"][:254] == 0)  # 252-day lookback + 2-day execution lag


def test_distinct_top_drops_near_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 1))
    m = np.hstack([base, base + rng.normal(scale=0.01, size=(300, 1)), rng.normal(size=(300, 2))])
    sr = np.array([3.0, 2.9, 1.0, 0.5])
    assert _distinct_top(m, sr, np.ones(4, bool)) == [0, 2, 3]
