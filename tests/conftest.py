import numpy as np
import pandas as pd
import pytest


def make_prices(n=600, seed=0, assets=("AAA", "BBB", "SPY", "IEF")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    cols = {}
    for a in assets:
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        o = c * np.exp(rng.normal(0, 0.002, n))
        cols[(a, "open")] = o
        cols[(a, "close")] = c
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def prices():
    return make_prices()
