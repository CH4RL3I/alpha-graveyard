"""Random search followed by genetic-algorithm refinement, on discovery data only.

Fitness is the discovery-period annualised Sharpe of the net-of-cost return series. Nothing in
this module sees validation data: the caller passes a price panel already cut at the split.
`trials` is the number of UNIQUE candidates evaluated (duplicates hit a cache and are not
counted). That number is what the Deflated Sharpe Ratio must be told.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COST_BPS, MIN_CHANGES
from .engine import Evaluator, ann_sharpe
from .grammar import Candidate, crossover, mutate, random_candidate

BAD = -10.0  # fitness of an untradeable candidate


@dataclass
class SearchResult:
    candidates: list[Candidate]  # unique, in evaluation order
    fitness: np.ndarray  # discovery Sharpe (BAD if untradeable)
    stage: list[str]  # "random" or "ga" per candidate
    seed: int
    trials: int


def _fitness(ev: Evaluator, cand: Candidate) -> float:
    if ev.n_changes(cand) < MIN_CHANGES:
        return BAD
    s = ann_sharpe(ev.returns(cand))
    return BAD if not np.isfinite(s) else s


def run_search(
    prices: pd.DataFrame,
    trials: int,
    seed: int,
    cost_bps: float = COST_BPS,
    random_frac: float = 0.6,
    pop_size: int = 300,
) -> SearchResult:
    ev = Evaluator(prices, cost_bps)
    rng = np.random.default_rng(seed)
    assets = ev.assets
    seen: dict[Candidate, float] = {}
    order: list[Candidate] = []
    stage: list[str] = []

    def evaluate(c: Candidate, tag: str) -> float:
        if c not in seen:
            seen[c] = _fitness(ev, c)
            order.append(c)
            stage.append(tag)
        return seen[c]

    n_random = min(trials, int(trials * random_frac))
    attempts = 0
    while len(order) < n_random and attempts < 50 * trials:
        evaluate(random_candidate(rng, assets), "random")
        attempts += 1

    def top(k: int) -> list[Candidate]:
        ranked = sorted(order, key=lambda c: -seen[c])  # stable: ties keep evaluation order
        return ranked[:k]

    pop = top(pop_size)
    stall = 0
    while len(order) < trials and stall < 200 and pop:
        before = len(order)
        children: list[Candidate] = []
        for _ in range(pop_size):
            a = _tournament(pop, seen, rng)
            if rng.random() < 0.5:
                child = crossover(a, _tournament(pop, seen, rng), rng)
            else:
                child = a
            for _ in range(1 + int(rng.random() < 0.5)):
                child = mutate(child, rng, assets)
            children.append(child)
        for c in children:
            if len(order) >= trials:
                break
            evaluate(c, "ga")
        pop = top(pop_size)
        stall = stall + 1 if len(order) == before else 0

    fit = np.array([seen[c] for c in order])
    return SearchResult(order, fit, stage, seed, len(order))


def _tournament(pop: list[Candidate], seen: dict, rng: np.random.Generator, k: int = 3):
    idx = rng.integers(len(pop), size=k)
    return max((pop[i] for i in idx), key=lambda c: seen[c])
