"""A small typed grammar of long/flat trading rules.

A rule is a boolean expression over indicators of one asset. It is either a Leaf (one indicator
condition) or a Node combining two rules with AND / OR. A Candidate binds a rule to one asset:
long when the rule is true at the close, flat otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# kind -> tuple of parameter choice lists, in order.
GRID: dict[str, tuple[tuple, ...]] = {
    # SMA(fast) > SMA(slow)
    "ma": ((5, 10, 20, 30, 50), (50, 100, 150, 200, 250)),
    # return over `lookback` days > threshold
    "mom": ((5, 10, 21, 42, 63, 126, 189, 252), (-0.05, -0.02, 0.0, 0.02, 0.05)),
    # Wilder RSI(n) above/below level
    "rsi": ((2, 3, 5, 7, 14, 21), (10, 20, 30, 40, 60, 70, 80, 90), (False, True)),
    # z-score of close vs SMA(n): above +thr, or below -thr
    "z": ((10, 20, 40, 60, 120), (0.5, 1.0, 1.5, 2.0), (False, True)),
    # realised vol ratio short/long above/below ratio
    "vol": ((10, 20, 40), (60, 120, 250), (0.6, 0.8, 1.0, 1.2, 1.5), (False, True)),
}
KINDS = tuple(GRID)
MAX_DEPTH = 3  # Node nesting depth; at most 8 leaves is never reached in practice
P_INVERT = 0.2


@dataclass(frozen=True)
class Leaf:
    kind: str
    params: tuple
    inv: bool = False


@dataclass(frozen=True)
class Node:
    op: str  # "and" | "or"
    left: Rule
    right: Rule


Rule = Leaf | Node


@dataclass(frozen=True)
class Candidate:
    asset: str
    rule: Rule


# ---------------------------------------------------------------- generation


def random_leaf(rng: np.random.Generator) -> Leaf:
    kind = KINDS[rng.integers(len(KINDS))]
    while True:
        params = tuple(choices[rng.integers(len(choices))] for choices in GRID[kind])
        if kind != "ma" or params[0] < params[1]:
            break
    return Leaf(kind, _native(params), bool(rng.random() < P_INVERT))


def random_rule(rng: np.random.Generator, depth: int = 0, max_depth: int = 2) -> Rule:
    p_node = (0.45, 0.30, 0.2)[min(depth, 2)]
    if depth < max_depth and rng.random() < p_node:
        op = "and" if rng.random() < 0.6 else "or"
        return Node(
            op, random_rule(rng, depth + 1, max_depth), random_rule(rng, depth + 1, max_depth)
        )
    return random_leaf(rng)


def random_candidate(rng: np.random.Generator, assets: list[str]) -> Candidate:
    return Candidate(assets[rng.integers(len(assets))], random_rule(rng))


def _native(params: tuple) -> tuple:
    return tuple(p.item() if hasattr(p, "item") else p for p in params)


# ---------------------------------------------------------------- tree utilities


def depth(rule: Rule) -> int:
    return 0 if isinstance(rule, Leaf) else 1 + max(depth(rule.left), depth(rule.right))


def paths(rule: Rule, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    out = [prefix]
    if isinstance(rule, Node):
        out += paths(rule.left, prefix + (0,)) + paths(rule.right, prefix + (1,))
    return out


def get(rule: Rule, path: tuple[int, ...]) -> Rule:
    for step in path:
        assert isinstance(rule, Node)
        rule = rule.right if step else rule.left
    return rule


def replace(rule: Rule, path: tuple[int, ...], new: Rule) -> Rule:
    if not path:
        return new
    assert isinstance(rule, Node)
    if path[0] == 0:
        return Node(rule.op, replace(rule.left, path[1:], new), rule.right)
    return Node(rule.op, rule.left, replace(rule.right, path[1:], new))


def n_leaves(rule: Rule) -> int:
    return 1 if isinstance(rule, Leaf) else n_leaves(rule.left) + n_leaves(rule.right)


# ---------------------------------------------------------------- genetic operators


def _mutate_leaf(leaf: Leaf, rng: np.random.Generator) -> Leaf:
    r = rng.random()
    if r < 0.15:
        return Leaf(leaf.kind, leaf.params, not leaf.inv)
    if r < 0.30:
        return random_leaf(rng)
    idx = int(rng.integers(len(leaf.params)))
    choices = GRID[leaf.kind][idx]
    pos = choices.index(leaf.params[idx])
    step = int(rng.choice([-1, 1]))
    new_pos = min(max(pos + step, 0), len(choices) - 1)
    params = list(leaf.params)
    params[idx] = choices[new_pos]
    params_t = tuple(params)
    if leaf.kind == "ma" and params_t[0] >= params_t[1]:
        return leaf
    return Leaf(leaf.kind, params_t, leaf.inv)


def mutate(cand: Candidate, rng: np.random.Generator, assets: list[str]) -> Candidate:
    asset, rule = cand.asset, cand.rule
    r = rng.random()
    if r < 0.15:
        asset = assets[rng.integers(len(assets))]
    else:
        ps = paths(rule)
        path = ps[rng.integers(len(ps))]
        sub = get(rule, path)
        if isinstance(sub, Leaf):
            if rng.random() < 0.2 and depth(rule) < MAX_DEPTH:
                op = "and" if rng.random() < 0.6 else "or"
                new: Rule = Node(op, sub, random_leaf(rng))
            else:
                new = _mutate_leaf(sub, rng)
        else:
            q = rng.random()
            if q < 0.4:
                new = Node("or" if sub.op == "and" else "and", sub.left, sub.right)
            elif q < 0.7:
                new = sub.left if rng.random() < 0.5 else sub.right  # prune
            else:
                new = random_rule(rng, max_depth=1)
        rule = replace(rule, path, new)
    return Candidate(asset, rule)


def crossover(a: Candidate, b: Candidate, rng: np.random.Generator) -> Candidate:
    pa = paths(a.rule)[rng.integers(len(paths(a.rule)))]
    pb = paths(b.rule)[rng.integers(len(paths(b.rule)))]
    child = replace(a.rule, pa, get(b.rule, pb))
    if depth(child) > MAX_DEPTH:
        child = a.rule
    return Candidate(a.asset if rng.random() < 0.5 else b.asset, child)


# ---------------------------------------------------------------- text and JSON


def describe(rule: Rule) -> str:
    if isinstance(rule, Leaf):
        p = rule.params
        if rule.kind == "ma":
            s = f"SMA{p[0]}>SMA{p[1]}"
        elif rule.kind == "mom":
            s = f"ret{p[0]}>{p[1]:+.0%}"
        elif rule.kind == "rsi":
            s = f"RSI{p[0]}{'>' if p[2] else '<'}{p[1]}"
        elif rule.kind == "z":
            s = f"z{p[0]}>{p[1]}" if p[2] else f"z{p[0]}<-{p[1]}"
        else:
            s = f"vol{p[0]}/{p[1]}{'>' if p[3] else '<'}{p[2]}"
        return f"NOT({s})" if rule.inv else s
    return f"({describe(rule.left)} {rule.op.upper()} {describe(rule.right)})"


def describe_candidate(c: Candidate) -> str:
    return f"{c.asset}: {describe(c.rule)}"


def rule_to_json(rule: Rule):
    if isinstance(rule, Leaf):
        return ["leaf", rule.kind, list(rule.params), rule.inv]
    return [rule.op, rule_to_json(rule.left), rule_to_json(rule.right)]


def rule_from_json(obj) -> Rule:
    if obj[0] == "leaf":
        return Leaf(obj[1], tuple(obj[2]), bool(obj[3]))
    return Node(obj[0], rule_from_json(obj[1]), rule_from_json(obj[2]))


def candidate_to_json(c: Candidate):
    return [c.asset, rule_to_json(c.rule)]


def candidate_from_json(obj) -> Candidate:
    return Candidate(obj[0], rule_from_json(obj[1]))
