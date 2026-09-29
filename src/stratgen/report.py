"""Out-of-sample audit of a finished search: tables, overfitting statistics, figures."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from overfit import (
    cscv_pbo,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from scipy.stats import spearmanr

from .baselines import baselines
from .config import COST_BPS, MIN_CHANGES, SPLIT_DATE, TRADING_DAYS
from .engine import Evaluator, ann_sharpe, perf
from .grammar import Candidate, candidate_from_json, candidate_to_json, describe_candidate
from .search import SearchResult

N_TOP = 10
PBO_BLOCKS = 10  # C(10,5) = 252 splits; S=16 needs ~15 GB of (splits x N) arrays at this N
CORR_CAP = 0.9  # a top strategy may not be >0.9 correlated (discovery) with a better one


def save_search(res: SearchResult, path: Path, meta: dict) -> None:
    payload = {
        "meta": meta | {"seed": res.seed, "trials": res.trials},
        "candidates": [candidate_to_json(c) for c in res.candidates],
        "is_sharpe": [round(float(x), 6) for x in res.fitness],
        "stage": res.stage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f, separators=(",", ":"))


def load_search(path: Path) -> tuple[dict, list[Candidate], list[str]]:
    with gzip.open(path, "rt") as f:
        p = json.load(f)
    return p["meta"], [candidate_from_json(c) for c in p["candidates"]], p["stage"]


def _sharpe_cols(x: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Annualised Sharpe of every column (NaN for constant columns), in float64 chunks."""
    out = np.empty(x.shape[1])
    for a in range(0, x.shape[1], chunk):
        blk = x[:, a : a + chunk].astype(float)
        sd = blk.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[a : a + chunk] = np.where(sd < 1e-14, np.nan, blk.mean(axis=0) / sd) * np.sqrt(
                TRADING_DAYS
            )
    return out


def _distinct_top(is_ret: np.ndarray, is_sr: np.ndarray, eligible: np.ndarray) -> list[int]:
    order = [i for i in np.argsort(-np.where(eligible, is_sr, -np.inf)) if eligible[i]]
    chosen: list[int] = []
    for i in order:
        if all(
            abs(np.corrcoef(is_ret[:, i], is_ret[:, j].astype(float))[0, 1]) < CORR_CAP
            for j in chosen
        ):
            chosen.append(int(i))
        if len(chosen) == N_TOP:
            break
    return chosen


def analyse(prices: pd.DataFrame, cands: list[Candidate], meta: dict) -> dict:
    ev = Evaluator(prices, meta["cost_bps"])
    is_mask = np.asarray(prices.index < pd.Timestamp(SPLIT_DATE))
    n_is = int(is_mask.sum())
    n_trials = len(cands)

    full = np.empty((len(prices), n_trials), dtype=np.float32)
    changes = np.empty(n_trials, dtype=int)
    for j, c in enumerate(cands):
        full[:, j] = ev.returns(c)
        changes[j] = ev.n_changes(c, upto=n_is)
    is_ret, oos_ret = full[:n_is], full[n_is:]

    is_sr, oos_sr = _sharpe_cols(is_ret), _sharpe_cols(oos_ret)
    tradeable = (changes >= MIN_CHANGES) & np.isfinite(is_sr)
    both = tradeable & np.isfinite(oos_sr)

    # PBO by CSCV on the discovery-period return matrix of all tradeable candidates.
    pbo = cscv_pbo(pd.DataFrame(is_ret[:, tradeable]), n_blocks=PBO_BLOCKS)

    # Cross-sectional variance of per-period Sharpe ratios across the trials, for the DSR.
    sr_var = float(np.nanvar(is_sr[tradeable] / np.sqrt(TRADING_DAYS), ddof=1))

    top = _distinct_top(is_ret, is_sr, tradeable)
    bl = baselines(ev)
    bl_is = {k: perf(v[:n_is]) for k, v in bl.items()}
    bl_oos = {k: perf(v[n_is:]) for k, v in bl.items()}

    oos_top_sr_pp = [sharpe_ratio(oos_ret[:, j].astype(float)) for j in top]
    var_top = float(np.nanvar(oos_top_sr_pp, ddof=1))
    spy_oos = bl_oos["Buy and hold SPY"]["sharpe"]
    rows = []
    for j in top:
        c = cands[j]
        asset_oos = ann_sharpe(ev.asset_returns(c.asset)[n_is:])
        rows.append(
            {
                "strategy": describe_candidate(c),
                "asset": c.asset,
                "is_sharpe": is_sr[j],
                "oos_sharpe": oos_sr[j],
                "oos_cagr": perf(oos_ret[:, j].astype(float))["cagr"],
                "oos_max_dd": perf(oos_ret[:, j].astype(float))["max_dd"],
                "asset_bh_oos_sharpe": asset_oos,
                "dsr_is": deflated_sharpe_ratio(is_ret[:, j].astype(float), n_trials, sr_var),
                "oos_psr0": probabilistic_sharpe_ratio(oos_ret[:, j].astype(float), 0.0),
                "oos_dsr_top10": deflated_sharpe_ratio(
                    oos_ret[:, j].astype(float), len(top), var_top
                ),
                "idx": j,
            }
        )
    table = pd.DataFrame(rows)
    table["survives"] = (
        (table.dsr_is >= 0.95) & (table.oos_psr0 >= 0.95) & (table.oos_sharpe > spy_oos)
    )

    top_decile = both & (is_sr >= np.nanquantile(is_sr[tradeable], 0.9))
    rho = spearmanr(is_sr[both], oos_sr[both])
    slope, intercept = np.polyfit(is_sr[both], oos_sr[both], 1)
    summary = {
        "seed": meta["seed"],
        "trials_unique_evaluated": n_trials,
        "tradeable": int(tradeable.sum()),
        "cost_bps_one_way": meta["cost_bps"],
        "split_date": SPLIT_DATE,
        "discovery_days": n_is,
        "validation_days": int(len(prices) - n_is),
        "discovery_range": [str(prices.index[0].date()), str(prices.index[n_is - 1].date())],
        "validation_range": [str(prices.index[n_is].date()), str(prices.index[-1].date())],
        "spearman_is_oos": float(rho.statistic),
        "ols_oos_on_is": {"slope": float(slope), "intercept": float(intercept)},
        "median_is_sharpe": float(np.nanmedian(is_sr[tradeable])),
        "median_oos_sharpe": float(np.nanmedian(oos_sr[both])),
        "share_is_sharpe_gt_1": float(np.mean(is_sr[tradeable] > 1.0)),
        "share_oos_sharpe_gt_0_given_is_gt_1": float(np.mean(oos_sr[both & (is_sr > 1.0)] > 0))
        if np.any(both & (is_sr > 1.0))
        else None,
        "top_decile_mean_is": float(np.nanmean(is_sr[top_decile])),
        "top_decile_mean_oos": float(np.nanmean(oos_sr[top_decile])),
        "n_beating_spy_oos": int(np.sum(oos_sr[both] > spy_oos)),
        "top10_mean_oos_percentile": float(
            np.mean([np.mean(oos_sr[both] < oos_sr[j]) for j in top if both[j]])
        ),
        "pbo": float(pbo.pbo),
        "pbo_prob_loss": float(pbo.prob_loss),
        "pbo_blocks": PBO_BLOCKS,
        "pbo_combinations": int(pbo.n_combinations),
        "pbo_n_strategies": int(pbo.n_strategies),
        "sr_variance_per_period": sr_var,
        "n_survivors_top10": int(table.survives.sum()),
        "baselines_is": bl_is,
        "baselines_oos": bl_oos,
    }
    curves = {
        "top": {describe_candidate(cands[j]): oos_ret[:, j].astype(float) for j in top[:5]},
        "baselines": {k: v[n_is:] for k, v in bl.items()},
        "dates": prices.index[n_is:],
    }
    pos_in_both = {int(j): k for k, j in enumerate(np.flatnonzero(both))}
    scatter = {
        "is": is_sr[both],
        "oos": oos_sr[both],
        "top_idx": [pos_in_both[j] for j in top if j in pos_in_both],
    }
    return {
        "summary": summary,
        "table": table.drop(columns="idx"),
        "curves": curves,
        "scatter": scatter,
        "pbo": pbo,
    }


def markdown_table(table: pd.DataFrame) -> str:
    cols = [
        ("strategy", "Strategy"),
        ("is_sharpe", "IS SR"),
        ("oos_sharpe", "OOS SR"),
        ("asset_bh_oos_sharpe", "Asset B&H OOS SR"),
        ("dsr_is", "DSR (IS, N trials)"),
        ("oos_psr0", "OOS PSR"),
        ("survives", "Survives"),
    ]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        cells = []
        for k, _ in cols:
            v = r[k]
            cells.append(
                ("yes" if v else "no") if k == "survives" else v if k == "strategy" else f"{v:.2f}"
            )
        lines.append("| " + " | ".join(str(x) for x in cells) + " |")
    return "\n".join(lines)


def baseline_markdown(s: dict) -> str:
    lines = [
        "| Baseline | IS SR | OOS SR | OOS CAGR | OOS vol | OOS max DD |",
        "|---|---|---|---|---|---|",
    ]
    for k in s["baselines_oos"]:
        i, o = s["baselines_is"][k], s["baselines_oos"][k]
        lines.append(
            f"| {k} | {i['sharpe']:.2f} | {o['sharpe']:.2f} | {o['cagr']:.1%} | "
            f"{o['vol']:.1%} | {o['max_dd']:.1%} |"
        )
    return "\n".join(lines)


def write_outputs(result: dict, results_dir: Path, fig_dir: Path) -> None:
    from . import plots

    results_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    result["table"].to_csv(results_dir / "top_strategies.csv", index=False, float_format="%.4f")
    (results_dir / "tables.md").write_text(
        markdown_table(result["table"]) + "\n\n" + baseline_markdown(result["summary"]) + "\n"
    )
    plots.decay(result["scatter"], result["summary"], fig_dir / "decay.png")
    plots.equity(result["curves"], fig_dir / "equity_oos.png")
    plots.pbo_hist(result["pbo"], fig_dir / "pbo_logits.png")


__all__ = ["COST_BPS", "TRADING_DAYS", "analyse", "load_search", "save_search", "write_outputs"]
