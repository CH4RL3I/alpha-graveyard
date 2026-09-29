"""Figures. Restrained style: one accent colour, grey for the bulk, direct annotation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#c0392b"
GREY = "#7f8c8d"
BLUE = "#2c5d8f"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 130,
    }
)


def decay(scatter: dict, s: dict, path: Path) -> None:
    x, y = scatter["is"], scatter["oos"]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.scatter(x, y, s=4, alpha=0.18, color=GREY, linewidths=0, label=f"all {len(x):,} candidates")
    tx, ty = x[scatter["top_idx"]], y[scatter["top_idx"]]
    ax.scatter(tx, ty, s=42, color=ACCENT, zorder=3, label="top 10 by discovery Sharpe")
    lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--", label="no decay (OOS = IS)")
    b = s["ols_oos_on_is"]
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b["intercept"] + b["slope"] * xs, color=BLUE, lw=1.6, label="OLS fit")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    d0, d1 = (y[:4] for y in s["discovery_range"])
    v0, v1 = (y[:4] for y in s["validation_range"])
    ax.set_xlabel(f"Discovery Sharpe (annualised, {d0}-{d1})")
    ax.set_ylabel(f"Held-out Sharpe (annualised, {v0}-{v1})")
    ax.set_title(
        f"Discovery vs held-out Sharpe: Spearman {s['spearman_is_oos']:.2f}, "
        f"slope {b['slope']:.2f}",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def equity(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.6))
    d = curves["dates"]
    palette = ["#e67e22", "#c0392b", "#8e44ad", "#d35400", "#e74c3c"]
    for (name, r), col in zip(curves["top"].items(), palette, strict=False):
        ax.plot(d, np.cumprod(1 + r), color=col, lw=1.0, alpha=0.8, label=name[:70])
    base_cols = ["black", BLUE, "#16a085", "#7f8c8d"]
    for (name, r), col in zip(curves["baselines"].items(), base_cols, strict=False):
        ax.plot(d, np.cumprod(1 + r), color=col, lw=2.0, label=name)
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        "Held-out period: top discovered strategies (thin) vs baselines (thick)",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pbo_hist(pbo, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    lam = pbo.logits[np.isfinite(pbo.logits)]
    ax.hist(lam, bins=25, color=GREY, alpha=0.85)
    ax.axvline(0, color=ACCENT, lw=1.6)
    ax.set_xlabel("Logit of the discovery winner's relative out-of-sample rank")
    ax.set_ylabel(f"CSCV splits (of {pbo.n_combinations:,})")
    ax.set_title(
        f"PBO = {pbo.pbo:.2f} over {pbo.n_strategies:,} candidates "
        "(share of splits left of the red line)",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
