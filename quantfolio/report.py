"""
Charts and analysis report.

Produces the standard portfolio-analysis visuals: efficient frontier,
correlation matrix, Monte Carlo fan, drawdowns, cumulative performance,
strategy comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import TRADING_DAYS
from .metrics import drawdown_series
from .montecarlo import MonteCarloResult

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    cloud: pd.DataFrame,
    highlights: dict[str, tuple[float, float]],
    assets: pd.DataFrame | None = None,
    path: str | Path = "efficient_frontier.png",
):
    """
    Efficient frontier + cloud of random portfolios.
    `highlights`: {name: (volatility, return)} for MaxSharpe, MinVol, etc.
    `assets`: DataFrame with 'volatility'/'return' columns for single assets.
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(cloud["volatility"], cloud["return"],
                    c=cloud["return"] / cloud["volatility"],
                    cmap="viridis", s=8, alpha=0.35)
    fig.colorbar(sc, label="Sharpe (rf=0)")
    ax.plot(frontier["volatility"], frontier["return"],
            color="crimson", lw=2.5, label="Efficient frontier")

    markers = {"MaxSharpe": ("*", 350), "MinVol": ("D", 90)}
    for name, (vol, ret) in highlights.items():
        mk, sz = markers.get(name, ("o", 90))
        ax.scatter([vol], [ret], marker=mk, s=sz, zorder=5,
                   edgecolors="black", label=name)

    if assets is not None:
        ax.scatter(assets["volatility"], assets["return"],
                   marker="x", color="gray", s=60)
        for tk, row in assets.iterrows():
            ax.annotate(tk, (row["volatility"], row["return"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annualized return")
    ax.set_title("Markowitz efficient frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation_matrix(returns: pd.DataFrame, path="correlation.png"):
    """Correlation matrix: the heart of diversification."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im)
    ax.set_title("Return correlation matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_monte_carlo(result: MonteCarloResult, path="monte_carlo.png", n_show=150):
    """Fan of simulated paths + percentile bands."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    days = np.arange(result.paths.shape[0])

    ax1.plot(days, result.paths[:, :n_show], color="steelblue", alpha=0.08, lw=0.7)
    for p, style in [(5, ":"), (50, "-"), (95, ":")]:
        ax1.plot(days, np.percentile(result.paths, p, axis=1),
                 color="crimson", ls=style, lw=2, label=f"P{p}")
    ax1.axhline(result.initial_value, color="black", lw=1, ls="--")
    ax1.set_xlabel("Trading days")
    ax1.set_ylabel("Portfolio value")
    ax1.set_title(f"Monte Carlo - {result.method}\n"
                  f"{result.paths.shape[1]:,} simulations")
    ax1.legend()

    ax2.hist(result.terminal_values, bins=60, color="steelblue",
             edgecolor="white", alpha=0.85)
    ax2.axvline(result.initial_value, color="black", ls="--", lw=1.2,
                label="Initial value")
    ax2.axvline(result.percentiles[50], color="crimson", lw=1.5, label="Median")
    ax2.set_xlabel("Final value")
    ax2.set_title("Terminal distribution")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drawdowns(returns: dict[str, pd.Series], path="drawdowns.png"):
    """Overlaid drawdown curves."""
    fig, ax = plt.subplots()
    for name, r in returns.items():
        dd = drawdown_series(r)
        ax.plot(dd.index, dd, lw=1.2, label=name)
        ax.fill_between(dd.index, dd, 0, alpha=0.12)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdowns: decline from the all-time high")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity_curves(curves: pd.DataFrame, path="equity_curves.png",
                       title="Backtest: strategy comparison"):
    """Value curves of the backtested strategies (log scale)."""
    fig, ax = plt.subplots()
    for col in curves.columns:
        ax.plot(curves.index, curves[col], lw=1.4, label=col)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value (log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_weights(strategies: dict[str, pd.Series], path="weights.png"):
    """Compared allocations, grouped bars."""
    df = pd.DataFrame(strategies)
    fig, ax = plt.subplots()
    df.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Weight")
    ax.set_title("Allocations by strategy")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
