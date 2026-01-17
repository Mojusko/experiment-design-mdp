#!/usr/bin/env python3
"""Plot post-rebuttal GPT feedback results.

Generates:
1. Lambda comparison figure (λ=0.1, 1, 10, 100 @ 50 train episodes)
2. Episode comparison figure (10-50 train episodes @ λ=100)
3. All styles grid with episode curves (10 themes × 5 episode points @ λ=100)
4. Summary bar chart
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
LLM_DIR = EXPERIMENTS_DIR / "llm"
CURVES_DIR = LLM_DIR / "analysis" / "holdout_curves"
FIGS_DIR = LLM_DIR / "analysis" / "figs"

ORDERED_STYLES = [
    "futuristic", "sunny", "forest", "landscape", "ancient",
    "noir", "watercolor", "cyberpunk", "minimalist", "medieval"
]

STYLE_TITLES = {
    "futuristic": "Futuristic",
    "sunny": "Sunny",
    "forest": "Forest",
    "landscape": "Landscape",
    "ancient": "Ancient",
    "noir": "Noir",
    "watercolor": "Watercolor",
    "cyberpunk": "Cyberpunk",
    "minimalist": "Minimalist",
    "medieval": "Medieval",
}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean_ci(values: List[float], ci_mult: float = 1.0) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, ci_mult * se


def plot_lambda_comparison(rows: List[Dict], outdir: Path, font_size: float = 14.0):
    """Plot accuracy vs lambda (0.1, 1, 10, 100) at 50 train episodes."""
    # Filter to 50 train episodes only
    filtered = [r for r in rows if int(r["estimation_size"]) == 50]

    # Group by lambda and algorithm
    by_lambda: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        lam = float(r["lambda"])
        alg = r["algorithm"]
        acc = float(r["accuracy"])
        by_lambda[lam][alg].append(acc)

    lambdas = sorted(by_lambda.keys())
    if not lambdas:
        print("No data for lambda comparison")
        return

    colors = {"design": "#1f77b4", "random": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for alg in ["design", "random"]:
        xs, means, cis = [], [], []
        for lam in lambdas:
            if alg in by_lambda[lam]:
                vals = by_lambda[lam][alg]
                m, ci = mean_ci(vals)
                xs.append(lam)
                means.append(m)
                cis.append(ci)

        if xs:
            xs_np = np.array(xs)
            means_np = np.array(means)
            cis_np = np.array(cis)
            ax.plot(xs_np, means_np, "-o", color=colors[alg], label=alg.capitalize())
            ax.fill_between(xs_np, means_np - cis_np, means_np + cis_np,
                          color=colors[alg], alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization λ", fontsize=font_size)
    ax.set_ylabel("Held-out accuracy", fontsize=font_size)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.tick_params(axis="both", labelsize=font_size - 1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=font_size - 1, loc="best")
    ax.set_title("50 training episodes", fontsize=font_size)

    fig.tight_layout()
    out_path = outdir / "gpt_lambda_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_episode_comparison(rows: List[Dict], outdir: Path, font_size: float = 14.0):
    """Plot accuracy vs train episodes (10-50) at λ=100."""
    # Filter to lambda=100
    filtered = [r for r in rows if float(r["lambda"]) == 100]

    # Group by episodes and algorithm
    by_ep: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        ep = int(r["estimation_size"])
        alg = r["algorithm"]
        acc = float(r["accuracy"])
        by_ep[ep][alg].append(acc)

    episodes = sorted(by_ep.keys())
    if not episodes:
        print("No data for episode comparison")
        return

    colors = {"design": "#1f77b4", "random": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for alg in ["design", "random"]:
        xs, means, cis = [], [], []
        for ep in episodes:
            if alg in by_ep[ep]:
                vals = by_ep[ep][alg]
                m, ci = mean_ci(vals)
                xs.append(ep)
                means.append(m)
                cis.append(ci)

        if xs:
            xs_np = np.array(xs)
            means_np = np.array(means)
            cis_np = np.array(cis)
            ax.plot(xs_np, means_np, "-o", color=colors[alg], label=alg.capitalize())
            ax.fill_between(xs_np, means_np - cis_np, means_np + cis_np,
                          color=colors[alg], alpha=0.15, linewidth=0)

    ax.set_xlabel("Training episodes", fontsize=font_size)
    ax.set_ylabel("Held-out accuracy", fontsize=font_size)
    ax.set_xticks(episodes)
    ax.tick_params(axis="both", labelsize=font_size - 1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=font_size - 1, loc="best")
    ax.set_title("λ = 100", fontsize=font_size)

    fig.tight_layout()
    out_path = outdir / "gpt_episode_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all_styles_grid(rows: List[Dict], outdir: Path, font_size: float = 10.0):
    """Plot per-style accuracy curves (all 10 styles) at λ=100."""
    # Filter to lambda=100
    filtered = [r for r in rows if float(r["lambda"]) == 100]

    # Group by style, episode, algorithm
    by_style: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in filtered:
        style = r["style"]
        ep = int(r["estimation_size"])
        alg = r["algorithm"]
        acc = float(r["accuracy"])
        by_style[style][ep][alg] = acc

    # Order styles
    styles = [s for s in ORDERED_STYLES if s in by_style]
    if not styles:
        print("No data for styles grid")
        return

    episodes = sorted(set(ep for sd in by_style.values() for ep in sd.keys()))

    colors = {"design": "#1f77b4", "random": "#7f7f7f"}

    n = len(styles)
    cols = 5
    rows_count = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(2.8 * cols, 2.2 * rows_count), sharey=True)
    if rows_count == 1:
        axes = np.array([axes])

    legend_handles = {}
    all_vals = []

    for idx, style in enumerate(styles):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        sd = by_style[style]

        for alg in ["design", "random"]:
            xs, ys = [], []
            for ep in episodes:
                if ep in sd and alg in sd[ep]:
                    xs.append(ep)
                    ys.append(sd[ep][alg])
                    all_vals.append(sd[ep][alg])
            if xs:
                handle, = ax.plot(xs, ys, "-o", color=colors[alg], lw=1.2, ms=4)
                legend_handles[alg] = handle

        ax.set_title(STYLE_TITLES.get(style, style), fontsize=font_size)
        ax.set_xticks(episodes)
        ax.tick_params(axis="both", which="major", labelsize=font_size - 1)
        ax.grid(True, alpha=0.2)
        if c == 0:
            ax.set_ylabel("Accuracy", fontsize=font_size)
        if r == rows_count - 1:
            ax.set_xlabel("Train ep.", fontsize=font_size - 1)

    # Hide unused subplots
    for idx in range(n, rows_count * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    # Set common y-axis limits
    if all_vals:
        ymin = max(0.0, min(all_vals) - 0.08)
        ymax = min(1.0, max(all_vals) + 0.08)
        for ax in axes.flat:
            if ax.has_data():
                ax.set_ylim(ymin, ymax)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    if all(k in legend_handles for k in ("design", "random")):
        fig.legend(
            [legend_handles["design"], legend_handles["random"]],
            ["Design", "Random"],
            loc="lower center",
            ncol=2,
            fontsize=font_size,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.suptitle("Per-style accuracy (λ=100, GPT-4.1-mini feedback)", fontsize=font_size + 1, y=0.99)

    out_path = outdir / "gpt_styles_all_lambda100.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(rows: List[Dict], outdir: Path, train_ep: int = 50, font_size: float = 14.0):
    """Plot summary bar chart at specified train episodes."""
    # Filter to lambda=100 and specified episodes
    filtered = [r for r in rows if float(r["lambda"]) == 100 and int(r["estimation_size"]) == train_ep]

    by_alg: Dict[str, List[float]] = defaultdict(list)
    for r in filtered:
        by_alg[r["algorithm"]].append(float(r["accuracy"]))

    if not by_alg:
        print("No data for bar chart")
        return

    colors = {"design": "#1f77b4", "random": "#7f7f7f", "guess": "#c7c7c7"}

    entries = []
    for alg, label in [("design", "ED-PBRL"), ("random", "Random Exp.")]:
        if alg in by_alg:
            m, ci = mean_ci(by_alg[alg])
            entries.append((label, m, ci, colors[alg]))
    entries.append(("Random Guess", 0.25, 0.0, colors["guess"]))

    fig, ax = plt.subplots(figsize=(4.6, 4.4))

    x = np.arange(len(entries))
    labels = [e[0] for e in entries]
    means = np.array([e[1] for e in entries])
    errs = np.array([e[2] for e in entries])
    colors_bar = [e[3] for e in entries]

    ax.bar(x, means, yerr=errs, color=colors_bar, capsize=6, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=font_size - 1)
    ax.set_ylabel("Held-out accuracy", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size - 1)
    ax.set_ylim(0.0, 0.7)
    ax.set_yticks(np.linspace(0.0, 0.6, 4))
    ax.set_title(f"Held-out accuracy\n({train_ep} training episodes, λ=100)", fontsize=font_size)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_path = outdir / f"gpt_accuracy_bar_lambda100_ep{train_ep}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CURVES_DIR / "post_rebuttal_all.csv"))
    parser.add_argument("--outdir", default=str(FIGS_DIR))
    parser.add_argument("--font-size", type=float, default=14.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.csv))
    print(f"Loaded {len(rows)} rows from {args.csv}")

    # Generate all figures
    plot_lambda_comparison(rows, outdir, args.font_size)
    plot_episode_comparison(rows, outdir, args.font_size)
    plot_all_styles_grid(rows, outdir, font_size=10.0)
    plot_summary_bar(rows, outdir, train_ep=50, font_size=args.font_size)

    print(f"\nAll figures saved to {outdir}")


if __name__ == "__main__":
    main()
