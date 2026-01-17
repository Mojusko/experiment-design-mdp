#!/usr/bin/env python3
"""Collect post-rebuttal GPT feedback results into CSVs for plotting.

Traverses results/human-feedback-post-rebuttal[-Nep]/lambda<L>/<theme>/<alg>/fold<F>/
and aggregates metrics across folds.

Outputs:
  - holdout_curves/post_rebuttal_lambda<L>.csv (one per lambda, all episode counts)
  - holdout_curves/post_rebuttal_all.csv (combined)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
LLM_DIR = EXPERIMENTS_DIR / "llm"
RESULTS_DIR = LLM_DIR / "results"
CURVES_DIR = LLM_DIR / "analysis" / "holdout_curves"

THEMES = [
    "ancient", "cyberpunk", "forest", "futuristic", "landscape",
    "medieval", "minimalist", "noir", "sunny", "watercolor"
]

LAMBDA_VALUES = [0.1, 1, 10, 100]
EPISODE_COUNTS = [10, 20, 30, 40, 50]


@dataclass
class MetricRow:
    theme: str
    lambda_val: float
    train_episodes: int
    algorithm: str
    accuracy: float
    accuracy_std: float
    correct: int
    comparisons: int
    num_folds: int


def get_results_dir(train_episodes: int) -> Path:
    """Get the results directory for a given train episode count."""
    if train_episodes == 50:
        return RESULTS_DIR / "human-feedback-post-rebuttal"
    else:
        return RESULTS_DIR / f"human-feedback-post-rebuttal-{train_episodes}ep"


def collect_fold_metrics(
    results_dir: Path,
    lambda_val: float,
    theme: str,
    algorithm: str,
) -> List[Dict]:
    """Collect metrics from all folds for a given configuration."""
    lambda_str = f"lambda{lambda_val}" if lambda_val != 0.1 else "lambda0.1"
    alg_dir = results_dir / lambda_str / theme / algorithm

    if not alg_dir.exists():
        return []

    metrics_list = []
    for fold_dir in sorted(alg_dir.glob("fold*")):
        for metrics_file in fold_dir.glob("metrics-*.json"):
            try:
                data = json.loads(metrics_file.read_text())
                metrics_list.append(data)
            except Exception:
                continue

    return metrics_list


def aggregate_metrics(
    metrics_list: List[Dict],
    theme: str,
    lambda_val: float,
    train_episodes: int,
    algorithm: str,
) -> MetricRow | None:
    """Aggregate metrics across folds."""
    if not metrics_list:
        return None

    accuracies = [m.get("benchmark_accuracy", 0.0) for m in metrics_list]
    correct_counts = [m.get("benchmark_correct_count", 0) for m in metrics_list]
    comparison_counts = [m.get("benchmark_comparisons_count", 0) for m in metrics_list]

    return MetricRow(
        theme=theme,
        lambda_val=lambda_val,
        train_episodes=train_episodes,
        algorithm=algorithm,
        accuracy=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        correct=int(np.mean(correct_counts)),
        comparisons=int(np.mean(comparison_counts)),
        num_folds=len(metrics_list),
    )


def collect_all_results() -> List[MetricRow]:
    """Collect all post-rebuttal results."""
    rows: List[MetricRow] = []

    for train_ep in EPISODE_COUNTS:
        results_dir = get_results_dir(train_ep)
        if not results_dir.exists():
            print(f"Warning: {results_dir} does not exist, skipping")
            continue

        # Check which lambdas are available
        available_lambdas = []
        for lam in LAMBDA_VALUES:
            lambda_str = f"lambda{lam}" if lam != 0.1 else "lambda0.1"
            if (results_dir / lambda_str).exists():
                available_lambdas.append(lam)

        for lam in available_lambdas:
            for theme in THEMES:
                for alg in ["design", "random"]:
                    metrics_list = collect_fold_metrics(results_dir, lam, theme, alg)
                    row = aggregate_metrics(metrics_list, theme, lam, train_ep, alg)
                    if row:
                        rows.append(row)

    return rows


def write_csv(rows: List[MetricRow], out_path: Path) -> None:
    """Write rows to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("feedback,lambda,style,bench_spec,estimation_size,algorithm,accuracy,accuracy_std,correct,comparisons,num_folds\n")
        for r in rows:
            # Use 'feedback' field to match existing plot script expectations
            feedback = f"{r.theme}_gpt-4.1-mini"
            f.write(
                f"{feedback},{r.lambda_val},{r.theme},10ep-test,{r.train_episodes},"
                f"{r.algorithm},{r.accuracy},{r.accuracy_std},{r.correct},{r.comparisons},{r.num_folds}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(CURVES_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)

    print("Collecting post-rebuttal results...")
    all_rows = collect_all_results()

    if not all_rows:
        print("No results found!")
        return

    # Write combined CSV
    combined_path = outdir / "post_rebuttal_all.csv"
    write_csv(all_rows, combined_path)
    print(f"Wrote {len(all_rows)} rows to {combined_path}")

    # Write per-lambda CSVs
    by_lambda: Dict[float, List[MetricRow]] = defaultdict(list)
    for r in all_rows:
        by_lambda[r.lambda_val].append(r)

    for lam, lam_rows in sorted(by_lambda.items()):
        lam_str = str(lam).replace(".", "p")
        lam_path = outdir / f"post_rebuttal_lambda{lam_str}.csv"
        write_csv(lam_rows, lam_path)
        print(f"Wrote {len(lam_rows)} rows to {lam_path}")

    # Summary
    print("\n=== Summary ===")
    for lam in sorted(by_lambda.keys()):
        lam_rows = by_lambda[lam]
        episodes = sorted(set(r.train_episodes for r in lam_rows))
        print(f"λ={lam}: {len(lam_rows)} rows, episodes: {episodes}")


if __name__ == "__main__":
    main()
