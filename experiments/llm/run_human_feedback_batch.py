#!/usr/bin/env python3
"""Batch runner for human feedback estimation.

Given a list of feedback files located in `human_feedbacks/`, this script
selects the matching visits file(s) under `human_feedback_visits/` based on the
lambda value stored in the feedback metadata, and dispatches the existing
`llm-human-feedback` Make target for each combination.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

# Mapping strings found in visit filenames to algorithm names expected by Hydra.
VISIT_ALGO_MAP = {
    "dsn": "design",
    "rand": "random",
}

VISIT_EPISODE_PATTERN = re.compile(r"ep(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback-files",
        required=True,
        help=(
            "Comma or whitespace separated list of feedback filenames located in "
            "human_feedbacks/. Both bare filenames (e.g. 1.jsonl) and paths "
            "relative to the llm directory are accepted."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help=(
            "Optional comma/space separated subset of algorithms to run. "
            "Supported values: design, random. Defaults to whichever visit files "
            "are found for the inferred lambda."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="results/human-feedback",
        help="Base directory (relative to llm/) where run outputs will be stored.",
    )
    parser.add_argument(
        "--feedback-num-rounds",
        type=int,
        default=None,
        help="Optional override for feedback.num_rounds",
    )
    parser.add_argument(
        "--num-benchmark-episodes",
        type=int,
        default=None,
        help="Optional override for num_benchmark_episodes",
    )
    parser.add_argument(
        "--benchmark-episodes",
        default="",
        help=(
            "Optional explicit benchmark episode selection, e.g. '70-79' or "
            "'design:65-74,random:70-79'."
        ),
    )
    parser.add_argument(
        "--estimation-episodes",
        default="",
        help=(
            "Optional explicit training episode selection, e.g. 'design:0-39,random:0-39'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the make commands that would be executed without running them.",
    )
    return parser.parse_args()


def split_tokens(value: str) -> List[str]:
    return [token for token in value.replace(",", " ").split() if token]


def normalise_feedback_path(raw: str, llm_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path

    candidates = [llm_dir / path, llm_dir / "human_feedbacks" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Default to human_feedbacks even if the file does not yet exist to provide
    # a useful error message downstream.
    return (llm_dir / "human_feedbacks" / path).resolve()


def format_lambda_for_dir(lamb: float) -> str:
    # Keep at most 4 decimal digits while preserving trailing zeros needed for directory names.
    formatted = f"{lamb:.4f}".rstrip("0").rstrip(".")
    return formatted


def find_visit_paths(lam_dir: Path, allowed_algorithms: Iterable[str]) -> List[tuple[Path, str]]:
    visit_pairs: List[tuple[Path, str]] = []
    if not lam_dir.is_dir():
        raise FileNotFoundError(f"Visits directory not found: {lam_dir}")

    allowed_set = {alg.lower() for alg in allowed_algorithms}
    for visit_path in sorted(lam_dir.glob("visits-feedback-*.pkl")):
        algo = None
        for key, name in VISIT_ALGO_MAP.items():
            if key in visit_path.name:
                algo = name
                break
        if algo is None:
            continue
        if allowed_set and algo not in allowed_set:
            continue
        visit_pairs.append((visit_path, algo))
    if allowed_set and not visit_pairs:
        raise ValueError(
            f"No visit files matching algorithms {sorted(allowed_set)} in {lam_dir}"
        )
    return visit_pairs


def dispatch_make(
    experiments_dir: Path,
    visit_rel: str,
    feedback_rel: str,
    results_rel: str,
    algorithm: str,
    dry_run: bool,
    lambda_override: Optional[str] = None,
    episodes_override: Optional[str] = None,
    num_rounds_override: Optional[str] = None,
    benchmark_override: Optional[str] = None,
    benchmark_spec_override: Optional[str] = None,
    estimation_spec_override: Optional[str] = None,
) -> None:
    cmd = [
        "make",
        "-C",
        str(experiments_dir),
        "llm-human-feedback",
        f"VISITS_PATH={visit_rel}",
        f"FEEDBACK_PATH={feedback_rel}",
        f"RESULTS_DIR={results_rel}",
        f"ALGORITHM={algorithm}",
    ]
    if lambda_override is not None:
        cmd.append(f"FEEDBACK_LAMBDA={lambda_override}")
    if episodes_override is not None:
        cmd.append(f"EXPERIMENT_EPISODES={episodes_override}")
    if num_rounds_override is not None:
        cmd.append(f"FEEDBACK_NUM_ROUNDS={num_rounds_override}")
    if benchmark_override is not None:
        cmd.append(f"NUM_BENCHMARK_EPISODES={benchmark_override}")
    if benchmark_spec_override:
        cmd.append(f"BENCHMARK_EPISODES={benchmark_spec_override}")
    if estimation_spec_override:
        cmd.append(f"ESTIMATION_EPISODES={estimation_spec_override}")
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    experiments_dir = Path(__file__).resolve().parents[1]
    llm_dir = experiments_dir / "llm"

    feedback_items = split_tokens(args.feedback_files)
    if not feedback_items:
        raise ValueError("No feedback files provided")

    requested_algorithms = split_tokens(args.algorithms)

    for item in feedback_items:
        feedback_path = normalise_feedback_path(item, llm_dir)
        if not feedback_path.exists():
            raise FileNotFoundError(f"Feedback file does not exist: {feedback_path}")
        if feedback_path.suffix != ".jsonl":
            raise ValueError(f"Feedback file must end with .jsonl: {feedback_path.name}")

        data = json.loads(feedback_path.read_text())
        metadata = data.get("metadata", {})
        lamb = metadata.get("lambda")
        visits_subdir = metadata.get("visits_subdir")

        if visits_subdir:
            visits_dir = llm_dir / "human_feedback_visits" / visits_subdir
        else:
            if lamb is None:
                raise KeyError(
                    f"Feedback metadata for {feedback_path.name} is missing 'lambda'; "
                    "cannot infer visits directory"
                )
            lam_str = format_lambda_for_dir(float(lamb))
            visits_dir = llm_dir / "human_feedback_visits" / f"lam{lam_str}"

        visit_pairs = find_visit_paths(visits_dir, requested_algorithms or VISIT_ALGO_MAP.values())

        feedback_rel = feedback_path.relative_to(llm_dir)
        feedback_name = feedback_path.stem
        lambda_override = None
        if lamb is not None:
            try:
                lambda_override = f"{float(lamb):g}"
            except (TypeError, ValueError):
                lambda_override = str(lamb)

        num_rounds_override = None
        if args.feedback_num_rounds is not None:
            num_rounds_override = str(args.feedback_num_rounds)

        benchmark_override = None
        if args.num_benchmark_episodes is not None:
            benchmark_override = str(args.num_benchmark_episodes)

        benchmark_spec_override = args.benchmark_episodes.strip() or None
        estimation_spec_override = args.estimation_episodes.strip() or None

        for visit_path, algorithm in visit_pairs:
            visit_rel = visit_path.relative_to(llm_dir)
            results_rel = Path(args.results_root) / feedback_name / algorithm
            episodes_override = None
            match = VISIT_EPISODE_PATTERN.search(visit_path.name)
            if match:
                episodes_override = match.group(1)
            dispatch_make(
                experiments_dir,
                str(visit_rel),
                str(feedback_rel),
                str(results_rel),
                algorithm,
                args.dry_run,
                lambda_override=lambda_override,
                episodes_override=episodes_override,
                num_rounds_override=num_rounds_override,
                benchmark_override=benchmark_override,
                benchmark_spec_override=benchmark_spec_override,
                estimation_spec_override=estimation_spec_override,
            )


if __name__ == "__main__":
    main()
