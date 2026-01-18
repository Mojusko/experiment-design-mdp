#!/usr/bin/env python3
"""Batch runner for human feedback estimation (post-rebuttal format).

Handles the new JSONL format from GPT feedback experiments:
- Line-by-line JSONL: {"image": "...", "model": "...", "preference": N}
- Directory structure: human_feedbacks_post_rebuttal/<lambda>/feedback/*.jsonl
- Visits located in: human_feedbacks_post_rebuttal/<lambda>/visits/*.pkl

Supports rotating test windows for cross-validation to reduce variance.
Converts to the format expected by experiment.py and dispatches make llm-human-feedback.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Mapping strings found in visit filenames to algorithm names expected by Hydra.
VISIT_ALGO_MAP = {
    "dsn": "design",
    "rand": "random",
}

# Pattern to extract algorithm from feedback filename (e.g., ancient_gpt-4.1-mini_dsn_lambda10.jsonl)
FEEDBACK_ALGO_PATTERN = re.compile(r"_(dsn|rand)_")

# Pattern to extract lambda from directory name (e.g., lambda10, lambda0.1)
LAMBDA_DIR_PATTERN = re.compile(r"lambda([\d.]+)")

# Pattern to parse image filename for episode/timestep
IMAGE_PATTERN = re.compile(r"alg-([a-zA-Z0-9_]+)_episode_(\d+)_timestep_(\d+)\.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback-dir",
        default="human_feedbacks_post_rebuttal",
        help="Base directory containing lambda subdirectories (relative to llm/)",
    )
    parser.add_argument(
        "--lambda-values",
        default="",
        help=(
            "Comma/space separated list of lambda values to process. "
            "If empty, processes all lambda directories found."
        ),
    )
    parser.add_argument(
        "--themes",
        default="",
        help=(
            "Comma/space separated list of themes to process (e.g., 'ancient,forest'). "
            "If empty, processes all themes found."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help=(
            "Comma/space separated subset of algorithms to run. "
            "Supported values: design, random. Defaults to both."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="results/human-feedback-post-rebuttal",
        help="Base directory (relative to llm/) where run outputs will be stored.",
    )
    parser.add_argument(
        "--feedback-num-rounds",
        type=int,
        default=None,
        help="Optional override for feedback.num_rounds",
    )
    # New arguments for rotating test windows
    parser.add_argument(
        "--num-train-episodes",
        type=int,
        default=None,
        help="Number of episodes to use for training (e.g., 50)",
    )
    parser.add_argument(
        "--num-test-episodes",
        type=int,
        default=None,
        help="Number of episodes to use for testing (e.g., 10)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Number of rotating folds to run. If not specified, runs all possible folds.",
    )
    # Legacy arguments for explicit episode selection
    parser.add_argument(
        "--num-benchmark-episodes",
        type=int,
        default=None,
        help="[Legacy] Optional override for num_benchmark_episodes (use --num-test-episodes instead)",
    )
    parser.add_argument(
        "--benchmark-episodes",
        default="",
        help="[Legacy] Optional explicit benchmark episode selection, e.g. '50-59'",
    )
    parser.add_argument(
        "--estimation-episodes",
        default="",
        help="[Legacy] Optional explicit training episode selection, e.g. '0-49'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the make commands that would be executed without running them.",
    )
    return parser.parse_args()


def split_tokens(value: str) -> List[str]:
    return [token for token in value.replace(",", " ").split() if token]


def convert_jsonl_to_json(jsonl_path: Path, lambda_value: str) -> dict:
    """Convert line-by-line JSONL to the JSON format expected by experiment.py."""
    preferences = []

    with open(jsonl_path, "r", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace and null bytes (file corruption)
            line = line.strip().replace("\x00", "")
            if not line:
                continue

            # Try to find JSON object in line (handles corruption with embedded JSON)
            json_start = line.find("{")
            if json_start == -1:
                continue
            line = line[json_start:]

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} in {jsonl_path.name}: {e}")
                continue

            # Convert from {"image": "...", "model": "...", "preference": N}
            # to {"filename": "...", "preference": N, "algorithm": "...", "episode": N}
            image_name = entry.get("image", "")
            preference = entry.get("preference")

            if image_name and preference is not None:
                pref_entry = {
                    "filename": image_name,
                    "preference": preference,
                }

                # Extract algorithm and episode from filename for benchmark selection
                match = IMAGE_PATTERN.search(image_name)
                if match:
                    pref_entry["algorithm"] = match.group(1)
                    pref_entry["episode"] = int(match.group(2))

                preferences.append(pref_entry)

    return {
        "metadata": {
            "lambda": float(lambda_value),
            "source": str(jsonl_path.name),
        },
        "preferences": preferences,
    }


def get_max_episode_from_preferences(preferences: List[dict]) -> int:
    """Extract the maximum episode number from preferences list."""
    max_episode = -1
    for pref in preferences:
        filename = pref.get("filename", "")
        match = IMAGE_PATTERN.search(filename)
        if match:
            episode = int(match.group(2))
            max_episode = max(max_episode, episode)
    return max_episode


def episodes_to_spec(episodes: List[int]) -> str:
    """Convert a list of episode numbers to a compact range spec string.

    E.g., [0, 1, 2, 5, 6, 7] -> "0-2,5-7"
    """
    if not episodes:
        return ""

    episodes = sorted(episodes)
    ranges = []
    start = episodes[0]
    end = episodes[0]

    for ep in episodes[1:]:
        if ep == end + 1:
            end = ep
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = ep
            end = ep

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def generate_rotating_folds(
    total_episodes: int,
    num_train: int,
    num_test: int,
    num_folds: Optional[int] = None,
) -> List[Tuple[str, str, int]]:
    """Generate rotating test windows for cross-validation.

    Returns list of (train_spec, test_spec, fold_index) tuples.
    Test windows rotate from end to start: [50-59], [40-49], [30-39], etc.
    Train uses exactly num_train episodes (first available excluding test).
    """
    if num_train + num_test > total_episodes:
        raise ValueError(
            f"num_train ({num_train}) + num_test ({num_test}) > total_episodes ({total_episodes})"
        )

    # Calculate maximum number of folds
    max_folds = total_episodes // num_test
    if num_folds is None:
        num_folds = max_folds
    else:
        num_folds = min(num_folds, max_folds)

    folds = []
    for fold_idx in range(num_folds):
        # Test window starts from the end and moves backward
        test_end = total_episodes - 1 - (fold_idx * num_test)
        test_start = test_end - num_test + 1

        if test_start < 0:
            break

        # Test spec is simple range
        test_spec = f"{test_start}-{test_end}"

        # Get all non-test episodes
        all_train_episodes = [i for i in range(total_episodes) if i < test_start or i > test_end]

        # Limit to num_train episodes (take first num_train)
        selected_train_episodes = all_train_episodes[:num_train]

        if not selected_train_episodes:
            # Edge case: no training episodes available
            continue

        train_spec = episodes_to_spec(selected_train_episodes)
        folds.append((train_spec, test_spec, fold_idx))

    return folds


def find_visit_file(visits_dir: Path, algorithm_code: str) -> Optional[Path]:
    """Find visit file matching the algorithm in the visits directory."""
    for visit_path in visits_dir.glob("visits-feedback-*.pkl"):
        if f"-{algorithm_code}-" in visit_path.name:
            return visit_path
    return None


def extract_algorithm_from_filename(filename: str) -> Optional[str]:
    """Extract algorithm code (dsn/rand) from feedback filename."""
    match = FEEDBACK_ALGO_PATTERN.search(filename)
    if match:
        return match.group(1)
    return None


def dispatch_make(
    experiments_dir: Path,
    visit_rel: str,
    feedback_rel: str,
    results_rel: str,
    algorithm: str,
    lambda_value: str,
    dry_run: bool,
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
        f"FEEDBACK_LAMBDA={lambda_value}",
    ]
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

    feedback_base_dir = llm_dir / args.feedback_dir
    if not feedback_base_dir.is_dir():
        raise FileNotFoundError(f"Feedback directory not found: {feedback_base_dir}")

    # Parse filter options
    requested_lambdas = set(split_tokens(args.lambda_values))
    requested_themes = set(t.lower() for t in split_tokens(args.themes))
    requested_algorithms = set(a.lower() for a in split_tokens(args.algorithms))

    # Find all lambda directories
    lambda_dirs = sorted(feedback_base_dir.glob("lambda*"))
    if not lambda_dirs:
        raise ValueError(f"No lambda directories found in {feedback_base_dir}")

    # Prepare overrides
    num_rounds_override = str(args.feedback_num_rounds) if args.feedback_num_rounds else None

    # Check if using rotating folds or legacy mode
    use_rotating_folds = args.num_train_episodes is not None and args.num_test_episodes is not None

    # Legacy overrides (only used if not using rotating folds)
    benchmark_override = str(args.num_benchmark_episodes) if args.num_benchmark_episodes else None
    legacy_benchmark_spec = args.benchmark_episodes.strip() or None
    legacy_estimation_spec = args.estimation_episodes.strip() or None

    for lambda_dir in lambda_dirs:
        # Extract lambda value from directory name
        match = LAMBDA_DIR_PATTERN.match(lambda_dir.name)
        if not match:
            print(f"Warning: Skipping directory with unexpected name: {lambda_dir.name}")
            continue

        lambda_value = match.group(1)

        # Filter by lambda if specified
        if requested_lambdas and lambda_value not in requested_lambdas:
            continue

        feedback_dir = lambda_dir / "feedback"
        visits_dir = lambda_dir / "visits"

        if not feedback_dir.is_dir():
            print(f"Warning: No feedback directory in {lambda_dir}")
            continue
        if not visits_dir.is_dir():
            print(f"Warning: No visits directory in {lambda_dir}")
            continue

        # Process each feedback file
        for feedback_path in sorted(feedback_dir.glob("*.jsonl")):
            # Extract theme from filename (first part before _gpt)
            theme = feedback_path.stem.split("_")[0].lower()

            # Filter by theme if specified
            if requested_themes and theme not in requested_themes:
                continue

            # Extract algorithm from filename
            algo_code = extract_algorithm_from_filename(feedback_path.name)
            if not algo_code:
                print(f"Warning: Could not extract algorithm from {feedback_path.name}")
                continue

            algorithm = VISIT_ALGO_MAP.get(algo_code)
            if not algorithm:
                print(f"Warning: Unknown algorithm code: {algo_code}")
                continue

            # Filter by algorithm if specified
            if requested_algorithms and algorithm not in requested_algorithms:
                continue

            # Find corresponding visit file
            visit_path = find_visit_file(visits_dir, algo_code)
            if not visit_path:
                print(f"Warning: No visit file found for {algo_code} in {visits_dir}")
                continue

            # Convert JSONL to expected JSON format
            converted_data = convert_jsonl_to_json(feedback_path, lambda_value)

            # Determine folds to run
            if use_rotating_folds:
                # Get total episodes from preferences
                total_episodes = get_max_episode_from_preferences(converted_data["preferences"]) + 1

                if total_episodes <= 0:
                    print(f"Warning: Could not determine episode count from {feedback_path.name}")
                    continue

                folds = generate_rotating_folds(
                    total_episodes,
                    args.num_train_episodes,
                    args.num_test_episodes,
                    args.num_folds,
                )

                if not folds:
                    print(f"Warning: No valid folds generated for {feedback_path.name}")
                    continue
            else:
                # Legacy mode: single run with explicit or default specs
                folds = [(legacy_estimation_spec, legacy_benchmark_spec, None)]

            # Run each fold
            for train_spec, test_spec, fold_idx in folds:
                # Write converted data to a temp file
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix=f"{feedback_path.stem}_converted_",
                    dir=llm_dir,
                    delete=False,
                ) as tmp_file:
                    json.dump(converted_data, tmp_file, indent=2)
                    converted_path = Path(tmp_file.name)

                try:
                    # Calculate relative paths
                    visit_rel = visit_path.relative_to(llm_dir)
                    feedback_rel = converted_path.relative_to(llm_dir)

                    # Build results directory path
                    if fold_idx is not None:
                        results_rel = Path(args.results_root) / f"lambda{lambda_value}" / theme / algorithm / f"fold{fold_idx}"
                    else:
                        results_rel = Path(args.results_root) / f"lambda{lambda_value}" / theme / algorithm

                    dispatch_make(
                        experiments_dir,
                        str(visit_rel),
                        str(feedback_rel),
                        str(results_rel),
                        algorithm,
                        lambda_value,
                        args.dry_run,
                        num_rounds_override=num_rounds_override,
                        benchmark_override=benchmark_override if not use_rotating_folds else None,
                        benchmark_spec_override=test_spec,
                        estimation_spec_override=train_spec,
                    )
                finally:
                    # Clean up temp file if not dry run
                    if not args.dry_run and converted_path.exists():
                        converted_path.unlink()


if __name__ == "__main__":
    main()
