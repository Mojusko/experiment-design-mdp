#!/usr/bin/env bash
set -euo pipefail

# Run Step 1 of the human feedback pipeline for lambdas {0.1, 0.01}
# and episodes {50, 80} using Make targets and optionally submit to Euler.
#
# Usage:
#   ./run_hf_step1.sh --user <eth_user> --remote-experiments-dir <remote_dir> [--dry-run] [--sleep-seconds N]
#
# Notes:
# - Requires experiments/submit_euler and passwordless SSH to euler.
# - Uses Make targets that emit one-liners; we still normalize with awk joiner.
# - REPEATS_LLM=1 is used to keep workload light.

USER_ARG=""
REMOTE_DIR=""
DRY_RUN=false
SLEEP_SECONDS=2
DEBUG=false
LOCAL=false

# Resolve script directories for robust relative paths
HERE=$(cd "$(dirname "$0")" && pwd)          # .../experiments
LLM_DIR="$HERE/llm"                              # .../experiments/llm

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      USER_ARG="$2"; shift 2 ;;
    --remote-experiments-dir)
      REMOTE_DIR="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --debug)
      DEBUG=true; shift ;;
    --local)
      LOCAL=true; shift ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--user <eth_user>] --remote-experiments-dir <remote_dir> [--local] [--dry-run] [--debug] [--sleep-seconds N]"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if ! $DEBUG; then
  if ! $LOCAL && [[ -z "$USER_ARG" ]]; then
    echo "Error: --user is required for remote execution."
    echo "Usage: $0 [--user <eth_user>] --remote-experiments-dir <remote_dir> [--local] [--dry-run] [--debug] [--sleep-seconds N]"
    exit 1
  fi
  if [[ -z "$REMOTE_DIR" ]]; then
    echo "Error: --remote-experiments-dir is required."
    echo "Usage: $0 [--user <eth_user>] --remote-experiments-dir <remote_dir> [--local] [--dry-run] [--debug] [--sleep-seconds N]"
    exit 1
  fi
fi

SERVER="${USER_ARG}@euler.ethz.ch"
PRECMD="cd ${REMOTE_DIR} && eval \"\$(conda shell.bash hook)\" && conda activate doexpy"

# Helper to join multiline Make output into single lines
join_lines() {
  awk 'BEGIN{ORS=""} { if (sub(/\\$/,"")) printf "%s ", $0; else print $0 "\n" }'
}

# Build the two make invocations
MAKE_BASE=( make llm-feedback-comparison-mul REPEATS_LLM=1 EPISODES_LLM_FEEDBACK_COMPARISON="50 80" )

# Generate unique TIMESTAMPs per batch to avoid collisions when running
# within the same minute. Include a lambda tag for clarity.
TS_L01="$(date +%Y-%m-%d-%H-%M-%S)-l01"
TS_L001="$(date +%Y-%m-%d-%H-%M-%S)-l001"

# Visits-only overrides: do not initialize GT scorer models; disable testers; run explore_only;
# restrict savers to VisitsSaver and ConfSaver. Use ++ to force-override existing key and quote
# list-valued overrides so Hydra gets them as single tokens.
EXTRA_VISITS_ONLY="explore_only=true ++experiment.scorer_model=null 'tester=[]' 'savers=[{_target_: components.saver.VisitsSaver, params: {filename: \"visits.pkl\"}}, {_target_: components.saver.ConfSaver, params: {filename: \"config_resolved.yaml\"}}]'"

# Lambda 0.1 (default) batch
MAKE_L01=( "${MAKE_BASE[@]}" "EXTRA=${EXTRA_VISITS_ONLY}" )

# Lambda 0.01 batch
EXTRA_L001="${EXTRA_VISITS_ONLY} feedback.lambda=0.01"
MAKE_L001=( "${MAKE_BASE[@]}" "EXTRA=${EXTRA_L001}" )

run_block() {
  local ts="$1"; shift
  local -a MAKE_CMD=("$@")
  echo "[Info] Generating jobs: TIMESTAMP=${ts} ${MAKE_CMD[*]} --dry-run"
  local cmds
  if ! cmds=$(TIMESTAMP="$ts" "${MAKE_CMD[@]}" --dry-run | join_lines); then
    echo "[Error] make --dry-run failed." >&2
    exit 1
  fi

  if $DRY_RUN; then
    echo "--- DRY RUN: Commands that would be submitted ---"
    echo "$cmds"
    echo "--- END DRY RUN BLOCK ---"
  else
    if [[ -z "$cmds" ]]; then
      echo "[Warn] No commands produced; skipping submission." >&2
    else
      if $LOCAL; then
        echo "$cmds" | ./submit_euler --local --precommand "$PRECMD"
      else
        echo "$cmds" | ./submit_euler --server "$SERVER" --precommand "$PRECMD"
      fi
    fi
  fi
}

if $DEBUG; then
  echo "[Debug] Running local quick sanity checks using config 'debug'."
  echo "[Debug] Ignoring remote submission and --dry-run."
  # Build two local runs (design and random) with minimal episodes; visits-only savers
  pushd "$LLM_DIR" >/dev/null
  set -x
  python run_exp.py --config-name=debug \
    experiment=feedback_comparison \
    algorithm=design \
    feedback=multinomial \
    explore_only=true \
    +experiment.scorer_model=null \
    'tester=[]' \
    'savers=[{_target_: components.saver.VisitsSaver, params: {filename: "visits.pkl"}}, {_target_: components.saver.ConfSaver, params: {filename: "config_resolved.yaml"}}]' \
    experiment.episodes=10 \
    seed=1 \
    results_dir="results/feedback-comparison" \
    experiment_id="ep10-1"

  python run_exp.py --config-name=debug \
    experiment=feedback_comparison \
    algorithm=random \
    feedback=multinomial \
    explore_only=true \
    +experiment.scorer_model=null \
    'tester=[]' \
    'savers=[{_target_: components.saver.VisitsSaver, params: {filename: "visits.pkl"}}, {_target_: components.saver.ConfSaver, params: {filename: "config_resolved.yaml"}}]' \
    experiment.episodes=10 \
    seed=1 \
    results_dir="results/feedback-comparison" \
    experiment_id="ep10-1"
  set +x
  popd >/dev/null
  echo "[Debug] Done."
else
# Lambda 0.1 (default) for episodes 50 and 80
run_block "$TS_L01" "${MAKE_L01[@]}"

  if ! $DRY_RUN; then
    echo "[Info] Sleeping ${SLEEP_SECONDS}s before next batch..."
    sleep "$SLEEP_SECONDS"
  fi

# Lambda 0.01 for episodes 50 and 80
run_block "$TS_L001" "${MAKE_L001[@]}"

  echo "[Info] Done."
fi
