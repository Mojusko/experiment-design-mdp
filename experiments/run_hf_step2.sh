#!/usr/bin/env bash
set -euo pipefail

# Step 2 of the human feedback pipeline: generate inspection images
# from stored visits (both design and random) for ep50 and ep80.
#
# Usage:
#   ./run_hf_step2.sh --results-dir <feedback-comparison-YYYY-MM-DD-HH-MM|PATH> [--dry-run]
#   ./run_hf_step2.sh --user <eth_user> --remote-experiments-dir <remote_dir> [--dry-run] [--sleep-seconds N]
#
# Notes:
# - Run from the repository root or any directory; script adjusts paths.
# - Expects the given name to exist under experiments/llm/results/.
# - Uses the Make target `llm-inspect-visits`, which relies on config_inspect.yaml
#   and VisitsImageSaver. Output goes to additional_tests/inspect_visits-.../
#   under the corresponding results directory, one subdir per visits file
#   (separate folders for ep50 and ep80 automatically).
# - REPEATS_LLM in experiments/Makefile controls split across seeds; set to 36.

HERE=$(cd "$(dirname "$0")" && pwd)         # .../experiments
ROOT=$(cd "$HERE/.." && pwd)                  # repo root
LLM_DIR="$HERE/llm"                            # .../experiments/llm
RESULTS_BASE="$LLM_DIR/results"                # .../experiments/llm/results

RESULTS_DIR_NAME=""
DRY_RUN=false
USER_ARG=""
REMOTE_DIR=""
SLEEP_SECONDS=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR_NAME="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --user)
      USER_ARG="$2"; shift 2 ;;
    --remote-experiments-dir)
      REMOTE_DIR="$2"; shift 2 ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --results-dir <feedback-comparison-YYYY-MM-DD-HH-MM> [--dry-run]"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$RESULTS_DIR_NAME" ]]; then
  echo "Error: --results-dir is required (e.g., feedback-comparison-2025-09-10-19-03)" >&2
  exit 1
fi

# Normalize results dir: accept either a bare folder name or a full/relative path
ABS_RESULTS_DIR=""
if [[ -d "$RESULTS_DIR_NAME" ]]; then
  ABS_RESULTS_DIR="$(cd "$RESULTS_DIR_NAME" && pwd)"
elif [[ -d "$ROOT/$RESULTS_DIR_NAME" ]]; then
  ABS_RESULTS_DIR="$(cd "$ROOT/$RESULTS_DIR_NAME" && pwd)"
elif [[ -d "$RESULTS_BASE/$RESULTS_DIR_NAME" ]]; then
  ABS_RESULTS_DIR="$RESULTS_BASE/$RESULTS_DIR_NAME"
else
  echo "Error: Directory not found: $RESULTS_DIR_NAME (tried also $ROOT/$RESULTS_DIR_NAME and $RESULTS_BASE/$RESULTS_DIR_NAME)" >&2
  exit 1
fi

# Strictly require exactly four visits files: (dsn|rand) x (50|80)
find_one() {
  local alg="$1"; local ep="$2"; local out
  mapfile -t out < <(find "$ABS_RESULTS_DIR" -maxdepth 1 -type f -name "visits-feedback-${alg}-mult-ep${ep}-*.pkl" | sort)
  if [[ ${#out[@]} -ne 1 ]]; then
    echo "Error: Expected exactly 1 file for alg='${alg}', ep='${ep}', but found ${#out[@]}." >&2
    echo "Searched pattern: $ABS_RESULTS_DIR/visits-feedback-${alg}-mult-ep${ep}-*.pkl" >&2
    printf '  Found: %s\n' "${out[@]}" >&2 || true
    exit 1
  fi
  echo "${out[0]}"
}

DSN50=$(find_one dsn 50)
DSN80=$(find_one dsn 80)
RAN50=$(find_one rand 50)
RAN80=$(find_one rand 80)

VISITS_FILES=("$DSN50" "$DSN80" "$RAN50" "$RAN80")

echo "Validated expected visits files:"
printf '  %s\n' "${VISITS_FILES[@]}"

join_lines() {
  awk 'BEGIN{ORS=""} { if (sub(/\\$/,"")) printf "%s ", $0; else print $0 "\n" }'
}

run_inspect_local() {
  local visits_abs="$1"
  # Convert absolute path under llm/ to a path relative to experiments/llm
  # Makefile's llm-inspect-visits runs 'cd llm' before invoking Python,
  # so VISITS_PATH must be relative to experiments/llm.
  local visits_rel
  visits_rel=${visits_abs#"$LLM_DIR/"}   # strip leading '.../experiments/llm/'
  visits_rel="${visits_rel#llm/}"        # fallback if path contained an extra 'llm/'
  # Ensure we pass 'results/...', not an absolute path
  if [[ "${visits_rel}" != results/* ]]; then
    echo "Internal error: expected visits_rel to start with 'results/': $visits_rel" >&2
    return 2
  fi

  local cmd=( make -C "$ROOT/experiments" llm-inspect-visits VISITS_PATH="$visits_rel" )
  if $DRY_RUN; then
    echo "DRY RUN: ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

if [[ -n "$USER_ARG" && -n "$REMOTE_DIR" ]]; then
  SERVER="${USER_ARG}@euler.ethz.ch"
  PRECMD="cd ${REMOTE_DIR} && conda activate doexpy"
  echo
  echo "Submitting inspection jobs remotely via submit_euler..."
  for vf in "${VISITS_FILES[@]}"; do
    # Build relative path for VISITS_PATH (relative to experiments/llm)
    visits_rel=${vf#"$LLM_DIR/"}
    visits_rel="${visits_rel#llm/}"
    if [[ "${visits_rel}" != results/* ]]; then
      echo "Internal error: expected visits_rel to start with 'results/': $visits_rel" >&2
      continue
    fi
    # Generate job commands with make --dry-run, then submit
    if ! cmds=$(make -C "$ROOT/experiments" llm-inspect-visits VISITS_PATH="$visits_rel" --dry-run | join_lines); then
      echo "[Error] make --dry-run failed for $visits_rel" >&2
      exit 1
    fi
    if $DRY_RUN; then
      echo "--- DRY RUN (remote) for $visits_rel ---"
      echo "$cmds"
      echo "--- END ---"
    else
      if [[ -n "$cmds" ]]; then
        echo "$cmds" | "$HERE/submit_euler" --server "$SERVER" --precommand "$PRECMD"
        echo "[Info] Sleeping ${SLEEP_SECONDS}s before next batch..."
        sleep "$SLEEP_SECONDS"
      else
        echo "[Warn] No commands produced for $visits_rel" >&2
      fi
    fi
  done
else
  echo
  echo "Running inspection jobs locally via make (llm-inspect-visits)..."
  for vf in "${VISITS_FILES[@]}"; do
    run_inspect_local "$vf"
  done
fi

echo "Done. Outputs will be under:"
echo "  $ABS_RESULTS_DIR/additional_tests/inspect_visits-<id>-<timestamp>/visit_images_inspect/"
