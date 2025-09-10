# Agent Guide: LLM Experiments and Human Feedback Pipeline

This guide summarizes how to run the LLM experiment jobs, their modes, and the end‑to‑end human feedback workflow. It is tailored for agents working in this repo and maps Make targets, configs, and expected inputs/outputs.

## Quick Glossary
- visits: Trajectory data collected during exploration (saved as `visits-*.pkl`).
- inspect: Generate grid images from visits for questionnaires.
- human feedback: JSON with participant choices over grid images and optional prompts.
- estimator: Learned preference model (`estimator.pt`).

## Key Make Targets (run from experiments/ directory)
- Full experiment (defaults): `llm-feedback-comparison{,-mul,-num}`
- Inspect visits → images: `llm-inspect-visits` (requires `VISITS_PATH`)
- Human feedback (train from JSON): `llm-human-feedback` (requires `VISITS_PATH`, `FEEDBACK_PATH`)
- Test‑only (generic): `llm-test-only` (combines estimator/visits/feedback paths)
- Re-evaluate stored visits: `llm-re-evaluate` (requires `EXP_DIR`)

Default algorithm and feedback are inferred from `VISITS_PATH` contents (e.g., `-dsn-` → design, `-rand-` → random, `-mult-` → multinomial, `-num-` → numerical). You can override with `ALGORITHM=...` and `FEEDBACK=...`.

Optional: set a custom results dir on any target with `RESULTS_DIR=...` (the Makefile toggles `override_results_dir=true`).

## Operation Modes (configs and purpose)

1) Full Experiment (explore → train with GT → test → save)
- Command: `cd experiments/llm && python run_exp.py` (or via Makefile jobs)
- Config: `conf/config.yaml` + `experiment/feedback_comparison.yaml` + `feedback/{multinomial|numerical}.yaml`
- Purpose: Baseline runs using ground‑truth scorer models for evaluation. Saves metrics, estimator, and visits.

2) Explore‑Only (only collect visits/images)
- Trigger: `explore_only=true` (e.g., via `conf/debug.yaml`)
- Savers allowed: `VisitsSaver`, `VisitsImageSaver`, `ConfSaver`.
- Purpose: Explore and save visits and optionally images, without fitting or testing.

3) Test‑Only (generic inference; selected by provided paths)
- Config: `conf/config_inference.yaml`
- Modes determined by args:
  - `estimator_path` only → Load estimator and run testers/savers.
  - `visits_path` only → Train from visits using GT labels, run testers.
  - `estimator_path + feedback_path` → Load estimator and use feedback data downstream.
  - `inspection_mode=true` → See Inspect Visits.

4) Inspect Visits (generate grid images from visits)
- Target: `make llm-inspect-visits VISITS_PATH=results/.../visits-....pkl`
- Config: `conf/config_inspect.yaml` (test_only + inspection).
- Output: `visit_images_inspect/*.png` named `alg-<design|random>_episode_XXX_timestep_YY.png`.
- Purpose: Produce questionnaire images from collected visits.

5) Human Feedback (train estimator from JSON)
- Target: `make llm-human-feedback VISITS_PATH=... FEEDBACK_PATH=...`
- Config: `conf/config_human_feedback.yaml`
  - `experiment.scorer_model: null` (no GT scorer; training purely from human feedback)
  - Testers: `HumanFeedbackBenchmarkTester`, `ImageGenerationTester (prompt_ranking_model: current_iteration_estimator)`
  - Savers: `LearnedEstimatorSaver`, `MetricsSaver`, `ImageGenerationSaver`, `ConfSaver`
- Purpose: Fit a model from human preferences and benchmark it (held‑out episodes) and generate images via the learned model.

6) Re‑Evaluate Stored Visits
- Target: `make llm-re-evaluate EXP_DIR=results/feedback-comparison-...`
- Purpose: Batch re‑testing on all `visits*.pkl` in a previous experiment folder.

## Human Feedback Pipeline (recommended sequence)

Step 1 — Collect visits (both design and random)
- Preferred multinomial runs:
  - `make llm-feedback-comparison-mul`
  - Produces e.g. `experiments/llm/results/feedback-comparison-<ts>/visits-feedback-{dsn|rand}-mult-ep<EP>-<SEED>.pkl`
- Notes:
  - Seeds count: `REPEATS_LLM` in Makefile (default 20).
  - Episodes set via `EPISODES_LLM_FEEDBACK_COMPARISON` in Makefile (now `50 80`).
  - You can pass extra Hydra overrides to Make recipes via `EXTRA`, e.g. `EXTRA='feedback.lambda=0.01'`.
  - To run a specific job, use the generated job name, e.g. `make -C experiments job-llm-feedback-dsn-mult-ep30-1`.

Step 2 — Generate images from visits (for the questionnaire)
- Example (design):
  - `make llm-inspect-visits VISITS_PATH=llm/results/feedback-comparison-YYYY-MM-DD-HH-MM/visits-feedback-dsn-mult-ep30-1.pkl`
- Output images live under an `additional_tests/inspect_visits-.../visit_images_inspect/` subdir of the visits’ results folder.

Step 3 — Prepare the questionnaire HTML
- Copy/sync generated images to the web folder:
  - `rsync -av <path-to>/visit_images_inspect/ experiments/llm/experiment_html/images/`
- Generate `imageList.js` for the questionnaire:
  - `python experiments/llm/generate_imagelist.py -d experiments/llm/experiment_html/images -o experiments/llm/experiment_html/js/imageList.js`
- Serve the static site (optional local server):
  - `python -m http.server -d experiments/llm/experiment_html 8000`
- The UI (index.html) will:
  - Randomize balanced episodes across algorithms.
  - Collect choices per image grid.
  - Store some episodes as held‑out benchmark (`benchmark_episode_keys`).
  - Prompt the user for their text prompts on `prompt.html`.

Step 4 — Collect feedback JSON
- Export via `prompt.html` (copy or download JSON). Minimal schema:
```
{
  "user_prompts": ["...", "..."],
  "preferences": [
    {"filename":"images/alg-design_episode_000_timestep_01.png","algorithm":"design","episode":0,"timestep":1,"preference":2},
    ...
  ],
  "benchmark_episode_keys": ["design-12","random-7", ...]
}
```

Step 5 — Train from human feedback and evaluate
- Command:
  - `make -C experiments llm-human-feedback VISITS_PATH=results/.../visits-feedback-dsn-mult-ep30-1.pkl FEEDBACK_PATH=/path/to/feedback.json [RESULTS_DIR=... ]`
- Config used: `conf/config_human_feedback.yaml` (no GT scorer).
- Output (under `additional_tests/run_human_feedback-...-<ts>/`):
  - `estimator.pt`, `metrics.json`, resolved config, and generated images.

## Test‑Only Cheatsheet (direct)
- Estimate from visits with GT (debugging):
  - `make llm-test-only VISITS_PATH=results/.../visits-....pkl [ALGORITHM=design|random] [FEEDBACK=multinomial|numerical] [RESULTS_DIR=...]`
- Load estimator and test:
  - `make llm-test-only ESTIMATOR_PATH=results/.../estimator.pt [RESULTS_DIR=...]`
- Load estimator + feedback JSON (for prompt/image generation flows):
  - `make llm-test-only ESTIMATOR_PATH=... FEEDBACK_PATH=... [RESULTS_DIR=...]`

## Submitting jobs to Euler (sbatch pipeline)

- From the `experiments/` directory, you can pipe one‑liner Make recipes to the helper:
  - `make <target> --dry-run | ./submit_euler --server=username@euler.ethz.ch --precommand 'cd experiments && conda activate doexpy'`

- Note on one‑liners: sbatch wrapping expects a single shell line per job. Our Make targets aim to emit single lines; if you encounter line‑continuations (`\`) in the dry‑run output, join them before piping. Example joiner:
  - `make <target> --dry-run | awk 'BEGIN{ORS=""} { if (sub(/\\$/,"")) printf "%s ", $0; else print $0 "\n" }' | ./submit_euler --server=... --precommand 'cd experiments && conda activate doexpy'`

Replace `<target>` with e.g. `llm-feedback-comparison-mul` or any of the job names (like `job-llm-feedback-dsn-mult-ep50-1`).

## Naming Conventions
- Experiment ID prefix: from `experiment.id_prefix` (e.g., `feedback`, `human-feedback`).
- Algorithm code: `dsn` (design), `rand` (random), `opt` (optim).
- Feedback code: `mult` (multinomial), `num` (numerical).
- Visits filename pattern: `visits-<prefix>-<alg>-<feed>-ep<episodes>-<seed>.pkl`
- Inspection image filename pattern (required by HTML tooling): `alg-<algorithm>_episode_XXX_timestep_YY.png`.

## Configs Reference
- `conf/config.yaml`: Standard full run (GT scorer present by default).
- `conf/config_inspect.yaml`: Inspection run (images from visits, no testers).
- `conf/config_human_feedback.yaml`: Human‑feedback training (no GT scorer), benchmarks + image generation.
- `conf/config_inference.yaml`: Generic test‑only modes.
- Feedback configs: `conf/feedback/{multinomial,numerical}.yaml`.
- Embedder configs: `conf/embedder/{clip,siglip2}.yaml` (override with `embedder=siglip2` if calling Python/Hydra directly).

## Troubleshooting
- ALGORITHM/FEEDBACK derivation: Makefile guesses from `VISITS_PATH` (`-dsn-`, `-rand-`, `-mult-`, `-num-`). Override if misdetected.
- RESULTS_DIR override: Add `RESULTS_DIR=...` to place outputs in a specific folder.
- Missing images in HTML: Ensure files are named `alg-..._episode_..._timestep_....png` and `imageList.js` was (re)generated.
- Inspect runs require visits; human‑feedback runs require both visits and a JSON file.

## FAQ / Tips

- Minimal visits‑only (skip GT/testers): simplest is to run `llm-feedback-comparison-mul` as‑is and just use the produced `visits-*.pkl`. If you truly need no GT/testers/savers, you can run a Python one‑liner with Hydra overrides to enable explore‑only and restrict savers, for example:
  - `cd experiments/llm && python run_exp.py --config-name=config algorithm=design feedback=multinomial explore_only=true +experiment.scorer_model=null tester=[] 'savers=[{_target_: components.saver.VisitsSaver, params: {filename: "visits.pkl"}}, {_target_: components.saver.ConfSaver, params: {filename: "config_resolved.yaml"}}]' experiment.episodes=50 seed=1 results_dir="results/feedback-comparison" experiment_id="ep50-1"`
  - This runs exploration only and saves visits/config. Repeat with `algorithm=random` to get both algorithms.
  - If you prefer, we can add a tiny `config_explore_visits.yaml` and/or a Make target to make this cleaner.

- Run feedback‑comparison for multiple lambdas and episodes (no Makefile changes):
  - For default lambda=0.1 and episodes 50, 80 using Make:
    - `make llm-feedback-comparison-mul EPISODES_LLM_FEEDBACK_COMPARISON="50 80"`
  - For lambda=0.01, invoke Python with an override (two runs):
    - `cd experiments/llm && python run_exp.py experiment=feedback_comparison algorithm=design feedback=multinomial feedback.lambda=0.01 experiment.episodes=50 seed=1 results_dir="results/feedback-comparison" experiment_id="ep50-1"`
    - `cd experiments/llm && python run_exp.py experiment=feedback_comparison algorithm=design feedback=multinomial feedback.lambda=0.01 experiment.episodes=80 seed=1 results_dir="results/feedback-comparison" experiment_id="ep80-1"`
  - You may generate the corresponding `random` algorithm runs by switching `algorithm=random`.
  - To send these to Euler, pipe each one‑liner to `./submit_euler` as shown above.
  - Alternative: via Make `EXTRA` passthrough variable:
    - `make llm-feedback-comparison-mul EPISODES_LLM_FEEDBACK_COMPARISON="50 80" EXTRA='feedback.lambda=0.01'`
    - Add `REPEATS_LLM=1` to reduce load.

## Helper Script for Step 1

- Script: `experiments/run_hf_step1.sh`
- Runs four jobs (lambda ∈ {0.1, 0.01} × episodes ∈ {50, 80}) using Make, and optionally submits to Euler.
- Ensures visits‑only collection: overrides `explore_only=true`, disables testers, sets `+experiment.scorer_model=null`, restricts savers to `VisitsSaver` + `ConfSaver` (no GT/scorers/testers in step 1).
- Usage:
  - `./run_hf_step1.sh --user <eth_user> --remote-experiments-dir <remote_dir> [--dry-run] [--sleep-seconds N]`
- Examples:
  - Dry run only (print commands):
    - `./run_hf_step1.sh --user your_nethz --remote-experiments-dir /cluster/home/your_nethz/experiment-design-mdp/experiments --dry-run`
  - Submit to Euler:
    - `./run_hf_step1.sh --user your_nethz --remote-experiments-dir /cluster/home/your_nethz/experiment-design-mdp/experiments`
