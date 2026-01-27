# Codex Guide: LLM Experiments (ED-PBRL)

This file adapts the `CLAUDE*.md`, challenge, and solution docs into concise, Codex-oriented instructions for working in `experiments/llm`.

## Scope And Canonical Sources

- Primary reference for workflow and project structure: `experiments/llm/CLAUDE.md`.
- Current best REINFORCE config and practical notes: `experiments/llm/CLAUDE.local.md`.
- Problem framing and vocabulary-free motivation: `experiments/llm/EXTENDED_CONTEXT_AND_CHALLENGE.md`.
- LLM-based design proposal and algorithm sketch: `experiments/llm/PROPOSED_SOLUTION.md`.
- Keep the challenge/solution docs unchanged unless explicitly asked; treat them as canonical references.
- When docs disagree on hyperparameters, prefer the most recent dated guidance. As of 2026-01-27, the `moving_avg` baseline with higher learning rate is the recommended REINFORCE setup.

## Working Directory And Entrypoints

- Run LLM experiments from `experiments/llm`.
- The main Hydra entrypoint is `experiments/llm/run_exp.py`.
- The vocabulary-free REINFORCE entrypoint is `experiments/llm/reinforce_word_level.py`.
- Batch human-feedback processing lives in `experiments/llm/run_human_feedback_batch_post_rebuttal.py`.

## Make Targets (From `experiments/Makefile`)

- Run make targets from `experiments`, or use `make -C experiments <target>` from the repo root.
- Key targets:
- `llm-feedback-comparison-mul`
- `llm-inspect-visits VISITS_PATH=results/.../visits-....pkl`
- `llm-human-feedback VISITS_PATH=... FEEDBACK_PATH=...`
- `llm-test-only VISITS_PATH=...`
- `llm-test-only ESTIMATOR_PATH=...`
- `llm-re-evaluate EXP_DIR=results/...`
- `llm-human-feedback-post-rebuttal NUM_TRAIN_EPISODES=50 NUM_TEST_EPISODES=10 NUM_FOLDS=3 LAMBDA_VALUES="10" THEMES="ancient forest" ALGORITHMS="design random"`

## Hydra Usage Patterns

- Standard run: `python run_exp.py`.
- Debug run: `python run_exp.py --config-name=debug`.
- Override groups and fields via CLI, for example:

```bash
python run_exp.py experiment=feedback_comparison algorithm=design feedback=multinomial experiment.episodes=50 seed=1
```

## Operation Modes (Mental Model)

- Full synthetic experiment: explore, train on ground truth, test, and save.
- Explore-only: set `explore_only=true` and limit savers to visits and config.
- Test-only inference: use `conf/config_inference.yaml` with `estimator_path`, `visits_path`, and optionally `feedback_path`.
- Inspect visits: generate grid images from a visits PKL via `llm-inspect-visits`.
- Human feedback: train from JSON or JSONL preference data via `llm-human-feedback` or the post-rebuttal batch target.

## Key Conventions And File Patterns

- Visits naming pattern: `visits-<prefix>-<alg>-<feed>-ep<episodes>-<seed>.pkl`.
- Algorithm tags: `dsn` for design (ED-PBRL) and `rand` for random.
- Feedback tags: `mult` for multinomial.
- Inspect image naming pattern must be: `alg-<algorithm>_episode_XXX_timestep_YY.png`.
- Some tooling infers algorithm and feedback from `VISITS_PATH`. If misdetected, override explicitly with make variables or Hydra overrides.

## REINFORCE Word-Level Guidance

- The REINFORCE script replaces fixed-vocabulary Frank-Wolfe optimization with LLM policy gradients.
- The main practical pitfall is mismatch between training and evaluation objectives. Use `--sum-trajectories` to align them.
- The recommended configuration as of 2026-01-27 is:

```bash
python reinforce_word_level.py \
    --model magicprompt \
    --samples 8 \
    --horizon 8 \
    --iterations 60 \
    --lambda-reg 1 \
    --lr 1e-3 \
    --baseline moving_avg \
    --optimizer sgd \
    --seed 123 \
    --init-noise 0 \
    --joint-update \
    --sum-trajectories \
    -T 8
```

Memory expectations:
- `M=8` and `T=8` typically require A100 80GB-class GPUs.
- Smaller `M` and `T` settings are needed for L4-class GPUs.

## Design Intent And Constraints

- Goal: remove the fixed vocabulary constraint and let the LLM explore open-ended prompts.
- Do not add a projection back to a fixed vocabulary.
- Prefer on-the-fly embedding computation over large precomputed matrices when working in the LLM setting.
- Keep the solution modular. Use existing components and Hydra structure where possible.

## Engineering Guidelines

- Keep solutions simple and explicit.
- Fail fast with clear errors rather than adding defensive complexity.
- Avoid hacks and one-off workarounds. Fix root causes.
- Respect existing naming conventions and file layouts.
- When changing experiment behavior, update relevant config files under `experiments/llm/conf`.
