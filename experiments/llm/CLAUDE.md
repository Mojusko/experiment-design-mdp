# Claude Code Guide for LLM Experiments

This guide provides complete documentation for AI assistants working in this codebase, covering workflow, conventions, and Hydra configuration.

---

## Project Overview: ED-PBRL

This codebase implements **ED-PBRL (Experimental Design for Preference-Based Reinforcement Learning)** from the paper:

> **Efficient Personalization of Generative Models via Optimal Experimental Design**
> Guy Schacht, Ziyad Sheebaelhamd, Riccardo De Santi, Mojmír Mutný, and Andreas Krause (2025)

### Core Idea

The goal is to efficiently learn a user's preferences for personalized image generation. Instead of randomly querying the user, ED-PBRL uses **Optimal Experimental Design (OED)** to select maximally informative preference queries, reducing the number of interactions needed.

### MDP Formulation

The problem is modeled as a **Markov Decision Process** where:
- **State s ∈ {0, 1, ..., H}**: Current timestep (position in prompt construction)
- **Action a ∈ vocabulary**: Token to append to the prompt
- **Horizon H**: Number of tokens per prompt (typically H=6)
- **Trajectory**: A sequence of tokens forming a complete prompt

### Reward Model

The user's latent preferences are modeled as a linear function over CLIP embeddings:
```
r(s) = θ*ᵀ φ(s)
```
Where:
- `θ*` is the unknown user preference vector
- `φ(s)` is the CLIP (ViT-L/14) embedding of the generated image

### Preference Model (Multinomial Logit)

Given K=4 options, the probability of preferring option i follows a softmax:
```
P(i | images) = exp(r(image_i)) / Σⱼ exp(r(image_j))
```

### Algorithms

| Code Name | Paper Name | Description |
|-----------|------------|-------------|
| `design` | **ED-PBRL** | Information-maximizing query selection using V-design scalarization and Frank-Wolfe/Convex-RL optimization |
| `random` | Baseline | Uniform random policy selection |

### Key Parameters

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| `K` | K | 4 | Number of policies/options per query |
| `horizon` | H | 6 | Tokens per prompt |
| `episodes` | T | 50-80 | Number of preference queries |
| `lambda` | λ | 0.01-100 | Regularization strength |

### Evaluation Metrics

- **Cosine Error**: `1 - cos(θ*, θ̂)` — measures direction alignment
- **Held-out Accuracy**: Preference prediction on unseen queries
- **Human Benchmark Accuracy**: Agreement with human preferences

### Technical Stack

- **Image Encoder**: CLIP ViT-L/14 (or SigLIP2)
- **Image Generator**: Stable Diffusion 1.4
- **Optimization**: Frank-Wolfe with Convex-RL for policy computation

### Vocabulary-Free Extension (REINFORCE)

The main paper uses a **fixed vocabulary** (~2500 words). ICLR reviewers raised scalability concerns:
- "State visitation measures very data hungry for high-dimensional state spaces"
- "Policy extraction computationally intensive for LLMs"

The `reinforce_word_level.py` script addresses this by replacing the fixed vocabulary with an LLM (MagicPrompt/GPT-2) that can generate **any prompt**:

| Fixed Vocabulary | Vocabulary-Free |
|------------------|-----------------|
| d_q ∈ ℝ^{2500} explicit | p_θ(prompt) via LLM |
| Frank-Wolfe (convex) | REINFORCE (policy gradient) |
| Pre-computed Φ matrix | On-the-fly CLIP embeddings |
| Convex-RL policy extraction | Direct LLM fine-tuning |

See `EXTENDED_CONTEXT_AND_CHALLENGE.md` and `PROPOSED_SOLUTION.md` for details.

---

## Core Principles

1. **KISS**: Keep it simple. Less code is better. Do one thing well.
2. **Fail Fast**: Crash with meaningful errors rather than over-handling edge cases.
3. **Modular Components**: Clear interfaces, defined goals.
4. **No Hacks**: Fix infrastructure, don't add workarounds.
5. **Documentation Matters**: Shared codebase - prioritize readability.

---

## Project Structure

```
experiments/llm/
├── conf/               # Hydra configurations
│   ├── config.yaml            # Standard full run
│   ├── config_inspect.yaml    # Inspection mode
│   ├── config_human_feedback.yaml  # Human feedback training
│   ├── config_inference.yaml  # Generic test-only
│   ├── debug.yaml             # Debug mode overrides
│   ├── embedder/      # CLIP, SigLIP2 configs
│   ├── feedback/      # Multinomial, numerical configs
│   └── experiment/    # Experiment-specific configs
├── components/         # Core modular components
│   ├── embedder.py
│   ├── feedback.py
│   ├── solver.py
│   ├── saver.py
│   ├── tester.py
│   └── ...
├── run_exp.py         # Main experiment runner
├── experiment.py      # Experiment orchestration
├── env.py             # Environment setup
└── Makefile           # Make targets for experiments
```

---

## Glossary

### Paper Terminology
- **θ* (theta-star)**: The true (unknown) user preference vector
- **θ̂ (theta-hat)**: The estimated preference vector learned from feedback
- **φ(s)**: CLIP embedding of the image generated from state/prompt s
- **V-design**: Scalarization method for experimental design optimization
- **Fisher Information**: Measures informativeness of a query for parameter estimation

### Codebase Terminology
- **visits**: Trajectory data collected during exploration (saved as `visits-*.pkl`). Each visit contains prompts, images, embeddings, and (optionally) preferences
- **episode**: One round of data collection with H timesteps. At each timestep, the user sees K=4 images and picks their favorite. So 1 episode = H preference decisions (e.g., 50 episodes × 6 timesteps = 300 total comparisons)
- **timestep**: Position within prompt construction (0 to H-1)
- **inspect**: Mode that generates grid images from visits for questionnaires
- **human feedback**: JSON/JSONL with participant choices over grid images
- **estimator**: Learned preference model weights (`estimator.pt`)
- **scorer_model**: Ground-truth preference model for synthetic experiments (e.g., `"ancient"`, `"forest"`)

---

## Architecture

### Components (see `components/`)

- `Embedder`: Computes φ(s) — CLIP ViT-L/14 or SigLIP2 image embeddings
- `Feedback`: Preference model — `MultinomialFeedback` (softmax over K options)
- `Solver`: Policy computation — implements the ED-PBRL algorithm (`design`) or random baseline (`random`)
- `Saver`: Result persistence — VisitsSaver, MetricsSaver, LearnedEstimatorSaver, ImageGenerationSaver
- `Tester`: Evaluation — HumanFeedbackBenchmarkTester (held-out accuracy), ImageGenerationTester (generates final images)

### High-Level Flow

1. **`run_exp.py`** (entry point):
   - Loads `conf/config.yaml` as the base config
   - Merges experiment config if `experiment=some_experiment` is specified
   - Debug mode: `--config-name=debug` merges `debug.yaml` on top

2. **`experiment.py`** (`LLMExperiment`):
   - Loads data and sets up the environment
   - Instantiates components (feedback, solver, tester, saver)
   - Runs exploration/optimization loop, testing, and saves results

3. **Results**: Saved under `results/` directory

---

## Operation Modes

### 1. Full Experiment (explore → train with GT → test → save)
- **Command**: `cd experiments/llm && python run_exp.py` (or via Makefile)
- **Config**: `conf/config.yaml` + `experiment/feedback_comparison.yaml` + `feedback/{multinomial|numerical}.yaml`
- **Purpose**: Synthetic experiments with known θ* (e.g., scorer_model="ancient"). Compares ED-PBRL vs random baseline by measuring cosine error to ground truth

### 2. Explore-Only (collect visits/images only)
- **Trigger**: `explore_only=true` (e.g., via `conf/debug.yaml`)
- **Savers allowed**: `VisitsSaver`, `VisitsImageSaver`, `ConfSaver`

### 3. Test-Only (generic inference)
- **Config**: `conf/config_inference.yaml`
- **Modes**:
  - `estimator_path` only → Load estimator and run testers/savers
  - `visits_path` only → Train from visits using GT labels, run testers
  - `estimator_path + feedback_path` → Load estimator and use feedback data

### 4. Inspect Visits (generate grid images)
- **Target**: `make llm-inspect-visits VISITS_PATH=results/.../visits-....pkl`
- **Config**: `conf/config_inspect.yaml`
- **Output**: `visit_images_inspect/*.png` named `alg-<design|random>_episode_XXX_timestep_YY.png`

### 5. Human Feedback (train from JSON preferences)
- **Target**: `make llm-human-feedback VISITS_PATH=... FEEDBACK_PATH=...`
- **Config**: `conf/config_human_feedback.yaml`
  - `experiment.scorer_model: null` (no GT scorer — θ* is unknown)
  - Learns θ̂ from real human/GPT preferences
  - Testers: `HumanFeedbackBenchmarkTester` (held-out accuracy), `ImageGenerationTester`
  - Savers: `LearnedEstimatorSaver`, `MetricsSaver`, `ImageGenerationSaver`, `ConfSaver`

### 6. Re-Evaluate Stored Visits
- **Target**: `make llm-re-evaluate EXP_DIR=results/feedback-comparison-...`
- **Purpose**: Batch re-testing on all `visits*.pkl` in a previous experiment folder

---

## Make Targets (run from `experiments/` directory)

```bash
# Full experiment with multinomial feedback
make llm-feedback-comparison-mul

# Generate inspection images from visits
make llm-inspect-visits VISITS_PATH=results/.../visits-....pkl

# Train from human feedback
make llm-human-feedback VISITS_PATH=... FEEDBACK_PATH=...

# Test-only mode
make llm-test-only VISITS_PATH=...
make llm-test-only ESTIMATOR_PATH=...

# Re-evaluate stored visits
make llm-re-evaluate EXP_DIR=results/...

# Post-rebuttal GPT feedback batch processing (with rotating folds)
make llm-human-feedback-post-rebuttal \
    NUM_TRAIN_EPISODES=50 NUM_TEST_EPISODES=10 \
    LAMBDA_VALUES="10" THEMES="ancient" \
    NUM_FOLDS=3
```

**Notes**:
- Algorithm and feedback are inferred from `VISITS_PATH` (e.g., `-dsn-` → design, `-rand-` → random, `-mult-` → multinomial). Override with `ALGORITHM=...` and `FEEDBACK=...`
- Custom results dir: `RESULTS_DIR=...`
- Extra Hydra overrides: `EXTRA='feedback.lambda=0.01'`

---

## Hydra Configuration

Configs are composable. Override with:
- **CLI**: `python run_exp.py algorithm=random feedback=multinomial`
- **Make**: `make llm-... EXTRA='feedback.lambda=0.01'`

### Key Config Groups
- `algorithm`: `design` (ED-PBRL), `random` (baseline)
- `feedback`: `multinomial` (K-way softmax)
- `embedder`: `clip` (ViT-L/14), `siglip2`
- `experiment`: `feedback_comparison`, `additivity`, `lambda_finding`, etc.

### Debug Mode
```bash
python run_exp.py --config-name=debug
python run_exp.py --config-name=debug experiment=feedback_comparison
```

### Defining New Experiments

1. Create config file under `conf/experiment/`, e.g. `new_experiment.yaml`:
   ```yaml
   name: "new_experiment"
   episodes: 50
   horizon: 5
   ```

2. Run with `experiment=new_experiment`:
   ```bash
   python run_exp.py experiment=new_experiment
   ```

3. Optionally add a Makefile target

---

## File Naming Conventions

**Visits files**:
```
visits-<prefix>-<alg>-<feed>-ep<episodes>-<seed>.pkl
```
- Algorithm: `dsn` (design = ED-PBRL), `rand` (random baseline)
- Feedback: `mult` (multinomial logit)

**Inspection images** (required format for HTML tooling):
```
alg-<algorithm>_episode_XXX_timestep_YY.png
```

---

## Human Feedback Pipeline

### Step 1 — Collect visits
```bash
make llm-feedback-comparison-mul
```
Produces: `results/feedback-comparison-<ts>/visits-feedback-{dsn|rand}-mult-ep<EP>-<SEED>.pkl`

**Options**:
- Seeds: `REPEATS_LLM` (default 20)
- Episodes: `EPISODES_LLM_FEEDBACK_COMPARISON` (default `50 80`)
- Lambda override: `EXTRA='feedback.lambda=0.01'`

### Step 2 — Generate images from visits
```bash
make llm-inspect-visits VISITS_PATH=llm/results/feedback-comparison-.../visits-feedback-dsn-mult-ep30-1.pkl
```
Output: `additional_tests/inspect_visits-.../visit_images_inspect/`

### Step 3 — Prepare questionnaire HTML
```bash
# Sync images
rsync -av <path-to>/visit_images_inspect/ experiments/llm/experiment_html/images/

# Generate imageList.js
python experiments/llm/generate_imagelist.py \
    -d experiments/llm/experiment_html/images \
    -o experiments/llm/experiment_html/js/imageList.js

# Serve locally (optional)
python -m http.server -d experiments/llm/experiment_html 8000
```

### Step 4 — Collect feedback JSON
Export via `prompt.html`. Schema:
```json
{
  "style_description": "...",
  "preferences": [
    {"filename":"images/alg-design_episode_000_timestep_01.png","algorithm":"design","episode":0,"timestep":1,"preference":2}
  ],
  "benchmark_episode_keys": ["design-12","random-7"],
  "metadata": {"lambda": 0.01, "responses_recorded": 600}
}
```

### Step 5 — Train and evaluate
```bash
make -C experiments llm-human-feedback \
    VISITS_PATH=results/.../visits-feedback-dsn-mult-ep30-1.pkl \
    FEEDBACK_PATH=/path/to/feedback.json
```
Output: `additional_tests/run_human_feedback-...-<ts>/` with `estimator.pt`, `metrics.json`, generated images

---

## Post-Rebuttal GPT Feedback Pipeline

For batch processing GPT feedback experiments with rotating cross-validation folds.

### Directory Structure
```
human_feedbacks_post_rebuttal/
├── lambda0.1/
│   ├── visits/     # PKL visit files
│   └── feedback/   # JSONL feedback files
├── lambda1/
├── lambda10/
└── lambda100/
```

### JSONL Format (line-by-line)
```json
{"image": "alg-design_episode_000_timestep_01.png", "model": "gpt-4.1-mini", "preference": 3}
```

### Usage
```bash
make llm-human-feedback-post-rebuttal \
    NUM_TRAIN_EPISODES=50 \
    NUM_TEST_EPISODES=10 \
    NUM_FOLDS=3 \
    LAMBDA_VALUES="10" \
    THEMES="ancient forest" \
    ALGORITHMS="design random"
```

### Rotating Folds (60 total episodes, 50 train, 10 test)
- Fold 0: test=50-59, train=0-49
- Fold 1: test=40-49, train=0-39,50-59
- Fold 2: test=30-39, train=0-29,40-59

### Output Structure
```
results/human-feedback-post-rebuttal/<lambda>/<theme>/<algorithm>/fold<N>/
├── estimator.pt
├── metrics.json
└── config_resolved.yaml
```

Script: `run_human_feedback_batch_post_rebuttal.py`

---

## Test-Only Cheatsheet

```bash
# Estimate from visits with GT (debugging)
make llm-test-only VISITS_PATH=results/.../visits-....pkl

# Load estimator and test
make llm-test-only ESTIMATOR_PATH=results/.../estimator.pt

# Load estimator + feedback JSON
make llm-test-only ESTIMATOR_PATH=... FEEDBACK_PATH=...
```

---

## Submitting Jobs to Euler (sbatch)

```bash
# Basic submission
make <target> --dry-run | ./submit_euler \
    --server=username@euler.ethz.ch \
    --precommand 'cd experiments && conda activate doexpy'

# Handle line continuations
make <target> --dry-run | \
    awk 'BEGIN{ORS=""} { if (sub(/\\$/,"")) printf "%s ", $0; else print $0 "\n" }' | \
    ./submit_euler --server=... --precommand 'cd experiments && conda activate doexpy'
```

---

## GPU Environment

- CUDA 11.8 (see `environment.yml`)
- Conda env: `doexpy-gpu` on remote, `doexpy-local` on local
- Default embedder: CLIP (override with `embedder=siglip2`)

### GCP 4-GPU Instance (l4-4gpu-instance)

**SSH Connection:**
```bash
ssh -i ~/atom/l4gpu-key l4gpuuser@<EXTERNAL_IP>
```

**Running experiments:**
```bash
ssh -i ~/atom/l4gpu-key l4gpuuser@<EXTERNAL_IP> \
  "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
   conda activate doexpy-gpu && \
   cd /home/ubuntu/experiment-design-mdp/experiments/llm && \
   python -u reinforce_word_level.py"
```

**Notes:**
- Key: `~/atom/l4gpu-key` (user: `l4gpuuser`)
- Conda is installed under `/home/ubuntu/miniconda3`
- Repo is at `/home/ubuntu/experiment-design-mdp`
- May need: `git config --global --add safe.directory /home/ubuntu/experiment-design-mdp`

---

## Debugging Tips

- Use `conf/debug.yaml`: sets `explore_only=true`, reduced episodes
- Make dry-run: `make <target> --dry-run` shows commands
- Results: `results/<experiment>-YYYY-MM-DD-HH-MM/`
- Algorithm/feedback auto-detected from VISITS_PATH (override if needed)
- Missing images in HTML? Check filename pattern: `alg-..._episode_..._timestep_....png`

---

## Common Patterns

**Running experiments**:
```bash
cd experiments/llm
python run_exp.py experiment=feedback_comparison \
    algorithm=design feedback=multinomial \
    experiment.episodes=50 seed=1
```

**Override results dir**:
```bash
make llm-... RESULTS_DIR=custom/path
```

**Multiple episodes/seeds**:
```bash
make llm-feedback-comparison-mul \
    EPISODES_LLM_FEEDBACK_COMPARISON="50 80" \
    REPEATS_LLM=20
```

**Minimal visits-only (no GT/testers)**:
```bash
cd experiments/llm && python run_exp.py \
    --config-name=config algorithm=design feedback=multinomial \
    explore_only=true +experiment.scorer_model=null tester=[] \
    'savers=[{_target_: components.saver.VisitsSaver, params: {filename: "visits.pkl"}}, {_target_: components.saver.ConfSaver, params: {filename: "config_resolved.yaml"}}]' \
    experiment.episodes=50 seed=1 \
    results_dir="results/feedback-comparison" experiment_id="ep50-1"
```

---

## Troubleshooting

- **ALGORITHM/FEEDBACK derivation**: Makefile guesses from `VISITS_PATH` (`-dsn-`, `-rand-`, `-mult-`, `-num-`). Override if misdetected.
- **RESULTS_DIR override**: Add `RESULTS_DIR=...` to place outputs in a specific folder.
- **Missing images in HTML**: Ensure files are named `alg-..._episode_..._timestep_....png` and `imageList.js` was regenerated.
- **Inspect runs**: Require visits file.
- **Human-feedback runs**: Require both visits and feedback JSON.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_exp.py` | Main entry point |
| `experiment.py` | Experiment orchestration |
| `components/feedback.py` | Feedback models |
| `components/solver.py` | Exploration algorithms |
| `components/saver.py` | Result savers |
| `components/tester.py` | Evaluation testers |
| `components/embedder.py` | Image encoders |
| `Makefile` | Job definitions and targets |
| `conf/*.yaml` | Hydra configurations |

---

## REINFORCE Word-Level Prompt Optimization

The `reinforce_word_level.py` script optimizes K=4 prompt-generating policies using REINFORCE to maximize Fisher information (D-optimal design).

### How It Works

1. **K=4 policies**: Each policy is a fine-tuned MagicPrompt (GPT-2 trained on Lexica.art prompts)
2. **Word-level embeddings**: For each policy, embed prefixes at each word position (K×H total embeddings)
3. **Fisher information**: Compute I = Σ_h I_h + λI where I_h captures embedding diversity
4. **D-optimal objective**: Maximize logdet(I)
5. **REINFORCE gradient**: ∇E[L] = E[L · Σ_w ∇log π(word_w)]

### CLI Arguments

```bash
python reinforce_word_level.py \
    --model magicprompt \      # Model: gpt2, magicprompt, distilgpt2-sd
    --samples 10 \             # M: Fisher samples for gradient estimation
    --horizon 8 \              # H: Words per prompt
    --iterations 100 \         # Number of optimization iterations
    --lambda-reg 0.5 \         # λ: Regularization strength
    --lr 1e-6 \                # Learning rate
    --baseline none \          # Baseline: none, weighted, per-word
    --design D \               # Design: D (logdet), A (-tr(I⁻¹)), V
    --optimizer sgd \          # Optimizer: sgd, adam, adamw
    --seed 123 \               # Base random seed
    --init-noise 0             # Std of noise added to init weights (0 = identical policies)
```

### Best Hyperparameters (as of 2025-01)

**Smooth trajectory (recommended for stability)**:
```bash
python reinforce_word_level.py \
    --model magicprompt --samples 10 --horizon 8 --iterations 100 \
    --lambda-reg 0.5 --lr 1e-6 --baseline none --optimizer sgd \
    --seed 123 --init-noise 0
```
- Monotone convergence in ~10 iterations
- Fluctuation: ±2.5 around optimum (most iterations near-best)
- Stable: can stop at almost any iteration after convergence

**More Fisher learning (noisier)**:
```bash
--lambda-reg 0.1 --lr 1e-6 --baseline none --optimizer sgd --init-noise 0
```
- Faster initial convergence (4 iterations to plateau)
- Fluctuation: ±10 around optimum
- Fisher term dominates more → more diversity learning

### Key Findings

| Parameter | Finding |
|-----------|---------|
| `--init-noise 0` | Start with identical policies; let optimization create diversity |
| `--baseline none` | Fastest convergence (simple L×sum(logprob)) |
| `--baseline weighted` | Slower, similar stability |
| `--lambda-reg` | Higher = smoother but less Fisher learning; 0.5 good balance |
| `--lr 1e-6` (SGD) | Good for D-optimal; 1e-7 too slow |
| `--lr 1e-5` (Adam) | Adam needs higher LR but not notably better than SGD |
| `--optimizer sgd` | Faster convergence than Adam for this problem |

### `<|endoftext|>` Handling

MagicPrompt is designed as a **prompt expander** (input: "Landscape of" → output: "Landscape of mountains, highly detailed..."). When used prefix-free, it generates from `<|endoftext|>` (GPT-2's BOS/EOS token).

**Current handling:**
1. **Block at start**: `<|endoftext|>` is masked (logit=-inf) for first 3 tokens to prevent empty generation
2. **Allow natural endings**: After 3 tokens, allow `<|endoftext|>` so prompts can end naturally
3. **Strip for CLIP**: `<|endoftext|>` is stripped before CLIP embedding (in `build_word_prefixes`)
4. **Clean display**: Stripped from printed output for readability

### GCP Instance

```bash
# SSH to l4-4gpu-instance
ssh -i ~/atom/l4gpu-key l4gpuuser@<EXTERNAL_IP>

# Run experiment
ssh -i ~/atom/l4gpu-key l4gpuuser@<IP> \
  "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
   conda activate doexpy-gpu && \
   cd /home/ubuntu/experiment-design-mdp/experiments/llm && \
   python -u reinforce_word_level.py --model magicprompt ..."

# Find current IP
~/atom/google-cloud-sdk/bin/gcloud compute instances list
```

---

## Development Guidelines

**DO**:
- Follow KISS - simple, clear solutions
- Use modular components with clear interfaces
- Write meaningful, actionable error messages
- Document complex logic
- Respect naming conventions
- Use existing components rather than creating new ones
- Test changes with debug config first
- Respect the Hydra config structure

**DON'T**:
- Add hacky workarounds (fix root cause)
- Over-handle impossible states
- Create monolithic functions
- Skip documentation
- Ignore existing patterns
- Create dependencies between unrelated components
- Bypass the Hydra config structure
