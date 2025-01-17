# Hydra-Based Framework: Quick Guide

This project uses [Hydra](https://github.com/facebookresearch/hydra) to manage configuration for LLM-based experimentation. Below is a short tutorial on how to navigate the folder structure, run experiments, define new experiments, and interact with the Makefile (including a debug mode).

---

## Folder Structure

```
your_project/
├─ llm/
│  ├─ Makefile
│  ├─ run_exp.py         # Main entry point
│  ├─ experiment.py      # Defines LLMExperiment logic
│  ├─ env.py             # Setup data/environment
│  ├─ components/        # Modular experiment components
│  │  ├─ feedback.py
│  │  ├─ solver.py
│  │  ├─ tester.py
│  │  └─ saver.py
│  └─ conf/
│     ├─ config.yaml
│     ├─ debug.yaml
│     ├─ experiment/
│     │  ├─ additivity.yaml
│     │  ├─ feedback_comparison.yaml
│     │  └─ lambda_finding.yaml
│     └─ feedback/
│        ├─ multinomial.yaml
│        └─ numerical.yaml
```

### High-Level Flow

1. **`run_exp.py`** (entry point):
   - Loads `conf/config.yaml` as the base config, plus any config files under `conf/**`, and integrates any command-line overrides.
   - If you specify `--config-name=debug`, it merges `conf/debug.yaml` **on top** of `config.yaml`.

2. **`experiment.py`** (`LLMExperiment`):
   - Loads data and sets up the environment (`env.py`).
   - Instantiates the necessary components from `components/` (feedback, solver, tester, saver).
   - Runs an exploration/optimization loop, then testing, and finally saves results.

3. **Results**:
   - Saved under the `results/` directory (default or overridden via config/CLI).
   - Reproducible via fixed seeds and controlled Hydra configs.

---

## Running Experiments

### 1. Using the Makefile

Inside `llm/Makefile`, there are predefined targets for various experiments. For example:

```bash
cd llm
make llm-additivity
```

This target might internally run something like:

```bash
python run_exp.py --config-name=experiment/additivity \
  experiment=additivity \
  saver.params.path="results/additivity/alg-numerical.txt" \
  ...
```

Which means:
- Hydra loads `conf/experiment/additivity.yaml` (plus `config.yaml`).
- Additional overrides are applied (like `saver.params.path`).
- The final composed config is used to run the experiment, and results are placed in `results/`.

### 2. Running Directly via Python

You can also run an experiment directly, for example:

```bash
cd llm
python run_exp.py --config-name=experiment/feedback_comparison \
  seed=123 \
  episodes=20
```

This will:
- Load `conf/config.yaml` + `conf/experiment/feedback_comparison.yaml`.
- Override `seed` and `episodes` to 123 and 20, respectively.

### 3. Debug Mode

To activate debug settings defined in `conf/debug.yaml`, run:

```bash
cd llm
python run_exp.py --config-name=debug
```

By default, this will:
- Start with `config.yaml`.
- Then apply any overrides in `debug.yaml` (such as smaller `vocab_size`, different feedback settings, fewer prompts, etc.).
- Print a message indicating debug mode is on.

You can also combine it with experiment overrides:

```bash
python run_exp.py --config-name=debug experiment=feedback_comparison
```

This merges `debug.yaml` on top of `config.yaml`, and also sets `experiment=feedback_comparison`.

---

## Defining New Experiments

1. **Create a new config file** under `conf/experiment/`, e.g. `new_experiment.yaml`:

   ```yaml
   # conf/experiment/new_experiment.yaml
   defaults:
     - override feedback: multinomial

   # Additional experiment-specific overrides
   episodes: 50
   horizon: 5
   ```

2. **Run** it by specifying `--config-name=experiment/new_experiment`, for example:

   ```bash
   cd llm
   python run_exp.py --config-name=experiment/new_experiment
   ```

3. **Adjust** the Makefile (optional) to add a new target (e.g., `make llm-new_experiment`) if you want quick shortcuts.  

---

## Components Overview

1. **`feedback.py`**  
   - Implements various ways of generating or simulating feedback, e.g.:
     - *Numerical feedback* (continuous)
     - *Multinomial feedback* (categorical)
   - The actual config references `conf/feedback/numerical.yaml` or `conf/feedback/multinomial.yaml`.

2. **`solver.py`**  
   - Contains different exploration or optimization algorithms (greedy, random, etc.).

3. **`tester.py`**  
   - Defines how final evaluation is performed (e.g., preference tests, error metrics).

4. **`saver.py`**  
   - Handles saving results (to files, logs, etc.).
   - Configurable via Hydra overrides (e.g., `saver.params.path`).

---

## Key Points

- **Hydra Merging**: 
  - `config.yaml` is your base config. 
  - Additional files in `conf/**` are included if referenced in the `defaults:` list or via `--config-name=some_dir/some_file`.
  - You can override any config field on the command line:  
    `python run_exp.py key=value nested.key=value ...`

- **Debugging**:
  - `conf/debug.yaml` can further reduce problem sizes or change certain parameters for quick testing.
  - Activate with `--config-name=debug`.

- **Makefile Integration**:
  - The Makefile in `llm/` automates calls to `run_exp.py` with specific overrides, giving you a simple command like `make llm-additivity`.
  - Useful for repeated experiments or quick iteration on multiple machines.

- **Extend with New Components**:
  - To add a new solver, put it in `components/solver.py` and reference it from your YAML.
  - To add a new experiment, drop a new file in `conf/experiment/` and follow the pattern above.

---

**Happy experimenting!** If you have any issues or want to add functionality, feel free to contribute new config files, components, or Makefile targets. Leverage Hydra to keep your code modular, reproducible, and easy to maintain.
