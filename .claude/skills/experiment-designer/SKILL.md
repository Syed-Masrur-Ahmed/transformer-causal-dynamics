---
name: experiment-designer
description: >
  Design and implement experiments for the transformer causal dynamics project.
  Use this skill whenever the user wants to create a new experiment, test different
  parameters, vary the architecture, or explore how the transformer learns stochastic
  process dynamics. Triggers on mentions of experiments, parameter sweeps, ablations,
  model comparisons, testing new configurations, new processes, or new embedding types.
  Also use when the user says things like "let's try", "what if we change", "sweep over",
  "compare models", or describes specific parameter values they want to test.
---

# Experiment Designer

You are designing an experiment for a research project studying whether transformers can learn causal dynamics from stochastic processes. The user will describe what they want to test — your job is to validate the design, create any needed helper code, and produce a runnable Jupyter notebook with relevant plots.

## Your Workflow

### 1. Understand the Experiment

Ask clarifying questions if needed. Extract:
- **What is being varied** (the independent variable): architecture params, physics params, training setup, process type, etc.
- **What is being measured** (the dependent variable): MSE, bias/variance decomposition, attention patterns, embedding properties, etc.
- **What stays fixed**: defaults from configs or user-specified values
- **Comparison baseline**: existing trained model, untrained model, analytical solution, etc.

### 2. Read the Current Codebase

Before creating anything, read the relevant source files to understand what already exists:

- **`src/model.py`** — current transformer architecture (embeddings, attention, output heads)
- **`src/mgf_dataset.py`** — current data generation and target computation
- **`src/utils.py`** — config loading, experiment ID management
- **`configs/`** — current default parameters (data.yaml, model.yaml, train.yaml, test.yaml)
- **`analysis_notebooks/`** — existing experiment notebooks (check for overlap or reusable patterns)
- **`experiments/`** — existing model checkpoints and test results

This is essential because the codebase evolves. Don't assume you know the current state — read the files to verify what functions exist, what parameters they accept, and what the current defaults are.

### 3. Validate the Experiment

Check that the proposed experiment is compatible with the codebase:

**Architecture constraints:**
- `n_head` must divide `d_model`
- `sequence_length` must not exceed `max_len` in positional encoding
- `d_output` must match the number of target coefficients
- Very large embedding dimensions may cause numerical issues — warn if concerning

**Data/physics constraints:**
- Verify that any new process or target computation is mathematically consistent
- Check that parameter ranges are physically meaningful
- Ensure derived quantities (like diffusion coefficients) maintain intended relationships

**Compatibility:**
- If the experiment requires modifying the model architecture, check that changes are backward-compatible or clearly scoped to this experiment
- If new helper functions are needed, verify they don't conflict with existing ones

If any constraint fails, tell the user and suggest a fix.

### 4. Cross-Reference Existing Work

Scan `analysis_notebooks/` and `experiments/` for overlap. If the proposed experiment is similar to existing work:
- Tell the user what already exists
- Suggest building on it rather than duplicating
- Offer to extend existing notebooks or create a new one that loads prior results for comparison

### 5. Decide Pipeline Strategy

- **Self-contained notebook**: If the experiment varies data generation, architecture, or training — generate fresh data and train inside the notebook. This is the default for sweeps and ablations.
- **Reuse existing artifacts**: If the experiment only needs new analysis of an already-trained model, load from `experiments/` and `data/`.
- **Hybrid**: Train fresh models but also load existing saved models for comparison.

### 6. Create Helper Functions (if needed)

If the experiment requires functionality not already in `src/`:
- Add new functions to the appropriate `src/` file
- Follow the existing code patterns and style in that file
- Keep functions general — parameterize rather than hardcode experiment-specific values

Examples of when new helpers are needed:
- A new stochastic process (new simulation function in `mgf_dataset.py`)
- A new embedding type (new class in `model.py`)
- A new target computation (new function in `mgf_dataset.py`)
- A new evaluation metric (could go in a new `src/metrics.py` or in the notebook)

### 7. Create the Notebook

Save to `analysis_notebooks/<descriptive_name>.ipynb`. Match the patterns and style of existing notebooks in the project.

**Required structure:**

1. **Title cell** (markdown): `# Experiment: [Descriptive Title]` with a brief description of what's being tested and why.

2. **Setup cell**: Follow the exact import and path-setup pattern used in existing notebooks. Read an existing notebook to get the current boilerplate — it includes `%config InlineBackend.figure_format = 'svg'`, path manipulation for project root, and standard imports.

3. **Config cell**: All experiment-specific parameters in one cell, clearly labeled with comments. Pull defaults from the config system (`load_full_config()`) and override only what the experiment changes.

4. **Helper functions cell**: Local functions for data generation, training, and evaluation specific to this experiment. Follow the pattern of functions like `make_data()`, `train_model()`, `eval_mse()` found in existing notebooks.

5. **Execution cells**: Run the experiment with `tqdm` progress bars for long loops.

6. **Plot cells**: Generate matplotlib plots matching the existing project style:
   - SVG format (set via `%config`)
   - Standard figure sizes: `(6, 5)` single, `(10, 4)` side-by-side, `(4*ncols, 8)` grids
   - LaTeX labels for mathematical quantities
   - Semi-log scale for quantities spanning orders of magnitude
   - `plt.tight_layout()` at the end
   - Colorbars for heatmaps/contours
   - Grid subplots with shared axes when comparing across conditions

7. **Summary cell** (markdown): Key observations from the results.

### 8. Suggest Follow-ups

After creating the notebook, briefly suggest:
- Interesting parameter variations to try next
- Existing results worth comparing against
- Whether findings warrant changes to defaults or further investigation
