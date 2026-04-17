---
name: add-experiment-docs
description: >
  Add Quarto documentation pages for experiments in the transformer causal dynamics project.
  Use this skill whenever the user wants to document an experiment, add a new page to the docs site,
  write up results, or create an experiment summary page. Triggers on phrases like "add docs for",
  "document experiment", "write up results", "add to the docs site", "create a docs page",
  "add a quarto page", or when the user describes an experiment and wants it recorded in the site.
  Also use when the user says they ran an experiment and wants to share or write it up.
---

# Add Experiment Docs

You are adding documentation for an experiment to the Quarto site in `docs/`. Your job is to create a well-structured `.qmd` page following the exact format and conventions already established in the project, then register it in the site's sidebar.

## Your Workflow

### 1. Understand What to Document

Ask the user (or infer from context) the following if not already clear:
- **Experiment name/title**: a concise descriptive title (e.g. "Experiment 3: Variable θ with Binary Outcomes")
- **What was varied**: the key independent variable(s)
- **What was measured**: metrics, plots, or evaluation outputs
- **Config parameters used**: theta, mu, sigma, dt, sequence length, epochs, batch size, learning rate, etc.
- **What images/plots will be shown**: names of the SVG files already saved (or to be saved) in `docs/images/`
- **Commands used**: the exact bash commands to generate data, train, and test

If the user says "the usual setup" or references defaults, read `configs/` to fill in the table accurately.

### 2. Read the Current Site Structure

Before writing anything:
- Read `docs/_quarto.yml` to see the current sidebar and identify the next experiment number or appropriate page name
- Skim one existing experiment page (e.g. `docs/experiment2.qmd` or `docs/theta_sweep.qmd`) to stay consistent with tone and structure

### 3. Choose the Filename

Follow these naming conventions:
- Sequential experiments: `experiment<N>.qmd` where N is the next available number
- Thematic/named experiments: a short lowercase-underscore slug, e.g. `binary_ou_sweep.qmd`

Default to sequential numbering unless the experiment has a distinctive identity worth naming.

### 4. Create the `.qmd` File

Save to `docs/<filename>.qmd`. Use this structure:

```
---
title: "Experiment N: Descriptive Title"
---

## Overview  (or "Setup" or "Training")

One or two sentences describing what this experiment tests and why it is interesting.

```bash
python scripts/data_gen.py
python scripts/train.py
```

## Parameters

| Parameter | Value |
|-----------|-------|
| ... | ... |

## Results

### Conditional Mean

![PLACEHOLDER — replace with a descriptive caption once the figure is ready](images/PLACEHOLDER_mean.svg)

### Conditional Variance

![PLACEHOLDER — replace with a descriptive caption once the figure is ready](images/PLACEHOLDER_variance.svg)

## Notes  (optional)

Any observations, caveats, or follow-up ideas.

See [Results](results.qmd) for detailed analysis.
```

**Formatting rules to follow exactly:**
- YAML frontmatter: `title` field only, double-quoted string
- Heading levels: H2 for top-level sections, H3 for sub-sections within Results
- Math notation: `$\theta$` for inline, `$$...$$` for display equations — never write theta as plain text when it should be a symbol
- Code blocks: triple backticks with `bash` language specifier for commands
- Tables: pipe format, left-aligned headers with dashes
- Image paths: always `images/filename.svg` (relative to docs/), never absolute paths
- Internal links: `[Link text](filename.qmd)` format

### 5. Handle Image Placeholders

For each expected plot or figure that does not yet exist in `docs/images/`:
- Write the image reference with a placeholder filename that clearly communicates what should go there
- Add a parenthetical note in the caption: `(PLACEHOLDER — replace with actual filename once figure is saved)`
- Use the project's SVG naming convention: `<experiment_slug>_<metric>.svg`, e.g. `binary_ou_constant_len_mean.svg`

If the user provides actual filenames, use them directly with a descriptive caption following the established style:

> ![Theta sweep results for conditional mean prediction (coef 0). Relative error across θ and sequence length for trained vs. untrained model.](images/single_theta_constant_len_mean.svg)

### 6. Build the Parameters Table

Pull values from the configs if the user says "defaults" — read `configs/data.yaml`, `configs/model.yaml`, and `configs/train.yaml` to get the current values. Common rows to include:

| Parameter | Value |
|-----------|-------|
| $\theta$ (or distribution) | fixed value or "lognormal-sampled" |
| $\mu$ | 0.0 |
| $\Delta t$ | 0.1 |
| Marginal variance | 0.2 (fixed across trajectories) |
| Sequence length | 100 (or range if variable) |
| Training trajectories | 1000 |
| Train/val split | 80/20 |
| Epochs | 150 |
| Batch size | 64 |
| Learning rate | $10^{-3}$ |
| Optimizer | Adam |
| Loss | MSE on conditional mean/variance |
| `d_model` | 20 |
| `n_head` | 1 |
| `n_layers` | 2 |

Only include the rows relevant to this experiment. Add rows for anything that was changed from defaults.

### 7. Register in `_quarto.yml`

After creating the `.qmd` file, open `docs/_quarto.yml` and add the new page under the `Experiments` section:

```yaml
      - section: "Experiments"
        contents:
          - experiment1.qmd
          - experiment2.qmd
          - experiment3.qmd
          - <new_filename>.qmd   # ← add here
```

Preserve indentation exactly — YAML is whitespace-sensitive. The page must appear in this file to show up in the sidebar.

### 8. Confirm with the User

After creating the file and updating `_quarto.yml`, tell the user:
- The path of the new `.qmd` file
- Which image placeholders they need to fill in (if any)
- The exact SVG filenames to save plots to so they match the references in the page
- How to preview locally: `cd docs && quarto preview`
