---
name: Test config should inherit physics params
description: Physics parameters in test.yaml must not duplicate data.yaml — test.py should fall back to the main physics config to avoid silent mismatches
type: feedback
---

Test sweep configs should not hardcode physics parameters (mu, dt, marginal_variance) that are already defined in data.yaml. Duplicating them causes silent mismatches when one is updated but not the other.

**Why:** User got degraded standard OU results because test.yaml had mu=0.0 while data.yaml had mu=5 after a config change. The model was tested on data from a different distribution than it was trained on.

**How to apply:** When adding new physics parameters or config fields, always check whether test.yaml or other configs duplicate them. Prefer falling back to the main config with sweep-level overrides only when intentionally testing out-of-distribution.
