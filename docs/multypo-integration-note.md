# MulTypo Integration Note

This pipeline uses the upstream **cisnlp/multypo** implementation via the published Python package `multypo`.

- Upstream component used: `multypo.generate_typos`
- Language setting: `english`
- Purpose: keyboard-aware typo perturbation for robustness evaluation in CI

The local script `scripts/multypo_robustness_eval.py` is only an evaluation harness/adapter around upstream functionality. It does **not** reimplement typo generation logic.
