# LLM Signature Framework — Vetting-Ready Edition

**Version:** 0.2.2

A prompt-as-code microframework that treats prompts as typed Python functions:
- **RuleOfOne:** one hard dependency — `pydantic>=2`.
- **KISS:** simple decorators, small modules, optional UI & extras.
- **Value/Complexity:** high ROI features with minimal code.

## CI
This repo ships a GitHub Actions workflow that runs **ruff** and **pytest** on Python 3.9–3.12 with an **80% coverage gate**.
(Replace `YOURUSER/llm-signature-framework` with your user/repo for the badge.)

![CI](https://github.com/YOURUSER/llm-signature-framework/actions/workflows/ci.yml/badge.svg)

### Coverage
CI enforces **>= 80%** statement coverage via `pytest-cov`. Adjust in `.github/workflows/ci.yml` if needed.

### Ruff auto-fix (label-triggered)
Add the **`ruff-fix`** label to a pull request to auto-run `ruff --fix` and push any formatting fixes back to the PR branch.


## Canvas snapshot
This repo includes a non-imported reference snapshot from the design canvas at:
```
canvas/core_canvas_v0_2_2.py
```
It’s for traceability only. The authoritative runtime sources are in:
```
src/llm_signature_framework/
```
