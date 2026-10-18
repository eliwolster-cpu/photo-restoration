# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
A Python CLI that conservatively restores scanned historical photos, one image at a time.
Pipeline: clean local damage (cracks, stains, scratches) → conservative sharpening → independent comparison with the original → save result + short report + status (`done` / `review` / `failed`).
Spec: [docs/spec.md](docs/spec.md) · Work plan: [docs/plan.md](docs/plan.md) (both in Hebrew).

## Core rules
- **The original file is never modified or deleted.** It is stored read-only and verified by hash.
- **Fidelity to the original beats beauty.** Don't invent detail. Inpaint only inside the damage mask, and leave pixels outside the mask unchanged.
- One processing pass per image, with no automatic fix loops. On doubt, set status `review` for a human. A suspected face change never results in automatic `done`.

## Scope
MVP only (spec §5). No colorization, Google Drive or agents until the user explicitly asks.
Don't add heavy model dependencies (LaMa, Real-ESRGAN, CodeFormer) until the user decides on tools and compute. There is no local GPU.

## Language and tools
- Python 3.11, virtual env in `.venv`, tests with pytest.
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest                          # all tests
pytest tests/test_x.py::test_y  # single test
python restore.py <image>       # run the CLI
```

## Structure (from docs/plan.md)
```
experiments/   # feasibility experiment (spec §6): OpenCV inpainting baseline, results.csv
restoration/   # pipeline modules: ingest, clean, sharpen, compare, decide, report
tests/         # pytest + small synthetic images
storage/       # git-ignored: originals/ (read-only), results/, reports/, jobs/
config.yaml    # fixed treatment settings and thresholds (thresholds live here, not in code)
restore.py     # CLI entry point
```
Output filenames include `restored` and the job ID.

## Secrets
Keys only in `.env`, which is git-ignored. Never put keys in code or commits.

## Workflow
1. Work on one step from `docs/plan.md` at a time, in order.
2. Run that step's "how to verify" check. `pytest` must pass before any commit.
3. Mark the step `[x]` in `docs/plan.md` and commit.

## Communication
Explanations to the user in Hebrew. Code, comments, and commit messages in English.
