# Direction Provenance Gates for Causal Editing

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Provenance](https://img.shields.io/badge/provenance-hermetic-blue)](docs/provenance.md)
[![Repro Gate](https://img.shields.io/badge/repro-gated-success)](scripts/ci_gates.sh)

Code, gates and pre-registered experiments for the paper in [`paper.md`](paper.md),
*Pin the Vector, Test the Control: Provenance Gates and Random-Direction Baselines for Activation Interventions*.

**What it shows**
- **Provenance gates, evaluated by fault injection.** Direction vectors are stored as hashed, hermetic artifacts. Of 13 injected faults, the hardened integrity gate catches 12. The 13th, a run silently redone end to end, is caught only by a ledger pinned in git. A regenerate-then-verify CI step catches just 2.
- **Random directions are not null controls.** In GPT-2 and Qwen3-0.6B, random directions shift a gender-bias metric by real, direction-specific amounts: in 15/18 cells as a plain unit projection, and in 18/18 when edits are matched to a fixed general-text KL budget. Support-avoiding and subspace-matched random directions behave the same as unconstrained ones.
- **Regeneration is not reproduction.** The same seed reproduces every vector hash on the same machine, but not on different hardware, where BLAS kernel dispatch changes a few float32 ULPs. Pin stored vectors; don't regenerate them.

Hypotheses and decision rules are pre-registered in [`docs/prereg_2026-09-28.md`](docs/prereg_2026-09-28.md).

## Quick Start

```bash
cd path/to/direction-provenance-gates
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
source scripts/activate_env.sh
python scripts/run_mvp.py --config configs/mvp.yaml
```

## One-Command Reproducibility Gate

```bash
./scripts/ci_gates.sh
```

This checks:
- family acceptance (`scripts/check_family_accept.py`)
- direction integrity (`scripts/check_direction_integrity.py`: hermetic copies, canonical form, semantic and raw hashes, expected files and controls)
- a fault-injection regression (`scripts/fault_injection.py` against `ledgers/fault_injection_expected.json`), so weakening the gate fails CI
- verification against the **pinned** ledger in `ledgers/`. CI never regenerates it; refresh it only with `UPDATE_LEDGER=1` and commit the diff
- repo-level artifact index generation

## Main Docs

- `paper.md`: the paper
- `docs/prereg_2026-09-28.md`: pre-registration, with addenda A (Qwen) and B (KL-matched design)
- `docs/provenance.md`: provenance contract and invariants
- `README_provenance.md`: quick provenance checklist

## Typical Workflow

1. Run a sweep, e.g. `./run_kl_sweeps.sh` (KL-matched controls, GPT-2 and Qwen3-0.6B). The full list of runners is in `paper.md` §8.
2. Gate it: `scripts/check_direction_integrity.py --strict --require-hermetic --check-canonical --expect-files 16 ...`
3. Analyse it: `scripts/analyze_mc_controls.py` (Cochran's Q, τ, paired tests), `scripts/calibrate_block_se.py` (SE calibration).
4. Run `./scripts/ci_gates.sh` before committing.

## Repository Layout

- `configs/`: experiment configs
- `docs/`: pre-registration, specs and provenance contract
- `ledgers/`: pinned artifact ledger and the expected fault-injection matrix (committed)
- `scripts/`: runners, analyzers, gates, manifests
- `src/`: metrics and telemetry utilities
- `runs/`: generated outputs/manifests (ignored in git)
- `paper.md`: paper draft
- `requirements-lock.txt`: frozen Python dependencies

## Citation

If you use this repository, cite the project artifact and methodology document:

```bibtex
@misc{christou2026directionprovenance,
  title        = {Pin the Vector, Test the Control: Provenance Gates and Random-Direction Baselines for Activation Interventions},
  author       = {Pantelis Christou},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {\\url{https://github.com/Pantelis23/direction-provenance-gates}}
}
```

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
