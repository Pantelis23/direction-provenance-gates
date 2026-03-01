# Direction Provenance Gates for Causal Editing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Provenance](https://img.shields.io/badge/provenance-hermetic-blue)](docs/provenance.md)
[![Repro Gate](https://img.shields.io/badge/repro-gated-success)](scripts/ci_gates.sh)

This repository packages matched-cost causal-direction experiments with a provenance-first workflow:
- hermetic direction artifacts (`.npy` copies inside run outputs)
- semantic + raw hash integrity checks
- family-level acceptance gates
- reproducible manifests and artifact index

See `paper.md` for the methods/results write-up.

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
- direction integrity (`scripts/check_direction_integrity.py`)
- manifest generation and verification
- repo-level artifact index generation

## Main Docs

- `docs/provenance.md`: provenance contract and invariants
- `README_provenance.md`: quick provenance checklist
- `paper.md`: experiment summary and results

## Typical Workflow

1. Generate new artifacts (example):
   - `./run_mc_orth_vs_supportavoid_mc4_newartifact.sh`
2. Run gates:
   - `./scripts/ci_gates.sh`
3. Generate paper figures/tables:
   - `.venv/bin/python scripts/make_paper_figures.py ...`

## Repository Layout

- `configs/`: experiment configs
- `docs/`: specs and provenance contract
- `scripts/`: runners, analyzers, gates, manifests
- `src/`: metrics and telemetry utilities
- `runs/`: generated outputs/manifests (ignored in git)
- `paper.md`: paper draft
- `requirements-lock.txt`: frozen Python dependencies

## Citation

If you use this repository, cite the project artifact and methodology document:

```bibtex
@misc{christou2026directionprovenance,
  title        = {Direction Provenance Gates for Causal Editing},
  author       = {Pantelis Christou},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {\\url{https://github.com/Pantelis23/direction-provenance-gates}}
}
```

## License

MIT. See `LICENSE`.
