# Provenance Quickstart

This project uses provenance-first direction artifacts for causal intervention runs.

## Read first
- Full spec: `docs/provenance.md`

## CI gate script
- `scripts/ci_gates.sh`

## Golden integrity gate (new artifacts)
```bash
cd path/to/direction-provenance-gates
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob 'runs/*_newartifact/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --strict
```

## Notes
- Use `--require-alias` only for post-hardening runs that include hermetic alias paths.
- Legacy runs can be checked without alias requirement.
