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
  --hash-mode both --require-alias --require-hermetic --check-canonical --strict \
  --expect-files <K> --expect-controls <c1>,<c2>
```

## Notes
- Use `--require-alias` only for post-hardening runs that include hermetic alias paths.
- Legacy runs can be checked without alias requirement.
- CI verifies against the ledger pinned in `ledgers/`; refresh it only with `UPDATE_LEDGER=1` and commit the diff.
- `scripts/fault_injection.py` shows which faults each gate catches.
