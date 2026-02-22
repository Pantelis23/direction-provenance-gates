# Provenance + Integrity (Directions + Support-Avoid)

This repo treats *direction vectors* as first-class artifacts for causal edits / probing.
Goal: make runs hermetic, auditable, and reproducible.

## Threat model (what this prevents)
- Silent direction regeneration (RNG drift / different seed usage / code changes).
- Path drift (relative paths that later resolve differently).
- Backfill ambiguity (reconstructing directions from generators instead of using exact vectors).
- Tie-sensitive “overlap” metrics causing false failures.
- Corrupted or swapped direction files (accidental overwrite / wrong run dir).

## Direction provenance contract

### What must be recorded per control
Each per-control record (in `matched_cost_summary.json` and `results_causal_summary.json`) should include:
- `direction_sha256`: SHA256 of **float32 payload bytes** (semantic hash)
- `direction_path`: canonical vector path (typically `.../direction.npy`)
- `direction_path_control`: control-specific alias path (`.../control_direction_<control>.npy`)
- `direction_path_hermetic`: hermetic copy inside matched-cost output dir
- `direction_path_control_hermetic`: hermetic alias copy inside matched-cost output dir

### Canonicalization rules
- direction is stored as `float32`, L2-normalized (`||d||=1`).
- semantic hash is computed as `sha256(d.astype(float32).reshape(-1).tobytes())`.

## Golden integrity gate

For NEW artifacts (expected to include hermetic alias fields):

```bash
cd path/to/direction-provenance-gates
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob 'runs/*/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --strict
```

Meaning:
- `semantic`: canonical direction payload hash matches recorded `direction_sha256`
- `raw`: canonical `.npy` bytes match alias `.npy` bytes (exact file identity check)
- `both`: run both checks
- `--require-alias`: missing alias is an error for new artifacts
- `--strict`: any issue -> non-zero exit

## Support-avoid invariants (tie-proof)

Support set is defined by source direction top-k indices by `abs(source)`.

Hard invariants for `orthogonal_random_support_avoid_top512`:
- `masked_support_mass == 0.0`
  - `sum(abs(v[support])) / sum(abs(v))`
- `support_intrusion_frac_eps == 0.0`
  - fraction of `abs(v[support]) > support_intrusion_eps`
- `support_intrusion_eps` is scale-aware:
  - `eps = finfo(float32).eps * 8 * (max(abs(v)) + 1e-12)`

Diagnostic only (not gated due to tie sensitivity):
- `support_overlap_frac`
- `support_overlap_count`

## Backfill policy (legacy artifacts)

Backfills should prefer exact stored direction vectors if available:
- Try `direction_path` / `direction_path_control`
- Only fall back to generator reconstruction if no stored vectors exist

Legacy artifacts may lack hermetic alias fields; do not run `--require-alias` on them.

## Recommended CI gates (minimal)

Direction integrity (new artifacts):
- `scripts/check_direction_integrity.py ... --hash-mode both --require-alias --strict`

Family accept gate:
- `scripts/check_family_accept.py ...`
