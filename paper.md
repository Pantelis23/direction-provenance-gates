# Provenance-First Integrity Gating for Direction-Based Causal Interventions

Pantelis Christou (independent)

## Abstract
Direction-vector interventions (e.g., activation steering, causal edits, probing) are widely used but often weakly auditable: direction vectors may be silently regenerated, paths may drift, and post-hoc backfills can depend on code state rather than exact artifacts. We present a minimal provenance protocol that treats direction vectors as first-class, hermetic artifacts with (i) canonical float32 normalization, (ii) semantic hashing over the float32 payload, (iii) hermetic copies embedded in downstream outputs, and (iv) an integrity gate that verifies both semantic identity and raw file identity between canonical and alias copies. We additionally introduce tie-proof invariants for validating support-avoid direction construction that avoid brittle overlap metrics. In a matched-cost case study (MC16) comparing an orthogonal random control to a support-avoid control, support-avoid sharply reduces envelope alignment to the source direction (mean abs cosine about 0.618 to 0.085) and reduces mean absolute effect and significance frequency, but non-null behavior persists, consistent with residual coupling beyond simple support overlap.

## Contributions
1. Direction provenance contract for per-control records:
   - `direction_sha256` (semantic hash: `sha256(float32_payload_bytes)`)
   - `direction_path` (canonical, typically `direction.npy`)
   - `direction_path_control` (control alias)
   - `direction_path_hermetic` and `direction_path_control_hermetic` (copied into matched-cost output)
2. Canonicalization: enforce float32, L2-normalized direction (`||d||=1`) before hashing and saving.
3. Integrity gate that verifies:
   - semantic hash vs recorded SHA
   - raw `.npy` byte identity between canonical and alias
   - optional alias requirement for new artifacts (`--require-alias`)
4. Tie-proof support-avoid invariants:
   - `masked_support_mass == 0`
   - `support_intrusion_frac_eps == 0`
   - overlap metrics are diagnostic-only
5. Legacy backfill protocol: prefer exact stored vectors; use generator reconstruction only when vectors are missing.
6. Runtime metadata embedded in summaries (`utc`, `python`, `numpy`, `hostname`) to diagnose environment drift.

## Threat Model
This protocol targets accidental or silent integrity failures common in direction experiments:
- silent direction regeneration (seed/RNG/code drift)
- path drift (relative paths resolving differently later)
- backfill ambiguity (generator-derived vectors instead of exact artifacts)
- tie-sensitive overlap metrics yielding false failures
- corrupted or swapped direction files

It does not prevent intentional adversarial falsification if an attacker records a wrong hash for a wrong vector.
We do not attempt to guarantee numerical reproducibility across BLAS/driver versions; we guarantee artifact identity and auditable provenance.

## Method

### Direction provenance contract
For each control record (`results_causal_summary.json` and `matched_cost_summary.json`) we persist:
- `direction_sha256`
- `direction_path`
- `direction_path_control`
- `direction_path_hermetic`
- `direction_path_control_hermetic`

Canonicalization:
- store `d` as float32 and L2-normalize before saving/hashing
- semantic hash: `sha256(d.astype(float32).reshape(-1).tobytes())`

Matched-cost outputs copy direction artifacts into:
- `out_dir/directions/{control}.direction.npy`
- `out_dir/directions/{control}.direction_control.npy`

### Integrity gate
`check_direction_integrity.py` supports three input modes with precedence:
1. `--seed-list` (file or `-` for stdin)
2. repeated `--seed-path` (path or glob)
3. fallback `--seed-glob`

Checks:
- `--hash-mode semantic`: semantic hash equals `direction_sha256`
- `--hash-mode raw`: exact `.npy` byte equality between canonical and alias
- `--hash-mode both`: both checks
- `--require-alias`: missing alias is an issue
- `--strict`: any issue returns non-zero

Golden gate (new artifacts):

```bash
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob 'runs/*/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --strict
```

`ci_gates.sh` environment overrides for custom paths:

```bash
# Newartifact seed summaries
export DIRECTION_GLOB='runs/*_newartifact/seed*.matched_cost_summary.json'
# Require paper assets (set 0 to skip paper figure/table presence guard)
export REQUIRE_PAPER_ASSETS=1
# Manifest outputs and asset globs
export ARTIFACTS_MANIFEST_OUT='runs/_artifacts_newartifact.json'
export ARTIFACTS_FIG_GLOB='runs/*_paper/*.png'
export ARTIFACTS_TABLE_GLOB='runs/*_paper/*.csv'
export ARTIFACTS_INDEX_OUT='runs/artifacts_index.json'
export ARTIFACTS_INDEX_GLOB='runs/_artifacts*.json'
```

### Tie-proof support-avoid invariants
Support is defined by source top-k indices of `abs(source)`.

Hard invariants for `orthogonal_random_support_avoid_top512`:
- `masked_support_mass = sum(abs(v[support])) / sum(abs(v)) == 0`
- `support_intrusion_frac_eps = mean(abs(v[support]) > support_intrusion_eps) == 0`

with scale-aware epsilon:
- `support_intrusion_eps = finfo(float32).eps * 8 * (max(abs(v)) + 1e-12)`

`support_overlap_frac` and `support_overlap_count` remain diagnostic-only due to tie sensitivity.

## Case Study: MC16 Matched-Cost Sweep
Controls:
- `orthogonal_random`
- `orthogonal_random_support_avoid_top512`

Family summary (`runs/mc_orth_vs_supportavoid_mc16_family.md`):
Here `sig_pos` means `CI_lo > 0`, `sig_neg` means `CI_hi < 0`, and `sig_two` means either condition.

- `orthogonal_random`:
  - matched_rate: 16/16
  - mean_abs_red: +0.001487
  - seed_boot_CI95(mean): [-0.002754, +0.005872]
  - sig_pos: 4/16, sig_neg: 2/16

- `orthogonal_random_support_avoid_top512`:
  - matched_rate: 16/16
  - mean_abs_red: +0.000211
  - seed_boot_CI95(mean): [-0.002457, +0.002836]
  - sig_pos: 1/16, sig_neg: 3/16

Paired matched-only delta (support-avoid minus orth):
- mean_delta: -0.0012753
- boot_CI95(mean_delta): [-0.0051734, +0.0024370]
- sign count: 9 negative / 7 positive

Interpretation: support-avoid reduces alignment and effect size tendency, but does not fully remove non-null behavior.

## New-Artifact Integrity Demonstration (K=4)
Using current code path and hermetic copying in matched-cost outputs:

```bash
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob 'runs/mc_orth_vs_supportavoid_mc4_newartifact/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --strict
```

Observed:
- `files=4 records_checked=8 ok_canon=8 ok_alias=8 missing_alias=0 mismatch_alias=0 issues=0`

This confirms end-to-end new-artifact integrity compliance.
On a fresh K=4 sweep generated under the current codepath, the integrity gate passed with `files=4 records_checked=8 ok_canon=8 ok_alias=8 issues=0`, confirming semantic+raw identity for canonical+alias hermetic direction artifacts.
Integrity here means: (i) semantic hash matches `direction_sha256`, and (ii) canonical and alias `.npy` are byte-identical (raw hash).

## Discussion
Provenance and integrity gating do not prove causal validity by themselves; they eliminate a major class of reproducibility failures that otherwise undermine causal claims. In this case study, support-avoid construction is verified by tie-proof invariants, yet occasional significant behavior remains, consistent with residual nonlinear/coupled response not explained by support overlap alone.

### Limitations
Integrity gating guarantees artifact identity, not causal validity. Support-avoid invariants guarantee no support leakage above epsilon, not that the model has no alternative pathways. Matched-cost calibration can mask or induce effects; because the paired mean-delta CI crosses zero at K=16, we do not claim `mean_delta != 0`.

## Figure captions
- **Fig 1 (abs cosine boxplot):** `abs_cos_to_source` for `orthogonal_random` versus `orthogonal_random_support_avoid_top512`; support-avoid collapses alignment.
- **Fig 2 (paired delta histogram):** per-seed paired delta (`support-avoid - orth`) in mean absolute reduction; distribution is slightly negative-centered while CI crosses zero.

## Reproducibility checklist
- New artifact direction integrity:
  - `check_direction_integrity.py --hash-mode both --require-alias --strict`
- Family accept gate:
  - `check_family_accept.py ...`
- Backfill policy:
  - prefer stored `direction_path(_control)`; generator fallback only if unavailable
- Runtime metadata:
  - present in top-level `runtime_meta` for both causal and matched-cost summaries
- Dependency lock:
  - `requirements-lock.txt` (exact Python deps used for figure + gate generation)
- Inputs to verify:
  - `requirements-lock.txt` (environment freeze)
  - `runs/_artifacts_newartifact.json` (per-run ledger: raw hashes + support-stats snapshot)
  - `runs/artifacts_index.json` (repo index: manifest hashes + git rev/dirty when available)
- Artifacts:
  - `runs/mc_orth_vs_supportavoid_mc16_paper/` (table + figures)
  - `runs/mc_orth_vs_supportavoid_mc4_newartifact/` (integrity-pass evidence)
  - `runs/_artifacts_newartifact.json` and `runs/artifacts_index.json` (artifact ledger + manifest index)

## Artifacts
- Provenance spec: `docs/provenance.md`
- Integrity gate: `scripts/check_direction_integrity.py`
- Backfill utility: `scripts/backfill_support_stats.py`
- Figure/table generator: `scripts/make_paper_figures.py`
- Single-command gate reproducer: `scripts/ci_gates.sh`
- MC16 paper outputs: `runs/mc_orth_vs_supportavoid_mc16_paper/`
- Artifact ledger: `runs/_artifacts_newartifact.json`
- Artifact index: `runs/artifacts_index.json`
