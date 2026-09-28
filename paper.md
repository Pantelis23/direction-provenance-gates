# Provenance-First Integrity Gating for Direction-Based Interventions

Pantelis Christou (independent)

## Abstract

Direction-vector interventions (activation steering, concept erasure, causal edits) are only as trustworthy as the vectors they apply, yet those vectors are usually treated as disposable by-products: regenerated from seeds, referenced by relative paths, and reconstructed after the fact. We describe a small provenance protocol that treats each direction as a hashed, hermetically copied artifact, and we evaluate it by injecting 13 realistic faults into a real experiment. A self-contained integrity gate (hash and file-identity checks against metadata written in the same run) catches 12 of the 13 once hardened; the original version caught 6. The remaining fault, a run silently redone end to end, is invisible to any self-contained check and is caught only by a ledger pinned outside the run (here, committed to git). A regenerate-then-verify CI step, which we had originally used, catches just 2 of the 13. As a worked example we re-run a 16-seed matched-cost control experiment on GPT-2 under the hardened gate. ⟨V2: reproducibility sentence — e.g. "Regenerating the directions from their seeds reproduced all 32 legacy hashes."⟩ ⟨V2: one-sentence scientific result — expected shape: "Random directions orthogonal to the source have real, direction-specific effects on the bias metric (Cochran's Q, p < 1e-5) that average to zero; avoiding the source's top-512 support does not shrink them, and neither does a subspace-matched random mask."⟩

## 1. Introduction

A direction-based intervention edits a hidden state h along a unit vector d. Claims about such interventions are claims about a specific d: "projecting out d reduces bias", "a random direction of equal cost does not". Those claims are only checkable if the exact d used in each run can be recovered later. In practice d is often regenerated from a seed, loaded from a path relative to whatever the working directory happens to be, or reconstructed by a later "backfill" script from generator code that may have changed.

None of these failure modes crash anything. The experiment still runs, the numbers still look plausible, and nothing tells you that the vector in the table is not the vector that produced it.

This paper makes three contributions:

1. **A provenance contract for direction vectors** (Section 3): canonical float32 unit vectors, a semantic hash of the payload, hermetic copies inside each run's output, and a ledger of hashes pinned in version control.
2. **A fault-injection evaluation of the gates** (Section 4): 13 injected faults, showing what a self-contained integrity gate can and cannot catch, and that a common CI pattern (regenerate the manifest, then verify it) catches almost nothing.
3. **A worked example** (Section 5): a 16-seed matched-cost comparison of random control directions on GPT-2, re-run under the gate, including a subspace-matched control that the original version of this analysis lacked.

## 2. Related work

**Direction-based interventions.** Linear directions in activation space are used to steer generation (Turner et al., 2023, *Activation Addition*; Li et al., 2023, *Inference-Time Intervention*; Zou et al., 2023, *Representation Engineering*) and to remove concepts (Ravfogel et al., 2020, *INLP*; Elazar et al., 2021, *Amnesic Probing*; Belrose et al., 2023, *LEACE*). Gender directions in embeddings go back at least to Bolukbasi et al. (2016). These works establish methods; none of them specifies how the resulting vectors should be stored and verified across runs.

**Controls for interventions.** Random-direction baselines are the standard control for directional effects. Our results (Section 5) add a caution: at matched cost, individual random directions have real, direction-specific effects, so a single random control is a noisy baseline.

**Reproducibility tooling.** Experiment trackers and data-versioning tools (e.g. MLflow, DVC) and reporting standards (Pineau et al., 2021; Mitchell et al., 2019; Gebru et al., 2021) record configurations, code versions and datasets. Our protocol is narrower and complementary: it pins the exact vector payloads that an intervention applied, and it is evaluated against injected faults, not just described.

## 3. Protocol

### 3.1 Threat model

We target accidental, silent integrity failures:

| threat | example |
|---|---|
| silent regeneration | seed handling or generator code changes; a "rerun" produces a different d |
| path drift | a relative path later resolves to a different file |
| backfill ambiguity | a post-hoc script reconstructs d from a generator instead of loading the stored vector |
| corruption / swap | truncated write, wrong file copied into the right name |
| incomplete runs | a control record or a whole seed summary missing, so a glob silently matches fewer files |

We do not defend against deliberate falsification, and we do not guarantee bit-exact numerical reproducibility across BLAS/driver versions. We guarantee that the vector recorded is the vector on disk, and that the set of vectors has not changed since it was pinned.

### 3.2 Direction contract

Every direction is stored as a 1-D float32 array with unit L2 norm. Its semantic hash is `sha256(d.astype(float32).reshape(-1).tobytes())`, which is independent of `.npy` header details. Each per-control record in a run summary stores this hash and two paths (a canonical `direction.npy` and a control-named alias). The matched-cost driver additionally copies both files into the run's own output directory (`directions/{control}.direction.npy`, `directions/{control}.direction_control.npy`) and records those hermetic paths. The field-level schema is in `docs/provenance.md`.

### 3.3 Gates

**Integrity gate** (`scripts/check_direction_integrity.py`). For every record it checks that:
- the hermetic copies exist (no silent fallback to the original run directory)
- the canonical file is float32, 1-D, finite and has unit norm
- its semantic hash matches the recorded hash, and the record-level and nested hashes agree
- the alias is byte-identical to the canonical file
- every expected control appears exactly once, and the expected number of seed summaries is present

All of these compare the run against metadata the same run wrote.

**Pinned ledger** (`ledgers/artifacts_newartifact.json`, checked by `scripts/check_artifacts_manifest.py`). It stores raw hashes of every seed summary and hermetic direction file, plus the recorded semantic hashes and support statistics. It is committed to git. CI verifies against it and never regenerates it; updating it requires an explicit `UPDATE_LEDGER=1`, which then shows up in `git diff`.

### 3.4 Support-avoid invariants

One of the controls in Section 5 is a random direction that avoids the source direction's *support*: the top-k coordinates of |source|. Checking this by comparing top-k index sets is fragile, because ties at the k-th value make the overlap depend on sort order. Instead we check two invariants that do not depend on the candidate vector's own ranking:
- `masked_support_mass = sum(|v[S]|) / sum(|v|) == 0`
- `support_intrusion_frac_eps = mean(|v[S]| > eps) == 0`, where `eps = 8 · float32_eps · max|v|`

Here S is the source's top-k set. S is itself computed with `argpartition` and so breaks ties arbitrarily; the generator and the checker use the same routine on the same array, so they agree, but S is not tie-invariant in an absolute sense. These invariants hold by construction for the support-avoid generator; their value is catching a changed or swapped generator.

## 4. Evaluating the gates by fault injection

`scripts/fault_injection.py` copies a clean 4-seed, 2-control run (8 direction records) into a sandbox, pins a ledger, applies one fault, and runs three checks: the integrity gate, the CI step as originally written (regenerate the manifest from the current files, then verify against it), and verification against the pinned ledger.

| fault | integrity gate, original | integrity gate, hardened | CI: regenerate + verify | pinned ledger |
|---|---|---|---|---|
| none (false-alarm check) | pass | pass | pass | pass |
| direction regenerated, both copies overwritten | caught | caught | missed | caught |
| one bit flipped in payload | caught | caught | missed | caught |
| `.npy` truncated | caught | caught | missed | caught |
| two controls' files swapped | caught | caught | missed | caught |
| alias diverged from canonical | caught | caught | missed | caught |
| hermetic copies deleted (originals remain) | **missed** | caught | caught | caught |
| all copies deleted | caught | caught | caught | caught |
| run redone end to end (vector + summary hash) | **missed** | **missed** | missed | caught |
| non-unit vector with self-consistent hash | **missed** | caught | missed | caught |
| record-level vs nested hash conflict | **missed** | caught | missed | caught |
| one control record dropped | **missed** | caught | missed | caught |
| empty results list | **missed** | caught | missed | caught |
| whole seed summary deleted | **missed** | caught | missed | caught |
| **caught (of 13 faults)** | **6** | **12** | **2** | **13** |

Three lessons:

1. **Silent fallbacks defeat hermeticity.** The original gate, on finding a hermetic copy missing, quietly checked the original run-directory file instead, and passed.
2. **Self-contained checks have a ceiling.** Any check that compares a run against metadata written by the same run cannot detect the run being consistently redone. That fault needs a reference that was fixed earlier and stored elsewhere.
3. **Regenerate-then-verify is a tautology.** It proves the files match themselves. It only catches faults that make generation itself fail.

The pinned ledger catches everything here, but only relative to the moment it was pinned: a fault present at pinning time (for example, a non-canonical vector) is pinned along with it. That is why the canonical-form check belongs in the integrity gate rather than only in the ledger.

## 5. Worked example: matched-cost random controls on GPT-2

### 5.1 Setup

**Model and data.** GPT-2 small (768-dim residual stream), layer 11. Sentences follow templates such as "{name} works as {article} {job}." for 5 gendered name pairs (John/Mary, Michael/Jennifer, …), 56 occupations and 8 templates. Each sentence is paired with a neutral baseline ("{name} works as a person.").

**Bias metric.** For a name pair, the bias is the difference in average token log-probability between the male-name and female-name sentence, minus the same difference for the baseline template. The effect of an intervention is `abs_red = |bias_before| − |bias_after|`, averaged over held-out (template, job) blocks (5% of blocks, fixed split). Positive means the intervention reduced the magnitude of the bias. CIs are 95% block-bootstrap intervals (5,000 resamples).

**Intervention.** At layer 11, on non-name tokens: h ← h − α (h·d) d. At α = 1 this projects d out. The matched α values here are about 5, so the component along d is reversed and amplified (about −4×), not removed.

**Source direction.** A ridge-regression direction (λ = 1) that predicts a name pair's baseline-adjusted bias from the difference of the two sentences' layer-11 mean-pooled hidden states (1,555 training pairs, train Pearson r = 0.94). It was trained with a 30% block holdout using the same block hash, so the 5% evaluation blocks are a subset of blocks it never saw. No control direction ever uses it directly; it only defines what the controls must be orthogonal to and which coordinates they must avoid.

**Controls.** One direction per seed and control (seeds 1–16):
- `orth`: Gaussian random, orthogonalised to the source.
- `support-avoid`: Gaussian random on the 256 coordinates *outside* the source's top-512 |coordinates|, orthogonalised to the source. Those 256 coordinates carry only 1.6% of the source's energy.
- `random-mask` (new): the same construction on 256 coordinates chosen uniformly at random. It matches support-avoid's dimensionality but not its avoidance, which separates "avoids the source's support" from "confined to one third of the coordinates".

**Matched cost.** For each direction, α is calibrated so that the mean drop in sentence log-probability (a fluency cost) matches a per-seed target, to within max(0.003, 2% of the target). Targets are fixed to the values used in the original sweep, so orth and support-avoid are exact re-runs.

**Statistics.** Per control: mean effect over seeds with a seed-bootstrap CI; per-seed significance counts; Cochran's Q across the 16 directions, using each direction's block-bootstrap SE, with a Monte Carlo p-value under the null that all directions share one effect; and the DerSimonian–Laird between-direction SD τ. Between controls: paired sign-flip permutation tests on the signed and absolute effects, and a seed-bootstrap CI for the ratio of τ values (`scripts/analyze_mc_controls.py`).

### 5.2 Results

⟨V2: integrity — "The sweep passes the hardened gate (48 records) and is pinned in the ledger."⟩

⟨V2: reproducibility — hashes and effect equality vs legacy for orth and support-avoid.⟩

⟨V2: table — per control: matched, mean [CI], mean |effect|, sig+/sig−, τ, Q p, α, abs-cos to source.⟩

⟨V2: pairwise — sign-flip p for signed and |effect|, τ ratios with CIs.⟩

⟨V2: figure — per-seed effect with CI, three controls side by side.⟩

**Reading the envelope-alignment numbers.** `abs_cos_to_source` is the cosine between |v| and |source|. It drops from about 0.62 (orth) to about 0.09 (support-avoid) *by construction*: support-avoid is exactly zero on the coordinates where |source| is large. It is a check that the construction worked, not a finding.

### 5.3 Interpretation

⟨V2: fill after results. Draft, valid only if v2 matches the legacy analysis: Random orthogonal directions are not null. At matched cost each one shifts the bias metric by a real, direction-specific amount (Q far above its null), in both directions, averaging out near zero. Avoiding the source's support does not reduce this spread, and neither does confining the direction to a random third of the coordinates. For intervention studies, one random control direction is therefore a noisy baseline. The comparison should be against a distribution of random directions, reporting between-direction spread and not only the mean.⟩

## 6. Limitations

- One model (GPT-2 small), one layer, one bias metric, one source direction. ⟨Optional: second model, e.g. Qwen3-0.6B.⟩
- The held-out set is small (5% of blocks), and all seeds share it. Per-seed CIs are therefore correlated, which is why we test heterogeneity with Q rather than by counting significant seeds.
- Q relies on the block-bootstrap SEs being accurate. If they are too small, Q is inflated.
- The gates establish artifact identity, not causal validity. The support-avoid invariants establish no leakage onto the source's top-k coordinates, not the absence of other pathways.
- The pinned ledger is only as trustworthy as its git history.

## 7. Reproducing

```bash
./run_mc16_v2_randmask.sh                                  # 16 seeds x 3 controls
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob 'runs/mc16_v2_randmask/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --require-hermetic --check-canonical --strict \
  --expect-files 16 \
  --expect-controls orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512
.venv/bin/python scripts/analyze_mc_controls.py \
  --seed-glob 'runs/mc16_v2_randmask/seed*.matched_cost_summary.json' \
  --controls orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512 \
  --reference-glob 'runs/mc_orth_vs_supportavoid_mc16/seed*.matched_cost_summary.json'
.venv/bin/python scripts/fault_injection.py               # Section 4 table
./scripts/ci_gates.sh                                      # verifies against the pinned ledger
```

Dependencies are frozen in `requirements-lock.txt`. Run summaries embed `runtime_meta` (UTC time, Python and NumPy versions, hostname).

## References

- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. *NeurIPS*.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS*.
- Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1).
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3).
- Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *TACL*.
- Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12).
- Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS*.
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.
- Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *JMLR*, 22.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI.
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. *ACL*.
- Turner, A. M., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.
- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.
