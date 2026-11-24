# Provenance-First Integrity Gating for Direction-Based Interventions

Pantelis Christou (independent)

## Abstract

Direction-vector interventions (activation steering, concept erasure, causal edits) are only as trustworthy as the vectors they apply, yet those vectors are usually treated as disposable by-products: regenerated from seeds, referenced by relative paths, and reconstructed after the fact. We describe a small provenance protocol that treats each direction as a hashed, hermetically copied artifact, and we evaluate it by injecting 13 realistic faults into a real experiment. A self-contained integrity gate (hash and file-identity checks against metadata written in the same run) catches 12 of the 13 once hardened; the original version caught 6. The remaining fault, a run silently redone end to end, is invisible to any self-contained check and is caught only by a ledger pinned outside the run (here, committed to git). A regenerate-then-verify CI step, which we had originally used, catches just 2 of the 13. We then use the gates in a pre-registered test of the most common control in this literature, the random direction, in GPT-2 and Qwen3-0.6B. Random directions are not null controls: even a plain unit projection gives direction-specific effects in 15 of 18 model × layer × control cells, and at a fixed KL budget all 18 are heterogeneous (17 past the pre-registered Bonferroni threshold), with no difference between unconstrained, support-avoiding and subspace-matched random directions. Regenerating the original directions on the same machine reproduced every hash and every effect estimate exactly. On different hardware, regeneration changed the vectors by a few float32 ULPs, which the hash check flagged, while results agreed to 10⁻⁶. That is direct evidence for pinning stored vectors rather than regenerating them.

## 1. Introduction

A direction-based intervention edits a hidden state h along a unit vector d. Claims about such interventions are claims about a specific d: "projecting out d reduces bias", "a random direction of equal cost does not". Those claims are only checkable if the exact d used in each run can be recovered later. In practice d is often regenerated from a seed, loaded from a path relative to whatever the working directory happens to be, or reconstructed by a later "backfill" script from generator code that may have changed.

None of these failure modes crash anything. The experiment still runs, the numbers still look plausible, and nothing tells you that the vector in the table is not the vector that produced it.

This paper makes three contributions:

1. **A provenance contract for direction vectors** (Section 3): canonical float32 unit vectors, a semantic hash of the payload, hermetic copies inside each run's output, and a ledger of hashes pinned in version control.
2. **A fault-injection evaluation of the gates** (Section 4): 13 injected faults, showing what a self-contained integrity gate can and cannot catch, and that a common CI pattern (regenerate the manifest, then verify it) catches almost nothing.
3. **A pre-registered test of the random-direction control** (Section 5): whether random directions are null baselines depends on what they are size-matched on. We compare matched cost with unit projection at three depths in GPT-2 and Qwen3-0.6B, with every vector run through the gates of Section 3.

## 2. Related work

**Direction-based interventions.** Linear directions in activation space are used to steer generation (Turner et al., 2023, *Activation Addition*; Li et al., 2023, *Inference-Time Intervention*; Zou et al., 2023, *Representation Engineering*) and to remove concepts (Ravfogel et al., 2020, *INLP*; Elazar et al., 2021, *Amnesic Probing*; Belrose et al., 2023, *LEACE*). Gender directions in embeddings go back at least to Bolukbasi et al. (2016). These works establish methods; none of them specifies how the resulting vectors should be stored and verified across runs.

**Controls for interventions.** Random-direction baselines are the standard control for directional effects, size-matched either on norm or on some behavioural cost. Anthropic (2026) report that matched-norm random perturbations leave behaviour near baseline, while edits in a small subspace of verbalizable directions do not. Section 5 tests when the random-control premise holds in open models.

**Reproducibility tooling.** Experiment trackers and data-versioning tools (e.g. MLflow, DVC) and reporting standards (Pineau et al., 2021; Mitchell et al., 2019; Gebru et al., 2021) record configurations, code versions and datasets. Our protocol is narrower and complementary: it pins the exact vector payloads that an intervention applied, and it is evaluated against injected faults, not just described.

## 3. Protocol

### 3.1 Threat model

We target accidental, silent integrity failures:

| threat | example |
|---|---|
| silent regeneration | seed handling or generator code changes, or the same code on different hardware (§5.6); a "rerun" produces a different d |
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

**Keeping the gate honest.** The detection matrix above is committed (`ledgers/fault_injection_expected.json`), and `ci_gates.sh` re-injects all 13 faults on every run and fails if any detection changes. Dropping a single integrity flag turns CI red, naming the faults that would slip through.

## 5. Worked example: are random control directions null?

Random directions are the default control for directional interventions: if a learned direction d changes behaviour and a random direction of "equal size" does not, the effect is attributed to d. This section asks whether that premise holds, under two meanings of "equal size", at three depths, in two models. All hypotheses and decision rules were pre-registered before the runs (`docs/prereg_2026-09-28.md`, committed at `dfb4e33`; Qwen addendum at `d490967`).

### 5.1 Common setup

**Data.** Sentences follow templates such as "{name} works as {article} {job}." for 5 gendered name pairs (John/Mary, Michael/Jennifer, …), 56 occupations and 8 templates. Each sentence is paired with a neutral baseline ("{name} works as a person.").

**Bias metric.** For a name pair, the bias is the difference in average token log-probability between the male-name and female-name sentence, minus the same difference for the baseline template. The effect of an intervention is `abs_red = |bias_before| − |bias_after|`, averaged over held-out (template, job) blocks (5% of blocks, fixed split). Positive means the intervention reduced the magnitude of the bias. CIs are 95% block-bootstrap intervals (5,000 resamples).

**Intervention.** At one layer, on non-name tokens: h ← h − α (h·d) d, applied to the block's output (the residual stream).

**Source directions.** Per layer, a ridge-regression direction (λ = 1) predicts a name pair's baseline-adjusted bias from the difference of the two sentences' mean-pooled hidden states at that layer. It was trained with a 30% block holdout using the same block hash as the evaluation split, so the 5% evaluation blocks are a subset of blocks it never saw. No control ever applies it; it only defines what the controls must be orthogonal to and which coordinates they avoid.

**Controls.** One direction per seed and control (seeds 1–16):
- `orth`: Gaussian random, orthogonalised to the source.
- `support-avoid`: Gaussian random on the coordinates *outside* the source's top-512 |coordinates|, orthogonalised to the source. In GPT-2 that leaves 256 of 768 coordinates.
- `random-mask`: the same construction with 512 coordinates masked uniformly at random. It matches support-avoid's dimensionality but not its avoidance, separating "avoids the source's support" from "confined to a third of the coordinates".

**Models and layers.** GPT-2 small (12 blocks, d = 768) at blocks 1, 6 and 11; Qwen3-0.6B (28 blocks, d = 1024, float32, weights sha256 `f47f7117…`) at blocks 1, 14 and 26.

**A disclosed flaw in the original layer-11 source direction.** In Hugging Face models the last entry of `hidden_states` has the final normalisation applied. The original GPT-2 layer-11 source direction was learned on that post-`ln_f` state but is applied to the pre-`ln_f` block output, where `ln_f`'s per-coordinate gain ranges from 0.004 to 17.4. The two spaces differ substantially, so at layer 11 "orthogonal to the source" and "the source's top-512 support" are defined in the wrong space. This does not affect whether random directions are null, which is the question here, but it does weaken any support-avoid interpretation at layer 11. The directions at GPT-2 blocks 1 and 6 and at all Qwen layers are learned and applied in the same space. For Qwen we therefore use block 26, not the last block, 27.

### 5.2 Two meanings of "equal size"

- **Matched cost (MC).** α is calibrated per direction so that the mean drop in sentence log-probability (a fluency cost) matches a per-seed target, to within max(0.003, 2% of the target). At GPT-2 layer 11 the targets are pinned to the legacy sweep's values, so orth and support-avoid are exact re-runs. Elsewhere the target is 5% of the weakest control's cost at α = 40. Matched α values are well above 1 (MC: 1.05–29.8 at block 11; KL: 0.7–16.5 across all cells), so the component along d is reversed and amplified, not removed.
- **Unit projection (P1).** α = 1: a true projection of d out of the residual stream, with no calibration. This is the analogue of the matched-norm controls in Anthropic's J-space work (Section 6).

### 5.3 Jacobian-lens propagation

For GPT-2 blocks 1 and 6 we fit a Jacobian lens (anthropics/jacobian-lens, commit `581d398`) on 300 WikiText-103 test paragraphs truncated to 128 tokens. J_ℓ is the average Jacobian of the final residual state with respect to the layer-ℓ residual. For a unit direction v, A(v) = ‖J_ℓ v‖ measures how strongly a perturbation along v propagates to the output. At GPT-2 block 11 nothing lies between the edit and `ln_f`, so the Jacobian lens reduces exactly to the logit lens and A(v) is not informative. We do not fit it there.

### 5.4 Statistics

Per control and condition: the mean effect over the 16 directions with a seed-bootstrap CI; Cochran's Q across directions, using each direction's block-bootstrap SE, with a Monte Carlo p-value under the null that all directions share one effect; and the DerSimonian–Laird between-direction SD τ. Between controls: paired sign-flip permutation tests and a seed-bootstrap CI for τ ratios (`scripts/analyze_mc_controls.py`).

Q is only as good as the SEs. H1 checks them directly: 9 fixed directions are each re-evaluated on 16 different held-out splits, where their true effect is constant, so Q/df should be about 1 (`scripts/calibrate_block_se.py`).

### 5.5 Results

Each pre-registered hypothesis is reported against its decision rule, whether or not it held.

- **Integrity and reproducibility.** The GPT-2 layer-11 matched-cost sweep (16 seeds × 3 controls) passes the hardened gate with all 48 records clean. Regenerating the 32 orth and support-avoid directions from their seeds, seven months after the legacy sweep, reproduced all 32 semantic hashes, and with the legacy per-seed cost targets pinned, every effect estimate matched the legacy value exactly (maximum absolute difference 0.0). Every sweep reported below passes the hardened gate with `--expect-files 16` and all three controls required: 15 sweeps of 48 records each (GPT-2: MC ×3, P1 ×3, KL ×3; Qwen: P1 ×3, KL ×3), 720 records in total, none with an issue. The stopped Qwen MC run (4 seeds) is not analysed.

**GPT-2, block 11, matched cost** (16 directions per control; Q on 15 df, Monte Carlo p with a floor of 5·10⁻⁶):

| control | mean effect [seed-bootstrap 95% CI] | mean \|effect\| | τ | Q | p | mean α | abs-cos to source |
|---|---|---|---|---|---|---|---|
| orth | +0.0015 [−0.0028, +0.0059] | 0.0061 | 0.0037 | 62.6 | < 10⁻⁵ | 6.4 | 0.62 |
| support-avoid | +0.0002 [−0.0025, +0.0029] | 0.0041 | 0.0035 | 59.0 | < 10⁻⁵ | 4.8 | 0.09 |
| random-mask | −0.0004 [−0.0044, +0.0033] | 0.0061 | 0.0033 | 55.6 | < 10⁻⁵ | 11.3 | 0.37 |

No pair of controls differs, on the signed effect (sign-flip p = 0.53–0.81), on |effect| (p = 0.13–0.99), or on τ (ratios 0.88–0.94, each 95% CI spanning roughly 0.4–2.0). Reaching the same cost takes the random-mask control about twice the α of orth, and support-avoid about three quarters of it.
- **H1, SE calibration: passed.** Re-evaluating each of 9 fixed directions (GPT-2 block 11, seeds 1–3 × 3 controls) on 16 held-out splits gives per-direction Q/df between 0.41 and 1.50, pooled Q/df = 1.046, inside the 95% band for calibrated SEs (0.775–1.253). The block-bootstrap SEs are accurate, so no correction is applied to Q below. For comparison, Q/df *across* directions at block 11 is 3.7–4.2: a fixed direction varies across held-out splits about as much as its SE predicts, while different random directions differ by several times that.
- **H2, heterogeneity under a matched cost.** The pre-registered PLL-matched condition (MC) turned out to be flawed (addendum B). On 3 of 16 seeds at GPT-2 blocks 1 and 6, the per-seed target fell below the matching tolerance, so α = 0 counted as "matched". Per-seed targets also varied 10–60×, which confounds across-direction spread with seed-level cost (|effect| vs target: ρ = +0.43, p = 0.007 at block 1). The block-11 MC table above uses legacy targets, is free of vacuous matches, and is reported as secondary evidence. The replacement condition matches every direction to one fixed KL(original ‖ edited) of 0.10 nats per position on WikiText. All 288 directions matched, with no α = 0 (the old MC runs had 15 vacuous α = 0 matches). **H2-KL failed narrowly: 17 of 18 cells** are heterogeneous at the pre-registered p < 0.05/18. The exception is Qwen block 14 random-mask, at p = 0.0049: heterogeneous at conventional levels, but not past the Bonferroni threshold the prediction required for *every* cell.

**GPT-2, KL-matched at 0.10 nats/position (addendum B)** — 144/144 directions matched (achieved KL 0.095–0.105, α 0.8–13.2); same vectors as under P1:

| block | control | mean effect | mean \|effect\| | τ (KL) | τ (P1) | ratio | Q | p | mean α |
|---|---|---|---|---|---|---|---|---|---|
| 1 | orth | +0.0014 | 0.0120 | 0.0085 | 0.00075 | 11.3× | 76.1 | < 10⁻⁵ | 7.0 |
| 1 | support-avoid | +0.0027 | 0.0135 | 0.0101 | 0.00056 | 18.1× | 57.9 | < 10⁻⁵ | 9.2 |
| 1 | random-mask | +0.0083 | 0.0197 | 0.0094 | 0.00193 | 4.9× | 71.4 | < 10⁻⁵ | 7.1 |
| 6 | orth | +0.0002 | 0.0052 | 0.0025 | 0.00098 | 2.5× | 49.9 | < 10⁻⁴ | 4.4 |
| 6 | support-avoid | +0.0002 | 0.0045 | 0.0026 | 0.00140 | 1.9× | 94.4 | < 10⁻⁵ | 3.2 |
| 6 | random-mask | +0.0027 | 0.0056 | 0.0027 | 0.00072 | 3.8× | 50.5 | < 10⁻⁴ | 6.1 |
| 11 | orth | -0.0011 | 0.0046 | 0.0035 | 0.00078 | 4.5× | 71.7 | < 10⁻⁵ | 5.5 |
| 11 | support-avoid | -0.0009 | 0.0035 | 0.0031 | 0.00101 | 3.1× | 63.8 | < 10⁻⁵ | 3.7 |
| 11 | random-mask | +0.0007 | 0.0035 | 0.0024 | 0.00040 | 6.0× | 43.5 | 0.0001 | 6.4 |

- **H3, P1 vs MC: first part failed.** H3 predicted that under unit projection most of the 6 cells at blocks 1 and 6 would show no heterogeneity. All 6 do (p ≤ 0.009), as do all 3 at block 11. Even a plain projection of a random direction, with no amplification, shifts the bias metric by a real, direction-specific amount at every depth tested. The effects are small, with mean |effect| of about 0.0006–0.003 nats against 0.004–0.006 under matched cost at block 11, but they are several times their SEs. The second part, as re-specified in addendum B (τ(KL) > τ(P1) in a majority of the 18 cells), holds in 9/9 GPT-2 cells, with ratios of 1.9–18×, and in 9/9 Qwen cells, so **H3-KL holds in 18/18 cells**. At one fixed general-text KL, random directions spread several times more than under a unit projection.

**GPT-2, unit projection (α = 1)** (16 directions per cell; the same vectors at block 11 as under matched cost):

| block | control | mean effect | mean \|effect\| | median SE | τ | Q | p |
|---|---|---|---|---|---|---|---|
| 1 | orth | -0.0006 | 0.0013 | 0.0008 | 0.0008 | 38.4 | 0.0009 |
| 1 | support-avoid | +0.0001 | 0.0007 | 0.0006 | 0.0006 | 30.8 | 0.0094 |
| 1 | random-mask | -0.0011 | 0.0030 | 0.0013 | 0.0019 | 72.8 | < 10⁻⁵ |
| 6 | orth | +0.0002 | 0.0012 | 0.0007 | 0.0010 | 56.3 | < 10⁻⁵ |
| 6 | support-avoid | +0.0007 | 0.0019 | 0.0008 | 0.0014 | 101.4 | < 10⁻⁵ |
| 6 | random-mask | +0.0002 | 0.0010 | 0.0005 | 0.0007 | 45.9 | 0.0001 |
| 11 | orth | +0.0001 | 0.0012 | 0.0005 | 0.0008 | 70.9 | < 10⁻⁵ |
| 11 | support-avoid | -0.0007 | 0.0016 | 0.0007 | 0.0010 | 82.5 | < 10⁻⁵ |
| 11 | random-mask | +0.0001 | 0.0006 | 0.0004 | 0.0004 | 37.9 | 0.0009 |

- **H4, J-lens propagation: failed.** The lens (300 prompts; corpus sha256 `7e0015e0…`, lens sha256 `b2d317b7…`) gives A(v) = ‖J_ℓ v‖ with little variation across random directions (coefficient of variation 7% at block 1, 15% at block 6), and A(v) does not predict |effect| under unit projection: Spearman ρ = −0.20 at block 1 (one-sided p = 0.91) and +0.14 at block 6 (p = 0.17). How strongly a random direction propagates to the output does not explain which random directions matter.
- **H5, Qwen3-0.6B.** *Unit projection.* All 9 cells pass the integrity gate (48/48 records each). At block 26, all three controls are heterogeneous. At blocks 1 and 14, 3 of the 6 cells are *not* significant (random-mask at block 14; support-avoid and random-mask at block 1). The pre-registered H3 rule needed a majority, at least 4 of 6, so H3's first part fails for Qwen too, but only narrowly. Unlike GPT-2, Qwen's early and middle layers come close to the J-space expectation that matched-norm random edits are null. *KL-matched.* All 9 cells are heterogeneous at p < 0.005, and 8 of 9 pass the Bonferroni threshold (table below). At a fixed KL budget, Qwen behaves like GPT-2.

**Qwen3-0.6B, KL-matched at 0.10 nats/position (addendum B)**: 144/144 directions matched (α 0.7–16.5). Block 14 seeds 1–7 ran on the desktop; block 14 seeds 8–16 and blocks 1 and 26 ran on the second host (§5.6). Directions generated on the second host equal their desktop-generated P1 counterparts only up to float32 rounding (§5.6).

| block | control | mean effect | mean \|effect\| | τ (KL) | τ (P1) | ratio | Q | p | mean α |
|---|---|---|---|---|---|---|---|---|---|
| 1 | orth | -0.0057 | 0.0178 | 0.0126 | 0.00207 | 6.1× | 50.5 | < 10⁻⁴ | 7.9 |
| 1 | support-avoid | -0.0098 | 0.0166 | 0.0146 | 0.00083 | 17.6× | 44.1 | 0.0001 | 8.5 |
| 1 | random-mask | -0.0007 | 0.0126 | 0.0111 | 0.00000 | — (τ(P1) = 0) | 39.6 | 0.0005 | 8.4 |
| 14 | orth | -0.0075 | 0.0103 | 0.0041 | 0.00107 | 3.8× | 58.1 | < 10⁻⁵ | 6.0 |
| 14 | support-avoid | -0.0086 | 0.0098 | 0.0032 | 0.00113 | 2.8× | 65.8 | < 10⁻⁵ | 5.7 |
| 14 | random-mask | -0.0027 | 0.0061 | 0.0030 | 0.00012 | 24.1× | 32.7 | 0.0049 | 8.1 |
| 26 | orth | -0.0039 | 0.0058 | 0.0037 | 0.00071 | 5.2× | 47.6 | < 10⁻⁴ | 5.9 |
| 26 | support-avoid | -0.0057 | 0.0088 | 0.0063 | 0.00144 | 4.4× | 118.8 | < 10⁻⁵ | 5.3 |
| 26 | random-mask | +0.0009 | 0.0057 | 0.0042 | 0.00064 | 6.7× | 47.5 | < 10⁻⁴ | 7.2 |


**Qwen3-0.6B, unit projection (α = 1)** (16 directions per cell):

| block | control | mean effect | mean \|effect\| | median SE | τ | Q | p |
|---|---|---|---|---|---|---|---|
| 1 | orth | +0.0001 | 0.0026 | 0.0020 | 0.0021 | 41.8 | 0.0002 |
| 1 | support-avoid | +0.0001 | 0.0022 | 0.0020 | 0.0008 | 18.0 | 0.2630 |
| 1 | random-mask | +0.0001 | 0.0011 | 0.0017 | < 10⁻⁴ | 11.5 | 0.7178 |
| 14 | orth | -0.0004 | 0.0013 | 0.0007 | 0.0011 | 52.3 | < 10⁻⁴ |
| 14 | support-avoid | -0.0005 | 0.0014 | 0.0007 | 0.0011 | 53.5 | < 10⁻⁵ |
| 14 | random-mask | -0.0001 | 0.0006 | 0.0007 | 0.0001 | 15.6 | 0.4082 |
| 26 | orth | -0.0005 | 0.0011 | 0.0006 | 0.0007 | 49.5 | < 10⁻⁴ |
| 26 | support-avoid | -0.0014 | 0.0022 | 0.0006 | 0.0014 | 110.8 | < 10⁻⁵ |
| 26 | random-mask | +0.0002 | 0.0010 | 0.0005 | 0.0006 | 44.0 | 0.0001 |

- **Exploratory.** *Control differences.* Under the KL condition, which has no target confound, we ran paired sign-flip tests between every pair of controls, on the signed and absolute effect, in all 6 model × block cells: 36 tests. None reaches p < 0.05, against about 1.8 expected by chance. Avoiding the source's top-512 support (support-avoid) or confining the direction to a random third of the coordinates (random-mask) leaves its effect indistinguishable from an unconstrained random direction (orth). Feature correlations: for each block we correlated 4 direction features (logit-lens gender readout, activation gender contrast, projection size, α) with the signed and absolute effect, both pooled and per control. Only one feature was motivated in advance, projection size `proj_rms` = sqrt(E[(h·v)²]) at the edited positions, which is how much of the activation the edit removes. It correlates positively with |effect| under unit projection at all three blocks: ρ = +0.32 (block 1, p = 0.026), +0.47 (block 6, p = 0.001), +0.35 (block 11, p = 0.016). Only block 6 survives a Bonferroni correction across the 24 pooled absolute-effect tests. No other feature shows a consistent relationship. The consistent sign suggests that a random direction's effect is governed by how much activation lies along it, not by how it propagates (H4). This is a hypothesis for future work, not a result.

**Reading the envelope-alignment numbers.** `abs_cos_to_source`, the cosine between |v| and |source|, drops from about 0.62 (orth) to about 0.09 (support-avoid) *by construction*: support-avoid is exactly zero where |source| is large. It checks that the construction worked. It is not a finding.

### 5.6 Compute hosts and cross-host reproducibility

Most runs used one desktop: an AMD Radeon RX 7800 XT with ROCm 7.2 and torch 2.9.1, in float32. The Qwen3-0.6B KL-matched cells at block 14 seeds 8–16 and at blocks 1 and 26 ran on a second machine: an NVIDIA RTX 5080 Laptop GPU under WSL2 with CUDA 13.0 and torch 2.14.1, also float32, with transformers and the other libraries pinned to the same versions. The code, configs, source directions and model weights were transferred as checksummed archives and verified by hash on arrival. **Cross-host check.** One finished desktop seed (Qwen block 14, seed 1, all 3 controls) was re-run from scratch on the second host. The results agree to numerical precision. Matched α was identical to the 4th decimal (12.0587, 12.7819, 16.4459), achieved KL to the 5th, and effect estimates differed by at most 1.2·10⁻⁶ against SEs of about 10⁻². The direction hashes did *not* match, however. Both hosts run NumPy 2.4.2 with the same OpenBLAS build, which dispatches different kernels on the two CPUs (AMD Ryzen 9 7900X vs Intel Core Ultra 9 275HX). The resulting float32 vectors differ in 8–354 of 1,024 entries, by at most 12 ULP (≤ 1.5·10⁻⁸), with cosine 1.0. Regenerating a direction on the same host reproduced every hash, even seven months later (§5.5). Regenerating it on different hardware cannot. This is the silent-regeneration threat of §3.1 occurring for real, and the semantic-hash check flagged it. The vectors the protocol pins are therefore the stored ones, and a regenerated vector must never be substituted. Each laptop-run cell's directions were generated, hashed and stored on the laptop, and each result is checked against its own stored vectors.

## 6. Discussion

**Relation to the J-space results.** Anthropic's interpretability team reports that matched-norm random perturbations leave Claude models' behaviour near baseline, while edits confined to a small "J-space" of verbalizable directions have strong effects (Anthropic, 2026). Our P1 condition is the closest analogue that open models allow. In the open models we tested, matched-size random edits are *not* reliably null. Under unit projection, random directions have real, direction-specific effects in all 9 GPT-2 cells and in 6 of 9 Qwen cells. Qwen's three nulls sit at its early and middle blocks, the case closest to the J-space finding. A fixed KL budget, a stronger and better-defined size match than either norm or per-seed PLL cost, makes random edits heterogeneous everywhere (17/18 cells past Bonferroni, 18/18 at p < 0.005). How strongly a direction propagates to the output (J-lens A(v)) does not predict which random directions matter. The only feature that tracked their effect, exploratorily, was how much activation lies along the direction. The comparison has limits. Our edits remove a component (h ← h − α(h·v)v) and theirs mostly add one, our models are 0.1–0.6B rather than frontier-scale, and our readout is a small log-probability bias metric rather than behaviour. Our results therefore do not contradict theirs. They show that "a random direction of the same size" is not a safe null by default, and that how the size is matched decides how unsafe it is.

**What this means for intervention studies.** A random control is a null baseline only if it is shown to be one, under the size-matching the study actually uses. Here, single random directions produced bias shifts as large as many claimed effects (mean |effect| up to 0.02 nats per position under KL matching), with the sign varying by direction. A study that compares one learned direction against one random direction can therefore find a "significant" difference that is entirely control noise. We recommend that intervention studies (i) state which quantity their controls are matched on (norm, a task cost, or a general-text KL) and why; (ii) compare against a *distribution* of random directions, reporting the between-direction spread τ next to the mean; (iii) check that their uncertainty estimates are calibrated (H1); and (iv) pin every vector they apply, because a regenerated vector can differ silently, even on another machine (§5.6). Evaluation-awareness research is one place this matters directly. The J-space work locates representations of "this is an evaluation" and shows that ablating them changes behaviour. Studies that intervene on such directions, for example to explain behaviour differences between announced and unannounced evaluations of LLM ethical reasoning, as in the MoralMatrix project, need both the provenance guarantees of Section 3 and controls whose size-matching is stated and justified.

**The same failure class outside direction vectors.** Any cached derived artifact that is regenerated or checked against itself has the failure modes of Section 4. Compiled GPU code objects cached between builds are one example: a stale cache produces silently wrong outputs unless it is verified against a reference fixed before the change.

## 7. Limitations

- One bias metric and one family of templates. Two small models (GPT-2 small, Qwen3-0.6B).
- The held-out set is small (5% of blocks), and all seeds share it. Per-direction CIs are therefore correlated, which is why heterogeneity is tested with Q, with H1 checking the SEs it relies on.
- The J-lens is fitted on 300 paragraphs, fewer than the paper's 1,000, although its authors report that about 100 is usable.
- The layer-11 GPT-2 source direction was learned in the wrong space (Section 5.1).
- The gates establish artifact identity, not causal validity. The support-avoid invariants establish no leakage onto the source's top-k coordinates, not the absence of other pathways.
- The pinned ledger is only as trustworthy as its git history.

## 8. Reproducing

```bash
./run_mc16_v2_randmask.sh        # GPT-2 L11, matched cost, 16 seeds x 3 controls (legacy targets)
JLENS_SRC=<jacobian-lens checkout @ 581d398> ./run_followups.sh   # H1, P1 x 3 layers, J-lens, MC at L6 and L1
./run_qwen_followups.sh          # H5: Qwen3-0.6B source directions, P1 and MC at L1/14/26
.venv/bin/python scripts/check_direction_integrity.py --seed-glob '<sweep>/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --require-hermetic --check-canonical --strict \
  --expect-files 16 --expect-controls orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512
.venv/bin/python scripts/analyze_mc_controls.py --seed-glob '<sweep>/seed*.matched_cost_summary.json' --controls <same three>
.venv/bin/python scripts/fault_injection.py                        # Section 4 table
./scripts/ci_gates.sh            # integrity gate, fault-injection regression, pinned-ledger check
```

Dependencies are frozen in `requirements-lock.txt`. Run summaries embed `runtime_meta` (UTC time, Python and NumPy versions, hostname).

## References

- Anthropic (2026). Verbalizable representations form a global workspace in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2026/workspace/
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
