# Pin the Vector, Test the Control: Provenance Gates and Random-Direction Baselines for Activation Interventions

Panteleimon-Panagiotis Christou (independent)

Code, pre-registration and pinned ledgers: https://github.com/Pantelis23/direction-provenance-gates

## Abstract

Claims about activation interventions are claims about specific direction vectors and about the random directions used as controls. We test both. First, we give a small provenance protocol that stores each direction as a hashed, hermetically copied artifact, and evaluate it by injecting 13 realistic faults into a real experiment. A hardened self-contained integrity gate catches 12 (the original caught 6). The 13th, a run silently redone end to end, is caught only by a hash ledger pinned outside the run, in git. A common regenerate-then-verify CI step catches 2. Second, in a pre-registered study on GPT-2 and Qwen3-0.6B (3 blocks each, 16 random directions per control and condition), random directions are not null controls. Even a plain unit projection has direction-specific effects on a gender-bias metric in 15 of 18 cells, and when every edit is matched to the same general-text KL budget, all 18 cells are heterogeneous (17 at the pre-registered Bonferroni level). The same holds for additive edits, even ones that move general-text KL by only about 0.001–0.006 (17/18 cells). Support-avoiding and subspace-matched random directions behave no differently from unconstrained ones, and Jacobian-lens propagation does not predict which directions matter. Finally, regenerating directions reproduced every hash on the same machine seven months later, but not on different hardware, where vectors shifted by a few float32 ULPs while results agreed to 10⁻⁶. Stored vectors, not regenerated ones, are what should be pinned.

## 1. Introduction

A direction-based intervention edits a hidden state h along a unit vector d. Claims about such interventions are claims about a specific d: "projecting out d reduces bias", "a random direction of equal cost does not". Those claims are only checkable if the exact d used in each run can be recovered later. In practice d is often regenerated from a seed, loaded from a path relative to whatever the working directory happens to be, or reconstructed by a later "backfill" script from generator code that may have changed.

None of these failure modes crash anything. The experiment still runs, the numbers still look plausible, and nothing tells you that the vector in the table is not the vector that produced it.

This paper makes three contributions. All code, the pre-registration with its dated addenda, the pinned ledgers and the fault-injection harness are public at https://github.com/Pantelis23/direction-provenance-gates.

1. **A provenance contract for direction vectors** (Section 3): canonical float32 unit vectors, a semantic hash of the payload, hermetic copies inside each run's output, and a ledger of hashes pinned in version control.
2. **A fault-injection evaluation of the gates** (Section 4): 13 injected faults, showing what a self-contained integrity gate can and cannot catch, and that a common CI pattern (regenerate the manifest, then verify it) catches almost nothing.
3. **A pre-registered test of the random-direction control** (Section 5): in GPT-2 and Qwen3-0.6B at three depths each, comparing a unit projection with edits matched to a fixed general-text KL budget, with every vector run through the gates of Section 3. Random directions turn out not to be null, and the reason is not how far they propagate.

## 2. Related work

**Direction-based interventions.** Linear directions in activation space are used to steer generation (Turner et al., 2023, *Activation Addition*; Li et al., 2023, *Inference-Time Intervention*; Zou et al., 2023, *Representation Engineering*) and to remove concepts (Ravfogel et al., 2020, *INLP*; Elazar et al., 2021, *Amnesic Probing*; Belrose et al., 2023, *LEACE*). Gender directions in embeddings go back at least to Bolukbasi et al. (2016). These works establish methods; none of them specifies how the resulting vectors should be stored and verified across runs.

**Controls for interventions.** Random-direction baselines are the standard control for directional effects, size-matched either on norm or on some behavioural cost. Anthropic (2026) report that matched-norm random perturbations leave behaviour near baseline, while edits in a small subspace of verbalizable directions do not. Section 5 tests when the random-control premise holds in open models.

**Reproducibility tooling.** Experiment trackers and data-versioning tools (e.g. MLflow, DVC) and reporting standards (Pineau et al., 2021; Mitchell et al., 2019; Gebru et al., 2021) record configurations, code versions and datasets. Our protocol is narrower and complementary: it pins the exact vector payloads that an intervention applied, and it is evaluated against injected faults, not just described.

## 3. Protocol

### 3.1 Threat model

We target accidental, silent integrity failures:

| threat | example |
|---|---|
| silent regeneration | seed handling or generator code changes, or the same code on different hardware (§5.5); a "rerun" produces a different d |
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

## 5. Are random control directions null?

Random directions are the default control for directional interventions: if a learned direction d changes behaviour and a random direction "of the same size" does not, the effect is attributed to d. We test that premise in two models, at three depths each, under three ways of making a random edit "the same size". The hypotheses and decision rules were pre-registered before the runs they concern: `docs/prereg_2026-09-28.md`, committed at `dfb4e33`, with the Qwen addendum at `d490967`, the KL-matched addendum at `1597fa6`, and the additive-steering addendum at `b8b1c63`. Every vector reported here passed the gates of Section 3.

### 5.1 Setup

**Data.** Sentences follow templates such as "{name} works as {article} {job}." for 5 gendered name pairs (John/Mary, Michael/Jennifer, …), 56 occupations and 8 templates. Each is paired with a neutral baseline ("{name} works as a person.").

**Bias metric.** For a name pair, the bias is the difference in average token log-probability between the male-name and the female-name sentence, minus the same difference for the baseline template. An intervention's effect is `abs_red = |bias_before| − |bias_after|`, in nats per token. It is averaged over held-out (template, job) blocks (a fixed 5% split), and positive means the bias magnitude shrank. CIs are 95% block-bootstrap intervals (5,000 resamples).

**Intervention.** At one block, on non-name tokens: h ← h − α (h·v) v, applied to the block's output (the residual stream).

**Source directions.** Per block, a ridge-regression direction (λ = 1) predicts a name pair's baseline-adjusted bias from the difference of the two sentences' mean-pooled hidden states. It was trained with a 30% block holdout using the same block hash as the evaluation split, so the evaluation blocks are unseen. No control ever applies the source; it only defines what the controls are orthogonal to and which coordinates they avoid.

**Controls.** One direction per seed (1–16) and control:
- `orth`: Gaussian random, orthogonalised to the source.
- `support-avoid`: Gaussian random on the coordinates *outside* the source's top-512 |coordinates|, orthogonalised to the source. In GPT-2 that leaves 256 of 768 coordinates.
- `random-mask`: the same construction with 512 coordinates masked uniformly at random. It matches support-avoid's dimensionality but not its avoidance.

**Models and blocks.** GPT-2 small (12 blocks, d = 768) at blocks 1, 6 and 11. Qwen3-0.6B (28 blocks, d = 1024, float32, weights sha256 `f47f7117…`) at blocks 1, 14 and 26.

**A disclosed flaw.** In Hugging Face models the last entry of `hidden_states` has the final normalisation applied. The original GPT-2 block-11 source direction was learned on that post-`ln_f` state but is applied to the pre-`ln_f` block output, where `ln_f`'s per-coordinate gain ranges from 0.004 to 17.4. At block 11, "orthogonal to the source" and "the source's top-512 support" are therefore defined in the wrong space. This does not affect whether random directions are null, but it weakens any support-avoid interpretation there. All other source directions are learned and applied in the same space, which is why Qwen uses block 26 rather than its last block, 27.

### 5.2 Three ways to make a random edit "the same size"

- **Unit projection (P1).** α = 1: a true projection of v out of the residual stream, with no calibration. This is the closest analogue to the matched-norm controls in Anthropic's J-space work (Section 6).
- **KL-matched (KL), the primary matched condition.** α is calibrated per direction so that the edit costs exactly the same general-language damage everywhere: mean next-token KL(original ‖ edited) = 0.10 nats per position (±5%), over every position of 100 WikiText-103 test paragraphs with the edit applied at all positions. The target is the same for every model, block, seed and control. It was chosen from a cost-only pilot on seeds 101–102, before any effect was measured (addendum B). The KL-to-original criterion follows heretic (p-e-w/heretic @ `3521f86`), which scores only the first generated token. The corpus is disjoint from the bias sentences, and KL is zero only for a no-op edit.
- **PLL-matched (MC), pre-registered but flawed, secondary.** α is calibrated so that the drop in log-probability of the bias sentences themselves matches a per-seed target. Two flaws surfaced (addendum B). This cost counts only log-prob *drops*, so on 3 of 16 seeds at GPT-2 blocks 1 and 6 the target fell below the matching tolerance and α = 0 counted as "matched" (15 vacuous matches in all). Targets also vary 10–60× across seeds, which confounds across-direction spread with seed-level cost (|effect| vs target: ρ = +0.43, p = 0.007 at block 1). We report MC only at GPT-2 block 11, where legacy targets left no vacuous matches. The Qwen MC run was stopped after 4 seeds.

Matched α values are well above 1 (KL: 0.7–16.5; MC at block 11: 1.05–29.8), so matched edits reverse and amplify the component along v rather than removing it.

**Additive edits** (addendum C). Projections *remove* a component; steering and the J-space controls mostly *add* one. With the same random directions, we therefore also ran h ← h + α v:
- **ADD-N (matched displacement):** one fixed α per block, equal to the pilot median of E|h·v| on WikiText. That is the typical displacement a unit projection makes along a random direction, so ADD-N is P1's additive twin. It is a gentle edit: KL is about 0.001–0.006 nats per position, interpolated from the pilot's α grid.
- **ADD-KL:** α calibrated to the same fixed KL = 0.10 nats per position as the KL condition, with per-block α ceilings fixed from a cost-only pilot.

### 5.3 Statistics

Per control and condition: the mean effect over the 16 directions with a seed-bootstrap CI; Cochran's Q across directions, using each direction's block-bootstrap SE, with a Monte Carlo p-value (floor 5·10⁻⁶) under the null that all directions share one effect; and the DerSimonian–Laird between-direction SD τ. Between controls: paired sign-flip permutation tests and seed-bootstrap CIs for τ ratios (`scripts/analyze_mc_controls.py`). Q is only as good as the SEs, so H1 checks them directly.

For GPT-2 blocks 1 and 6 we also fit a Jacobian lens (anthropics/jacobian-lens @ `581d398`) on 300 WikiText-103 paragraphs of up to 128 tokens. J_ℓ is the average Jacobian of the final residual with respect to the block-ℓ residual. For a unit direction v, A(v) = ‖J_ℓ v‖ measures how strongly a perturbation along v propagates to the output. At GPT-2 block 11 nothing lies between the edit and `ln_f`, so the lens reduces to the logit lens and A(v) carries no information.

### 5.4 Results

| hypothesis | prediction | outcome |
|---|---|---|
| H1 | block-bootstrap SEs are calibrated (pooled Q/df within its null band) | **held**: 1.046, band 0.775–1.253 |
| H3, part 1 | under P1, most cells at early/middle blocks show no heterogeneity | **failed**: GPT-2 0/6 null; Qwen 3/6 null (4 needed) |
| H2-KL | under KL, all 18 cells heterogeneous at p < 0.05/18 | **failed narrowly**: 17/18 (the 18th at p = 0.0049) |
| H3-KL | τ(KL) > τ(P1) in a majority of the 18 cells | **held**: 18/18 |
| H4 | J-lens propagation A(v) predicts \|effect\| under P1 (ρ > 0) | **failed**: ρ = −0.20 (block 1), +0.14 (block 6) |
| H5 | Qwen3-0.6B replicates H2 and H3 | split into the per-model rows above (Qwen's cells are counted in H3, H2-KL and H3-KL). Its PLL-matched part was superseded with MC by addendum B. |
| H6 | ADD-N (P1-sized additive edits) is null in a majority of the 18 cells | **failed**: 17/18 heterogeneous at p < 0.05 |
| H7 | ADD-KL heterogeneous at p < 0.05/18 in all 18 cells | **failed narrowly**: 15/18 (all 18 at p < 0.05) |

**Integrity and reproducibility.** Fifteen sweeps of 48 records each (GPT-2: P1, KL and MC; Qwen: P1 and KL; three blocks each) pass the hardened gate with `--expect-files 16` and all three controls required: 720 records, none with an issue. Regenerating the 32 original orth and support-avoid directions from their seeds on the same machine, seven months after the legacy sweep, reproduced all 32 hashes. With the legacy cost targets pinned, it also reproduced every effect estimate exactly (maximum difference 0.0). Section 5.5 covers what happens on other hardware.

**H1: the SEs are calibrated.** Nine fixed directions (GPT-2 block 11, seeds 1–3 × 3 controls) were each re-evaluated on 16 held-out splits, where their true effect is constant. Per-direction Q/df ranged from 0.41 to 1.50, with pooled Q/df = 1.046, inside the 95% band for calibrated SEs. So a fixed direction varies across splits about as much as its SE predicts, while *different* random directions differ by up to several times that (Q/df up to about 8 in the heterogeneous cells below). No correction is applied.

**Unit projection: random directions are not null.** Even a plain projection, with no amplification, shifts the bias metric by a real, direction-specific amount in all 9 GPT-2 cells (p ≤ 0.009) and in 6 of 9 Qwen cells. Qwen's three nulls are random-mask at block 14 and support-avoid and random-mask at block 1, so its early and middle blocks come closest to a null control. The effects are small, with mean |effect| of 0.0006–0.003 nats, but several times their SEs.

**GPT-2, unit projection (α = 1)**, 16 directions per cell:

| block | control | mean effect | mean \|effect\| | median SE | τ | Q | p |
|---|---|---|---|---|---|---|---|
| 1 | orth | −0.0006 | 0.0013 | 0.0008 | 0.0008 | 38.4 | 0.0009 |
| 1 | support-avoid | +0.0001 | 0.0007 | 0.0006 | 0.0006 | 30.8 | 0.0094 |
| 1 | random-mask | −0.0011 | 0.0030 | 0.0013 | 0.0019 | 72.8 | < 10⁻⁵ |
| 6 | orth | +0.0002 | 0.0012 | 0.0007 | 0.0010 | 56.3 | < 10⁻⁵ |
| 6 | support-avoid | +0.0007 | 0.0019 | 0.0008 | 0.0014 | 101.4 | < 10⁻⁵ |
| 6 | random-mask | +0.0002 | 0.0010 | 0.0005 | 0.0007 | 45.9 | 0.0001 |
| 11 | orth | +0.0001 | 0.0012 | 0.0005 | 0.0008 | 70.9 | < 10⁻⁵ |
| 11 | support-avoid | −0.0007 | 0.0016 | 0.0007 | 0.0010 | 82.5 | < 10⁻⁵ |
| 11 | random-mask | +0.0001 | 0.0006 | 0.0004 | 0.0004 | 37.9 | 0.0009 |

**Qwen3-0.6B, unit projection (α = 1)**:

| block | control | mean effect | mean \|effect\| | median SE | τ | Q | p |
|---|---|---|---|---|---|---|---|
| 1 | orth | +0.0001 | 0.0026 | 0.0020 | 0.0021 | 41.8 | 0.0002 |
| 1 | support-avoid | +0.0001 | 0.0022 | 0.0020 | 0.0008 | 18.0 | 0.2630 |
| 1 | random-mask | +0.0001 | 0.0011 | 0.0017 | < 10⁻⁴ | 11.5 | 0.7178 |
| 14 | orth | −0.0004 | 0.0013 | 0.0007 | 0.0011 | 52.3 | < 10⁻⁴ |
| 14 | support-avoid | −0.0005 | 0.0014 | 0.0007 | 0.0011 | 53.5 | < 10⁻⁵ |
| 14 | random-mask | −0.0001 | 0.0006 | 0.0007 | 0.0001 | 15.6 | 0.4082 |
| 26 | orth | −0.0005 | 0.0011 | 0.0006 | 0.0007 | 49.5 | < 10⁻⁴ |
| 26 | support-avoid | −0.0014 | 0.0022 | 0.0006 | 0.0014 | 110.8 | < 10⁻⁵ |
| 26 | random-mask | +0.0002 | 0.0010 | 0.0005 | 0.0006 | 44.0 | 0.0001 |

**KL-matched: heterogeneous everywhere.** All 288 directions matched the KL target (achieved 0.095–0.105), and none needed α = 0. Every one of the 18 cells is heterogeneous at p < 0.005, and 17 pass the pre-registered Bonferroni threshold (p < 0.05/18). The exception is Qwen block 14 random-mask, at p = 0.0049, so H2-KL fails narrowly. In every cell the between-direction spread is larger than under unit projection (τ ratio at least 1.9 in every cell, up to 24, and unbounded in the one cell where τ(P1) = 0), so H3-KL holds 18/18. At one fixed general-language cost, random directions move the bias metric by up to 0.02 nats on average, with signs that vary by direction.

**GPT-2, KL-matched at 0.10 nats/position.** These are the same vectors as under P1, with identical hashes:

| block | control | mean effect | mean \|effect\| | τ (KL) | τ (P1) | ratio | Q | p | mean α |
|---|---|---|---|---|---|---|---|---|---|
| 1 | orth | +0.0014 | 0.0120 | 0.0085 | 0.00075 | 11.3× | 76.1 | < 10⁻⁵ | 7.0 |
| 1 | support-avoid | +0.0027 | 0.0135 | 0.0101 | 0.00056 | 18.1× | 57.9 | < 10⁻⁵ | 9.2 |
| 1 | random-mask | +0.0083 | 0.0197 | 0.0094 | 0.00193 | 4.9× | 71.4 | < 10⁻⁵ | 7.1 |
| 6 | orth | +0.0002 | 0.0052 | 0.0025 | 0.00098 | 2.5× | 49.9 | < 10⁻⁴ | 4.4 |
| 6 | support-avoid | +0.0002 | 0.0045 | 0.0026 | 0.00140 | 1.9× | 94.4 | < 10⁻⁵ | 3.2 |
| 6 | random-mask | +0.0027 | 0.0056 | 0.0027 | 0.00072 | 3.8× | 50.5 | < 10⁻⁴ | 6.1 |
| 11 | orth | −0.0011 | 0.0046 | 0.0035 | 0.00078 | 4.5× | 71.7 | < 10⁻⁵ | 5.5 |
| 11 | support-avoid | −0.0009 | 0.0035 | 0.0031 | 0.00101 | 3.1× | 63.8 | < 10⁻⁵ | 3.7 |
| 11 | random-mask | +0.0007 | 0.0035 | 0.0024 | 0.00040 | 6.0× | 43.5 | 0.0001 | 6.4 |

**Qwen3-0.6B, KL-matched at 0.10 nats/position.** Block 14 seeds 1–7 ran on the desktop; block 14 seeds 8–16 and blocks 1 and 26 ran on a second machine (§5.5). Vectors generated there equal their desktop-generated P1 counterparts only up to float32 rounding (cosine 1.0):

| block | control | mean effect | mean \|effect\| | τ (KL) | τ (P1) | ratio | Q | p | mean α |
|---|---|---|---|---|---|---|---|---|---|
| 1 | orth | −0.0057 | 0.0178 | 0.0126 | 0.00207 | 6.1× | 50.5 | < 10⁻⁴ | 7.9 |
| 1 | support-avoid | −0.0098 | 0.0166 | 0.0146 | 0.00083 | 17.6× | 44.1 | 0.0001 | 8.5 |
| 1 | random-mask | −0.0007 | 0.0126 | 0.0111 | 0.00000 | — (τ(P1) = 0) | 39.6 | 0.0005 | 8.4 |
| 14 | orth | −0.0075 | 0.0103 | 0.0041 | 0.00107 | 3.8× | 58.1 | < 10⁻⁵ | 6.0 |
| 14 | support-avoid | −0.0086 | 0.0098 | 0.0032 | 0.00113 | 2.8× | 65.8 | < 10⁻⁵ | 5.7 |
| 14 | random-mask | −0.0027 | 0.0061 | 0.0030 | 0.00012 | 24.1× | 32.7 | 0.0049 | 8.1 |
| 26 | orth | −0.0039 | 0.0058 | 0.0037 | 0.00071 | 5.2× | 47.6 | < 10⁻⁴ | 5.9 |
| 26 | support-avoid | −0.0057 | 0.0088 | 0.0063 | 0.00144 | 4.4× | 118.8 | < 10⁻⁵ | 5.3 |
| 26 | random-mask | +0.0009 | 0.0057 | 0.0042 | 0.00064 | 6.7× | 47.5 | < 10⁻⁴ | 7.2 |

**Additive edits: still not null.** All 12 additive sweeps (576 records) pass the hardened gate. These runs all used the second host (§5.5). *ADD-N.* Adding a random direction the size of a unit projection's displacement barely changes general text (KL ≈ 0.001–0.006 nats per position, interpolated from the pilot's α grid), yet 17 of 18 cells are heterogeneous at p < 0.05, so H6 fails. The only null cell is Qwen block 1 random-mask, the same cell that was null under P1. The effects are small (mean |effect| 0.0005–0.004) but about 1.5–3× their SEs in the heterogeneous cells. *ADD-KL.* All 288 directions matched (achieved KL 0.095–0.105). All 18 cells are heterogeneous at p < 0.05 and 15 at the pre-registered Bonferroni level, so H7 fails narrowly. The three exceptions are all at block 1 (GPT-2 orth; Qwen support-avoid and random-mask), where additive effects are largest (mean |effect| 0.012–0.022) but also noisiest. At the same KL, additive random edits spread at least as much as projections: τ(ADD-KL)/τ(KL) has a median of 1.5 and is above 1 in 16 of 18 cells. Switching from removing a component to adding one therefore does not make random directions null.

**ADD-N (matched displacement, α fixed per block):**

| model | block | control | α | mean effect | mean \|effect\| | median SE | τ | Q | p |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 1 | orth | 2.797 | +0.0011 | 0.0025 | 0.0017 | 0.0021 | 45.6 | < 10⁻⁴ |
| GPT-2 | 1 | support-avoid | 2.797 | +0.0001 | 0.0028 | 0.0018 | 0.0021 | 42.2 | 0.0002 |
| GPT-2 | 1 | random-mask | 2.797 | +0.0004 | 0.0041 | 0.0020 | 0.0040 | 78.1 | < 10⁻⁵ |
| GPT-2 | 6 | orth | 3.955 | +0.0000 | 0.0016 | 0.0008 | 0.0014 | 63.5 | < 10⁻⁵ |
| GPT-2 | 6 | support-avoid | 3.955 | +0.0001 | 0.0020 | 0.0009 | 0.0020 | 97.0 | < 10⁻⁵ |
| GPT-2 | 6 | random-mask | 3.955 | +0.0004 | 0.0017 | 0.0009 | 0.0017 | 72.7 | < 10⁻⁵ |
| GPT-2 | 11 | orth | 8.455 | +0.0002 | 0.0006 | 0.0003 | 0.0006 | 94.1 | < 10⁻⁵ |
| GPT-2 | 11 | support-avoid | 8.455 | +0.0002 | 0.0006 | 0.0003 | 0.0006 | 102.2 | < 10⁻⁵ |
| GPT-2 | 11 | random-mask | 8.455 | −0.0000 | 0.0005 | 0.0003 | 0.0005 | 65.8 | < 10⁻⁵ |
| Qwen | 1 | orth | 0.221 | +0.0003 | 0.0020 | 0.0013 | 0.0018 | 42.2 | 0.0002 |
| Qwen | 1 | support-avoid | 0.221 | +0.0004 | 0.0016 | 0.0014 | 0.0015 | 32.4 | 0.0058 |
| Qwen | 1 | random-mask | 0.221 | −0.0002 | 0.0011 | 0.0013 | 0.0005 | 18.0 | 0.2586 |
| Qwen | 14 | orth | 2.834 | −0.0005 | 0.0025 | 0.0013 | 0.0025 | 72.3 | < 10⁻⁵ |
| Qwen | 14 | support-avoid | 2.834 | −0.0002 | 0.0021 | 0.0014 | 0.0021 | 48.7 | < 10⁻⁴ |
| Qwen | 14 | random-mask | 2.834 | −0.0003 | 0.0023 | 0.0012 | 0.0020 | 54.9 | < 10⁻⁵ |
| Qwen | 26 | orth | 8.969 | −0.0000 | 0.0007 | 0.0002 | 0.0007 | 143.3 | < 10⁻⁵ |
| Qwen | 26 | support-avoid | 8.969 | −0.0001 | 0.0009 | 0.0003 | 0.0009 | 180.9 | < 10⁻⁵ |
| Qwen | 26 | random-mask | 8.969 | +0.0001 | 0.0006 | 0.0003 | 0.0008 | 168.1 | < 10⁻⁵ |

**ADD-KL (0.10 nats/position):**

| model | block | control | mean effect | mean \|effect\| | τ | Q | p | mean α |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | 1 | orth | +0.0202 | 0.0216 | 0.0094 | 29.5 | 0.0141 | 15.1 |
| GPT-2 | 1 | support-avoid | +0.0125 | 0.0175 | 0.0112 | 36.4 | 0.0017 | 14.6 |
| GPT-2 | 1 | random-mask | +0.0127 | 0.0182 | 0.0124 | 43.2 | 0.0001 | 14.7 |
| GPT-2 | 6 | orth | +0.0042 | 0.0082 | 0.0071 | 70.7 | < 10⁻⁵ | 19.6 |
| GPT-2 | 6 | support-avoid | +0.0055 | 0.0113 | 0.0095 | 83.6 | < 10⁻⁵ | 20.2 |
| GPT-2 | 6 | random-mask | +0.0062 | 0.0093 | 0.0074 | 64.9 | < 10⁻⁵ | 20.0 |
| GPT-2 | 11 | orth | +0.0014 | 0.0042 | 0.0037 | 91.1 | < 10⁻⁵ | 62.1 |
| GPT-2 | 11 | support-avoid | +0.0021 | 0.0050 | 0.0048 | 110.8 | < 10⁻⁵ | 62.7 |
| GPT-2 | 11 | random-mask | +0.0003 | 0.0038 | 0.0032 | 68.3 | < 10⁻⁵ | 62.1 |
| Qwen | 1 | orth | −0.0056 | 0.0182 | 0.0172 | 60.3 | < 10⁻⁵ | 2.3 |
| Qwen | 1 | support-avoid | −0.0015 | 0.0127 | 0.0099 | 29.0 | 0.0157 | 2.4 |
| Qwen | 1 | random-mask | +0.0090 | 0.0124 | 0.0077 | 25.4 | 0.0440 | 2.4 |
| Qwen | 14 | orth | −0.0021 | 0.0106 | 0.0112 | 73.3 | < 10⁻⁵ | 13.6 |
| Qwen | 14 | support-avoid | −0.0020 | 0.0101 | 0.0091 | 43.4 | 0.0001 | 13.5 |
| Qwen | 14 | random-mask | −0.0036 | 0.0092 | 0.0076 | 37.7 | 0.0010 | 13.5 |
| Qwen | 26 | orth | +0.0010 | 0.0068 | 0.0069 | 132.5 | < 10⁻⁵ | 99.8 |
| Qwen | 26 | support-avoid | +0.0002 | 0.0097 | 0.0095 | 191.8 | < 10⁻⁵ | 99.4 |
| Qwen | 26 | random-mask | +0.0018 | 0.0067 | 0.0078 | 125.1 | < 10⁻⁵ | 98.5 |

**H4: propagation does not explain it.** A(v) barely varies across random directions (coefficient of variation 7% at block 1, 15% at block 6), and it does not predict |effect| under P1 (Spearman ρ = −0.20, one-sided p = 0.91 at block 1; ρ = +0.14, p = 0.17 at block 6). Lens provenance: corpus sha256 `7e0015e0…`, lens sha256 `b2d317b7…`.

**Exploratory.**
- *Control type does not matter.* Under KL, paired sign-flip tests between every pair of controls, on the signed and absolute effect, in all 6 model × block cells (36 tests) give no p < 0.05, against about 1.8 expected by chance. Under ADD-N and ADD-KL the same 36 tests give 1 and 2 hits, again at chance. Avoiding the source's support, or confining a direction to a random third of the coordinates, leaves its effect indistinguishable from an unconstrained random direction.
- *What predicts a direction's effect.* Per block, we correlated 4 direction features (logit-lens gender readout, activation gender contrast, projection size, α) with the signed and absolute effect. The only feature motivated in advance was projection size, `proj_rms` = sqrt(E[(h·v)²]) at the edited positions, i.e. how much activation the edit removes. It correlates positively with |effect| under P1 at all three GPT-2 blocks: ρ = +0.32 (p = 0.026), +0.47 (p = 0.001) and +0.35 (p = 0.016). Only block 6 survives Bonferroni across the 24 pooled absolute-effect tests, and no other feature is consistent. Together with H4, this suggests a random direction's effect depends on how much activation lies along it, not on how far it propagates. That is a hypothesis, not a result.
- *Secondary: PLL-matched at GPT-2 block 11.* All three controls are heterogeneous here too (Q = 55.6–62.6, p < 10⁻⁵), and no pair of controls differs (sign-flip p = 0.13–0.99; τ ratios 0.88–0.94, CIs about 0.4–2.0). Reaching the same PLL cost takes random-mask about twice orth's α. Note that `abs_cos_to_source`, the cosine between |v| and |source|, drops from about 0.62 to 0.09 for support-avoid *by construction*: it checks that the construction worked, not a finding.

| control | mean effect [seed-bootstrap 95% CI] | mean \|effect\| | τ | Q | p | mean α | abs-cos to source |
|---|---|---|---|---|---|---|---|
| orth | +0.0015 [−0.0028, +0.0059] | 0.0061 | 0.0037 | 62.6 | < 10⁻⁵ | 6.4 | 0.62 |
| support-avoid | +0.0002 [−0.0025, +0.0029] | 0.0041 | 0.0035 | 59.0 | < 10⁻⁵ | 4.8 | 0.09 |
| random-mask | −0.0004 [−0.0044, +0.0033] | 0.0061 | 0.0033 | 55.6 | < 10⁻⁵ | 11.3 | 0.37 |

### 5.5 Compute hosts and cross-host reproducibility

Most runs used one desktop: an AMD Radeon RX 7800 XT with ROCm 7.2 and torch 2.9.1, in float32. The Qwen3-0.6B KL-matched cells at block 14 seeds 8–16 and at blocks 1 and 26, and all additive runs, ran on a second machine: an NVIDIA RTX 5080 Laptop GPU under WSL2 with CUDA 13.0 and torch 2.14.1, also float32, with transformers and the other libraries pinned to the same versions. The code, configs, source directions and model weights were transferred as checksummed archives and verified by hash on arrival. **Cross-host check.** One finished desktop seed (Qwen block 14, seed 1, all 3 controls) was re-run from scratch on the second host. The results agree to numerical precision. Matched α was identical to the 4th decimal (12.0587, 12.7819, 16.4459), achieved KL to the 5th, and effect estimates differed by at most 1.2·10⁻⁶ against SEs of about 10⁻². The direction hashes did *not* match, however. Both hosts run NumPy 2.4.2 with the same OpenBLAS build, which dispatches different kernels on the two CPUs (AMD Ryzen 9 7900X vs Intel Core Ultra 9 275HX). The resulting float32 vectors differ in 8–354 of 1,024 entries, by at most 12 ULP (≤ 1.5·10⁻⁸), with cosine 1.0. Regenerating a direction on the same host reproduced every hash, even seven months later (§5.4). Regenerating it on different hardware cannot. This is the silent-regeneration threat of §3.1 occurring for real, and the semantic-hash check flagged it. The vectors the protocol pins are therefore the stored ones, and a regenerated vector must never be substituted. Each laptop-run cell's directions were generated, hashed and stored on the laptop, and each result is checked against its own stored vectors.

## 6. Discussion

**Relation to the J-space results.** Anthropic's interpretability team reports that matched-norm random perturbations leave Claude models' behaviour near baseline, while edits confined to a small "J-space" of verbalizable directions have strong effects (Anthropic, 2026). Our P1 condition is the closest analogue that open models allow. In the open models we tested, matched-size random edits are *not* reliably null. Under unit projection, random directions have real, direction-specific effects in all 9 GPT-2 cells and in 6 of 9 Qwen cells. Qwen's three nulls sit at its early and middle blocks, the case closest to the J-space finding. A fixed KL budget, a stronger and better-defined size match than either norm or per-seed PLL cost, makes random edits heterogeneous everywhere (17/18 cells past Bonferroni, 18/18 at p < 0.005). How strongly a direction propagates to the output (J-lens A(v)) does not predict which random directions matter. The only feature that tracked their effect, exploratorily, was how much activation lies along the direction. The comparison has limits, but one of them is now tested. Their controls mostly *add* a component, and our additive random edits are not null either: 17/18 cells are heterogeneous even at a displacement that moves general-text KL by only about 0.001–0.006. What remains is that our models are 0.1–0.6B rather than frontier-scale, and that our readout is a small log-probability bias metric rather than behaviour. Our results therefore do not contradict theirs. They show that "a random direction of the same size" is not a safe null by default, and that how the size is matched decides how unsafe it is.

**What this means for intervention studies.** A random control is a null baseline only if it is shown to be one, under the size-matching the study actually uses. Here, single random directions produced bias shifts as large as many claimed effects (mean |effect| up to 0.02 nats per position under KL matching), with the sign varying by direction. A study that compares one learned direction against one random direction can therefore find a "significant" difference that is entirely control noise. We recommend that intervention studies (i) state which quantity their controls are matched on (norm, a task cost, or a general-text KL) and why; (ii) compare against a *distribution* of random directions, reporting the between-direction spread τ next to the mean; (iii) check that their uncertainty estimates are calibrated (H1); and (iv) pin every vector they apply, because a regenerated vector can differ silently, even on another machine (§5.5). Evaluation-awareness research is one place this matters directly. The J-space work locates representations of "this is an evaluation" and shows that ablating them changes behaviour. Studies that intervene on such directions, for example to explain behaviour differences between announced and unannounced evaluations of LLM ethical reasoning, need both the provenance guarantees of Section 3 and controls whose size-matching is stated and justified.

**The same failure class outside direction vectors.** Any cached derived artifact that is regenerated or checked against itself has the failure modes of Section 4. Compiled GPU code objects cached between builds are one example: a stale cache produces silently wrong outputs unless it is verified against a reference fixed before the change.

## 7. Limitations

- One bias metric and one family of templates, in two small models (GPT-2 small, 124M; Qwen3-0.6B). Edits are removals (h ← h − α(h·v)v) and additions (h ← h + αv) at a single block; multi-block or learned steering vectors were not tested.
- The held-out set is small (5% of blocks) and shared by all seeds, so per-direction CIs are correlated. That is why heterogeneity is tested with Q, and why H1 checks the SEs it relies on.
- The pre-registered PLL-matched condition was flawed (§5.2) and was replaced by a KL-matched condition in a dated addendum, after the flaw was found but before any KL-matched run. Its results are reported as secondary.
- The GPT-2 block-11 source direction was learned in the wrong space (§5.1).
- Part of the Qwen KL-matched data came from a second machine (§5.5). Results agree across machines to 10⁻⁶ on a re-run seed, but those vectors are not bit-identical to the desktop's.
- The Jacobian lens was fitted on 300 paragraphs, fewer than the original authors' 1,000, although they report that about 100 is usable.
- The gates establish artifact identity, not causal validity. The support-avoid invariants show no leakage onto the source's top-k coordinates, not the absence of other pathways. The pinned ledger is only as trustworthy as its git history.

## 8. Reproducing

```bash
# Gates and the Section 4 table
./scripts/ci_gates.sh                     # integrity gate, fault-injection regression, pinned-ledger check
.venv/bin/python scripts/fault_injection.py

# Section 5 (pre-registration: docs/prereg_2026-09-28.md)
./run_mc16_v2_randmask.sh                 # GPT-2 block 11, PLL-matched, legacy targets pinned
JLENS_SRC=<jacobian-lens @ 581d398> ./run_followups.sh   # H1, P1 x 3 blocks, J-lens fit, PLL-matched at blocks 6/1
./run_qwen_followups.sh                   # Qwen3-0.6B source directions and P1 (its MC part was stopped)
./run_kl_sweeps.sh                        # KL-matched, GPT-2 x 3 blocks, then Qwen x 3 blocks
./run_qwen_kl_laptop.sh                   # the second-machine variant used for part of the Qwen KL data (WSL2/CUDA)
./run_add_laptop.sh                       # additive steering, ADD-N and ADD-KL (addendum C; second machine)

# Check and analyse any sweep
.venv/bin/python scripts/check_direction_integrity.py --seed-glob '<sweep>/seed*.matched_cost_summary.json' \
  --hash-mode both --require-alias --require-hermetic --check-canonical --strict --expect-files 16 \
  --expect-controls orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512
.venv/bin/python scripts/analyze_mc_controls.py --seed-glob '<sweep>/seed*.matched_cost_summary.json' --controls <same three>
.venv/bin/python scripts/calibrate_block_se.py --seeds 1,2,3        # H1
.venv/bin/python scripts/jspace_propagation.py --jlens-src <checkout> --sweep '1=runs/p1_early/seed*.json' --sweep '6=runs/p1_mid/seed*.json'   # H4
```

Dependencies are frozen in `requirements-lock.txt`. Run summaries embed `runtime_meta` (UTC time, Python and NumPy versions, hostname).

## References

- Anthropic (2026). jacobian-lens: companion code for the global workspace paper. Software, https://github.com/anthropics/jacobian-lens, commit `581d398`.
- Anthropic (2026). Verbalizable representations form a global workspace in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2026/workspace/
- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. *NeurIPS*.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS*.
- Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1).
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3).
- Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *TACL*.
- Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12).
- Grattafiori, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783. Model used: Llama-3.2-1B-Instruct, via the mirror `unsloth/Llama-3.2-1B-Instruct` @ `5a8abab4` (weights sha256 `1ff795ff…`).
- Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS*.
- Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer sentinel mixture models. *ICLR*. arXiv:1609.07843. (WikiText-103.)
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.
- Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *JMLR*, 22.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI.
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. *ACL*.
- Turner, A. M., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.
- Weidmann, P. E., et al. (2025–2026). heretic. Software, https://github.com/p-e-w/heretic, commit `3521f86`.
- Yang, A., et al. (2025). Qwen3 technical report. arXiv:2505.09388.
- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.
