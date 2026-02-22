# Direction Provenance + Matched-Cost Causal Editing

This repo contains bias/casual-direction experiments with a reproducibility-first workflow:
- matched-cost causal control sweeps
- hermetic direction artifacts + hashes
- integrity/family gates for publishable runs

## Repository Map

- `configs/`: experiment configurations
- `docs/`: specs and provenance contract
- `scripts/`: run, aggregate, integrity, and manifest tools
- `src/`: core metrics/telemetry utilities
- `runs/`: run outputs and generated manifests
- `paper.md`: methods/results write-up

## Reproducibility Gates

Single command (new artifacts + paper assets):

```bash
cd /home/pantelis/Desktop/Projects/Work/bias_embedding_exp
./scripts/ci_gates.sh
```

Reference docs:
- `docs/provenance.md`
- `README_provenance.md`

## Quickstart

```bash
cd /home/pantelis/Desktop/Projects/Work/bias_embedding_exp
source .venv/bin/activate
python -m pip install -r requirements.txt
source scripts/activate_env.sh
python scripts/run_mvp.py --config configs/mvp.yaml
```

## What This Covers

- Embedding fragility across layers, pooling, and embedding models
- Cross-metric agreement between embedding-based bias and probability-based bias
- Layerwise attribution of bias deltas
- Telemetry: timing, throughput, memory, GPU info, and reproducibility context

## AtlasLM Integration

This project is set up to import from the local AtlasLM repo via `PYTHONPATH`.
Run `source scripts/activate_env.sh` to add:
- `/home/pantelis/Desktop/Projects/Work/AtlasLM`
- `/home/pantelis/Desktop/Projects/Work/AtlasLM/src`

This allows direct reuse of AtlasLM modules without repackaging.

## Data Format (SEAT/WEAT)

`configs/mvp.yaml` expects SEAT/WEAT-style tests under `data.seat_tests`:

- `X`, `Y`: target sentence sets
- `A`, `B`: attribute sentence sets

Each test yields an effect size, permutation p-values, and diagnostics (anisotropy, singular values, conicity).

You can provide sentence lists directly or use templating:

- Direct list: `- "This person is a doctor."`
- Templated: `templates: ["This person is a {t}."]` + `targets: ["doctor", "engineer"]`

When using `pooling: "target"`, each sentence must include a `target` term (or a template `targets` list).

## Data Format (PLL Group Bias)

`data.pll_templates` defines counterfactual group swaps for PLL:

- `templates`: list of template strings (preferred for diversity)
- `group_pairs`: optional paired list `[[A, B], ...]` for matched names
- `groupA`, `groupB`: list or string of group fillers (names)
- `jobs`: list of job entries; each can be `\"engineer\"` or `{job: \"engineer\", article: \"an\"}`
- `baseline_template` + `adjust_by_baseline`: optional baseline adjustment to reduce name priors
- `baseline_templates`: per-template baselines (must align with `templates`)

PLL bias is computed as `avg_logprob(groupA) - avg_logprob(groupB)` per template.
When `adjust_by_baseline` is true, we compute per-name deltas:
`Δ(name) = PLL(job sentence) - PLL(baseline sentence)` and then take group differences.

## Random Split Control

Optional `data.pll_random_split` performs K random equal splits of a name pool and reports the distribution of adjusted biases. Outputs:
- `results_pll_random_split.jsonl`
- summary stats in `results_pll_summary.json`

PLL results are written to `results_pll.jsonl` and a summary to `results_pll_summary.json`.

## Layout

- `configs/` experiment configs
- `docs/` experiment spec, notes, provenance
- `scripts/` entrypoints and gates
- `src/` experiment code
- `runs/` run outputs, manifests, paper assets
- `results/` aggregated results

Per-run outputs:
- `results_embed.jsonl` (SEAT/WEAT)
- `results_pll.jsonl` (PLL)
- `embedding_holm.json` (Holm correction)
- `results_pll_summary.json` (PLL stats)

## Next Steps

- Point `configs/mvp.yaml` to your dataset and model.
- Extend telemetry in `src/telemetry/logger.py` as needed.
