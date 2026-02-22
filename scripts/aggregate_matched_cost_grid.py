#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _pass_flags(
    rec_none: dict,
    rec_rand: dict | None,
    rec_shuf: dict | None,
    rec_orth: dict,
    rec_absm: dict | None = None,
) -> dict[str, object]:
    none_s = rec_none.get("final_summary", {})
    rand_s = (rec_rand or {}).get("final_summary", {})
    shuf_s = (rec_shuf or {}).get("final_summary", {})
    orth_s = rec_orth.get("final_summary", {})
    absm_s = (rec_absm or {}).get("final_summary", {})

    none_abs = none_s.get("mean_abs_reduction_adj")
    none_ci = (none_s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
    rand_abs = rand_s.get("mean_abs_reduction_adj")
    rand_ci = (rand_s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
    orth_abs = orth_s.get("mean_abs_reduction_adj")
    orth_shape = (orth_s.get("direction_shape_stats") or {})
    orth_abs_cos = orth_shape.get("abs_cos_to_source")
    orth_top512_mass = orth_shape.get("top512_mass")
    orth_ci = (orth_s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
    shuf_ci = (shuf_s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
    absm_abs = absm_s.get("mean_abs_reduction_adj")
    absm_ci = (absm_s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]

    none_pass = (
        none_abs is not None
        and none_ci[0] is not None
        and float(none_abs) > 0.0
        and float(none_ci[0]) > 0.0
    )
    delta_rand = (
        (float(none_abs) - float(rand_abs))
        if none_abs is not None and rand_abs is not None
        else None
    )
    shuf_abs = shuf_s.get("mean_abs_reduction_adj")
    delta_shuf = (
        (float(none_abs) - float(shuf_abs))
        if none_abs is not None and shuf_abs is not None
        else None
    )
    delta_absm = (
        (float(none_abs) - float(absm_abs))
        if none_abs is not None and absm_abs is not None
        else None
    )
    delta_orth = (
        (float(none_abs) - float(orth_abs))
        if none_abs is not None and orth_abs is not None
        else None
    )

    def _sig_pos(ci: list[float | None]) -> bool:
        return ci[0] is not None and float(ci[0]) > 0.0

    def _sig_neg(ci: list[float | None]) -> bool:
        return ci[1] is not None and float(ci[1]) < 0.0

    def _sig_two_sided(ci: list[float | None]) -> bool:
        return _sig_pos(ci) or _sig_neg(ci)

    def _not_sig_pos(ci: list[float | None]) -> bool:
        return not _sig_pos(ci)

    rand_pass = _not_sig_pos(rand_ci) if rec_rand is not None else True
    shuf_pass = _not_sig_pos(shuf_ci) if rec_shuf is not None else True
    orth_pass = _not_sig_pos(orth_ci)
    absm_pass = _not_sig_pos(absm_ci) if rec_absm is not None else True
    shuf_stress_pass = (not _sig_two_sided(shuf_ci)) if rec_shuf is not None else True
    absm_stress_pass = (not _sig_two_sided(absm_ci)) if rec_absm is not None else True
    none_beats_random = (delta_rand is not None and float(delta_rand) > 0.0) if rec_rand is not None else True
    none_beats_shuffled = delta_shuf is not None and float(delta_shuf) > 0.0
    none_beats_absm = delta_absm is not None and float(delta_absm) > 0.0
    none_beats_orth = delta_orth is not None and float(delta_orth) > 0.0

    def _cost_err(rec: dict) -> float | None:
        c = rec.get("cost_err")
        if c is not None:
            try:
                return float(c)
            except Exception:
                return None
        s = rec.get("final_summary", {})
        lp = s.get("mean_logprob_change_pair")
        tgt = rec.get("target_cost")
        if lp is None or tgt is None:
            return None
        return abs(max(0.0, -float(lp)) - float(tgt))

    none_cost_err = _cost_err(rec_none)
    random_cost_err = _cost_err(rec_rand or {})
    shuffled_cost_err = _cost_err(rec_shuf or {})
    orth_cost_err = _cost_err(rec_orth)
    absm_cost_err = _cost_err(rec_absm or {})

    def _tol(rec: dict) -> float:
        tol = rec.get("match_tol", 0.002)
        try:
            return float(tol)
        except Exception:
            return 0.002

    def _matched(rec: dict, ce: float | None) -> bool:
        m = rec.get("matched")
        if isinstance(m, bool):
            return m
        tol_f = _tol(rec)
        return ce is not None and ce <= tol_f

    none_matched = _matched(rec_none, none_cost_err)
    random_matched = _matched(rec_rand or {}, random_cost_err) if rec_rand is not None else True
    shuffled_matched = _matched(rec_shuf or {}, shuffled_cost_err) if rec_shuf is not None else True
    orth_matched = _matched(rec_orth, orth_cost_err)
    absm_matched = _matched(rec_absm or {}, absm_cost_err) if rec_absm is not None else True

    # Strict core nulls: ONLY orthogonal_random (deterministic null expectation).
    pass_strict_core = (
        none_pass
        and none_beats_orth
        and none_matched
        and orth_matched
        and orth_pass
    )
    pass_strict_with_shuffled = pass_strict_core and (
        (rec_shuf is None) or (shuffled_matched and shuf_stress_pass)
    )
    pass_strict_with_stress = pass_strict_core and (
        (rec_shuf is None or (shuffled_matched and shuf_stress_pass))
        and (rec_absm is None or (absm_matched and absm_stress_pass))
    )
    pass_strict_with_orth = pass_strict_core
    overall_pass_effect_only = bool(none_pass and none_matched)
    diagnostic_orth_sig_pos = bool(rec_orth is not None and orth_matched and _sig_pos(orth_ci))
    diagnostic_orth_sig_neg = bool(rec_orth is not None and orth_matched and _sig_neg(orth_ci))
    diagnostic_orth_abs_cos_hi = None
    fail_reasons: list[str] = []

    checks = [
        ("none", rec_none, none_cost_err, none_matched),
        ("orthogonal_random", rec_orth, orth_cost_err, orth_matched),
    ]
    if rec_rand is not None:
        checks.append(("random", rec_rand, random_cost_err, random_matched))
    if rec_shuf is not None:
        checks.append(("shuffled", rec_shuf, shuffled_cost_err, shuffled_matched))
    if rec_absm is not None:
        checks.append(("abs_marginal_matched_random_orth", rec_absm, absm_cost_err, absm_matched))
    for ctrl, rec, ce, matched in checks:
        tol_f = _tol(rec)
        if ce is None:
            fail_reasons.append(f"{ctrl}:missing_cost_err")
        elif ce > tol_f:
            fail_reasons.append(f"{ctrl}:cost_err_gt_tol")
        if not matched:
            fail_reasons.append(f"{ctrl}:not_matched")

    if not none_pass:
        fail_reasons.append("none:non_positive_effect_or_ci")
    if not none_beats_orth:
        fail_reasons.append("none:not_better_than_orthogonal_random")
    if rec_rand is not None and (not none_beats_random):
        fail_reasons.append("stress:none_not_better_than_random")
    if rec_shuf is not None and (not none_beats_shuffled):
        fail_reasons.append("stress:none_not_better_than_shuffled")
    if rec_absm is not None and (not none_beats_absm):
        fail_reasons.append("stress:none_not_better_than_abs_marginal_matched_random_orth")
    if rec_rand is not None and random_matched and (not rand_pass):
        fail_reasons.append("random:significant_positive_reduction")
    if rec_shuf is not None and shuffled_matched and (not shuf_stress_pass):
        fail_reasons.append("shuffled:significant_two_sided_effect")
    if orth_matched and (not orth_pass):
        fail_reasons.append("orthogonal_random:significant_positive_reduction")
    if rec_absm is not None and absm_matched and (not absm_stress_pass):
        fail_reasons.append("abs_marginal_matched_random_orth:significant_two_sided_effect")

    cost_items = [
        ("none", none_cost_err),
        ("random", random_cost_err if rec_rand is not None else None),
        ("shuffled", shuffled_cost_err if rec_shuf is not None else None),
        ("orthogonal_random", orth_cost_err),
        ("abs_marginal_matched_random_orth", absm_cost_err if rec_absm is not None else None),
    ]
    cost_items = [(k, float(v)) for k, v in cost_items if v is not None]
    worst_ctrl = None
    max_cost_err = None
    if cost_items:
        worst_ctrl, max_cost_err = max(cost_items, key=lambda x: x[1])

    return {
        "none_pass": none_pass,
        "none_beats_random": none_beats_random,
        "none_beats_shuffled": none_beats_shuffled,
        "none_abs": none_abs,
        "none_ci_low": none_ci[0],
        "none_ci_high": none_ci[1],
        "random_abs": rand_abs,
        "random_ci_low": rand_ci[0],
        "random_ci_high": rand_ci[1],
        "orthogonal_random_abs": orth_abs,
        "orthogonal_random_ci_low": orth_ci[0],
        "orthogonal_random_ci_high": orth_ci[1],
        "shuffled_abs": shuf_abs,
        "shuffled_ci_low": shuf_ci[0],
        "shuffled_ci_high": shuf_ci[1],
        "abs_marginal_matched_random_orth_abs": absm_abs,
        "abs_marginal_matched_random_orth_ci_low": absm_ci[0],
        "abs_marginal_matched_random_orth_ci_high": absm_ci[1],
        "delta_none_minus_random": delta_rand,
        "delta_none_minus_orthogonal_random": delta_orth,
        "delta_none_minus_shuffled": delta_shuf,
        "delta_none_minus_abs_marginal_matched_random_orth": delta_absm,
        "none_matched": none_matched,
        "random_matched": random_matched,
        "random_near_zero": rand_pass,
        "orthogonal_random_matched": orth_matched,
        "shuffled_matched": shuffled_matched,
        "abs_marginal_matched_random_orth_matched": absm_matched,
        "none_cost_err": none_cost_err,
        "random_cost_err": random_cost_err,
        "orthogonal_random_cost_err": orth_cost_err,
        "shuffled_cost_err": shuffled_cost_err,
        "abs_marginal_matched_random_orth_cost_err": absm_cost_err,
        "orthogonal_random_near_zero": orth_pass,
        "shuffled_near_zero": shuf_pass,
        "abs_marginal_matched_random_orth_near_zero": absm_pass,
        "shuffled_stress_non_significant": shuf_stress_pass,
        "abs_marginal_matched_random_orth_stress_non_significant": absm_stress_pass,
        "random_pass": rand_pass,
        "orthogonal_random_pass": orth_pass,
        "shuffled_pass": shuf_pass,
        "abs_marginal_matched_random_orth_pass": absm_pass,
        "none_beats_orthogonal_random": none_beats_orth,
        "none_beats_abs_marginal_matched_random_orth": none_beats_absm,
        "pass_strict": pass_strict_core,
        "pass_strict_core": pass_strict_core,
        "pass_strict_with_shuffled": pass_strict_with_shuffled,
        "pass_strict_with_stress": pass_strict_with_stress,
        "pass_strict_with_orthogonal_random": pass_strict_with_orth,
        "overall_pass": overall_pass_effect_only,
        "overall_pass_effect_only": overall_pass_effect_only,
        "overall_pass_with_shuffled_stress": pass_strict_with_shuffled,
        "overall_pass_with_stress": pass_strict_with_stress,
        "diagnostic_pass_strict_core": pass_strict_core,
        "diagnostic_orth_sig_pos": diagnostic_orth_sig_pos,
        "diagnostic_orth_sig_neg": diagnostic_orth_sig_neg,
        "diagnostic_orth_abs_cos_to_source": orth_abs_cos,
        "diagnostic_orth_top512_mass": orth_top512_mass,
        "diagnostic_orth_abs_cos_hi": diagnostic_orth_abs_cos_hi,
        "fail_reasons": sorted(set(fail_reasons)),
        "max_cost_err": max_cost_err,
        "worst_ctrl": worst_ctrl,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/mc_s*_*/matched_cost_summary.json")
    ap.add_argument("--min-schema-version", type=int, default=0)
    ap.add_argument(
        "--require-fields",
        default="",
        help="Comma-separated top-level fields required in matched_cost_summary.json",
    )
    ap.add_argument("--out-prefix", default="runs/matched_cost_grid_compact")
    ap.add_argument(
        "--verbose-run-ids",
        action="store_true",
        help="Keep full global run_ids arrays in JSON output",
    )
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.runs_glob))
    required_fields = [f.strip() for f in str(args.require_fields).split(",") if f.strip()]
    rows: list[dict] = []
    grouped_per_run: dict[tuple[str, int, str, float, float, str], dict[str, dict]] = {}
    skipped_schema = 0
    skipped_fields = 0

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) < int(args.min_schema_version):
            skipped_schema += 1
            continue
        if any((f not in data) for f in required_fields):
            skipped_fields += 1
            continue
        a = data.get("args", {})
        holdout_seed = int(a.get("holdout_seed", -1))
        control_seed_raw = a.get("control_seed")
        control_seed = None
        if control_seed_raw is not None:
            try:
                control_seed = int(control_seed_raw)
            except Exception:
                control_seed = None
        if control_seed is None:
            m = re.search(r"mc_orth_valid_seed(\d{4})", str(data.get("run_id", p.parent.name)))
            if m:
                control_seed = int(m.group(1))
        if control_seed is None:
            control_seed = -1
        layer = str(a.get("layer", "unknown"))
        target = float(a.get("target_logprob_change", 0.0))
        target_cost = data.get("target_cost")
        target_cost_key = round(float(target_cost), 6) if target_cost is not None else round(target, 6)
        match_tol_used = data.get("match_tol_used")
        match_tol_key = round(float(match_tol_used), 6) if match_tol_used is not None else 0.0
        cost_source = str(data.get("cost_source", "unknown"))
        run_id = str(data.get("run_id", p.parent.name))

        by_ctrl: dict[str, dict] = {}
        for r in data.get("results", []):
            ctrl = str(r.get("control"))
            s = r.get("final_summary", {})
            ci = (s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
            target_cost = r.get("target_cost")
            final_cost = r.get("final_cost")
            if final_cost is None and s.get("mean_logprob_change_pair") is not None:
                final_cost = max(0.0, -float(s.get("mean_logprob_change_pair")))
            cost_err = r.get("cost_err")
            if cost_err is None and final_cost is not None and target_cost is not None:
                cost_err = abs(float(final_cost) - float(target_cost))
            match_tol = r.get("match_tol", 0.002)
            matched = r.get("matched")
            if matched is None and cost_err is not None:
                matched = bool(float(cost_err) <= float(match_tol))
            unmatched = r.get("unmatched")
            if unmatched is None and matched is not None:
                unmatched = (not bool(matched))
            row = {
                "run_id": run_id,
                "seed": holdout_seed,
                "control_seed": control_seed,
                "layer": layer,
                "target_logprob_change": target,
                "target_cost_key": target_cost_key,
                "match_tol_used": match_tol_used,
                "cost_source": cost_source,
                "control": ctrl,
                "matched_alpha": r.get("matched_alpha"),
                "final_cost": final_cost,
                "cost_err": cost_err,
                "match_tol": match_tol,
                "matched": matched,
                "unmatched": unmatched,
                "calibration_mode_used": r.get("calibration_mode_used"),
                "fallback_applied": r.get("fallback_applied"),
                "mean_logprob_change_pair": s.get("mean_logprob_change_pair"),
                "mean_abs_reduction_adj": s.get("mean_abs_reduction_adj"),
                "ci95_lo": ci[0],
                "ci95_hi": ci[1],
                "n_rows": s.get("n_rows"),
                "n_blocks": s.get("n_blocks"),
                "final_run_dir": r.get("final_run_dir"),
            }
            rows.append(row)
            r = dict(r)
            r.setdefault("target_cost", target_cost)
            r.setdefault("final_cost", final_cost)
            r.setdefault("cost_err", cost_err)
            r.setdefault("match_tol", match_tol)
            r.setdefault("matched", matched)
            r.setdefault("unmatched", unmatched)
            by_ctrl[ctrl] = r
        grouped_per_run[(run_id, control_seed, layer, target_cost_key, match_tol_key, cost_source)] = by_ctrl

    rows.sort(
        key=lambda r: (
            int(r["seed"]),
            int(r["control_seed"]),
            ["early", "mid", "late"].index(r["layer"]) if r["layer"] in {"early", "mid", "late"} else 99,
            str(r["run_id"]),
            float(r["target_logprob_change"]),
            [
                "none",
                "random",
                "orthogonal_random",
                "orthogonal_random_support_avoid_top512",
                "abs_marginal_matched_random_orth",
                "shuffled",
            ].index(r["control"])
            if r["control"]
            in {
                "none",
                "random",
                "orthogonal_random",
                "orthogonal_random_support_avoid_top512",
                "abs_marginal_matched_random_orth",
                "shuffled",
            }
            else 99,
        )
    )

    out_tsv = Path(f"{args.out_prefix}.tsv")
    out_csv = Path(f"{args.out_prefix}.csv")
    out_json = Path(f"{args.out_prefix}.json")

    fieldnames = [
        "run_id",
        "seed",
        "control_seed",
        "layer",
        "target_logprob_change",
        "target_cost_key",
        "match_tol_used",
        "cost_source",
        "control",
        "matched_alpha",
        "final_cost",
        "cost_err",
        "match_tol",
        "matched",
        "unmatched",
        "calibration_mode_used",
        "fallback_applied",
        "mean_logprob_change_pair",
        "mean_abs_reduction_adj",
        "ci95_lo",
        "ci95_hi",
        "n_rows",
        "n_blocks",
        "final_run_dir",
    ]

    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(fieldnames) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[k] is None else str(r[k]) for k in fieldnames) + "\n")

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    checks = []
    grouped_global: dict[tuple[int, str, float, float, str], list[dict]] = {}
    grouped_global_noseed: dict[tuple[str, float, float, str], list[dict]] = {}
    for key, ctrls in grouped_per_run.items():
        run_id, control_seed, layer, target_cost_key, match_tol_key, cost_source = key
        if not all(c in ctrls for c in ("none", "orthogonal_random")):
            continue
        flags = _pass_flags(
            ctrls["none"],
            ctrls.get("random"),
            ctrls.get("shuffled"),
            ctrls["orthogonal_random"],
            ctrls.get("abs_marginal_matched_random_orth"),
        )
        run_item = {
            "run_id": run_id,
            "seed": control_seed,
            "controls": ctrls,
            "flags": flags,
        }
        grouped_global.setdefault((control_seed, layer, target_cost_key, match_tol_key, cost_source), []).append(
            run_item
        )
        grouped_global_noseed.setdefault((layer, target_cost_key, match_tol_key, cost_source), []).append(run_item)
        checks.append(
            {
                "run_id": run_id,
                "seed": control_seed,
                "layer": layer,
                "target_cost_key": target_cost_key,
                "match_tol_key": match_tol_key,
                "cost_source": cost_source,
                **flags,
            }
        )

    def _quantile(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        arr = sorted(float(v) for v in vals)
        if len(arr) == 1:
            return arr[0]
        qf = min(max(float(q), 0.0), 1.0)
        pos = (len(arr) - 1) * qf
        lo = int(pos)
        hi = min(lo + 1, len(arr) - 1)
        frac = pos - lo
        return arr[lo] * (1.0 - frac) + arr[hi] * frac

    orth_abs_cos_vals = [
        float(c["diagnostic_orth_abs_cos_to_source"])
        for c in checks
        if c.get("diagnostic_orth_abs_cos_to_source") is not None
    ]
    orth_top512_vals = [
        float(c["diagnostic_orth_top512_mass"])
        for c in checks
        if c.get("diagnostic_orth_top512_mass") is not None
    ]
    orth_abs_cos_q75 = _quantile(orth_abs_cos_vals, 0.75)
    orth_top512_q75 = _quantile(orth_top512_vals, 0.75)

    # Relative-high diagnostics are more informative than a fixed threshold in this regime.
    for chk in checks:
        v = chk.get("diagnostic_orth_abs_cos_to_source")
        chk["diagnostic_orth_abs_cos_hi_rel"] = (
            v is not None and orth_abs_cos_q75 is not None and float(v) > float(orth_abs_cos_q75)
        )
        t = chk.get("diagnostic_orth_top512_mass")
        chk["diagnostic_orth_top512_hi_rel"] = (
            t is not None and orth_top512_q75 is not None and float(t) > float(orth_top512_q75)
        )

    for run_items in grouped_global.values():
        for item in run_items:
            flags = item.get("flags", {})
            v = flags.get("diagnostic_orth_abs_cos_to_source")
            flags["diagnostic_orth_abs_cos_hi_rel"] = (
                v is not None and orth_abs_cos_q75 is not None and float(v) > float(orth_abs_cos_q75)
            )
            t = flags.get("diagnostic_orth_top512_mass")
            flags["diagnostic_orth_top512_hi_rel"] = (
                t is not None and orth_top512_q75 is not None and float(t) > float(orth_top512_q75)
            )

    def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        arr = [float(v) for v in vals]
        mean = sum(arr) / len(arr)
        if len(arr) < 2:
            return float(mean), 0.0
        var = sum((v - mean) ** 2 for v in arr) / (len(arr) - 1)
        return float(mean), float(var ** 0.5)

    def _build_global_checks(groups: dict, include_seed: bool) -> list[dict]:
        out: list[dict] = []
        for key, run_items in sorted(groups.items()):
            if include_seed:
                seed, layer, target_cost_key, match_tol_key, cost_source = key
            else:
                layer, target_cost_key, match_tol_key, cost_source = key
                seed = None

            run_ids = [str(x["run_id"]) for x in run_items]
            seeds = sorted({int(x.get("seed", -1)) for x in run_items if x.get("seed") is not None})
            pass_overall_vals = [bool(x["flags"].get("overall_pass")) for x in run_items]
            pass_strict_vals = [bool(x["flags"].get("pass_strict")) for x in run_items]
            pass_strict_core_vals = [bool(x["flags"].get("pass_strict_core")) for x in run_items]
            pass_strict_shuf_vals = [bool(x["flags"].get("pass_strict_with_shuffled")) for x in run_items]
            pass_strict_orth_vals = [bool(x["flags"].get("pass_strict_with_orthogonal_random")) for x in run_items]
            diag_core_vals = [bool(x["flags"].get("diagnostic_pass_strict_core")) for x in run_items]
            diag_orth_pos_vals = [bool(x["flags"].get("diagnostic_orth_sig_pos")) for x in run_items]
            diag_orth_neg_vals = [bool(x["flags"].get("diagnostic_orth_sig_neg")) for x in run_items]
            diag_orth_abs_cos_hi_vals = [
                bool(x["flags"].get("diagnostic_orth_abs_cos_hi")) for x in run_items
            ]
            diag_orth_abs_cos_hi_rel_vals = [
                bool(x["flags"].get("diagnostic_orth_abs_cos_hi_rel")) for x in run_items
            ]
            diag_orth_top512_hi_rel_vals = [
                bool(x["flags"].get("diagnostic_orth_top512_hi_rel")) for x in run_items
            ]
            overall_pass_rate = (
                sum(1 for v in pass_overall_vals if v) / len(pass_overall_vals)
                if pass_overall_vals
                else 0.0
            )
            strict_pass_rate = (
                sum(1 for v in pass_strict_vals if v) / len(pass_strict_vals)
                if pass_strict_vals
                else 0.0
            )
            strict_core_pass_rate = (
                sum(1 for v in pass_strict_core_vals if v) / len(pass_strict_core_vals)
                if pass_strict_core_vals
                else 0.0
            )
            strict_with_shuffled_pass_rate = (
                sum(1 for v in pass_strict_shuf_vals if v) / len(pass_strict_shuf_vals)
                if pass_strict_shuf_vals
                else 0.0
            )
            strict_with_orthogonal_random_pass_rate = (
                sum(1 for v in pass_strict_orth_vals if v) / len(pass_strict_orth_vals)
                if pass_strict_orth_vals
                else 0.0
            )
            diagnostic_core_seed_rate = (
                sum(1 for v in diag_core_vals if v) / len(diag_core_vals) if diag_core_vals else 0.0
            )
            diagnostic_orth_sig_pos_rate = (
                sum(1 for v in diag_orth_pos_vals if v) / len(diag_orth_pos_vals)
                if diag_orth_pos_vals
                else 0.0
            )
            diagnostic_orth_sig_neg_rate = (
                sum(1 for v in diag_orth_neg_vals if v) / len(diag_orth_neg_vals)
                if diag_orth_neg_vals
                else 0.0
            )
            diagnostic_orth_abs_cos_hi_rate = (
                sum(1 for v in diag_orth_abs_cos_hi_vals if v) / len(diag_orth_abs_cos_hi_vals)
                if diag_orth_abs_cos_hi_vals
                else 0.0
            )
            diagnostic_orth_abs_cos_hi_rel_rate = (
                sum(1 for v in diag_orth_abs_cos_hi_rel_vals if v) / len(diag_orth_abs_cos_hi_rel_vals)
                if diag_orth_abs_cos_hi_rel_vals
                else 0.0
            )
            diagnostic_orth_top512_hi_rel_rate = (
                sum(1 for v in diag_orth_top512_hi_rel_vals if v) / len(diag_orth_top512_hi_rel_vals)
                if diag_orth_top512_hi_rel_vals
                else 0.0
            )

            def _collect(control: str, field: str) -> list[float]:
                vals: list[float] = []
                for item in run_items:
                    rec = item["controls"].get(control, {})
                    if field == "mean_abs_reduction_adj":
                        s = rec.get("final_summary", {})
                        v = s.get("mean_abs_reduction_adj")
                    else:
                        v = rec.get(field)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                return vals

            n_alpha_mean, n_alpha_std = _mean_std(_collect("none", "matched_alpha"))
            r_alpha_mean, r_alpha_std = _mean_std(_collect("random", "matched_alpha"))
            s_alpha_mean, s_alpha_std = _mean_std(_collect("shuffled", "matched_alpha"))
            o_alpha_mean, o_alpha_std = _mean_std(_collect("orthogonal_random", "matched_alpha"))
            a_alpha_mean, a_alpha_std = _mean_std(_collect("abs_marginal_matched_random_orth", "matched_alpha"))
            n_cost_mean, n_cost_std = _mean_std(_collect("none", "cost_err"))
            r_cost_mean, r_cost_std = _mean_std(_collect("random", "cost_err"))
            s_cost_mean, s_cost_std = _mean_std(_collect("shuffled", "cost_err"))
            o_cost_mean, o_cost_std = _mean_std(_collect("orthogonal_random", "cost_err"))
            a_cost_mean, a_cost_std = _mean_std(_collect("abs_marginal_matched_random_orth", "cost_err"))
            n_abs_mean, n_abs_std = _mean_std(_collect("none", "mean_abs_reduction_adj"))
            r_abs_mean, r_abs_std = _mean_std(_collect("random", "mean_abs_reduction_adj"))
            s_abs_mean, s_abs_std = _mean_std(_collect("shuffled", "mean_abs_reduction_adj"))
            o_abs_mean, o_abs_std = _mean_std(_collect("orthogonal_random", "mean_abs_reduction_adj"))
            a_abs_mean, a_abs_std = _mean_std(_collect("abs_marginal_matched_random_orth", "mean_abs_reduction_adj"))
            orth_abs_cos_vals: list[float] = []
            for item in run_items:
                v = item["flags"].get("diagnostic_orth_abs_cos_to_source")
                if v is None:
                    continue
                try:
                    orth_abs_cos_vals.append(float(v))
                except Exception:
                    continue
            orth_abs_cos_mean, orth_abs_cos_std = _mean_std(orth_abs_cos_vals)

            out.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "target_cost_key": target_cost_key,
                    "match_tol_key": match_tol_key,
                    "cost_source": cost_source,
                    "n_runs": len(run_items),
                    "run_ids": run_ids,
                    "run_ids_count": len(run_ids),
                    "run_ids_sample": ",".join(run_ids[:10]),
                    "n_unique_seeds": len(seeds),
                    "seeds_sample": ",".join(str(x) for x in seeds[:10]),
                    "overall_pass_rate": float(overall_pass_rate),
                    "overall_effect_only_pass_rate": float(overall_pass_rate),
                    "strict_pass_rate": float(strict_pass_rate),
                    "strict_core_pass_rate": float(strict_core_pass_rate),
                    "strict_with_shuffled_pass_rate": float(strict_with_shuffled_pass_rate),
                    "strict_with_orthogonal_random_pass_rate": float(strict_with_orthogonal_random_pass_rate),
                    "diagnostic_core_seed_rate": float(diagnostic_core_seed_rate),
                    "diagnostic_orth_sig_pos_rate": float(diagnostic_orth_sig_pos_rate),
                    "diagnostic_orth_sig_neg_rate": float(diagnostic_orth_sig_neg_rate),
                    "diagnostic_orth_abs_cos_hi_rate": float(diagnostic_orth_abs_cos_hi_rate),
                    "diagnostic_orth_abs_cos_hi_rel_rate": float(diagnostic_orth_abs_cos_hi_rel_rate),
                    "diagnostic_orth_top512_hi_rel_rate": float(diagnostic_orth_top512_hi_rel_rate),
                    "diagnostic_orth_abs_cos_mean": orth_abs_cos_mean,
                    "diagnostic_orth_abs_cos_std": orth_abs_cos_std,
                    "none_alpha_mean": n_alpha_mean,
                    "none_alpha_std": n_alpha_std,
                    "random_alpha_mean": r_alpha_mean,
                    "random_alpha_std": r_alpha_std,
                    "shuffled_alpha_mean": s_alpha_mean,
                    "shuffled_alpha_std": s_alpha_std,
                    "orthogonal_random_alpha_mean": o_alpha_mean,
                    "orthogonal_random_alpha_std": o_alpha_std,
                    "abs_marginal_matched_random_orth_alpha_mean": a_alpha_mean,
                    "abs_marginal_matched_random_orth_alpha_std": a_alpha_std,
                    "none_cost_err_mean": n_cost_mean,
                    "none_cost_err_std": n_cost_std,
                    "random_cost_err_mean": r_cost_mean,
                    "random_cost_err_std": r_cost_std,
                    "shuffled_cost_err_mean": s_cost_mean,
                    "shuffled_cost_err_std": s_cost_std,
                    "orthogonal_random_cost_err_mean": o_cost_mean,
                    "orthogonal_random_cost_err_std": o_cost_std,
                    "abs_marginal_matched_random_orth_cost_err_mean": a_cost_mean,
                    "abs_marginal_matched_random_orth_cost_err_std": a_cost_std,
                    "none_abs_reduction_mean": n_abs_mean,
                    "none_abs_reduction_std": n_abs_std,
                    "random_abs_reduction_mean": r_abs_mean,
                    "random_abs_reduction_std": r_abs_std,
                    "shuffled_abs_reduction_mean": s_abs_mean,
                    "shuffled_abs_reduction_std": s_abs_std,
                    "orthogonal_random_abs_reduction_mean": o_abs_mean,
                    "orthogonal_random_abs_reduction_std": o_abs_std,
                    "abs_marginal_matched_random_orth_abs_reduction_mean": a_abs_mean,
                    "abs_marginal_matched_random_orth_abs_reduction_std": a_abs_std,
                }
            )
        return out

    def _write_global_table(records: list[dict], out_stem: str) -> tuple[Path, Path]:
        out_tsv_path = Path(f"{out_stem}.tsv")
        out_csv_path = Path(f"{out_stem}.csv")
        table_rows: list[dict] = []
        for rec in records:
            row = dict(rec)
            row.pop("run_ids", None)
            table_rows.append(row)
        preferred_fields = [
            "layer",
            "seed",
            "target_cost_key",
            "match_tol_key",
            "cost_source",
            "n_runs",
            "n_unique_seeds",
            "seeds_sample",
            "overall_pass_rate",
            "overall_effect_only_pass_rate",
            "strict_pass_rate",
            "strict_with_shuffled_pass_rate",
            "strict_with_orthogonal_random_pass_rate",
            "strict_core_pass_rate",
            "diagnostic_core_seed_rate",
            "diagnostic_orth_sig_pos_rate",
            "diagnostic_orth_sig_neg_rate",
            "diagnostic_orth_abs_cos_hi_rate",
            "diagnostic_orth_abs_cos_hi_rel_rate",
            "diagnostic_orth_top512_hi_rel_rate",
            "diagnostic_orth_abs_cos_mean",
            "diagnostic_orth_abs_cos_std",
            "none_alpha_mean",
            "none_alpha_std",
            "random_alpha_mean",
            "random_alpha_std",
            "abs_marginal_matched_random_orth_alpha_mean",
            "abs_marginal_matched_random_orth_alpha_std",
            "shuffled_alpha_mean",
            "shuffled_alpha_std",
            "orthogonal_random_alpha_mean",
            "orthogonal_random_alpha_std",
            "none_cost_err_mean",
            "none_cost_err_std",
            "random_cost_err_mean",
            "random_cost_err_std",
            "abs_marginal_matched_random_orth_cost_err_mean",
            "abs_marginal_matched_random_orth_cost_err_std",
            "shuffled_cost_err_mean",
            "shuffled_cost_err_std",
            "orthogonal_random_cost_err_mean",
            "orthogonal_random_cost_err_std",
            "none_abs_reduction_mean",
            "none_abs_reduction_std",
            "random_abs_reduction_mean",
            "random_abs_reduction_std",
            "abs_marginal_matched_random_orth_abs_reduction_mean",
            "abs_marginal_matched_random_orth_abs_reduction_std",
            "shuffled_abs_reduction_mean",
            "shuffled_abs_reduction_std",
            "orthogonal_random_abs_reduction_mean",
            "orthogonal_random_abs_reduction_std",
            "run_ids_count",
            "run_ids_sample",
        ]
        present = {k for r in table_rows for k in r.keys()} if table_rows else set()
        extras = sorted(present - set(preferred_fields))
        g_fields = [k for k in preferred_fields if k in present] + extras
        with out_tsv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=g_fields, delimiter="\t")
            w.writeheader()
            for r in table_rows:
                w.writerow(r)
        with out_csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=g_fields)
            w.writeheader()
            for r in table_rows:
                w.writerow(r)
        return out_tsv_path, out_csv_path

    global_checks = _build_global_checks(grouped_global, include_seed=True)
    global_checks_noseed = _build_global_checks(grouped_global_noseed, include_seed=False)
    out_global_tsv, out_global_csv = _write_global_table(global_checks, f"{args.out_prefix}_global")
    out_global_noseed_tsv, out_global_noseed_csv = _write_global_table(
        global_checks_noseed,
        f"{args.out_prefix}_global_noseed",
    )
    if args.verbose_run_ids:
        global_checks_json = global_checks
        global_checks_noseed_json = global_checks_noseed
    else:
        global_checks_json = [{k: v for k, v in rec.items() if k != "run_ids"} for rec in global_checks]
        global_checks_noseed_json = [
            {k: v for k, v in rec.items() if k != "run_ids"} for rec in global_checks_noseed
        ]

    out_json.write_text(
        json.dumps(
            {
                "runs_glob": args.runs_glob,
                "min_schema_version": int(args.min_schema_version),
                "require_fields": required_fields,
                "n_summary_files": len(files),
                "n_skipped_schema": skipped_schema,
                "n_skipped_missing_fields": skipped_fields,
                "n_rows": len(rows),
                "checks": checks,
                "global_checks": global_checks_json,
                "global_checks_noseed": global_checks_noseed_json,
                "pass_strict": sum(1 for c in checks if c.get("pass_strict")),
                "pass_strict_core": sum(1 for c in checks if c.get("pass_strict_core")),
                "pass_strict_with_shuffled": sum(1 for c in checks if c.get("pass_strict_with_shuffled")),
                "pass_strict_with_orthogonal_random": sum(
                    1 for c in checks if c.get("pass_strict_with_orthogonal_random")
                ),
                "diagnostic_pass_strict_core": sum(
                    1 for c in checks if c.get("diagnostic_pass_strict_core")
                ),
                "diagnostic_orth_sig_pos": sum(1 for c in checks if c.get("diagnostic_orth_sig_pos")),
                "diagnostic_orth_sig_neg": sum(1 for c in checks if c.get("diagnostic_orth_sig_neg")),
                "diagnostic_orth_abs_cos_hi": sum(1 for c in checks if c.get("diagnostic_orth_abs_cos_hi")),
                "diagnostic_orth_abs_cos_hi_rel": sum(
                    1 for c in checks if c.get("diagnostic_orth_abs_cos_hi_rel")
                ),
                "diagnostic_orth_top512_hi_rel": sum(
                    1 for c in checks if c.get("diagnostic_orth_top512_hi_rel")
                ),
                "diagnostic_orth_abs_cos_q75": orth_abs_cos_q75,
                "diagnostic_orth_top512_q75": orth_top512_q75,
                "overall_pass_with_shuffled_stress": sum(
                    1 for c in checks if c.get("overall_pass_with_shuffled_stress")
                ),
                "overall_pass_effect_only": sum(1 for c in checks if c.get("overall_pass_effect_only")),
                "overall_pass": sum(1 for c in checks if c["overall_pass"]),
                "overall_total": len(checks),
                "global_total": len(global_checks),
                "global_noseed_total": len(global_checks_noseed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_tsv}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_global_tsv}")
    print(f"Wrote {out_global_csv}")
    print(f"Wrote {out_global_noseed_tsv}")
    print(f"Wrote {out_global_noseed_csv}")
    print(f"Wrote {out_json}")
    print(
        f"Checks overall(effect_only): {sum(1 for c in checks if c.get('overall_pass'))}/{len(checks)} | "
        f"diagnostic_strict_core_seed_rate: {sum(1 for c in checks if c.get('diagnostic_pass_strict_core'))}/{len(checks)} | "
        f"diagnostic_orth_sig_pos: {sum(1 for c in checks if c.get('diagnostic_orth_sig_pos'))}/{len(checks)} | "
        f"diagnostic_orth_sig_neg: {sum(1 for c in checks if c.get('diagnostic_orth_sig_neg'))}/{len(checks)} | "
        f"diagnostic_orth_abs_cos_hi_rel: {sum(1 for c in checks if c.get('diagnostic_orth_abs_cos_hi_rel'))}/{len(checks)} | "
        f"diagnostic_orth_top512_hi_rel: {sum(1 for c in checks if c.get('diagnostic_orth_top512_hi_rel'))}/{len(checks)} | "
        f"overall+shuf(stress): {sum(1 for c in checks if c.get('overall_pass_with_shuffled_stress'))}/{len(checks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
