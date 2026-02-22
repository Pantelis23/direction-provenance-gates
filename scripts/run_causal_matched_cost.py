#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import socket
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


RUN_RE = re.compile(r"Run complete:\s*(runs/\S+)")


def _parse_run_dir(output: str) -> Path:
    m = RUN_RE.search(output)
    if not m:
        raise RuntimeError("Could not parse run directory from run_causal.py output.")
    return Path(m.group(1))


def _call_run_causal(
    base_cmd: list[str],
    control: str,
    alpha: float,
    bootstrap_samples: int,
    run_name: str,
) -> tuple[Path, dict]:
    cmd = list(base_cmd)
    cmd += [
        "--direction-control",
        control,
        "--alpha",
        str(alpha),
        "--bootstrap-samples",
        str(int(bootstrap_samples)),
        "--run-name",
        run_name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        tail = "\n".join(combined.splitlines()[-40:])
        raise RuntimeError(
            "run_causal.py failed"
            f" (control={control}, alpha={alpha}, rc={proc.returncode}).\n{tail}"
        )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    run_dir = _parse_run_dir(combined)
    summary_path = run_dir / "results_causal_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return run_dir, summary


def _resolve_existing_path(path_str: str | None, run_dir: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(str(path_str))
    if p.is_file():
        return p
    if not p.is_absolute():
        # Try relative to current run directory (in case summary used relative paths).
        p2 = (run_dir / p).resolve()
        if p2.is_file():
            return p2
    return None


def _persist_direction_artifacts(summary: dict, run_dir: Path, out_dir: Path, control: str) -> dict[str, object]:
    direction_sha = summary.get("direction_sha256")
    src_vec = _resolve_existing_path(summary.get("direction_path"), run_dir)
    src_vec_ctrl = _resolve_existing_path(summary.get("direction_path_control"), run_dir)
    dst_root = out_dir / "directions"
    dst_root.mkdir(parents=True, exist_ok=True)

    out: dict[str, object] = {
        "direction_sha256": direction_sha,
        "direction_path": summary.get("direction_path"),
        "direction_path_control": summary.get("direction_path_control"),
    }
    if src_vec is not None:
        dst = dst_root / f"{control}.direction.npy"
        shutil.copyfile(src_vec, dst)
        out["direction_path_hermetic"] = str(dst)
    if src_vec_ctrl is not None:
        dst = dst_root / f"{control}.direction_control.npy"
        shutil.copyfile(src_vec_ctrl, dst)
        out["direction_path_control_hermetic"] = str(dst)
    return out


def _cost(summary: dict, mode: str = "neg_only", source: str = "pair") -> float:
    if source == "pll":
        v = summary.get("mean_pll_change")
        if v is None:
            return float("inf")
        val = float(v)
        if mode == "abs":
            return abs(val)
        # For PLL-based quality cost, positive means worse quality.
        return max(0.0, val)
    v = summary.get("mean_logprob_change_pair")
    if v is None:
        return float("inf")
    lp = float(v)
    if mode == "abs":
        return abs(lp)
    # neg_only: positive logprob change implies no cost.
    return max(0.0, -lp)


def _resolve_match_tol(match_tol_arg: str | float, target_cost: float) -> float:
    raw = str(match_tol_arg).strip().lower()
    if raw == "auto":
        return max(0.003, 0.02 * float(target_cost))
    return float(raw)


def _finite_float(x: object) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _fmt_float(x: object, precision: int = 6) -> str:
    v = _finite_float(x)
    if v is None:
        return "na"
    return f"{v:.{int(precision)}g}"


def _runtime_meta() -> dict[str, str]:
    return {
        "utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hostname": socket.gethostname(),
    }


def _progress(msg: str) -> None:
    print(f"[matched-cost] {msg}", flush=True)


def _clamp_alpha(alpha: float, alpha_min: float, alpha_max: float) -> float:
    return min(max(float(alpha), float(alpha_min)), float(alpha_max))


def _infer_increasing_from_trials(trials: list[dict]) -> bool:
    points: list[tuple[float, float]] = []
    for t in trials:
        a = _finite_float(t.get("alpha"))
        c = _finite_float(t.get("cost"))
        if a is None or c is None:
            continue
        points.append((a, c))
    if len(points) < 2:
        return True
    lo = min(points, key=lambda x: x[0])
    hi = max(points, key=lambda x: x[0])
    return bool(hi[1] >= lo[1])


def _calibrate_alpha(
    base_cmd: list[str],
    control: str,
    target_cost: float,
    alpha_min: float,
    alpha_max: float,
    search_iters: int,
    calib_bootstrap_samples: int,
    run_prefix: str,
    cost_mode: str = "neg_only",
    cost_source: str = "pair",
) -> tuple[float, dict, list[dict], list[str]]:
    trials: list[dict] = []
    notes: list[str] = []

    def run_trial(alpha: float, tag: str) -> tuple[dict, float]:
        alpha_req = _finite_float(alpha)
        if alpha_req is None:
            alpha_req = 0.5 * (float(alpha_min) + float(alpha_max))
            notes.append("nonfinite_alpha_requested")
            tag = f"{tag}_nanfix"
        alpha_eval = _clamp_alpha(alpha_req, alpha_min, alpha_max)
        run_name = f"{run_prefix}_{control}_cal_{tag}_a{alpha_eval:g}"
        run_dir, summary = _call_run_causal(
            base_cmd=base_cmd,
            control=control,
            alpha=alpha_eval,
            bootstrap_samples=calib_bootstrap_samples,
            run_name=run_name,
        )
        c_raw = _cost(summary, mode=cost_mode, source=cost_source)
        c = _finite_float(c_raw)
        if c is None:
            notes.append("nonfinite_cost")
        rec = {
            "alpha": float(alpha_eval),
            "cost": (float(c) if c is not None else None),
            "tag": str(tag),
            "mean_logprob_change_pair": summary.get("mean_logprob_change_pair"),
            "mean_pll_change": summary.get("mean_pll_change"),
            "run_dir": str(run_dir),
        }
        trials.append(rec)
        return summary, (float(c) if c is not None else float("inf"))

    _summary_lo, cost_lo = run_trial(float(alpha_min), "lo")
    _summary_hi, cost_hi = run_trial(float(alpha_max), "hi")

    lo = float(alpha_min)
    hi = float(alpha_max)
    clo = _finite_float(cost_lo)
    chi = _finite_float(cost_hi)
    boundary = sorted(
        [t for t in trials if str(t.get("tag")) in {"lo", "hi"}],
        key=lambda t: float(t.get("alpha", 0.0)),
    )
    if clo is None or chi is None:
        notes.append("nonfinite_boundary_cost")
    elif (target_cost - clo) * (target_cost - chi) > 0:
        notes.append("target_not_bracketed")
        if len(boundary) == 2:
            best_boundary = (
                boundary[0]
                if abs(target_cost - clo) <= abs(target_cost - chi)
                else boundary[1]
            )
            best_run = json.loads(
                (Path(best_boundary["run_dir"]) / "results_causal_summary.json").read_text(encoding="utf-8")
            )
            return float(best_boundary["alpha"]), best_run, trials, sorted(set(notes))

    increasing = _infer_increasing_from_trials(trials)
    for i in range(max(0, int(search_iters))):
        mid = 0.5 * (lo + hi)
        _summary_mid, cost_mid = run_trial(mid, f"it{i}")
        cmid = _finite_float(cost_mid)
        if cmid is None:
            notes.append("nonfinite_mid_cost")
            continue
        if increasing:
            if cmid < target_cost:
                lo = mid
            else:
                hi = mid
        else:
            if cmid < target_cost:
                hi = mid
            else:
                lo = mid

    best = min(
        trials,
        key=lambda r: (
            abs(float(_finite_float(r.get("cost"))) - target_cost)
            if _finite_float(r.get("cost")) is not None
            else float("inf")
        ),
    )
    best_alpha = float(best["alpha"])
    best_run = json.loads((Path(best["run_dir"]) / "results_causal_summary.json").read_text(encoding="utf-8"))
    return best_alpha, best_run, trials, sorted(set(notes))


def _calibrate_alpha_probe(
    base_cmd: list[str],
    control: str,
    target_cost: float,
    alpha_min: float,
    alpha_max: float,
    alpha_probe: float,
    calib_bootstrap_samples: int,
    run_prefix: str,
    cost_mode: str = "neg_only",
    cost_source: str = "pair",
) -> tuple[float, dict, list[dict], list[str]]:
    trials: list[dict] = []
    notes: list[str] = []

    def run_trial(alpha: float, tag: str) -> tuple[dict, float]:
        alpha_req = _finite_float(alpha)
        if alpha_req is None:
            alpha_req = 0.5 * (float(alpha_min) + float(alpha_max))
            notes.append("nonfinite_alpha_requested")
            tag = f"{tag}_nanfix"
        alpha_eval = _clamp_alpha(alpha_req, alpha_min, alpha_max)
        run_name = f"{run_prefix}_{control}_cal_{tag}_a{alpha_eval:g}"
        run_dir, summary = _call_run_causal(
            base_cmd=base_cmd,
            control=control,
            alpha=alpha_eval,
            bootstrap_samples=calib_bootstrap_samples,
            run_name=run_name,
        )
        c_raw = _cost(summary, mode=cost_mode, source=cost_source)
        c = _finite_float(c_raw)
        if c is None:
            notes.append("nonfinite_cost")
        rec = {
            "alpha": float(alpha_eval),
            "cost": (float(c) if c is not None else None),
            "tag": str(tag),
            "mean_logprob_change_pair": summary.get("mean_logprob_change_pair"),
            "mean_pll_change": summary.get("mean_pll_change"),
            "run_dir": str(run_dir),
        }
        trials.append(rec)
        return summary, (float(c) if c is not None else float("inf"))

    # Probe boundaries once so reachability is explicit and symmetric for signed search.
    run_trial(float(alpha_min), "lo")
    if float(alpha_max) != float(alpha_min):
        run_trial(float(alpha_max), "hi")
    boundary = sorted(
        [t for t in trials if str(t.get("tag")) in {"lo", "hi"}],
        key=lambda t: float(t.get("alpha", 0.0)),
    )
    increasing = _infer_increasing_from_trials(trials)
    if len(boundary) == 2:
        clo = _finite_float(boundary[0].get("cost"))
        chi = _finite_float(boundary[1].get("cost"))
        if clo is None or chi is None:
            notes.append("nonfinite_boundary_cost")
        else:
            increasing = bool(chi >= clo)
            notes.append(f"probe_boundary_increasing={str(increasing).lower()}")
        if clo is None or chi is None:
            pass
        elif (target_cost - clo) * (target_cost - chi) > 0:
            notes.append("target_not_bracketed")
            best_boundary = (
                boundary[0]
                if abs(target_cost - clo) <= abs(target_cost - chi)
                else boundary[1]
            )
            best_run = json.loads(
                (Path(best_boundary["run_dir"]) / "results_causal_summary.json").read_text(encoding="utf-8")
            )
            return float(best_boundary["alpha"]), best_run, trials, sorted(set(notes))

    probe_alpha = min(max(float(alpha_probe), float(alpha_min)), float(alpha_max))
    probe_cost_f: float | None = None
    probe_ok = False
    for k in range(4):
        _probe_summary, probe_cost = run_trial(probe_alpha, f"probe{k}")
        probe_cost_f = _finite_float(probe_cost)
        if probe_cost_f is not None and probe_cost_f > 1e-6:
            probe_ok = True
            break
        next_probe = probe_alpha
        if probe_alpha < float(alpha_max):
            if probe_alpha <= 0.0:
                next_probe = min(float(alpha_max), max(1e-6, 0.1 * float(alpha_max)))
            else:
                next_probe = min(float(alpha_max), 2.0 * probe_alpha)
        if abs(next_probe - probe_alpha) <= 1e-12:
            break
        notes.append("probe_degenerate_escalate")
        probe_alpha = next_probe
    if not probe_ok:
        notes.append("probe_always_degenerate")

    if probe_cost_f is None:
        est_alpha = 0.5 * (float(alpha_min) + float(alpha_max))
        notes.append("nonfinite_probe_cost")
    elif probe_cost_f <= 1e-12:
        est_alpha = float(alpha_min) if float(target_cost) <= 0.0 else float(alpha_max)
        notes.append("degenerate_probe_cost")
    else:
        if increasing:
            ratio_raw = float(target_cost) / probe_cost_f
        else:
            ratio_raw = probe_cost_f / max(float(target_cost), 1e-12)
            notes.append("probe_ratio_inverted_for_decreasing_cost")
        ratio = _finite_float(ratio_raw)
        if ratio is None:
            est_alpha = float(probe_alpha)
            notes.append("nonfinite_ratio")
        else:
            est_alpha = probe_alpha * ratio
        est_alpha = _clamp_alpha(est_alpha, alpha_min, alpha_max)
    if _finite_float(est_alpha) is None:
        est_alpha = float(probe_alpha)
        notes.append("nonfinite_alpha_est")
    _est_summary, _est_cost = run_trial(est_alpha, "est")

    # Optional one-step refinement when probe/est bracket the target cost.
    est_cost_f = _finite_float(_est_cost)
    if (
        est_alpha != probe_alpha
        and probe_cost_f is not None
        and est_cost_f is not None
        and (probe_cost_f - target_cost) * (est_cost_f - target_cost) < 0
    ):
        denom = _finite_float(est_cost_f - probe_cost_f)
        if denom is None or abs(denom) < 1e-12:
            alpha_interp = 0.5 * (probe_alpha + est_alpha)
            notes.append("degenerate_interp_slope")
        else:
            alpha_interp = probe_alpha + (target_cost - probe_cost_f) * (est_alpha - probe_alpha) / denom
        alpha_interp = _clamp_alpha(alpha_interp, alpha_min, alpha_max)
        # Avoid exact duplicate trial.
        if abs(alpha_interp - probe_alpha) > 1e-9 and abs(alpha_interp - est_alpha) > 1e-9:
            run_trial(alpha_interp, "interp")

    best = min(
        trials,
        key=lambda r: (
            abs(float(_finite_float(r.get("cost"))) - target_cost)
            if _finite_float(r.get("cost")) is not None
            else float("inf")
        ),
    )
    best_alpha = float(best["alpha"])
    best_run = json.loads((Path(best["run_dir"]) / "results_causal_summary.json").read_text(encoding="utf-8"))
    return best_alpha, best_run, trials, sorted(set(notes))


def _best_bracket_from_trials(
    trials: list[dict],
    target: float,
    increasing: bool,
) -> tuple[float, float, float, float] | None:
    pts: list[tuple[float, float]] = []
    for t in trials:
        a = _finite_float(t.get("alpha"))
        c = _finite_float(t.get("cost"))
        if a is None or c is None:
            continue
        pts.append((float(a), float(c)))
    if not pts:
        return None
    below = [(a, c) for a, c in pts if c <= float(target)]
    above = [(a, c) for a, c in pts if c >= float(target)]
    if not below or not above:
        return None
    if increasing:
        a_lo, c_lo = max(below, key=lambda x: x[0])
        a_hi, c_hi = min(above, key=lambda x: x[0])
    else:
        a_lo, c_lo = max(above, key=lambda x: x[0])
        a_hi, c_hi = min(below, key=lambda x: x[0])
    if float(a_hi) <= float(a_lo) + 1e-12:
        return None
    return float(a_lo), float(c_lo), float(a_hi), float(c_hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--d-path", default=None)
    ap.add_argument("--variant-key", default=None)
    ap.add_argument("--variant-hash", default=None)
    ap.add_argument("--layer", default="late")
    ap.add_argument("--mode", choices=["project_out", "flip", "add"], default="project_out")
    ap.add_argument("--alpha-policy", choices=["same", "opposite", "only_a", "only_b"], default="same")
    ap.add_argument("--intervention-scope", choices=["all", "name", "non_name"], default="all")
    ap.add_argument("--filter-template", default="gender_names_jobs")
    ap.add_argument("--eval-split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--control-seed", type=int, default=None, help="Optional seed passed to run_causal controls.")
    ap.add_argument(
        "--direction-seed",
        type=int,
        default=None,
        help="Optional direction seed passed to run_causal (defaults to control-seed there).",
    )
    ap.add_argument("--controls", default="none,random,shuffled")
    ap.add_argument("--target-logprob-change", type=float, default=-0.02)
    ap.add_argument("--target-pll-change", default=None, help="PLL quality target (float) or 'auto'.")
    ap.add_argument(
        "--auto-target-frac",
        type=float,
        default=0.5,
        help="When --target-pll-change=auto, use frac * min(max_reachable_cost_control).",
    )
    ap.add_argument("--calibration-mode", choices=["bisection", "probe"], default="bisection")
    ap.add_argument("--alpha-min", type=float, default=0.05)
    ap.add_argument("--alpha-max", type=float, default=4.0)
    ap.add_argument("--alpha-probe", type=float, default=1.0)
    ap.add_argument("--search-iters", type=int, default=6)
    ap.add_argument("--calib-bootstrap-samples", type=int, default=0)
    ap.add_argument("--match-check-bootstrap-samples", type=int, default=0)
    ap.add_argument("--final-bootstrap-samples", type=int, default=10000)
    ap.add_argument("--match-tol", default="0.002", help="Absolute match tolerance or 'auto'.")
    ap.add_argument("--fallback-on-unmatched", choices=["none", "bisection"], default="bisection")
    ap.add_argument("--cost-mode", choices=["neg_only", "abs"], default="neg_only")
    ap.add_argument("--run-name", default="causal_matched_cost")
    args = ap.parse_args()
    global_notes: list[str] = []

    # Prevent empty evaluation sets in holdout=0 diagnostic runs.
    if float(args.holdout_frac) <= 0.0 and str(args.eval_split) == "test":
        args.eval_split = "all"
        global_notes.append("eval_split_forced_all_due_to_holdout0")

    auto_target_pll = False
    max_reachable_cost_control: dict[str, float] = {}
    if args.target_pll_change is not None:
        pll_target_raw = str(args.target_pll_change).strip().lower()
        cost_source = "pll"
        if pll_target_raw == "auto":
            auto_target_pll = True
            target_cost = None
        else:
            target_cost = max(0.0, float(pll_target_raw))
    else:
        target_cost = max(0.0, -float(args.target_logprob_change))
        cost_source = "pair"
    control_alias = {
        "marginal_matched_random_orth": "signed_resample_marginal_orth",
    }
    raw_controls = [c.strip() for c in str(args.controls).split(",") if c.strip()]
    controls: list[str] = []
    seen_controls: set[str] = set()
    for c in raw_controls:
        cc = control_alias.get(c, c)
        if cc != c:
            global_notes.append(f"control_alias:{c}->{cc}")
        if cc in seen_controls:
            global_notes.append(f"control_dedup_skipped:{cc}")
            continue
        seen_controls.add(cc)
        controls.append(cc)
    if not controls:
        raise ValueError("No controls provided.")
    _progress(
        "start "
        f"run_name={args.run_name} controls={','.join(controls)} "
        f"cost_source={cost_source} holdout_seed={int(args.holdout_seed)}"
    )

    base_cmd = [
        sys.executable,
        "scripts/run_causal.py",
        "--config",
        args.config,
        "--source-run",
        args.source_run,
        "--mode",
        args.mode,
        "--alpha-policy",
        args.alpha_policy,
        "--intervention-scope",
        args.intervention_scope,
        "--layer",
        args.layer,
        "--filter-template",
        args.filter_template,
        "--eval-split",
        args.eval_split,
        "--holdout-frac",
        str(float(args.holdout_frac)),
        "--holdout-seed",
        str(int(args.holdout_seed)),
    ]
    if args.control_seed is not None:
        base_cmd += ["--control-seed", str(int(args.control_seed))]
    if args.direction_seed is not None:
        base_cmd += ["--direction-seed", str(int(args.direction_seed))]
    if args.d_path:
        base_cmd += ["--d-path", args.d_path]
    if args.variant_key:
        base_cmd += ["--variant-key", args.variant_key]
    if args.variant_hash:
        base_cmd += ["--variant-hash", args.variant_hash]

    out_root = Path("runs")
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"{args.run_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"output_dir={out_dir}")

    results: list[dict] = []
    all_trials: dict[str, list[dict]] = {}
    match_tol_used = None

    if auto_target_pll:
        _progress(
            "auto_target_pre start "
            f"frac={_fmt_float(args.auto_target_frac)} alpha_range=[{_fmt_float(args.alpha_min)}, {_fmt_float(args.alpha_max)}]"
        )
        pre_trials: dict[str, list[dict]] = {}
        for control in controls:
            _progress(f"auto_target_pre control={control}")
            _best_alpha_pre, _best_run_pre, trials_pre, _notes_pre = _calibrate_alpha_probe(
                base_cmd=base_cmd,
                control=control,
                target_cost=0.0,
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_probe=float(args.alpha_probe),
                calib_bootstrap_samples=int(args.calib_bootstrap_samples),
                run_prefix=f"{args.run_name}_s{args.holdout_seed}_{args.layer}_auto",
                cost_mode=str(args.cost_mode),
                cost_source=cost_source,
            )
            for t in trials_pre:
                t.setdefault("phase", "auto_target_pre")
            pre_trials[control] = trials_pre
            max_reachable_cost_control[control] = max(
                (
                    float(_finite_float(t.get("cost")))
                    if _finite_float(t.get("cost")) is not None
                    else 0.0
                )
                for t in trials_pre
            )
        if not max_reachable_cost_control:
            raise RuntimeError("Could not compute auto PLL target: no controls/trials available.")
        auto_target_frac = float(args.auto_target_frac)
        if not (0.0 < auto_target_frac <= 1.0):
            raise ValueError("--auto-target-frac must be in (0, 1].")
        target_cost = auto_target_frac * min(max_reachable_cost_control.values())
        _progress(
            "auto_target_pre done "
            f"target_cost={_fmt_float(target_cost)} min_cap={_fmt_float(min(max_reachable_cost_control.values()))}"
        )
        for control, trials_pre in pre_trials.items():
            all_trials.setdefault(control, []).extend(trials_pre)
    match_tol_used = _resolve_match_tol(args.match_tol, float(target_cost))
    _progress(f"match_tol={_fmt_float(match_tol_used)} target_cost={_fmt_float(target_cost)}")

    for control in controls:
        _progress(f"control={control} stage=calibration mode={args.calibration_mode}")
        mode_used = args.calibration_mode
        fallback_applied = False
        control_notes: list[str] = []
        if args.calibration_mode == "probe":
            best_alpha, _best_summary, trials, notes = _calibrate_alpha_probe(
                base_cmd=base_cmd,
                control=control,
                target_cost=target_cost,
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_probe=float(args.alpha_probe),
                calib_bootstrap_samples=int(args.calib_bootstrap_samples),
                run_prefix=f"{args.run_name}_s{args.holdout_seed}_{args.layer}",
                cost_mode=str(args.cost_mode),
                cost_source=cost_source,
            )
            control_notes.extend(notes)
        else:
            best_alpha, _best_summary, trials, notes = _calibrate_alpha(
                base_cmd=base_cmd,
                control=control,
                target_cost=target_cost,
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                search_iters=int(args.search_iters),
                calib_bootstrap_samples=int(args.calib_bootstrap_samples),
                run_prefix=f"{args.run_name}_s{args.holdout_seed}_{args.layer}",
                cost_mode=str(args.cost_mode),
                cost_source=cost_source,
            )
            control_notes.extend(notes)
        for t in trials:
            t.setdefault("phase", "initial")
        _progress(
            f"control={control} stage=calibration_done "
            f"alpha_est={_fmt_float(best_alpha)} n_trials={len(trials)} notes={len(set(control_notes))}"
        )

        # Reachability short-circuit: if target is outside sampled cost range, skip fallback/final.
        finite_trials: list[tuple[dict, float]] = []
        for t in trials:
            c = _finite_float(t.get("cost"))
            if c is not None:
                finite_trials.append((t, float(c)))
        reach_unreachable = False
        reach_best: dict | None = None
        reach_cost: float | None = None
        if finite_trials:
            c_min = min(c for _, c in finite_trials)
            c_max = max(c for _, c in finite_trials)
            if target_cost < (c_min - float(match_tol_used)) or target_cost > (c_max + float(match_tol_used)):
                reach_unreachable = True
                reach_best, reach_cost = min(finite_trials, key=lambda tc: abs(tc[1] - target_cost))
        else:
            reach_unreachable = True
            control_notes.append("nonfinite_sampled_costs")
            if trials:
                reach_best = trials[0]

        if reach_unreachable and reach_best is not None:
            reach_run_dir = Path(str(reach_best.get("run_dir")))
            reach_summary = json.loads((reach_run_dir / "results_causal_summary.json").read_text(encoding="utf-8"))
            direction_meta = _persist_direction_artifacts(
                summary=reach_summary,
                run_dir=reach_run_dir,
                out_dir=out_dir,
                control=control,
            )
            reach_summary = dict(reach_summary)
            reach_summary.update({k: v for k, v in direction_meta.items() if k.startswith("direction_")})
            control_notes.append("target_unreachable_in_sampled_alpha_range")
            all_trials[control] = trials
            results.append(
                {
                    "control": control,
                    "matched_alpha": float(reach_best.get("alpha", args.alpha_max)),
                    "target_cost": target_cost,
                    "final_cost": (float(reach_cost) if reach_cost is not None and math.isfinite(reach_cost) else None),
                    "cost_err": (
                        abs(float(reach_cost) - target_cost)
                        if reach_cost is not None and math.isfinite(reach_cost)
                        else None
                    ),
                    "match_tol": float(match_tol_used),
                    "matched": False,
                    "unmatched": True,
                    "calibration_mode_used": mode_used,
                    "fallback_applied": False,
                    "feedback_steps": 0,
                    "refine_steps": 0,
                    "calibration_notes": sorted(set(control_notes)),
                    "final_bootstrap_used": 0,
                    "final_run_dir": str(reach_run_dir),
                    "direction_sha256": direction_meta.get("direction_sha256"),
                    "direction_path": direction_meta.get("direction_path"),
                    "direction_path_control": direction_meta.get("direction_path_control"),
                    "direction_path_hermetic": direction_meta.get("direction_path_hermetic"),
                    "direction_path_control_hermetic": direction_meta.get("direction_path_control_hermetic"),
                    "final_summary": reach_summary,
                }
            )
            continue

        check_name = f"{args.run_name}_s{args.holdout_seed}_{args.layer}_{control}_check"
        check_run_dir, check_summary = _call_run_causal(
            base_cmd=base_cmd,
            control=control,
            alpha=best_alpha,
            bootstrap_samples=int(args.match_check_bootstrap_samples),
            run_name=check_name,
        )
        final_run_dir = check_run_dir
        final_summary = check_summary
        final_bootstrap_used = int(args.match_check_bootstrap_samples)

        final_cost_raw = _cost(check_summary, mode=str(args.cost_mode), source=cost_source)
        final_cost_f = _finite_float(final_cost_raw)
        if final_cost_f is None:
            control_notes.append("nonfinite_final_cost")
            final_cost = None
            cost_err = None
        else:
            final_cost = float(final_cost_f)
            cost_err = abs(final_cost - target_cost)
        matched = bool(cost_err is not None and math.isfinite(cost_err) and cost_err <= float(match_tol_used))
        feedback_steps = 0
        _progress(
            f"control={control} stage=match_check "
            f"alpha={_fmt_float(best_alpha)} cost={_fmt_float(final_cost)} "
            f"cost_err={_fmt_float(cost_err)} matched={matched}"
        )

        # Correct systematic under/overshoot with a damped multiplicative update.
        increasing = _infer_increasing_from_trials(trials)
        for step in range(5):
            if matched:
                break
            c_now = _finite_float(final_cost)
            if c_now is None:
                control_notes.append("feedback_nonfinite_cost")
                break
            ratio = _finite_float(float(target_cost) / max(c_now, 1e-12))
            if ratio is None:
                control_notes.append("feedback_nonfinite_ratio")
                break
            ratio = min(max(ratio, 0.5), 2.0)
            gamma = 0.7
            a_now = _finite_float(best_alpha)
            if a_now is None:
                control_notes.append("feedback_nonfinite_alpha")
                break
            if increasing:
                a_next = a_now * (ratio**gamma)
            else:
                a_next = a_now / (ratio**gamma)
            a_next = _clamp_alpha(a_next, args.alpha_min, args.alpha_max)
            if abs(a_next - a_now) <= 1e-9:
                control_notes.append("feedback_stalled")
                break
            feedback_steps += 1
            check_run_dir, check_summary = _call_run_causal(
                base_cmd=base_cmd,
                control=control,
                alpha=a_next,
                bootstrap_samples=int(args.match_check_bootstrap_samples),
                run_name=f"{check_name}_fbk{step}",
            )
            c_trial_raw = _cost(check_summary, mode=str(args.cost_mode), source=cost_source)
            c_trial = _finite_float(c_trial_raw)
            if c_trial is None:
                control_notes.append("feedback_nonfinite_cost_eval")
            trials.append(
                {
                    "alpha": float(a_next),
                    "cost": (float(c_trial) if c_trial is not None else None),
                    "tag": f"fbk{step}",
                    "phase": "feedback",
                    "mean_logprob_change_pair": check_summary.get("mean_logprob_change_pair"),
                    "mean_pll_change": check_summary.get("mean_pll_change"),
                    "run_dir": str(check_run_dir),
                }
            )
            final_run_dir = check_run_dir
            final_summary = check_summary
            final_bootstrap_used = int(args.match_check_bootstrap_samples)
            best_alpha = float(a_next)
            final_cost = (float(c_trial) if c_trial is not None else None)
            cost_err = abs(final_cost - target_cost) if (final_cost is not None and math.isfinite(final_cost)) else None
            matched = bool(cost_err is not None and math.isfinite(cost_err) and cost_err <= float(match_tol_used))

        if (
            not matched
            and args.fallback_on_unmatched == "bisection"
            and mode_used != "bisection"
        ):
            _progress(f"control={control} stage=fallback_bisection")
            fallback_applied = True
            mode_used = "bisection"
            best_alpha_fb, _best_summary_fb, trials_fb, notes_fb = _calibrate_alpha(
                base_cmd=base_cmd,
                control=control,
                target_cost=target_cost,
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                search_iters=int(args.search_iters),
                calib_bootstrap_samples=int(args.calib_bootstrap_samples),
                run_prefix=f"{args.run_name}_s{args.holdout_seed}_{args.layer}_{control}_fb",
                cost_mode=str(args.cost_mode),
                cost_source=cost_source,
            )
            control_notes.extend(notes_fb)
            for t in trials_fb:
                t["phase"] = "fallback_bisection"
            trials.extend(trials_fb)
            check_run_dir, check_summary = _call_run_causal(
                base_cmd=base_cmd,
                control=control,
                alpha=best_alpha_fb,
                bootstrap_samples=int(args.match_check_bootstrap_samples),
                run_name=f"{check_name}_fb",
            )
            final_run_dir = check_run_dir
            final_summary = check_summary
            final_bootstrap_used = int(args.match_check_bootstrap_samples)
            best_alpha = best_alpha_fb
            final_cost_raw = _cost(check_summary, mode=str(args.cost_mode), source=cost_source)
            final_cost_f = _finite_float(final_cost_raw)
            if final_cost_f is None:
                control_notes.append("fallback_nonfinite_cost")
                final_cost = None
                cost_err = None
            else:
                final_cost = float(final_cost_f)
                cost_err = abs(final_cost - target_cost)
            matched = bool(cost_err is not None and math.isfinite(cost_err) and cost_err <= float(match_tol_used))

        # Local bracket refine: propose one interior point per iter from the best target bracket.
        refine_steps = 0
        if not matched:
            _progress(f"control={control} stage=refine start")
            for rk in range(10):
                bracket = _best_bracket_from_trials(
                    trials=trials,
                    target=float(target_cost),
                    increasing=bool(increasing),
                )
                if bracket is None:
                    control_notes.append("refine_no_bracket")
                    break
                a_lo, c_lo, a_hi, c_hi = bracket
                denom = _finite_float(float(c_hi) - float(c_lo))
                if denom is None or abs(float(denom)) < 1e-12:
                    a_next = 0.5 * (float(a_lo) + float(a_hi))
                    control_notes.append("refine_deg_slope_midpoint")
                else:
                    a_next = float(a_lo) + (float(target_cost) - float(c_lo)) * (
                        float(a_hi) - float(a_lo)
                    ) / float(denom)
                # Keep the proposal inside the local bracket first, then global bounds.
                a_minb = min(float(a_lo), float(a_hi))
                a_maxb = max(float(a_lo), float(a_hi))
                a_next = max(a_minb, min(a_maxb, float(a_next)))
                eps = 1e-6 * max(1.0, abs(float(a_hi) - float(a_lo)))
                if abs(a_next - float(a_lo)) < 1e-12:
                    a_next = min(a_maxb, float(a_lo) + eps)
                elif abs(a_next - float(a_hi)) < 1e-12:
                    a_next = max(a_minb, float(a_hi) - eps)
                a_next = _clamp_alpha(a_next, args.alpha_min, args.alpha_max)
                if _finite_float(best_alpha) is not None and abs(float(a_next) - float(best_alpha)) <= 1e-9:
                    control_notes.append("refine_stalled")
                    break
                check_run_dir, check_summary = _call_run_causal(
                    base_cmd=base_cmd,
                    control=control,
                    alpha=a_next,
                    bootstrap_samples=int(args.match_check_bootstrap_samples),
                    run_name=f"{check_name}_ref{rk}_a{a_next:g}",
                )
                c_trial_raw = _cost(check_summary, mode=str(args.cost_mode), source=cost_source)
                c_trial = _finite_float(c_trial_raw)
                trials.append(
                    {
                        "alpha": float(a_next),
                        "cost": (float(c_trial) if c_trial is not None else None),
                        "tag": f"ref{rk}",
                        "phase": "refine",
                        "mean_logprob_change_pair": check_summary.get("mean_logprob_change_pair"),
                        "mean_pll_change": check_summary.get("mean_pll_change"),
                        "run_dir": str(check_run_dir),
                    }
                )
                if c_trial is None:
                    control_notes.append("refine_nonfinite_cost_eval")
                    break
                refine_steps += 1
                final_run_dir = check_run_dir
                final_summary = check_summary
                final_bootstrap_used = int(args.match_check_bootstrap_samples)
                best_alpha = float(a_next)
                final_cost = float(c_trial)
                cost_err = abs(final_cost - target_cost)
                matched = bool(cost_err <= float(match_tol_used))
                if matched:
                    break
            _progress(
                f"control={control} stage=refine_done "
                f"refine_steps={int(refine_steps)} alpha={_fmt_float(best_alpha)} "
                f"cost_err={_fmt_float(cost_err)} matched={matched}"
            )

        if matched and refine_steps > 0 and "feedback_stalled" in control_notes:
            control_notes = [n for n in control_notes if n != "feedback_stalled"]
            control_notes.append("feedback_stalled_then_refined")

        # Only spend full bootstrap budget when cost is matched.
        if matched and int(args.final_bootstrap_samples) > int(args.match_check_bootstrap_samples):
            _progress(
                f"control={control} stage=final_bootstrap samples={int(args.final_bootstrap_samples)}"
            )
            final_name = f"{args.run_name}_s{args.holdout_seed}_{args.layer}_{control}_final"
            final_run_dir, final_summary = _call_run_causal(
                base_cmd=base_cmd,
                control=control,
                alpha=best_alpha,
                bootstrap_samples=int(args.final_bootstrap_samples),
                run_name=final_name,
            )
            final_bootstrap_used = int(args.final_bootstrap_samples)
            final_cost_raw = _cost(final_summary, mode=str(args.cost_mode), source=cost_source)
            final_cost_f = _finite_float(final_cost_raw)
            if final_cost_f is None:
                control_notes.append("final_nonfinite_cost")
                final_cost = None
                cost_err = None
            else:
                final_cost = float(final_cost_f)
                cost_err = abs(final_cost - target_cost)
            matched = bool(cost_err is not None and math.isfinite(cost_err) and cost_err <= float(match_tol_used))
            _progress(
                f"control={control} stage=final_bootstrap_done "
                f"cost={_fmt_float(final_cost)} cost_err={_fmt_float(cost_err)} matched={matched}"
            )

        existing_trials = all_trials.get(control, [])
        all_trials[control] = existing_trials + trials
        direction_meta = _persist_direction_artifacts(
            summary=final_summary,
            run_dir=final_run_dir,
            out_dir=out_dir,
            control=control,
        )
        final_summary = dict(final_summary)
        final_summary.update({k: v for k, v in direction_meta.items() if k.startswith("direction_")})
        results.append(
            {
                "control": control,
                "matched_alpha": best_alpha,
                "target_cost": target_cost,
                "final_cost": final_cost,
                "cost_err": cost_err,
                "match_tol": float(match_tol_used),
                "matched": matched,
                "unmatched": (not matched),
                "calibration_mode_used": mode_used,
                "fallback_applied": fallback_applied,
                "feedback_steps": int(feedback_steps),
                "refine_steps": int(refine_steps),
                "calibration_notes": sorted(set(control_notes)),
                "final_bootstrap_used": final_bootstrap_used,
                "final_run_dir": str(final_run_dir),
                "direction_sha256": direction_meta.get("direction_sha256"),
                "direction_path": direction_meta.get("direction_path"),
                "direction_path_control": direction_meta.get("direction_path_control"),
                "direction_path_hermetic": direction_meta.get("direction_path_hermetic"),
                "direction_path_control_hermetic": direction_meta.get("direction_path_control_hermetic"),
                "final_summary": final_summary,
            }
        )
        _progress(
            f"control={control} done matched={matched} alpha={_fmt_float(best_alpha)} "
            f"cost_err={_fmt_float(cost_err)}"
        )

    out_json = out_dir / "matched_cost_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "runtime_meta": _runtime_meta(),
                "run_id": out_dir.name,
                "args": vars(args),
                "target_cost": target_cost,
                "cost_source": cost_source,
                "target_cost_auto": auto_target_pll,
                "auto_target_frac": float(args.auto_target_frac),
                "match_tol_arg": str(args.match_tol),
                "match_tol_used": float(match_tol_used),
                "max_reachable_cost_control": max_reachable_cost_control,
                "notes": global_notes,
                "results": results,
                "calibration_trials": all_trials,
                "all_trials": all_trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_tsv = out_dir / "matched_cost.tsv"
    with out_tsv.open("w", encoding="utf-8") as fh:
        fh.write(
            "\t".join(
                [
                    "control",
                    "matched_alpha",
                "target_cost",
                "cost_source",
                "final_cost",
                "cost_err",
                "match_tol",
                "matched",
                "unmatched",
                "calibration_mode_used",
                "fallback_applied",
                "final_bootstrap_used",
                "mean_logprob_change_pair",
                "mean_abs_reduction_adj",
                "ci95_lo",
                    "ci95_hi",
                    "n_rows",
                    "n_blocks",
                    "final_run_dir",
                ]
            )
            + "\n"
        )
        for rec in results:
            s = rec["final_summary"]
            ci = (s.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
            fh.write(
                "\t".join(
                    [
                        rec["control"],
                        str(rec["matched_alpha"]),
                        str(rec["target_cost"]),
                        str(cost_source),
                        str(rec["final_cost"]),
                        str(rec["cost_err"]),
                        str(rec["match_tol"]),
                        str(rec["matched"]),
                        str(rec["unmatched"]),
                        str(rec["calibration_mode_used"]),
                        str(rec["fallback_applied"]),
                        str(rec["final_bootstrap_used"]),
                        str(s.get("mean_logprob_change_pair")),
                        str(s.get("mean_abs_reduction_adj")),
                        str(ci[0]),
                        str(ci[1]),
                        str(s.get("n_rows")),
                        str(s.get("n_blocks")),
                        rec["final_run_dir"],
                    ]
                )
                + "\n"
            )

    _progress(f"done controls={len(results)}")
    print(f"Run complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
