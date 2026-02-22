#!/usr/bin/env python3
import argparse
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-md", default="runs/mc_shuf_matched_mcK16_family_controls.md")
    ap.add_argument("--agg-json", default="runs/mc_family_optA_effect_only_test.json")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--max_stress_sig_rate", type=float, default=0.50)
    ap.add_argument(
        "--seed-glob",
        default="",
        help="Optional glob of per-seed matched_cost_summary.json files for generator invariant checks.",
    )
    ap.add_argument(
        "--support-control",
        default="orthogonal_random_support_avoid_top512",
        help="Control name used for support-avoid invariant checks.",
    )
    ap.add_argument(
        "--max-masked-support-mass",
        type=float,
        default=1e-6,
        help="Maximum allowed masked_support_mass for support-avoid control.",
    )
    ap.add_argument(
        "--max-support-overlap-frac",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed support_overlap_frac for support-avoid control. "
            "Disabled by default because overlap can be tie-sensitive."
        ),
    )
    ap.add_argument(
        "--max-support-intrusion-frac-eps",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed support_intrusion_frac_eps for support-avoid control "
            "(tie-proof support invariant; default hard zero)."
        ),
    )
    args = ap.parse_args()

    # Keep this input explicit in the contract, even if parsed by a separate gate script.
    family_md = Path(args.family_md)
    if not family_md.is_file():
        raise SystemExit(f"missing family markdown: {family_md}")

    agg_path = Path(args.agg_json)
    if not agg_path.is_file():
        raise SystemExit(f"missing aggregate json: {agg_path}")

    j = json.loads(agg_path.read_text(encoding="utf-8"))
    checks = j.get("checks") or []
    K = len(checks)
    if K != args.K:
        raise SystemExit(f"K mismatch: got {K}, expected {args.K}")

    eff = sum(1 for c in checks if c.get("overall_pass"))
    if eff != K:
        raise SystemExit(f"effect_only overall_pass expected {K}/{K}, got {eff}/{K}")

    orth_pos = sum(1 for c in checks if c.get("diagnostic_orth_sig_pos"))
    orth_neg = sum(1 for c in checks if c.get("diagnostic_orth_sig_neg"))
    if orth_pos > 0.80 * K or orth_neg > 0.80 * K:
        raise SystemExit(f"orth sign skew suspicious: pos={orth_pos}/{K} neg={orth_neg}/{K}")

    stress_fails = sum(
        1 for c in checks if c.get("overall_pass_with_shuffled_stress") is False
    )
    stress_rate = stress_fails / K
    if stress_rate > args.max_stress_sig_rate:
        raise SystemExit(
            f"stress_fail_rate too high: {stress_rate:.3f} > {args.max_stress_sig_rate:.3f}"
        )

    if args.seed_glob:
        files = sorted(glob.glob(args.seed_glob))
        if not files:
            raise SystemExit(f"no files matched --seed-glob: {args.seed_glob}")
        bad: list[str] = []
        for p in files:
            summary = json.loads(Path(p).read_text(encoding="utf-8"))
            rec = None
            for r in summary.get("results", []):
                if str(r.get("control")) == str(args.support_control):
                    rec = r
                    break
            if rec is None:
                bad.append(f"{p}:missing_control={args.support_control}")
                continue
            shape = ((rec.get("final_summary") or {}).get("direction_shape_stats") or {})
            msm = shape.get("masked_support_mass")
            sof = shape.get("support_overlap_frac")
            sif = shape.get("support_intrusion_frac_eps")
            if msm is None:
                bad.append(f"{p}:missing_support_stats")
                continue
            if float(msm) > float(args.max_masked_support_mass):
                bad.append(
                    f"{p}:masked_support_mass={float(msm):.6g}>{float(args.max_masked_support_mass):.6g}"
                )
            if sif is None:
                bad.append(f"{p}:missing_support_intrusion_frac_eps")
            elif float(sif) > float(args.max_support_intrusion_frac_eps):
                bad.append(
                    f"{p}:support_intrusion_frac_eps={float(sif):.6g}>{float(args.max_support_intrusion_frac_eps):.6g}"
                )
            if (
                sof is not None
                and
                args.max_support_overlap_frac is not None
                and float(sof) > float(args.max_support_overlap_frac)
            ):
                bad.append(
                    f"{p}:support_overlap_frac={float(sof):.6g}>{float(args.max_support_overlap_frac):.6g}"
                )
        if bad:
            raise SystemExit(
                "support-avoid invariant failures:\n" + "\n".join(f"- {x}" for x in bad)
            )

    print("OK: family accept gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
