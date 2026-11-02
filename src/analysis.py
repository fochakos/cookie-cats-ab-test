"""Cookie Cats A/B test: does moving the first gate from level 30 to level 40 help?

Run from the project root:  python src/analysis.py
Writes every number quoted in the README to reports/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stats_tools import (beta_posterior, bootstrap_mean_diff, bootstrap_prop_diff, holm,
                         mde_two_prop, n_per_arm, srm_test, two_prop_test)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cookie_cats.csv"
REPORTS = ROOT / "reports"

CONTROL, TREATMENT = "gate_30", "gate_40"   # gate_30 is the version already live
ALPHA = 0.05
PRIMARY = "retention_7"                    # decided before looking at results
SECONDARY = "retention_1"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    assert df.userid.is_unique, "duplicate players"
    assert df.notna().all().all(), "missing values"
    assert set(df.version) == {CONTROL, TREATMENT}
    return df


def srm_checks(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    counts = df.version.value_counts()[[CONTROL, TREATMENT]]
    overall = srm_test(counts.tolist())
    overall["counts"] = counts.to_dict()

    # Is the imbalance concentrated in one slice of players or spread evenly?
    # userid is spread uniformly from 0 to 10M (it looks random, not sequential),
    # so this can't locate a time window, but it does show whether the skew is local.
    df = df.assign(decile=pd.qcut(df.userid, 10, labels=False) + 1)
    rows = []
    for d, g in df.groupby("decile"):
        c = g.version.value_counts().reindex([CONTROL, TREATMENT], fill_value=0)
        t = srm_test(c.tolist())
        rows.append({"userid_decile": d, "userid_min": g.userid.min(), "userid_max": g.userid.max(),
                     CONTROL: c[CONTROL], TREATMENT: c[TREATMENT],
                     "share_gate_40": c[TREATMENT] / c.sum(), "p_value": t["p_value"]})
    by_decile = pd.DataFrame(rows)
    # Is the imbalance the same across deciles, or concentrated? (chi-square of homogeneity)
    homog = stats.chi2_contingency(by_decile[[CONTROL, TREATMENT]].to_numpy())
    overall["homogeneity_p_value"] = float(homog.pvalue)
    overall["deciles_with_p_below_0.05"] = int((by_decile.p_value < 0.05).sum())
    return overall, by_decile


def retention_tests(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    g = df.groupby("version")
    results, rows = {}, []
    for metric in [PRIMARY, SECONDARY]:
        x = g[metric].sum()
        n = g[metric].count()
        t = two_prop_test(x[CONTROL], n[CONTROL], x[TREATMENT], n[TREATMENT], ALPHA)
        boot = bootstrap_prop_diff(x[CONTROL], n[CONTROL], x[TREATMENT], n[TREATMENT])
        bayes = beta_posterior(x[CONTROL], n[CONTROL], x[TREATMENT], n[TREATMENT])
        np.save(REPORTS / f"bootstrap_{metric}.npy", boot)
        results[metric] = {
            **t.__dict__,
            "boot_ci_low": float(np.percentile(boot, 2.5)),
            "boot_ci_high": float(np.percentile(boot, 97.5)),
            "boot_share_below_zero": float((boot < 0).mean()),
            **bayes,
            "n_control": int(n[CONTROL]), "n_treatment": int(n[TREATMENT]),
            "x_control": int(x[CONTROL]), "x_treatment": int(x[TREATMENT]),
        }
        rows.append({"metric": metric, "gate_30": t.p_a, "gate_40": t.p_b,
                     "diff_pp": 100 * t.diff, "ci_low_pp": 100 * t.ci_low, "ci_high_pp": 100 * t.ci_high,
                     "relative_lift": t.rel_lift, "p_value": t.p_value,
                     "prob_gate_40_better": bayes["prob_b_better"]})
    adjusted = holm({m: results[m]["p_value"] for m in results})
    for r in rows:
        r["p_value_holm"] = adjusted[r["metric"]]
        results[r["metric"]]["p_value_holm"] = adjusted[r["metric"]]
    return results, pd.DataFrame(rows)


def srm_worst_case(df: pd.DataFrame) -> dict:
    """Stress test: suppose gate_30 lost players from the log until it matched gate_40's size,
    and every lost player churned (the assumption least favourable to gate_30).
    Does gate_30 still have higher 7-day retention?"""
    n = df.version.value_counts()
    x = df.groupby("version")[PRIMARY].sum()
    missing = int(n[TREATMENT] - n[CONTROL])
    t = two_prop_test(x[CONTROL], n[CONTROL] + missing, x[TREATMENT], n[TREATMENT], ALPHA)
    return {"missing_players_added": missing, "diff": t.diff, "ci_low": t.ci_low,
            "ci_high": t.ci_high, "p_value": t.p_value, "p_control": t.p_a}


def engagement(df: pd.DataFrame) -> dict:
    a = df.loc[df.version == CONTROL, "sum_gamerounds"].to_numpy()
    b = df.loc[df.version == TREATMENT, "sum_gamerounds"].to_numpy()
    top = df.sum_gamerounds.max()
    # One gate_30 player logged 49,854 rounds in 14 days (about 3,500 a day): almost
    # certainly a bot or a logging error. It inflates the gate_30 mean and standard deviation,
    # so the mean comparison is reported without it.
    a_clean = a[a != top]
    boot = bootstrap_mean_diff(a_clean.astype(float), b.astype(float))
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "outlier_rounds": int(top),
        "median_control": float(np.median(a)), "median_treatment": float(np.median(b)),
        "mean_control_raw": float(a.mean()), "mean_control_clean": float(a_clean.mean()),
        "mean_treatment": float(b.mean()),
        "mean_diff_clean": float(b.mean() - a_clean.mean()),
        "mean_diff_ci_low": float(np.percentile(boot, 2.5)),
        "mean_diff_ci_high": float(np.percentile(boot, 97.5)),
        "mann_whitney_p": float(mw.pvalue),
        "share_zero_rounds_control": float((a == 0).mean()),
        "share_zero_rounds_treatment": float((b == 0).mean()),
        "share_30plus_control": float((a >= 30).mean()),
        "share_30plus_treatment": float((b >= 30).mean()),
    }


def power(df: pd.DataFrame) -> dict:
    n = df.version.value_counts().min()
    base7 = df.loc[df.version == CONTROL, PRIMARY].mean()
    base1 = df.loc[df.version == CONTROL, SECONDARY].mean()
    return {
        "n_per_arm_used": int(n),
        "mde_pp_retention_7": 100 * mde_two_prop(base7, n),
        "mde_pp_retention_1": 100 * mde_two_prop(base1, n),
        "n_needed_retention_1_observed_effect": n_per_arm(
            base1, df.loc[df.version == TREATMENT, SECONDARY].mean() - base1),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()

    srm, srm_by_decile = srm_checks(df)
    srm["worst_case_retention_7"] = srm_worst_case(df)
    retention, summary = retention_tests(df)
    eng = engagement(df)
    pw = power(df)

    srm_by_decile.to_csv(REPORTS / "srm_by_userid_decile.csv", index=False)
    summary.to_csv(REPORTS / "retention_summary.csv", index=False)
    out = {"players": len(df), "srm": srm, "retention": retention, "engagement": eng, "power": pw}
    (REPORTS / "results.json").write_text(json.dumps(out, indent=2, default=float))

    r7, r1 = retention[PRIMARY], retention[SECONDARY]
    print(f"Players: {len(df):,}   split: {srm['counts']}   SRM p = {srm['p_value']:.4f}"
          f"   homogeneity p = {srm['homogeneity_p_value']:.3f}")
    for name, r in [("7-day", r7), ("1-day", r1)]:
        print(f"{name}: {r['p_a']:.4f} -> {r['p_b']:.4f}  diff {100*r['diff']:+.2f} pp "
              f"[{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}]  rel {100*r['rel_lift']:+.1f}%  "
              f"p = {r['p_value']:.4f} (Holm {r['p_value_holm']:.4f})  "
              f"P(gate_40 better) = {r['prob_b_better']:.3f}")
    print(f"Rounds: median {eng['median_control']:.0f} vs {eng['median_treatment']:.0f}, "
          f"clean mean diff {eng['mean_diff_clean']:+.2f} "
          f"[{eng['mean_diff_ci_low']:+.2f}, {eng['mean_diff_ci_high']:+.2f}], MW p = {eng['mann_whitney_p']:.3f}")
    wc = srm["worst_case_retention_7"]
    print(f"SRM worst case (+{wc['missing_players_added']} churned gate_30 players): "
          f"diff {100*wc['diff']:+.2f} pp [{100*wc['ci_low']:+.2f}, {100*wc['ci_high']:+.2f}] p = {wc['p_value']:.4f}")
    print(f"MDE at 80% power: 7-day {pw['mde_pp_retention_7']:.2f} pp, 1-day {pw['mde_pp_retention_1']:.2f} pp")
    print(srm_by_decile.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
