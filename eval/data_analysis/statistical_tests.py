"""
Statistical significance testing for method comparisons.

Uses non-parametric tests appropriate for small N (6 participants),
ordinal Likert data, and within-subject repeated-measures design.

Outputs:
  - outputs/statistical_results.json
  - outputs/statistical_report.txt
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

METHODS = ["gp_refined", "user_customized", "baseline_text", "style_transfer"]
PAIRWISE_COMPARISONS = [
    ("gp_refined", "baseline_text"),
    ("gp_refined", "style_transfer"),
    ("user_customized", "baseline_text"),
    ("user_customized", "style_transfer"),
    ("gp_refined", "user_customized"),
]


def load_scores():
    return pd.read_csv(OUTPUT_DIR / "location_scores.csv")


def participant_means(scores: pd.DataFrame) -> pd.DataFrame:
    """Average over locations and sessions to get one value per (participant, method)."""
    return (
        scores.groupby(["participant", "method"])["score"]
        .mean()
        .unstack("method")
        .reindex(columns=METHODS)
    )


# ---------------------------------------------------------------------------
# 1. Friedman Test
# ---------------------------------------------------------------------------
def friedman_test(pm: pd.DataFrame) -> dict:
    data = [pm[m].values for m in METHODS]
    stat, p = stats.friedmanchisquare(*data)
    return {
        "test": "Friedman",
        "chi2": round(float(stat), 4),
        "df": len(METHODS) - 1,
        "p_value": round(float(p), 4),
        "significance": _sig_label(p),
        "n_participants": len(pm),
    }


# ---------------------------------------------------------------------------
# 2. Wilcoxon Signed-Rank (pairwise, Holm-corrected)
# ---------------------------------------------------------------------------
def wilcoxon_pairwise(pm: pd.DataFrame) -> list[dict]:
    results = []
    raw_ps = []

    for m1, m2 in PAIRWISE_COMPARISONS:
        diff = pm[m1].values - pm[m2].values
        n = len(diff)
        nonzero = diff[diff != 0]

        if len(nonzero) < 2:
            results.append({
                "comparison": f"{m1} vs {m2}",
                "mean_diff": round(float(np.mean(diff)), 3),
                "statistic": None,
                "p_raw": None,
                "p_corrected": None,
                "effect_size_r": None,
                "significance": "insufficient_data",
                "n": n,
                "n_nonzero": len(nonzero),
            })
            raw_ps.append(1.0)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat_result = stats.wilcoxon(pm[m1].values, pm[m2].values)
                stat_val = stat_result.statistic
                p_val = stat_result.pvalue
            except ValueError:
                stat_val = np.nan
                p_val = 1.0

        z = stats.norm.ppf(1 - p_val / 2) if p_val < 1.0 else 0.0
        r = z / np.sqrt(n) if n > 0 else 0.0

        results.append({
            "comparison": f"{m1} vs {m2}",
            "mean_diff": round(float(np.mean(diff)), 3),
            "statistic": round(float(stat_val), 4) if not np.isnan(stat_val) else None,
            "p_raw": round(float(p_val), 4),
            "effect_size_r": round(float(r), 3),
            "n": n,
            "n_nonzero": len(nonzero),
        })
        raw_ps.append(p_val)

    corrected = _holm_bonferroni(raw_ps)
    for i, res in enumerate(results):
        res["p_corrected"] = round(corrected[i], 4)
        res["significance"] = _sig_label(corrected[i])

    return results


# ---------------------------------------------------------------------------
# 3. Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(
    scores: pd.DataFrame,
    n_boot: int = 10000,
    ci: float = 0.95,
) -> list[dict]:
    rng = np.random.default_rng(42)
    results = []

    pm = participant_means(scores)
    participants = pm.index.values

    comparisons = [
        ("gp_refined", "baseline_text", "gp_uplift_vs_baseline"),
        ("user_customized", "baseline_text", "custom_uplift_vs_baseline"),
        ("gp_refined", "style_transfer", "gp_uplift_vs_style"),
        ("user_customized", "style_transfer", "custom_uplift_vs_style"),
        ("gp_refined", "user_customized", "gp_vs_custom"),
    ]

    for m1, m2, label in comparisons:
        diffs = pm[m1].values - pm[m2].values
        observed = np.mean(diffs)

        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_means[b] = np.mean(diffs[idx])

        alpha = 1 - ci
        lo = np.percentile(boot_means, 100 * alpha / 2)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

        results.append({
            "comparison": label,
            "observed_mean_diff": round(float(observed), 3),
            "ci_lower": round(float(lo), 3),
            "ci_upper": round(float(hi), 3),
            "ci_level": ci,
            "excludes_zero": bool(lo > 0 or hi < 0),
        })

    return results


# ---------------------------------------------------------------------------
# 4. Spearman correlations (GP quality vs outcome)
# ---------------------------------------------------------------------------
def correlation_tests(scores: pd.DataFrame) -> list[dict]:
    results = []

    gp_diag = pd.read_csv(OUTPUT_DIR / "gp_diagnostics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "session_summary.csv")

    gp_scores = (
        scores[scores["method"] == "gp_refined"]
        .groupby("session_id")["score"]
        .mean()
    )

    last_round = gp_diag.loc[gp_diag.groupby("session_id")["round"].idxmax()]
    merged = last_round.set_index("session_id")[["spearman_rho", "pairwise_accuracy"]].join(
        gp_scores.rename("mean_gp_score"), how="inner"
    )

    if len(merged) >= 4:
        for gp_col, label in [
            ("spearman_rho", "final_spearman_vs_gp_score"),
            ("pairwise_accuracy", "final_accuracy_vs_gp_score"),
        ]:
            valid = merged.dropna(subset=[gp_col, "mean_gp_score"])
            if len(valid) >= 4:
                rho, p = stats.spearmanr(valid[gp_col], valid["mean_gp_score"])
                results.append({
                    "test": "Spearman correlation",
                    "variables": label,
                    "rho": round(float(rho), 3),
                    "p_value": round(float(p), 4),
                    "n": len(valid),
                    "significance": _sig_label(p),
                })

    rounds_scores = summary.set_index("session_id")[["n_rounds"]].join(
        gp_scores.rename("mean_gp_score"), how="inner"
    )
    if len(rounds_scores) >= 4:
        rho, p = stats.spearmanr(rounds_scores["n_rounds"], rounds_scores["mean_gp_score"])
        results.append({
            "test": "Spearman correlation",
            "variables": "n_rounds_vs_gp_score",
            "rho": round(float(rho), 3),
            "p_value": round(float(p), 4),
            "n": len(rounds_scores),
            "significance": _sig_label(p),
        })

    return results


# ---------------------------------------------------------------------------
# 5. Target vs Non-target (Mann-Whitney U)
# ---------------------------------------------------------------------------
def target_transfer_test(scores: pd.DataFrame) -> dict:
    for method in ["gp_refined", "user_customized"]:
        mscores = scores[scores["method"] == method]
        target = mscores[mscores["is_target_room"]]["score"].values
        transfer = mscores[~mscores["is_target_room"]]["score"].values

        if len(target) >= 3 and len(transfer) >= 3:
            stat, p = stats.mannwhitneyu(target, transfer, alternative="two-sided")
            yield {
                "test": "Mann-Whitney U",
                "comparison": f"{method}: target_room vs transferred",
                "target_mean": round(float(np.mean(target)), 3),
                "transfer_mean": round(float(np.mean(transfer)), 3),
                "U_statistic": round(float(stat), 4),
                "p_value": round(float(p), 4),
                "n_target": len(target),
                "n_transfer": len(transfer),
                "significance": _sig_label(p),
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sig_label(p: float) -> str:
    if p < 0.01:
        return "** (p < .01)"
    elif p < 0.05:
        return "* (p < .05)"
    elif p < 0.10:
        return "~ (marginal)"
    return "n.s."


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [1.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(p * (n - rank), 1.0)
    for i in range(1, len(indexed)):
        orig_i = indexed[i][0]
        orig_prev = indexed[i - 1][0]
        corrected[orig_i] = max(corrected[orig_i], corrected[orig_prev])
    return corrected


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    fr = all_results["friedman"]
    lines.append("1. FRIEDMAN TEST (overall method effect)")
    lines.append(f"   chi2({fr['df']}) = {fr['chi2']}, p = {fr['p_value']}")
    lines.append(f"   N = {fr['n_participants']} participants")
    lines.append(f"   Result: {fr['significance']}")
    lines.append("")

    lines.append("2. WILCOXON SIGNED-RANK TESTS (pairwise, Holm-corrected)")
    for w in all_results["wilcoxon"]:
        lines.append(f"   {w['comparison']}")
        lines.append(f"     mean diff = {w['mean_diff']}, W = {w['statistic']}, "
                      f"p_raw = {w['p_raw']}, p_corr = {w['p_corrected']}, "
                      f"r = {w['effect_size_r']}")
        lines.append(f"     {w['significance']}")
    lines.append("")

    lines.append("3. BOOTSTRAP 95% CONFIDENCE INTERVALS")
    for b in all_results["bootstrap"]:
        lines.append(f"   {b['comparison']}: mean = {b['observed_mean_diff']} "
                      f"[{b['ci_lower']}, {b['ci_upper']}] "
                      f"{'(excludes 0)' if b['excludes_zero'] else '(includes 0)'}")
    lines.append("")

    lines.append("4. SPEARMAN CORRELATIONS (GP quality vs outcome)")
    for c in all_results["correlations"]:
        lines.append(f"   {c['variables']}: rho = {c['rho']}, p = {c['p_value']}, "
                      f"n = {c['n']} -- {c['significance']}")
    lines.append("")

    lines.append("5. MANN-WHITNEY U (target room vs transferred rooms)")
    for t in all_results["target_transfer"]:
        lines.append(f"   {t['comparison']}")
        lines.append(f"     target mean = {t['target_mean']}, "
                      f"transfer mean = {t['transfer_mean']}")
        lines.append(f"     U = {t['U_statistic']}, p = {t['p_value']} -- "
                      f"{t['significance']}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    scores = load_scores()
    pm = participant_means(scores)

    all_results = {}

    print("Running Friedman test...")
    all_results["friedman"] = friedman_test(pm)

    print("Running Wilcoxon signed-rank tests...")
    all_results["wilcoxon"] = wilcoxon_pairwise(pm)

    print("Computing bootstrap CIs...")
    all_results["bootstrap"] = bootstrap_ci(scores)

    print("Running correlation tests...")
    all_results["correlations"] = correlation_tests(scores)

    print("Running target vs transfer tests...")
    all_results["target_transfer"] = list(target_transfer_test(scores))

    with open(OUTPUT_DIR / "statistical_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    report = generate_report(all_results)
    with open(OUTPUT_DIR / "statistical_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nResults saved to {OUTPUT_DIR / 'statistical_results.json'}")
    print(f"Report saved to {OUTPUT_DIR / 'statistical_report.txt'}")


if __name__ == "__main__":
    main()
