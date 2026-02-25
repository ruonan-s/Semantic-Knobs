"""
Deep-dive analysis: GP-Refined vs User Customized.

Focused comparison with:
  - Paired differences by participant, location, target/transfer
  - Conditional win analysis (when does GP beat Custom?)
  - GP diagnostic predictors of the gap
  - Score agreement analysis
  - Additional statistical tests
  - 6 targeted figures
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

GP_COLOR = "#E91E63"
CUSTOM_COLOR = "#FFB74D"
TIE_COLOR = "#9E9E9E"


def load():
    scores = pd.read_csv(OUTPUT_DIR / "location_scores.csv")
    gp_diag = pd.read_csv(OUTPUT_DIR / "gp_diagnostics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "session_summary.csv")
    drift = pd.read_csv(OUTPUT_DIR / "preference_drift.csv")
    return scores, gp_diag, summary, drift


def build_paired(scores: pd.DataFrame) -> pd.DataFrame:
    pivot = scores.pivot_table(
        index=["session_id", "location", "participant", "preset", "is_target_room"],
        columns="method", values="score",
    ).reset_index()
    pivot["diff"] = pivot["gp_refined"] - pivot["user_customized"]
    pivot["winner"] = np.where(
        pivot["diff"] > 0, "GP wins",
        np.where(pivot["diff"] < 0, "Custom wins", "Tie")
    )
    pivot["abs_diff"] = pivot["diff"].abs()
    return pivot


def _save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# =========================================================================
# Statistical tests specific to GP vs Custom
# =========================================================================
def run_stats(paired, scores, gp_diag, summary):
    results = {}
    report = []
    report.append("=" * 70)
    report.append("GP-REFINED vs USER CUSTOMIZED: DEEP-DIVE ANALYSIS")
    report.append("=" * 70)

    # --- Overall ---
    diffs = paired["diff"].values
    report.append(f"\nOverall: N={len(diffs)} location-level pairs")
    report.append(f"  GP wins: {(diffs > 0).sum()}, Ties: {(diffs == 0).sum()}, Custom wins: {(diffs < 0).sum()}")
    report.append(f"  Mean diff: {np.mean(diffs):.3f}, Median: {np.median(diffs):.1f}")
    report.append(f"  SD: {np.std(diffs, ddof=1):.3f}")

    # Wilcoxon on location-level pairs
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 5:
        stat, p = scipy_stats.wilcoxon(nonzero)
        r = scipy_stats.norm.ppf(1 - p / 2) / np.sqrt(len(nonzero)) if p < 1 else 0
        report.append(f"\n  Wilcoxon signed-rank (location-level): W={stat:.1f}, p={p:.4f}, r={r:.3f}")
        results["wilcoxon_location"] = {"W": float(stat), "p": round(float(p), 4), "r": round(r, 3)}
    else:
        report.append("  Wilcoxon: insufficient non-zero pairs")

    # Sign test (binomial)
    n_gp = int((diffs > 0).sum())
    n_custom = int((diffs < 0).sum())
    n_decided = n_gp + n_custom
    if n_decided >= 1:
        p_sign = scipy_stats.binom_test(n_gp, n_decided, 0.5) if hasattr(scipy_stats, 'binom_test') else \
                 scipy_stats.binomtest(n_gp, n_decided, 0.5).pvalue
        report.append(f"  Sign test: GP={n_gp}, Custom={n_custom}, p={p_sign:.4f}")
        results["sign_test"] = {"gp_wins": n_gp, "custom_wins": n_custom,
                                "p": round(float(p_sign), 4)}

    # --- Participant-level ---
    report.append("\n" + "-" * 60)
    report.append("BY PARTICIPANT (participant-level mean diff)")
    p_means = paired.groupby("participant")["diff"].mean()
    report.append(str(p_means.round(3)))

    p_vals = p_means.values
    if len(p_vals[p_vals != 0]) >= 2:
        stat, p = scipy_stats.wilcoxon(p_vals[p_vals != 0])
        report.append(f"\n  Wilcoxon (participant-level): W={stat:.1f}, p={p:.4f}")
        results["wilcoxon_participant"] = {"W": float(stat), "p": round(float(p), 4)}

    # Bootstrap CI on participant-level mean
    rng = np.random.default_rng(42)
    boot = np.array([np.mean(rng.choice(p_vals, len(p_vals), replace=True))
                     for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    report.append(f"  Bootstrap 95% CI on participant-level mean: [{ci_lo:.3f}, {ci_hi:.3f}]")
    results["bootstrap_participant"] = {"mean": round(float(np.mean(p_vals)), 3),
                                         "ci_lo": round(float(ci_lo), 3),
                                         "ci_hi": round(float(ci_hi), 3)}

    # --- Target vs Transfer ---
    report.append("\n" + "-" * 60)
    report.append("TARGET ROOM vs TRANSFERRED ROOMS")
    for label, mask in [("Target room", paired["is_target_room"]),
                        ("Transferred", ~paired["is_target_room"])]:
        sub = paired[mask]["diff"]
        w, l = (sub > 0).sum(), (sub < 0).sum()
        report.append(f"  {label}: mean={sub.mean():.3f}, GP wins={w}, Custom wins={l}, ties={(sub==0).sum()}")

    target_diffs = paired[paired["is_target_room"]]["diff"].values
    transfer_diffs = paired[~paired["is_target_room"]]["diff"].values
    if len(target_diffs) >= 3 and len(transfer_diffs) >= 3:
        stat, p = scipy_stats.mannwhitneyu(target_diffs, transfer_diffs, alternative="two-sided")
        report.append(f"  Mann-Whitney U (target vs transfer diff): U={stat:.1f}, p={p:.4f}")
        results["target_vs_transfer"] = {"U": float(stat), "p": round(float(p), 4),
                                          "target_mean": round(float(np.mean(target_diffs)), 3),
                                          "transfer_mean": round(float(np.mean(transfer_diffs)), 3)}

    # --- By location ---
    report.append("\n" + "-" * 60)
    report.append("BY LOCATION")
    loc_stats = paired.groupby("location").agg(
        mean_diff=("diff", "mean"),
        gp_wins=("diff", lambda x: (x > 0).sum()),
        custom_wins=("diff", lambda x: (x < 0).sum()),
        ties=("diff", lambda x: (x == 0).sum()),
        n=("diff", "count"),
    ).sort_values("mean_diff", ascending=False)
    report.append(str(loc_stats.round(3)))

    # --- GP quality as predictor ---
    report.append("\n" + "-" * 60)
    report.append("GP DIAGNOSTICS AS PREDICTORS OF GP-CUSTOM GAP")

    session_diff = paired.groupby("session_id")["diff"].mean()
    last = gp_diag.loc[gp_diag.groupby("session_id")["round"].idxmax()].set_index("session_id")
    merged = summary.set_index("session_id")[["n_rounds", "hitl_duration_s"]].join(
        last[["spearman_rho", "pairwise_accuracy", "image_variance"]]
    ).join(session_diff.rename("gp_custom_diff"))

    predictor_results = []
    for col in ["n_rounds", "spearman_rho", "pairwise_accuracy", "image_variance", "hitl_duration_s"]:
        valid = merged.dropna(subset=[col, "gp_custom_diff"])
        if len(valid) >= 4:
            rho, p = scipy_stats.spearmanr(valid[col], valid["gp_custom_diff"])
            report.append(f"  {col}: rho={rho:.3f}, p={p:.3f}")
            predictor_results.append({"variable": col, "rho": round(rho, 3), "p": round(p, 3)})
    results["predictors"] = predictor_results

    # --- Score agreement ---
    report.append("\n" + "-" * 60)
    report.append("SCORE AGREEMENT / CONCORDANCE")

    both_high = ((paired["gp_refined"] >= 6) & (paired["user_customized"] >= 6)).sum()
    both_low = ((paired["gp_refined"] <= 4) & (paired["user_customized"] <= 4)).sum()
    gp_high_only = ((paired["gp_refined"] >= 6) & (paired["user_customized"] < 6)).sum()
    custom_high_only = ((paired["gp_refined"] < 6) & (paired["user_customized"] >= 6)).sum()
    report.append(f"  Both score >= 6: {both_high}/{len(paired)} ({both_high/len(paired)*100:.0f}%)")
    report.append(f"  Both score <= 4: {both_low}/{len(paired)} ({both_low/len(paired)*100:.0f}%)")
    report.append(f"  Only GP >= 6: {gp_high_only} ({gp_high_only/len(paired)*100:.0f}%)")
    report.append(f"  Only Custom >= 6: {custom_high_only} ({custom_high_only/len(paired)*100:.0f}%)")

    rho, p = scipy_stats.spearmanr(paired["gp_refined"], paired["user_customized"])
    report.append(f"  Spearman correlation between GP and Custom scores: rho={rho:.3f}, p={p:.4f}")
    results["score_correlation"] = {"rho": round(rho, 3), "p": round(p, 4)}

    # Cohen's kappa on high/low classification
    from sklearn.metrics import cohen_kappa_score
    gp_high = (paired["gp_refined"] >= 6).astype(int)
    custom_high = (paired["user_customized"] >= 6).astype(int)
    kappa = cohen_kappa_score(gp_high, custom_high)
    report.append(f"  Cohen's kappa (high >= 6 vs not): {kappa:.3f}")
    results["cohen_kappa"] = round(kappa, 3)

    # --- Conditional: when GP has advantage ---
    report.append("\n" + "-" * 60)
    report.append("WHEN DOES GP-REFINED HAVE AN ADVANTAGE?")
    session_merged = merged.copy()
    session_merged["gp_advantage"] = session_merged["gp_custom_diff"] > 0
    for grp_label, grp in [("GP advantage", session_merged[session_merged["gp_advantage"]]),
                           ("Custom advantage", session_merged[~session_merged["gp_advantage"]])]:
        if len(grp) == 0:
            continue
        report.append(f"\n  {grp_label} sessions (n={len(grp)}):")
        for col in ["n_rounds", "spearman_rho", "pairwise_accuracy"]:
            if col in grp.columns:
                report.append(f"    {col}: mean={grp[col].mean():.3f}, median={grp[col].median():.3f}")

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)
    return results, report_text, merged


# =========================================================================
# Figures
# =========================================================================
def fig_paired_difference_distribution(paired):
    """Histogram of GP - Custom score differences."""
    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = paired["diff"].values
    bins = np.arange(-5.5, 6.5, 1)

    colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR) for d in sorted(set(diffs))]

    n, bin_edges, patches = ax.hist(diffs, bins=bins, edgecolor="white", alpha=0.8, color="#78909C")
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        center = left_edge + 0.5
        if center > 0:
            patch.set_facecolor(GP_COLOR)
        elif center < 0:
            patch.set_facecolor(CUSTOM_COLOR)
        else:
            patch.set_facecolor(TIE_COLOR)

    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    mean_d = np.mean(diffs)
    ax.axvline(mean_d, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_d:.2f}")

    ax.set_xlabel("Score Difference (GP-Refined - User Customized)")
    ax.set_ylabel("Count (locations)")
    ax.set_title("Paired Score Differences: GP-Refined vs User Customized")
    gp_patch = mpatches.Patch(color=GP_COLOR, label=f"GP wins ({(diffs>0).sum()})")
    custom_patch = mpatches.Patch(color=CUSTOM_COLOR, label=f"Custom wins ({(diffs<0).sum()})")
    tie_patch = mpatches.Patch(color=TIE_COLOR, label=f"Ties ({(diffs==0).sum()})")
    ax.legend(handles=[gp_patch, tie_patch, custom_patch, ax.lines[-1]], fontsize=9)
    _save(fig, "gp_vs_custom_diff_histogram.png")


def fig_participant_paired_bars(paired):
    """Per-participant: GP vs Custom scores side by side."""
    pm = paired.groupby("participant")[["gp_refined", "user_customized"]].mean()
    pm = pm.sort_values("gp_refined", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pm))
    w = 0.35
    ax.bar(x - w / 2, pm["gp_refined"], w, color=GP_COLOR, alpha=0.85, label="GP-Refined", edgecolor="white")
    ax.bar(x + w / 2, pm["user_customized"], w, color=CUSTOM_COLOR, alpha=0.85, label="User Customized", edgecolor="white")

    for i, (gp, cu) in enumerate(zip(pm["gp_refined"], pm["user_customized"])):
        diff = gp - cu
        symbol = "+" if diff > 0 else ""
        ax.text(i, max(gp, cu) + 0.15, f"{symbol}{diff:.2f}", ha="center", fontsize=8,
                color=GP_COLOR if diff > 0 else (CUSTOM_COLOR if diff < 0 else "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(pm.index, fontsize=10)
    ax.set_ylabel("Mean Score")
    ax.set_title("GP-Refined vs User Customized: Per Participant")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, "gp_vs_custom_by_participant.png")


def fig_location_breakdown(paired):
    """Per-location win/tie/loss stacked bars."""
    loc_stats = paired.groupby("location").agg(
        gp_wins=("diff", lambda x: (x > 0).sum()),
        ties=("diff", lambda x: (x == 0).sum()),
        custom_wins=("diff", lambda x: (x < 0).sum()),
    ).sort_values("gp_wins", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(loc_stats))
    ax.barh(y, loc_stats["gp_wins"], color=GP_COLOR, alpha=0.85, label="GP wins", edgecolor="white")
    ax.barh(y, loc_stats["ties"], left=loc_stats["gp_wins"], color=TIE_COLOR,
            alpha=0.6, label="Ties", edgecolor="white")
    ax.barh(y, loc_stats["custom_wins"],
            left=loc_stats["gp_wins"] + loc_stats["ties"],
            color=CUSTOM_COLOR, alpha=0.85, label="Custom wins", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(loc_stats.index, fontsize=10)
    ax.set_xlabel("Number of Sessions")
    ax.set_title("GP-Refined vs User Customized: Win/Tie/Loss by Location")
    ax.legend(fontsize=9)
    _save(fig, "gp_vs_custom_by_location.png")


def fig_target_vs_transfer(paired):
    """Compare diff distribution in target room vs transferred rooms."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (label, mask) in zip(axes, [("Target Room", paired["is_target_room"]),
                                        ("Transferred Rooms", ~paired["is_target_room"])]):
        sub = paired[mask]["diff"].values
        bins = np.arange(-5.5, 6.5, 1)
        n, _, patches = ax.hist(sub, bins=bins, edgecolor="white", alpha=0.8)
        for patch, left in zip(patches, bins[:-1]):
            c = left + 0.5
            patch.set_facecolor(GP_COLOR if c > 0 else (CUSTOM_COLOR if c < 0 else TIE_COLOR))

        mean_d = np.mean(sub)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.axvline(mean_d, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label}\nmean diff = {mean_d:.2f}, n={len(sub)}")
        ax.set_xlabel("GP - Custom")

    axes[0].set_ylabel("Count")
    fig.suptitle("GP vs Custom: Target Room vs Style Transfer", fontsize=12, y=1.04)
    fig.tight_layout()
    _save(fig, "gp_vs_custom_target_transfer.png")


def fig_scatter_paired(paired):
    """Scatter: GP score vs Custom score, each point is a location."""
    fig, ax = plt.subplots(figsize=(6, 6))

    jitter_rng = np.random.default_rng(0)
    jx = jitter_rng.uniform(-0.15, 0.15, len(paired))
    jy = jitter_rng.uniform(-0.15, 0.15, len(paired))

    colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR)
              for d in paired["diff"]]
    ax.scatter(paired["user_customized"] + jx, paired["gp_refined"] + jy,
               c=colors, s=40, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=5)

    ax.plot([0.5, 7.5], [0.5, 7.5], "k--", alpha=0.3, label="Equal")
    ax.set_xlabel("User Customized Score")
    ax.set_ylabel("GP-Refined Score")
    ax.set_title("Score Concordance: GP-Refined vs User Customized")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_aspect("equal")

    gp_p = mpatches.Patch(color=GP_COLOR, label="GP higher")
    cu_p = mpatches.Patch(color=CUSTOM_COLOR, label="Custom higher")
    ti_p = mpatches.Patch(color=TIE_COLOR, label="Tie")
    ax.legend(handles=[gp_p, ti_p, cu_p], fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)
    _save(fig, "gp_vs_custom_scatter.png")


def fig_gp_diagnostics_vs_gap(merged):
    """Scatter: GP model quality predicting whether GP > Custom."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    predictors = [
        ("spearman_rho", "Final Spearman ρ"),
        ("pairwise_accuracy", "Final Pairwise Accuracy"),
        ("n_rounds", "Number of Rounds"),
    ]
    for ax, (col, xlabel) in zip(axes, predictors):
        valid = merged.dropna(subset=[col, "gp_custom_diff"])
        colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR)
                  for d in valid["gp_custom_diff"]]
        ax.scatter(valid[col], valid["gp_custom_diff"], c=colors, s=60,
                   edgecolors="white", zorder=5)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

        for sid, row in valid.iterrows():
            short = sid.split("_")[1]
            ax.annotate(short, (row[col], row["gp_custom_diff"]),
                        fontsize=7, alpha=0.6, xytext=(4, 4),
                        textcoords="offset points")

        if len(valid) >= 4:
            rho, p = scipy_stats.spearmanr(valid[col], valid["gp_custom_diff"])
            ax.text(0.05, 0.95, f"ρ={rho:.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean (GP - Custom)")
        ax.set_title(f"GP Advantage vs {xlabel}")

    fig.tight_layout()
    _save(fig, "gp_diagnostics_vs_gap.png")


# =========================================================================
# main
# =========================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scores, gp_diag, summary, drift = load()
    paired = build_paired(scores)

    print("Running GP vs Custom deep-dive analysis...")
    results, report_text, merged = run_stats(paired, scores, gp_diag, summary)

    with open(OUTPUT_DIR / "gp_vs_custom_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUTPUT_DIR / "gp_vs_custom_report.txt", "w") as f:
        f.write(report_text)

    print(report_text)

    print("\nGenerating GP vs Custom figures...")
    fig_paired_difference_distribution(paired)
    fig_participant_paired_bars(paired)
    fig_location_breakdown(paired)
    fig_target_vs_transfer(paired)
    fig_scatter_paired(paired)
    fig_gp_diagnostics_vs_gap(merged)

    print(f"\nReport: {OUTPUT_DIR / 'gp_vs_custom_report.txt'}")
    print(f"Data:   {OUTPUT_DIR / 'gp_vs_custom_results.json'}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
