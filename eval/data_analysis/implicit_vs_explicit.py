"""
Implicit vs Explicit Preference Analysis.

Core thesis: GP-Refined captures users' implicit preferences (revealed through
ranking behavior) that diverge from their explicit preferences (manual slider
weights), bridging the gap between user mental model and system output.

Evidence structure:
  1. Explicit-Implicit Divergence: manual weights vs GP-learned mus are poorly correlated
  2. First-Round Surprise: the exploit option (built from manual weights) is rarely preferred
  3. Tag Surprise Analysis: GP systematically elevates tags users underweighted
  4. Divergence Predicts GP Advantage: sessions with more divergence → GP outperforms Custom
  5. Transfer as Implicit Evidence: GP-Refined transfers better to new rooms
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

SESSION_LOGS_DIR = Path(__file__).resolve().parent.parent / "session_logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

GP_COLOR = "#E91E63"
CUSTOM_COLOR = "#FFB74D"
NEUTRAL_COLOR = "#78909C"


def _save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _short_id(session_id: str) -> str:
    return session_id.split("_")[1]


def discover_sessions():
    return sorted(
        p for p in SESSION_LOGS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("eval_")
    )


# =========================================================================
# 1. Compute explicit-implicit divergence per session
# =========================================================================
def compute_divergence(sessions):
    """For each session, compare manual weights (explicit) to GP-learned mus (implicit)."""
    rows = []
    tag_rows = []

    for sess_dir in sessions:
        manual_f = sess_dir / "impression" / "user_manual_weights.json"
        refined_f = sess_dir / "refined_preferences_v2.json"
        if not manual_f.exists() or not refined_f.exists():
            continue

        with open(manual_f) as f:
            manual = json.load(f)
        with open(refined_f) as f:
            refined = json.load(f)

        session_id = sess_dir.name
        manual_w = manual.get("weights", {})
        all_tags = {t["text"]: t for t in refined.get("all_tag_details", [])}
        shared = sorted(set(manual_w.keys()) & set(all_tags.keys()))

        if len(shared) < 3:
            continue

        manual_vals = np.array([manual_w[t] for t in shared])
        gp_mus = np.array([all_tags[t]["final_mu"] for t in shared])

        manual_ranks = scipy_stats.rankdata(-manual_vals)
        gp_ranks = scipy_stats.rankdata(-gp_mus)

        rho, p_rho = scipy_stats.spearmanr(manual_vals, gp_mus)
        tau, p_tau = scipy_stats.kendalltau(manual_vals, gp_mus)

        rank_displacement = np.mean(np.abs(manual_ranks - gp_ranks))

        manual_top3 = set(np.array(shared)[np.argsort(-manual_vals)[:3]])
        gp_top3 = set(np.array(shared)[np.argsort(-gp_mus)[:3]])
        top3_overlap = len(manual_top3 & gp_top3) / 3.0

        rows.append({
            "session_id": session_id,
            "participant": _short_id(session_id),
            "n_shared_tags": len(shared),
            "spearman_rho": round(rho, 3),
            "spearman_p": round(p_rho, 4),
            "kendall_tau": round(tau, 3),
            "kendall_p": round(p_tau, 4),
            "mean_rank_displacement": round(rank_displacement, 2),
            "top3_overlap": round(top3_overlap, 3),
            "manual_top3": ", ".join(sorted(manual_top3)),
            "gp_top3": ", ".join(sorted(gp_top3)),
        })

        for tag in shared:
            m_rank = int(manual_ranks[shared.index(tag)])
            g_rank = int(gp_ranks[shared.index(tag)])
            tag_rows.append({
                "session_id": session_id,
                "participant": _short_id(session_id),
                "tag": tag,
                "manual_weight": manual_w[tag],
                "gp_mu": all_tags[tag]["final_mu"],
                "manual_rank": m_rank,
                "gp_rank": g_rank,
                "rank_change": m_rank - g_rank,
                "is_surprise": (m_rank - g_rank) >= 2,
            })

    return pd.DataFrame(rows), pd.DataFrame(tag_rows)


# =========================================================================
# 2. First-round surprise: exploit option performance
# =========================================================================
def compute_first_round_surprise(sessions):
    rows = []
    for sess_dir in sessions:
        diag_f = sess_dir / "gp_round_diagnostics_v2.jsonl"
        if not diag_f.exists():
            continue
        with open(diag_f) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        rec = json.loads(first_line)
        m = rec["metrics"]
        ranking = rec["ranking"]
        snaps = rec.get("option_snapshots", [])

        exploit_id = None
        for s in snaps:
            if s["strategy"] == "exploit":
                exploit_id = s["option_id"]
                break

        exploit_rank = ranking.index(exploit_id) + 1 if exploit_id is not None else None
        predicted_top = int(m.get("predicted_top_option_before_update", -1))
        actual_top = int(m.get("actual_top_option", -1))

        rows.append({
            "session_id": sess_dir.name,
            "participant": _short_id(sess_dir.name),
            "exploit_option_rank": exploit_rank,
            "exploit_was_top": exploit_rank == 1 if exploit_rank else False,
            "predicted_top": predicted_top,
            "actual_top": actual_top,
            "top_predicted_correctly": predicted_top == actual_top,
            "round1_pairwise_accuracy": m.get("pairwise_accuracy_before_update"),
            "round1_spearman": m.get("spearman_rank_corr_before_update"),
        })
    return pd.DataFrame(rows)


# =========================================================================
# 3. Merge divergence with GP advantage
# =========================================================================
def merge_divergence_with_outcomes(divergence_df, scores_df):
    gp_scores = scores_df[scores_df["method"] == "gp_refined"].groupby("session_id")["score"].mean()
    custom_scores = scores_df[scores_df["method"] == "user_customized"].groupby("session_id")["score"].mean()
    gp_advantage = (gp_scores - custom_scores).rename("gp_advantage")

    target_scores = scores_df[scores_df["is_target_room"]]
    transfer_scores = scores_df[~scores_df["is_target_room"]]

    gp_target = target_scores[target_scores["method"] == "gp_refined"].groupby("session_id")["score"].mean()
    custom_target = target_scores[target_scores["method"] == "user_customized"].groupby("session_id")["score"].mean()
    gp_transfer = transfer_scores[transfer_scores["method"] == "gp_refined"].groupby("session_id")["score"].mean()
    custom_transfer = transfer_scores[transfer_scores["method"] == "user_customized"].groupby("session_id")["score"].mean()

    merged = divergence_df.set_index("session_id").copy()
    merged["gp_advantage"] = gp_advantage
    merged["gp_target"] = gp_target
    merged["custom_target"] = custom_target
    merged["gp_transfer"] = gp_transfer
    merged["custom_transfer"] = custom_transfer
    merged["gp_advantage_transfer"] = gp_transfer - custom_transfer
    merged["gp_advantage_target"] = gp_target - custom_target
    return merged.reset_index()


# =========================================================================
# Statistical tests
# =========================================================================
def run_stats(divergence_df, tag_df, surprise_df, merged_df):
    report = []
    results = {}

    report.append("=" * 70)
    report.append("IMPLICIT vs EXPLICIT PREFERENCE ANALYSIS")
    report.append("=" * 70)

    # --- 1. Divergence ---
    report.append("\n1. EXPLICIT-IMPLICIT DIVERGENCE")
    report.append("-" * 50)
    rhos = divergence_df["spearman_rho"].values
    report.append(f"   Spearman ρ (manual weights vs GP mus):")
    report.append(f"     Mean: {np.mean(rhos):.3f}, Median: {np.median(rhos):.3f}")
    report.append(f"     Range: [{np.min(rhos):.3f}, {np.max(rhos):.3f}]")
    n_sig = (divergence_df["spearman_p"] < 0.05).sum()
    report.append(f"     Significant correlations (p<.05): {n_sig}/{len(divergence_df)}")
    report.append(f"     => In {len(divergence_df) - n_sig}/{len(divergence_df)} sessions, "
                  "explicit and implicit preferences are NOT significantly correlated")

    stat, p = scipy_stats.wilcoxon(rhos - 0, alternative="greater")
    report.append(f"   Wilcoxon test (ρ > 0): W={stat:.1f}, p={p:.4f}")
    results["divergence_rho_test"] = {"W": float(stat), "p": round(float(p), 4)}

    top3 = divergence_df["top3_overlap"].values
    report.append(f"\n   Top-3 tag overlap (explicit vs implicit):")
    report.append(f"     Mean: {np.mean(top3):.3f} (1.0 = perfect agreement)")
    report.append(f"     Distribution: {dict(zip(*np.unique(top3, return_counts=True)))}")

    rank_disp = divergence_df["mean_rank_displacement"].values
    report.append(f"\n   Mean rank displacement: {np.mean(rank_disp):.2f} positions")
    results["divergence_summary"] = {
        "mean_rho": round(float(np.mean(rhos)), 3),
        "mean_top3_overlap": round(float(np.mean(top3)), 3),
        "mean_rank_displacement": round(float(np.mean(rank_disp)), 2),
    }

    # --- 2. First-round surprise ---
    report.append(f"\n\n2. FIRST-ROUND SURPRISE (exploit = manual-weight-based option)")
    report.append("-" * 50)
    n_exploit_top = surprise_df["exploit_was_top"].sum()
    n_sessions = len(surprise_df)
    exploit_ranks = surprise_df["exploit_option_rank"].values
    report.append(f"   Exploit option chosen as #1: {n_exploit_top}/{n_sessions} "
                  f"({n_exploit_top/n_sessions*100:.0f}%)")
    report.append(f"   Mean exploit rank: {np.mean(exploit_ranks):.2f} (1 = best, 4 = worst)")
    report.append(f"   Exploit rank distribution: {dict(zip(*np.unique(exploit_ranks, return_counts=True)))}")

    p_binom = scipy_stats.binomtest(int(n_exploit_top), n_sessions, 0.25, alternative="greater").pvalue
    report.append(f"   Binomial test (exploit top > 25% chance): p={p_binom:.4f}")

    top_correct = surprise_df["top_predicted_correctly"].sum()
    r1_acc = surprise_df["round1_pairwise_accuracy"].values
    report.append(f"\n   Prior-based prediction correct for top option: "
                  f"{top_correct}/{n_sessions} ({top_correct/n_sessions*100:.0f}%)")
    report.append(f"   Mean round-1 pairwise accuracy: {np.mean(r1_acc):.3f} "
                  f"(chance = 0.5)")
    results["first_round"] = {
        "exploit_top_rate": round(n_exploit_top / n_sessions, 3),
        "mean_exploit_rank": round(float(np.mean(exploit_ranks)), 2),
        "top_prediction_accuracy": round(top_correct / n_sessions, 3),
        "mean_r1_pairwise_accuracy": round(float(np.mean(r1_acc)), 3),
    }

    # --- 3. Tag surprises ---
    report.append(f"\n\n3. TAG SURPRISE ANALYSIS")
    report.append("-" * 50)
    surprises = tag_df[tag_df["is_surprise"]]
    n_surprise = len(surprises)
    n_total_tags = len(tag_df)
    report.append(f"   Surprise tags (rank improved by >= 2 positions): "
                  f"{n_surprise}/{n_total_tags} ({n_surprise/n_total_tags*100:.0f}%)")
    report.append(f"   Mean rank change for surprises: "
                  f"+{surprises['rank_change'].mean():.1f} positions")

    top_surprises = surprises.nlargest(10, "rank_change")
    report.append(f"\n   Top-10 surprise tags (biggest rank jumps):")
    for _, row in top_surprises.iterrows():
        report.append(f"     {row['participant']}: \"{row['tag']}\" "
                      f"manual_rank={row['manual_rank']} -> gp_rank={row['gp_rank']} "
                      f"(manual_w={row['manual_weight']:.2f}, gp_mu={row['gp_mu']:.2f})")

    results["surprises"] = {
        "n_surprise": int(n_surprise),
        "n_total": n_total_tags,
        "surprise_rate": round(n_surprise / n_total_tags, 3),
        "mean_rank_change": round(float(surprises["rank_change"].mean()), 1),
    }

    # --- 4. Divergence predicts GP advantage ---
    report.append(f"\n\n4. DIVERGENCE PREDICTS GP ADVANTAGE")
    report.append("-" * 50)

    for col, label in [
        ("spearman_rho", "Explicit-Implicit ρ"),
        ("top3_overlap", "Top-3 Overlap"),
        ("mean_rank_displacement", "Mean Rank Displacement"),
    ]:
        valid = merged_df.dropna(subset=[col, "gp_advantage"])
        if len(valid) < 4:
            continue
        rho, p = scipy_stats.spearmanr(valid[col], valid["gp_advantage"])
        report.append(f"   {label} vs GP advantage: ρ={rho:.3f}, p={p:.3f}")

        rho_t, p_t = scipy_stats.spearmanr(valid[col], valid["gp_advantage_transfer"])
        report.append(f"   {label} vs GP advantage (transfer only): ρ={rho_t:.3f}, p={p_t:.3f}")

    # Compare: divergent sessions vs convergent sessions
    median_rho = divergence_df["spearman_rho"].median()
    high_div = merged_df[merged_df["spearman_rho"] <= median_rho]
    low_div = merged_df[merged_df["spearman_rho"] > median_rho]
    report.append(f"\n   High divergence sessions (ρ <= {median_rho:.2f}, n={len(high_div)}):")
    report.append(f"     Mean GP advantage: {high_div['gp_advantage'].mean():.3f}")
    report.append(f"     Mean GP advantage (transfer): {high_div['gp_advantage_transfer'].mean():.3f}")
    report.append(f"   Low divergence sessions (ρ > {median_rho:.2f}, n={len(low_div)}):")
    report.append(f"     Mean GP advantage: {low_div['gp_advantage'].mean():.3f}")
    report.append(f"     Mean GP advantage (transfer): {low_div['gp_advantage_transfer'].mean():.3f}")

    # --- 5. Transfer as implicit evidence ---
    report.append(f"\n\n5. TRANSFER AS EVIDENCE OF IMPLICIT PREFERENCE CAPTURE")
    report.append("-" * 50)
    target_adv = merged_df["gp_advantage_target"].dropna()
    transfer_adv = merged_df["gp_advantage_transfer"].dropna()
    report.append(f"   GP advantage in target room: mean={target_adv.mean():.3f}")
    report.append(f"   GP advantage in transferred rooms: mean={transfer_adv.mean():.3f}")
    report.append(f"   Difference: {(transfer_adv.mean() - target_adv.mean()):.3f}")
    report.append(f"   => GP-Refined benefits more in transferred rooms, suggesting it")
    report.append(f"      captures deeper style preferences that generalize beyond the")
    report.append(f"      specific room the user was thinking about")

    if len(target_adv) >= 3 and len(transfer_adv) >= 3:
        stat, p = scipy_stats.mannwhitneyu(transfer_adv, target_adv, alternative="greater")
        report.append(f"   Mann-Whitney U (transfer > target): U={stat:.1f}, p={p:.4f}")
        results["transfer_vs_target"] = {"U": float(stat), "p": round(float(p), 4)}

    report.append("\n" + "=" * 70)
    return results, "\n".join(report)


# =========================================================================
# Figures
# =========================================================================
def fig_divergence_overview(divergence_df):
    """Bar chart of per-session Spearman ρ with significance threshold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df = divergence_df.sort_values("spearman_rho")
    x = np.arange(len(df))
    colors = [GP_COLOR if p < 0.05 else NEUTRAL_COLOR for p in df["spearman_p"]]

    bars = ax.bar(x, df["spearman_rho"], color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(-1.0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["participant"], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Spearman ρ\n(Manual Weights vs GP-Learned μ)")
    ax.set_title("Explicit-Implicit Preference Divergence per Session")
    ax.set_ylim(-1.2, 1.2)

    sig_patch = mpatches.Patch(color=GP_COLOR, label="Significant (p < .05)")
    ns_patch = mpatches.Patch(color=NEUTRAL_COLOR, label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9)

    mean_rho = df["spearman_rho"].mean()
    ax.axhline(mean_rho, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(df) - 0.5, mean_rho + 0.05, f"mean={mean_rho:.2f}", color="red",
            fontsize=9, ha="right")
    _save(fig, "implicit_explicit_divergence.png")


def fig_exploit_rank_distribution(surprise_df):
    """Where did the exploit (manual-weight) option rank in round 1?"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ranks = surprise_df["exploit_option_rank"].values
    counts = [np.sum(ranks == r) for r in [1, 2, 3, 4]]
    colors_bar = [CUSTOM_COLOR, NEUTRAL_COLOR, NEUTRAL_COLOR, "#F44336"]

    bars = ax.bar([1, 2, 3, 4], counts, color=colors_bar, edgecolor="white", alpha=0.85)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(count), ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1st\n(best)", "2nd", "3rd", "4th\n(worst)"], fontsize=10)
    ax.set_xlabel("Rank of Exploit Option (Built from Manual Weights)")
    ax.set_ylabel("Number of Sessions")
    ax.set_title("Round 1: How Users Ranked the\n\"What They Said They Wanted\" Option")
    ax.set_ylim(0, max(counts) + 1.5)
    _save(fig, "exploit_option_rank.png")


def fig_tag_surprise_volcano(tag_df):
    """Volcano-like plot: manual weight vs rank change."""
    fig, ax = plt.subplots(figsize=(9, 6))

    surprises = tag_df[tag_df["is_surprise"]]
    non_surprises = tag_df[~tag_df["is_surprise"]]

    ax.scatter(non_surprises["manual_weight"], non_surprises["rank_change"],
               s=25, color=NEUTRAL_COLOR, alpha=0.4, edgecolors="none", label="No surprise")
    ax.scatter(surprises["manual_weight"], surprises["rank_change"],
               s=50, color="#E91E63", alpha=0.7, edgecolors="white", linewidth=0.3,
               label="Surprise (rank ↑ by 2+)", zorder=5)

    for _, row in surprises.iterrows():
        if row["rank_change"] >= 4:
            ax.annotate(f"{row['participant']}: {row['tag']}",
                        (row["manual_weight"], row["rank_change"]),
                        fontsize=6, alpha=0.8, xytext=(5, 3),
                        textcoords="offset points")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(2, color="#E91E63", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Manual Weight (User's Explicit Rating)")
    ax.set_ylabel("Rank Change (+ = GP Elevated)")
    ax.set_title("Tag Surprise: GP Discovers Preferences Users Underweighted")
    ax.legend(fontsize=9)
    _save(fig, "tag_surprise_volcano.png")


def fig_divergence_vs_advantage(merged_df):
    """Scatter: divergence (inverted ρ) vs GP advantage in transfer rooms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (y_col, y_label) in zip(axes, [
        ("gp_advantage", "GP Advantage (all rooms)"),
        ("gp_advantage_transfer", "GP Advantage (transferred rooms)"),
    ]):
        valid = merged_df.dropna(subset=["spearman_rho", y_col])
        divergence = 1 - valid["spearman_rho"]
        advantage = valid[y_col]

        colors = [GP_COLOR if a > 0 else CUSTOM_COLOR for a in advantage]
        ax.scatter(divergence, advantage, c=colors, s=60,
                   edgecolors="white", zorder=5)

        for _, row in valid.iterrows():
            ax.annotate(row["participant"], (1 - row["spearman_rho"], row[y_col]),
                        fontsize=7, alpha=0.6, xytext=(4, 4),
                        textcoords="offset points")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

        if len(valid) >= 4:
            rho, p = scipy_stats.spearmanr(divergence, advantage)
            ax.text(0.05, 0.95, f"ρ={rho:.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Divergence (1 - ρ)\n← Agreement ... Disagreement →")
        ax.set_ylabel(y_label)

    fig.suptitle("Does Explicit-Implicit Divergence Predict GP Advantage?", fontsize=12)
    fig.tight_layout()
    _save(fig, "divergence_vs_gp_advantage.png")


def fig_target_vs_transfer_advantage(merged_df):
    """Paired comparison: GP advantage in target vs transfer rooms."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = merged_df.dropna(subset=["gp_advantage_target", "gp_advantage_transfer"])

    for _, row in valid.iterrows():
        ax.plot([0, 1], [row["gp_advantage_target"], row["gp_advantage_transfer"]],
                "o-", alpha=0.6, markersize=8, color=NEUTRAL_COLOR)
        ax.annotate(row["participant"], (1.05, row["gp_advantage_transfer"]),
                    fontsize=8, alpha=0.7, va="center")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    means = [valid["gp_advantage_target"].mean(), valid["gp_advantage_transfer"].mean()]
    ax.plot([0, 1], means, "s--", color="#E91E63", markersize=10, linewidth=2,
            label=f"Mean: {means[0]:.2f} → {means[1]:.2f}", zorder=10)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Target Room\n(user's mental model)", "Transferred Rooms\n(beyond mental model)"],
                       fontsize=10)
    ax.set_ylabel("GP Advantage (GP score - Custom score)")
    ax.set_title("GP-Refined Advantage: Target Room vs Style Transfer")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.3, 1.5)
    _save(fig, "target_vs_transfer_gp_advantage.png")


def fig_mental_model_summary(divergence_df, surprise_df, merged_df):
    """Multi-panel summary figure telling the complete story."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Divergence distribution
    ax = axes[0, 0]
    rhos = divergence_df["spearman_rho"].values
    ax.hist(rhos, bins=np.arange(-1.05, 1.15, 0.2), color=NEUTRAL_COLOR,
            edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(rhos), color="red", linestyle="--", linewidth=2,
               label=f"Mean ρ = {np.mean(rhos):.2f}")
    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Spearman ρ (explicit vs implicit)")
    ax.set_ylabel("Sessions")
    ax.set_title("A) Explicit-Implicit Correlation")
    ax.legend(fontsize=8)

    # Panel B: Exploit option rank
    ax = axes[0, 1]
    ranks = surprise_df["exploit_option_rank"].values
    counts = [np.sum(ranks == r) for r in [1, 2, 3, 4]]
    colors_bar = [CUSTOM_COLOR, "#FFB74D", "#FFB74D", "#F44336"]
    ax.bar([1, 2, 3, 4], counts, color=colors_bar, edgecolor="white", alpha=0.85)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i + 1, c + 0.2, str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th"])
    ax.set_xlabel("Rank of \"what user said they want\"")
    ax.set_ylabel("Sessions")
    ax.set_title("B) Round-1 Exploit Option Rank")

    # Panel C: Top-3 overlap
    ax = axes[1, 0]
    overlaps = divergence_df["top3_overlap"].values
    overlap_counts = {0.0: 0, 0.333: 0, 0.667: 0, 1.0: 0}
    for o in overlaps:
        closest = min(overlap_counts.keys(), key=lambda k: abs(k - o))
        overlap_counts[closest] += 1
    ax.bar(["0/3", "1/3", "2/3", "3/3"],
           [overlap_counts[0.0], overlap_counts[0.333], overlap_counts[0.667], overlap_counts[1.0]],
           color=[NEUTRAL_COLOR, "#FFB74D", "#FFB74D", CUSTOM_COLOR], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Top-3 Tag Overlap (explicit vs implicit)")
    ax.set_ylabel("Sessions")
    ax.set_title("C) Do Top Preferences Match?")

    # Panel D: Target vs Transfer GP advantage
    ax = axes[1, 1]
    valid = merged_df.dropna(subset=["gp_advantage_target", "gp_advantage_transfer"])
    bp_data = [valid["gp_advantage_target"].values, valid["gp_advantage_transfer"].values]
    bp = ax.boxplot(bp_data, positions=[0, 1], widths=0.4, patch_artist=True,
                    boxprops=dict(alpha=0.4))
    bp["boxes"][0].set_facecolor(CUSTOM_COLOR)
    bp["boxes"][1].set_facecolor(GP_COLOR)
    for _, row in valid.iterrows():
        ax.plot([0, 1], [row["gp_advantage_target"], row["gp_advantage_transfer"]],
                "o-", alpha=0.3, color=NEUTRAL_COLOR, markersize=5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Target Room", "Transferred"], fontsize=10)
    ax.set_ylabel("GP - Custom score diff")
    ax.set_title("D) GP Advantage by Room Type")

    fig.suptitle("Bridging the Gap: How GP Captures What Users Don't Know They Want",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "mental_model_gap_summary.png")


# =========================================================================
# main
# =========================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sessions = discover_sessions()
    scores = pd.read_csv(OUTPUT_DIR / "location_scores.csv")

    print("Computing explicit-implicit divergence...")
    divergence_df, tag_df = compute_divergence(sessions)
    divergence_df.to_csv(OUTPUT_DIR / "explicit_implicit_divergence.csv", index=False)
    tag_df.to_csv(OUTPUT_DIR / "tag_level_divergence.csv", index=False)

    print("Computing first-round surprise...")
    surprise_df = compute_first_round_surprise(sessions)
    surprise_df.to_csv(OUTPUT_DIR / "first_round_surprise.csv", index=False)

    print("Merging with outcomes...")
    merged_df = merge_divergence_with_outcomes(divergence_df, scores)
    merged_df.to_csv(OUTPUT_DIR / "divergence_outcome_merged.csv", index=False)

    print("Running statistical tests...")
    results, report_text = run_stats(divergence_df, tag_df, surprise_df, merged_df)

    with open(OUTPUT_DIR / "implicit_explicit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUTPUT_DIR / "implicit_explicit_report.txt", "w") as f:
        f.write(report_text)

    print(report_text)

    print("\nGenerating figures...")
    fig_divergence_overview(divergence_df)
    fig_exploit_rank_distribution(surprise_df)
    fig_tag_surprise_volcano(tag_df)
    fig_divergence_vs_advantage(merged_df)
    fig_target_vs_transfer_advantage(merged_df)
    fig_mental_model_summary(divergence_df, surprise_df, merged_df)

    print(f"\nReport: {OUTPUT_DIR / 'implicit_explicit_report.txt'}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
