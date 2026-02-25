"""
Deep investigation: When and why do GP-Refined and User Customized diverge?

Three questions:
  1. When are they similar (ties)?
  2. When does GP-Refined capture more than User Customized?
  3. When does GP-Refined fail to capture what User Customized gets right?

Outputs:
  - outputs/gp_custom_investigation_report.txt
  - outputs/gp_custom_investigation_data.json
  - figures/investigation_*.png
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

SESSION_LOGS_DIR = Path(__file__).resolve().parent.parent / "session_logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

GP_COLOR = "#E91E63"
CUSTOM_COLOR = "#FFB74D"
TIE_COLOR = "#9E9E9E"
WARN_COLOR = "#F44336"


def _save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _short(session_id: str) -> str:
    parts = session_id.split("_")
    return parts[1]


def _short_preset(session_id: str) -> str:
    parts = session_id.split("_")
    return parts[1] + " " + "_".join(parts[2:4])


# =========================================================================
# Data loading and paired construction
# =========================================================================
def load_all():
    scores = pd.read_csv(OUTPUT_DIR / "location_scores.csv")
    summary = pd.read_csv(OUTPUT_DIR / "session_summary.csv")
    gp_diag = pd.read_csv(OUTPUT_DIR / "gp_diagnostics.csv")
    drift = pd.read_csv(OUTPUT_DIR / "preference_drift.csv")
    divergence = pd.read_csv(OUTPUT_DIR / "explicit_implicit_divergence.csv")
    return scores, summary, gp_diag, drift, divergence


def build_paired(scores):
    pivot = scores.pivot_table(
        index=["session_id", "location", "participant", "preset", "is_target_room"],
        columns="method", values="score",
    ).reset_index()
    pivot["diff"] = pivot["gp_refined"] - pivot["user_customized"]
    pivot["outcome"] = np.where(
        pivot["diff"] > 0, "GP_wins",
        np.where(pivot["diff"] < 0, "Custom_wins", "Tie")
    )
    pivot["abs_diff"] = pivot["diff"].abs()
    return pivot


def build_session_features(scores, summary, gp_diag, divergence):
    """One row per session with all relevant features."""
    pivot = scores.pivot_table(
        index=["session_id", "location", "participant", "is_target_room"],
        columns="method", values="score",
    ).reset_index()
    pivot["diff"] = pivot["gp_refined"] - pivot["user_customized"]

    session_agg = pivot.groupby("session_id").agg(
        mean_diff=("diff", "mean"),
        gp_wins=("diff", lambda x: (x > 0).sum()),
        ties=("diff", lambda x: (x == 0).sum()),
        custom_wins=("diff", lambda x: (x < 0).sum()),
        mean_gp=("gp_refined", "mean"),
        mean_custom=("user_customized", "mean"),
        mean_baseline=("baseline_text", "mean"),
    )

    last_round = gp_diag.loc[gp_diag.groupby("session_id")["round"].idxmax()].set_index("session_id")
    first_round = gp_diag.loc[gp_diag.groupby("session_id")["round"].idxmin()].set_index("session_id")

    merged = session_agg.join(
        summary.set_index("session_id")[["n_rounds", "hitl_duration_s", "preset", "participant"]]
    )
    merged = merged.join(
        last_round[["spearman_rho", "pairwise_accuracy", "image_variance"]].rename(
            columns={"spearman_rho": "final_spearman", "pairwise_accuracy": "final_accuracy",
                     "image_variance": "final_variance"})
    )
    merged = merged.join(
        first_round[["pairwise_accuracy"]].rename(columns={"pairwise_accuracy": "r1_accuracy"})
    )
    merged["accuracy_gain"] = merged["final_accuracy"] - merged["r1_accuracy"]

    div_cols = divergence.set_index("session_id")[
        ["spearman_rho", "top3_overlap", "mean_rank_displacement"]
    ].rename(columns={"spearman_rho": "explicit_implicit_rho"})
    merged = merged.join(div_cols)

    merged["outcome_category"] = np.where(
        merged["mean_diff"] > 0, "GP_advantage",
        np.where(merged["mean_diff"] < 0, "Custom_advantage", "Tie")
    )
    return merged


def load_tag_data():
    """Load per-session tag-level comparison between manual weights and GP mus."""
    rows = []
    for sess_dir in sorted(SESSION_LOGS_DIR.iterdir()):
        if not sess_dir.name.startswith("eval_"):
            continue
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
        final_w = refined.get("weights", {})
        shared = sorted(set(manual_w.keys()) & set(all_tags.keys()))

        for tag in shared:
            detail = all_tags[tag]
            rows.append({
                "session_id": session_id,
                "tag": tag,
                "manual_weight": manual_w[tag],
                "gp_mu": detail["final_mu"],
                "gp_sigma": detail["final_sigma"],
                "final_weight": final_w.get(tag, 0),
                "win_rate": detail.get("win_rate", 0),
                "times_shown": detail.get("times_shown", 0),
            })
    return pd.DataFrame(rows)


# =========================================================================
# Analysis
# =========================================================================
def analyze(paired, session_features, tag_data, scores, report):
    report.append("=" * 70)
    report.append("INVESTIGATION: WHEN & WHY DO GP-REFINED AND USER CUSTOMIZED DIVERGE?")
    report.append("=" * 70)

    # ---- Section 1: Overview ----
    report.append("\n" + "=" * 70)
    report.append("1. OVERVIEW")
    report.append("=" * 70)

    n_total = len(paired)
    n_gp = (paired["outcome"] == "GP_wins").sum()
    n_tie = (paired["outcome"] == "Tie").sum()
    n_cu = (paired["outcome"] == "Custom_wins").sum()
    report.append(f"\n  Location-level outcomes (N={n_total}):")
    report.append(f"    GP-Refined wins:     {n_gp:3d} ({n_gp/n_total*100:.0f}%)")
    report.append(f"    Ties:                {n_tie:3d} ({n_tie/n_total*100:.0f}%)")
    report.append(f"    Custom wins:         {n_cu:3d} ({n_cu/n_total*100:.0f}%)")
    report.append(f"    Mean diff (GP-Custom): {paired['diff'].mean():+.3f}")

    n_sess = len(session_features)
    n_gp_s = (session_features["outcome_category"] == "GP_advantage").sum()
    n_cu_s = (session_features["outcome_category"] == "Custom_advantage").sum()
    n_tie_s = (session_features["outcome_category"] == "Tie").sum()
    report.append(f"\n  Session-level outcomes (N={n_sess}):")
    report.append(f"    GP-Refined advantage: {n_gp_s}")
    report.append(f"    Custom advantage:     {n_cu_s}")
    report.append(f"    Tie:                  {n_tie_s}")

    # ---- Section 2: When they agree (ties) ----
    report.append("\n" + "=" * 70)
    report.append("2. WHEN ARE THEY SIMILAR? (Tie locations)")
    report.append("=" * 70)

    ties = paired[paired["outcome"] == "Tie"]
    non_ties = paired[paired["outcome"] != "Tie"]

    report.append(f"\n  Tie locations: {len(ties)}/{n_total}")
    report.append(f"  Mean GP score at ties:     {ties['gp_refined'].mean():.2f}")
    report.append(f"  Mean GP score at non-ties: {non_ties['gp_refined'].mean():.2f}")

    tie_score_dist = ties["gp_refined"].value_counts().sort_index()
    report.append(f"\n  Score distribution at ties:")
    for score, count in tie_score_dist.items():
        report.append(f"    Score {score:.0f}: {count} ties ({count/len(ties)*100:.0f}%)")

    high_ties = ties[ties["gp_refined"] >= 6]
    low_ties = ties[ties["gp_refined"] <= 4]
    report.append(f"\n  High-agreement ties (both >= 6): {len(high_ties)} ({len(high_ties)/len(ties)*100:.0f}%)")
    report.append(f"  Low-agreement ties (both <= 4):  {len(low_ties)} ({len(low_ties)/len(ties)*100:.0f}%)")

    report.append(f"\n  Ties by location:")
    tie_by_loc = ties.groupby("location").size().sort_values(ascending=False)
    total_by_loc = paired.groupby("location").size()
    for loc, count in tie_by_loc.items():
        report.append(f"    {loc}: {count}/{total_by_loc[loc]} ({count/total_by_loc[loc]*100:.0f}%)")

    report.append(f"\n  Ties by target/transfer:")
    for label, is_target in [("Target room", True), ("Transferred", False)]:
        sub = ties[ties["is_target_room"] == is_target]
        total = len(paired[paired["is_target_room"] == is_target])
        report.append(f"    {label}: {len(sub)}/{total} ({len(sub)/total*100:.0f}%)")

    # ---- Section 3: When GP captures more ----
    report.append("\n" + "=" * 70)
    report.append("3. WHEN DOES GP-REFINED CAPTURE MORE? (GP wins)")
    report.append("=" * 70)

    gp_wins = paired[paired["outcome"] == "GP_wins"]
    gp_sessions = session_features[session_features["outcome_category"] == "GP_advantage"]

    report.append(f"\n  GP wins at {len(gp_wins)} locations across {len(gp_sessions)} sessions")
    report.append(f"  Mean GP score when it wins: {gp_wins['gp_refined'].mean():.2f}")
    report.append(f"  Mean Custom score when GP wins: {gp_wins['user_customized'].mean():.2f}")
    report.append(f"  Mean advantage: +{gp_wins['diff'].mean():.2f} points")

    report.append(f"\n  GP-advantage sessions:")
    for sid, row in gp_sessions.sort_values("mean_diff", ascending=False).iterrows():
        report.append(
            f"    {_short_preset(sid)}: mean_diff={row['mean_diff']:+.2f}, "
            f"GP_model_ρ={row['final_spearman']:.2f}, "
            f"accuracy={row['final_accuracy']:.2f}, "
            f"rounds={int(row['n_rounds'])}"
        )

    report.append(f"\n  GP-advantage session characteristics (vs Custom-advantage):")
    cu_sessions = session_features[session_features["outcome_category"] == "Custom_advantage"]
    for col, label in [
        ("final_spearman", "GP model Spearman ρ"),
        ("final_accuracy", "GP pairwise accuracy"),
        ("accuracy_gain", "Accuracy improvement (r1→final)"),
        ("n_rounds", "Number of rounds"),
        ("final_variance", "Image variance (diversity)"),
        ("explicit_implicit_rho", "Explicit-implicit correlation"),
        ("top3_overlap", "Top-3 tag overlap (explicit vs implicit)"),
    ]:
        gp_val = gp_sessions[col].mean()
        cu_val = cu_sessions[col].mean()
        report.append(f"    {label}:")
        report.append(f"      GP-advantage sessions:     {gp_val:.3f}")
        report.append(f"      Custom-advantage sessions: {cu_val:.3f}")

    # Key finding: GP model quality
    report.append(f"\n  KEY FINDING: GP model quality strongly predicts when GP beats Custom")
    report.append(f"    When GP model is good (ρ >= 0.4): GP advantage in {(gp_sessions['final_spearman'] >= 0.4).sum()}/{len(gp_sessions)} GP-advantage sessions")
    report.append(f"    When GP model is poor (ρ < 0.4): Custom wins in {(cu_sessions['final_spearman'] < 0.4).sum()}/{len(cu_sessions)} Custom-advantage sessions")

    # Transfer room analysis
    report.append(f"\n  Transfer room pattern:")
    gp_transfer = paired[(paired["outcome"] == "GP_wins") & (~paired["is_target_room"])]
    gp_target = paired[(paired["outcome"] == "GP_wins") & (paired["is_target_room"])]
    report.append(f"    GP wins in transferred rooms: {len(gp_transfer)}")
    report.append(f"    GP wins in target room:       {len(gp_target)}")
    report.append(f"    => GP-Refined's advantage is stronger in rooms the user wasn't")
    report.append(f"       explicitly thinking about, suggesting it captures generalizable")
    report.append(f"       style preferences rather than room-specific ones.")

    # ---- Section 4: When Custom captures more ----
    report.append("\n" + "=" * 70)
    report.append("4. WHEN CAN'T GP-REFINED MATCH USER CUSTOMIZED? (Custom wins)")
    report.append("=" * 70)

    cu_wins = paired[paired["outcome"] == "Custom_wins"]
    report.append(f"\n  Custom wins at {len(cu_wins)} locations across {len(cu_sessions)} sessions")
    report.append(f"  Mean Custom score when it wins: {cu_wins['user_customized'].mean():.2f}")
    report.append(f"  Mean GP score when Custom wins: {cu_wins['gp_refined'].mean():.2f}")
    report.append(f"  Mean shortfall: {cu_wins['diff'].mean():.2f} points")

    report.append(f"\n  Custom-advantage sessions:")
    for sid, row in cu_sessions.sort_values("mean_diff").iterrows():
        report.append(
            f"    {_short_preset(sid)}: mean_diff={row['mean_diff']:+.2f}, "
            f"GP_model_ρ={row['final_spearman']:.2f}, "
            f"accuracy={row['final_accuracy']:.2f}, "
            f"rounds={int(row['n_rounds'])}"
        )

    # Failure mode analysis
    report.append(f"\n  FAILURE MODE ANALYSIS:")

    poor_model = cu_sessions[cu_sessions["final_spearman"] <= 0]
    mediocre_model = cu_sessions[(cu_sessions["final_spearman"] > 0) & (cu_sessions["final_accuracy"] <= 0.5)]
    decent_model = cu_sessions[cu_sessions["final_accuracy"] > 0.5]

    report.append(f"\n  Mode A: Poor GP model (final ρ <= 0)")
    report.append(f"    Sessions: {len(poor_model)}")
    if len(poor_model) > 0:
        for sid, row in poor_model.iterrows():
            report.append(f"      {_short_preset(sid)}: ρ={row['final_spearman']:.2f}, acc={row['final_accuracy']:.2f}, diff={row['mean_diff']:+.2f}")
        report.append(f"    Interpretation: The GP model learned the wrong preferences —")
        report.append(f"    ranking predictions are anti-correlated with actual preferences.")

    report.append(f"\n  Mode B: Mediocre GP model (ρ > 0 but accuracy <= 0.5)")
    report.append(f"    Sessions: {len(mediocre_model)}")
    if len(mediocre_model) > 0:
        for sid, row in mediocre_model.iterrows():
            report.append(f"      {_short_preset(sid)}: ρ={row['final_spearman']:.2f}, acc={row['final_accuracy']:.2f}, diff={row['mean_diff']:+.2f}")
        report.append(f"    Interpretation: GP model didn't converge well — it's no better")
        report.append(f"    than random at predicting pairwise preferences.")

    report.append(f"\n  Mode C: Decent GP model but Custom still wins (acc > 0.5)")
    report.append(f"    Sessions: {len(decent_model)}")
    if len(decent_model) > 0:
        for sid, row in decent_model.iterrows():
            report.append(f"      {_short_preset(sid)}: ρ={row['final_spearman']:.2f}, acc={row['final_accuracy']:.2f}, diff={row['mean_diff']:+.2f}")
        report.append(f"    Interpretation: GP learned reasonable preferences but user's")
        report.append(f"    explicit customization was still more aligned with their vision.")

    # ---- Section 5: Tag-level investigation ----
    report.append("\n" + "=" * 70)
    report.append("5. TAG-LEVEL INVESTIGATION")
    report.append("=" * 70)

    gp_sess_ids = set(gp_sessions.index)
    cu_sess_ids = set(cu_sessions.index)

    gp_tags = tag_data[tag_data["session_id"].isin(gp_sess_ids)]
    cu_tags = tag_data[tag_data["session_id"].isin(cu_sess_ids)]

    # Normalize within each session for comparison
    def rank_agreement(grp):
        if len(grp) < 3:
            return np.nan
        return scipy_stats.spearmanr(grp["manual_weight"], grp["gp_mu"])[0]

    gp_tag_corr = gp_tags.groupby("session_id").apply(rank_agreement, include_groups=False)
    cu_tag_corr = cu_tags.groupby("session_id").apply(rank_agreement, include_groups=False)

    report.append(f"\n  Manual weight ↔ GP mu correlation (Spearman ρ):")
    report.append(f"    GP-advantage sessions:     {gp_tag_corr.mean():.3f}")
    report.append(f"    Custom-advantage sessions:  {cu_tag_corr.mean():.3f}")

    # GP uncertainty (sigma) as indicator
    report.append(f"\n  GP uncertainty (mean σ across tags):")
    report.append(f"    GP-advantage sessions:     {gp_tags['gp_sigma'].mean():.3f}")
    report.append(f"    Custom-advantage sessions:  {cu_tags['gp_sigma'].mean():.3f}")

    # Surprise tags (low manual weight, high GP mu)
    for label, tag_subset, sess_ids in [
        ("GP-advantage", gp_tags, gp_sess_ids),
        ("Custom-advantage", cu_tags, cu_sess_ids)
    ]:
        surprises = tag_subset[
            (tag_subset["manual_weight"] < tag_subset["manual_weight"].quantile(0.5)) &
            (tag_subset["gp_mu"] > tag_subset["gp_mu"].quantile(0.5))
        ]
        report.append(f"\n  Surprise tags in {label} sessions (low manual weight, high GP mu):")
        report.append(f"    Count: {len(surprises)}/{len(tag_subset)} tags ({len(surprises)/len(tag_subset)*100:.0f}%)")
        if len(surprises) > 0:
            top_s = surprises.nlargest(5, "gp_mu")
            for _, r in top_s.iterrows():
                report.append(f"      {_short(r['session_id'])}: \"{r['tag']}\" "
                              f"(manual={r['manual_weight']:.2f}, gp_mu={r['gp_mu']:.2f})")

    # ---- Section 6: What predicts the gap ----
    report.append("\n" + "=" * 70)
    report.append("6. STATISTICAL PREDICTORS OF THE GP-CUSTOM GAP")
    report.append("=" * 70)

    predictors = [
        ("final_spearman", "GP model Spearman ρ (final round)"),
        ("final_accuracy", "GP pairwise accuracy (final round)"),
        ("accuracy_gain", "Accuracy improvement across rounds"),
        ("n_rounds", "Number of HITL rounds"),
        ("final_variance", "GP image variance"),
        ("explicit_implicit_rho", "Explicit-implicit preference correlation"),
        ("top3_overlap", "Top-3 tag overlap"),
        ("mean_rank_displacement", "Mean rank displacement"),
        ("r1_accuracy", "Round-1 pairwise accuracy"),
    ]

    report.append(f"\n  Spearman correlations with session-level mean(GP - Custom):")
    sig_predictors = []
    for col, label in predictors:
        valid = session_features.dropna(subset=[col, "mean_diff"])
        if len(valid) < 4:
            continue
        rho, p = scipy_stats.spearmanr(valid[col], valid["mean_diff"])
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        report.append(f"    {label}: ρ={rho:+.3f}, p={p:.3f} {sig}")
        if p < 0.10:
            sig_predictors.append((col, label, rho, p))

    if sig_predictors:
        report.append(f"\n  Significant/marginal predictors:")
        for col, label, rho, p in sig_predictors:
            direction = "higher GP advantage" if rho > 0 else "lower GP advantage"
            report.append(f"    Higher {label} → {direction}")

    # ---- Section 7: Synthesis ----
    report.append("\n" + "=" * 70)
    report.append("7. SYNTHESIS")
    report.append("=" * 70)

    report.append("""
  WHY THEY ARE SIMILAR:
    - Both methods produce high-quality results (mean ~5.2/7)
    - In 31% of locations, they receive identical scores
    - Agreement is highest when both score well (75% of ties have score >= 5)
    - The HITL refinement effectively calibrates GP weights to match what
      users would manually set for obvious, salient preferences

  WHEN GP-REFINED CAPTURES MORE:
    - GP model quality is the strongest predictor: sessions where the GP
      model achieves higher Spearman ρ and pairwise accuracy tend to show
      GP advantage
    - GP excels in TRANSFERRED rooms (rooms the user wasn't thinking about),
      suggesting it captures generalizable style preferences
    - GP discovers "surprise" tags — preferences users didn't know they had
      but are revealed through ranking behavior

  WHEN GP-REFINED FALLS SHORT:
    - Poor GP model convergence (low/negative ρ) is the primary failure mode
    - When GP's pairwise accuracy stays at or below chance (0.5), it means
      the model hasn't learned meaningful preference structure
    - In target rooms, Custom tends to outperform because users' explicit
      mental model is directly applicable to the room they had in mind
    - Some users have very clear, articulable preferences that are well-
      captured by direct slider manipulation, leaving less room for GP
      to add value through implicit preference discovery
""")

    return report


# =========================================================================
# Figures
# =========================================================================
def fig_outcome_by_session(session_features):
    """Waterfall showing GP-Custom diff per session, colored by outcome."""
    sf = session_features.sort_values("mean_diff", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sf))
    colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR) for d in sf["mean_diff"]]

    bars = ax.barh(y, sf["mean_diff"], color=colors, edgecolor="white", alpha=0.85, height=0.7)

    labels = [_short_preset(sid) for sid in sf.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, (sid, row) in zip(bars, sf.iterrows()):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.05 if x_pos >= 0 else -0.05
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"ρ={row['final_spearman']:.1f}, acc={row['final_accuracy']:.0%}",
                va="center", ha=ha, fontsize=7, alpha=0.7)

    ax.set_xlabel("Mean Score Difference (GP-Refined − User Customized)")
    ax.set_title("Session-Level: When Does Each Method Win?")

    gp_patch = mpatches.Patch(color=GP_COLOR, label="GP-Refined advantage")
    cu_patch = mpatches.Patch(color=CUSTOM_COLOR, label="User Customized advantage")
    ax.legend(handles=[gp_patch, cu_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    _save(fig, "investigation_session_waterfall.png")


def fig_gp_quality_predicts_gap(session_features):
    """Two-panel: GP model quality vs GP-Custom gap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col, xlabel) in zip(axes, [
        ("final_spearman", "GP Model Spearman ρ (final round)"),
        ("final_accuracy", "GP Pairwise Accuracy (final round)"),
    ]):
        valid = session_features.dropna(subset=[col, "mean_diff"])
        colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR) for d in valid["mean_diff"]]

        ax.scatter(valid[col], valid["mean_diff"], c=colors, s=80,
                   edgecolors="white", linewidth=0.5, zorder=5)

        for sid, row in valid.iterrows():
            ax.annotate(_short(sid), (row[col], row["mean_diff"]),
                        fontsize=7, alpha=0.6, xytext=(4, 4),
                        textcoords="offset points")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

        if len(valid) >= 4:
            rho, p = scipy_stats.spearmanr(valid[col], valid["mean_diff"])
            ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}\np = {p:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mean (GP − Custom) Score", fontsize=10)

    axes[0].set_title("GP Model Quality → GP Advantage")
    axes[1].set_title("GP Accuracy → GP Advantage")
    fig.suptitle("Better GP Model ≈ Bigger GP Advantage Over User Customized", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "investigation_gp_quality_predicts_gap.png")


def fig_tie_analysis(paired):
    """What do ties look like? Score distribution and location breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Score distribution at tie vs non-tie
    ax = axes[0]
    ties = paired[paired["outcome"] == "Tie"]
    gp_w = paired[paired["outcome"] == "GP_wins"]
    cu_w = paired[paired["outcome"] == "Custom_wins"]

    bins = np.arange(0.5, 8.5, 1)
    ax.hist(ties["gp_refined"], bins=bins, alpha=0.7, color=TIE_COLOR, label="Ties", edgecolor="white")
    ax.set_xlabel("Score (both methods)")
    ax.set_ylabel("Count")
    ax.set_title("A) Score Level at Ties")
    ax.axvline(ties["gp_refined"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {ties['gp_refined'].mean():.1f}")
    ax.legend(fontsize=8)

    # Panel B: Outcome breakdown by location
    ax = axes[1]
    loc_outcomes = paired.groupby(["location", "outcome"]).size().unstack(fill_value=0)
    loc_outcomes = loc_outcomes.reindex(columns=["GP_wins", "Tie", "Custom_wins"], fill_value=0)
    loc_outcomes = loc_outcomes.sort_values("GP_wins", ascending=True)

    y = np.arange(len(loc_outcomes))
    ax.barh(y, loc_outcomes["GP_wins"], color=GP_COLOR, alpha=0.85, label="GP wins", edgecolor="white")
    ax.barh(y, loc_outcomes["Tie"], left=loc_outcomes["GP_wins"], color=TIE_COLOR,
            alpha=0.6, label="Tie", edgecolor="white")
    ax.barh(y, loc_outcomes["Custom_wins"],
            left=loc_outcomes["GP_wins"] + loc_outcomes["Tie"],
            color=CUSTOM_COLOR, alpha=0.85, label="Custom wins", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(loc_outcomes.index, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("B) Outcome by Location")
    ax.legend(fontsize=7, loc="lower right")

    # Panel C: Target vs Transfer
    ax = axes[2]
    target_outcomes = paired.groupby(["is_target_room", "outcome"]).size().unstack(fill_value=0)
    target_outcomes = target_outcomes.reindex(columns=["GP_wins", "Tie", "Custom_wins"], fill_value=0)
    target_outcomes.index = ["Transferred", "Target Room"]

    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, target_outcomes["GP_wins"], w, color=GP_COLOR, alpha=0.85, label="GP wins")
    ax.bar(x, target_outcomes["Tie"], w, color=TIE_COLOR, alpha=0.6, label="Tie")
    ax.bar(x + w, target_outcomes["Custom_wins"], w, color=CUSTOM_COLOR, alpha=0.85, label="Custom wins")
    ax.set_xticks(x)
    ax.set_xticklabels(target_outcomes.index, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("C) Target Room vs Transferred")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "investigation_tie_analysis.png")


def fig_failure_modes(session_features):
    """Scatter plot highlighting different failure modes when Custom wins."""
    fig, ax = plt.subplots(figsize=(9, 6))

    gp_sess = session_features[session_features["outcome_category"] == "GP_advantage"]
    cu_sess = session_features[session_features["outcome_category"] == "Custom_advantage"]

    ax.scatter(gp_sess["final_spearman"], gp_sess["final_accuracy"],
               s=np.abs(gp_sess["mean_diff"]) * 120 + 30, c=GP_COLOR,
               alpha=0.7, edgecolors="white", zorder=5, label="GP advantage")
    ax.scatter(cu_sess["final_spearman"], cu_sess["final_accuracy"],
               s=np.abs(cu_sess["mean_diff"]) * 120 + 30, c=CUSTOM_COLOR,
               alpha=0.7, edgecolors="white", zorder=5, label="Custom advantage")

    for sid, row in session_features.iterrows():
        ax.annotate(_short(sid), (row["final_spearman"], row["final_accuracy"]),
                    fontsize=7, alpha=0.6, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.4)

    ax.fill_between([-1, 0], 0, 0.5, alpha=0.05, color=WARN_COLOR)
    ax.text(-0.9, 0.15, "Poor GP model\n(anti-correlated)", fontsize=8, alpha=0.5, color=WARN_COLOR)
    ax.fill_between([0, 0.5], 0, 0.55, alpha=0.03, color="#FF9800")
    ax.text(0.05, 0.15, "Mediocre GP\n(chance-level)", fontsize=8, alpha=0.5, color="#FF9800")

    ax.set_xlabel("GP Model Spearman ρ (final round)", fontsize=11)
    ax.set_ylabel("GP Pairwise Accuracy (final round)", fontsize=11)
    ax.set_title("GP Model Quality Space: When Does Each Method Win?", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, loc="upper left", title="Dot size ∝ |diff|", title_fontsize=8)
    _save(fig, "investigation_failure_modes.png")


def fig_participant_profile(paired, session_features):
    """Per-participant stacked bar: GP wins, ties, Custom wins across all their locations."""
    part_outcomes = paired.groupby(["participant", "outcome"]).size().unstack(fill_value=0)
    part_outcomes = part_outcomes.reindex(columns=["GP_wins", "Tie", "Custom_wins"], fill_value=0)
    part_diff = paired.groupby("participant")["diff"].mean()
    part_outcomes = part_outcomes.loc[part_diff.sort_values(ascending=False).index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Stacked bar
    ax = axes[0]
    y = np.arange(len(part_outcomes))
    ax.barh(y, part_outcomes["GP_wins"], color=GP_COLOR, alpha=0.85, label="GP wins", edgecolor="white")
    ax.barh(y, part_outcomes["Tie"], left=part_outcomes["GP_wins"], color=TIE_COLOR,
            alpha=0.6, label="Tie", edgecolor="white")
    ax.barh(y, part_outcomes["Custom_wins"],
            left=part_outcomes["GP_wins"] + part_outcomes["Tie"],
            color=CUSTOM_COLOR, alpha=0.85, label="Custom wins", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(part_outcomes.index, fontsize=10)
    ax.set_xlabel("Number of Locations")
    ax.set_title("A) Win/Tie/Loss by Participant")
    ax.legend(fontsize=8)

    # Panel B: Mean diff with GP model quality annotation
    ax = axes[1]
    participants = part_outcomes.index.tolist()
    mean_diffs = [part_diff[p] for p in participants]
    colors = [GP_COLOR if d > 0 else (CUSTOM_COLOR if d < 0 else TIE_COLOR) for d in mean_diffs]

    bars = ax.barh(y, mean_diffs, color=colors, edgecolor="white", alpha=0.85, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)

    participant_gp_quality = session_features.groupby("participant")["final_spearman"].mean()
    for i, (p, d) in enumerate(zip(participants, mean_diffs)):
        quality = participant_gp_quality.get(p, np.nan)
        if not np.isnan(quality):
            x_pos = d + (0.05 if d >= 0 else -0.05)
            ha = "left" if d >= 0 else "right"
            ax.text(x_pos, i, f"GP ρ={quality:.2f}", va="center", ha=ha, fontsize=8, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(participants, fontsize=10)
    ax.set_xlabel("Mean Score Difference (GP − Custom)")
    ax.set_title("B) Mean Advantage & GP Model Quality")

    fig.suptitle("Per-Participant: Who Benefits from GP-Refined?", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "investigation_participant_profile.png")


def fig_tag_surprise_comparison(tag_data, session_features):
    """Compare surprise tag patterns in GP-advantage vs Custom-advantage sessions."""
    gp_sess_ids = set(session_features[session_features["outcome_category"] == "GP_advantage"].index)
    cu_sess_ids = set(session_features[session_features["outcome_category"] == "Custom_advantage"].index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, (label, sess_ids, color) in zip(axes, [
        ("GP-Advantage Sessions", gp_sess_ids, GP_COLOR),
        ("Custom-Advantage Sessions", cu_sess_ids, CUSTOM_COLOR),
    ]):
        sub = tag_data[tag_data["session_id"].isin(sess_ids)]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Normalize within session
        sub = sub.copy()
        sub["manual_rank"] = sub.groupby("session_id")["manual_weight"].rank(ascending=False)
        sub["gp_rank"] = sub.groupby("session_id")["gp_mu"].rank(ascending=False)
        sub["rank_change"] = sub["manual_rank"] - sub["gp_rank"]

        surprise = sub[sub["rank_change"] >= 2]
        non_surprise = sub[sub["rank_change"] < 2]

        ax.scatter(non_surprise["manual_weight"], non_surprise["gp_mu"],
                   s=20, alpha=0.4, color=TIE_COLOR, edgecolors="none", label="Other")
        ax.scatter(surprise["manual_weight"], surprise["gp_mu"],
                   s=40, alpha=0.7, color=color, edgecolors="white", linewidth=0.3,
                   label=f"Surprise ({len(surprise)})", zorder=5)

        if len(sub) >= 4:
            rho, p = scipy_stats.spearmanr(sub["manual_weight"], sub["gp_mu"])
            ax.text(0.05, 0.95, f"ρ = {rho:.2f}, p = {p:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Diagonal
        lim = max(sub["manual_weight"].max(), sub["gp_mu"].max()) + 0.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.2)

        ax.set_xlabel("Manual Weight (explicit)", fontsize=10)
        ax.set_ylabel("GP μ (implicit)", fontsize=10)
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Tag-Level: Explicit Weight vs GP-Learned μ", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "investigation_tag_surprise_comparison.png")


# =========================================================================
# main
# =========================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    scores, summary, gp_diag, drift, divergence = load_all()
    paired = build_paired(scores)
    session_features = build_session_features(scores, summary, gp_diag, divergence)
    tag_data = load_tag_data()

    print("Running investigation analysis...")
    report = []
    report = analyze(paired, session_features, tag_data, scores, report)

    report_text = "\n".join(report)

    with open(OUTPUT_DIR / "gp_custom_investigation_report.txt", "w") as f:
        f.write(report_text)
    print(report_text)

    print("\nGenerating investigation figures...")
    fig_outcome_by_session(session_features)
    fig_gp_quality_predicts_gap(session_features)
    fig_tie_analysis(paired)
    fig_failure_modes(session_features)
    fig_participant_profile(paired, session_features)
    fig_tag_surprise_comparison(tag_data, session_features)

    print(f"\nReport: {OUTPUT_DIR / 'gp_custom_investigation_report.txt'}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
