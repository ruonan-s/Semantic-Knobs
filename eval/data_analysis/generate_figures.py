"""
Generate publication-ready figures from parsed session data.

All figures saved to figures/ directory.
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
QUESTIONNAIRE_DIR = Path(__file__).resolve().parent / "questionnaire"

METHOD_ORDER = ["gp_refined", "user_customized", "baseline_text", "style_transfer"]
METHOD_LABELS = {
    "gp_refined": "GP-Refined",
    "user_customized": "User Customized",
    "baseline_text": "Baseline (Text)",
    "style_transfer": "Style Transfer",
}
METHOD_COLORS = {
    "gp_refined": "#E91E63",
    "user_customized": "#FFB74D",
    "baseline_text": "#78909C",
    "style_transfer": "#546E7A",
}
PARTICIPANT_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "d"]


def load_all():
    data = {}
    data["scores"] = pd.read_csv(OUTPUT_DIR / "location_scores.csv")
    data["summary"] = pd.read_csv(OUTPUT_DIR / "session_summary.csv")
    data["gp"] = pd.read_csv(OUTPUT_DIR / "gp_diagnostics.csv")
    data["drift"] = pd.read_csv(OUTPUT_DIR / "preference_drift.csv")
    data["uplifts"] = pd.read_csv(OUTPUT_DIR / "uplifts.csv")
    stat_file = OUTPUT_DIR / "statistical_results.json"
    if stat_file.exists():
        with open(stat_file) as f:
            data["stats"] = json.load(f)
    data["pref_summary"] = pd.read_csv(OUTPUT_DIR / "preference_drift_summary.csv")
    return data


def _method_label(m):
    return METHOD_LABELS.get(m, m)


def _save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# =========================================================================
# 1. Method Score Comparison (box + swarm)
# =========================================================================
def fig_method_scores_boxswarm(scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = scores.copy()
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER]
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER}

    sns.boxplot(
        data=plot_df, x="method_label", y="score", hue="method_label",
        order=order, palette=palette, width=0.5, fliersize=0, ax=ax,
        boxprops=dict(alpha=0.4), legend=False,
    )

    participants = sorted(plot_df["participant"].unique())
    for i, p in enumerate(participants):
        sub = plot_df[plot_df["participant"] == p]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(sub))
        x_positions = [order.index(ml) + j for ml, j in zip(sub["method_label"], jitter)]
        ax.scatter(
            x_positions, sub["score"],
            marker=PARTICIPANT_MARKERS[i % len(PARTICIPANT_MARKERS)],
            s=25, alpha=0.7, label=p, zorder=5, edgecolors="white", linewidth=0.3,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Score (1-7)")
    ax.set_title("Method Score Comparison")
    ax.legend(title="Participant", fontsize=7, title_fontsize=8, loc="lower left")
    ax.set_ylim(0.5, 7.5)
    _save(fig, "method_scores_boxswarm.png")


# =========================================================================
# 2. Per-Participant Method Profiles (slope chart)
# =========================================================================
def fig_participant_profiles(scores):
    pm = (
        scores.groupby(["participant", "method"])["score"]
        .mean()
        .unstack("method")
        .reindex(columns=METHOD_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHOD_ORDER))
    participants = pm.index.tolist()
    for i, p in enumerate(participants):
        vals = pm.loc[p].values
        ax.plot(x, vals, "-", label=p, alpha=0.8,
                marker=PARTICIPANT_MARKERS[i % len(PARTICIPANT_MARKERS)],
                markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in METHOD_ORDER], fontsize=9)
    ax.set_ylabel("Mean Score")
    ax.set_title("Per-Participant Method Profiles")
    ax.legend(title="Participant", fontsize=8)
    ax.set_ylim(0, 8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "participant_method_profiles.png")


# =========================================================================
# 3. Uplift Histogram
# =========================================================================
def fig_uplift_histogram(uplifts):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-5.5, 7.5, 1)

    ax.hist(uplifts["gp_vs_baseline"].dropna(), bins=bins, alpha=0.6,
            label="GP-Refined vs Baseline", color=METHOD_COLORS["gp_refined"],
            edgecolor="white")
    ax.hist(uplifts["custom_vs_baseline"].dropna(), bins=bins, alpha=0.5,
            label="User Customized vs Baseline", color=METHOD_COLORS["user_customized"],
            edgecolor="white")

    for col, color, label in [
        ("gp_vs_baseline", METHOD_COLORS["gp_refined"], "GP mean"),
        ("custom_vs_baseline", METHOD_COLORS["user_customized"], "Custom mean"),
    ]:
        mean_val = uplifts[col].mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=2, label=f"{label}={mean_val:.2f}")

    ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Score Uplift vs Baseline (Text)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Score Uplifts over Baseline")
    ax.legend(fontsize=8)
    _save(fig, "uplift_histogram.png")


# =========================================================================
# 4. Score Heatmap (participant x location)
# =========================================================================
def fig_score_heatmap(scores):
    gp = scores[scores["method"] == "gp_refined"]
    pivot = gp.pivot_table(index="participant", columns="location", values="score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlGnBu",
        vmin=1, vmax=7, ax=ax, linewidths=0.5,
    )
    ax.set_title("GP-Refined Scores by Participant x Location")
    ax.set_ylabel("")
    _save(fig, "score_heatmap.png")


# =========================================================================
# 5. Win Rate Bar Chart
# =========================================================================
def fig_win_rates(scores):
    def _compute_rates(scores_df):
        methods = METHOD_ORDER
        n_locations = scores_df.groupby(["session_id", "location"]).ngroups
        top1 = {}
        top2 = {}
        for m in methods:
            top1[m] = 0
            top2[m] = 0

        for (sid, loc), grp in scores_df.groupby(["session_id", "location"]):
            sorted_g = grp.sort_values("score", ascending=False)
            best_score = sorted_g.iloc[0]["score"]
            winners = sorted_g[sorted_g["score"] == best_score]["method"].tolist()
            for m in methods:
                if m in winners:
                    top1[m] += 1

            if len(sorted_g) >= 2:
                second_score = sorted_g.iloc[1]["score"]
                top2_methods = sorted_g[sorted_g["score"] >= second_score]["method"].tolist()
            else:
                top2_methods = winners
            for m in methods:
                if m in top2_methods:
                    top2[m] += 1

        return {m: top1[m] / n_locations for m in methods}, {m: top2[m] / n_locations for m in methods}

    top1_rates, top2_rates = _compute_rates(scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHOD_ORDER))
    w = 0.35
    bars1 = ax.bar(x - w / 2, [top1_rates[m] for m in METHOD_ORDER], w,
                   color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.9,
                   label="Top-1 Win Rate", edgecolor="white")
    bars2 = ax.bar(x + w / 2, [top2_rates[m] for m in METHOD_ORDER], w,
                   color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.5,
                   label="Top-2 Rate", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in METHOD_ORDER], fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Win Rates by Method")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", fontsize=8)

    _save(fig, "win_rates.png")


# =========================================================================
# 6. Win Rates by Room Condition (Overall / Target / Transferred)
# =========================================================================
def fig_win_rates_by_room_condition(scores):
    def _compute_rates(scores_df):
        methods = METHOD_ORDER
        n_locations = scores_df.groupby(["session_id", "location"]).ngroups
        if n_locations == 0:
            return {m: np.nan for m in methods}, {m: np.nan for m in methods}
        top1 = {m: 0 for m in methods}
        top2 = {m: 0 for m in methods}

        for (_, _), grp in scores_df.groupby(["session_id", "location"]):
            sorted_g = grp.sort_values("score", ascending=False)
            best_score = sorted_g.iloc[0]["score"]
            winners = sorted_g[sorted_g["score"] == best_score]["method"].tolist()
            for m in methods:
                if m in winners:
                    top1[m] += 1

            if len(sorted_g) >= 2:
                second_score = sorted_g.iloc[1]["score"]
                top2_methods = sorted_g[sorted_g["score"] >= second_score]["method"].tolist()
            else:
                top2_methods = winners
            for m in methods:
                if m in top2_methods:
                    top2[m] += 1

        return (
            {m: top1[m] / n_locations for m in methods},
            {m: top2[m] / n_locations for m in methods},
        )

    conditions = [
        ("Overall", scores),
        ("Target Room", scores[scores["is_target_room"] == True]),
        ("Transferred", scores[scores["is_target_room"] == False]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (label, sub_scores) in zip(axes, conditions):
        top1_rates, top2_rates = _compute_rates(sub_scores)
        x = np.arange(len(METHOD_ORDER))
        w = 0.35
        bars1 = ax.bar(
            x - w / 2,
            [top1_rates[m] for m in METHOD_ORDER],
            w,
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.9,
            label="Top-1",
            edgecolor="white",
        )
        bars2 = ax.bar(
            x + w / 2,
            [top2_rates[m] for m in METHOD_ORDER],
            w,
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.5,
            label="Top-2",
            edgecolor="white",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(m) for m in METHOD_ORDER], rotation=20, ha="right", fontsize=9)
        ax.set_title(label)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.2)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", fontsize=7)

    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Win Rates by Condition", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "win_rates_room_conditions.png")


# =========================================================================
# 7. GP Learning Curves
# =========================================================================
def fig_gp_learning_curves(gp):
    sessions = gp["session_id"].unique()
    n = len(sessions)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)

    for i, sid in enumerate(sorted(sessions)):
        ax = axes[i // ncols][i % ncols]
        sub = gp[gp["session_id"] == sid].sort_values("round")
        short_id = sid.split("_")[1] + " " + "_".join(sid.split("_")[2:-4])

        ax.plot(sub["round"], sub["pairwise_accuracy"], "o-", color="#2196F3",
                label="Pairwise Acc.", markersize=5)
        ax.set_ylabel("Accuracy", color="#2196F3", fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="y", labelcolor="#2196F3", labelsize=7)

        ax2 = ax.twinx()
        ax2.plot(sub["round"], sub["spearman_rho"], "s--", color="#E91E63",
                 label="Spearman ρ", markersize=5)
        ax2.set_ylabel("Spearman ρ", color="#E91E63", fontsize=8)
        ax2.set_ylim(-1.1, 1.1)
        ax2.tick_params(axis="y", labelcolor="#E91E63", labelsize=7)

        ax.set_xlabel("Round", fontsize=8)
        ax.set_title(short_id, fontsize=9)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="lower right")

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("GP Learning Curves per Session", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "gp_learning_curves.png")


# =========================================================================
# 8. GP Quality vs Outcome
# =========================================================================
def fig_gp_quality_vs_outcome(scores, gp):
    gp_scores = (
        scores[scores["method"] == "gp_refined"]
        .groupby("session_id")["score"].mean()
    )
    last_round = gp.loc[gp.groupby("session_id")["round"].idxmax()]
    merged = last_round.set_index("session_id")[["spearman_rho"]].join(
        gp_scores.rename("mean_gp_score"), how="inner"
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged["spearman_rho"], merged["mean_gp_score"],
               s=60, color="#2196F3", edgecolors="white", zorder=5)

    for sid, row in merged.iterrows():
        short = sid.split("_")[1]
        ax.annotate(short, (row["spearman_rho"], row["mean_gp_score"]),
                     fontsize=7, alpha=0.7, xytext=(5, 5),
                     textcoords="offset points")

    if len(merged) >= 3:
        z = np.polyfit(merged["spearman_rho"], merged["mean_gp_score"], 1)
        x_line = np.linspace(merged["spearman_rho"].min() - 0.1,
                              merged["spearman_rho"].max() + 0.1, 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
        rho, p = scipy_stats.spearmanr(merged["spearman_rho"], merged["mean_gp_score"])
        ax.text(0.05, 0.95, f"ρ = {rho:.2f}, p = {p:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top")

    ax.set_xlabel("Final Spearman ρ (GP model)")
    ax.set_ylabel("Mean GP-Refined Score")
    ax.set_title("GP Model Quality vs Downstream Score")
    _save(fig, "gp_quality_vs_outcome.png")


# =========================================================================
# 9. Warning Distribution
# =========================================================================
def fig_warning_distribution(warnings_file=None):
    warn_path = OUTPUT_DIR / "gp_warnings.csv"
    if not warn_path.exists():
        print("  Skipping warning_distribution (no data)")
        return
    warnings = pd.read_csv(warn_path)
    if warnings.empty:
        print("  Skipping warning_distribution (empty)")
        return

    pivot = (
        warnings.groupby(["session_id", "warning_type"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = [s.split("_")[1] + " " + s.split("_")[2] for s in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_xlabel("Warning Count")
    ax.set_title("GP Refinement Warnings by Session")
    ax.legend(title="Warning Type", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "warning_distribution.png")


# =========================================================================
# 10. Tag Weight Drift (dumbbell)
# =========================================================================
def fig_tag_weight_drift(drift):
    sessions = drift["session_id"].unique()
    n = len(sessions)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)

    for i, sid in enumerate(sorted(sessions)):
        ax = axes[i // ncols][i % ncols]
        sub = drift[drift["session_id"] == sid].dropna(subset=["initial_weight", "final_weight"])
        sub = sub.nlargest(10, "initial_weight")
        if sub.empty:
            ax.set_visible(False)
            continue

        tags = sub["tag"].tolist()[::-1]
        y = np.arange(len(tags))

        for j, tag in enumerate(tags):
            row = sub[sub["tag"] == tag].iloc[0]
            iw, fw = row["initial_weight"], row["final_weight"]
            color = "#4CAF50" if fw >= iw else "#F44336"
            ax.plot([iw, fw], [j, j], "-", color=color, linewidth=1.5, alpha=0.7)
            ax.scatter([iw], [j], color="#FF9800", s=30, zorder=5)
            ax.scatter([fw], [j], color="#2196F3", s=30, zorder=5)

        ax.set_yticks(y)
        ax.set_yticklabels(tags, fontsize=7)
        short_id = sid.split("_")[1] + " " + "_".join(sid.split("_")[2:-4])
        ax.set_title(short_id, fontsize=9)
        ax.set_xlabel("Weight", fontsize=8)

    handles = [
        plt.scatter([], [], color="#FF9800", s=30, label="Initial"),
        plt.scatter([], [], color="#2196F3", s=30, label="Final"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Tag Weight Drift (Top-10 by Initial Weight)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "tag_weight_drift.png")


# =========================================================================
# 11. Entropy Shift
# =========================================================================
def fig_entropy_shift(pref_summary):
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = pref_summary.dropna(subset=["initial_entropy", "final_entropy"])

    ax.scatter(valid["initial_entropy"], valid["final_entropy"],
               s=60, color="#2196F3", edgecolors="white", zorder=5)

    for _, row in valid.iterrows():
        short = row["session_id"].split("_")[1]
        ax.annotate(short, (row["initial_entropy"], row["final_entropy"]),
                     fontsize=7, alpha=0.7, xytext=(5, 3),
                     textcoords="offset points")

    lim_lo = min(valid["initial_entropy"].min(), valid["final_entropy"].min()) - 0.1
    lim_hi = max(valid["initial_entropy"].max(), valid["final_entropy"].max()) + 0.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.3, label="No change")

    ax.set_xlabel("Initial Weight Entropy")
    ax.set_ylabel("Final Weight Entropy")
    ax.set_title("Preference Entropy: Before vs After Refinement")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    _save(fig, "entropy_shift.png")


# =========================================================================
# 12. Session Timeline Gantt
# =========================================================================
def fig_session_timelines(summary):
    fig, ax = plt.subplots(figsize=(10, 5))
    sessions = summary.sort_values("total_duration_s", ascending=True)
    y = np.arange(len(sessions))
    labels = [s.split("_")[1] + " " + "_".join(s.split("_")[2:-4])
              for s in sessions["session_id"]]

    hitl = sessions["hitl_duration_s"].fillna(0).values
    gen_total = (sessions["avg_generation_s"].fillna(0) * sessions["n_locations_rated"].fillna(0)).values
    remaining = sessions["total_duration_s"].values - hitl - gen_total

    ax.barh(y, hitl, color="#2196F3", alpha=0.8, label="HITL Refinement", edgecolor="white")
    ax.barh(y, gen_total, left=hitl, color="#4CAF50", alpha=0.8,
            label="Image Generation", edgecolor="white")
    ax.barh(y, remaining, left=hitl + gen_total, color="#FF9800", alpha=0.5,
            label="Other (setup, ranking)", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Session Duration Breakdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "session_timelines.png")


# =========================================================================
# 13. Time vs Quality
# =========================================================================
def fig_time_vs_quality(scores, summary):
    gp_scores = (
        scores[scores["method"] == "gp_refined"]
        .groupby("session_id")["score"].mean()
    )
    baseline_scores = (
        scores[scores["method"] == "baseline_text"]
        .groupby("session_id")["score"].mean()
    )
    uplift = (gp_scores - baseline_scores).rename("uplift")

    merged = summary.set_index("session_id")[["hitl_duration_s", "n_rounds"]].join(
        uplift, how="inner"
    ).dropna()
    merged["hitl_minutes"] = merged["hitl_duration_s"] / 60

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        merged["hitl_minutes"], merged["uplift"],
        s=merged["n_rounds"] * 30, c="#2196F3",
        alpha=0.7, edgecolors="white", zorder=5,
    )

    for sid, row in merged.iterrows():
        short = sid.split("_")[1]
        ax.annotate(short, (row["hitl_minutes"], row["uplift"]),
                     fontsize=7, alpha=0.7, xytext=(5, 5),
                     textcoords="offset points")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("HITL Duration (minutes)")
    ax.set_ylabel("Mean Uplift (GP-Refined - Baseline)")
    ax.set_title("Interaction Time vs Quality Gain")

    sizes = [1, 3, 6]
    for s in sizes:
        ax.scatter([], [], s=s * 30, c="#2196F3", alpha=0.5, label=f"{s} rounds")
    ax.legend(title="# Rounds", fontsize=8, loc="upper left")
    _save(fig, "time_vs_quality.png")


# =========================================================================
# 14. Average Rank (aggregate across all participants)
# =========================================================================
def fig_average_rank(scores):
    METHOD_RANK_LABELS = {
        "gp_refined": "Ours",
        "user_customized": "User Customized",
        "baseline_text": "Text-Only",
        "style_transfer": "Text+Image",
    }
    RANK_COLORS = {
        "Ours": METHOD_COLORS["gp_refined"],
        "User Customized": METHOD_COLORS["user_customized"],
        "Text-Only": METHOD_COLORS["baseline_text"],
        "Text+Image": METHOD_COLORS["style_transfer"],
    }

    avg_rank = scores.groupby("method")["rank_position"].mean().reindex(METHOD_ORDER)
    labels = [METHOD_RANK_LABELS.get(m, m) for m in METHOD_ORDER]
    values = [avg_rank[m] for m in METHOD_ORDER]
    colors = [RANK_COLORS[lab] for lab in labels]

    # 95% bootstrap confidence interval for mean rank per method.
    rng = np.random.default_rng(42)
    n_boot = 3000
    ci_lows = []
    ci_highs = []
    for method in METHOD_ORDER:
        sample = scores.loc[scores["method"] == method, "rank_position"].dropna().to_numpy()
        if len(sample) == 0:
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
            continue
        boot_means = []
        for _ in range(n_boot):
            draw = rng.choice(sample, size=len(sample), replace=True)
            boot_means.append(draw.mean())
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        ci_lows.append(values[METHOD_ORDER.index(method)] - ci_lo)
        ci_highs.append(ci_hi - values[METHOD_ORDER.index(method)])

    # Wilcoxon signed-rank p-values for paired rank differences vs Ours.
    pivot = scores.pivot_table(
        index=["participant", "session_id", "location"],
        columns="method",
        values="rank_position",
    )

    def _p_to_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    significance_labels = {}
    for method in METHOD_ORDER:
        if method == "gp_refined":
            significance_labels[method] = "ref"
            continue
        paired = pivot[["gp_refined", method]].dropna()
        if len(paired) < 5:
            significance_labels[method] = "n/a"
            continue
        try:
            stat = scipy_stats.wilcoxon(
                paired["gp_refined"],
                paired[method],
                alternative="two-sided",
            )
            significance_labels[method] = _p_to_stars(stat.pvalue)
        except ValueError:
            significance_labels[method] = "n.s."

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(METHOD_ORDER))
    yerr = np.array([ci_lows, ci_highs])
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="white",
        alpha=0.9,
        yerr=yerr,
        capsize=6,
        error_kw={"elinewidth": 1.4, "ecolor": "#333333"},
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        # Place labels above the upper CI whisker to avoid overlap.
        upper_ci = ci_highs[i] if not np.isnan(ci_highs[i]) else 0.0
        value_y = val + upper_ci + 0.10
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value_y,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        method = METHOD_ORDER[i]
        if method != "gp_refined":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value_y + 0.4,
                significance_labels[method],
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average Rank (lower is better)", fontsize=11)
    ax.set_title("Average Rank with 95% CI and Wilcoxon Significance vs Ours", fontsize=12)
    ax.set_ylim(0, 4.5)
    ax.grid(axis="y", alpha=0.2)

    n_participants = scores["participant"].nunique()
    n_sessions = scores["session_id"].nunique()
    n_ratings = len(scores) // len(METHOD_ORDER)
    ax.text(
        0.99,
        0.02,
        f"N={n_participants} participants, {n_sessions} sessions, {n_ratings} location ratings",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        alpha=0.7,
    )

    fig.tight_layout()
    _save(fig, "average_rank.png")


# =========================================================================
# 15. Average Rating (aggregate across all participants)
# =========================================================================
def fig_average_rating(scores):
    METHOD_SCORE_LABELS = {
        "gp_refined": "Ours",
        "user_customized": "User Customized",
        "baseline_text": "Text-Only",
        "style_transfer": "Text+Image",
    }
    SCORE_COLORS = {
        "Ours": METHOD_COLORS["gp_refined"],
        "User Customized": METHOD_COLORS["user_customized"],
        "Text-Only": METHOD_COLORS["baseline_text"],
        "Text+Image": METHOD_COLORS["style_transfer"],
    }

    avg_score = scores.groupby("method")["score"].mean().reindex(METHOD_ORDER)
    labels = [METHOD_SCORE_LABELS.get(m, m) for m in METHOD_ORDER]
    values = [avg_score[m] for m in METHOD_ORDER]
    colors = [SCORE_COLORS[lab] for lab in labels]

    # 95% bootstrap confidence interval for mean score per method.
    rng = np.random.default_rng(42)
    n_boot = 3000
    ci_lows = []
    ci_highs = []
    for method in METHOD_ORDER:
        sample = scores.loc[scores["method"] == method, "score"].dropna().to_numpy()
        if len(sample) == 0:
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
            continue
        boot_means = []
        for _ in range(n_boot):
            draw = rng.choice(sample, size=len(sample), replace=True)
            boot_means.append(draw.mean())
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        ci_lows.append(values[METHOD_ORDER.index(method)] - ci_lo)
        ci_highs.append(ci_hi - values[METHOD_ORDER.index(method)])

    # Wilcoxon signed-rank p-values for paired score differences vs Ours.
    pivot = scores.pivot_table(
        index=["participant", "session_id", "location"],
        columns="method",
        values="score",
    )

    def _p_to_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    significance_labels = {}
    for method in METHOD_ORDER:
        if method == "gp_refined":
            significance_labels[method] = "ref"
            continue
        paired = pivot[["gp_refined", method]].dropna()
        if len(paired) < 5:
            significance_labels[method] = "n/a"
            continue
        try:
            stat = scipy_stats.wilcoxon(
                paired["gp_refined"],
                paired[method],
                alternative="two-sided",
            )
            significance_labels[method] = _p_to_stars(stat.pvalue)
        except ValueError:
            significance_labels[method] = "n.s."

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(METHOD_ORDER))
    yerr = np.array([ci_lows, ci_highs])
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="white",
        alpha=0.9,
        yerr=yerr,
        capsize=6,
        error_kw={"elinewidth": 1.4, "ecolor": "#333333"},
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        upper_ci = ci_highs[i] if not np.isnan(ci_highs[i]) else 0.0
        value_y = val + upper_ci + 0.10
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value_y,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        method = METHOD_ORDER[i]
        if method != "gp_refined":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value_y + 0.15,
                significance_labels[method],
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    ax.axhline(4.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average Rating Score (higher is better)", fontsize=11)
    ax.set_title("Average Rating with 95% CI and Wilcoxon Significance vs Ours", fontsize=12)
    ax.set_ylim(0, 7.8)
    ax.grid(axis="y", alpha=0.2)

    n_participants = scores["participant"].nunique()
    n_sessions = scores["session_id"].nunique()
    n_ratings = len(scores) // len(METHOD_ORDER)
    ax.text(
        0.99,
        0.02,
        f"N={n_participants} participants, {n_sessions} sessions, {n_ratings} location ratings",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        alpha=0.7,
    )

    fig.tight_layout()
    _save(fig, "average_rating.png")


# =========================================================================
# 16. Average Rating by Room Condition (overall/target/transferred)
# =========================================================================
def fig_average_rating_by_room_condition(scores):
    condition_rows = []
    condition_masks = {
        "Overall": scores["is_target_room"].notna(),
        "Target Room": scores["is_target_room"] == True,
        "Transferred": scores["is_target_room"] == False,
    }

    for condition, mask in condition_masks.items():
        sub = scores[mask]
        for method in METHOD_ORDER:
            vals = sub.loc[sub["method"] == method, "score"].dropna()
            if len(vals) == 0:
                continue
            condition_rows.append({
                "condition": condition,
                "method": method,
                "mean_score": vals.mean(),
                "sem": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            })

    df = pd.DataFrame(condition_rows)
    if df.empty:
        print("  Skipping average_rating_room_condition (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = METHOD_ORDER
    condition_order = ["Overall", "Target Room", "Transferred"]
    condition_alphas = {
        "Overall": 0.95,
        "Target Room": 0.70,
        "Transferred": 0.45,
    }
    condition_hatches = {
        "Overall": "",
        "Target Room": "//",
        "Transferred": "..",
    }
    x = np.arange(len(methods))
    w = 0.22

    for i, condition in enumerate(condition_order):
        sub = df[df["condition"] == condition].set_index("method").reindex(methods)
        y = sub["mean_score"].values
        yerr = 1.96 * sub["sem"].values
        offset = (i - 1) * w
        bar_colors = [METHOD_COLORS[m] for m in methods]
        bars = ax.bar(
            x + offset,
            y,
            width=w,
            color=bar_colors,
            edgecolor="white",
            alpha=condition_alphas[condition],
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )
        for bar in bars:
            bar.set_hatch(condition_hatches[condition])

        # Legend handle per condition using neutral swatches.
        ax.bar(
            [],
            [],
            color="#546E7A",
            alpha=condition_alphas[condition],
            hatch=condition_hatches[condition],
            label=condition,
        )
        for bar, val, err in zip(bars, y, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (err if not np.isnan(err) else 0) + 0.06,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=10)
    ax.set_ylabel("Average Rating Score (1-7)")
    ax.set_title("Average Rating by Method with Room Conditions")
    ax.set_ylim(0, 7.8)
    ax.axhline(4.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout()
    _save(fig, "average_rating_room_conditions.png")


# =========================================================================
# 17. Average Rank by Room Condition (overall/target/transferred)
# =========================================================================
def fig_average_rank_by_room_condition(scores):
    condition_rows = []
    condition_masks = {
        "Overall": scores["is_target_room"].notna(),
        "Target Room": scores["is_target_room"] == True,
        "Transferred": scores["is_target_room"] == False,
    }

    for condition, mask in condition_masks.items():
        sub = scores[mask]
        for method in METHOD_ORDER:
            vals = sub.loc[sub["method"] == method, "rank_position"].dropna()
            if len(vals) == 0:
                continue
            condition_rows.append({
                "condition": condition,
                "method": method,
                "mean_rank": vals.mean(),
                "sem": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            })

    df = pd.DataFrame(condition_rows)
    if df.empty:
        print("  Skipping average_rank_room_condition (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = METHOD_ORDER
    condition_order = ["Overall", "Target Room", "Transferred"]
    condition_alphas = {
        "Overall": 0.95,
        "Target Room": 0.70,
        "Transferred": 0.45,
    }
    condition_hatches = {
        "Overall": "",
        "Target Room": "//",
        "Transferred": "..",
    }
    x = np.arange(len(methods))
    w = 0.22

    for i, condition in enumerate(condition_order):
        sub = df[df["condition"] == condition].set_index("method").reindex(methods)
        y = sub["mean_rank"].values
        yerr = 1.96 * sub["sem"].values
        offset = (i - 1) * w
        bar_colors = [METHOD_COLORS[m] for m in methods]
        bars = ax.bar(
            x + offset,
            y,
            width=w,
            color=bar_colors,
            edgecolor="white",
            alpha=condition_alphas[condition],
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )
        for bar in bars:
            bar.set_hatch(condition_hatches[condition])

        ax.bar(
            [],
            [],
            color="#546E7A",
            alpha=condition_alphas[condition],
            hatch=condition_hatches[condition],
            label=condition,
        )
        for bar, val, err in zip(bars, y, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (err if not np.isnan(err) else 0) + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=10)
    ax.set_ylabel("Average Rank (lower is better)")
    ax.set_title("Average Rank by Method with Room Conditions")
    ax.set_ylim(0, 4.7)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout()
    _save(fig, "average_rank_room_conditions.png")


# =========================================================================
# 18. Average Rank per Participant (grouped bar)
# =========================================================================
def fig_average_rank_per_participant(scores):
    METHOD_RANK_LABELS = {
        "gp_refined": "Ours",
        "user_customized": "User Customized",
        "baseline_text": "Text-Only",
        "style_transfer": "Text+Image",
    }
    RANK_COLORS = METHOD_COLORS

    pm = (
        scores.groupby(["participant", "method"])["rank_position"]
        .mean()
        .unstack("method")
        .reindex(columns=METHOD_ORDER)
    )
    participants = sorted(pm.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(participants))
    n_methods = len(METHOD_ORDER)
    w = 0.8 / n_methods

    for i, m in enumerate(METHOD_ORDER):
        offset = (i - n_methods / 2 + 0.5) * w
        vals = [pm.loc[p, m] for p in participants]
        ax.bar(x + offset, vals, w, color=RANK_COLORS[m], alpha=0.85,
               label=METHOD_RANK_LABELS[m], edgecolor="white")

    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, fontsize=10)
    ax.set_ylabel("Average Rank (lower is better)", fontsize=11)
    ax.set_title("Average Rank by Participant", fontsize=13)
    ax.set_ylim(0, 4.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "average_rank_per_participant.png")


# =========================================================================
# 19. Participant Association: Rank + Q1 + Q3
# =========================================================================
def fig_participant_rank_questionnaire_association(scores):
    pre_file = QUESTIONNAIRE_DIR / "pre_session.csv"
    post_file = QUESTIONNAIRE_DIR / "post_session.csv"
    if not pre_file.exists() or not post_file.exists():
        print("  Skipping rank_questionnaire_association (missing questionnaire CSVs)")
        return

    q1_col = "I have a clear sense of what [descriptor] means to me in terms of a physical space."
    q2_col = "I could describe my ideal [descriptor] space to an interior designer in concrete terms."
    q3_col = "I can mentally picture what a [descriptor] environment would look like for me."
    id_col = "Your participant ID"

    pre = pd.read_csv(pre_file)
    post = pd.read_csv(post_file)
    for df in (pre, post):
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()

    # Aggregate questionnaire at participant level across descriptors.
    q_participant = (
        pre[[id_col, q1_col, q2_col, q3_col]]
        .rename(columns={q1_col: "q1_pre", q2_col: "q2_pre", q3_col: "q3_pre"})
        .merge(
            post[[id_col, q1_col, q2_col, q3_col]].rename(
                columns={q1_col: "q1_post", q2_col: "q2_post", q3_col: "q3_post"}
            ),
            on=id_col,
            how="inner",
        )
    )
    q_participant["q1_pre"] = pd.to_numeric(q_participant["q1_pre"], errors="coerce")
    q_participant["q1_post"] = pd.to_numeric(q_participant["q1_post"], errors="coerce")
    q_participant["q2_pre"] = pd.to_numeric(q_participant["q2_pre"], errors="coerce")
    q_participant["q2_post"] = pd.to_numeric(q_participant["q2_post"], errors="coerce")
    q_participant["q3_pre"] = pd.to_numeric(q_participant["q3_pre"], errors="coerce")
    q_participant["q3_post"] = pd.to_numeric(q_participant["q3_post"], errors="coerce")
    q_participant = (
        q_participant.groupby(id_col, as_index=False)[
            ["q1_pre", "q1_post", "q2_pre", "q2_post", "q3_pre", "q3_post"]
        ]
        .mean()
        .rename(columns={id_col: "participant"})
    )

    rank_participant = (
        scores[scores["method"].isin(["gp_refined", "user_customized"])]
        .groupby(["participant", "method"])["rank_position"]
        .mean()
        .unstack("method")
        .rename(columns={"gp_refined": "ours_rank", "user_customized": "custom_rank"})
        .reset_index()
    )

    merged = rank_participant.merge(q_participant, on="participant", how="inner")
    if merged.empty:
        print("  Skipping rank_questionnaire_association (no matched participant data)")
        return
    merged["q1_delta"] = merged["q1_post"] - merged["q1_pre"]
    merged["q2_delta"] = merged["q2_post"] - merged["q2_pre"]
    merged["q3_delta"] = merged["q3_post"] - merged["q3_pre"]

    merged = merged.sort_values("ours_rank")
    participants = merged["participant"].tolist()
    x = np.arange(len(participants))

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # Panel 1: Ours vs User Customized rank
    ax = axes[0]
    w = 0.36
    ax.bar(
        x - w / 2,
        merged["ours_rank"].values,
        color=METHOD_COLORS["gp_refined"],
        edgecolor="white",
        alpha=0.9,
        width=w,
        label="Ours",
    )
    ax.bar(
        x + w / 2,
        merged["custom_rank"].values,
        color=METHOD_COLORS["user_customized"],
        edgecolor="white",
        alpha=0.9,
        width=w,
        label="User Customized",
    )
    for i, v in enumerate(merged["ours_rank"].values):
        ax.text(i - w / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(merged["custom_rank"].values):
        ax.text(i + w / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("Avg Rank\n(lower better)")
    ax.set_title("Participant Association: Method Rank with Q1/Q3 Self-Assessment")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: Q1 pre/post
    ax = axes[1]
    for _, row in merged.iterrows():
        idx = participants.index(row["participant"])
        ax.plot([idx - 0.12, idx + 0.12], [row["q1_pre"], row["q1_post"]],
                color="#B0BEC5", linewidth=1.2, alpha=0.9)
        ax.scatter(idx - 0.12, row["q1_pre"], color=METHOD_COLORS["baseline_text"], s=35, zorder=3)
        ax.scatter(idx + 0.12, row["q1_post"], color=METHOD_COLORS["gp_refined"], s=35, zorder=3)
    rho_q1_ours, p_q1_ours = scipy_stats.spearmanr(merged["ours_rank"], merged["q1_delta"], nan_policy="omit")
    rho_q1_custom, p_q1_custom = scipy_stats.spearmanr(merged["custom_rank"], merged["q1_delta"], nan_policy="omit")
    ax.text(0.99, 0.96, f"Ours: Spearman(rank, Q1 delta)={rho_q1_ours:.2f}, p={p_q1_ours:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["gp_refined"])
    ax.text(0.99, 0.89, f"Custom: Spearman(rank, Q1 delta)={rho_q1_custom:.2f}, p={p_q1_custom:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["user_customized"])
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel("Q1 score")
    ax.set_title("Q1: Clear Sense (Pre -> Post)")
    ax.grid(axis="y", alpha=0.2)

    # Panel 3: Q2 pre/post
    ax = axes[2]
    for _, row in merged.iterrows():
        idx = participants.index(row["participant"])
        ax.plot([idx - 0.12, idx + 0.12], [row["q2_pre"], row["q2_post"]],
                color="#B0BEC5", linewidth=1.2, alpha=0.9)
        ax.scatter(idx - 0.12, row["q2_pre"], color=METHOD_COLORS["baseline_text"], s=35, zorder=3)
        ax.scatter(idx + 0.12, row["q2_post"], color=METHOD_COLORS["gp_refined"], s=35, zorder=3)
    rho_q2_ours, p_q2_ours = scipy_stats.spearmanr(merged["ours_rank"], merged["q2_delta"], nan_policy="omit")
    rho_q2_custom, p_q2_custom = scipy_stats.spearmanr(merged["custom_rank"], merged["q2_delta"], nan_policy="omit")
    ax.text(0.99, 0.96, f"Ours: Spearman(rank, Q2 delta)={rho_q2_ours:.2f}, p={p_q2_ours:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["gp_refined"])
    ax.text(0.99, 0.89, f"Custom: Spearman(rank, Q2 delta)={rho_q2_custom:.2f}, p={p_q2_custom:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["user_customized"])
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel("Q2 score")
    ax.set_title("Q2: Concrete Description (Pre -> Post)")
    ax.grid(axis="y", alpha=0.2)

    # Panel 4: Q3 pre/post
    ax = axes[3]
    for _, row in merged.iterrows():
        idx = participants.index(row["participant"])
        ax.plot([idx - 0.12, idx + 0.12], [row["q3_pre"], row["q3_post"]],
                color="#B0BEC5", linewidth=1.2, alpha=0.9)
        ax.scatter(idx - 0.12, row["q3_pre"], color=METHOD_COLORS["baseline_text"], s=35, zorder=3)
        ax.scatter(idx + 0.12, row["q3_post"], color=METHOD_COLORS["gp_refined"], s=35, zorder=3)
    rho_q3_ours, p_q3_ours = scipy_stats.spearmanr(merged["ours_rank"], merged["q3_delta"], nan_policy="omit")
    rho_q3_custom, p_q3_custom = scipy_stats.spearmanr(merged["custom_rank"], merged["q3_delta"], nan_policy="omit")
    ax.text(0.99, 0.96, f"Ours: Spearman(rank, Q3 delta)={rho_q3_ours:.2f}, p={p_q3_ours:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["gp_refined"])
    ax.text(0.99, 0.89, f"Custom: Spearman(rank, Q3 delta)={rho_q3_custom:.2f}, p={p_q3_custom:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=METHOD_COLORS["user_customized"])
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel("Q3 score")
    ax.set_title("Q3: Mental Picture (Pre -> Post)")
    ax.grid(axis="y", alpha=0.2)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(participants, fontsize=10)
    axes[3].set_xlabel("Participant")

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS["baseline_text"], label="Pre"),
        mpatches.Patch(color=METHOD_COLORS["gp_refined"], label="Post"),
    ]
    axes[1].legend(handles=legend_handles, fontsize=8, loc="lower left")

    fig.tight_layout()
    _save(fig, "rank_questionnaire_association.png")


# =========================================================================
# 20. Descriptor-wise Ratings by Method
# =========================================================================
def fig_descriptor_rating_by_method(scores):
    descriptor_scores = (
        scores.groupby(["preset", "method"])["score"]
        .mean()
        .reset_index()
    )
    descriptor_scores["method_label"] = descriptor_scores["method"].map(METHOD_LABELS)
    descriptor_order = sorted(descriptor_scores["preset"].dropna().unique())
    method_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=descriptor_scores,
        x="preset",
        y="score",
        hue="method_label",
        order=descriptor_order,
        hue_order=method_labels,
        palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER},
        ax=ax,
    )
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Average Rating (1-7)")
    ax.set_title("Average Rating by Descriptor and Method")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Method", fontsize=8, title_fontsize=9)
    ax.set_ylim(0, 7.2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "descriptor_rating_results.png")


# =========================================================================
# 21. Descriptor-wise Ranking by Method
# =========================================================================
def fig_descriptor_rank_by_method(scores):
    descriptor_ranks = (
        scores.groupby(["preset", "method"])["rank_position"]
        .mean()
        .reset_index()
    )
    descriptor_ranks["method_label"] = descriptor_ranks["method"].map(METHOD_LABELS)
    descriptor_order = sorted(descriptor_ranks["preset"].dropna().unique())
    method_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=descriptor_ranks,
        x="preset",
        y="rank_position",
        hue="method_label",
        order=descriptor_order,
        hue_order=method_labels,
        palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER},
        ax=ax,
    )
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Average Rank (lower is better)")
    ax.set_title("Average Rank by Descriptor and Method")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.legend(title="Method", fontsize=8, title_fontsize=9)
    ax.set_ylim(0, 4.5)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "descriptor_rank_results.png")


# =========================================================================
# main
# =========================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()

    print("Generating figures...")

    fig_method_scores_boxswarm(data["scores"])
    fig_participant_profiles(data["scores"])
    fig_uplift_histogram(data["uplifts"])
    fig_score_heatmap(data["scores"])
    fig_win_rates(data["scores"])
    fig_win_rates_by_room_condition(data["scores"])
    fig_gp_learning_curves(data["gp"])
    fig_gp_quality_vs_outcome(data["scores"], data["gp"])
    fig_warning_distribution()
    fig_tag_weight_drift(data["drift"])
    fig_entropy_shift(data["pref_summary"])
    fig_session_timelines(data["summary"])
    fig_time_vs_quality(data["scores"], data["summary"])
    fig_average_rank(data["scores"])
    fig_average_rating(data["scores"])
    fig_average_rating_by_room_condition(data["scores"])
    fig_average_rank_by_room_condition(data["scores"])
    fig_average_rank_per_participant(data["scores"])
    fig_participant_rank_questionnaire_association(data["scores"])
    fig_descriptor_rating_by_method(data["scores"])
    fig_descriptor_rank_by_method(data["scores"])

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
