"""
Compute aggregated metrics from parsed CSVs.

Reads from outputs/ CSVs produced by parse_sessions.py and prints
summary tables + saves augmented CSVs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as shannon_entropy

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_tables():
    return {
        "scores": pd.read_csv(OUTPUT_DIR / "location_scores.csv"),
        "summary": pd.read_csv(OUTPUT_DIR / "session_summary.csv"),
        "gp": pd.read_csv(OUTPUT_DIR / "gp_diagnostics.csv"),
        "drift": pd.read_csv(OUTPUT_DIR / "preference_drift.csv"),
        "warnings": pd.read_csv(OUTPUT_DIR / "gp_warnings.csv"),
    }


# ---------------------------------------------------------------------------
# A. Method Comparison Metrics
# ---------------------------------------------------------------------------
def method_comparison(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}

    grand_mean = scores.groupby("method")["score"].agg(["mean", "std", "count"])
    grand_mean.columns = ["mean_score", "std_score", "n_observations"]
    results["grand_mean"] = grand_mean.round(3)

    participant_mean = (
        scores.groupby(["participant", "method"])["score"]
        .mean()
        .unstack("method")
    )
    results["participant_mean"] = participant_mean.round(3)

    def _win_stats(group):
        max_score = group["score"].max()
        winners = group[group["score"] == max_score]
        return winners["method"].tolist()

    location_winners = (
        scores.groupby(["session_id", "location"])
        .apply(_win_stats, include_groups=False)
        .reset_index(name="winners")
    )
    n_locations = len(location_winners)
    methods = scores["method"].unique()
    win_counts = {}
    top2_counts = {}
    for m in methods:
        win_counts[m] = location_winners["winners"].apply(lambda w: m in w).sum()

    sorted_locs = (
        scores.groupby(["session_id", "location", "method"])["score"]
        .mean()
        .reset_index()
    )
    for key, grp in sorted_locs.groupby(["session_id", "location"]):
        top2 = grp.nlargest(2, "score")["method"].tolist()
        for m in methods:
            top2_counts[m] = top2_counts.get(m, 0) + (1 if m in top2 else 0)

    win_df = pd.DataFrame({
        "method": methods,
        "top1_wins": [win_counts[m] for m in methods],
        "top1_rate": [win_counts[m] / n_locations for m in methods],
        "top2_count": [top2_counts.get(m, 0) for m in methods],
        "top2_rate": [top2_counts.get(m, 0) / n_locations for m in methods],
    }).set_index("method").round(3)
    results["win_rates"] = win_df

    pivot = scores.pivot_table(
        index=["session_id", "location", "participant"],
        columns="method", values="score"
    ).reset_index()
    uplift_rows = []
    for _, row in pivot.iterrows():
        base = row.get("baseline_text")
        style = row.get("style_transfer")
        gp = row.get("gp_refined")
        custom = row.get("user_customized")
        uplift_rows.append({
            "session_id": row["session_id"],
            "location": row["location"],
            "participant": row["participant"],
            "gp_vs_baseline": gp - base if pd.notna(gp) and pd.notna(base) else None,
            "gp_vs_style": gp - style if pd.notna(gp) and pd.notna(style) else None,
            "custom_vs_baseline": custom - base if pd.notna(custom) and pd.notna(base) else None,
            "custom_vs_style": custom - style if pd.notna(custom) and pd.notna(style) else None,
            "gp_vs_custom": gp - custom if pd.notna(gp) and pd.notna(custom) else None,
        })
    uplift_df = pd.DataFrame(uplift_rows)
    results["uplifts"] = uplift_df

    target_vs_transfer = (
        scores.groupby(["is_target_room", "method"])["score"]
        .agg(["mean", "std", "count"])
    )
    target_vs_transfer.columns = ["mean_score", "std_score", "n"]
    results["target_vs_transfer"] = target_vs_transfer.round(3)

    return results


# ---------------------------------------------------------------------------
# B. GP Quality Metrics
# ---------------------------------------------------------------------------
def gp_quality(gp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}

    last_round = gp.loc[gp.groupby("session_id")["round"].idxmax()]
    results["final_round"] = last_round[
        ["session_id", "round", "pairwise_accuracy", "spearman_rho", "kendall_tau",
         "image_variance", "top_tag", "top_mu"]
    ].round(3)

    def _acc_slope(grp):
        if len(grp) < 2:
            return np.nan
        x = grp["round"].values.astype(float)
        y = grp["pairwise_accuracy"].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(x[mask], y[mask], 1)[0]

    slopes = gp.groupby("session_id").apply(_acc_slope, include_groups=False).rename("accuracy_slope")
    results["accuracy_slopes"] = slopes.round(4)

    first_round = gp.loc[gp.groupby("session_id")["round"].idxmin()]
    sigma_cols = last_round[["session_id"]].copy()
    sigma_cols["first_image_var"] = first_round["image_variance"].values
    sigma_cols["last_image_var"] = last_round["image_variance"].values
    sigma_cols["var_change"] = sigma_cols["last_image_var"] - sigma_cols["first_image_var"]
    results["variance_change"] = sigma_cols.round(4)

    return results


# ---------------------------------------------------------------------------
# C. Interaction Cost Metrics
# ---------------------------------------------------------------------------
def interaction_cost(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["time_per_round_s"] = (df["hitl_duration_s"] / df["n_rounds"]).round(1)
    return df[["session_id", "participant", "total_duration_s", "hitl_duration_s",
               "n_rounds", "time_per_round_s", "avg_generation_s", "n_locations_rated"]]


# ---------------------------------------------------------------------------
# D. Preference Drift Metrics
# ---------------------------------------------------------------------------
def preference_drift_metrics(drift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session_id, grp in drift.groupby("session_id"):
        has_both = grp.dropna(subset=["initial_weight", "final_weight"])

        if len(has_both) >= 5:
            init_ranked = has_both.nlargest(5, "initial_weight")["tag"].tolist()
            final_ranked = has_both.nlargest(5, "final_weight")["tag"].tolist()
            overlap = len(set(init_ranked) & set(final_ranked))
            union = len(set(init_ranked) | set(final_ranked))
            top5_jaccard = overlap / union if union else 0
        else:
            top5_jaccard = np.nan

        if len(has_both) > 0:
            iw = has_both["initial_weight"].values
            fw = has_both["final_weight"].values
            iw_norm = iw / iw.sum() if iw.sum() > 0 else iw
            fw_norm = fw / fw.sum() if fw.sum() > 0 else fw
            init_entropy = shannon_entropy(iw_norm + 1e-12)
            final_entropy = shannon_entropy(fw_norm + 1e-12)
            max_delta = has_both["delta_weight"].abs().max()
            from scipy.stats import spearmanr
            rho, _ = spearmanr(
                has_both["initial_weight"].values,
                has_both["final_weight"].values,
            )
        else:
            init_entropy = final_entropy = max_delta = rho = np.nan

        rows.append({
            "session_id": session_id,
            "n_shared_tags": len(has_both),
            "top5_jaccard": round(top5_jaccard, 3),
            "initial_entropy": round(init_entropy, 3),
            "final_entropy": round(final_entropy, 3),
            "entropy_change": round(final_entropy - init_entropy, 3) if not np.isnan(final_entropy) else np.nan,
            "max_abs_delta_weight": round(max_delta, 3) if not np.isnan(max_delta) else np.nan,
            "rank_spearman_rho": round(rho, 3) if not np.isnan(rho) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Warning summary
# ---------------------------------------------------------------------------
def warning_summary(warnings: pd.DataFrame) -> pd.DataFrame:
    if warnings.empty:
        return pd.DataFrame()
    return (
        warnings.groupby(["session_id", "warning_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    tables = load_tables()

    print("=" * 60)
    print("A. METHOD COMPARISON METRICS")
    print("=" * 60)
    mc = method_comparison(tables["scores"])
    print("\nGrand mean scores by method:")
    print(mc["grand_mean"])
    print("\nPer-participant mean scores:")
    print(mc["participant_mean"])
    print("\nWin rates (top-1 and top-2):")
    print(mc["win_rates"])
    print("\nUplift summary:")
    uplift_summary = mc["uplifts"][
        ["gp_vs_baseline", "gp_vs_style", "custom_vs_baseline",
         "custom_vs_style", "gp_vs_custom"]
    ].describe().round(3)
    print(uplift_summary)
    print("\nTarget room vs transferred rooms:")
    print(mc["target_vs_transfer"])

    mc["uplifts"].to_csv(OUTPUT_DIR / "uplifts.csv", index=False)

    print("\n" + "=" * 60)
    print("B. GP QUALITY METRICS")
    print("=" * 60)
    gq = gp_quality(tables["gp"])
    print("\nFinal-round diagnostics:")
    print(gq["final_round"].to_string())
    print("\nAccuracy slope (improvement per round):")
    print(gq["accuracy_slopes"])
    print("\nImage variance change (first -> last round):")
    print(gq["variance_change"].to_string())

    print("\n" + "=" * 60)
    print("C. INTERACTION COST METRICS")
    print("=" * 60)
    ic = interaction_cost(tables["summary"])
    print(ic.to_string())

    print("\n" + "=" * 60)
    print("D. PREFERENCE DRIFT METRICS")
    print("=" * 60)
    pd_metrics = preference_drift_metrics(tables["drift"])
    print(pd_metrics.to_string())
    pd_metrics.to_csv(OUTPUT_DIR / "preference_drift_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("E. WARNING SUMMARY")
    print("=" * 60)
    ws = warning_summary(tables["warnings"])
    if not ws.empty:
        print(ws.to_string())
        ws.to_csv(OUTPUT_DIR / "warning_summary.csv", index=False)
    else:
        print("No warnings found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
