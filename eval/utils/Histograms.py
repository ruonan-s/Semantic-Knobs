import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Shared condition palette across all auto-generated plots
# Keep these colors consistent in rank_histogram, score_by_condition,
# average_rank, and tag_weight_butterfly.
CONDITION_COLORS = {
    "text": "#6B7280",            # gray
    "text+image": "#5D576B",      # custom
    "user_customized": "#F7D08A", # custom
    "Ours": "#F7567C",            # custom
}
CONDITION_ORDER = ["text", "text+image", "user_customized", "Ours"]


def categorize_method(filename: str) -> str:
    """Categorize image filenames into method types."""
    if filename.startswith("eval_alpha_"):
        return "Ours"
    elif filename == "user_customized.png":
        return "user_customized"
    elif filename in ("sd_style_transfer.png", "llm_style_transfer.png"):
        return "text+image"
    elif filename == "sd_baseline_text.png":
        return "text"
    else:
        # Everything else (Calm_*, Inviting_*, Refreshing_*, Cozy_*, etc.) is "text"
        return "text"


def load_rank_order(json_path: str) -> dict:
    """Load rank order JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_rank_histogram(data: dict) -> dict:
    """
    Compute histogram of ranks for each method.
    Returns: {method: {rank: count}}
    
    Supports both old format (string filenames) and new format (dict with "image" and "score").
    """
    rankings = data.get("rankings", {})
    histogram = defaultdict(lambda: defaultdict(int))
    
    for room, ranks in rankings.items():
        for rank_str, value in ranks.items():
            rank = int(rank_str)
            
            # Handle both old format (string) and new format (dict)
            if isinstance(value, str):
                # Old format: {"1": "filename.png"}
                filename = value
            elif isinstance(value, dict):
                # New format: {"1": {"image": "filename.png", "score": 6}}
                filename = value.get("image", "")
            else:
                filename = ""
            
            method = categorize_method(filename)
            histogram[method][rank] += 1
    
    return histogram


def plot_rank_histogram(histogram: dict, output_path: str = None, title: str = "Rank Distribution by Method", show: bool = True):
    """
    Plot a grouped bar chart showing rank distribution for each method.
    X-axis: Ranks (1..N)
    Colors: Different methods
    """
    methods = [m for m in CONDITION_ORDER if m in histogram]
    # Preserve unknown methods if they appear
    methods += [m for m in sorted(histogram.keys()) if m not in methods]
    all_ranks = sorted({rank for counts in histogram.values() for rank in counts.keys()})
    ranks = all_ranks if all_ranks else [1, 2, 3, 4]
    
    # Prepare data matrix: rows = methods, cols = ranks
    data_matrix = np.zeros((len(methods), len(ranks)))
    for i, method in enumerate(methods):
        for j, rank in enumerate(ranks):
            data_matrix[i, j] = histogram[method].get(rank, 0)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ranks))  # x-axis is now ranks
    n_methods = len(methods)
    width = 0.8 / n_methods  # divide available space among methods
    
    # Create grouped bars - one bar per method at each rank position
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        color = CONDITION_COLORS.get(method, '#9CA3AF')
        bars = ax.bar(x + offset, data_matrix[i, :], width, 
                     label=method, color=color, 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    ax.legend(title='Method', loc='upper right', fontsize=9)
    ax.set_ylim(0, max(data_matrix.max() + 2, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def compute_score_statistics(data: dict) -> dict:
    """
    Compute preference score statistics for each method.
    Returns: {method: {"scores": [list of scores], "mean": float, "std": float}}
    
    Only works with new format that includes "score" field.
    """
    rankings = data.get("rankings", {})
    score_data = defaultdict(list)
    
    for room, ranks in rankings.items():
        for rank_str, value in ranks.items():
            # Only process new format with scores
            if isinstance(value, dict) and "score" in value and "image" in value:
                filename = value["image"]
                score = value["score"]
                method = categorize_method(filename)
                score_data[method].append(score)
    
    # Compute statistics
    statistics = {}
    for method, scores in score_data.items():
        if scores:
            statistics[method] = {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
    
    return statistics


def categorize_condition(filename: str) -> str:
    """
    Categorize image filenames into condition types for score analysis.
    
    Returns:
    - "text": Text-only baseline (sd_baseline_text.png or generic baselines)
    - "text+image": Style transfer (sd_style_transfer.png or llm_style_transfer.png)
    - "user_customized": User customized manual tags/weights baseline
    - "Ours": eval_alpha_* images (custom embedding fusion)
    """
    if filename == "user_customized.png":
        return "user_customized"
    if filename in ("sd_style_transfer.png", "llm_style_transfer.png"):
        return "text+image"
    elif filename.startswith("eval_alpha_"):
        return "Ours"
    elif filename == "sd_baseline_text.png":
        return "text"
    else:
        # Everything else (Calm_*, Inviting_*, Refreshing_*, Cozy_*, etc.) is "text"
        return "text"


def plot_score_by_condition(data: dict, output_path: str = None, title: str = "Preference Score Distribution by Condition", show: bool = True):
    """
    Plot preference score distribution for each condition (method).
    Shows box plots of scores (1-7) for each condition.
    
    Conditions:
    1. Text-only baseline (sd_baseline_text.png or generic) → "text"
    2. Img2img style transfer (sd_style_transfer.png) → "text+image"
    3. User customized baseline (user_customized.png) → "user_customized"
    4. Custom embedding fusion (eval_alpha_1.00_*) → "Ours"
    
    Only works with new format that includes "score" field.
    """
    rankings = data.get("rankings", {})
    condition_scores = defaultdict(list)  # {condition: [list of scores]}
    
    # Collect scores for each condition
    for room, ranks in rankings.items():
        for rank_str, value in ranks.items():
            if isinstance(value, dict) and "score" in value and "image" in value:
                filename = value["image"]
                score = value["score"]
                condition = categorize_condition(filename)
                if condition:
                    condition_scores[condition].append(score)
    
    if not condition_scores:
        print("[HISTOGRAM] No score data available for score-by-condition plot")
        return None, None
    
    # Define order of conditions
    conditions = [c for c in CONDITION_ORDER if c in condition_scores]
    conditions += [c for c in sorted(condition_scores.keys()) if c not in conditions]
    score_data = [condition_scores[c] for c in conditions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create box plot with different colors for each condition
    positions = list(range(1, len(conditions) + 1))
    
    bp = ax.boxplot(score_data, positions=positions, widths=0.6, 
                    patch_artist=True,
                    medianprops=dict(color='#000000', linewidth=2),
                    whiskerprops=dict(color='#1E3A8A', linewidth=1.5),
                    capprops=dict(color='#1E3A8A', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#EF4444', markersize=6, alpha=0.5))
    
    # Set colors for each box
    for patch, condition in zip(bp['boxes'], conditions):
        color = CONDITION_COLORS.get(condition, '#9CA3AF')
        patch.set_facecolor(color)
        patch.set_edgecolor('#1E3A8A')
        patch.set_linewidth(1.5)
        patch.set_alpha(0.7)
    
    # Add mean markers
    means = [np.mean(scores) for scores in score_data]
    ax.plot(positions, means, 'D', color='#059669', markersize=8, label='Mean', zorder=3)
    
    # Add value labels for mean
    for i, pos in enumerate(positions):
        mean_val = means[i]
        n = len(score_data[i])
        
        # Mean label
        ax.text(pos + 0.15, mean_val, f'{mean_val:.2f}', 
               va='center', ha='left', fontsize=9, color='#059669', fontweight='bold')
        
        # Count label
        ax.text(pos, 0.5, f'n={n}', 
               va='center', ha='center', fontsize=8, color='#666', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Preference Score (1-7)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0.5, 7.5)
    ax.set_yticks(range(1, 8))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Score-by-condition chart saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_average_rank(histogram: dict, output_path: str = None, title: str = "Average Rank by Method", show: bool = True):
    """
    Plot average rank for each method (lower is better).
    """
    methods = []
    avg_ranks = []
    
    for method, ranks in histogram.items():
        total_rank = sum(rank * count for rank, count in ranks.items())
        total_count = sum(ranks.values())
        if total_count > 0:
            methods.append(method)
            avg_ranks.append(total_rank / total_count)
    
    # Sort by average rank (lower is better)
    sorted_indices = np.argsort(avg_ranks)
    methods = [methods[i] for i in sorted_indices]
    avg_ranks = [avg_ranks[i] for i in sorted_indices]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = [CONDITION_COLORS.get(m, "#9CA3AF") for m in methods]
    bars = ax.barh(methods, avg_ranks, color=colors, edgecolor= None, linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, avg_ranks):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)
    
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    max_rank = max((rank for counts in histogram.values() for rank in counts.keys()), default=4)
    ax.set_xlim(0, max_rank + 1)
    ax.axvline(x=(max_rank + 1) / 2, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Average rank chart saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def _load_weight_map(path: Path, tags_key: str = "selected_tags") -> dict:
    """Load tag->weight map from a json file."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    weights = data.get("weights", {}) or {}
    tags = data.get(tags_key, []) or list(weights.keys())

    out = {}
    for tag in tags:
        try:
            out[str(tag)] = float(weights.get(tag, 0.0))
        except (TypeError, ValueError):
            out[str(tag)] = 0.0
    return out


def plot_tag_weight_butterfly(
    session_dir: str,
    output_path: str = None,
    title: str = "User vs GP Tag Weights",
    show: bool = True,
    normalize_weights: bool = True
):
    """
    Plot butterfly (back-to-back) chart comparing manual user weights vs GP-refined weights.

    Groups:
    - Shared tags (both)
    - User-only tags
    - GP-only tags
    """
    session_path = Path(session_dir)
    user_path = session_path / "impression" / "user_manual_weights.json"
    gp_path = session_path / "refined_preferences_v2.json"

    user_weights = _load_weight_map(user_path, tags_key="selected_tags")
    gp_weights = _load_weight_map(gp_path, tags_key="tags")

    if not user_weights or not gp_weights:
        print("[HISTOGRAM] Skipping butterfly chart (missing user manual weights or GP refined weights)")
        return None, None

    # Normalize each side independently so magnitudes are directly comparable
    # (user sliders can be arbitrary in [0,1], GP is softmax-normalized).
    if normalize_weights:
        u_sum = sum(user_weights.values())
        g_sum = sum(gp_weights.values())
        if u_sum > 0:
            user_weights = {k: v / u_sum for k, v in user_weights.items()}
        if g_sum > 0:
            gp_weights = {k: v / g_sum for k, v in gp_weights.items()}

    user_tags = set(user_weights.keys())
    gp_tags = set(gp_weights.keys())

    shared = list(user_tags & gp_tags)
    user_only = list(user_tags - gp_tags)
    gp_only = list(gp_tags - user_tags)

    # Sort each group by max(weight) descending
    shared.sort(key=lambda t: max(user_weights.get(t, 0.0), gp_weights.get(t, 0.0)), reverse=True)
    user_only.sort(key=lambda t: user_weights.get(t, 0.0), reverse=True)
    gp_only.sort(key=lambda t: gp_weights.get(t, 0.0), reverse=True)

    rows = shared + user_only + gp_only
    if not rows:
        print("[HISTOGRAM] Skipping butterfly chart (no tags to plot)")
        return None, None

    y = np.arange(len(rows))
    user_vals = np.array([user_weights.get(t, 0.0) for t in rows], dtype=float)
    gp_vals = np.array([gp_weights.get(t, 0.0) for t in rows], dtype=float)

    labels = list(rows)

    height = max(6, min(20, 2.8 + 0.45 * len(rows)))
    fig = plt.figure(figsize=(11, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.35], wspace=0.02)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1], sharey=ax_l)
    ax_r = fig.add_subplot(gs[0, 2], sharey=ax_l)

    # Clean palette
    bg = "#F7F8FA"
    shared_user_c = CONDITION_COLORS["user_customized"]  # purple
    shared_gp_c = CONDITION_COLORS["Ours"]               # amber
    center_text = "#2F3747"
    dim_text = "#8A97AB"

    # Uniform side colors and label color per request
    left_colors = [shared_user_c] * len(rows)
    right_colors = [shared_gp_c] * len(rows)
    label_colors = ["#000000"] * len(rows)

    fig.patch.set_facecolor(bg)
    ax_l.set_facecolor(bg)
    ax_c.set_facecolor(bg)
    ax_r.set_facecolor(bg)

    # Split-panel butterfly with center label column
    bar_h = 0.72
    ax_l.barh(y, -user_vals, color=left_colors, alpha=1.0, edgecolor=bg, linewidth=1.0, height=bar_h)
    ax_r.barh(y, gp_vals, color=right_colors, alpha=1.0, edgecolor=bg, linewidth=1.0, height=bar_h)

    # Group separators (shared / user-only / gp-only)
    shared_n = len(shared)
    user_only_n = len(user_only)
    if shared_n > 0 and user_only_n > 0:
        ax_l.axhline(shared_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)
        ax_c.axhline(shared_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)
        ax_r.axhline(shared_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)
    if shared_n + user_only_n > 0 and len(gp_only) > 0:
        ax_l.axhline(shared_n + user_only_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)
        ax_c.axhline(shared_n + user_only_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)
        ax_r.axhline(shared_n + user_only_n - 0.5, color="#D7DBE3", linestyle="-", linewidth=1.0)

    # Symmetric x limits
    user_max = float(user_vals.max()) if len(user_vals) > 0 else 0.0
    gp_max = float(gp_vals.max()) if len(gp_vals) > 0 else 0.0
    x_max = float(max(user_max, gp_max, 0.01)) * 1.22
    ax_l.set_xlim(-x_max, 0)
    ax_r.set_xlim(0, x_max)

    # Hide default axes in center; keep top ticks on left/right
    ax_c.set_xlim(0, 1)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    for ax_side in (ax_l, ax_r):
        ax_side.set_yticks([])
        ax_side.tick_params(axis="y", length=0)
        ax_side.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False, length=0)
        for s in ax_side.spines.values():
            s.set_visible(False)
        ax_side.set_xticks([])

    # first row at top
    ax_l.invert_yaxis()

    # Center labels (with membership markers already encoded in text)
    for yi, label, c in zip(y, labels, label_colors):
        ax_c.text(0.5, yi, label, ha="center", va="center", fontsize=11.5, color=c, fontweight="600")

    # Title
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=center_text, y=0.965)

    # Legend-like headers at top (left and right), concise
    fig.text(0.06, 0.905, "User-defined", ha="left", va="bottom",
             fontsize=11.5, color="#4C5A6F", fontweight="600")

    fig.text(0.94, 0.905, "GP-refined", ha="right", va="bottom",
             fontsize=11.5, color="#4C5A6F", fontweight="600")

    # Value labels for every non-zero bar
    pad = x_max * 0.02
    for yi, v in enumerate(user_vals):
        if v > 0:
            ax_l.text(-v - pad, yi, f"{v:.3f}", ha="right", va="center", fontsize=9, color=dim_text)
    for yi, v in enumerate(gp_vals):
        if v > 0:
            ax_r.text(v + pad, yi, f"{v:.3f}", ha="left", va="center", fontsize=9, color=dim_text)

    # Summary statistics above plot
    union_n = len(user_tags | gp_tags)
    jaccard = (len(shared) / union_n) if union_n > 0 else 0.0
    summary = (
        f"Shared: {len(shared)}   |   User-only: {len(user_only)}   |   GP-only: {len(gp_only)}"
        f"   |   Jaccard: {jaccard:.2f}"
    )
    fig.text(0.5, 0.948, summary, ha="center", va="top", fontsize=9.5, color=dim_text)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Tag butterfly chart saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax_l, ax_c, ax_r)


def generate_histograms(json_path: str, output_dir: str = None, show: bool = False):
    """
    Generate histograms from rank order JSON (for programmatic use).
    Called automatically after all rankings are completed.
    """
    print(f"[HISTOGRAM] Loading rank order from: {json_path}")
    data = load_rank_order(json_path)
    session_name = data.get("session_log", "unknown_session")
    num_rankings = len(data.get("rankings", {}))
    print(f"[HISTOGRAM] Session: {session_name}, Rankings: {num_rankings}")
    
    # Compute histogram
    histogram = compute_rank_histogram(data)
    
    # Compute score statistics if available
    score_stats = compute_score_statistics(data)
    if score_stats:
        print(f"[HISTOGRAM] Preference scores found - computing statistics")
        for method, stats in sorted(score_stats.items()):
            print(f"  {method}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, n={stats['count']}")
    
    # Set up output paths
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot histograms (save only, no display)
    hist_path = output_dir / "rank_histogram.png"
    avg_path = output_dir / "average_rank.png"
    score_condition_path = output_dir / "score_by_condition.png"
    butterfly_path = output_dir / "tag_weight_butterfly.png"
    
    print(f"[HISTOGRAM] Saving rank_histogram.png to: {hist_path}")
    plot_rank_histogram(histogram, str(hist_path), 
                       title=f"Rank Distribution - {session_name}", show=show)
    
    print(f"[HISTOGRAM] Saving average_rank.png to: {avg_path}")
    plot_average_rank(histogram, str(avg_path),
                     title=f"Average Rank - {session_name}", show=show)
    
    # Generate score-by-condition plot if score data is available
    if score_stats:
        print(f"[HISTOGRAM] Saving score_by_condition.png to: {score_condition_path}")
        plot_score_by_condition(data, str(score_condition_path),
                               title=f"Preference Score by Condition - {session_name}", show=show)

    # Generate user-vs-GP butterfly chart if both weight files exist
    print(f"[HISTOGRAM] Saving tag_weight_butterfly.png to: {butterfly_path}")
    plot_tag_weight_butterfly(
        session_dir=str(Path(json_path).parent),
        output_path=str(butterfly_path),
        title=f"User vs GP Tag Weights - {session_name}",
        show=show
    )
    
    print(f"[HISTOGRAM] ✅ Generated histograms for {session_name}")
    return histogram


def main(json_path: str, output_dir: str = None):
    """
    Main function to generate histograms from rank order JSON.
    """
    # Load data
    data = load_rank_order(json_path)
    session_name = data.get("session_log", "unknown_session")
    
    print(f"Processing session: {session_name}")
    print(f"Number of rooms: {len(data.get('rankings', {}))}")
    
    # Compute histogram
    histogram = compute_rank_histogram(data)
    
    # Print summary
    print("\nRank Distribution Summary:")
    print("-" * 50)
    for method, ranks in sorted(histogram.items()):
        total = sum(ranks.values())
        avg_rank = sum(r * c for r, c in ranks.items()) / total if total > 0 else 0
        print(f"{method}:")
        print(f"  Ranks: {dict(sorted(ranks.items()))}")
        print(f"  Average Rank: {avg_rank:.2f}")
    
    # Set up output paths
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute score statistics if available
    score_stats = compute_score_statistics(data)
    
    # Plot histograms
    hist_path = output_dir / "rank_histogram.png"
    avg_path = output_dir / "average_rank.png"
    score_condition_path = output_dir / "score_by_condition.png"
    butterfly_path = output_dir / "tag_weight_butterfly.png"
    
    plot_rank_histogram(histogram, str(hist_path), 
                       title=f"Rank Distribution - {session_name}", show=True)
    plot_average_rank(histogram, str(avg_path),
                     title=f"Average Rank - {session_name}", show=True)
    
    # Generate score-by-condition plot if score data is available
    if score_stats:
        plot_score_by_condition(data, str(score_condition_path),
                               title=f"Preference Score by Condition - {session_name}", show=True)

    plot_tag_weight_butterfly(
        session_dir=str(Path(json_path).parent),
        output_path=str(butterfly_path),
        title=f"User vs GP Tag Weights - {session_name}",
        show=True
    )
    
    return histogram


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Default to the sample file
        json_path = "/home/nancy/Semantic-Knobs/eval/session_logs/eval_test_5_Calm_Home_Office_Sample_2026-01-27_19-20-14/rank_order.json"
    else:
        json_path = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(json_path, output_dir)

