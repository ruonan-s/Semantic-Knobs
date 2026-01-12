import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def categorize_method(filename: str) -> str:
    """Categorize image filenames into method types."""
    if filename.startswith("eval_alpha_"):
        return "Ours"
    elif filename == "llm_style_transfer.png":
        return "LLM text+image"
    elif filename == "llm_baseline_tags.png":
        return "LLM text+tags"
    else:
        # Everything else (Calm_*, Inviting_*, Refreshing_*, Cozy_*, etc.) is "LLM text"
        return "LLM text"


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
    X-axis: Ranks (1, 2, 3, 4)
    Colors: Different methods
    """
    methods = sorted(histogram.keys())
    ranks = [1, 2, 3, 4]
    
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
    
    # Color scheme for methods - similar colors for related methods
    # Baselines (LLM): shades of gray/blue
    # Ours: red
    method_color_map = {
        'LLM text': '#4B5563',             # Dark gray
        'LLM text+tags': '#6B7280',        # Gray
        'LLM text+image': '#9CA3AF',       # Light gray
        'Ours': '#DC2626',                 # Red
    }
    
    # Create grouped bars - one bar per method at each rank position
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        color = method_color_map.get(method, '#888888')
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
    - "LLM text": Original/reference images (any file not matching the special cases below)
    - "LLM text+image": llm_style_transfer.png
    - "LLM text+tags": llm_baseline_tags.png
    - "Ours": eval_alpha_* images
    """
    if filename == "llm_style_transfer.png":
        return "LLM text+image"
    elif filename == "llm_baseline_tags.png":
        return "LLM text+tags"
    elif filename.startswith("eval_alpha_"):
        return "Ours"
    else:
        # Everything else (Calm_*, Inviting_*, Refreshing_*, Cozy_*, etc.) is "LLM text"
        return "LLM text"


def plot_score_by_condition(data: dict, output_path: str = None, title: str = "Preference Score Distribution by Condition", show: bool = True):
    """
    Plot preference score distribution for each condition (method).
    Shows box plots of scores (1-7) for each condition.
    
    Conditions:
    1. Original images (Refreshing_*, Cozy_*, etc.) → "LLM text"
    2. "llm_style_transfer.png" → "LLM text+image"
    3. "llm_baseline_tags.png" → "LLM text+tags"
    4. "eval_alpha_1.00_*" → "Ours"
    
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
    condition_order = ["LLM text", "LLM text+image", "LLM text+tags", "Ours"]
    conditions = [c for c in condition_order if c in condition_scores]
    score_data = [condition_scores[c] for c in conditions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot with different colors for each condition
    positions = list(range(1, len(conditions) + 1))
    
    # Color scheme for conditions
    condition_colors = {
        'LLM text': '#6B7280',       # Gray
        'LLM text+image': '#3B82F6', # Blue
        'LLM text+tags': '#8B5CF6',  # Purple
        'Ours': '#DC2626',           # Red
    }
    
    bp = ax.boxplot(score_data, positions=positions, widths=0.6, 
                    patch_artist=True,
                    medianprops=dict(color='#000000', linewidth=2),
                    whiskerprops=dict(color='#1E3A8A', linewidth=1.5),
                    capprops=dict(color='#1E3A8A', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#EF4444', markersize=6, alpha=0.5))
    
    # Set colors for each box
    for patch, condition in zip(bp['boxes'], conditions):
        color = condition_colors.get(condition, '#93C5FD')
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
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.barh(methods, avg_ranks, color=colors, edgecolor= None, linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, avg_ranks):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)
    
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 5)
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Average rank chart saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


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
    
    plot_rank_histogram(histogram, str(hist_path), 
                       title=f"Rank Distribution - {session_name}", show=True)
    plot_average_rank(histogram, str(avg_path),
                     title=f"Average Rank - {session_name}", show=True)
    
    # Generate score-by-condition plot if score data is available
    if score_stats:
        plot_score_by_condition(data, str(score_condition_path),
                               title=f"Preference Score by Condition - {session_name}", show=True)
    
    return histogram


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Default to the sample file
        json_path = "/home/nancy/Semantic-Knobs/eval/session_logs/eval_P04_Inviting_Livingroom_Sample_2026-01-08_17-33-49/rank_order.json"
    else:
        json_path = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(json_path, output_dir)

