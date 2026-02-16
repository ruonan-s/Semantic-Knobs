#!/usr/bin/env python3
"""
Aggregate visualization script for user study data.
Generates comprehensive charts across all sessions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.patches as mpatches

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

def categorize_method(filename: str) -> str:
    """Categorize image filenames into method types."""
    if filename.startswith("eval_alpha_"):
        return "Ours"
    elif filename == "user_customized.png":
        return "user_customized"
    elif filename == "llm_style_transfer.png":
        return "LLM text+image"
    elif filename == "sd_style_transfer.png":
        return "LLM text+image"
    elif filename == "llm_baseline_tags.png":
        return "LLM text+tags"
    elif filename == "sd_baseline_tags.png":
        return "LLM text+tags"
    else:
        return "LLM text"

def extract_adjective(folder_name: str) -> str:
    """Extract adjective from session folder name.
    
    Example: 'eval_P01_Calm_Home_Office_Sample_2026-01-08_09-51-23' -> 'Calm'
    """
    parts = folder_name.split('_')
    if len(parts) >= 3:
        # Adjective is the third part (index 2)
        return parts[2]
    return "Unknown"

def load_all_sessions(session_logs_dir: str):
    """Load all rank_order.json files from session directories."""
    base_path = Path(session_logs_dir)
    all_data = []
    
    for session_dir in sorted(base_path.iterdir()):
        if session_dir.is_dir() and session_dir.name.startswith("eval_P"):
            rank_file = session_dir / "rank_order.json"
            if rank_file.exists():
                with open(rank_file, 'r') as f:
                    data = json.load(f)
                    # Extract participant ID from folder name
                    parts = session_dir.name.split('_')
                    participant = parts[1] if len(parts) > 1 else "Unknown"
                    data['participant'] = participant
                    data['session_folder'] = session_dir.name
                    # Extract adjective from folder name
                    data['adjective'] = extract_adjective(session_dir.name)
                    all_data.append(data)
    
    return all_data

# Locations to exclude from analysis
EXCLUDED_LOCATIONS = {'Restaurant'}

def aggregate_data(all_sessions):
    """Aggregate data across all sessions."""
    # Data structures
    rank_counts = defaultdict(lambda: defaultdict(int))  # {method: {rank: count}}
    scores_by_method = defaultdict(list)
    scores_by_location = defaultdict(lambda: defaultdict(list))
    scores_by_participant = defaultdict(lambda: defaultdict(list))
    ranks_by_participant = defaultdict(lambda: defaultdict(list))  # {participant: {method: [ranks]}}
    first_place_by_location = defaultdict(lambda: defaultdict(int))
    
    # Adjective-based data structures
    ranks_by_adjective = defaultdict(lambda: defaultdict(list))  # {adjective: {method: [ranks]}}
    scores_by_adjective = defaultdict(lambda: defaultdict(list))  # {adjective: {method: [scores]}}
    first_place_by_adjective = defaultdict(lambda: defaultdict(int))  # {adjective: {method: count}}
    
    for session in all_sessions:
        participant = session.get('participant', 'Unknown')
        adjective = session.get('adjective', 'Unknown')
        rankings = session.get('rankings', {})
        
        for location, ranks in rankings.items():
            # Skip excluded locations
            if location in EXCLUDED_LOCATIONS:
                continue
            for rank_str, value in ranks.items():
                rank = int(rank_str)
                
                if isinstance(value, dict):
                    filename = value.get("image", "")
                    score = value.get("score", 0)
                else:
                    filename = value
                    score = 0
                
                method = categorize_method(filename)
                
                # Rank counts
                rank_counts[method][rank] += 1
                
                # Ranks by participant
                ranks_by_participant[participant][method].append(rank)
                
                # Ranks by adjective
                ranks_by_adjective[adjective][method].append(rank)
                
                # Scores
                if score > 0:
                    scores_by_method[method].append(score)
                    scores_by_location[location][method].append(score)
                    scores_by_participant[participant][method].append(score)
                    scores_by_adjective[adjective][method].append(score)
                
                # First place counts
                if rank == 1:
                    first_place_by_location[location][method] += 1
                    first_place_by_adjective[adjective][method] += 1
    
    return {
        'rank_counts': rank_counts,
        'scores_by_method': scores_by_method,
        'scores_by_location': scores_by_location,
        'scores_by_participant': scores_by_participant,
        'ranks_by_participant': ranks_by_participant,
        'first_place_by_location': first_place_by_location,
        'ranks_by_adjective': ranks_by_adjective,
        'scores_by_adjective': scores_by_adjective,
        'first_place_by_adjective': first_place_by_adjective,
    }

# Color scheme
METHOD_COLORS = {
    'LLM text': '#6B7280',       # Gray
    'LLM text+image': '#3B82F6', # Blue
    'user_customized': '#2563EB',# Strong blue
    'Ours': '#DC2626',           # Red
}
METHOD_ORDER = ['LLM text', 'LLM text+image', 'user_customized', 'Ours']

def plot_aggregate_rank_distribution(rank_counts, output_path=None):
    """Plot 1: Aggregate rank distribution as grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = [1, 2, 3, 4]
    x = np.arange(len(ranks))
    width = 0.18
    
    for i, method in enumerate(METHOD_ORDER):
        if method in rank_counts:
            values = [rank_counts[method].get(r, 0) for r in ranks]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=method, 
                         color=METHOD_COLORS[method], edgecolor='white', linewidth=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Rank (1 = Best)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Rank Distribution by Method (All Sessions)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    ax.legend(title='Method', loc='upper right')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_score_boxplot(scores_by_method, output_path=None):
    """Plot 2: Preference score box plots."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [m for m in METHOD_ORDER if m in scores_by_method]
    data = [scores_by_method[m] for m in methods]
    positions = range(1, len(methods) + 1)
    
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    # Add mean diamonds
    means = [np.mean(d) for d in data]
    ax.scatter(positions, means, marker='D', color='#059669', s=80, zorder=5, label='Mean')
    
    # Add statistics
    for i, (pos, m) in enumerate(zip(positions, methods)):
        mean_val = means[i]
        n = len(data[i])
        ax.text(pos, mean_val + 0.2, f'{mean_val:.2f}', ha='center', fontsize=9, 
               fontweight='bold', color='#059669')
        ax.text(pos, 0.6, f'n={n}', ha='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Preference Score (1-7)', fontweight='bold')
    ax.set_title('Preference Score Distribution by Method', fontweight='bold', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 8)
    ax.set_yticks(range(1, 8))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_first_place_by_location(first_place_by_location, output_path=None):
    """Plot 3: First place wins by location - stacked bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    locations = sorted(first_place_by_location.keys())
    x = np.arange(len(locations))
    width = 0.6
    
    bottom = np.zeros(len(locations))
    
    for method in METHOD_ORDER:
        values = [first_place_by_location[loc].get(method, 0) for loc in locations]
        ax.bar(x, values, width, label=method, color=METHOD_COLORS[method],
               bottom=bottom, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Location', fontweight='bold')
    ax.set_ylabel('First Place Wins', fontweight='bold')
    ax.set_title('First Place Wins by Location', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(locations, rotation=30, ha='right')
    ax.legend(title='Method', loc='upper right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_average_rank_comparison(rank_counts, output_path=None):
    """Plot 4: Average rank comparison (horizontal bar chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    avg_ranks = {}
    for method in METHOD_ORDER:
        if method in rank_counts:
            ranks = rank_counts[method]
            total = sum(ranks.values())
            avg = sum(r * c for r, c in ranks.items()) / total if total > 0 else 0
            avg_ranks[method] = avg
    
    # Sort by average rank (lower is better)
    sorted_methods = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
    values = [avg_ranks[m] for m in sorted_methods]
    colors = [METHOD_COLORS[m] for m in sorted_methods]
    
    bars = ax.barh(sorted_methods, values, color=colors, edgecolor='white', height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Average Rank (lower is better)', fontweight='bold')
    ax.set_title('Average Rank by Method', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 4)
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_participant_heatmap(scores_by_participant, output_path=None):
    """Plot 5: Participant preferences heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    participants = sorted(scores_by_participant.keys())
    methods = METHOD_ORDER
    
    # Build matrix of average scores
    matrix = np.zeros((len(participants), len(methods)))
    for i, p in enumerate(participants):
        for j, m in enumerate(methods):
            scores = scores_by_participant[p].get(m, [])
            matrix[i, j] = np.mean(scores) if scores else np.nan
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=7)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Score', fontweight='bold')
    
    # Add value annotations
    for i in range(len(participants)):
        for j in range(len(methods)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 3.5 or val > 5.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants)
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Participant', fontweight='bold')
    ax.set_title('Average Score by Participant and Method', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_win_rate_pie(rank_counts, output_path=None):
    """Plot 6: First place win rate pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    first_place = {m: rank_counts[m].get(1, 0) for m in METHOD_ORDER if m in rank_counts}
    
    methods = list(first_place.keys())
    values = list(first_place.values())
    colors = [METHOD_COLORS[m] for m in methods]
    
    wedges, texts, autotexts = ax.pie(values, labels=methods, colors=colors,
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values))})',
                                       startangle=90, explode=[0.02]*len(methods),
                                       textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('First Place Win Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_rank_variance(rank_counts, scores_by_method, output_path=None):
    """Plot 7: Rank variance comparison - dual bar chart showing consistency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate statistics for each method
    stats = {}
    for method in METHOD_ORDER:
        if method in rank_counts:
            ranks = rank_counts[method]
            # Expand ranks to list
            rank_list = []
            for r, count in ranks.items():
                rank_list.extend([r] * count)
            
            scores = scores_by_method.get(method, [])
            
            stats[method] = {
                'avg_rank': np.mean(rank_list),
                'rank_std': np.std(rank_list),
                'rank_var': np.var(rank_list),
                'score_std': np.std(scores) if scores else 0,
            }
    
    methods = [m for m in METHOD_ORDER if m in stats]
    
    # Sort by rank variance (most consistent first)
    methods_sorted = sorted(methods, key=lambda m: stats[m]['rank_var'])
    
    # Plot 1: Rank Variance
    ax1 = axes[0]
    x = np.arange(len(methods_sorted))
    rank_vars = [stats[m]['rank_var'] for m in methods_sorted]
    colors = [METHOD_COLORS[m] for m in methods_sorted]
    
    bars1 = ax1.bar(x, rank_vars, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for bar, val in zip(bars1, rank_vars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Method', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Rank Variance', fontweight='bold', fontsize=12)
    ax1.set_title('Rank Variance\n(lower = more consistent)', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_sorted, rotation=15, ha='right', fontsize=10)
    ax1.set_ylim(0, max(rank_vars) * 1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(y=np.mean(rank_vars), color='gray', linestyle='--', alpha=0.7, label='Mean')
    ax1.legend(loc='upper right')
    
    # Plot 2: Score Standard Deviation
    ax2 = axes[1]
    score_stds = [stats[m]['score_std'] for m in methods_sorted]
    
    bars2 = ax2.bar(x, score_stds, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for bar, val in zip(bars2, score_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Method', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Score Std Dev', fontweight='bold', fontsize=12)
    ax2.set_title('Score Standard Deviation\n(lower = more consistent)', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_sorted, rotation=15, ha='right', fontsize=10)
    ax2.set_ylim(0, max(score_stds) * 1.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(y=np.mean(score_stds), color='gray', linestyle='--', alpha=0.7, label='Mean')
    ax2.legend(loc='upper right')
    
    plt.suptitle('Consistency Comparison by Method', fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_avg_rank_by_participant(ranks_by_participant, output_path=None):
    """Plot 8: Average rank by participant for each method."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    participants = sorted(ranks_by_participant.keys())
    methods = METHOD_ORDER
    
    x = np.arange(len(participants))
    width = 0.18
    
    for i, method in enumerate(methods):
        avg_ranks = []
        for p in participants:
            ranks = ranks_by_participant[p].get(method, [])
            avg_ranks.append(np.mean(ranks) if ranks else 0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, avg_ranks, width, label=method, 
                     color=METHOD_COLORS[method], edgecolor='white', linewidth=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Participant', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Rank (lower = better)', fontweight='bold', fontsize=12)
    ax.set_title('Average Rank by Participant', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, fontsize=11)
    ax.set_ylim(0, 4.5)
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.legend(title='Method', loc='upper right', fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Invert y-axis interpretation note
    ax.text(0.02, 0.98, '↓ Lower is better', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', style='italic', color='gray')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_avg_rank_by_adjective(ranks_by_adjective, output_path=None):
    """Plot 9: Average rank by adjective for each method."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    adjectives = sorted(ranks_by_adjective.keys())
    methods = METHOD_ORDER
    
    x = np.arange(len(adjectives))
    width = 0.18
    
    for i, method in enumerate(methods):
        avg_ranks = []
        for adj in adjectives:
            ranks = ranks_by_adjective[adj].get(method, [])
            avg_ranks.append(np.mean(ranks) if ranks else 0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, avg_ranks, width, label=method, 
                     color=METHOD_COLORS[method], edgecolor='white', linewidth=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Adjective', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Rank (lower = better)', fontweight='bold', fontsize=12)
    ax.set_title('Average Rank by Adjective', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(adjectives, fontsize=11)
    ax.set_ylim(0, 4.5)
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.legend(title='Method', loc='upper right', fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Invert y-axis interpretation note
    ax.text(0.02, 0.98, '↓ Lower is better', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', style='italic', color='gray')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_first_place_by_adjective(first_place_by_adjective, output_path=None):
    """Plot 10: First place wins by adjective - stacked bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    adjectives = sorted(first_place_by_adjective.keys())
    x = np.arange(len(adjectives))
    width = 0.6
    
    bottom = np.zeros(len(adjectives))
    
    for method in METHOD_ORDER:
        values = [first_place_by_adjective[adj].get(method, 0) for adj in adjectives]
        bars = ax.bar(x, values, width, label=method, color=METHOD_COLORS[method],
               bottom=bottom, edgecolor='white', linewidth=0.5)
        
        # Add value labels inside bars if there's space
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_y() + bar.get_height()/2,
                       f'{int(val)}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
        bottom += values
    
    ax.set_xlabel('Adjective', fontweight='bold')
    ax.set_ylabel('First Place Wins', fontweight='bold')
    ax.set_title('First Place Wins by Adjective', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(adjectives, fontsize=11)
    ax.legend(title='Method', loc='upper right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_adjective_heatmap(scores_by_adjective, output_path=None):
    """Plot 11: Adjective preferences heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    adjectives = sorted(scores_by_adjective.keys())
    methods = METHOD_ORDER
    
    # Build matrix of average scores
    matrix = np.zeros((len(adjectives), len(methods)))
    for i, adj in enumerate(adjectives):
        for j, m in enumerate(methods):
            scores = scores_by_adjective[adj].get(m, [])
            matrix[i, j] = np.mean(scores) if scores else np.nan
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=7)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Score', fontweight='bold')
    
    # Add value annotations
    for i in range(len(adjectives)):
        for j in range(len(methods)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 3.5 or val > 5.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_yticks(range(len(adjectives)))
    ax.set_yticklabels(adjectives)
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Adjective', fontweight='bold')
    ax.set_title('Average Score by Adjective and Method', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def plot_adjective_rank_variance(ranks_by_adjective, output_path=None):
    """Plot 12: Rank variance by adjective for each method."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    adjectives = sorted(ranks_by_adjective.keys())
    methods = METHOD_ORDER
    
    x = np.arange(len(adjectives))
    width = 0.18
    
    for i, method in enumerate(methods):
        variances = []
        for adj in adjectives:
            ranks = ranks_by_adjective[adj].get(method, [])
            variances.append(np.var(ranks) if ranks else 0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, variances, width, label=method, 
                     color=METHOD_COLORS[method], edgecolor='white', linewidth=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Adjective', fontweight='bold', fontsize=12)
    ax.set_ylabel('Rank Variance (lower = more consistent)', fontweight='bold', fontsize=12)
    ax.set_title('Rank Variance by Adjective', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(adjectives, fontsize=11)
    ax.legend(title='Method', loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add note
    ax.text(0.02, 0.98, '↓ Lower is better (more consistent)', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', style='italic', color='gray')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    plt.show()
    return fig

def main():
    # Configuration
    SESSION_LOGS_DIR = "/home/nancy/Semantic-Knobs/eval/session_logs"
    OUTPUT_DIR = Path(SESSION_LOGS_DIR) / "aggregate_analysis"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("Loading all sessions...")
    all_sessions = load_all_sessions(SESSION_LOGS_DIR)
    print(f"Loaded {len(all_sessions)} sessions")
    
    # Print adjective distribution
    adjective_counts = defaultdict(int)
    for session in all_sessions:
        adjective_counts[session.get('adjective', 'Unknown')] += 1
    print(f"Adjectives found: {dict(adjective_counts)}")
    
    print("\nAggregating data...")
    agg = aggregate_data(all_sessions)
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_aggregate_rank_distribution(agg['rank_counts'], 
                                     OUTPUT_DIR / "1_rank_distribution.png")
    
    plot_score_boxplot(agg['scores_by_method'], 
                      OUTPUT_DIR / "2_score_boxplot.png")
    
    plot_first_place_by_location(agg['first_place_by_location'],
                                 OUTPUT_DIR / "3_first_place_by_location.png")
    
    plot_average_rank_comparison(agg['rank_counts'],
                                OUTPUT_DIR / "4_average_rank.png")
    
    plot_participant_heatmap(agg['scores_by_participant'],
                            OUTPUT_DIR / "5_participant_heatmap.png")
    
    plot_win_rate_pie(agg['rank_counts'],
                     OUTPUT_DIR / "6_win_rate_pie.png")
    
    plot_rank_variance(agg['rank_counts'], agg['scores_by_method'],
                      OUTPUT_DIR / "7_rank_variance.png")
    
    plot_avg_rank_by_participant(agg['ranks_by_participant'],
                                OUTPUT_DIR / "8_avg_rank_by_participant.png")
    
    # New adjective-based visualizations
    plot_avg_rank_by_adjective(agg['ranks_by_adjective'],
                              OUTPUT_DIR / "9_avg_rank_by_adjective.png")
    
    plot_first_place_by_adjective(agg['first_place_by_adjective'],
                                  OUTPUT_DIR / "10_first_place_by_adjective.png")
    
    plot_adjective_heatmap(agg['scores_by_adjective'],
                          OUTPUT_DIR / "11_adjective_heatmap.png")
    
    plot_adjective_rank_variance(agg['ranks_by_adjective'],
                                OUTPUT_DIR / "12_adjective_rank_variance.png")
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Collect rank variance data for comparison table
    rank_stats = {}
    
    for method in METHOD_ORDER:
        if method in agg['scores_by_method']:
            scores = agg['scores_by_method'][method]
            ranks = agg['rank_counts'][method]
            total_ranks = sum(ranks.values())
            avg_rank = sum(r * c for r, c in ranks.items()) / total_ranks
            
            # Calculate rank variance: expand ranks to individual data points
            rank_list = []
            for r, count in ranks.items():
                rank_list.extend([r] * count)
            rank_variance = np.var(rank_list)
            rank_std = np.std(rank_list)
            
            rank_stats[method] = {
                'avg_rank': avg_rank,
                'rank_std': rank_std,
                'rank_var': rank_variance,
                'avg_score': np.mean(scores),
                'score_std': np.std(scores),
                'first_place': ranks.get(1, 0),
                'first_pct': ranks.get(1, 0) / total_ranks * 100,
                'total': total_ranks
            }
            
            print(f"\n{method}:")
            print(f"  Avg Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
            print(f"  Avg Rank:  {avg_rank:.2f} ± {rank_std:.2f} (var={rank_variance:.2f})")
            print(f"  1st Place: {ranks.get(1, 0)} ({ranks.get(1,0)/total_ranks*100:.1f}%)")
    
    # Print rank variance comparison table
    print("\n" + "="*60)
    print("RANK VARIANCE COMPARISON (lower = more consistent)")
    print("="*60)
    sorted_by_var = sorted(rank_stats.items(), key=lambda x: x[1]['rank_var'])
    print(f"\n{'Method':<20} {'Avg Rank':<12} {'Rank Std':<12} {'Rank Var':<12} {'Score Std':<12}")
    print("-" * 68)
    for method, stats in sorted_by_var:
        print(f"{method:<20} {stats['avg_rank']:<12.2f} {stats['rank_std']:<12.2f} {stats['rank_var']:<12.2f} {stats['score_std']:<12.2f}")
    
    # Print adjective-based statistics
    print("\n" + "="*60)
    print("STATISTICS BY ADJECTIVE (P01-P06)")
    print("="*60)
    
    for adjective in sorted(agg['ranks_by_adjective'].keys()):
        print(f"\n{'='*40}")
        print(f"ADJECTIVE: {adjective}")
        print(f"{'='*40}")
        
        # Calculate stats for each method under this adjective
        adj_stats = []
        for method in METHOD_ORDER:
            ranks = agg['ranks_by_adjective'][adjective].get(method, [])
            scores = agg['scores_by_adjective'][adjective].get(method, [])
            first_place = agg['first_place_by_adjective'][adjective].get(method, 0)
            
            if ranks:
                avg_rank = np.mean(ranks)
                rank_std = np.std(ranks)
                rank_var = np.var(ranks)
                total = len(ranks)
                first_pct = first_place / total * 100 if total > 0 else 0
                avg_score = np.mean(scores) if scores else 0
                score_std = np.std(scores) if scores else 0
                
                adj_stats.append({
                    'method': method,
                    'avg_rank': avg_rank,
                    'rank_std': rank_std,
                    'rank_var': rank_var,
                    'avg_score': avg_score,
                    'score_std': score_std,
                    'first_place': first_place,
                    'first_pct': first_pct,
                    'total': total
                })
        
        # Print table for this adjective
        print(f"\n{'Method':<18} {'Avg Rank':<10} {'Rank Var':<10} {'Avg Score':<10} {'1st Place':<12}")
        print("-" * 60)
        for s in sorted(adj_stats, key=lambda x: x['avg_rank']):
            print(f"{s['method']:<18} {s['avg_rank']:<10.2f} {s['rank_var']:<10.2f} {s['avg_score']:<10.2f} {s['first_place']} ({s['first_pct']:.1f}%)")
    
    # Summary comparison across adjectives
    print("\n" + "="*60)
    print("CROSS-ADJECTIVE COMPARISON")
    print("="*60)
    print("\nBest performing method (lowest avg rank) by adjective:")
    for adjective in sorted(agg['ranks_by_adjective'].keys()):
        best_method = None
        best_avg = float('inf')
        for method in METHOD_ORDER:
            ranks = agg['ranks_by_adjective'][adjective].get(method, [])
            if ranks:
                avg = np.mean(ranks)
                if avg < best_avg:
                    best_avg = avg
                    best_method = method
        if best_method:
            print(f"  {adjective:<15}: {best_method:<18} (avg rank: {best_avg:.2f})")

if __name__ == "__main__":
    main()