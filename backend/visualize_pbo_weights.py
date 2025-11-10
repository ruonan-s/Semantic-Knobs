#!/usr/bin/env python3
"""
Interactive PBO Weight Evolution Visualization

Shows how concept weights evolve across refinement rounds:
- Dashed lines: Average weights across all 4 proposals per round (top 10 concepts)
- Solid lines: Weights of selected images per round (top 10 concepts)
- Both share the same color per concept for clarity

Usage:
    python visualize_pbo_weights.py <session_path>

Example:
    python visualize_pbo_weights.py sessions/[fast]_A_refreshing_space_by_the_sea_2025-11-07_02-33-39/impression_refinement/
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    USE_PLOTLY = True
except ImportError:
    import matplotlib.pyplot as plt
    USE_PLOTLY = False
    print("⚠️  Plotly not found. Install with: pip install plotly")
    print("   Falling back to matplotlib (less interactive)")


def load_round_weights(round_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """Load weight proposals and concept labels from a round directory."""
    weights_file = round_dir / "weights.json"
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights.json found in {round_dir}")

    with open(weights_file, 'r') as f:
        data = json.load(f)

    proposals = [np.array(p) for p in data['proposals']]
    labels = data['concept_labels']
    return proposals, labels


def find_all_rounds(session_path: Path) -> List[int]:
    """Find all round directories in the session path."""
    rounds = []
    for item in session_path.iterdir():
        if item.is_dir() and item.name.startswith('round_'):
            try:
                round_num = int(item.name.split('_')[1])
                rounds.append(round_num)
            except (IndexError, ValueError):
                continue
    return sorted(rounds)


def get_selections_interactive(num_rounds: int) -> List[int]:
    """Interactively ask user which images were selected each round."""
    print("\n" + "="*80)
    print("SELECTION INPUT")
    print("="*80)
    print("For each round, enter which image you selected (0-3)")
    print("Or press Enter to use defaults from logs (if available)\n")

    default_selections = [0, 0, 0, 3, 2, 3, 2, 2, 3, 2, 3, 0]
    selections = []
    for i in range(1, num_rounds + 1):
        default = default_selections[i-1] if i-1 < len(default_selections) else 0
        response = input(f"Round {i}: Which image did you select? [0-3, default={default}]: ").strip()
        if response == '':
            selections.append(default)
        else:
            try:
                idx = int(response)
                if 0 <= idx <= 3:
                    selections.append(idx)
                else:
                    print(f"  Invalid input, using default: {default}")
                    selections.append(default)
            except ValueError:
                print(f"  Invalid input, using default: {default}")
                selections.append(default)
    return selections


def calculate_top_concepts(all_round_avgs: List[np.ndarray], top_k: int = 10) -> List[int]:
    """Identify top K concepts based on average weight across all rounds."""
    stacked = np.stack(all_round_avgs, axis=0)
    overall_avg = stacked.mean(axis=0)
    top_indices = np.argsort(-overall_avg)[:top_k]
    return top_indices.tolist()


# ------------------ Plotly Visualization ------------------
def create_plotly_visualization(
    rounds: List[int],
    concept_labels: List[str],
    avg_weights: List[np.ndarray],
    selected_weights: List[np.ndarray],
    top_concepts: List[int]
):
    """Create interactive Plotly visualization."""
    from plotly.colors import qualitative
    palette = qualitative.Plotly  # 10-color palette

    fig = go.Figure()
    num_rounds = len(rounds)
    x_vals = list(range(1, num_rounds + 1))

    # Average (dashed)
    for i, concept_idx in enumerate(top_concepts):
        concept_label = concept_labels[concept_idx]
        color = palette[i % len(palette)]
        y_vals_avg = [avg_weights[r][concept_idx] for r in range(num_rounds)]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals_avg,
            mode='lines+markers',
            name=f'{concept_label} (avg)',
            line=dict(color=color, dash='dash', width=2),
            marker=dict(size=6, color=color),
            legendgroup=f'concept_{concept_idx}',
            showlegend=True
        ))

    # Selected (solid)
    for i, concept_idx in enumerate(top_concepts):
        concept_label = concept_labels[concept_idx]
        color = palette[i % len(palette)]
        y_vals_sel = [selected_weights[r][concept_idx] for r in range(num_rounds)]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals_sel,
            mode='lines+markers',
            name=f'{concept_label} (selected)',
            line=dict(color=color, width=3),
            marker=dict(size=8, symbol='star', color=color),
            legendgroup=f'concept_{concept_idx}',
            showlegend=False  # keep legend compact
        ))

    fig.update_layout(
        title=dict(text='PBO Concept Weight Evolution Across Rounds', font=dict(size=20)),
        xaxis=dict(title='Round', tickmode='linear', tick0=1, dtick=1, gridcolor='lightgray'),
        yaxis=dict(title='Concept Weight', gridcolor='lightgray'),
        hovermode='x unified',
        template='plotly_white',
        width=1400,
        height=800,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, font=dict(size=10))
    )
    return fig


# ------------------ Matplotlib Visualization ------------------
def create_matplotlib_visualization(
    rounds: List[int],
    concept_labels: List[str],
    avg_weights: List[np.ndarray],
    selected_weights: List[np.ndarray],
    top_concepts: List[int]
):
    """Create matplotlib visualization (fallback)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 8))
    num_rounds = len(rounds)
    x_vals = list(range(1, num_rounds + 1))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(top_concepts))]

    for i, concept_idx in enumerate(top_concepts):
        concept_label = concept_labels[concept_idx]
        color = colors[i]
        y_vals_avg = [avg_weights[r][concept_idx] for r in range(num_rounds)]
        y_vals_sel = [selected_weights[r][concept_idx] for r in range(num_rounds)]

        # Average (dashed)
        ax.plot(x_vals, y_vals_avg, '--o', color=color, label=f'{concept_label} (avg)', alpha=0.9)
        # Selected (solid)
        ax.plot(x_vals, y_vals_sel, '-*', color=color, linewidth=2, markersize=10, label='_nolegend_')

    ax.set_xlabel('Round', fontsize=14)
    ax.set_ylabel('Concept Weight', fontsize=14)
    ax.set_title('PBO Concept Weight Evolution Across Rounds', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    return fig


# ------------------ Main Execution ------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_path = Path(sys.argv[1])
    if not session_path.exists():
        print(f"❌ Error: Path not found: {session_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("PBO WEIGHT EVOLUTION VISUALIZATION")
    print("="*80)
    print(f"Session: {session_path}")

    rounds = find_all_rounds(session_path)
    if not rounds:
        print(f"❌ Error: No round directories found in {session_path}")
        sys.exit(1)
    print(f"Found {len(rounds)} rounds: {rounds}")

    all_proposals, all_labels = [], None
    for round_num in rounds:
        round_dir = session_path / f"round_{round_num}"
        proposals, labels = load_round_weights(round_dir)
        all_proposals.append(proposals)
        if all_labels is None:
            all_labels = labels
    print(f"Loaded {len(all_proposals)} rounds with {len(all_labels)} concepts each")

    selections = get_selections_interactive(len(rounds))
    print(f"\nSelections: {selections}")

    avg_weights = [np.mean(np.stack(p, axis=0), axis=0) for p in all_proposals]
    selected_weights = [p[selections[i]] for i, p in enumerate(all_proposals)]

    top_concepts = calculate_top_concepts(avg_weights, top_k=10)
    print(f"\nTop 10 concepts by average weight:")
    for i, idx in enumerate(top_concepts):
        mean_val = np.mean([a[idx] for a in avg_weights])
        print(f"  {i+1}. {all_labels[idx]}: {mean_val:.4f}")

    print(f"\n📊 Creating visualization...")
    if USE_PLOTLY:
        fig = create_plotly_visualization(rounds, all_labels, avg_weights, selected_weights, top_concepts)
        output_file = session_path / "pbo_weight_evolution.html"
        fig.write_html(str(output_file))
        print(f"✅ Saved interactive visualization to: {output_file}")
        try:
            import webbrowser
            webbrowser.open(f"file://{output_file.absolute()}")
            print(f"🌐 Opening in browser...")
        except Exception:
            print(f"   Open manually: {output_file}")
    else:
        fig = create_matplotlib_visualization(rounds, all_labels, avg_weights, selected_weights, top_concepts)
        output_file = session_path / "pbo_weight_evolution.png"
        fig.savefig(str(output_file), dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_file}")
        plt.show()

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
