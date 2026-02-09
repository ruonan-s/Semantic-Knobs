"""
Session V2 Analyzer - Retroactive analysis of tag-level GP sessions

Analyzes hitl_state_v2.json and generates detailed logs.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def analyze_session_v2(session_folder: Path) -> Dict[str, Any]:
    """Analyze a V2 HITL session."""
    session_folder = Path(session_folder)
    
    # Load V2 state
    state_file = session_folder / "hitl_state_v2.json"
    if not state_file.exists():
        raise FileNotFoundError(f"hitl_state_v2.json not found in {session_folder}")
    
    with open(state_file) as f:
        state = json.load(f)
    
    # Load tag preferences
    tag_prefs = None
    tag_prefs_path = session_folder / "impression" / "tag_preferences.json"
    if tag_prefs_path.exists():
        with open(tag_prefs_path) as f:
            tag_prefs = json.load(f)
    
    analysis = {
        "metadata": {
            "session_id": state.get("session_id", "unknown"),
            "version": state.get("version", "2.0"),
            "analysis_timestamp": datetime.now().isoformat(),
            "rounds_completed": state.get("round_count", 0),
            "is_converged": state.get("is_converged", False),
        },
        "tag_preferences": tag_prefs,
        "rounds": [],
        "tag_statistics": {},
        "tag_trajectories": {},
        "warnings": [],
    }
    
    # Track tag statistics
    tag_stats = defaultdict(lambda: {
        "times_shown": 0,
        "times_in_winner": 0,
        "times_in_loser": 0,
        "strategies_used": defaultdict(int),
        "mu_trajectory": [],
        "sigma_trajectory": [],
        "weight_trajectory": [],
    })
    
    rankings = state.get("rankings_history", [])
    compositions = state.get("compositions_history", [])
    tag_states = state.get("tag_states", {})
    
    # Analyze each round
    for round_idx, (ranking, round_comps) in enumerate(zip(rankings, compositions)):
        round_num = round_idx + 1
        
        winner_idx = ranking[0]
        loser_idx = ranking[-1]
        
        round_data = {
            "round_num": round_num,
            "ranking": ranking,
            "options": [],
            "pairwise_comparisons": [],
            "diagnostics": {},
        }
        
        # Analyze each option
        all_mus_this_round = []
        all_sigmas_this_round = []
        
        for comp in round_comps:
            opt_id = comp["option_id"]
            is_winner = opt_id == winner_idx
            is_loser = opt_id == loser_idx
            
            tags = comp.get("tag_labels", [])
            weights = comp.get("weights", [])
            mus = comp.get("mus", [])
            sigmas = comp.get("sigmas", [])
            strategy = comp.get("strategy", "unknown")
            
            all_mus_this_round.extend(mus)
            all_sigmas_this_round.extend(sigmas)
            
            option_data = {
                "option_id": opt_id,
                "strategy": strategy,
                "is_winner": is_winner,
                "is_loser": is_loser,
                "position": ranking.index(opt_id) + 1,
                "tags": [
                    {
                        "text": tag,
                        "weight": round(w, 4),
                        "mu": round(m, 4),
                        "sigma": round(s, 4),
                    }
                    for tag, w, m, s in zip(tags, weights, mus, sigmas)
                ],
                "avg_mu": round(np.mean(mus), 4) if mus else 0,
                "avg_sigma": round(np.mean(sigmas), 4) if sigmas else 0,
                "avg_weight": round(np.mean(weights), 4) if weights else 0,
            }
            round_data["options"].append(option_data)
            
            # Update tag statistics
            for tag, weight, mu, sigma in zip(tags, weights, mus, sigmas):
                stats = tag_stats[tag]
                stats["times_shown"] += 1
                stats["strategies_used"][strategy] += 1
                stats["mu_trajectory"].append((round_num, mu))
                stats["sigma_trajectory"].append((round_num, sigma))
                stats["weight_trajectory"].append((round_num, weight))
                
                if is_winner:
                    stats["times_in_winner"] += 1
                if is_loser:
                    stats["times_in_loser"] += 1
        
        # Generate pairwise comparisons
        for i, better_id in enumerate(ranking):
            for worse_id in ranking[i+1:]:
                better_comp = round_comps[better_id]
                worse_comp = round_comps[worse_id]
                
                better_tags = set(better_comp.get("tag_labels", []))
                worse_tags = set(worse_comp.get("tag_labels", []))
                
                comparison = {
                    "better_option": better_id,
                    "worse_option": worse_id,
                    "rank_distance": ranking.index(worse_id) - ranking.index(better_id),
                    "tags_only_in_better": list(better_tags - worse_tags),
                    "tags_only_in_worse": list(worse_tags - better_tags),
                    "shared_tags": list(better_tags & worse_tags),
                    "num_differentiating": len(better_tags - worse_tags) + len(worse_tags - better_tags),
                }
                round_data["pairwise_comparisons"].append(comparison)
                
                # Check for low-information comparisons
                if comparison["num_differentiating"] < 2:
                    analysis["warnings"].append({
                        "type": "low_info_comparison",
                        "round": round_num,
                        "message": f"Round {round_num}: Comparison {better_id} > {worse_id} has only {comparison['num_differentiating']} differentiating tags"
                    })
        
        # Round diagnostics
        round_data["diagnostics"] = {
            "mu_distribution": {
                "min": round(min(all_mus_this_round), 4) if all_mus_this_round else 0,
                "max": round(max(all_mus_this_round), 4) if all_mus_this_round else 0,
                "mean": round(np.mean(all_mus_this_round), 4) if all_mus_this_round else 0,
                "std": round(np.std(all_mus_this_round), 4) if all_mus_this_round else 0,
            },
            "sigma_distribution": {
                "min": round(min(all_sigmas_this_round), 4) if all_sigmas_this_round else 0,
                "max": round(max(all_sigmas_this_round), 4) if all_sigmas_this_round else 0,
                "mean": round(np.mean(all_sigmas_this_round), 4) if all_sigmas_this_round else 0,
            },
            "winning_strategy": round_comps[winner_idx].get("strategy", "unknown"),
            "losing_strategy": round_comps[loser_idx].get("strategy", "unknown"),
        }
        
        analysis["rounds"].append(round_data)
    
    # Finalize tag statistics
    for tag, stats in tag_stats.items():
        if stats["times_shown"] > 0:
            stats["win_rate"] = round(stats["times_in_winner"] / stats["times_shown"], 3)
            stats["lose_rate"] = round(stats["times_in_loser"] / stats["times_shown"], 3)
            
            # Get final mu/sigma from tag_states if available
            for tag_id, tag_data in tag_states.items():
                if tag_data.get("text") == tag:
                    stats["final_mu"] = tag_data.get("mu", 0)
                    stats["final_sigma"] = tag_data.get("sigma", 0)
                    stats["category"] = tag_data.get("category", "unknown")
                    break
            
            # Convert defaultdict
            stats["strategies_used"] = dict(stats["strategies_used"])
    
    # Sort tags by final mu (or win rate if no final state)
    sorted_tags = sorted(
        tag_stats.items(),
        key=lambda x: (x[1].get("final_mu", 0), x[1].get("win_rate", 0)),
        reverse=True
    )
    
    analysis["tag_statistics"] = {
        "total_unique_tags": len(tag_stats),
        "tags_by_utility": [
            {"tag": tag, **{k: v for k, v in stats.items() if k not in ["mu_trajectory", "sigma_trajectory", "weight_trajectory"]}}
            for tag, stats in sorted_tags
        ],
        "top_10_by_final_mu": [
            {
                "tag": tag,
                "final_mu": stats.get("final_mu", 0),
                "final_sigma": stats.get("final_sigma", 0),
                "win_rate": stats.get("win_rate", 0),
                "times_shown": stats["times_shown"],
            }
            for tag, stats in sorted_tags[:10]
        ],
    }
    
    # Tag trajectories
    analysis["tag_trajectories"] = {
        tag: {
            "mu": stats["mu_trajectory"],
            "sigma": stats["sigma_trajectory"],
        }
        for tag, stats in sorted_tags[:20]  # Top 20 tags
    }
    
    return analysis


def generate_summary(analysis: Dict) -> str:
    """Generate human-readable summary."""
    lines = []
    
    meta = analysis["metadata"]
    lines.append("=" * 70)
    lines.append("V2 SESSION ANALYSIS (Tag-Level GP)")
    lines.append("=" * 70)
    lines.append(f"Session: {meta['session_id']}")
    lines.append(f"Rounds: {meta['rounds_completed']}")
    lines.append(f"Converged: {meta['is_converged']}")
    lines.append(f"Analysis Time: {meta['analysis_timestamp']}")
    lines.append("")
    
    # Tag preferences
    if analysis.get("tag_preferences"):
        prefs = analysis["tag_preferences"]
        lines.append("-" * 70)
        lines.append("EXPLORATION PREFERENCES")
        lines.append("-" * 70)
        lines.append(f"  Positive: {len(prefs.get('positive', []))} tags")
        lines.append(f"  Neutral: {len(prefs.get('neutral', []))} tags")
        lines.append(f"  Negative: {len(prefs.get('negative', []))} tags")
        lines.append("")
    
    # Round summaries
    for round_data in analysis["rounds"]:
        lines.append("-" * 70)
        lines.append(f"ROUND {round_data['round_num']}")
        lines.append("-" * 70)
        
        lines.append("  Options:")
        for opt in round_data["options"]:
            marker = "🏆" if opt["is_winner"] else ("💀" if opt["is_loser"] else "  ")
            top_tags = [t["text"] for t in opt["tags"][:3]]
            lines.append(f"    {marker}[{opt['option_id']}] {opt['strategy']:10s} | "
                        f"pos={opt['position']} | avg_μ={opt['avg_mu']:+.2f} | "
                        f"{', '.join(top_tags)}...")
        
        lines.append(f"  Ranking: {round_data['ranking']}")
        lines.append(f"  Comparisons: {len(round_data['pairwise_comparisons'])}")
        
        diag = round_data["diagnostics"]
        lines.append(f"  μ range: [{diag['mu_distribution']['min']:.2f}, {diag['mu_distribution']['max']:.2f}]")
        lines.append(f"  Winner strategy: {diag['winning_strategy']}")
        lines.append("")
    
    # Tag statistics
    lines.append("=" * 70)
    lines.append("TAG STATISTICS (by final μ)")
    lines.append("=" * 70)
    
    tag_stats = analysis["tag_statistics"]
    lines.append(f"Total unique tags: {tag_stats['total_unique_tags']}")
    lines.append("")
    lines.append("Top 15 by Final Utility (μ):")
    
    for i, tag_data in enumerate(tag_stats["tags_by_utility"][:15]):
        final_mu = tag_data.get("final_mu", "N/A")
        final_sigma = tag_data.get("final_sigma", "N/A")
        win_rate = tag_data.get("win_rate", 0)
        shown = tag_data.get("times_shown", 0)
        category = tag_data.get("category", "?")[:3]
        
        if isinstance(final_mu, float):
            lines.append(f"  {i+1:2d}. {tag_data['tag'][:30]:30s} | "
                        f"μ={final_mu:+.3f} σ={final_sigma:.3f} | "
                        f"win={win_rate:.0%} | shown={shown} [{category}]")
        else:
            lines.append(f"  {i+1:2d}. {tag_data['tag'][:30]:30s} | "
                        f"win={win_rate:.0%} | shown={shown}")
    
    # Strategy analysis
    lines.append("")
    lines.append("=" * 70)
    lines.append("STRATEGY ANALYSIS")
    lines.append("=" * 70)
    
    strategy_wins = defaultdict(int)
    strategy_losses = defaultdict(int)
    strategy_appearances = defaultdict(int)
    
    for round_data in analysis["rounds"]:
        for opt in round_data["options"]:
            strategy = opt["strategy"]
            strategy_appearances[strategy] += 1
            if opt["is_winner"]:
                strategy_wins[strategy] += 1
            if opt["is_loser"]:
                strategy_losses[strategy] += 1
    
    for strategy in ["exploit", "explore", "ucb", "challenger"]:
        wins = strategy_wins.get(strategy, 0)
        losses = strategy_losses.get(strategy, 0)
        total = strategy_appearances.get(strategy, 0)
        if total > 0:
            lines.append(f"  {strategy:12s}: {wins} wins, {losses} losses out of {total} appearances "
                        f"({wins/total:.0%} win rate)")
    
    # Warnings
    if analysis.get("warnings"):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"WARNINGS ({len(analysis['warnings'])})")
        lines.append("=" * 70)
        for w in analysis["warnings"][:15]:
            lines.append(f"  [{w['type']}] {w['message']}")
        if len(analysis["warnings"]) > 15:
            lines.append(f"  ... and {len(analysis['warnings']) - 15} more")
    
    return "\n".join(lines)


def save_analysis(analysis: Dict, output_folder: Path) -> Tuple[str, str]:
    """Save analysis to files."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_folder / "session_analysis_v2.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Save summary
    summary_path = output_folder / "session_analysis_v2_summary.txt"
    summary = generate_summary(analysis)
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return str(json_path), str(summary_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze V2 HITL session")
    parser.add_argument("session_folder", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.session_folder / "analysis"
    
    print(f"Analyzing V2 session: {args.session_folder}")
    analysis = analyze_session_v2(args.session_folder)
    
    json_path, summary_path = save_analysis(analysis, args.output)
    
    print(f"Analysis saved:")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    
    # Print summary
    print("\n" + generate_summary(analysis))


if __name__ == "__main__":
    main()
