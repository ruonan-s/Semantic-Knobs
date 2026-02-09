"""
Retroactive Session Log Analyzer

Analyzes existing HITL session data and generates a detailed log 
similar to the GP refinement logger output.

Works with sessions that used the composition-based GP system.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


def analyze_session(session_folder: Path) -> Dict[str, Any]:
    """
    Analyze a completed HITL session and generate detailed logs.
    
    Args:
        session_folder: Path to session folder
        
    Returns:
        Analysis results dictionary
    """
    session_folder = Path(session_folder)
    
    # Load session data
    hitl_state_path = session_folder / "hitl_state.json"
    eval_log_path = session_folder / "eval_log.json"
    tag_prefs_path = session_folder / "impression" / "tag_preferences.json"
    final_selection_path = session_folder / "final_selection.json"
    
    if not hitl_state_path.exists():
        raise FileNotFoundError(f"hitl_state.json not found in {session_folder}")
    
    with open(hitl_state_path) as f:
        hitl_state = json.load(f)
    
    eval_log = None
    if eval_log_path.exists():
        with open(eval_log_path) as f:
            eval_log = json.load(f)
    
    tag_prefs = None
    if tag_prefs_path.exists():
        with open(tag_prefs_path) as f:
            tag_prefs = json.load(f)
    
    final_selection = None
    if final_selection_path.exists():
        with open(final_selection_path) as f:
            final_selection = json.load(f)
    
    # Build analysis
    analysis = {
        "metadata": {
            "session_id": hitl_state.get("session_id", "unknown"),
            "analysis_timestamp": datetime.now().isoformat(),
            "rounds_completed": hitl_state.get("round_count", 0),
            "is_converged": hitl_state.get("is_converged", False),
            "base_prompt": hitl_state.get("base_prompt", ""),
        },
        "tag_preferences": tag_prefs,
        "rounds": [],
        "tag_statistics": {},
        "final_selection": final_selection,
        "warnings": [],
    }
    
    # Analyze each round
    rankings = hitl_state.get("rankings_history", [])
    compositions = hitl_state.get("compositions_history", [])
    
    # Track tag statistics
    tag_stats = defaultdict(lambda: {
        "times_shown": 0,
        "times_in_winner": 0,
        "times_in_loser": 0,
        "total_weight": 0.0,
        "weight_when_winning": 0.0,
        "weight_when_losing": 0.0,
        "ucb_scores": [],
        "positions": [],  # 1st, 2nd, 3rd, 4th place counts
    })
    
    for round_idx, (ranking, round_compositions) in enumerate(zip(rankings, compositions)):
        round_num = round_idx + 1
        
        # Get timing from eval_log if available
        round_timestamp = None
        gp_variance = None
        if eval_log:
            for event in eval_log.get("events", []):
                if event.get("type") == "hitl_ranking" and event.get("data", {}).get("round") == round_num:
                    round_timestamp = event.get("timestamp")
                    gp_variance = event.get("data", {}).get("gp_variance")
                    break
        
        round_data = {
            "round_num": round_num,
            "timestamp": round_timestamp,
            "ranking": ranking,
            "gp_variance": gp_variance,
            "options": [],
            "pairwise_comparisons": [],
        }
        
        # Analyze each option
        winning_option_idx = ranking[0]
        losing_option_idx = ranking[-1]
        
        for opt_idx, composition in enumerate(round_compositions):
            tags = composition.get("tag_labels", [])
            weights = composition.get("weights", [])
            ucb_scores = composition.get("ucb_scores", [])
            
            position = ranking.index(opt_idx) + 1  # 1-indexed position
            
            option_data = {
                "option_id": opt_idx,
                "position": position,
                "is_winner": opt_idx == winning_option_idx,
                "is_loser": opt_idx == losing_option_idx,
                "tags": [
                    {
                        "text": tag,
                        "weight": round(w, 4),
                        "ucb_score": round(ucb, 4) if ucb_scores else None,
                    }
                    for tag, w, ucb in zip(
                        tags, 
                        weights, 
                        ucb_scores if ucb_scores else [None] * len(tags)
                    )
                ],
                "avg_weight": round(np.mean(weights), 4) if weights else 0,
                "avg_ucb": round(np.mean(ucb_scores), 4) if ucb_scores else None,
            }
            round_data["options"].append(option_data)
            
            # Update tag statistics
            for tag, weight, ucb in zip(tags, weights, ucb_scores if ucb_scores else [None] * len(tags)):
                stats = tag_stats[tag]
                stats["times_shown"] += 1
                stats["total_weight"] += weight
                stats["positions"].append(position)
                if ucb is not None:
                    stats["ucb_scores"].append(ucb)
                
                if opt_idx == winning_option_idx:
                    stats["times_in_winner"] += 1
                    stats["weight_when_winning"] += weight
                if opt_idx == losing_option_idx:
                    stats["times_in_loser"] += 1
                    stats["weight_when_losing"] += weight
        
        # Generate pairwise comparisons
        for i, better_id in enumerate(ranking):
            for worse_id in ranking[i+1:]:
                better_tags = set(round_compositions[better_id].get("tag_labels", []))
                worse_tags = set(round_compositions[worse_id].get("tag_labels", []))
                
                comparison = {
                    "better_option": better_id,
                    "worse_option": worse_id,
                    "rank_distance": ranking.index(worse_id) - ranking.index(better_id),
                    "tags_only_in_better": list(better_tags - worse_tags),
                    "tags_only_in_worse": list(worse_tags - better_tags),
                    "shared_tags": list(better_tags & worse_tags),
                }
                round_data["pairwise_comparisons"].append(comparison)
                
                # Check for low-information comparisons
                if len(better_tags - worse_tags) + len(worse_tags - better_tags) < 2:
                    analysis["warnings"].append({
                        "type": "low_info_comparison",
                        "round": round_num,
                        "message": f"Round {round_num}: Comparison {better_id} > {worse_id} has few differentiating tags"
                    })
        
        analysis["rounds"].append(round_data)
    
    # Finalize tag statistics
    for tag, stats in tag_stats.items():
        if stats["times_shown"] > 0:
            stats["win_rate"] = round(stats["times_in_winner"] / stats["times_shown"], 3)
            stats["lose_rate"] = round(stats["times_in_loser"] / stats["times_shown"], 3)
            stats["avg_weight"] = round(stats["total_weight"] / stats["times_shown"], 4)
            stats["avg_position"] = round(np.mean(stats["positions"]), 2)
            
            if stats["times_in_winner"] > 0:
                stats["avg_weight_when_winning"] = round(
                    stats["weight_when_winning"] / stats["times_in_winner"], 4
                )
            else:
                stats["avg_weight_when_winning"] = None
                
            if stats["ucb_scores"]:
                stats["final_ucb"] = round(stats["ucb_scores"][-1], 4)
                stats["ucb_trend"] = round(
                    stats["ucb_scores"][-1] - stats["ucb_scores"][0], 4
                ) if len(stats["ucb_scores"]) > 1 else 0
            
            # Clean up internal tracking
            del stats["total_weight"]
            del stats["weight_when_winning"]
            del stats["weight_when_losing"]
            del stats["positions"]
            del stats["ucb_scores"]
    
    # Sort tags by win rate
    sorted_tags = sorted(
        tag_stats.items(),
        key=lambda x: (x[1].get("win_rate", 0), -x[1].get("lose_rate", 1)),
        reverse=True
    )
    
    analysis["tag_statistics"] = {
        "total_unique_tags": len(tag_stats),
        "tags_by_win_rate": [
            {"tag": tag, **stats}
            for tag, stats in sorted_tags
        ],
        "top_10_by_win_rate": [
            {"tag": tag, "win_rate": stats.get("win_rate", 0), "times_shown": stats["times_shown"]}
            for tag, stats in sorted_tags[:10]
        ],
    }
    
    # Check for consistency issues
    _check_consistency(analysis)
    
    return analysis


def _check_consistency(analysis: Dict) -> None:
    """Check for potential issues in the session."""
    
    tag_stats = analysis["tag_statistics"].get("tags_by_win_rate", [])
    
    # Check if final selection matches high win-rate tags
    if analysis.get("final_selection") and "tags" in analysis["final_selection"]:
        final_tags = set(analysis["final_selection"]["tags"])
        top_win_tags = set(t["tag"] for t in tag_stats[:10])
        
        overlap = len(final_tags & top_win_tags)
        if overlap < 5:
            analysis["warnings"].append({
                "type": "final_mismatch",
                "message": f"Only {overlap}/10 final tags match top win-rate tags"
            })
    
    # Check for tags that won often but weren't in final
    for tag_data in tag_stats[:15]:
        if tag_data["win_rate"] > 0.5 and tag_data["times_shown"] >= 3:
            if analysis.get("final_selection"):
                final_tags = analysis["final_selection"].get("tags", [])
                if tag_data["tag"] not in final_tags:
                    analysis["warnings"].append({
                        "type": "high_winrate_excluded",
                        "message": f"Tag '{tag_data['tag']}' has {tag_data['win_rate']:.0%} win rate but not in final"
                    })


def save_analysis(analysis: Dict, output_folder: Path) -> Tuple[str, str]:
    """Save analysis to JSON and text summary files."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_folder / "session_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Save text summary
    summary_path = output_folder / "session_analysis_summary.txt"
    summary = generate_text_summary(analysis)
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return str(json_path), str(summary_path)


def generate_text_summary(analysis: Dict) -> str:
    """Generate a human-readable summary."""
    lines = []
    
    meta = analysis["metadata"]
    lines.append("=" * 70)
    lines.append("SESSION ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Session: {meta['session_id']}")
    lines.append(f"Base Prompt: {meta.get('base_prompt', 'N/A')}")
    lines.append(f"Rounds: {meta['rounds_completed']}")
    lines.append(f"Converged: {meta['is_converged']}")
    lines.append(f"Analysis Time: {meta['analysis_timestamp']}")
    lines.append("")
    
    # Tag preferences from exploration
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
        if round_data.get("gp_variance"):
            lines.append(f"  GP Variance: {round_data['gp_variance']:.4f}")
        lines.append("-" * 70)
        
        lines.append("  Options:")
        for opt in round_data["options"]:
            marker = "🏆" if opt["is_winner"] else ("💀" if opt["is_loser"] else "  ")
            top_tags = [t["text"] for t in opt["tags"][:3]]
            lines.append(f"    {marker}[{opt['option_id']}] pos={opt['position']} | "
                        f"avg_w={opt['avg_weight']:.3f} | {', '.join(top_tags)}...")
        
        lines.append(f"  Ranking: {round_data['ranking']}")
        lines.append(f"  Comparisons: {len(round_data['pairwise_comparisons'])}")
        
        # Show one interesting comparison
        if round_data["pairwise_comparisons"]:
            comp = round_data["pairwise_comparisons"][0]  # Best vs second
            lines.append(f"  Top comparison ({comp['better_option']} > {comp['worse_option']}):")
            if comp["tags_only_in_better"]:
                lines.append(f"    Only in winner: {comp['tags_only_in_better'][:3]}...")
            if comp["tags_only_in_worse"]:
                lines.append(f"    Only in loser: {comp['tags_only_in_worse'][:3]}...")
        lines.append("")
    
    # Tag statistics
    lines.append("=" * 70)
    lines.append("TAG STATISTICS")
    lines.append("=" * 70)
    
    tag_stats = analysis["tag_statistics"]
    lines.append(f"Total unique tags: {tag_stats['total_unique_tags']}")
    lines.append("")
    lines.append("Top 15 by Win Rate:")
    
    for i, tag_data in enumerate(tag_stats["tags_by_win_rate"][:15]):
        win_rate = tag_data.get("win_rate", 0)
        lose_rate = tag_data.get("lose_rate", 0)
        shown = tag_data.get("times_shown", 0)
        avg_pos = tag_data.get("avg_position", 0)
        lines.append(f"  {i+1:2d}. {tag_data['tag'][:30]:30s} | "
                    f"win={win_rate:.0%} lose={lose_rate:.0%} | "
                    f"shown={shown} avg_pos={avg_pos:.1f}")
    
    # Final selection
    if analysis.get("final_selection"):
        lines.append("")
        lines.append("=" * 70)
        lines.append("FINAL SELECTION")
        lines.append("=" * 70)
        final = analysis["final_selection"]
        if "tags" in final:
            for i, tag in enumerate(final["tags"]):
                weight = final.get("weights", {}).get(tag, 0)
                lines.append(f"  {i+1:2d}. {tag[:35]:35s} | weight={weight:.3f}")
    
    # Warnings
    if analysis.get("warnings"):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"WARNINGS ({len(analysis['warnings'])})")
        lines.append("=" * 70)
        for w in analysis["warnings"][:20]:
            lines.append(f"  [{w['type']}] {w['message']}")
        if len(analysis["warnings"]) > 20:
            lines.append(f"  ... and {len(analysis['warnings']) - 20} more")
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze HITL session logs")
    parser.add_argument("session_folder", type=Path, help="Path to session folder")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output folder (default: session_folder/analysis)")
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.session_folder / "analysis"
    
    print(f"Analyzing session: {args.session_folder}")
    analysis = analyze_session(args.session_folder)
    
    json_path, summary_path = save_analysis(analysis, args.output)
    
    print(f"Analysis saved:")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    
    # Print summary to stdout
    print("\n" + "=" * 70)
    print(generate_text_summary(analysis))


if __name__ == "__main__":
    main()
