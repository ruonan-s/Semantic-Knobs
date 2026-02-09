"""
GP Refinement Logger

Comprehensive logging for debugging and validating the GP refinement process.

Tracks:
- Initialization: tags, priors, categories
- Per-round: options, weights, diversity, images
- Per-ranking: comparisons, updates, tag trajectories
- Final: selection, statistics, convergence analysis

Saves structured JSON logs for post-hoc analysis.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class TagSnapshot:
    """Snapshot of a tag's state at a point in time."""
    tag_id: str
    text: str
    category: str
    mu: float
    sigma: float
    times_shown: int = 0
    times_in_winner: int = 0
    times_in_loser: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "tag_id": self.tag_id,
            "text": self.text,
            "category": self.category,
            "mu": round(self.mu, 6),
            "sigma": round(self.sigma, 6),
            "times_shown": self.times_shown,
            "times_in_winner": self.times_in_winner,
            "times_in_loser": self.times_in_loser,
        }


@dataclass 
class OptionLog:
    """Log entry for a single option."""
    option_id: int
    strategy: str
    tags: List[Dict[str, Any]]  # [{text, tag_id, mu, sigma, weight}, ...]
    avg_mu: float
    avg_sigma: float
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "option_id": self.option_id,
            "strategy": self.strategy,
            "tags": self.tags,
            "avg_mu": round(self.avg_mu, 4),
            "avg_sigma": round(self.avg_sigma, 4),
            "image_path": self.image_path,
        }


@dataclass
class DiversityCheck:
    """Log of diversity check between options."""
    option_a: int
    option_b: int
    shared_tags: List[str]
    overlap_count: int
    overlap_fraction: float
    max_allowed: float
    passed: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PairwiseComparison:
    """Log of a single pairwise comparison."""
    better_option_id: int
    worse_option_id: int
    rank_distance: int
    preference_strength: float
    tags_only_in_better: List[str]
    tags_only_in_worse: List[str]
    shared_tags: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "better_option_id": self.better_option_id,
            "worse_option_id": self.worse_option_id,
            "rank_distance": self.rank_distance,
            "preference_strength": round(self.preference_strength, 3),
            "tags_only_in_better": self.tags_only_in_better,
            "tags_only_in_worse": self.tags_only_in_worse,
            "shared_tags": self.shared_tags,
            "num_differentiating": len(self.tags_only_in_better) + len(self.tags_only_in_worse),
        }


@dataclass
class TagUpdate:
    """Log of an update to a tag."""
    tag_id: str
    text: str
    mu_before: float
    mu_after: float
    mu_delta: float
    sigma_before: float
    sigma_after: float
    update_reason: str  # "in_better" or "in_worse"
    from_comparison: str  # e.g., "2 > 1"
    
    def to_dict(self) -> Dict:
        return {
            "tag_id": self.tag_id,
            "text": self.text,
            "mu_before": round(self.mu_before, 4),
            "mu_after": round(self.mu_after, 4),
            "mu_delta": round(self.mu_delta, 4),
            "sigma_before": round(self.sigma_before, 4),
            "sigma_after": round(self.sigma_after, 4),
            "update_reason": self.update_reason,
            "from_comparison": self.from_comparison,
        }


@dataclass
class RoundLog:
    """Complete log for a single round."""
    round_num: int
    timestamp: str
    beta: float
    max_overlap: float
    learning_rate: float
    
    # Options generated
    options: List[OptionLog] = field(default_factory=list)
    
    # Diversity checks
    diversity_checks: List[DiversityCheck] = field(default_factory=list)
    
    # User ranking
    ranking: Optional[List[int]] = None
    
    # Pairwise comparisons
    comparisons: List[PairwiseComparison] = field(default_factory=list)
    
    # Tag updates
    tag_updates: List[TagUpdate] = field(default_factory=list)
    
    # Top tags after round
    top_10_after: List[Dict] = field(default_factory=list)
    
    # Diagnostic metrics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "parameters": {
                "beta": round(self.beta, 4),
                "max_overlap": round(self.max_overlap, 2),
                "learning_rate": round(self.learning_rate, 4),
            },
            "options": [o.to_dict() for o in self.options],
            "diversity_checks": [d.to_dict() for d in self.diversity_checks],
            "ranking": self.ranking,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "tag_updates": [u.to_dict() for u in self.tag_updates],
            "top_10_after": self.top_10_after,
            "diagnostics": self.diagnostics,
        }


class GPRefinementLogger:
    """
    Comprehensive logger for GP refinement sessions.
    
    Usage:
        logger = GPRefinementLogger(session_id, output_dir)
        logger.log_initialization(tags, config)
        
        # Each round:
        logger.start_round(round_num, beta, max_overlap, learning_rate)
        logger.log_option(option_id, strategy, tags, weights)
        logger.log_diversity_check(...)
        logger.log_ranking(ranking)
        logger.log_comparison(...)
        logger.log_tag_update(...)
        logger.end_round(top_tags, diagnostics)
        
        # At end:
        logger.log_final_selection(tags, weights)
        logger.save()
    """
    
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        
        # Initialization data
        self.init_data: Dict[str, Any] = {}
        
        # Round logs
        self.rounds: List[RoundLog] = []
        self.current_round: Optional[RoundLog] = None
        
        # Tag trajectories (track μ over rounds)
        self.tag_trajectories: Dict[str, List[Tuple[int, float, float]]] = {}  # tag_id -> [(round, mu, sigma), ...]
        
        # Final data
        self.final_data: Dict[str, Any] = {}
        
        # Warnings/issues detected
        self.warnings: List[Dict[str, Any]] = []
    
    # =========================================================================
    # Initialization Logging
    # =========================================================================
    
    def log_initialization(
        self,
        positive_tags: List[str],
        neutral_tags: List[str],
        selected_image_tags: set,
        tag_states: Dict[str, Any],  # tag_id -> TagState
        config: Dict[str, Any],
    ) -> None:
        """Log the initialization of the GP refiner."""
        
        # Categorize tags
        categories = {
            "positive_selected": [],
            "positive_other": [],
            "neutral_selected": [],
            "neutral_other": [],
        }
        
        all_tags = []
        for tag_id, tag_state in tag_states.items():
            tag_info = {
                "tag_id": tag_id,
                "text": tag_state.text,
                "category": tag_state.category.value,
                "initial_mu": round(tag_state.mu, 4),
                "initial_sigma": round(tag_state.sigma, 4),
            }
            all_tags.append(tag_info)
            categories[tag_state.category.value].append(tag_state.text)
            
            # Initialize trajectory
            self.tag_trajectories[tag_id] = [(0, tag_state.mu, tag_state.sigma)]
        
        self.init_data = {
            "session_id": self.session_id,
            "timestamp": self.start_time.isoformat(),
            "config": config,
            "input_summary": {
                "total_positive": len(positive_tags),
                "total_neutral": len(neutral_tags),
                "total_tags": len(tag_states),
                "in_selected_images": len(selected_image_tags),
            },
            "categories": {
                k: {"count": len(v), "tags": v}
                for k, v in categories.items()
            },
            "all_tags": all_tags,
            "prior_ordering": "positive_selected > positive_other > neutral_selected > neutral_other",
        }
        
        # Check for potential issues
        if len(positive_tags) < 7:
            self._add_warning("low_positive_count", 
                f"Only {len(positive_tags)} positive tags. May not have enough diversity.")
        
        if len(neutral_tags) < 3:
            self._add_warning("low_neutral_count",
                f"Only {len(neutral_tags)} neutral tags. Exploration may be limited.")
        
        print(f"[GPLogger] Initialized: {len(tag_states)} tags logged")
    
    # =========================================================================
    # Round Logging
    # =========================================================================
    
    def start_round(
        self,
        round_num: int,
        beta: float,
        max_overlap: float,
        learning_rate: float,
    ) -> None:
        """Start logging a new round."""
        self.current_round = RoundLog(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            beta=beta,
            max_overlap=max_overlap,
            learning_rate=learning_rate,
        )
        print(f"[GPLogger] Started round {round_num}")
    
    def log_option(
        self,
        option_id: int,
        strategy: str,
        tag_ids: List[str],
        tag_texts: List[str],
        tag_mus: List[float],
        tag_sigmas: List[float],
        weights: Optional[List[float]] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Log a generated option."""
        if not self.current_round:
            return
        
        tags = []
        for i, (tid, text, mu, sigma) in enumerate(zip(tag_ids, tag_texts, tag_mus, tag_sigmas)):
            tag_entry = {
                "tag_id": tid,
                "text": text,
                "mu": round(mu, 4),
                "sigma": round(sigma, 4),
                "weight": round(weights[i], 4) if weights else None,
            }
            tags.append(tag_entry)
        
        option = OptionLog(
            option_id=option_id,
            strategy=strategy,
            tags=tags,
            avg_mu=float(np.mean(tag_mus)),
            avg_sigma=float(np.mean(tag_sigmas)),
            image_path=image_path,
        )
        self.current_round.options.append(option)
    
    def log_diversity_check(
        self,
        option_a: int,
        option_b: int,
        shared_tags: List[str],
        max_allowed: float,
    ) -> None:
        """Log a diversity check between two options."""
        if not self.current_round:
            return
        
        overlap_count = len(shared_tags)
        # Assume 10 tags per option
        overlap_fraction = overlap_count / 10.0
        passed = overlap_fraction <= max_allowed
        
        check = DiversityCheck(
            option_a=option_a,
            option_b=option_b,
            shared_tags=shared_tags,
            overlap_count=overlap_count,
            overlap_fraction=round(overlap_fraction, 2),
            max_allowed=max_allowed,
            passed=passed,
        )
        self.current_round.diversity_checks.append(check)
        
        if not passed:
            self._add_warning("diversity_violation",
                f"Round {self.current_round.round_num}: Options {option_a} and {option_b} "
                f"have {overlap_fraction:.0%} overlap (max {max_allowed:.0%})")
    
    def log_ranking(self, ranking: List[int]) -> None:
        """Log the user's ranking."""
        if not self.current_round:
            return
        self.current_round.ranking = ranking
    
    def log_comparison(
        self,
        better_option_id: int,
        worse_option_id: int,
        ranking: List[int],
        better_tags: set,
        worse_tags: set,
    ) -> None:
        """Log a pairwise comparison."""
        if not self.current_round:
            return
        
        rank_better = ranking.index(better_option_id)
        rank_worse = ranking.index(worse_option_id)
        rank_distance = rank_worse - rank_better
        
        # Preference strength by rank distance
        strength_map = {3: 1.0, 2: 0.6, 1: 0.3}
        preference_strength = strength_map.get(rank_distance, 0.5)
        
        tags_only_in_better = list(better_tags - worse_tags)
        tags_only_in_worse = list(worse_tags - better_tags)
        shared_tags = list(better_tags & worse_tags)
        
        comparison = PairwiseComparison(
            better_option_id=better_option_id,
            worse_option_id=worse_option_id,
            rank_distance=rank_distance,
            preference_strength=preference_strength,
            tags_only_in_better=tags_only_in_better,
            tags_only_in_worse=tags_only_in_worse,
            shared_tags=shared_tags,
        )
        self.current_round.comparisons.append(comparison)
        
        # Check for low-information comparisons
        if len(tags_only_in_better) + len(tags_only_in_worse) < 2:
            self._add_warning("low_info_comparison",
                f"Round {self.current_round.round_num}: Comparison {better_option_id} > {worse_option_id} "
                f"has only {len(tags_only_in_better) + len(tags_only_in_worse)} differentiating tags")
    
    def log_tag_update(
        self,
        tag_id: str,
        text: str,
        mu_before: float,
        mu_after: float,
        sigma_before: float,
        sigma_after: float,
        update_reason: str,
        from_comparison: str,
    ) -> None:
        """Log an update to a tag's utilities."""
        if not self.current_round:
            return
        
        update = TagUpdate(
            tag_id=tag_id,
            text=text,
            mu_before=mu_before,
            mu_after=mu_after,
            mu_delta=mu_after - mu_before,
            sigma_before=sigma_before,
            sigma_after=sigma_after,
            update_reason=update_reason,
            from_comparison=from_comparison,
        )
        self.current_round.tag_updates.append(update)
        
        # Check for extreme updates
        if abs(mu_after - mu_before) > 0.5:
            self._add_warning("large_mu_update",
                f"Round {self.current_round.round_num}: Tag '{text}' had large μ update: "
                f"{mu_before:.3f} → {mu_after:.3f}")
    
    def end_round(
        self,
        tag_states: Dict[str, Any],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End the current round and compute diagnostics."""
        if not self.current_round:
            return
        
        round_num = self.current_round.round_num
        
        # Get top 10 tags
        sorted_tags = sorted(
            tag_states.values(),
            key=lambda t: t.mu,
            reverse=True
        )[:10]
        
        self.current_round.top_10_after = [
            {
                "rank": i + 1,
                "tag_id": t.tag_id,
                "text": t.text,
                "mu": round(t.mu, 4),
                "sigma": round(t.sigma, 4),
            }
            for i, t in enumerate(sorted_tags)
        ]
        
        # Update trajectories
        for tag_id, tag_state in tag_states.items():
            if tag_id not in self.tag_trajectories:
                self.tag_trajectories[tag_id] = []
            self.tag_trajectories[tag_id].append((round_num, tag_state.mu, tag_state.sigma))
        
        # Compute diagnostics
        all_mus = [t.mu for t in tag_states.values()]
        all_sigmas = [t.sigma for t in tag_states.values()]
        
        self.current_round.diagnostics = {
            "mu_distribution": {
                "min": round(min(all_mus), 4),
                "max": round(max(all_mus), 4),
                "mean": round(np.mean(all_mus), 4),
                "std": round(np.std(all_mus), 4),
            },
            "sigma_distribution": {
                "min": round(min(all_sigmas), 4),
                "max": round(max(all_sigmas), 4),
                "mean": round(np.mean(all_sigmas), 4),
            },
            "tags_updated": len(self.current_round.tag_updates),
            "unique_tags_updated": len(set(u.tag_id for u in self.current_round.tag_updates)),
            "comparisons_made": len(self.current_round.comparisons),
            "avg_differentiating_tags": np.mean([
                len(c.tags_only_in_better) + len(c.tags_only_in_worse)
                for c in self.current_round.comparisons
            ]) if self.current_round.comparisons else 0,
        }
        
        if diagnostics:
            self.current_round.diagnostics.update(diagnostics)
        
        # Check for convergence issues
        if np.mean(all_sigmas) < 0.15:
            self._add_warning("early_convergence",
                f"Round {round_num}: Mean σ is {np.mean(all_sigmas):.3f}. "
                f"Uncertainty may be collapsing too fast.")
        
        if np.std(all_mus) < 0.1:
            self._add_warning("low_mu_variance",
                f"Round {round_num}: μ variance is {np.std(all_mus):.3f}. "
                f"Tags may not be differentiating well.")
        
        self.rounds.append(self.current_round)
        self.current_round = None
        
        print(f"[GPLogger] Ended round {round_num}: {len(self.rounds[-1].tag_updates)} tag updates")
    
    # =========================================================================
    # Final Logging
    # =========================================================================
    
    def log_final_selection(
        self,
        final_tags: List[str],
        weights: Dict[str, float],
        tag_states: Dict[str, Any],
    ) -> None:
        """Log the final tag selection."""
        
        # Build detailed final tag info
        final_tag_details = []
        for i, tag_text in enumerate(final_tags):
            # Find tag state
            tag_state = None
            for ts in tag_states.values():
                if ts.text == tag_text:
                    tag_state = ts
                    break
            
            if tag_state:
                final_tag_details.append({
                    "rank": i + 1,
                    "text": tag_text,
                    "weight": round(weights.get(tag_text, 0), 4),
                    "final_mu": round(tag_state.mu, 4),
                    "final_sigma": round(tag_state.sigma, 4),
                    "category": tag_state.category.value,
                    "times_shown": tag_state.times_shown,
                    "times_in_winner": tag_state.times_in_winner,
                    "times_in_loser": tag_state.times_in_loser,
                    "win_rate": round(
                        tag_state.times_in_winner / tag_state.times_shown, 3
                    ) if tag_state.times_shown > 0 else None,
                })
        
        # Compute trajectories for final tags
        final_trajectories = {}
        for tag_text in final_tags:
            for tag_id, ts in tag_states.items():
                if ts.text == tag_text and tag_id in self.tag_trajectories:
                    final_trajectories[tag_text] = [
                        {"round": r, "mu": round(m, 4), "sigma": round(s, 4)}
                        for r, m, s in self.tag_trajectories[tag_id]
                    ]
                    break
        
        # Summary statistics
        total_comparisons = sum(len(r.comparisons) for r in self.rounds)
        total_updates = sum(len(r.tag_updates) for r in self.rounds)
        
        self.final_data = {
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": len(self.rounds),
            "total_comparisons": total_comparisons,
            "total_tag_updates": total_updates,
            "final_tags": final_tag_details,
            "final_trajectories": final_trajectories,
            "convergence_analysis": self._analyze_convergence(tag_states),
            "warnings_summary": {
                "total": len(self.warnings),
                "by_type": self._group_warnings(),
            },
        }
        
        print(f"[GPLogger] Final selection logged: {len(final_tags)} tags")
    
    def _analyze_convergence(self, tag_states: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well the refinement converged."""
        
        # Check if top tags are stable
        if len(self.rounds) < 2:
            return {"status": "insufficient_rounds"}
        
        # Compare top 10 between last two rounds
        last_top = set(t["text"] for t in self.rounds[-1].top_10_after)
        prev_top = set(t["text"] for t in self.rounds[-2].top_10_after) if len(self.rounds) > 1 else set()
        
        overlap = len(last_top & prev_top)
        
        # Check μ stability
        final_mus = [t.mu for t in tag_states.values()]
        mu_spread = max(final_mus) - min(final_mus)
        
        # Check σ (should be low for decided tags)
        final_sigmas = [t.sigma for t in tag_states.values()]
        
        return {
            "top_10_stability": f"{overlap}/10 tags same as previous round",
            "mu_spread": round(mu_spread, 4),
            "avg_final_sigma": round(np.mean(final_sigmas), 4),
            "min_final_sigma": round(min(final_sigmas), 4),
            "converged": overlap >= 8 and np.mean(final_sigmas) < 0.3,
        }
    
    def _add_warning(self, warning_type: str, message: str) -> None:
        """Add a warning to the log."""
        self.warnings.append({
            "type": warning_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"[GPLogger WARNING] {message}")
    
    def _group_warnings(self) -> Dict[str, int]:
        """Group warnings by type."""
        groups = {}
        for w in self.warnings:
            t = w["type"]
            groups[t] = groups.get(t, 0) + 1
        return groups
    
    # =========================================================================
    # Saving
    # =========================================================================
    
    def save(self) -> str:
        """Save the complete log to a JSON file."""
        
        log_data = {
            "metadata": {
                "session_id": self.session_id,
                "log_version": "1.0",
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            },
            "initialization": self.init_data,
            "rounds": [r.to_dict() for r in self.rounds],
            "final": self.final_data,
            "warnings": self.warnings,
            "tag_trajectories": {
                tag_id: [
                    {"round": r, "mu": round(m, 4), "sigma": round(s, 4)}
                    for r, m, s in trajectory
                ]
                for tag_id, trajectory in self.tag_trajectories.items()
            },
        }
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        # Save main log
        log_path = self.output_dir / "gp_refinement_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, cls=NumpyEncoder)
        
        # Save human-readable summary
        summary_path = self.output_dir / "gp_refinement_summary.txt"
        self._save_summary(summary_path)
        
        print(f"[GPLogger] Saved log to {log_path}")
        print(f"[GPLogger] Saved summary to {summary_path}")
        
        return str(log_path)
    
    def _save_summary(self, path: Path) -> None:
        """Save a human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("GP REFINEMENT SESSION SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Started: {self.start_time.isoformat()}")
        lines.append(f"Rounds: {len(self.rounds)}")
        lines.append("")
        
        # Initialization summary
        if self.init_data:
            lines.append("-" * 70)
            lines.append("INITIALIZATION")
            lines.append("-" * 70)
            inp = self.init_data.get("input_summary", {})
            lines.append(f"  Positive tags: {inp.get('total_positive', 0)}")
            lines.append(f"  Neutral tags: {inp.get('total_neutral', 0)}")
            lines.append(f"  Total: {inp.get('total_tags', 0)}")
            lines.append("")
        
        # Round summaries
        for r in self.rounds:
            lines.append("-" * 70)
            lines.append(f"ROUND {r.round_num} (β={r.beta:.2f})")
            lines.append("-" * 70)
            
            lines.append("  Options:")
            for opt in r.options:
                top_tags = [t["text"] for t in opt.tags[:3]]
                lines.append(f"    [{opt.option_id}] {opt.strategy:12s} | "
                           f"avg_μ={opt.avg_mu:+.2f} | {', '.join(top_tags)}...")
            
            if r.ranking:
                lines.append(f"  Ranking: {r.ranking}")
            
            lines.append(f"  Comparisons: {len(r.comparisons)}")
            lines.append(f"  Tags updated: {r.diagnostics.get('unique_tags_updated', 0)}")
            
            lines.append("  Top 5 after round:")
            for t in r.top_10_after[:5]:
                lines.append(f"    {t['rank']}. {t['text'][:25]:25s} μ={t['mu']:+.3f} σ={t['sigma']:.3f}")
            lines.append("")
        
        # Final selection
        if self.final_data and "final_tags" in self.final_data:
            lines.append("=" * 70)
            lines.append("FINAL SELECTION")
            lines.append("=" * 70)
            for t in self.final_data["final_tags"]:
                win_rate = f"{t['win_rate']:.0%}" if t.get('win_rate') is not None else "N/A"
                lines.append(f"  {t['rank']:2d}. {t['text'][:30]:30s} | "
                           f"w={t['weight']:.3f} | μ={t['final_mu']:+.3f} | "
                           f"win_rate={win_rate}")
            lines.append("")
            
            conv = self.final_data.get("convergence_analysis", {})
            lines.append(f"  Convergence: {conv.get('converged', 'Unknown')}")
            lines.append(f"  Top-10 stability: {conv.get('top_10_stability', 'N/A')}")
        
        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"WARNINGS ({len(self.warnings)})")
            lines.append("=" * 70)
            for w in self.warnings:
                lines.append(f"  [{w['type']}] {w['message']}")
        
        with open(path, 'w') as f:
            f.write("\n".join(lines))


# ============================================================================
# Integration helper
# ============================================================================

def create_logger(session_id: str, session_folder: Path) -> GPRefinementLogger:
    """Create a logger for a GP refinement session."""
    log_dir = Path(session_folder) / "gp_refinement"
    return GPRefinementLogger(session_id, log_dir)
