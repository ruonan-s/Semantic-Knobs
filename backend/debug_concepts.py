"""
Debug script for concept refinement system
Tracks all interactions and state changes for debugging
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

DEBUG_LOG_DIR = "sessions/debug_logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


class ConceptDebugger:
    """Tracks and logs all concept refinement interactions"""
    
    def __init__(self, session_id: str, stage: str):
        self.session_id = session_id
        self.stage = stage
        self.log_file = os.path.join(
            DEBUG_LOG_DIR, 
            f"{session_id}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.events = []
        self.current_state = None
        
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event with timestamp"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.events.append(event)
        
        # Also print to console for real-time debugging
        print(f"\n{'='*80}")
        print(f"[CONCEPT DEBUG] {event_type.upper()}")
        print(f"Time: {event['timestamp']}")
        print(f"{'='*80}")
        self._print_data(data)
        print(f"{'='*80}\n")
        
        # Save to file
        self._save_to_file()
    
    def _print_data(self, data: Dict, indent: int = 0):
        """Pretty print data recursively"""
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_data(value, indent + 1)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"{prefix}{key}: ({len(value)} items)")
                    for i, item in enumerate(value[:3]):  # Show first 3
                        print(f"{prefix}  [{i}]:")
                        self._print_data(item, indent + 2)
                    if len(value) > 3:
                        print(f"{prefix}  ... and {len(value) - 3} more")
                else:
                    print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: {value}")
    
    def _save_to_file(self):
        """Save all events to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'session_id': self.session_id,
                    'stage': self.stage,
                    'events': self.events,
                    'final_state': self.current_state
                }, f, indent=2, default=str)  # Use default=str for non-serializable objects
        except Exception as e:
            print(f"Error saving debug log: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def log_initialization(self, raw_tags: List, concepts: List, initial_states: Dict):
        """Log initialization phase"""
        self.log_event('initialization', {
            'total_raw_tags': len(raw_tags),
            'tags_sample': [
                {'id': t.id, 'text': t.text, 'image_id': t.image_id, 'concept_id': t.concept_id}
                for t in raw_tags[:10]
            ],
            'total_concepts': len(concepts),
            'concepts': [
                {
                    'id': c.id,
                    'label': c.label,
                    'member_count': len(c.member_tag_ids),
                    'member_tags': getattr(c, 'member_tags', [])[:5] if hasattr(c, 'member_tags') else []
                }
                for c in concepts
            ],
            'initial_weights': {
                c.id: {
                    'w': initial_states[c.id].w,
                    'ema_w': initial_states[c.id].ema_w,
                    'like_count': initial_states[c.id].like_count,
                    'dislike_count': initial_states[c.id].dislike_count
                }
                for c in concepts
            }
        })
        
    def log_categorization(self, concepts: List, concept_states: Dict, categorized: Dict):
        """Log categorization logic"""
        K = len(concepts)
        w_base = 1.0 / K if K > 0 else 0.0
        delta = 0.2 / K if K > 0 else 0.0
        
        details = []
        for concept in concepts:
            state = concept_states[concept.id]
            w = state.w
            
            if w >= w_base + delta:
                category = 'positive'
            elif w <= w_base - delta:
                category = 'negative'
            else:
                category = 'neutral'
            
            details.append({
                'id': concept.id,
                'label': concept.label,
                'w': w,
                'ema_w': state.ema_w,
                'w_base': w_base,
                'delta': delta,
                'threshold_positive': w_base + delta,
                'threshold_negative': w_base - delta,
                'computed_category': category,
                'in_positive_list': concept.id in categorized.get('positive', []),
                'in_neutral_list': concept.id in categorized.get('neutral', []),
                'in_negative_list': concept.id in categorized.get('negative', [])
            })
        
        self.log_event('categorization', {
            'K': K,
            'w_base': w_base,
            'delta': delta,
            'threshold_positive': w_base + delta,
            'threshold_negative': w_base - delta,
            'categorized_counts': {
                'positive': len(categorized.get('positive', [])),
                'neutral': len(categorized.get('neutral', [])),
                'negative': len(categorized.get('negative', []))
            },
            'categorized_ids': categorized,
            'concept_details': details
        })
    
    def log_tag_interaction(self, tag_id: str, preference: str, 
                           concept_id: str, before_state: Dict, after_state: Dict):
        """Log tag click interaction"""
        self.log_event('tag_interaction', {
            'tag_id': tag_id,
            'preference': preference,
            'concept_id': concept_id,
            'before': {
                'like_count': before_state.get('like_count', 0),
                'dislike_count': before_state.get('dislike_count', 0),
                'w': before_state.get('w', 0),
                'ema_w': before_state.get('ema_w', 0),
                'score': before_state.get('score', 0),
                'liked_tags': list(before_state.get('liked_tags', set())),
                'disliked_tags': list(before_state.get('disliked_tags', set()))
            },
            'after': {
                'like_count': after_state.get('like_count', 0),
                'dislike_count': after_state.get('dislike_count', 0),
                'w': after_state.get('w', 0),
                'ema_w': after_state.get('ema_w', 0),
                'score': after_state.get('score', 0),
                'liked_tags': list(after_state.get('liked_tags', set())),
                'disliked_tags': list(after_state.get('disliked_tags', set()))
            },
            'changes': {
                'like_count_delta': after_state.get('like_count', 0) - before_state.get('like_count', 0),
                'dislike_count_delta': after_state.get('dislike_count', 0) - before_state.get('dislike_count', 0),
                'w_delta': after_state.get('w', 0) - before_state.get('w', 0),
                'ema_w_delta': after_state.get('ema_w', 0) - before_state.get('ema_w', 0),
                'was_toggled_off': after_state.get('like_count', 0) < before_state.get('like_count', 0) or 
                                  after_state.get('dislike_count', 0) < before_state.get('dislike_count', 0)
            }
        })
    
    def log_weight_computation(self, concepts: List, concept_states: Dict, 
                              scores: Dict, weights: Dict):
        """Log weight computation details"""
        self.log_event('weight_computation', {
            'total_concepts': len(concepts),
            'computation_details': [
                {
                    'id': c.id,
                    'label': c.label,
                    'like_count': concept_states[c.id].like_count,
                    'dislike_count': concept_states[c.id].dislike_count,
                    'rank_bonus': concept_states[c.id].rank_bonus,
                    'rank_penalty': concept_states[c.id].rank_penalty,
                    'raw_score': scores.get(c.id, 0),
                    'final_score': concept_states[c.id].score,
                    'weight_w': weights.get(c.id, 0),
                    'weight_ema_w': concept_states[c.id].ema_w
                }
                for c in concepts
            ],
            'weight_sum': sum(weights.values()),
            'score_range': {
                'min': min(scores.values()) if scores else 0,
                'max': max(scores.values()) if scores else 0
            }
        })
    
    def log_ranking_update(self, positive_ids: List[str], negative_ids: List[str],
                          rank_bonuses: Dict, rank_penalties: Dict):
        """Log ranking update"""
        self.log_event('ranking_update', {
            'positive_list': positive_ids,
            'negative_list': negative_ids,
            'rank_bonuses': {
                cid: {'rank': i, 'bonus': rank_bonuses.get(cid, 0)}
                for i, cid in enumerate(positive_ids)
            },
            'rank_penalties': {
                cid: {'rank': i, 'penalty': rank_penalties.get(cid, 0)}
                for i, cid in enumerate(negative_ids)
            }
        })
    
    def log_image_effects(self, image_effects: Dict, incidence_matrix: Dict):
        """Log image effect calculations"""
        self.log_event('image_effects', {
            'effects': image_effects,
            'incidence_matrix': incidence_matrix,
            'effect_range': {
                'min': min(image_effects.values()) if image_effects else 0,
                'max': max(image_effects.values()) if image_effects else 0
            }
        })
    
    def update_state(self, state: Dict):
        """Update current state snapshot"""
        self.current_state = state
        self._save_to_file()


# Global debugger instances
_debuggers: Dict[str, ConceptDebugger] = {}


def get_debugger(session_id: str, stage: str) -> ConceptDebugger:
    """Get or create debugger for session/stage"""
    key = f"{session_id}_{stage}"
    if key not in _debuggers:
        _debuggers[key] = ConceptDebugger(session_id, stage)
    return _debuggers[key]


def log_concept_event(session_id: str, stage: str, event_type: str, data: Dict):
    """Convenience function to log an event"""
    debugger = get_debugger(session_id, stage)
    debugger.log_event(event_type, data)

