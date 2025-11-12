"""
Preference-Driven Tag Refinement (Single-Round)
Implements concept building, weight learning, and preview from raw image tags.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

# Use CLIP for embeddings
import torch
import clip

# Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering

# Import debug utilities
from debug_concepts import get_debugger

# Load CLIP model globally (load once, reuse)
print("Loading CLIP model (ViT-L/14)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
print(f"CLIP ViT-L/14 model loaded on {device}")


# ============================================================================
# Parameters (defaults from spec)
# ============================================================================
# Agglomerative clustering parameters
DISTANCE_THRESHOLD = 0.2  # Stop merging when cosine distance exceeds this (0-2 range)
                           # Lower = tighter clusters (more concepts)
                           # Higher = looser clusters (fewer concepts)
MIN_CLUSTERS = 3           # Minimum number of clusters (safety bound)
MAX_CLUSTERS = 100         # Maximum number of clusters (safety bound)

# Weight computation parameters
TAU = 2.0               # softmax temperature
A = 1.0                 # like strength
B = 1.0                 # dislike strength
BETA_POS = 0.3          # rank intensity positive
BETA_NEG = 0.3          # rank intensity negative
TAU_POS = 1.0           # rank decay positive
TAU_NEG = 2.0           # rank decay negative
S_CAP = 3.0             # negative score clamp
GAMMA_NEG = 0.7         # negative effect strength in image preview


# ============================================================================
# Data Models
# ============================================================================
@dataclass
class RawTag:
    """Per-image tag with embedding"""
    id: str
    text: str
    embedding: List[float]  # CLIP text embedding, unit-norm
    concept_id: Optional[str] = None
    image_id: str = ""


@dataclass
class Concept:
    """Stage-local concept (synonym cluster)"""
    id: str
    label: str
    centroid: List[float]  # unit-norm centroid
    member_tag_ids: List[str]


@dataclass
class ConceptState:
    """Learned state for a concept"""
    like_count: int = 0
    dislike_count: int = 0
    rank_bonus: float = 0.0
    rank_penalty: float = 0.0
    score: float = 0.0
    w: float = 0.0          # normalized weight
    liked_tags: set = None  # Track which tags have been liked
    disliked_tags: set = None  # Track which tags have been disliked
    
    def __post_init__(self):
        if self.liked_tags is None:
            self.liked_tags = set()
        if self.disliked_tags is None:
            self.disliked_tags = set()


# ============================================================================
# Text Normalization (DISABLED - CLIP handles variations naturally)
# ============================================================================
def normalize_text(text: str) -> str:
    """No normalization - just basic cleanup"""
    return text.strip()


def simple_lemmatize(text: str) -> str:
    """No lemmatization - CLIP understands plural/singular naturally"""
    return text


# ============================================================================
# Embedding via CLIP
# ============================================================================
def get_text_embedding(text: str) -> np.ndarray:
    """Get normalized embedding for text using CLIP"""
    try:
        with torch.no_grad():
            # Tokenize text
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            # Get text features
            text_features = clip_model.encode_text(text_tokens)
            # Normalize to unit length
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Convert to numpy
            embedding = text_features.cpu().numpy()[0].astype(np.float32)
        return embedding
    except Exception as e:
        print(f"Error getting CLIP embedding for '{text}': {e}")
        # Return random normalized vector as fallback (ViT-L/14 dimension is 768)
        embedding = np.random.randn(768).astype(np.float32)
        return embedding / np.linalg.norm(embedding)


def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings for multiple texts in batch using CLIP"""
    try:
        with torch.no_grad():
            # Tokenize all texts
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            # Get text features
            text_features = clip_model.encode_text(text_tokens)
            # Normalize to unit length
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Convert to numpy
            embeddings = text_features.cpu().numpy().astype(np.float32)
            # Return as list of arrays
            return [embedding for embedding in embeddings]
    except Exception as e:
        print(f"Error getting batch CLIP embeddings: {e}")
        # Return random normalized vectors as fallback (ViT-L/14 dimension is 768)
        embeddings = []
        for _ in texts:
            embedding = np.random.randn(768).astype(np.float32)
            embeddings.append(embedding / np.linalg.norm(embedding))
        return embeddings


# ============================================================================
# Concept Building (K-Means Clustering)
# ============================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit vectors"""
    return float(np.dot(a, b))


def find_elbow_point(inertias: list, k_values: list) -> int:
    """
    Find elbow point using the maximum distance from line method.
    Returns the K value at the elbow.
    """
    if len(inertias) < 3:
        return k_values[len(inertias) // 2]
    
    inertias = np.array(inertias, dtype=np.float64)
    k_values = np.array(k_values, dtype=np.float64)
    
    # Create line from first to last point
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    # Calculate perpendicular distance from each point to the line
    distances = []
    for i in range(len(k_values)):
        point = np.array([k_values[i], inertias[i]])
        point_vec = point - p1
        # Project point onto line, then compute perpendicular distance
        projection = np.dot(point_vec, line_vec) / (line_len ** 2) * line_vec
        perpendicular = point_vec - projection
        distance = np.linalg.norm(perpendicular)
        distances.append(distance)
    
    # Find the point with maximum distance (the elbow)
    elbow_idx = np.argmax(distances)
    return int(k_values[elbow_idx])


# No longer needed - agglomerative determines K automatically
# Kept for reference/comparison
def estimate_optimal_k_legacy(n_tags: int, embeddings: np.ndarray, 
                       min_k: int = MIN_CLUSTERS, 
                       max_k: int = MAX_CLUSTERS) -> int:
    """LEGACY: K-means elbow method (no longer used with agglomerative)"""
    # This function is kept for reference but not called
    # Agglomerative clustering determines K automatically via distance_threshold
    k_target = max(min_k, min(max_k, n_tags // 3))
    return k_target
    optimal_k = find_elbow_point(inertias, k_range[:len(inertias)])
    
    print(f"  ✓ Elbow point found at K={optimal_k}")
    print(f"  Inertia at elbow: {inertias[k_range[:len(inertias)].index(optimal_k)]:.2f}")
    print(f"  Average tags per cluster: {n_tags / optimal_k:.1f}")
    
    return optimal_k


def build_concepts(raw_tags: List[RawTag]) -> Tuple[List[Concept], Dict[str, str]]:
    """
    Build concepts using Agglomerative Clustering on tag embeddings.
    Uses cosine distance and automatically determines number of clusters.
    Returns: (concepts, tag_id_to_concept_id)
    """
    n = len(raw_tags)
    if n == 0:
        return [], {}
    
    print(f"\n[AGGLOMERATIVE CLUSTERING] Building concepts from {n} tags")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD} (cosine distance)")
    print(f"  Linkage: average")
    
    # Prepare embeddings matrix
    embeddings = np.array([tag.embedding for tag in raw_tags])
    
    # Run Agglomerative Clustering
    # n_clusters=None means use distance_threshold to determine K automatically
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric='cosine',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Determine actual number of clusters created
    K = len(set(cluster_labels))
    print(f"  Automatically determined K={K} clusters from distance threshold")
    
    # Safety bounds check
    if K < MIN_CLUSTERS:
        print(f"  ⚠️ Warning: Only {K} clusters created (min={MIN_CLUSTERS}). Consider lowering distance_threshold.")
    elif K > MAX_CLUSTERS:
        print(f"  ⚠️ Warning: {K} clusters created (max={MAX_CLUSTERS}). Consider raising distance_threshold.")
    
    # Group tags by cluster
    components = [[] for _ in range(K)]
    for i, label in enumerate(cluster_labels):
        components[label].append(i)
    
    # Filter out empty clusters (shouldn't happen but just in case)
    components = [comp for comp in components if len(comp) > 0]
    
    print(f"  Created {len(components)} non-empty clusters")
    
    # Compute cluster statistics
    cluster_sizes = [len(comp) for comp in components]
    print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
    
    # Create Concept objects
    concepts = []
    tag_id_to_concept_id = {}
    
    for comp_idx, component in enumerate(components):
        concept_id = f"concept_{comp_idx}"
        
        # Get member tags
        member_tags = [raw_tags[i] for i in component]
        member_tag_ids = [tag.id for tag in member_tags]
        
        # Compute centroid
        embeddings = np.array([tag.embedding for tag in member_tags])
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Choose label: shortest, most frequent tag
        tag_texts = [tag.text for tag in member_tags]
        tag_counts = defaultdict(int)
        for text in tag_texts:
            tag_counts[text] += 1
        
        # Sort by frequency desc, then length asc
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], len(x[0])))
        label = sorted_tags[0][0] if sorted_tags else f"concept_{comp_idx}"
        
        concept = Concept(
            id=concept_id,
            label=label,
            centroid=centroid.tolist(),
            member_tag_ids=member_tag_ids
        )
        concepts.append(concept)
        
        # Map tags to concept
        for tag in member_tags:
            tag.concept_id = concept_id
            tag_id_to_concept_id[tag.id] = concept_id
    
    # Log final concept summary
    print(f"\n[CONCEPT CLUSTERING] Created {len(concepts)} concepts:")
    for i, concept in enumerate(concepts[:15]):  # Show first 15 concepts
        member_count = len(concept.member_tag_ids)
        member_texts = [tag.text for tag in raw_tags if tag.id in concept.member_tag_ids]
        unique_member_texts = list(dict.fromkeys(member_texts))  # Deduplicate while preserving order
        
        if member_count > 1:
            print(f"  Concept {i+1}: '{concept.label}' ({member_count} tag instances, {len(unique_member_texts)} unique)")
            print(f"    Unique tags: {', '.join(unique_member_texts[:5])}")
            if len(unique_member_texts) > 5:
                print(f"    ... +{len(unique_member_texts) - 5} more")
        else:
            print(f"  Concept {i+1}: '{concept.label}' (singleton)")
    
    if len(concepts) > 15:
        print(f"  ... +{len(concepts) - 15} more concepts")
    
    return concepts, tag_id_to_concept_id


# ============================================================================
# Weight Computation
# ============================================================================
def compute_weights(
    concepts: List[Concept],
    concept_states: Dict[str, ConceptState],
    tau: float = TAU,
    a: float = A,
    b: float = B
) -> Dict[str, ConceptState]:
    """
    Compute concept weights from interaction signals.
    Updates concept_states in place and returns it.
    """
    if not concepts:
        return concept_states
    
    K = len(concepts)
    epsilon = 0.002 / K
    
    # Step 1: Compute raw scores
    scores = {}
    for concept in concepts:
        state = concept_states[concept.id]
        score = (a * state.like_count - 
                b * state.dislike_count + 
                state.rank_bonus - 
                state.rank_penalty)
        
        # Apply guardrails
        max_rank = 2 * a
        state.rank_bonus = np.clip(state.rank_bonus, -max_rank, max_rank)
        state.rank_penalty = np.clip(state.rank_penalty, -max_rank, max_rank)
        score = np.clip(score, -S_CAP, S_CAP)
        
        scores[concept.id] = score
        state.score = score
    
    # Step 2: Softmax with temperature
    score_values = np.array([scores[c.id] for c in concepts])
    exp_scores = np.exp(score_values / tau)
    weights = exp_scores / np.sum(exp_scores)
    
    # Apply floor and renormalize
    weights = np.maximum(weights, epsilon)
    weights = weights / np.sum(weights)
    
    # Step 3: Update states with new weights
    scores_dict = {}
    weights_dict = {}
    for i, concept in enumerate(concepts):
        state = concept_states[concept.id]
        new_w = float(weights[i])
        
        scores_dict[concept.id] = state.score
        weights_dict[concept.id] = new_w
        
        state.w = new_w
    
    # Minimal debug output (removed verbose logging for speed)
    # Only log weight sum as sanity check
    # print(f"[WEIGHT] Updated {len(concepts)} concepts, sum={sum(weights_dict.values()):.4f}")
    
    return concept_states


# ============================================================================
# Ranking Bonus/Penalty Computation
# ============================================================================
def compute_rank_bonus(rank: int, beta: float = BETA_POS, tau: float = TAU_POS) -> float:
    """Compute rank bonus for position in positive list (0-indexed)"""
    return beta * np.exp(-rank / tau)


def compute_rank_penalty(rank: int, beta: float = BETA_NEG, tau: float = TAU_NEG) -> float:
    """Compute rank penalty for position in negative list (0-indexed)"""
    return beta * np.exp(-rank / tau)


def update_rank_bonuses(
    positive_concept_ids: List[str],
    negative_concept_ids: List[str],
    concept_states: Dict[str, ConceptState]
) -> None:
    """Update rank bonuses/penalties based on ordered lists"""
    # Reset all bonuses/penalties
    for state in concept_states.values():
        state.rank_bonus = 0.0
        state.rank_penalty = 0.0
    
    # Apply bonuses to positive list
    for rank, concept_id in enumerate(positive_concept_ids):
        if concept_id in concept_states:
            concept_states[concept_id].rank_bonus = compute_rank_bonus(rank)
    
    # Apply penalties to negative list
    for rank, concept_id in enumerate(negative_concept_ids):
        if concept_id in concept_states:
            concept_states[concept_id].rank_penalty = compute_rank_penalty(rank)


# ============================================================================
# Auto Categorization
# ============================================================================
def categorize_concepts(
    concepts: List[Concept],
    concept_states: Dict[str, ConceptState]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Auto-categorize concepts into Positive, Neutral, Negative based on weights and scores.
    Returns: (positive_ids, neutral_ids, negative_ids)
    """
    if not concepts:
        return [], [], []
    
    K = len(concepts)
    
    # Get all weights and scores for adaptive thresholds
    weights = np.array([concept_states[c.id].w for c in concepts])
    scores = np.array([concept_states[c.id].score for c in concepts])
    
    # Compute statistics
    w_mean = np.mean(weights)
    w_std = np.std(weights)
    w_median = np.median(weights)
    
    # Also compute score statistics for better categorization
    s_mean = np.mean(scores)
    s_median = np.median(scores)
    
    # Adaptive thresholds based on distribution
    # Use median + std instead of fixed 1/K formula since softmax creates skew
    threshold_factor = 0.5  # Standard deviations from median (increased from 0.25)
    
    positive_threshold = w_median + threshold_factor * w_std
    negative_threshold = w_median - threshold_factor * w_std
    
    # Ensure thresholds are reasonable (not too tight)
    min_gap = 0.015  # Minimum 1.5% gap
    if positive_threshold - negative_threshold < 2 * min_gap:
        positive_threshold = w_median + min_gap
        negative_threshold = w_median - min_gap
    
    positive = []
    neutral = []
    negative = []
    
    print(f"\n[CATEGORIZATION] K={K}, w_mean={w_mean:.6f}, w_median={w_median:.6f}, w_std={w_std:.6f}")
    print(f"  Score stats: s_mean={s_mean:.3f}, s_median={s_median:.3f}")
    print(f"  Thresholds: positive >= {positive_threshold:.6f}, negative <= {negative_threshold:.6f}")
    
    # Log concepts with likes/dislikes for debugging
    concepts_with_interaction = []
    for concept in concepts:
        state = concept_states[concept.id]
        if state.like_count > 0 or state.dislike_count > 0:
            concepts_with_interaction.append({
                'label': concept.label,
                'likes': state.like_count,
                'dislikes': state.dislike_count,
                'score': state.score,
                'w': state.w
            })
    
    if concepts_with_interaction:
        print(f"\n  📊 Concepts with interactions:")
        for c in concepts_with_interaction:
            print(f"    {c['label']}: likes={c['likes']}, dislikes={c['dislikes']}, score={c['score']:.3f}, w={c['w']:.6f}")
    
    for concept in concepts:
        state = concept_states[concept.id]
        w = state.w
        score = state.score
        
        # Enhanced categorization logic:
        # 1. If a concept has more dislikes than likes, strongly prefer negative category
        # 2. Use both weight and score for categorization
        has_net_dislikes = state.dislike_count > state.like_count
        has_any_dislike = state.dislike_count > 0
        
        if has_net_dislikes or (has_any_dislike and w <= negative_threshold):
            # Concepts with net dislikes should be negative
            negative.append(concept.id)
            category = 'NEGATIVE'
        elif w >= positive_threshold:
            positive.append(concept.id)
            category = 'POSITIVE'
        elif w <= negative_threshold:
            negative.append(concept.id)
            category = 'NEGATIVE'
        else:
            neutral.append(concept.id)
            category = 'NEUTRAL'
        
        # Only print first 10 and concepts with interactions
        if len(positive) + len(neutral) + len(negative) <= 10 or state.like_count > 0 or state.dislike_count > 0:
            print(f"  {concept.label}: w={w:.6f}, score={score:.3f}, likes={state.like_count}, dislikes={state.dislike_count} -> {category}")
    
    # Sort by weight descending
    positive.sort(key=lambda cid: concept_states[cid].w, reverse=True)
    # For negative: sort ASCENDING so lowest weight (most negative) comes first
    negative.sort(key=lambda cid: concept_states[cid].w, reverse=False)
    neutral.sort(key=lambda cid: concept_states[cid].w, reverse=True)
    
    print(f"  Result: {len(positive)} positive, {len(neutral)} neutral, {len(negative)} negative")
    
    return positive, neutral, negative


# ============================================================================
# Image Effect Preview
# ============================================================================
def compute_image_effects(
    image_ids: List[str],
    incidence_matrix: Dict[str, Dict[str, int]],  # image_id -> concept_id -> count
    concepts: List[Concept],
    concept_states: Dict[str, ConceptState]
) -> Dict[str, float]:
    """
    Compute effect score for each image based on concept weights.
    Returns: {image_id: effect_score}
    """
    if not concepts:
        return {img_id: 0.0 for img_id in image_ids}
    
    K = len(concepts)
    w_base = 1.0 / K
    
    effects = {}
    for image_id in image_ids:
        effect = 0.0
        for concept in concepts:
            state = concept_states[concept.id]
            w = state.w
            
            # Get incidence (0 or 1)
            a_ic = incidence_matrix.get(image_id, {}).get(concept.id, 0)
            
            if a_ic > 0:
                w_plus = max(0, w - w_base)
                w_minus = max(0, w_base - w)
                effect += w_plus - GAMMA_NEG * w_minus
        
        effects[image_id] = effect
    
    return effects


# ============================================================================
# Session Management
# ============================================================================
class ConceptRefinementSession:
    """Manages concept refinement state for a stage"""
    
    def __init__(self, session_id: str, stage: str, image_ids: List[str]):
        self.session_id = session_id
        self.stage = stage
        self.image_ids = image_ids
        
        self.raw_tags: List[RawTag] = []
        self.concepts: List[Concept] = []
        self.concept_states: Dict[str, ConceptState] = {}
        self.tag_to_concept: Dict[str, str] = {}  # tag_id -> concept_id
        self.incidence_matrix: Dict[str, Dict[str, int]] = {}  # image_id -> concept_id -> count
        
        self.history: List[Dict] = []  # For undo/redo
        self.initialized = False
        self._explicit_categorization: Optional[Dict] = None  # User-specified categorization
    
    def initialize_from_tags(self, image_tags: Dict[str, List[str]]) -> None:
        """
        Initialize concepts from image tags.
        image_tags: {image_id: [tag1, tag2, ...]}
        """
        print(f"🔧 Initializing concepts for {self.stage} stage...")
        
        # Step 1: Collect and normalize raw tags
        all_texts = []
        tag_metadata = []
        
        for image_id, tags in image_tags.items():
            for tag_index, tag_text in enumerate(tags):
                normalized = normalize_text(tag_text)
                lemmatized = simple_lemmatize(normalized)
                all_texts.append(lemmatized)
                tag_metadata.append({
                    'original': tag_text,
                    'normalized': lemmatized,
                    'image_id': image_id,
                    'tag_index': tag_index  # Per-image index
                })
        
        if not all_texts:
            print("⚠️  No tags found, skipping concept initialization")
            self.initialized = True
            return
        
        # Step 2: Get embeddings in batch
        print(f"📊 Getting embeddings for {len(all_texts)} tags...")
        embeddings = get_batch_embeddings(all_texts)
        
        # Step 3: Create RawTag objects
        for i, (text, embedding, meta) in enumerate(zip(all_texts, embeddings, tag_metadata)):
            tag_id = f"tag_{self.stage}_{meta['image_id']}_{meta['tag_index']}"
            raw_tag = RawTag(
                id=tag_id,
                text=text,
                embedding=embedding.tolist(),
                image_id=meta['image_id']
            )
            self.raw_tags.append(raw_tag)
            print(f"  Created tag: {tag_id} -> '{text}' (image: {meta['image_id']})")
        
        # Step 4: Build concepts using Agglomerative Clustering
        print(f"🔨 Building concepts with Agglomerative Clustering...")
        self.concepts, self.tag_to_concept = build_concepts(self.raw_tags)
        print(f"✅ Created {len(self.concepts)} concepts from {len(self.raw_tags)} tags")
        
        # Step 5: Build incidence matrix
        for image_id in self.image_ids:
            self.incidence_matrix[image_id] = defaultdict(int)
        
        for tag in self.raw_tags:
            if tag.concept_id:
                self.incidence_matrix[tag.image_id][tag.concept_id] += 1
        
        # Step 6: Initialize concept states
        K = len(self.concepts)
        initial_weight = 1.0 / K if K > 0 else 0.0
        
        for concept in self.concepts:
            self.concept_states[concept.id] = ConceptState(
                w=initial_weight
            )
        
        self.initialized = True
        print(f"✅ Concept refinement initialized for {self.stage}")
        
        # Debug logging
        debugger = get_debugger(self.session_id, self.stage)
        debugger.log_initialization(self.raw_tags, self.concepts, self.concept_states)
        
        # Log initial categorization
        categorized = self.get_categorized_concepts()
        debugger.log_categorization(self.concepts, self.concept_states, categorized)
    
    def handle_tag_click(self, tag_id: str, preference: str) -> None:
        """
        Handle like/dislike on a tag -> affects its concept.
        preference: 'positive', 'negative', or 'neutral' (toggle off)
        """
        if tag_id not in self.tag_to_concept:
            print(f"⚠️  Tag {tag_id} not found in concept mapping")
            return
        
        concept_id = self.tag_to_concept[tag_id]
        if concept_id not in self.concept_states:
            print(f"⚠️  Concept {concept_id} not found in states")
            return
        
        state = self.concept_states[concept_id]
        
        # Store before state
        before_state = {
            'like_count': state.like_count,
            'dislike_count': state.dislike_count,
            'w': state.w,
            'score': state.score,
            'liked_tags': state.liked_tags.copy(),
            'disliked_tags': state.disliked_tags.copy()
        }
        
        # Handle toggle logic using per-tag tracking
        action_taken = ""
        if preference == 'positive':
            if tag_id in state.liked_tags:
                # Already liked this tag, toggle off (UNDO)
                state.liked_tags.remove(tag_id)
                state.like_count = max(0, state.like_count - 1)
                action_taken = "UNDO_LIKE"
                print(f"  ✖️ Toggled OFF like for tag {tag_id} (undo)")
            else:
                # Not liked yet, add like
                was_disliked = tag_id in state.disliked_tags
                state.liked_tags.add(tag_id)
                state.like_count += 1
                # Remove from disliked if it was there
                if was_disliked:
                    state.disliked_tags.remove(tag_id)
                    state.dislike_count = max(0, state.dislike_count - 1)
                    action_taken = "REVERSE_TO_LIKE"
                    print(f"  🔄 Reversed from dislike to like for tag {tag_id}")
                else:
                    action_taken = "ADD_LIKE"
                    print(f"  ✔️ Added like for tag {tag_id}")
        elif preference == 'negative':
            if tag_id in state.disliked_tags:
                # Already disliked this tag, toggle off (UNDO)
                state.disliked_tags.remove(tag_id)
                state.dislike_count = max(0, state.dislike_count - 1)
                action_taken = "UNDO_DISLIKE"
                print(f"  ✖️ Toggled OFF dislike for tag {tag_id} (undo)")
            else:
                # Not disliked yet, add dislike
                was_liked = tag_id in state.liked_tags
                state.disliked_tags.add(tag_id)
                state.dislike_count += 1
                # Remove from liked if it was there
                if was_liked:
                    state.liked_tags.remove(tag_id)
                    state.like_count = max(0, state.like_count - 1)
                    action_taken = "REVERSE_TO_DISLIKE"
                    print(f"  🔄 Reversed from like to dislike for tag {tag_id}")
                else:
                    action_taken = "ADD_DISLIKE"
                    print(f"  ✔️ Added dislike for tag {tag_id}")
        
        # Recompute weights
        self.concept_states = compute_weights(
            self.concepts,
            self.concept_states
        )
        
        # Clear explicit categorization - go back to auto mode after tag interaction
        self._explicit_categorization = None
        
        # Store after state
        after_state = {
            'like_count': state.like_count,
            'dislike_count': state.dislike_count,
            'w': state.w,
            'score': state.score,
            'liked_tags': state.liked_tags.copy(),
            'disliked_tags': state.disliked_tags.copy()
        }
        
        # Minimal logging for speed
        delta_w = after_state['w'] - before_state['w']
        print(f"[TAG CLICK] {action_taken} → {concept_id}: "
              f"likes={before_state['like_count']}→{after_state['like_count']}, "
              f"dislikes={before_state['dislike_count']}→{after_state['dislike_count']}, "
              f"Δw={delta_w:+.4f}")
        
        # NOTE: Debug logging disabled for performance during rapid tag interactions
        # Uncomment if detailed interaction logs are needed:
        # debugger = get_debugger(self.session_id, self.stage)
        # debugger.log_tag_interaction(tag_id, preference, concept_id, before_state, after_state)
        # categorized = self.get_categorized_concepts()
        # debugger.log_categorization(self.concepts, self.concept_states, categorized)
    
    def handle_image_selection(self, image_id: str, boost_amount: float = 0.5) -> None:
        """
        Boost weights for all concepts present in the selected image.
        boost_amount: how much to boost (default 0.5 = half a like)
        """
        if image_id not in self.incidence_matrix:
            print(f"⚠️  Image {image_id} not found in incidence matrix")
            return
        
        # Find all concepts in this image
        concepts_in_image = [
            concept_id for concept_id, count in self.incidence_matrix[image_id].items()
            if count > 0
        ]
        
        print(f"\n[IMAGE SELECTION] {image_id}")
        print(f"  Boosting {len(concepts_in_image)} concepts by {boost_amount}")
        
        # Boost like count for each concept (fractional boost)
        for concept_id in concepts_in_image:
            if concept_id in self.concept_states:
                state = self.concept_states[concept_id]
                # Add fractional boost to like count
                state.like_count += boost_amount
                
                # Find concept label
                concept = next((c for c in self.concepts if c.id == concept_id), None)
                if concept:
                    print(f"    {concept.label}: like_count {state.like_count - boost_amount:.2f} -> {state.like_count:.2f}")
        
        # Recompute weights
        self.concept_states = compute_weights(
            self.concepts,
            self.concept_states
        )
        
        # Clear explicit categorization - go back to auto mode after image selection
        self._explicit_categorization = None
        
        # Debug logging
        debugger = get_debugger(self.session_id, self.stage)
        debugger.log_event('image_selection', {
            'image_id': image_id,
            'boost_amount': boost_amount,
            'concepts_boosted': concepts_in_image,
            'concept_count': len(concepts_in_image)
        })
        
        # Log new categorization (will use auto-categorization now)
        categorized = self.get_categorized_concepts()
        debugger.log_categorization(self.concepts, self.concept_states, categorized)
    
    def update_rankings(self, positive_ids: List[str], negative_ids: List[str]) -> None:
        """Update concept rankings - respects explicit user list placement"""
        update_rank_bonuses(positive_ids, negative_ids, self.concept_states)
        
        # Collect rank bonuses/penalties for logging
        rank_bonuses = {cid: self.concept_states[cid].rank_bonus 
                       for cid in positive_ids if cid in self.concept_states}
        rank_penalties = {cid: self.concept_states[cid].rank_penalty 
                         for cid in negative_ids if cid in self.concept_states}
        
        # Debug logging
        debugger = get_debugger(self.session_id, self.stage)
        debugger.log_ranking_update(positive_ids, negative_ids, rank_bonuses, rank_penalties)
        
        # Recompute weights
        self.concept_states = compute_weights(
            self.concepts,
            self.concept_states
        )
        
        # IMPORTANT: Store explicit categorization without auto-recategorization
        # The user has explicitly placed concepts in these lists, so respect that
        # Neutral list contains all concepts not in positive or negative
        all_concept_ids = {c.id for c in self.concepts}
        explicit_ids = set(positive_ids) | set(negative_ids)
        neutral_ids = list(all_concept_ids - explicit_ids)
        
        # Cache the explicit categorization (used by get_categorized_concepts)
        self._explicit_categorization = {
            'positive': positive_ids,
            'neutral': neutral_ids,
            'negative': negative_ids
        }
        
        print(f"\n[RANKING UPDATE] Explicit categorization stored:")
        print(f"  Positive: {len(positive_ids)} concepts")
        print(f"  Neutral: {len(neutral_ids)} concepts")
        print(f"  Negative: {len(negative_ids)} concepts")
        
        # Log the explicit categorization
        debugger.log_categorization(self.concepts, self.concept_states, self._explicit_categorization)
    
    def get_categorized_concepts(self) -> Dict:
        """Get concepts categorized into positive/neutral/negative"""
        # If we have explicit categorization from ranking update, use that
        if hasattr(self, '_explicit_categorization') and self._explicit_categorization:
            print(f"[GET_CATEGORIZED] Using explicit categorization")
            return self._explicit_categorization
        
        # Otherwise, use auto-categorization based on weights
        print(f"[GET_CATEGORIZED] Using auto-categorization")
        positive, neutral, negative = categorize_concepts(
            self.concepts,
            self.concept_states
        )
        
        return {
            'positive': positive,
            'neutral': neutral,
            'negative': negative
        }

    def get_current_weights_for_pbo(self) -> np.ndarray:
        """
        Get current UI weights as numpy array for PBO stabilization.

        Returns:
            weights: np.ndarray of shape (K,) with current w values
        """
        if not self.concepts:
            return np.array([])

        K = len(self.concepts)
        weights = np.zeros(K, dtype=np.float32)

        for i, concept in enumerate(self.concepts):
            state = self.concept_states.get(concept.id)
            if state:
                weights[i] = state.w

        # Ensure simplex (should already be, but safety check)
        w_sum = weights.sum()
        if w_sum > 1e-8:
            weights = weights / w_sum
        else:
            weights = np.ones(K, dtype=np.float32) / K

        return weights

    def get_negative_concept_ids(self) -> set:
        """
        Get concept IDs that user has expressed negative preference for.

        Returns:
            set of concept IDs where dislike_count > like_count
        """
        negatives = set()

        for concept in self.concepts:
            state = self.concept_states.get(concept.id)
            if state and state.dislike_count > state.like_count:
                negatives.add(concept.id)

        return negatives
    
    def get_image_effects(self) -> Dict[str, float]:
        """Get effect scores for all images"""
        return compute_image_effects(
            self.image_ids,
            self.incidence_matrix,
            self.concepts,
            self.concept_states
        )
    
    def to_dict(self) -> Dict:
        """Serialize to dict for API response"""
        concepts_output = []
        for concept in self.concepts:
            # Get all matching tags
            matching_tags = [tag for tag in self.raw_tags if tag.id in concept.member_tag_ids]
            member_tag_texts = [tag.text for tag in matching_tags]
            unique_texts = list(dict.fromkeys(member_tag_texts))
            
            # Debug: log if we find unexpected duplicates
            if len(member_tag_texts) != len(unique_texts):
                print(f"[DEDUP WARNING] Concept '{concept.label}':")
                print(f"  member_tag_ids: {concept.member_tag_ids}")
                print(f"  Found {len(matching_tags)} tags: {[f'{t.id}={t.text}' for t in matching_tags]}")
                print(f"  Texts: {member_tag_texts}")
                print(f"  Unique: {unique_texts}")
            
            concepts_output.append({
                **asdict(concept),
                'state': self._serialize_concept_state(self.concept_states.get(concept.id, ConceptState())),
                'member_tags': unique_texts
            })
        
        return {
            'session_id': self.session_id,
            'stage': self.stage,
            'concepts': concepts_output,
            'categorized': self.get_categorized_concepts(),
            'image_effects': self.get_image_effects(),
            'incidence_matrix': {
                img_id: dict(concepts) 
                for img_id, concepts in self.incidence_matrix.items()
            },
            'tag_preferences': self.get_tag_preferences()
        }
    
    def _serialize_concept_state(self, state: ConceptState) -> Dict:
        """Serialize ConceptState to dict, converting sets to lists"""
        return {
            'like_count': state.like_count,
            'dislike_count': state.dislike_count,
            'rank_bonus': state.rank_bonus,
            'rank_penalty': state.rank_penalty,
            'score': state.score,
            'w': state.w,
            'liked_tags': list(state.liked_tags) if state.liked_tags else [],
            'disliked_tags': list(state.disliked_tags) if state.disliked_tags else []
        }
    
    def get_tag_preferences(self) -> Dict[str, str]:
        """
        Get preference status for all tags.
        Returns: { tag_id: 'positive' | 'negative' | None }
        """
        print(f"\n[GET_TAG_PREFS] Getting preferences for {len(self.raw_tags)} tags")
        
        # Debug: show which concepts have liked/disliked tags
        for concept_id, state in self.concept_states.items():
            if state.liked_tags or state.disliked_tags:
                print(f"  Concept {concept_id}: liked={state.liked_tags}, disliked={state.disliked_tags}")
        
        tag_prefs = {}
        for tag in self.raw_tags:
            concept_id = tag.concept_id
            if concept_id in self.concept_states:
                state = self.concept_states[concept_id]
                if tag.id in state.liked_tags:
                    tag_prefs[tag.id] = 'positive'
                    print(f"  ✓ Tag {tag.id} -> positive")
                elif tag.id in state.disliked_tags:
                    tag_prefs[tag.id] = 'negative'
                    print(f"  ✓ Tag {tag.id} -> negative")
                else:
                    tag_prefs[tag.id] = None
            else:
                tag_prefs[tag.id] = None
        
        positive_count = sum(1 for v in tag_prefs.values() if v == 'positive')
        negative_count = sum(1 for v in tag_prefs.values() if v == 'negative')
        null_count = sum(1 for v in tag_prefs.values() if v is None)
        
        print(f"[GET_TAG_PREFS] Result: {positive_count} positive, {negative_count} negative, {null_count} null")
        
        return tag_prefs
    
    def save_concept_weights(self, session_folder: str) -> None:
        """
        Save learned concept weights to disk for warm-starting refinement stages.
        Saves to: <session_folder>/<stage>/concept_weights.json
        """
        import os
        
        if not self.initialized:
            print(f"[SAVE_WEIGHTS] Session not initialized, skipping save")
            return
        
        stage_folder = os.path.join(session_folder, self.stage)
        os.makedirs(stage_folder, exist_ok=True)
        
        weights_file = os.path.join(stage_folder, "concept_weights.json")
        
        # Prepare weights data
        weights_data = {
            'stage': self.stage,
            'session_id': self.session_id,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'num_concepts': len(self.concepts),
            'concept_weights': []
        }
        
        for concept in self.concepts:
            state = self.concept_states.get(concept.id, ConceptState())
            weights_data['concept_weights'].append({
                'concept_id': concept.id,
                'label': concept.label,
                'weight': state.w,
                'score': state.score,
                'like_count': state.like_count,
                'dislike_count': state.dislike_count,
                'member_tag_ids': concept.member_tag_ids
            })
        
        # Sort by weight descending for readability
        weights_data['concept_weights'].sort(key=lambda x: x['weight'], reverse=True)
        
        with open(weights_file, 'w') as f:
            __import__('json').dump(weights_data, f, indent=2)
        
        print(f"[SAVE_WEIGHTS] ✅ Saved {len(self.concepts)} concept weights to {weights_file}")
        # Create sample list outside f-string to avoid backslash syntax error
        sample_weights = [f"{c['label']}: {c['weight']:.3f}" for c in weights_data['concept_weights'][:5]]
        print(f"[SAVE_WEIGHTS] Top 5 by weight: {sample_weights}")
    
    def load_concept_weights_from_base_stage(self, session_folder: str) -> bool:
        """
        Load learned weights from the base stage (e.g., impression → impression_refinement).
        Returns True if weights were loaded, False otherwise.
        """
        import os
        
        # Extract base stage name (remove _refinement suffix)
        if not self.stage.endswith('_refinement'):
            print(f"[LOAD_WEIGHTS] Stage '{self.stage}' is not a refinement stage, skipping")
            return False
        
        base_stage = self.stage.replace('_refinement', '')
        base_stage_folder = os.path.join(session_folder, base_stage)
        weights_file = os.path.join(base_stage_folder, "concept_weights.json")
        
        if not os.path.exists(weights_file):
            print(f"[LOAD_WEIGHTS] No weights file found at {weights_file}, starting with uniform weights")
            return False
        
        try:
            with open(weights_file, 'r') as f:
                weights_data = __import__('json').load(f)
            
            print(f"[LOAD_WEIGHTS] 📂 Loading {len(weights_data['concept_weights'])} weights from {base_stage}")
            
            # Create a mapping: label → weight data
            weight_map = {w['label']: w for w in weights_data['concept_weights']}
            
            # Apply weights to matching concepts (by label)
            matched_count = 0
            for concept in self.concepts:
                if concept.label in weight_map:
                    prev_weight_data = weight_map[concept.label]
                    state = self.concept_states.get(concept.id, ConceptState())
                    
                    # Transfer learned weights
                    state.w = prev_weight_data.get('weight', state.w)
                    state.score = prev_weight_data.get('score', state.score)
                    
                    self.concept_states[concept.id] = state
                    matched_count += 1
                    
                    print(f"[LOAD_WEIGHTS]   ✓ {concept.label}: w={state.w:.3f}")
            
            print(f"[LOAD_WEIGHTS] ✅ Loaded weights for {matched_count}/{len(self.concepts)} concepts from {base_stage}")
            
            if matched_count < len(self.concepts):
                print(f"[LOAD_WEIGHTS] ⚠️  {len(self.concepts) - matched_count} concepts have no previous weights (will use computed values)")
            
            return matched_count > 0
            
        except Exception as e:
            print(f"[LOAD_WEIGHTS] ❌ Error loading weights: {e}")
            return False


# Global session store
refinement_sessions: Dict[str, ConceptRefinementSession] = {}


def get_or_create_session(session_id: str, stage: str, image_ids: List[str]) -> ConceptRefinementSession:
    """Get existing or create new refinement session"""
    key = f"{session_id}_{stage}"
    
    if key not in refinement_sessions:
        refinement_sessions[key] = ConceptRefinementSession(session_id, stage, image_ids)
    
    return refinement_sessions[key]

