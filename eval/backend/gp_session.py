"""
GP Exploration Session Wrapper

Bridges the AdaptivePreferenceSystem (GP-based preference learning) to the existing
eval_server interface, providing compatibility with ConceptRefinementSession patterns.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Import GP system
from exploration_GP import (
    AdaptivePreferenceSystem,
    RawTag,
    InteractionRound,
    DisplayConcept
)


class GPExplorationSession:
    """
    Wrapper class that bridges the GP-based AdaptivePreferenceSystem to the 
    existing eval_server interface.
    
    Provides the same interface as ConceptRefinementSession but uses GP for
    preference learning instead of softmax on like/dislike counts.
    """
    
    def __init__(
        self,
        session_id: str,
        stage: str,
        image_ids: List[str],
        embedding_dim: int = 768,
        n_inducing: int = 64,
        device: str = None
    ):
        self.session_id = session_id
        self.stage = stage
        self.image_ids = image_ids
        
        # Initialize GP system
        self.gp_system = AdaptivePreferenceSystem(
            embedding_dim=embedding_dim,
            n_inducing=n_inducing,
            device=device
        )
        
        # Track raw tags by ID for lookups
        self.raw_tags: Dict[str, RawTag] = {}
        self.tag_to_image: Dict[str, str] = {}  # tag_id -> image_id
        
        # Track current interaction state
        self.tag_states: Dict[str, str] = {}  # tag_id -> 'liked'/'neutral'/'disliked'
        self.selected_image_id: Optional[str] = None
        
        # For building InteractionRound
        self.image_tags: Dict[str, List[RawTag]] = {}  # image_id -> [RawTag, ...]
        
        self.initialized = False
        self._clip_model = None
        self._clip_device = None
    
    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None:
            import torch
            import clip
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, _ = clip.load("ViT-L/14", device=self._clip_device)
            print(f"[GPSession] Loaded CLIP ViT-L/14 on {self._clip_device}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get CLIP embedding for text."""
        self._load_clip()
        import torch
        import clip
        
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self._clip_device)
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()[0].astype(np.float32)
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get CLIP embeddings for multiple texts."""
        self._load_clip()
        import torch
        import clip
        
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self._clip_device)
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy().astype(np.float32)
            return [emb for emb in embeddings]
    
    def initialize_from_tags(self, image_tags: Dict[str, List[str]]) -> None:
        """
        Initialize session from image tags.
        
        Args:
            image_tags: {image_id: [tag_text1, tag_text2, ...]}
        """
        print(f"[GPSession] Initializing GP exploration for {self.stage} stage...")
        
        # Collect all tag texts
        all_texts = []
        tag_metadata = []
        
        for image_id, tags in image_tags.items():
            for tag_idx, tag_text in enumerate(tags):
                all_texts.append(tag_text.strip())
                tag_metadata.append({
                    'text': tag_text.strip(),
                    'image_id': image_id,
                    'tag_idx': tag_idx
                })
        
        if not all_texts:
            print("[GPSession] No tags found, skipping initialization")
            self.initialized = True
            return
        
        # Get embeddings in batch
        print(f"[GPSession] Getting embeddings for {len(all_texts)} tags...")
        embeddings = self._get_batch_embeddings(all_texts)
        
        # Create RawTag objects
        for i, (text, embedding, meta) in enumerate(zip(all_texts, embeddings, tag_metadata)):
            tag_id = f"tag_{self.stage}_{meta['image_id']}_{meta['tag_idx']}"
            image_idx = self.image_ids.index(meta['image_id']) if meta['image_id'] in self.image_ids else i
            
            raw_tag = RawTag(
                id=tag_id,
                text=text,
                embedding=embedding,
                source_image_idx=image_idx
            )
            
            self.raw_tags[tag_id] = raw_tag
            self.tag_to_image[tag_id] = meta['image_id']
            self.tag_states[tag_id] = 'neutral'
            
            # Store by image
            if meta['image_id'] not in self.image_tags:
                self.image_tags[meta['image_id']] = []
            self.image_tags[meta['image_id']].append(raw_tag)
        
        # Add all tags to GP system (without interaction)
        for tag in self.raw_tags.values():
            self.gp_system.all_tags[tag.id] = tag
        
        self.initialized = True
        print(f"[GPSession] Initialized with {len(self.raw_tags)} tags across {len(self.image_tags)} images")
    
    def handle_tag_click(self, tag_id: str, preference: str) -> None:
        """
        Handle like/dislike on a tag.
        
        Args:
            tag_id: The tag ID
            preference: 'positive', 'negative', or 'neutral'
        """
        if tag_id not in self.raw_tags:
            print(f"[GPSession] Warning: Tag {tag_id} not found")
            return
        
        # Map preference to GP format
        pref_map = {
            'positive': 'liked',
            'negative': 'disliked',
            'neutral': 'neutral'
        }
        
        # Toggle logic - if already in state, toggle off
        current_state = self.tag_states.get(tag_id, 'neutral')
        new_gp_state = pref_map.get(preference, 'neutral')
        
        if current_state == new_gp_state:
            # Toggle off
            self.tag_states[tag_id] = 'neutral'
            print(f"[GPSession] Tag {tag_id}: {current_state} -> neutral (toggle off)")
        else:
            self.tag_states[tag_id] = new_gp_state
            print(f"[GPSession] Tag {tag_id}: {current_state} -> {new_gp_state}")
    
    def handle_image_selection(self, image_id: str, boost_amount: float = 0.5) -> None:
        """
        Handle image selection.
        
        Args:
            image_id: Selected image ID
            boost_amount: Not used in GP mode, kept for interface compatibility
        """
        self.selected_image_id = image_id
        print(f"[GPSession] Selected image: {image_id}")
        
        # Process the interaction round
        self._process_interaction_round()
    
    def _process_interaction_round(self) -> List[DisplayConcept]:
        """
        Process current tag states as an interaction round.
        
        Creates an InteractionRound from current state and processes through GP.
        """
        if not self.image_ids:
            return []
        
        # Build images list for InteractionRound
        images = []
        for image_id in self.image_ids:
            if image_id in self.image_tags:
                images.append(self.image_tags[image_id])
            else:
                images.append([])
        
        # Determine selected index
        selected_idx = 0
        if self.selected_image_id and self.selected_image_id in self.image_ids:
            selected_idx = self.image_ids.index(self.selected_image_id)
        
        # Create InteractionRound
        interaction = InteractionRound(
            images=images,
            selected_image_idx=selected_idx,
            tag_states=self.tag_states.copy()
        )
        
        # Process through GP system
        concepts = self.gp_system.process_interaction(interaction, refit=True, verbose=True)
        
        print(f"[GPSession] Processed interaction round, got {len(concepts)} concepts")
        return concepts
    
    def get_concepts(self) -> List[DisplayConcept]:
        """Get current concepts."""
        return self.gp_system.get_concepts()
    
    def get_categorized_concepts(self) -> Dict[str, List[str]]:
        """Get concepts categorized into positive/neutral/negative."""
        concepts = self.get_concepts()
        
        positive = [c.id for c in concepts if c.category == 'positive']
        negative = [c.id for c in concepts if c.category == 'negative']
        neutral = [c.id for c in concepts if c.category == 'neutral']
        
        return {
            'positive': positive,
            'neutral': neutral,
            'negative': negative
        }
    
    def get_image_effects(self) -> Dict[str, float]:
        """Get effect scores for images (not tracked in GP mode)."""
        return {img_id: 0.0 for img_id in self.image_ids}
    
    def get_tag_preferences(self) -> Dict[str, Optional[str]]:
        """Get preference status for all tags."""
        prefs = {}
        for tag_id, state in self.tag_states.items():
            if state == 'liked':
                prefs[tag_id] = 'positive'
            elif state == 'disliked':
                prefs[tag_id] = 'negative'
            else:
                prefs[tag_id] = None
        return prefs
    
    def save_concept_weights(self, session_folder: str) -> str:
        """
        Save learned concept weights to disk.
        
        Args:
            session_folder: Path to the session folder
        
        Returns:
            Path to the saved file
        """
        return self.gp_system.save_concept_weights(session_folder, self.stage)
    
    def save_raw_tag_weights(
        self,
        session_folder: str,
        k: int = 10,
        min_cos_distance: float = 0.15
    ) -> str:
        """
        Save top-K raw tag weights to disk (bypasses clustering).
        
        This saves individual tag weights based on GP utilities,
        with cosine deduplication to avoid redundant tags.
        
        Args:
            session_folder: Path to the session folder
            k: Number of top tags to save
            min_cos_distance: Minimum cosine distance for deduplication
        
        Returns:
            Path to the saved file
        """
        return self.gp_system.save_raw_tag_weights(
            session_folder, 
            self.stage,
            k=k,
            min_cos_distance=min_cos_distance
        )
    
    def get_top_k_tags_for_generation(
        self,
        k: int = 10,
        min_cos_distance: float = 0.15
    ):
        """
        Get top-K tags by GP utility with cosine deduplication.
        
        Returns:
            (tag_texts, weights, tag_details)
        """
        return self.gp_system.get_top_k_tags_for_generation(
            k=k,
            min_cos_distance=min_cos_distance
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for API response."""
        base_dict = self.gp_system.to_dict()
        base_dict['session_id'] = self.session_id
        base_dict['stage'] = self.stage
        base_dict['tag_preferences'] = self.get_tag_preferences()
        
        # Build incidence matrix from image_tags
        incidence_matrix = {}
        concepts = self.get_concepts()
        concept_labels = {c.representative_tag: c.id for c in concepts}
        
        for image_id, tags in self.image_tags.items():
            incidence_matrix[image_id] = {}
            for tag in tags:
                # Find which concept this tag belongs to
                for concept in concepts:
                    if tag.text in concept.member_tags:
                        if concept.id not in incidence_matrix[image_id]:
                            incidence_matrix[image_id][concept.id] = 0
                        incidence_matrix[image_id][concept.id] += 1
                        break
        
        base_dict['incidence_matrix'] = incidence_matrix
        
        return base_dict
    
    def process_round_manually(self) -> List[DisplayConcept]:
        """
        Manually trigger processing of current interaction state.
        
        Call this when user explicitly wants to update the GP model
        without selecting an image.
        """
        return self._process_interaction_round()


# Global session store for GP sessions
gp_exploration_sessions: Dict[str, GPExplorationSession] = {}


def get_or_create_gp_session(
    session_id: str,
    stage: str,
    image_ids: List[str]
) -> GPExplorationSession:
    """Get existing or create new GP exploration session."""
    key = f"{session_id}_{stage}"
    
    if key not in gp_exploration_sessions:
        gp_exploration_sessions[key] = GPExplorationSession(
            session_id=session_id,
            stage=stage,
            image_ids=image_ids
        )
    
    return gp_exploration_sessions[key]


def clear_gp_session(session_id: str, stage: str) -> None:
    """Clear a GP session from memory."""
    key = f"{session_id}_{stage}"
    if key in gp_exploration_sessions:
        del gp_exploration_sessions[key]
        print(f"[GPSession] Cleared session {key}")

