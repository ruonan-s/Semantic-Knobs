"""
Adaptive Preference Learning System for Visual Tag Exploration

Key principle: Learn preferences on raw embeddings, cluster only for UI display.
The GP naturally handles "soft clustering" through the kernel.
"""

import numpy as np
import torch
import gpytorch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RawTag:
    """A single visual tag extracted from an image."""
    id: str
    text: str
    embedding: np.ndarray  # CLIP embedding (768-dim for ViT-L/14)
    source_image_idx: int

@dataclass 
class PreferencePair:
    """A preference: embedding_a is preferred over embedding_b."""
    embedding_a: np.ndarray
    embedding_b: np.ndarray
    strength: float = 1.0

@dataclass
class InteractionRound:
    """Data from one round of user interaction."""
    images: List[List[RawTag]]  # 4 images, each with ~12 tags
    selected_image_idx: int
    tag_states: Dict[str, str]  # tag_id -> 'liked'/'neutral'/'disliked'

@dataclass
class DisplayConcept:
    """A concept cluster for UI display (created after learning)."""
    id: str
    representative_tag: str
    member_tags: List[str]
    centroid: np.ndarray
    mean_utility: float
    utility_std: float
    category: str  # 'positive'/'neutral'/'negative'


# ============================================================================
# PREFERENCE EXTRACTION
# ============================================================================

class PreferenceExtractor:
    """
    Extract preference pairs from EXPLICIT user clicks only.
    
    Only creates pairs from explicit clicks, with weak neutral constraints:
    - Liked > Disliked (strongest signal)
    - Liked in selected image > Liked in other images (weak signal)
    - Liked > Neutral (very weak, optional)
    - Neutral > Disliked (very weak, optional)
    """
    
    def __init__(
        self,
        strength_liked_vs_disliked: float = 1.0,
        strength_liked_selected_vs_liked_other: float = 0.3,
        strength_liked_vs_neutral: float = 0.15,
        strength_neutral_vs_disliked: float = 0.10,
    ):
        self.strengths = {
            'liked_vs_disliked': strength_liked_vs_disliked,
            'liked_selected_vs_liked_other': strength_liked_selected_vs_liked_other,
            'liked_vs_neutral': strength_liked_vs_neutral,
            'neutral_vs_disliked': strength_neutral_vs_disliked,
        }
    
    def extract(self, interaction: InteractionRound) -> List[PreferencePair]:
        """
        Extract preference pairs from explicit user clicks only.
        
        Only creates pairs between:
        - Explicitly liked tags > Explicitly disliked tags
        - Liked in selected image > Liked in other images (if user clicked both)
        
        Weakly constrains:
        - liked > neutral
        - neutral > disliked
        """
        pairs = []
        selected_idx = interaction.selected_image_idx
        
        # Categorize ONLY explicitly clicked tags
        liked_selected = []  # Liked tags in selected image
        liked_other = []     # Liked tags in non-selected images
        disliked_all = []    # All disliked tags
        neutral_all = []     # All neutral tags
        
        for img_idx, tags in enumerate(interaction.images):
            is_selected = (img_idx == selected_idx)
            for tag in tags:
                state = interaction.tag_states.get(tag.id, 'neutral')
                emb = tag.embedding
                
                if state == 'liked':
                    if is_selected:
                        liked_selected.append(emb)
                    else:
                        liked_other.append(emb)
                elif state == 'disliked':
                    disliked_all.append(emb)
                else:
                    neutral_all.append(emb)
        
        all_liked = liked_selected + liked_other
        
        # === ONLY EXPLICIT CLICK PAIRS ===
        
        # 1. Liked > Disliked (main signal from explicit feedback)
        for a in all_liked:
            for b in disliked_all:
                pairs.append(PreferencePair(
                    a, b, self.strengths['liked_vs_disliked']
                ))
        
        # 2. Liked in selected > Liked in other (weak signal)
        #    Only if user explicitly liked tags in both selected and other images
        if liked_selected and liked_other:
            for a in liked_selected:
                for b in liked_other:
                    pairs.append(PreferencePair(
                        a, b, self.strengths['liked_selected_vs_liked_other']
                    ))

        # 3. Weak constraint: liked > neutral
        if all_liked and neutral_all and self.strengths['liked_vs_neutral'] > 0:
            for a in all_liked:
                for b in neutral_all:
                    pairs.append(PreferencePair(
                        a, b, self.strengths['liked_vs_neutral']
                    ))

        # 4. Weak constraint: neutral > disliked
        if neutral_all and disliked_all and self.strengths['neutral_vs_disliked'] > 0:
            for a in neutral_all:
                for b in disliked_all:
                    pairs.append(PreferencePair(
                        a, b, self.strengths['neutral_vs_disliked']
                    ))
        
        return pairs
    
    def get_pair_statistics(self, pairs: List[PreferencePair]) -> Dict:
        """Get statistics about extracted pairs for debugging."""
        if not pairs:
            return {'total': 0}
        
        strengths = [p.strength for p in pairs]
        return {
            'total': len(pairs),
            'mean_strength': np.mean(strengths),
            'max_strength': np.max(strengths),
            'strength_distribution': {
                s: sum(1 for p in pairs if p.strength == s)
                for s in set(strengths)
            }
        }


# ============================================================================
# GAUSSIAN PROCESS PREFERENCE MODEL
# ============================================================================

class PreferenceGP(gpytorch.models.ApproximateGP):
    """Variational GP for learning utility function over embedding space."""
    
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
        )
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class PreferenceLearner:
    """
    Learn user preferences directly in CLIP embedding space via GP.
    
    Models a latent utility function f(e) where preferences imply:
    f(e_preferred) > f(e_less_preferred)
    
    Uses probit likelihood: P(a > b) = Φ((f(a) - f(b)) / (√2 * σ))
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,  # CLIP ViT-L/14 dimension
        n_inducing: int = 64,
        noise_scale: float = 1.0,
        device: str = None
    ):
        self.embedding_dim = embedding_dim
        self.n_inducing = n_inducing
        self.noise_scale = noise_scale
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.preference_pairs: List[PreferencePair] = []
        self.all_embeddings: List[np.ndarray] = []
        
        self.model: Optional[PreferenceGP] = None
        self.is_fitted = False
        
    def add_preferences(self, pairs: List[PreferencePair]):
        """Add preference pairs from new interaction."""
        self.preference_pairs.extend(pairs)
        for pair in pairs:
            self.all_embeddings.append(pair.embedding_a)
            self.all_embeddings.append(pair.embedding_b)
        self.is_fitted = False
    
    def clear(self):
        """Clear all learned preferences."""
        self.preference_pairs = []
        self.all_embeddings = []
        self.model = None
        self.is_fitted = False
    
    def _select_inducing_points(self) -> torch.Tensor:
        """Select diverse inducing points via k-means++ initialization."""
        all_emb = np.stack(self.all_embeddings)
        
        if len(all_emb) <= self.n_inducing:
            points = all_emb
        else:
            # K-means++ selection for diversity
            n_points = all_emb.shape[0]
            selected = [np.random.randint(n_points)]
            
            for _ in range(self.n_inducing - 1):
                selected_pts = all_emb[selected]
                dists = np.min(
                    np.linalg.norm(
                        all_emb[:, None] - selected_pts[None, :], 
                        axis=2
                    ),
                    axis=1
                )
                probs = dists ** 2
                prob_sum = probs.sum()
                
                # Handle edge case where all distances are zero (identical points)
                if prob_sum < 1e-10:
                    # Fall back to uniform random selection
                    remaining = list(set(range(n_points)) - set(selected))
                    if remaining:
                        selected.append(np.random.choice(remaining))
                    else:
                        # All points already selected (shouldn't happen)
                        break
                else:
                    probs /= prob_sum
                    selected.append(np.random.choice(n_points, p=probs))
            
            points = all_emb[selected]
        
        return torch.tensor(points, dtype=torch.float32, device=self.device)
    
    def fit(self, n_epochs: int = 100, lr: float = 0.01, verbose: bool = False):
        """Fit GP model to accumulated preferences."""
        n_pairs = len(self.preference_pairs)
        if n_pairs < 3:
            if verbose:
                print(f"Only {n_pairs} pairs - need at least 3")
            return
        
        # Prepare tensors
        emb_a = torch.tensor(
            np.stack([p.embedding_a for p in self.preference_pairs]),
            dtype=torch.float32, device=self.device
        )
        emb_b = torch.tensor(
            np.stack([p.embedding_b for p in self.preference_pairs]),
            dtype=torch.float32, device=self.device
        )
        strengths = torch.tensor(
            [p.strength for p in self.preference_pairs],
            dtype=torch.float32, device=self.device
        )
        
        # Initialize model
        inducing_points = self._select_inducing_points()
        self.model = PreferenceGP(inducing_points).to(self.device)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        
        sqrt2 = np.sqrt(2)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output_a = self.model(emb_a)
            output_b = self.model(emb_b)
            
            # Probit likelihood for preference
            diff_mean = output_a.mean - output_b.mean
            z = diff_mean / (self.noise_scale * sqrt2)
            log_prob = torch.distributions.Normal(0, 1).cdf(z).clamp(1e-6, 1-1e-6).log()
            
            # Weighted NLL + KL divergence
            nll = -(log_prob * strengths).sum() / strengths.sum()
            kl = self.model.variational_strategy.kl_divergence().sum() / n_pairs
            loss = nll + kl
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and epoch % 25 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f} NLL={nll.item():.4f} KL={kl.item():.4f}")
        
        self.model.eval()
        self.is_fitted = True
        
        if verbose:
            print(f"Fitted on {n_pairs} preference pairs")
    
    def predict_utility(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict utility and uncertainty for embeddings.
        
        Returns: (mean_utility, std_uncertainty)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        if not self.is_fitted or self.model is None:
            n = embeddings.shape[0]
            return np.zeros(n), np.ones(n) * 10.0
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            output = self.model(x)
            mean = output.mean.cpu().numpy()
            std = output.variance.sqrt().cpu().numpy()
        
        return mean, std
    
    def predict_preference_prob(
        self, 
        embedding_a: np.ndarray, 
        embedding_b: np.ndarray
    ) -> float:
        """Predict P(a preferred over b)."""
        mean_a, var_a = self.predict_utility(embedding_a.reshape(1, -1))
        mean_b, var_b = self.predict_utility(embedding_b.reshape(1, -1))
        
        diff_mean = mean_a[0] - mean_b[0]
        diff_std = np.sqrt(var_a[0]**2 + var_b[0]**2 + 2 * self.noise_scale**2)
        
        from scipy.stats import norm
        return float(norm.cdf(diff_mean / diff_std))
    
    def get_kernel_lengthscales(self) -> Optional[np.ndarray]:
        """Get learned ARD lengthscales (indicates dimension importance)."""
        if not self.is_fitted or self.model is None:
            return None
        with torch.no_grad():
            ls = self.model.covar_module.base_kernel.lengthscale.cpu().numpy()
        return ls.flatten()


# ============================================================================
# PREFERENCE-INFORMED CLUSTERING (For UI)
# ============================================================================

class PreferenceInformedClusterer:
    """
    Cluster tags AFTER learning preferences.
    
    Distance combines:
    - Semantic similarity (CLIP embedding distance)
    - Utility similarity (similar preference = closer)
    - Uncertainty similarity (uncertain things together)
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.5,
        utility_weight: float = 0.3,
        uncertainty_weight: float = 0.2,
        distance_threshold: float = 0.35,
        min_clusters: int = 3,
        max_clusters: int = 25
    ):
        self.semantic_weight = semantic_weight
        self.utility_weight = utility_weight
        self.uncertainty_weight = uncertainty_weight
        self.distance_threshold = distance_threshold
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
    
    def _compute_distance_matrix(
        self,
        embeddings: np.ndarray,
        utilities: np.ndarray,
        uncertainties: np.ndarray
    ) -> np.ndarray:
        """Compute preference-informed pairwise distances."""
        # Semantic distance (cosine) - clamp to [0, 2] range
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-8)
        cosine_sim = normalized @ normalized.T
        # Clamp cosine similarity to [-1, 1] to avoid floating point issues
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        semantic_dist = 1 - cosine_sim
        
        # Utility distance (normalized)
        u_range = utilities.max() - utilities.min() + 1e-8
        utility_dist = np.abs(utilities[:, None] - utilities[None, :]) / u_range
        
        # Uncertainty distance (normalized)
        unc_range = uncertainties.max() - uncertainties.min() + 1e-8
        unc_dist = np.abs(uncertainties[:, None] - uncertainties[None, :]) / unc_range
        
        # Combined - ensure non-negative
        combined = (
            self.semantic_weight * semantic_dist +
            self.utility_weight * utility_dist +
            self.uncertainty_weight * unc_dist
        )
        return np.maximum(combined, 0)
    
    def cluster(
        self,
        tags: List[RawTag],
        preference_learner: PreferenceLearner
    ) -> List[DisplayConcept]:
        """Cluster tags into display concepts using learned preferences."""
        if len(tags) == 0:
            return []
        
        if len(tags) == 1:
            emb = tags[0].embedding
            u, s = preference_learner.predict_utility(emb.reshape(1, -1))
            return [DisplayConcept(
                id="concept_0",
                representative_tag=tags[0].text,
                member_tags=[tags[0].text],
                centroid=emb / np.linalg.norm(emb),
                mean_utility=float(u[0]),
                utility_std=float(s[0]),
                category='positive' if u[0] > 0.5 else ('negative' if u[0] < -0.5 else 'neutral')
            )]
        
        # Get predictions
        embeddings = np.stack([t.embedding for t in tags])
        utilities, uncertainties = preference_learner.predict_utility(embeddings)
        
        # Compute distances and cluster
        dist_matrix = self._compute_distance_matrix(embeddings, utilities, uncertainties)
        np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is zero
        
        # Clamp negative values to zero (can occur due to floating point errors)
        dist_matrix = np.maximum(dist_matrix, 0)
        
        # Make matrix symmetric (average of upper and lower triangles)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method='average')
        
        # Determine clusters
        labels = fcluster(Z, t=self.distance_threshold, criterion='distance')
        n_clusters = len(np.unique(labels))
        
        if n_clusters < self.min_clusters:
            labels = fcluster(Z, t=self.min_clusters, criterion='maxclust')
        elif n_clusters > self.max_clusters:
            labels = fcluster(Z, t=self.max_clusters, criterion='maxclust')
        
        # Build concepts
        concepts = []
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_tags = [t for t, m in zip(tags, mask) if m]
            cluster_emb = embeddings[mask]
            cluster_util = utilities[mask]
            cluster_unc = uncertainties[mask]
            
            # Centroid (normalized)
            centroid = cluster_emb.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            
            # Representative: highest utility
            rep_idx = np.argmax(cluster_util)
            representative = cluster_tags[rep_idx].text
            
            # Aggregates
            mean_u = float(cluster_util.mean())
            mean_s = float(cluster_unc.mean())
            
            # Category
            if mean_u > 0.5:
                cat = 'positive'
            elif mean_u < -0.5:
                cat = 'negative'
            else:
                cat = 'neutral'
            
            concepts.append(DisplayConcept(
                id=f"concept_{cluster_id}",
                representative_tag=representative,
                member_tags=[t.text for t in cluster_tags],
                centroid=centroid,
                mean_utility=mean_u,
                utility_std=mean_s,
                category=cat
            ))
        
        # Sort by utility
        concepts.sort(key=lambda c: c.mean_utility, reverse=True)
        
        # Re-index
        for i, c in enumerate(concepts):
            c.id = f"concept_{i}"
        
        return concepts


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class AdaptivePreferenceSystem:
    """
    Complete system for learning tag preferences without pre-clustering.
    
    Usage:
        system = AdaptivePreferenceSystem()
        
        # After each user interaction
        concepts = system.process_interaction(interaction)
        
        # Get weights for generation
        weights = system.get_concept_weights()
        
        # Predict for new tags
        predictions = system.predict_for_new_tags(new_tags)
        
        # Get exploration suggestions
        suggestions = system.suggest_exploration(candidates)
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,  # CLIP ViT-L/14 dimension
        n_inducing: int = 64,
        device: str = None
    ):
        self.extractor = PreferenceExtractor()
        self.learner = PreferenceLearner(
            embedding_dim=embedding_dim,
            n_inducing=n_inducing,
            device=device
        )
        self.clusterer = PreferenceInformedClusterer()
        
        self.all_tags: Dict[str, RawTag] = {}
        self.interaction_count = 0
        
    def process_interaction(
        self,
        interaction: InteractionRound,
        refit: bool = True,
        verbose: bool = False
    ) -> List[DisplayConcept]:
        """
        Process user interaction and return updated concepts.
        """
        self.interaction_count += 1
        
        # Store tags
        for image_tags in interaction.images:
            for tag in image_tags:
                self.all_tags[tag.id] = tag
        
        # Extract preferences from EXPLICIT clicks only
        pairs = self.extractor.extract(interaction)
        self.learner.add_preferences(pairs)
        
        # Always log pair extraction for debugging
        stats = self.extractor.get_pair_statistics(pairs)
        n_liked = sum(1 for s in interaction.tag_states.values() if s == 'liked')
        n_disliked = sum(1 for s in interaction.tag_states.values() if s == 'disliked')
        print(f"[GP] Round {self.interaction_count}: {n_liked} liked, {n_disliked} disliked → {stats['total']} pairs")
        
        # Fit model if we have enough pairs
        total_pairs = len(self.learner.preference_pairs)
        if refit and total_pairs >= 3:
            print(f"[GP] Fitting model on {total_pairs} total preference pairs...")
            self.learner.fit(n_epochs=100, verbose=verbose)
            print(f"[GP] Model fitted. is_fitted={self.learner.is_fitted}")
        else:
            print(f"[GP] Not fitting: refit={refit}, total_pairs={total_pairs} (need >= 3)")
        
        # Cluster for display
        return self.clusterer.cluster(list(self.all_tags.values()), self.learner)
    
    def get_concepts(self) -> List[DisplayConcept]:
        """Get current concepts without processing new interaction."""
        return self.clusterer.cluster(list(self.all_tags.values()), self.learner)
    
    def get_concept_weights(self, tau: float = 1.0) -> Dict[str, float]:
        """Get softmax-normalized weights for concepts."""
        concepts = self.get_concepts()
        if not concepts:
            return {}
        
        utilities = np.array([c.mean_utility for c in concepts])
        exp_u = np.exp(utilities / tau)
        weights = exp_u / exp_u.sum()
        
        return {c.id: float(w) for c, w in zip(concepts, weights)}
    
    def predict_for_new_tags(
        self,
        new_tags: List[RawTag]
    ) -> List[Tuple[RawTag, float, float]]:
        """
        Predict preferences for unseen tags.
        
        Returns: [(tag, utility, uncertainty), ...] sorted by utility
        """
        if not new_tags:
            return []
        
        embeddings = np.stack([t.embedding for t in new_tags])
        utilities, uncertainties = self.learner.predict_utility(embeddings)
        
        results = list(zip(new_tags, utilities, uncertainties))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def suggest_exploration(
        self,
        candidates: List[RawTag],
        n: int = 5,
        exploration_weight: float = 0.5
    ) -> List[RawTag]:
        """
        Suggest tags to show user using Upper Confidence Bound.
        
        UCB = utility + exploration_weight * uncertainty
        
        Balances exploitation (show liked things) with exploration
        (show uncertain things to learn more).
        """
        if not candidates:
            return []
        
        embeddings = np.stack([t.embedding for t in candidates])
        utilities, uncertainties = self.learner.predict_utility(embeddings)
        
        ucb = utilities + exploration_weight * uncertainties
        top_idx = np.argsort(ucb)[-n:][::-1]
        
        return [candidates[i] for i in top_idx]
    
    def get_top_k_tags_for_generation(
        self,
        k: int = 10,
        min_cos_distance: float = 0.15,
        min_utility: float = 0.0,
        tau: float = 1.0
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Get top-K tags by GP utility with cosine deduplication.
        
        This method bypasses clustering and directly selects the top tags
        based on their individual GP utilities, with deduplication to avoid
        semantically redundant tags.
        
        Args:
            k: Number of tags to return
            min_cos_distance: Minimum cosine distance between selected tags (0-2 range)
                             Higher = more diverse. 0.15 means cos_sim < 0.85
            min_utility: Minimum utility threshold (skip negative preferences)
            tau: Temperature for softmax normalization
        
        Returns:
            (tag_texts, weights, tag_details) where:
                - tag_texts: List of tag text strings
                - weights: numpy array of softmax-normalized weights
                - tag_details: List of dicts with full tag info
        """
        if not self.all_tags:
            return [], np.array([]), []
        
        tags = list(self.all_tags.values())
        embeddings = np.stack([t.embedding for t in tags])
        utilities, uncertainties = self.learner.predict_utility(embeddings)
        
        # Sort by utility (descending)
        sorted_indices = np.argsort(utilities)[::-1]
        
        selected_indices = []
        selected_embs = []
        
        for idx in sorted_indices:
            if len(selected_indices) >= k:
                break
            
            # Skip if utility is below threshold
            if utilities[idx] < min_utility:
                continue
            
            emb = embeddings[idx]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            
            # Check cosine distance to already selected tags
            is_duplicate = False
            for sel_emb in selected_embs:
                cos_sim = np.dot(emb_norm, sel_emb)
                cos_distance = 1 - cos_sim
                if cos_distance < min_cos_distance:  # Too similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_indices.append(idx)
                selected_embs.append(emb_norm)
        
        if not selected_indices:
            return [], np.array([]), []
        
        # Build output
        tag_texts = [tags[i].text for i in selected_indices]
        selected_utilities = np.array([utilities[i] for i in selected_indices])
        selected_uncertainties = np.array([uncertainties[i] for i in selected_indices])
        
        # Softmax normalization
        exp_u = np.exp(selected_utilities / tau)
        weights = exp_u / (exp_u.sum() + 1e-8)
        
        # Build detailed info
        tag_details = []
        for i, idx in enumerate(selected_indices):
            tag_details.append({
                'tag_id': tags[idx].id,
                'text': tags[idx].text,
                'utility': float(selected_utilities[i]),
                'uncertainty': float(selected_uncertainties[i]),
                'weight': float(weights[i]),
                'source_image_idx': tags[idx].source_image_idx
            })
        
        return tag_texts, weights, tag_details
    
    def _get_all_positive_tags_with_weights(
        self,
        min_cos_distance: float = 0.15,
        tau: float = 1.0,
        min_utility: float = 0.3  # Threshold for "positive" based on GP utility
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Get ALL tags with positive GP utility, with learned weights.
        
        Unlike get_top_k_tags_for_generation which limits to top-K,
        this returns ALL positive-utility tags to preserve the full user preference.
        
        Args:
            min_cos_distance: Minimum cosine distance for deduplication
            tau: Temperature for softmax normalization
            min_utility: Minimum GP utility to be considered "positive"
        
        Returns:
            (tag_texts, weights, tag_details)
        """
        if not self.all_tags:
            return [], np.array([]), []
        
        # Get all tags and their GP utilities
        tags = list(self.all_tags.values())
        embeddings = np.stack([t.embedding for t in tags])
        utilities, uncertainties = self.learner.predict_utility(embeddings)
        
        # Select tags with positive utility (GP learned they are preferred)
        positive_indices = [i for i, u in enumerate(utilities) if u >= min_utility]
        
        if not positive_indices:
            # Fallback: if no positive-utility tags, use top-K by utility
            return self.get_top_k_tags_for_generation(k=10, min_cos_distance=min_cos_distance)
        
        # Get positive tags
        positive_tags = [tags[i] for i in positive_indices]
        positive_utilities = utilities[positive_indices]
        positive_uncertainties = uncertainties[positive_indices]
        positive_embeddings = embeddings[positive_indices]
        
        if not positive_tags:
            return [], np.array([]), []
        
        # Sort by utility (descending) for deduplication priority
        sorted_indices = np.argsort(positive_utilities)[::-1]
        
        # Deduplicate by cosine distance (keep highest utility when similar)
        selected_indices = []
        selected_embs = []
        
        for idx in sorted_indices:
            emb = positive_embeddings[idx]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            
            # Check cosine distance to already selected tags
            is_duplicate = False
            for sel_emb in selected_embs:
                cos_sim = np.dot(emb_norm, sel_emb)
                cos_distance = 1 - cos_sim
                if cos_distance < min_cos_distance:  # Too similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_indices.append(idx)
                selected_embs.append(emb_norm)
        
        if not selected_indices:
            return [], np.array([]), []
        
        # Build output
        tag_texts = [positive_tags[i].text for i in selected_indices]
        selected_utilities = np.array([positive_utilities[i] for i in selected_indices])
        selected_uncertainties = np.array([positive_uncertainties[i] for i in selected_indices])
        
        # Softmax normalization
        exp_u = np.exp(selected_utilities / tau)
        weights = exp_u / (exp_u.sum() + 1e-8)
        
        # Build detailed info
        tag_details = []
        for i, idx in enumerate(selected_indices):
            tag = positive_tags[idx]
            tag_details.append({
                'tag_id': tag.id,
                'text': tag.text,
                'utility': float(selected_utilities[i]),
                'uncertainty': float(selected_uncertainties[i]),
                'weight': float(weights[i]),
                'source_image_idx': tag.source_image_idx,
                'category': 'positive'  # GP utility >= threshold
            })
        
        return tag_texts, weights, tag_details
    
    def save_raw_tag_weights(
        self,
        session_folder: str,
        stage: str = "impression",
        k: int = None,  # None = all positive tags
        min_cos_distance: float = 0.15,
        include_all_positive: bool = True
    ) -> str:
        """
        Save raw tag weights to disk (no clustering).
        
        This saves individual tag weights based on GP utilities,
        bypassing the concept clustering step for more faithful
        representation of learned preferences.
        
        Saves to: <session_folder>/<stage>/concept_weights.json
        (replaces the clustered version)
        
        Args:
            session_folder: Path to the session folder
            stage: Stage name (default: "impression")
            k: Number of top tags to save (None = all positive tags)
            min_cos_distance: Minimum cosine distance for deduplication
            include_all_positive: If True, include all liked tags (ignore k)
        
        Returns:
            Path to the saved file
        """
        import os
        import json
        from datetime import datetime
        
        if include_all_positive:
            # Get ALL positive (liked) tags with their GP-learned weights
            tag_texts, weights, tag_details = self._get_all_positive_tags_with_weights(
                min_cos_distance=min_cos_distance
            )
        else:
            # Legacy: Get top-K tags
            tag_texts, weights, tag_details = self.get_top_k_tags_for_generation(
                k=k or 10,
                min_cos_distance=min_cos_distance
            )
        
        if not tag_texts:
            print("[GP SAVE_RAW_WEIGHTS] No tags to save")
            return ""
        
        # Build concept_weights format (compatible with existing slider generation)
        # Each tag becomes its own "concept" with weight = its individual GP utility
        concept_weights_list = []
        for i, detail in enumerate(tag_details):
            concept_weights_list.append({
                'concept_id': f"tag_{i}",
                'label': detail['text'],
                'weight': detail['weight'],
                'score': detail['utility'],
                'utility_std': detail['uncertainty'],
                'category': 'positive' if detail['utility'] > 0.5 else (
                    'negative' if detail['utility'] < -0.5 else 'neutral'
                ),
                'like_count': 0,
                'dislike_count': 0,
                'member_tag_ids': [detail['text']],  # Single member (itself)
                'is_raw_tag': True  # Flag to indicate this is a raw tag, not clustered
            })
        
        weights_data = {
            'stage': stage,
            'session_id': os.path.basename(session_folder),
            'timestamp': datetime.now().isoformat(),
            'num_concepts': len(tag_details),
            'gp_mode': True,
            'raw_tag_mode': True,  # Flag to indicate raw tag mode
            'n_preference_pairs': len(self.learner.preference_pairs),
            'is_fitted': self.learner.is_fitted,
            'deduplication_params': {
                'k': k,
                'min_cos_distance': min_cos_distance
            },
            'concept_weights': concept_weights_list,
            # Also save the raw details for debugging
            'raw_tag_details': tag_details
        }
        
        # Save to file
        stage_folder = os.path.join(session_folder, stage)
        os.makedirs(stage_folder, exist_ok=True)
        weights_file = os.path.join(stage_folder, "concept_weights.json")
        
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        mode_str = "positive" if include_all_positive else f"top-{k}"
        print(f"[GP SAVE_RAW_WEIGHTS] Saved {len(tag_details)} {mode_str} tags to {weights_file}")
        top_3 = [f"{d['text']}: {d['weight']:.3f} (u={d['utility']:.2f})" for d in tag_details[:3]]
        print(f"[GP SAVE_RAW_WEIGHTS] Top 3: {top_3}")
        
        return weights_file
    
    def get_statistics(self) -> Dict:
        """Get system statistics for debugging."""
        return {
            'n_interactions': self.interaction_count,
            'n_tags': len(self.all_tags),
            'n_preference_pairs': len(self.learner.preference_pairs),
            'is_fitted': self.learner.is_fitted,
            'n_concepts': len(self.get_concepts()) if self.learner.is_fitted else 0,
        }
    
    def save_concept_weights(self, session_folder: str, stage: str = "impression") -> str:
        """
        Save learned concept weights to disk in the format expected by slider generation.
        
        Saves to: <session_folder>/<stage>/concept_weights.json
        
        Args:
            session_folder: Path to the session folder
            stage: Stage name (default: "impression")
        
        Returns:
            Path to the saved file
        """
        import os
        import json
        from datetime import datetime
        
        # Get concepts and compute weights
        concepts = self.get_concepts()
        if not concepts:
            print("[GP SAVE_WEIGHTS] No concepts to save")
            return ""
        
        # Get softmax-normalized weights
        tau = 1.0
        utilities = np.array([c.mean_utility for c in concepts])
        exp_u = np.exp(utilities / tau)
        weights = exp_u / exp_u.sum()
        
        # Build concept_weights data structure
        concept_weights_list = []
        for i, (concept, weight) in enumerate(zip(concepts, weights)):
            concept_weights_list.append({
                'concept_id': concept.id,
                'label': concept.representative_tag,
                'weight': float(weight),
                'score': float(concept.mean_utility),
                'utility_std': float(concept.utility_std),
                'category': concept.category,
                'like_count': 0,  # GP doesn't track this directly
                'dislike_count': 0,
                'member_tag_ids': concept.member_tags  # Using tag texts as IDs for display
            })
        
        # Sort by weight descending
        concept_weights_list.sort(key=lambda x: x['weight'], reverse=True)
        
        weights_data = {
            'stage': stage,
            'session_id': os.path.basename(session_folder),
            'timestamp': datetime.now().isoformat(),
            'num_concepts': len(concepts),
            'gp_mode': True,  # Flag to indicate GP-derived weights
            'n_preference_pairs': len(self.learner.preference_pairs),
            'is_fitted': self.learner.is_fitted,
            'concept_weights': concept_weights_list
        }
        
        # Save to file
        stage_folder = os.path.join(session_folder, stage)
        os.makedirs(stage_folder, exist_ok=True)
        weights_file = os.path.join(stage_folder, "concept_weights.json")
        
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"[GP SAVE_WEIGHTS] Saved {len(concepts)} concepts to {weights_file}")
        top_3 = [f"{c['label']}: {c['weight']:.3f}" for c in concept_weights_list[:3]]
        print(f"[GP SAVE_WEIGHTS] Top 3: {top_3}")
        
        return weights_file
    
    def to_dict(self) -> Dict:
        """
        Serialize current state to dict for API response.
        
        Returns a format compatible with the existing ConceptRefinementSession.to_dict()
        """
        concepts = self.get_concepts()
        
        # Get softmax weights
        tau = 1.0
        if concepts:
            utilities = np.array([c.mean_utility for c in concepts])
            exp_u = np.exp(utilities / tau)
            weights = exp_u / exp_u.sum()
        else:
            weights = []
        
        # Build concepts output
        concepts_output = []
        for i, (concept, w) in enumerate(zip(concepts, weights)):
            concepts_output.append({
                'id': concept.id,
                'label': concept.representative_tag,
                'centroid': concept.centroid.tolist() if isinstance(concept.centroid, np.ndarray) else concept.centroid,
                'member_tag_ids': concept.member_tags,
                'member_tags': concept.member_tags,
                'state': {
                    'like_count': 0,
                    'dislike_count': 0,
                    'rank_bonus': 0.0,
                    'rank_penalty': 0.0,
                    'score': concept.mean_utility,
                    'w': float(w),
                    'utility_std': concept.utility_std,
                    'liked_tags': [],
                    'disliked_tags': []
                }
            })
        
        # Categorize by utility thresholds
        positive = [c['id'] for c in concepts_output if c['state']['score'] > 0.5]
        negative = [c['id'] for c in concepts_output if c['state']['score'] < -0.5]
        neutral = [c['id'] for c in concepts_output if -0.5 <= c['state']['score'] <= 0.5]
        
        # Sort by weight
        positive.sort(key=lambda cid: next(c['state']['w'] for c in concepts_output if c['id'] == cid), reverse=True)
        negative.sort(key=lambda cid: next(c['state']['w'] for c in concepts_output if c['id'] == cid))
        neutral.sort(key=lambda cid: next(c['state']['w'] for c in concepts_output if c['id'] == cid), reverse=True)
        
        return {
            'session_id': '',  # Set by caller
            'stage': 'impression',
            'concepts': concepts_output,
            'categorized': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            },
            'image_effects': {},  # GP doesn't track per-image effects
            'incidence_matrix': {},
            'tag_preferences': self._get_tag_preferences()
        }
    
    def _get_tag_preferences(self) -> Dict[str, Optional[str]]:
        """Get preference status for all tags (for UI display)."""
        # In GP mode, we don't have explicit tag preferences
        # Return empty preferences - the UI will show tags as neutral
        return {tag_id: None for tag_id in self.all_tags.keys()}
    
    def reset(self):
        """Reset all learned state."""
        self.all_tags.clear()
        self.learner.clear()
        self.interaction_count = 0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate system usage with synthetic data."""
    import random
    
    # Simulate CLIP embeddings (normally from CLIP model)
    def make_embedding(seed: int) -> np.ndarray:
        np.random.seed(seed)
        emb = np.random.randn(768)  # CLIP ViT-L/14 dimension
        return emb / np.linalg.norm(emb)
    
    # Create synthetic tags
    tag_texts = [
        "warm lighting", "cool shadows", "soft glow", "harsh contrast",
        "wooden texture", "metal surface", "fabric folds", "glass reflection",
        "cozy atmosphere", "minimalist", "cluttered", "spacious",
        "golden hour", "blue tones", "earth colors", "monochrome"
    ]
    
    def make_tags_for_image(image_idx: int, offset: int) -> List[RawTag]:
        tags = []
        for i in range(12):
            idx = (offset + i) % len(tag_texts)
            tags.append(RawTag(
                id=f"img{image_idx}_tag{i}",
                text=tag_texts[idx],
                embedding=make_embedding(idx * 100 + image_idx),
                source_image_idx=image_idx
            ))
        return tags
    
    # Initialize system
    system = AdaptivePreferenceSystem(n_inducing=32)
    
    # Simulate interactions
    for round_num in range(3):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num + 1}")
        print('='*50)
        
        # Generate 4 images with tags
        images = [make_tags_for_image(i, round_num * 4 + i) for i in range(4)]
        
        # Simulate user selection and tagging
        selected = random.randint(0, 3)
        tag_states = {}
        
        for img_idx, img_tags in enumerate(images):
            for tag in img_tags:
                # Simulate: user likes "warm" things, dislikes "harsh" things
                if "warm" in tag.text or "cozy" in tag.text or "soft" in tag.text:
                    tag_states[tag.id] = 'liked'
                elif "harsh" in tag.text or "cold" in tag.text:
                    tag_states[tag.id] = 'disliked'
                else:
                    tag_states[tag.id] = 'neutral'
        
        interaction = InteractionRound(
            images=images,
            selected_image_idx=selected,
            tag_states=tag_states
        )
        
        # Process
        concepts = system.process_interaction(interaction, verbose=True)
        
        # Show results
        print(f"\nConcepts ({len(concepts)}):")
        for c in concepts[:5]:  # Top 5
            print(f"  [{c.category:8s}] {c.representative_tag:20s} "
                  f"utility={c.mean_utility:+.2f} ± {c.utility_std:.2f} "
                  f"({len(c.member_tags)} tags)")
        
        print(f"\nStatistics: {system.get_statistics()}")
    
    # Test prediction for new tag
    print(f"\n{'='*50}")
    print("PREDICTION FOR NEW TAG")
    print('='*50)
    
    new_tag = RawTag(
        id="new_1",
        text="warm amber glow",  # Should be predicted as liked
        embedding=make_embedding(999),
        source_image_idx=-1
    )
    
    predictions = system.predict_for_new_tags([new_tag])
    tag, util, unc = predictions[0]
    print(f"'{tag.text}': utility={util:+.2f} ± {unc:.2f}")


if __name__ == "__main__":
    example_usage()

