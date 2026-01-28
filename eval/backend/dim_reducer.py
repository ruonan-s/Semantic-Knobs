"""
Dimensionality Reducer for CLIP Embedding Space

Reduces 768-dim CLIP embeddings to a lower-dimensional space where
the GP can effectively learn the utility function.

The meaningful semantic variation for interior design concepts lies
on a much lower-dimensional manifold. PCA finds this subspace.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA


class EmbeddingReducer:
    """
    Reduce CLIP embeddings from 768-dim to a lower-dimensional space.
    
    The GP operates in this reduced space for better:
    - Convergence speed (fewer dimensions to explore)
    - Uncertainty estimation (better coverage)
    - Computational efficiency
    
    After sampling in reduced space, embeddings are projected back
    to 768-dim for SDXL injection.
    """
    
    def __init__(
        self,
        n_components: int = 32,
        variance_threshold: Optional[float] = None
    ):
        """
        Initialize the embedding reducer.
        
        Args:
            n_components: Target dimensionality (default 32)
            variance_threshold: If set, choose n_components to explain 
                               this fraction of variance (e.g., 0.95)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca: Optional[PCA] = None
        self.is_fitted = False
        
        # Store original space mean for reconstruction
        self._mean = None
        self._explained_variance_ratio = None
    
    def fit(self, embeddings: np.ndarray) -> "EmbeddingReducer":
        """
        Fit PCA on the available embeddings.
        
        Should be called with all tag embeddings (positive + negative + neutral)
        to capture the full semantic space of interest.
        
        Args:
            embeddings: (N, 768) array of CLIP embeddings
        
        Returns:
            self for chaining
        """
        if len(embeddings) < 3:
            print(f"[DimReducer] Warning: Only {len(embeddings)} embeddings, need at least 3")
            # Fallback: use identity (no reduction)
            self.n_components = min(self.n_components, len(embeddings))
        
        # Determine n_components
        if self.variance_threshold is not None:
            # First fit with all components to find optimal
            temp_pca = PCA(n_components=min(len(embeddings) - 1, 768))
            temp_pca.fit(embeddings)
            
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            n_for_threshold = np.searchsorted(cumsum, self.variance_threshold) + 1
            self.n_components = min(n_for_threshold, len(embeddings) - 1, 100)
            print(f"[DimReducer] Using {self.n_components} components for {self.variance_threshold*100:.0f}% variance")
        
        # Fit PCA
        self.n_components = min(self.n_components, len(embeddings) - 1, 768)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(embeddings)
        
        self._mean = embeddings.mean(axis=0)
        self._explained_variance_ratio = self.pca.explained_variance_ratio_
        self.is_fitted = True
        
        total_variance = sum(self._explained_variance_ratio)
        print(f"[DimReducer] Fitted PCA: 768 → {self.n_components} dims "
              f"({total_variance*100:.1f}% variance explained)")
        
        return self
    
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings to reduced space.
        
        Args:
            embeddings: (N, 768) or (768,) array
        
        Returns:
            (N, n_components) or (n_components,) reduced embeddings
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings.reshape(1, -1)
        
        reduced = self.pca.transform(embeddings)
        
        return reduced[0] if single else reduced
    
    def reconstruct(self, reduced: np.ndarray) -> np.ndarray:
        """
        Project reduced embeddings back to 768-dim space.
        
        Args:
            reduced: (N, n_components) or (n_components,) array
        
        Returns:
            (N, 768) or (768,) reconstructed embeddings
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        single = reduced.ndim == 1
        if single:
            reduced = reduced.reshape(1, -1)
        
        reconstructed = self.pca.inverse_transform(reduced)
        
        # Normalize to unit sphere (CLIP embeddings are normalized)
        norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        reconstructed = reconstructed / (norms + 1e-8)
        
        return reconstructed[0] if single else reconstructed
    
    def reduce_and_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce and normalize in reduced space."""
        reduced = self.reduce(embeddings)
        if reduced.ndim == 1:
            return reduced / (np.linalg.norm(reduced) + 1e-8)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        return reduced / (norms + 1e-8)
    
    @property
    def dim(self) -> int:
        """Get the reduced dimensionality."""
        return self.n_components if self.is_fitted else 768
    
    @property
    def explained_variance(self) -> float:
        """Get total explained variance ratio."""
        if self._explained_variance_ratio is None:
            return 0.0
        return float(sum(self._explained_variance_ratio))


def slerp_reduced(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation in reduced space.
    
    Works the same as regular SLERP but in lower dimensions.
    """
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-8)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        return v0_norm * (1 - t) + v1_norm * t
    
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * v0_norm + \
           (np.sin(t * theta) / sin_theta) * v1_norm
