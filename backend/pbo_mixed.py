"""
Preferential Bayesian Optimization (PBO) for Tag-Concept Refinement

Implements preference-based optimization over concept weight mixtures using:
- Mixture embeddings: z = L2_normalize(w @ MU) where MU are concept centroids
- Laplace approximation for GP preference likelihood
- Batch acquisition with 4 strategies (Thompson, EI, variance, diverse)
- Negative concept constraints via soft penalties
- Candidate coalescing and pruning for efficiency
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
from collections import defaultdict

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Kernel
    from scipy.optimize import minimize
    from scipy.special import expit  # logistic sigmoid
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[PBO] Warning: sklearn/scipy not available. Using fallback surrogate.")

# ============================================================================
# Parameters (defaults from spec)
# ============================================================================
KERNEL_LENGTH_SCALE = 0.6
KERNEL_SIGMA_F = 1.0
POOL_SIZE = 2048
BATCH_DIVERSITY_MAX_COS = 0.95
NEGATIVE_PENALTY_RHO = 0.03
NEGATIVE_PENALTY_LAMBDA = 10.0
PER_CONCEPT_CAP = 0.35
SOFTMAX_TAU_W = 0.7
MAX_CANDIDATES = 200
COALESCE_COSINE_THRESHOLD = 0.995

EPS = 1e-8


# ============================================================================
# Cosine-RBF Kernel for mixture embeddings
# ============================================================================
class CosineRBFKernel(Kernel):
    """Cosine-RBF kernel: k(z,z') = σ²_f * exp(-(1 - cos(z,z')) / ℓ)"""

    def __init__(self, length_scale=KERNEL_LENGTH_SCALE, sigma_f=KERNEL_SIGMA_F):
        self.length_scale = length_scale
        self.sigma_f = sigma_f

    def __call__(self, X, Y=None, eval_gradient=False):
        """Compute kernel matrix"""
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        # Cosine similarity (assuming unit-norm rows)
        cos_sim = np.dot(X, Y.T)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        # RBF on cosine distance
        cos_dist = 1.0 - cos_sim
        K = self.sigma_f ** 2 * np.exp(-cos_dist / self.length_scale)

        if eval_gradient:
            # Gradient computation (simplified - not fully optimized)
            return K, np.zeros((K.shape[0], K.shape[1], 2))  # dummy gradient
        return K

    @property
    def theta(self):
        return np.log([self.length_scale, self.sigma_f])

    @theta.setter
    def theta(self, value):
        self.length_scale = np.exp(value[0])
        self.sigma_f = np.exp(value[1])

    @property
    def bounds(self):
        return np.log([[1e-2, 5.0], [0.1, 10.0]])

    def diag(self, X):
        return np.full(X.shape[0], self.sigma_f ** 2)

    def is_stationary(self):
        return True


# ============================================================================
# Data Models
# ============================================================================
@dataclass
class Candidate:
    """A candidate weight mixture"""
    id: str
    w: np.ndarray  # weight vector on simplex (K,)
    z: np.ndarray  # mixture embedding (d,) - L2 normalized

    def to_dict(self):
        return {
            'id': self.id,
            'w': self.w.tolist(),
            'z': self.z.tolist()
        }


@dataclass
class Duel:
    """A pairwise preference"""
    better_id: str
    worse_id: str
    strength: float = 1.0  # 0.5 for weak (snapshots), 1.0 for strong (favorites)

    def to_dict(self):
        return asdict(self)


# ============================================================================
# Helper Functions
# ============================================================================
def normalize_simplex(w: np.ndarray) -> np.ndarray:
    """Project to probability simplex"""
    w = np.maximum(0.0, np.asarray(w, dtype=np.float32))
    s = w.sum()
    return w / (s + EPS) if s > EPS else np.ones_like(w) / len(w)


def softmax(phi: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax: w = exp(phi/tau) / sum(exp(phi/tau))"""
    phi = np.asarray(phi, dtype=np.float32)
    phi_scaled = phi / tau
    phi_scaled = phi_scaled - np.max(phi_scaled)  # numerical stability
    exp_phi = np.exp(phi_scaled)
    return exp_phi / (exp_phi.sum() + EPS)


def logit_to_weights(phi: np.ndarray, tau: float = SOFTMAX_TAU_W, cap: float = PER_CONCEPT_CAP) -> np.ndarray:
    """Convert logits to weights with cap and renormalization"""
    w = softmax(phi, tau)
    # Apply per-concept cap
    w = np.minimum(w, cap)
    # Renormalize
    return normalize_simplex(w)


def compute_mixture_embedding(w: np.ndarray, MU: np.ndarray) -> np.ndarray:
    """
    Compute mixture embedding: z = L2_normalize(w @ MU)

    Args:
        w: weight vector (K,)
        MU: concept centroids matrix (K, d), rows are L2-normalized

    Returns:
        z: mixture embedding (d,), L2-normalized
    """
    w = normalize_simplex(w)
    z = np.dot(w, MU)  # (d,)
    norm = np.linalg.norm(z)
    if norm < EPS:
        # Degenerate case - return uniform over first axis
        z = np.zeros_like(z)
        z[0] = 1.0
        return z
    return z / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def project_sdxl(w: np.ndarray, top_k: int = 15, jitter: float = 0.01) -> np.ndarray:
    """
    Project weight vector to SDXL-compatible form (keep top-K tags only).
    
    Args:
        w: weight vector (K,)
        top_k: number of top tags to keep
        jitter: small random noise to avoid exact duplicates
    
    Returns:
        w_proj: projected weight vector (K,) with only top-K non-zero
    """
    # Ensure non-negative and normalized
    w = np.maximum(w, 0.0)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    
    K = len(w)
    top_k = min(top_k, K)
    
    # Get top-K indices
    idx = np.argsort(-w)[:top_k]
    
    # Create new weight vector (only top-K non-zero)
    w_proj = np.zeros(K, dtype=np.float32)
    w_proj[idx] = w[idx]
    
    # Add tiny jitter to avoid exact duplicates (only to selected indices)
    if jitter > 0:
        noise = np.random.normal(0.0, jitter, size=K)
        # Only perturb non-zero entries
        noise_mask = (w_proj > 0)
        w_proj[noise_mask] = w_proj[noise_mask] + noise[noise_mask]
        # Clip to non-negative
        w_proj = np.maximum(w_proj, 0.0)
    
    # Renormalize
    w_proj = normalize_simplex(w_proj)
    
    # ---- Soft cap after projection to prevent top-k amplification ----
    if (w_proj > PER_CONCEPT_CAP).any():
        # Cap the dominant concepts
        excess = np.sum(w_proj[w_proj > PER_CONCEPT_CAP] - PER_CONCEPT_CAP)
        w_proj[w_proj > PER_CONCEPT_CAP] = PER_CONCEPT_CAP
        
        # Redistribute excess over remaining positive entries
        tail_mask = w_proj > 0
        tail_mass = np.sum(w_proj[tail_mask])
        if tail_mass > 0 and excess > 0:
            w_proj[tail_mask] += excess * (w_proj[tail_mask] / tail_mass)
    
    return normalize_simplex(w_proj)


def local_around(
    w_center: np.ndarray, 
    alpha_scale: float = 30.0,
    top_k: int = 15,
    rng: Optional[np.random.RandomState] = None,
    num_rounds: Optional[int] = None
) -> np.ndarray:
    """
    Generate Dirichlet sample around a center weight vector.
    
    Args:
        w_center: center weight vector (K,)
        alpha_scale: concentration parameter (higher = tighter around center)
        top_k: project result to top-K
        rng: random state
        num_rounds: number of rounds completed (for adaptive alpha_scale)
    
    Returns:
        w_local: sampled weight vector (K,)
    """
    if rng is None:
        rng = np.random
    
    w_center = normalize_simplex(w_center)
    
    # ---- Cap individual concepts before Dirichlet to prevent runaway concentration ----
    w_center_capped = np.minimum(w_center, PER_CONCEPT_CAP)
    w_center_capped = normalize_simplex(w_center_capped)
    
    # ---- Mix in uniform to maintain diversity ----
    mix_uniform = 0.15  # 15% uniform keeps tail concepts alive
    K = len(w_center)
    w_base = (1.0 - mix_uniform) * w_center_capped + mix_uniform * (np.ones(K, dtype=np.float32) / K)
    w_base = normalize_simplex(w_base)
    
    # ---- Adaptive alpha_scale (gentler growth) ----
    if num_rounds is not None:
        alpha_scale = min(8.0 + 2.0 * num_rounds, 20.0)  # 8 → 20 (instead of 20 → 50)
    
    # Dirichlet concentration
    alpha = alpha_scale * (w_base + 1e-6)
    alpha = np.maximum(alpha, 1e-6)  # ensure all alphas > 0
    w_local = rng.dirichlet(alpha)
    
    # Project to SDXL format
    return project_sdxl(w_local, top_k=top_k, jitter=0.005)


# ============================================================================
# PBO Class
# ============================================================================
class PBO:
    """
    Preferential Bayesian Optimization over concept weight mixtures.

    Uses mixture embeddings z = L2_normalize(w @ MU) as features for GP.
    Learns from pairwise preferences (duels) via Laplace approximation.
    Proposes batches of 4 diverse candidates using multi-strategy acquisition.
    """

    def __init__(
        self,
        MU: np.ndarray,  # concept centroids (K, d)
        concept_ids: List[str],  # concept IDs
        kernel_length_scale: float = KERNEL_LENGTH_SCALE,
        kernel_sigma_f: float = KERNEL_SIGMA_F,
        random_state: int = 42,
        concept_weights: Optional[np.ndarray] = None  # Initial concept weights (ema_w) for warm start
    ):
        self.MU = np.asarray(MU, dtype=np.float32)  # (K, d)
        self.K, self.d = self.MU.shape
        self.concept_ids = concept_ids
        self.rng = np.random.RandomState(random_state)

        # Kernel parameters
        self.length_scale = kernel_length_scale
        self.sigma_f = kernel_sigma_f

        # Storage
        self.candidates: Dict[str, Candidate] = {}  # id -> Candidate
        self.duels: List[Duel] = []

        # Surrogate
        self.gp = None
        self.fitted = False

        # Counters
        self._cid_counter = 0
        
        # Initial weights for warm start (sorted indices for cold start proposals)
        if concept_weights is not None:
            self.concept_weights = np.asarray(concept_weights, dtype=np.float32)
            # Sort by weight descending for cold start
            self.sorted_indices = np.argsort(-self.concept_weights)
            print(f"[PBO] Initialized with concept weights (warm start mode)")
            print(f"  Top 3 concepts by weight: {[self.concept_ids[i] for i in self.sorted_indices[:3]]}")
        else:
            self.concept_weights = np.ones(self.K, dtype=np.float32) / self.K
            self.sorted_indices = np.arange(self.K)
            print(f"[PBO] Initialized without weights (cold start mode)")

        print(f"[PBO] Initialized with K={self.K} concepts, d={self.d} embedding dim")

    def compute_mixture_embedding(self, w: np.ndarray) -> np.ndarray:
        """
        Compute mixture embedding for a weight vector.
        
        Args:
            w: weight vector (K,)
            
        Returns:
            z: mixture embedding (d,), L2-normalized
        """
        return compute_mixture_embedding(w, self.MU)

    def _generate_candidate_id(self) -> str:
        """Generate unique candidate ID"""
        self._cid_counter += 1
        return f"cand_{self._cid_counter:04d}"

    def add_candidate(self, w: np.ndarray, candidate_id: Optional[str] = None) -> str:
        """
        Add a candidate weight vector.

        Args:
            w: weight vector (K,)
            candidate_id: optional ID (auto-generated if None)

        Returns:
            candidate_id
        """
        w = normalize_simplex(w)
        z = compute_mixture_embedding(w, self.MU)

        # Check for near-duplicates (coalesce)
        # BUT: Don't coalesce during early rounds (cold start) to ensure diversity
        # We need at least a few distinct candidates for GP to fit properly
        allow_coalescing = len(self.candidates) >= 10  # Allow coalescing after ~2-3 rounds
        
        if allow_coalescing:
            for cid, cand in self.candidates.items():
                cos_sim = cosine_similarity(z, cand.z)
                if cos_sim > COALESCE_COSINE_THRESHOLD:
                    print(f"[PBO] Coalescing candidate (cos={cos_sim:.4f}) into {cid}")
                    return cid
        else:
            print(f"[PBO] Coalescing disabled (early rounds: {len(self.candidates)} candidates)")

        # Create new candidate
        if candidate_id is None:
            candidate_id = self._generate_candidate_id()

        candidate = Candidate(id=candidate_id, w=w, z=z)
        self.candidates[candidate_id] = candidate

        # Prune if too many candidates
        if len(self.candidates) > MAX_CANDIDATES:
            self._prune_candidates()

        print(f"[PBO] Added candidate {candidate_id} (total: {len(self.candidates)})")
        return candidate_id

    def add_preference(self, better_id: str, worse_id: str, strength: float = 1.0) -> None:
        """
        Add a pairwise preference (duel).

        Args:
            better_id: ID of preferred candidate
            worse_id: ID of less preferred candidate
            strength: 0.5 for weak (snapshot), 1.0 for strong (favorite)
        """
        if better_id not in self.candidates:
            print(f"[PBO] Warning: better_id={better_id} not found")
            return
        if worse_id not in self.candidates:
            print(f"[PBO] Warning: worse_id={worse_id} not found")
            return

        duel = Duel(better_id=better_id, worse_id=worse_id, strength=strength)
        self.duels.append(duel)
        print(f"[PBO] Added duel: {better_id} ≻ {worse_id} (strength={strength})")

    def _prune_candidates(self) -> None:
        """Prune candidates to MAX_CANDIDATES using FIFO + diversity"""
        if len(self.candidates) <= MAX_CANDIDATES:
            return

        print(f"[PBO] Pruning candidates from {len(self.candidates)} to {MAX_CANDIDATES}")

        # Keep most recent candidates (FIFO)
        # Sort by candidate ID (assumes sequential IDs)
        sorted_cands = sorted(self.candidates.items(), key=lambda x: x[0], reverse=True)
        keep_ids = {cid for cid, _ in sorted_cands[:MAX_CANDIDATES]}

        # Remove old candidates
        self.candidates = {cid: cand for cid, cand in self.candidates.items() if cid in keep_ids}

        # Remove duels involving pruned candidates
        self.duels = [d for d in self.duels
                      if d.better_id in self.candidates and d.worse_id in self.candidates]

        print(f"[PBO] After pruning: {len(self.candidates)} candidates, {len(self.duels)} duels")

    def fit(self) -> None:
        """
        Fit GP surrogate using Laplace approximation on preference likelihood.

        This is the expensive step - only call on "finalize" triggers.
        """
        if len(self.candidates) < 2 or len(self.duels) == 0:
            print(f"[PBO] Not enough data to fit (candidates={len(self.candidates)}, duels={len(self.duels)})")
            self.fitted = False
            return

        print(f"\n[PBO FIT] Fitting GP with {len(self.candidates)} candidates, {len(self.duels)} duels")

        # Build feature matrix Z (N, d)
        cand_list = list(self.candidates.values())
        Z = np.vstack([c.z for c in cand_list])
        cand_id_to_idx = {c.id: i for i, c in enumerate(cand_list)}

        # Convert duels to utility observations via Copeland scores
        # Count wins and losses for each candidate
        wins = defaultdict(int)
        losses = defaultdict(int)

        for duel in self.duels:
            wins[duel.better_id] += duel.strength
            losses[duel.worse_id] += duel.strength

        # Compute utility via logit of win rate
        y = np.zeros(len(cand_list), dtype=np.float32)
        for i, cand in enumerate(cand_list):
            w = wins.get(cand.id, 0)
            l = losses.get(cand.id, 0)
            total = w + l
            if total > 0:
                frac = w / (total + EPS)
                frac = np.clip(frac, 0.05, 0.95)  # avoid inf
                y[i] = np.log(frac / (1.0 - frac))  # logit
            else:
                y[i] = 0.0

        print(f"[PBO FIT] Utility range: [{y.min():.3f}, {y.max():.3f}]")

        # Fit GP
        if HAS_SKLEARN:
            try:
                kernel = CosineRBFKernel(
                    length_scale=self.length_scale,
                    sigma_f=self.sigma_f
                )
                from sklearn.gaussian_process.kernels import WhiteKernel
                kernel = kernel + WhiteKernel(noise_level=1e-3)

                self.gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=0,
                    n_restarts_optimizer=2
                )
                self.gp.fit(Z, y)
                self.fitted = True

                # Log learned hyperparameters
                learned_kernel = self.gp.kernel_
                print(f"[PBO FIT] Learned kernel: {learned_kernel}")
                print(f"[PBO FIT] Log-marginal-likelihood: {self.gp.log_marginal_likelihood_value_:.3f}")

            except Exception as e:
                print(f"[PBO FIT] GP fit failed: {e}. Using fallback.")
                self.fitted = False
        else:
            print(f"[PBO FIT] sklearn not available. Using fallback surrogate.")
            self.fitted = False

    def _predict(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict utility mean and std for embeddings Z.

        Args:
            Z: embeddings (N, d)

        Returns:
            mu: mean (N,)
            std: standard deviation (N,)
        """
        if not self.fitted or self.gp is None:
            # Cold start - uniform uncertainty
            mu = np.zeros(len(Z), dtype=np.float32)
            std = np.ones(len(Z), dtype=np.float32)
            return mu, std

        mu, std = self.gp.predict(Z, return_std=True)
        return mu, std

    def _expected_improvement(
        self,
        mu: np.ndarray,
        std: np.ndarray,
        best_mu: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """Expected Improvement acquisition function"""
        std = np.maximum(std, 1e-9)
        z = (mu - best_mu - xi) / std

        # Use scipy's erf for CDF
        from scipy.special import erf
        cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

        ei = (mu - best_mu - xi) * cdf + std * pdf
        ei = np.where(std <= 1e-9, 0.0, ei)
        return ei

    def propose_batch(
        self,
        q: int = 4,
        negatives: Optional[set] = None,
        pool_size: int = POOL_SIZE,
        max_cos: float = BATCH_DIVERSITY_MAX_COS,
        w_current: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Propose a batch of q diverse candidates using multi-strategy acquisition.

        Args:
            q: number of candidates to propose
            negatives: set of negative concept IDs (for soft penalty)
            pool_size: number of candidates to optimize over
            max_cos: maximum pairwise cosine similarity (diversity constraint)
            w_current: current UI weights (for Dirichlet seeding)

        Returns:
            List of q weight vectors (each K,)
        """
        print(f"\n[PBO PROPOSE] Generating batch of {q} candidates")
        print(f"  Pool size: {pool_size}")
        print(f"  Negatives: {negatives}")
        print(f"  Max cosine: {max_cos}")

        if negatives is None:
            negatives = set()

        # Get negative indices
        neg_indices = [i for i, cid in enumerate(self.concept_ids) if cid in negatives]

        # Cold start - use learned weights with controlled perturbations
        # Instead of one-hot corners, create meaningful variations of learned weights
        if not self.fitted or len(self.candidates) < 2:
            print("[PBO PROPOSE] Cold start - generating perturbations of learned weights")
            proposals = []
            
            # Get learned weights (normalized)
            w_learned = self.concept_weights.copy()
            w_learned = w_learned / (w_learned.sum() + EPS)
            uniform = np.ones(self.K, dtype=np.float32) / self.K
            
            # Get top concept indices for reference
            top_3_indices = self.sorted_indices[:3]
            mid_tier_indices = self.sorted_indices[3:7] if self.K >= 7 else self.sorted_indices[3:]
            
            # Strategy 1: Learned Baseline - use weights directly from exploration
            w1 = w_learned.copy()
            w1 = normalize_simplex(w1)  # Explicit normalization
            proposals.append(w1)
            top_3_labels = [self.concept_ids[i] for i in top_3_indices]
            print(f"  [1/4] Learned Baseline: Top-3={top_3_labels}")
            print(f"        Weights: [{', '.join([f'{w1[i]:.3f}' for i in top_3_indices])}]")
            
            # Strategy 2: Top-Heavy - amplify top 3 concepts, dampen others
            w2 = w_learned.copy()
            for i in range(self.K):
                if i in top_3_indices:
                    w2[i] *= 1.5  # Boost favorites
                else:
                    w2[i] *= 0.5  # Dampen others
            w2 = np.maximum(w2, 0.0)  # Clip to non-negative
            w2 = normalize_simplex(w2)  # Renormalize
            proposals.append(w2)
            print(f"  [2/4] Top-Heavy: Amplify top-3 (×1.5), dampen rest (×0.5)")
            print(f"        Weights: [{', '.join([f'{w2[i]:.3f}' for i in top_3_indices])}]")
            
            # Strategy 3: Diversified - boost mid-tier concepts (rank 4-7)
            w3 = w_learned.copy()
            for i in range(self.K):
                if i in top_3_indices:
                    w3[i] *= 0.7  # Reduce dominance of top concepts
                elif i in mid_tier_indices:
                    w3[i] *= 1.8  # Boost promising mid-tier concepts
                else:
                    w3[i] *= 0.5  # Dampen low-weight concepts
            w3 = np.maximum(w3, 0.0)  # Clip to non-negative
            w3 = normalize_simplex(w3)  # Renormalize
            proposals.append(w3)
            mid_tier_labels = [self.concept_ids[i] for i in mid_tier_indices] if len(mid_tier_indices) > 0 else ["none"]
            print(f"  [3/4] Diversified: Boost mid-tier concepts (rank 4-7)")
            print(f"        Mid-tier={mid_tier_labels[:3]}")
            print(f"        Top-3 weights: [{', '.join([f'{w3[i]:.3f}' for i in top_3_indices])}]")
            
            # Strategy 4: Smoothed - blend with uniform (reduce extremes)
            w4 = 0.7 * w_learned + 0.3 * uniform
            w4 = np.maximum(w4, 0.0)  # Clip to non-negative
            w4 = normalize_simplex(w4)  # Renormalize
            proposals.append(w4)
            print(f"  [4/4] Smoothed: 70% learned + 30% uniform (balanced exploration)")
            print(f"        Weights: [{', '.join([f'{w4[i]:.3f}' for i in top_3_indices])}]")
            
            print(f"\n[PBO PROPOSE] Generated {len(proposals)} cold start proposals based on learned preferences")
            return proposals[:q]

        # ====================================================================
        # Principled 4-Candidate Design (Round 2+)
        # ====================================================================
        # Strategy: First 3 candidates emphasize learned weights (exploitation)
        #           Last candidate introduces variance (exploration)
        # A: Best from GP (exploit)
        # B: Tight local around best (exploit with small perturbation)
        # C: Medium local around best (exploit with moderate perturbation)
        # D: Diverse exploration (high uncertainty, far from A/B/C)
        # ====================================================================
        
        print("[PBO PROPOSE] Round 2+ - Learned-emphasis design (3 exploit + 1 explore)")
        proposals = []
        
        # Get w_best from GP posterior
        w_best = self._get_best_candidate()
        if w_best is None:
            # Fallback: use learned weights
            w_best = self.concept_weights.copy()
        
        print(f"  w_best (from GP): max={w_best.max():.3f}, top-3 concepts: "
              f"{[self.concept_ids[i] for i in np.argsort(-w_best)[:3]]}")
        
        num_rounds = len(self.duels) // 3 + 1  # rough estimate
        
        # ----------------------------------------------------------------
        # Candidate A: Anchor/Exploit (Best-So-Far)
        # ----------------------------------------------------------------
        print(f"\n  [A] Anchor/Exploit: w_best from GP")
        w_A = project_sdxl(w_best, top_k=15, jitter=0.01)
        proposals.append(w_A)
        print(f"      Top-3 weights: {sorted(w_A, reverse=True)[:3]}")
        
        # ----------------------------------------------------------------
        # Candidate B: Tight Local Refinement
        # ----------------------------------------------------------------
        print(f"\n  [B] Tight Local: Small perturbation around w_best")
        # Tight concentration - stays very close to w_best
        w_B = local_around(w_best, alpha_scale=15.0, top_k=15, rng=self.rng, num_rounds=num_rounds)
        proposals.append(w_B)
        print(f"      Top-3 weights: {sorted(w_B, reverse=True)[:3]}")
        
        # ----------------------------------------------------------------
        # Candidate C: Medium Local Refinement
        # ----------------------------------------------------------------
        print(f"\n  [C] Medium Local: Moderate perturbation around w_best")
        # Medium concentration - allows more variation
        w_C = local_around(w_best, alpha_scale=10.0, top_k=15, rng=self.rng, num_rounds=num_rounds)
        proposals.append(w_C)
        print(f"      Top-3 weights: {sorted(w_C, reverse=True)[:3]}")
        
        # ----------------------------------------------------------------
        # Candidate D: Diverse Explorer (Uncertainty-Guided)
        # ----------------------------------------------------------------
        print(f"\n  [D] Diverse Explorer: High σ, far from A/B/C")
        w_D = self._generate_diverse_candidate(w_A, w_B, neg_indices, pool_size=pool_size)
        if w_D is not None:
            proposals.append(w_D)
            print(f"      Top-3 weights: {sorted(w_D, reverse=True)[:3]}")
        else:
            # Fallback: random Dirichlet
            w_D = self.rng.dirichlet(np.ones(self.K))
            w_D = project_sdxl(w_D, top_k=15)
            proposals.append(w_D)
            print(f"      (Fallback random) Top-3 weights: {sorted(w_D, reverse=True)[:3]}")
        
        # Check diversity
        self._check_diversity(proposals, max_cos)
        
        return proposals[:q]

    def _generate_starts(self, w_current: Optional[np.ndarray]) -> List[np.ndarray]:
        """Generate multi-start initialization points using GP knowledge"""
        starts = []

        # PRIORITY 1: Use top-performing candidates from GP
        # These are the candidates that the model has learned are good!
        if self.fitted and len(self.candidates) >= 2:
            # Get all candidates and their predicted utilities
            candidate_weights = []
            candidate_ids = []
            for cid, cand in self.candidates.items():
                candidate_weights.append(cand.w)
                candidate_ids.append(cid)
            
            if len(candidate_weights) > 0:
                candidate_weights = np.array(candidate_weights)
                Z_candidates = np.array([compute_mixture_embedding(w, self.MU) for w in candidate_weights])
                mu_candidates, _ = self._predict(Z_candidates)
                
                # Sort by predicted utility (best first)
                top_indices = np.argsort(-mu_candidates)[:min(3, len(candidate_weights))]
                
                # Add top-performing candidates as start points
                for idx in top_indices:
                    starts.append(candidate_weights[idx])

        # PRIORITY 2: Current UI weights (if provided)
        if w_current is not None:
            starts.append(normalize_simplex(w_current))
            
            # Top-K boosted (emphasize top 3 concepts from current)
            top_k = np.argsort(-w_current)[:3]
            w_boost = np.zeros(self.K, dtype=np.float32)
            w_boost[top_k] = 1.0 / len(top_k)
            starts.append(normalize_simplex(w_boost))

        # PRIORITY 3: Dirichlet samples around learned preferences
        # If we have GP knowledge, sample around top candidates
        # Otherwise, sample around w_current
        if len(starts) > 0:
            # Use the best starts as seeds for Dirichlet sampling
            for seed_w in starts[:2]:  # Use top 2 starts as seeds
                # LOWER concentration (20.0) to allow MORE exploration
                # Too high concentration causes all samples to be too similar!
                alpha = 20.0 * normalize_simplex(seed_w)
                for _ in range(2):
                    w_dir = self.rng.dirichlet(alpha + EPS)
                    starts.append(normalize_simplex(w_dir))
        
        # Add some random exploration
        for _ in range(2):
            w_unif = self.rng.dirichlet(np.ones(self.K))
            starts.append(normalize_simplex(w_unif))

        return starts

    def _optimize_acquisition(
        self,
        strategy: str,
        starts: List[np.ndarray],
        pool_size: int,
        neg_indices: List[int],
        current_proposals: List[np.ndarray],
        max_cos: float
    ) -> Optional[np.ndarray]:
        """Optimize acquisition function using given strategy"""

        # Sample pool from starts
        pool = []
        
        # Strategy-specific noise AND pool composition
        if strategy == 'exploit':
            noise_std = 0.15  # Moderate focus (not too tight)
            pct_from_starts = 0.7  # 70% from learned starts
        elif strategy == 'thompson':
            noise_std = 0.3  # Balanced
            pct_from_starts = 0.5  # 50% from starts, 50% random
        elif strategy == 'ei':
            noise_std = 0.35  # Slightly more exploration
            pct_from_starts = 0.4  # 40% from starts
        else:  # diverse
            noise_std = 0.5  # Maximum exploration (back to original!)
            pct_from_starts = 0.3  # Only 30% from starts, 70% random for diversity
        
        # Generate samples from start points
        per_start = max(1, int(pool_size * pct_from_starts) // len(starts))

        for w_start in starts:
            # Convert to logits
            phi_start = np.log(w_start + EPS) * SOFTMAX_TAU_W

            # Add strategy-specific noise
            for _ in range(per_start):
                phi_noisy = phi_start + self.rng.randn(self.K) * noise_std
                w = logit_to_weights(phi_noisy)
                pool.append(w)

        # Fill remaining pool with random samples
        # This proportion varies by strategy (more for diverse, less for exploit)
        while len(pool) < pool_size:
            w_rand = self.rng.dirichlet(np.ones(self.K))
            w = logit_to_weights(np.log(w_rand + EPS))
            pool.append(w)

        pool = np.array(pool[:pool_size])  # (pool_size, K)

        # Compute embeddings
        Z_pool = np.array([compute_mixture_embedding(w, self.MU) for w in pool])

        # Predict utilities
        mu, std = self._predict(Z_pool)

        # Apply negative penalty
        if neg_indices:
            penalty = np.sum(np.maximum(0, pool[:, neg_indices] - NEGATIVE_PENALTY_RHO), axis=1)
            penalty *= NEGATIVE_PENALTY_LAMBDA
            mu = mu - penalty

        # Compute acquisition scores
        if strategy == 'exploit':
            # Pure exploitation - maximize posterior mean (learned preference)
            # This converges toward what the user has selected
            scores = mu
        
        elif strategy == 'thompson':
            # Thompson sampling - sample from posterior
            samples = mu + std * self.rng.randn(len(mu))
            scores = samples

        elif strategy == 'ei':
            # Expected Improvement
            best_mu = np.max([self.gp.predict(c.z.reshape(1, -1))[0]
                             for c in self.candidates.values()]) if self.candidates else 0.0
            scores = self._expected_improvement(mu, std, best_mu)

        elif strategy == 'variance':
            # Max variance (pure exploration)
            scores = std

        elif strategy == 'diverse':
            # Diverse explorer - balance variance and distance from current proposals
            scores = std.copy()
            if current_proposals:
                Z_current = np.array([compute_mixture_embedding(w, self.MU) for w in current_proposals])
                # Penalize candidates close to current proposals
                for z_curr in Z_current:
                    cos_sim = np.dot(Z_pool, z_curr)
                    scores = scores - 0.5 * np.maximum(0, cos_sim - max_cos)

        else:
            scores = mu  # default to posterior mean

        # Select best
        best_idx = np.argmax(scores)
        w_best = pool[best_idx]

        print(f"    Acquisition: score={scores[best_idx]:.4f}, mu={mu[best_idx]:.4f}, std={std[best_idx]:.4f}")

        return w_best

    def _get_best_candidate(self) -> Optional[np.ndarray]:
        """
        Get w_best from GP posterior (highest predicted utility).
        
        Returns:
            w_best: weight vector with highest posterior mean, or None if not fitted
        """
        if not self.fitted or not self.candidates:
            return None
        
        best_cand = None
        best_mu = -np.inf
        
        for cand in self.candidates.values():
            mu, _ = self._predict(cand.z.reshape(1, -1))
            if mu[0] > best_mu:
                best_mu = mu[0]
                best_cand = cand
        
        return best_cand.w if best_cand else None
    
    def _generate_diverse_candidate(
        self,
        w_A: np.ndarray,
        w_B: np.ndarray,
        neg_indices: List[int],
        pool_size: int = 1000
    ) -> Optional[np.ndarray]:
        """
        Generate uncertainty-guided diverse candidate (Candidate C).
        
        High posterior uncertainty + far from A/B.
        
        Args:
            w_A: Candidate A weights
            w_B: Candidate B weights
            neg_indices: indices of negative concepts
            pool_size: number of candidates to sample
        
        Returns:
            w_C: diverse candidate, or None if failed
        """
        # Sample pool: mix of uniform and light bias toward learned weights
        pool = []
        
        # 70% uniform exploration
        for _ in range(int(pool_size * 0.7)):
            w = self.rng.dirichlet(np.ones(self.K))
            pool.append(w)
        
        # 30% biased toward learned weights (but still diverse)
        for _ in range(int(pool_size * 0.3)):
            alpha = 5.0 * (self.concept_weights + EPS)  # low concentration
            w = self.rng.dirichlet(alpha)
            pool.append(w)
        
        pool = np.array(pool)
        
        # Project all to SDXL format
        pool = np.array([project_sdxl(w, top_k=15, jitter=0) for w in pool])
        
        # Compute embeddings
        Z_pool = np.array([compute_mixture_embedding(w, self.MU) for w in pool])
        
        # Predict utilities
        mu, std = self._predict(Z_pool)
        
        # Apply negative penalty
        if neg_indices:
            penalty = np.sum(np.maximum(0, pool[:, neg_indices] - NEGATIVE_PENALTY_RHO), axis=1)
            penalty *= NEGATIVE_PENALTY_LAMBDA
            mu = mu - penalty
        
        # Compute L1 distance from A and B
        Z_A = compute_mixture_embedding(w_A, self.MU)
        Z_B = compute_mixture_embedding(w_B, self.MU)
        
        dist_from_A = np.array([1.0 - np.dot(z, Z_A) for z in Z_pool])  # 1 - cos similarity
        dist_from_B = np.array([1.0 - np.dot(z, Z_B) for z in Z_pool])
        avg_distance = (dist_from_A + dist_from_B) / 2.0
        
        # Score: balance uncertainty and diversity
        # λ controls diversity weight (higher = prefer more distant candidates)
        lambda_diversity = 0.5
        scores = std + lambda_diversity * avg_distance
        
        # Select best
        best_idx = np.argmax(scores)
        w_C = pool[best_idx]
        
        print(f"      Acquisition: std={std[best_idx]:.4f}, mu={mu[best_idx]:.4f}, "
              f"dist_AB={avg_distance[best_idx]:.4f}")
        
        return w_C
    
    def _generate_thompson_candidate(
        self,
        w_best: np.ndarray,
        w_diverse: np.ndarray,
        neg_indices: List[int],
        pool_size: int = 1000
    ) -> Optional[np.ndarray]:
        """
        Generate Thompson/EI candidate (Candidate D).
        
        Sample from posterior to find high-upside candidate.
        
        Args:
            w_best: best candidate so far
            w_diverse: diverse candidate
            neg_indices: indices of negative concepts
            pool_size: number of candidates to sample
        
        Returns:
            w_D: Thompson/EI candidate, or None if failed
        """
        # Create pool around w_best and w_diverse
        pool = []
        
        # 50% around w_best (exploit)
        for _ in range(pool_size // 2):
            alpha = 20.0 * (w_best + EPS)
            w = self.rng.dirichlet(alpha)
            pool.append(w)
        
        # 25% around w_diverse (explore)
        for _ in range(pool_size // 4):
            alpha = 15.0 * (w_diverse + EPS)
            w = self.rng.dirichlet(alpha)
            pool.append(w)
        
        # 25% neutral/uniform
        for _ in range(pool_size // 4):
            w = self.rng.dirichlet(np.ones(self.K))
            pool.append(w)
        
        pool = np.array(pool)
        
        # Project to SDXL format
        pool = np.array([project_sdxl(w, top_k=15, jitter=0) for w in pool])
        
        # Compute embeddings
        Z_pool = np.array([compute_mixture_embedding(w, self.MU) for w in pool])
        
        # Predict utilities
        mu, std = self._predict(Z_pool)
        
        # Apply negative penalty
        if neg_indices:
            penalty = np.sum(np.maximum(0, pool[:, neg_indices] - NEGATIVE_PENALTY_RHO), axis=1)
            penalty *= NEGATIVE_PENALTY_LAMBDA
            mu = mu - penalty
        
        # Thompson sampling: f̃(w) = μ(w) + ξ * σ(w) * ε
        # ξ controls exploration (higher = more exploration)
        # Adaptive: higher in early rounds, lower in later rounds
        num_rounds = len(self.duels) // 3 + 1
        xi = max(1.0, 2.0 - num_rounds * 0.2)  # 2.0 → 1.0
        
        epsilon = self.rng.randn(len(mu))
        f_tilde = mu + xi * std * epsilon
        
        # Select best
        best_idx = np.argmax(f_tilde)
        w_D = pool[best_idx]
        
        print(f"      Acquisition: f̃={f_tilde[best_idx]:.4f}, μ={mu[best_idx]:.4f}, "
              f"σ={std[best_idx]:.4f}, ξ={xi:.2f}")
        
        return w_D

    def _check_diversity(self, proposals: List[np.ndarray], max_cos: float) -> None:
        """Check pairwise diversity of proposals"""
        if len(proposals) < 2:
            return

        Z_props = np.array([compute_mixture_embedding(w, self.MU) for w in proposals])

        print(f"\n[PBO PROPOSE] Diversity check:")
        max_sim = 0.0
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                cos_sim = cosine_similarity(Z_props[i], Z_props[j])
                max_sim = max(max_sim, cos_sim)
                print(f"  cos(prop_{i}, prop_{j}) = {cos_sim:.4f}")

        if max_sim > max_cos:
            print(f"  ⚠️  Warning: Max cosine {max_sim:.4f} > threshold {max_cos:.4f}")
        else:
            print(f"  ✅ Diversity satisfied (max cos: {max_sim:.4f})")

    def best(self) -> Optional[np.ndarray]:
        """Return current best candidate (highest posterior mean)"""
        if not self.fitted or not self.candidates:
            return None

        best_cand = None
        best_mu = -np.inf

        for cand in self.candidates.values():
            mu, _ = self._predict(cand.z.reshape(1, -1))
            if mu[0] > best_mu:
                best_mu = mu[0]
                best_cand = cand

        return best_cand.w if best_cand else None

    def to_dict(self) -> Dict:
        """Serialize state"""
        return {
            'K': self.K,
            'd': self.d,
            'concept_ids': self.concept_ids,
            'num_candidates': len(self.candidates),
            'num_duels': len(self.duels),
            'fitted': self.fitted,
            'candidates': [c.to_dict() for c in self.candidates.values()],
            'duels': [d.to_dict() for d in self.duels]
        }
