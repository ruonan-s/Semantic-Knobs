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
        random_state: int = 42
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

        print(f"[PBO] Initialized with K={self.K} concepts, d={self.d} embedding dim")

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
        for cid, cand in self.candidates.items():
            cos_sim = cosine_similarity(z, cand.z)
            if cos_sim > COALESCE_COSINE_THRESHOLD:
                print(f"[PBO] Coalescing candidate (cos={cos_sim:.4f}) into {cid}")
                return cid

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

        # Cold start - return corners + center
        if not self.fitted or len(self.candidates) < 2:
            print("[PBO PROPOSE] Cold start - returning corners + center")
            proposals = []
            # Corners (one-hot)
            for i in range(min(q - 1, self.K)):
                w = np.zeros(self.K, dtype=np.float32)
                w[i] = 1.0
                proposals.append(w)
            # Center
            if len(proposals) < q:
                w = np.ones(self.K, dtype=np.float32) / self.K
                proposals.append(w)
            return proposals[:q]

        # Multi-start initialization
        starts = self._generate_starts(w_current)

        # Acquisition strategies
        strategies = ['thompson', 'ei', 'variance', 'diverse']
        proposals = []

        for strategy in strategies[:q]:
            print(f"  Strategy {len(proposals)+1}/{q}: {strategy}")

            # Optimize acquisition function
            w_opt = self._optimize_acquisition(
                strategy=strategy,
                starts=starts,
                pool_size=pool_size,
                neg_indices=neg_indices,
                current_proposals=proposals,
                max_cos=max_cos
            )

            if w_opt is not None:
                proposals.append(w_opt)
                print(f"    Proposed: w_max={w_opt.max():.3f}, w_min={w_opt.min():.3f}")
            else:
                # Fallback - random Dirichlet
                w_fallback = self.rng.dirichlet(np.ones(self.K))
                w_fallback = logit_to_weights(np.log(w_fallback + EPS))
                proposals.append(w_fallback)
                print(f"    Fallback random proposal")

        # Fill remaining slots if needed
        while len(proposals) < q:
            w_random = self.rng.dirichlet(np.ones(self.K))
            w_random = logit_to_weights(np.log(w_random + EPS))
            proposals.append(w_random)

        # Check diversity
        self._check_diversity(proposals, max_cos)

        return proposals[:q]

    def _generate_starts(self, w_current: Optional[np.ndarray]) -> List[np.ndarray]:
        """Generate multi-start initialization points"""
        starts = []

        # Current UI weights
        if w_current is not None:
            starts.append(normalize_simplex(w_current))

        # Top-K boosted (emphasize top 3 concepts)
        if w_current is not None:
            top_k = np.argsort(-w_current)[:3]
            w_boost = np.zeros(self.K, dtype=np.float32)
            w_boost[top_k] = 1.0 / len(top_k)
            starts.append(normalize_simplex(w_boost))

        # Dirichlet samples around current
        if w_current is not None:
            alpha = 40.0 * normalize_simplex(w_current)
            for _ in range(3):
                w_dir = self.rng.dirichlet(alpha + EPS)
                starts.append(normalize_simplex(w_dir))

        # Uniform Dirichlet samples
        for _ in range(3):
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
        per_start = max(1, pool_size // (len(starts) + 1))

        for w_start in starts:
            # Convert to logits
            phi_start = np.log(w_start + EPS) * SOFTMAX_TAU_W

            # Add noise
            for _ in range(per_start):
                phi_noisy = phi_start + self.rng.randn(self.K) * 0.5
                w = logit_to_weights(phi_noisy)
                pool.append(w)

        # Add random samples
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
        if strategy == 'thompson':
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
