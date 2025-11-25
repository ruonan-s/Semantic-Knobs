from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Kernel, WhiteKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[PBO] Warning: sklearn not available. Using fallback surrogate.")

# ----------------------------------------------------------------------------
# Logging setup (plugin can override; script __main__ will configure basicConfig)
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ============================================================================
# Parameters
# ============================================================================
KERNEL_LENGTH_SCALE = 0.6
KERNEL_SIGMA_F = 1.0

NEGATIVE_PENALTY_RHO = 0.03
NEGATIVE_PENALTY_LAMBDA = 10.0

POOL_SIZE = 1024
BATCH_DIVERSITY_MAX_COS = 0.95
MAX_CANDIDATES = 200
PER_CONCEPT_CAP = 0.35

EPS = 1e-8


# ============================================================================
# Cosine-RBF Kernel
# ============================================================================
class CosineRBFKernel(Kernel):
    """Cosine-RBF kernel: k(z,z') = σ²_f * exp(-(1 - cos(z,z')) / ℓ)"""

    def __init__(self, length_scale=KERNEL_LENGTH_SCALE, sigma_f=KERNEL_SIGMA_F):
        self.length_scale = length_scale
        self.sigma_f = sigma_f

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        cos_sim = np.dot(X, Y.T)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        cos_dist = 1.0 - cos_sim
        K = self.sigma_f ** 2 * np.exp(-cos_dist / self.length_scale)

        if eval_gradient:
            return K, np.zeros((K.shape[0], K.shape[1], 2))
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
    id: str
    w: np.ndarray  # (K,)
    z: np.ndarray  # (d,)

    def to_dict(self):
        return {
            "id": self.id,
            "w": self.w.tolist(),
            "z": self.z.tolist(),
        }


@dataclass
class Duel:
    better_id: str
    worse_id: str
    strength: float = 1.0  # 0.5 weak, 1.0 strong

    def to_dict(self):
        return asdict(self)


# ============================================================================
# Helper Functions
# ============================================================================
def normalize_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, np.asarray(w, dtype=np.float32))
    s = w.sum()
    if s <= EPS:
        return np.ones_like(w, dtype=np.float32) / len(w)
    return w / s


def compute_mixture_embedding(w: np.ndarray, MU: np.ndarray) -> np.ndarray:
    """
    z = L2_normalize(w @ MU)
    MU: (K, d), rows are assumed L2-normalized
    """
    w = normalize_simplex(w)
    z = np.dot(w, MU)  # (d,)
    norm = np.linalg.norm(z)
    if norm < EPS:
        z = np.zeros_like(z)
        z[0] = 1.0
        return z
    return z / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    )


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
        alpha_scale = min(8.0 + 2.0 * num_rounds, 20.0)  # 8 → 20
    
    # Dirichlet concentration
    alpha = alpha_scale * (w_base + 1e-6)
    alpha = np.maximum(alpha, 1e-6)  # ensure all alphas > 0
    w_local = rng.dirichlet(alpha)
    
    # Project to SDXL format
    return project_sdxl(w_local, top_k=top_k, jitter=0.005)


# ============================================================================
# Debug helpers
# ============================================================================
def format_topk(w: np.ndarray, concept_ids: List[str], k: int = 5) -> str:
    """Return a string like: 'tag1=0.31, tag7=0.22, ...' for top-k weights."""
    w = normalize_simplex(w)
    idx = np.argsort(-w)[:k]
    parts = [f"{concept_ids[i]}={w[i]:.3f}" for i in idx]
    return ", ".join(parts)


# ============================================================================
# PBO Class
# ============================================================================
class PBO:
    """
    Preferential Bayesian Optimization over concept weight mixtures,
    with a single exploration–exploitation acquisition:

        score(w) = mu(w) + lambda_sigma * sigma(w) - penalty_neg(w)

    Returns *raw* weight vectors on the simplex. You can project them
    to SDXL format (top-K, jitter, etc.) outside this class.
    """

    def __init__(
        self,
        MU: np.ndarray,                 # (K, d) concept centroids
        concept_ids: List[str],         # length-K list of IDs
        kernel_length_scale: float = KERNEL_LENGTH_SCALE,
        kernel_sigma_f: float = KERNEL_SIGMA_F,
        random_state: int = 42,
        concept_weights: Optional[np.ndarray] = None,  # for warm start / cold start prior
    ):
        self.MU = np.asarray(MU, dtype=np.float32)
        self.K, self.d = self.MU.shape
        self.concept_ids = concept_ids
        self.rng = np.random.RandomState(random_state)

        self.length_scale = kernel_length_scale
        self.sigma_f = kernel_sigma_f

        self.candidates: Dict[str, Candidate] = {}
        self.duels: List[Duel] = []

        self.gp: Optional[GaussianProcessRegressor] = None
        self.fitted: bool = False

        self._cid_counter = 0

        # Prior concept weights (e.g., from tag frequencies or exploration)
        if concept_weights is not None:
            self.concept_weights = normalize_simplex(
                np.asarray(concept_weights, dtype=np.float32)
            )
            logger.info("[PBO] Warm start with provided concept weights.")
        else:
            self.concept_weights = np.ones(self.K, dtype=np.float32) / self.K
            logger.info("[PBO] Cold start with uniform concept weights.")

        logger.info(f"[PBO] Initialized with K={self.K}, d={self.d}")

    # ------------------------------------------------------
    # Candidate & preference management
    # ------------------------------------------------------
    def _generate_candidate_id(self) -> str:
        self._cid_counter += 1
        return f"cand_{self._cid_counter:04d}"

    def compute_mixture_embedding(self, w: np.ndarray) -> np.ndarray:
        return compute_mixture_embedding(w, self.MU)

    def add_candidate(self, w: np.ndarray, candidate_id: Optional[str] = None) -> str:
        w = normalize_simplex(w)
        z = compute_mixture_embedding(w, self.MU)

        if candidate_id is None:
            candidate_id = self._generate_candidate_id()

        self.candidates[candidate_id] = Candidate(id=candidate_id, w=w, z=z)
        logger.debug(f"[PBO] Added candidate {candidate_id} | top tags: {format_topk(w, self.concept_ids)}")

        if len(self.candidates) > MAX_CANDIDATES:
            self._prune_candidates()

        return candidate_id

    def add_preference(self, better_id: str, worse_id: str, strength: float = 1.0) -> None:
        if better_id not in self.candidates:
            logger.warning(f"[PBO] better_id={better_id} not found.")
            return
        if worse_id not in self.candidates:
            logger.warning(f"[PBO] worse_id={worse_id} not found.")
            return

        self.duels.append(Duel(better_id=better_id, worse_id=worse_id, strength=strength))
        logger.debug(f"[PBO] Added duel: {better_id} ≻ {worse_id} (strength={strength})")

    def _prune_candidates(self) -> None:
        if len(self.candidates) <= MAX_CANDIDATES:
            return
        # Keep most recent by ID ordering
        sorted_items = sorted(self.candidates.items(), key=lambda kv: kv[0], reverse=True)
        keep_ids = {cid for cid, _ in sorted_items[:MAX_CANDIDATES]}
        before = len(self.candidates)
        self.candidates = {cid: cand for cid, cand in self.candidates.items() if cid in keep_ids}
        self.duels = [
            d for d in self.duels
            if d.better_id in self.candidates and d.worse_id in self.candidates
        ]
        logger.info(f"[PBO] Pruned candidates from {before} → {len(self.candidates)}")

    # ------------------------------------------------------
    # GP fitting
    # ------------------------------------------------------
    def fit(self) -> None:
        """
        Fit GP surrogate using Copeland-score-style utilities derived from duels.
        """
        if len(self.candidates) < 2 or len(self.duels) == 0:
            logger.info("[PBO] Not enough data to fit GP.")
            self.fitted = False
            return

        cand_list = list(self.candidates.values())
        Z = np.vstack([c.z for c in cand_list])

        # wins / losses
        wins = defaultdict(float)
        losses = defaultdict(float)
        for d in self.duels:
            wins[d.better_id] += d.strength
            losses[d.worse_id] += d.strength

        # utility via logit(win_rate)
        y = np.zeros(len(cand_list), dtype=np.float32)
        win_rates = []
        for i, c in enumerate(cand_list):
            w = wins.get(c.id, 0.0)
            l = losses.get(c.id, 0.0)
            total = w + l
            if total > 0:
                frac = w / (total + EPS)
                frac = np.clip(frac, 0.05, 0.95)
                win_rates.append(frac)
                y[i] = np.log(frac / (1.0 - frac))
            else:
                win_rates.append(0.5)
                y[i] = 0.0

        win_rates = np.array(win_rates)
        logger.info(
            f"[PBO] Fitting GP on {len(cand_list)} candidates, {len(self.duels)} duels. "
            f"Utility range [{y.min():.3f}, {y.max():.3f}], "
            f"win-rate mean={win_rates.mean():.3f}, std={win_rates.std():.3f}"
        )

        if not HAS_SKLEARN:
            logger.warning("[PBO] sklearn not available; GP disabled.")
            self.fitted = False
            return

        kernel = CosineRBFKernel(length_scale=self.length_scale, sigma_f=self.sigma_f)
        kernel = kernel + WhiteKernel(noise_level=1e-3)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=0,
            n_restarts_optimizer=2,
        )
        try:
            self.gp.fit(Z, y)
            self.fitted = True
            logger.info(
                f"[PBO] GP fitted. Log-marginal-likelihood = "
                f"{self.gp.log_marginal_likelihood_value_:.3f}"
            )
            logger.debug(f"[PBO] Learned kernel: {self.gp.kernel_}")
        except Exception as e:
            logger.error(f"[PBO] GP fit failed: {e}")
            self.fitted = False

    def _predict(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict (mu, std) for embeddings Z.
        """
        Z = np.atleast_2d(Z)
        if not self.fitted or self.gp is None:
            mu = np.zeros(Z.shape[0], dtype=np.float32)
            std = np.ones(Z.shape[0], dtype=np.float32)
            return mu, std
        mu, std = self.gp.predict(Z, return_std=True)
        return mu, std

    # ------------------------------------------------------
    # Acquisition: pool sampling + single score
    # ------------------------------------------------------
    def _sample_pool(
        self,
        pool_size: int,
        w_best: Optional[np.ndarray],
        w_current: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build a pool of candidate weight vectors.
        - Some around w_best (if any)
        - Some around w_current (if provided)
        - Remaining from broad Dirichlet / prior concept_weights
        """
        pool: List[np.ndarray] = []

        # 1) Around w_best
        if w_best is not None:
            w_best = normalize_simplex(w_best)
            alpha_best = 25.0 * w_best  # fairly concentrated
            for _ in range(pool_size // 3):
                pool.append(normalize_simplex(self.rng.dirichlet(alpha_best + EPS)))

        # 2) Around w_current (UI slider)
        if w_current is not None:
            w_current = normalize_simplex(w_current)
            alpha_curr = 20.0 * w_current
            for _ in range(pool_size // 6):
                pool.append(normalize_simplex(self.rng.dirichlet(alpha_curr + EPS)))

        # 3) Around concept_weights prior
        alpha_prior = 10.0 * self.concept_weights
        for _ in range(pool_size // 3):
            pool.append(normalize_simplex(self.rng.dirichlet(alpha_prior + EPS)))

        # 4) Pure exploration
        while len(pool) < pool_size:
            pool.append(normalize_simplex(self.rng.dirichlet(np.ones(self.K))))

        return np.array(pool[:pool_size], dtype=np.float32)

    def _sample_pool_cold_start(
        self,
        pool_size: int,
        w_current: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Cold-start pool builder with heavier bias toward learned priors.
        """
        pool: List[np.ndarray] = []

        # 1) Around current UI weights (if provided) - ~10%
        if w_current is not None:
            w_current = normalize_simplex(w_current)
            alpha_curr = 20.0 * (w_current + EPS)
            num_curr = max(1, pool_size // 10)
            for _ in range(num_curr):
                pool.append(normalize_simplex(self.rng.dirichlet(alpha_curr)))

        # 2) Around concept_weights prior - ~60%
        alpha_prior = 15.0 * (self.concept_weights + EPS)
        num_prior = max(1, int(pool_size * 0.6))
        for _ in range(num_prior):
            pool.append(normalize_simplex(self.rng.dirichlet(alpha_prior)))

        # 3) Pure exploration - remaining samples (~30%)
        while len(pool) < pool_size:
            pool.append(normalize_simplex(self.rng.dirichlet(np.ones(self.K))))

        return np.array(pool[:pool_size], dtype=np.float32)

    def _evaluate_pool(
        self,
        pool: np.ndarray,
        neg_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For a set of weight vectors pool (N, K):
        - Compute embeddings Z_pool
        - Predict mu, std
        - Apply negative concept penalty to mu
        """
        Z_pool = np.vstack([compute_mixture_embedding(w, self.MU) for w in pool])
        mu, std = self._predict(Z_pool)

        if neg_indices:
            # penalty if negative concepts exceed small threshold
            excess = np.maximum(0.0, pool[:, neg_indices] - NEGATIVE_PENALTY_RHO)
            penalty = NEGATIVE_PENALTY_LAMBDA * np.sum(excess, axis=1)
            mu = mu - penalty

        return pool, Z_pool, mu, std

    def _select_diverse_top_k(
        self,
        pool: np.ndarray,
        Z_pool: np.ndarray,
        scores: np.ndarray,
        k: int,
        max_cos: float,
    ) -> List[np.ndarray]:
        """
        Greedy selection of top-k candidates with cosine diversity constraint.
        """
        order = np.argsort(-scores)  # descending score
        selected_indices: List[int] = []

        for idx in order:
            if len(selected_indices) >= k:
                break
            z = Z_pool[idx]

            # Check cosine similarity to already selected
            too_close = False
            for j in selected_indices:
                cos_sim = cosine_similarity(z, Z_pool[j])
                if cos_sim > max_cos:
                    too_close = True
                    break

            if not too_close:
                selected_indices.append(idx)

        # If we couldn't satisfy diversity for all k, fill remaining
        if len(selected_indices) < k:
            for idx in order:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= k:
                        break

        return [pool[i] for i in selected_indices]

    # ------------------------------------------------------
    # Public proposal API
    # ------------------------------------------------------
    def propose_batch(
        self,
        q: int = 4,
        negatives: Optional[set] = None,
        pool_size: int = POOL_SIZE,
        max_cos: float = BATCH_DIVERSITY_MAX_COS,
        w_current: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Propose a batch of q weight vectors using a single acquisition:

            score(w) = mu(w) + lambda_sigma * std(w) - penalty_neg(w)

        Early rounds: higher lambda_sigma → more exploration.
        Later rounds: lower lambda_sigma → more exploitation.
        """
        if negatives is None:
            negatives = set()

        neg_indices = [i for i, cid in enumerate(self.concept_ids) if cid in negatives]

        # Approximate "round number" from number of duels
        num_rounds = max(1, len(self.duels) // 3)

        # Exploration–exploitation schedule:
        # round 1: ~1.5, round 5: ~1.0, floor at 0.3
        lambda_sigma = max(0.3, 1.5 - 0.1 * (num_rounds - 1))

        logger.info(
            f"[PBO] propose_batch: q={q}, pool_size={pool_size}, "
            f"num_rounds≈{num_rounds}, lambda_sigma={lambda_sigma:.3f}, "
            f"negatives={len(neg_indices)}, fitted={self.fitted}"
        )

        # Cold-start: if no GP, use pure exploration (sigma = 1 everywhere)
        if not self.fitted or self.gp is None:
            logger.info("[PBO] propose_batch: GP not fitted yet → pure exploration.")
            pool = self._sample_pool_cold_start(pool_size, w_current=w_current)
            pool, Z_pool, mu, std = self._evaluate_pool(pool, neg_indices)
            prior_alignment = np.array(
                [np.dot(normalize_simplex(w), self.concept_weights) for w in pool],
                dtype=np.float32,
            )
            lambda_prior = 0.8
            scores = (1.0 - lambda_prior) * std + lambda_prior * prior_alignment
            logger.info(
                f"[PBO] Cold-start scoring | prior_align mean={prior_alignment.mean():.3f} "
                f"max={prior_alignment.max():.3f}"
            )
        else:
            w_best = self.best()
            if w_best is not None:
                logger.info(f"[PBO] Current best by GP: {format_topk(w_best, self.concept_ids)}")
            pool = self._sample_pool(pool_size, w_best=w_best, w_current=w_current)
            pool, Z_pool, mu, std = self._evaluate_pool(pool, neg_indices)
            scores = mu + lambda_sigma * std

        logger.debug(
            f"[PBO] Pool stats: mu[{mu.min():.3f}, {mu.max():.3f}], "
            f"std[{std.min():.3f}, {std.max():.3f}], "
            f"score[{scores.min():.3f}, {scores.max():.3f}]"
        )

        proposals = self._select_diverse_top_k(pool, Z_pool, scores, q, max_cos)

        for i, w in enumerate(proposals):
            logger.info(f"[PBO] Proposal {i}: {format_topk(w, self.concept_ids)}")

        return proposals

    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------
    def best(self) -> Optional[np.ndarray]:
        """
        Return the weight vector of the currently best candidate (max mu).
        """
        if not self.fitted or not self.candidates:
            return None

        best_mu = -np.inf
        best_w = None
        for cand in self.candidates.values():
            mu, _ = self._predict(cand.z.reshape(1, -1))
            if mu[0] > best_mu:
                best_mu = mu[0]
                best_w = cand.w
        return best_w

    def to_dict(self) -> Dict:
        return {
            "K": self.K,
            "d": self.d,
            "concept_ids": self.concept_ids,
            "num_candidates": len(self.candidates),
            "num_duels": len(self.duels),
            "fitted": self.fitted,
            "candidates": [c.to_dict() for c in self.candidates.values()],
            "duels": [d.to_dict() for d in self.duels],
        }


# ============================================================================
# Simulation script for sanity-checking PBO behavior
# ============================================================================
def _simulate_pbo(num_rounds: int = 8, K: int = 20, d: int = 32, save_debug: bool = True) -> None:
    """
    Synthetic test:
    - Random MU
    - True underlying preference over a subset of concepts
    - Simulate a user that always chooses the proposal with maximum dot(true_w, w)
    - Check if PBO best() drifts toward true_w over rounds
    
    Args:
        num_rounds: Number of rounds to simulate
        K: Number of concepts
        d: Embedding dimension
        save_debug: If True, save debug log to backend/sessions/debug_logs/
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    logger.info("=== Starting PBO simulation ===")

    # Random concept centroids (L2-normalized)
    MU = np.random.randn(K, d).astype(np.float32)
    MU /= np.linalg.norm(MU, axis=1, keepdims=True) + EPS
    concept_ids = [f"c{i}" for i in range(K)]

    # True user preference: favor 3 random concepts
    true_idx = np.random.choice(K, size=3, replace=False)
    true_w = np.zeros(K, dtype=np.float32)
    true_w[true_idx] = 1.0
    true_w = normalize_simplex(true_w)

    logger.info(
        "True user favorites: " +
        ", ".join(f"{concept_ids[i]} (idx={i})" for i in true_idx)
    )
    logger.info("True w top tags: " + format_topk(true_w, concept_ids))

    pbo = PBO(MU, concept_ids, concept_weights=np.ones(K, dtype=np.float32) / K)

    # Initialize debug log structure
    session_id = f"[pbo_sim]_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    debug_log = {
        "session_id": session_id,
        "stage": "simulation",
        "events": []
    }
    
    # Add initialization event
    concepts = [
        {
            "id": cid,
            "label": cid,
            "member_count": 1,
            "member_tags": []
        }
        for cid in concept_ids
    ]
    
    initial_weights = {
        cid: {
            "w": 1.0 / K,
            "like_count": 0,
            "dislike_count": 0
        }
        for cid in concept_ids
    }
    
    debug_log["events"].append({
        "timestamp": datetime.now().isoformat(),
        "event_type": "initialization",
        "data": {
            "total_concepts": K,
            "concepts": concepts,
            "initial_weights": initial_weights,
            "true_favorites": [concept_ids[i] for i in true_idx]
        }
    })

    for r in range(num_rounds):
        logger.info(f"\n===== ROUND {r+1} / {num_rounds} =====")

        # 1) Propose batch
        proposals_w = pbo.propose_batch(q=4)

        # 2) Register as candidates, log each
        cand_ids = []
        for j, w in enumerate(proposals_w):
            cid = pbo.add_candidate(w)
            cand_ids.append(cid)
            logger.info(
                f"[Round {r+1}] Candidate {j} (id={cid}) top tags: "
                f"{format_topk(w, concept_ids)}"
            )

        # 3) Simulated user choice: argmax dot(true_w, w)
        scores_true = [float(np.dot(true_w, w)) for w in proposals_w]
        winner_idx = int(np.argmax(scores_true))
        winner_cid = cand_ids[winner_idx]
        winner_w = proposals_w[winner_idx]

        logger.info(
            f"[Round {r+1}] Simulated user picks candidate {winner_idx} "
            f"(id={winner_cid}), true-score={scores_true[winner_idx]:.4f}"
        )

        # Log image selection event (simulated)
        debug_log["events"].append({
            "timestamp": datetime.now().isoformat(),
            "event_type": "image_selection",
            "data": {
                "image_id": f"proposal_{winner_idx}",
                "candidate_id": winner_cid,
                "true_score": scores_true[winner_idx],
                "concepts_boosted": [
                    concept_ids[i] for i in np.argsort(-winner_w)[:5]
                ],
                "concept_count": int(np.sum(winner_w > 0.01))
            }
        })

        # 4) Log duels: winner ≻ others
        for j, cid in enumerate(cand_ids):
            if j == winner_idx:
                continue
            pbo.add_preference(winner_cid, cid, strength=1.0)

        # 5) Fit GP after this round
        pbo.fit()

        # 6) Check current best and log categorization
        best_w = pbo.best()
        if best_w is not None:
            align = float(np.dot(true_w, normalize_simplex(best_w)))
            logger.info(
                f"[Round {r+1}] PBO best top tags: {format_topk(best_w, concept_ids)}"
            )
            logger.info(
                f"[Round {r+1}] Alignment with true prefs (dot product): {align:.4f}"
            )
            
            # Log categorization event with concept details
            w_base = 1.0 / K
            delta = 0.2 * w_base  # 20% threshold for categorization
            threshold_positive = w_base + delta
            threshold_negative = w_base - delta
            
            concept_details = []
            for i, cid in enumerate(concept_ids):
                w = float(best_w[i])
                if w > threshold_positive:
                    category = "positive"
                elif w < threshold_negative:
                    category = "negative"
                else:
                    category = "neutral"
                
                concept_details.append({
                    "id": cid,
                    "label": cid,
                    "w": w,
                    "w_base": w_base,
                    "delta": delta,
                    "threshold_positive": threshold_positive,
                    "threshold_negative": threshold_negative,
                    "computed_category": category,
                    "in_positive_list": category == "positive",
                    "in_neutral_list": category == "neutral",
                    "in_negative_list": category == "negative"
                })
            
            categorized_ids = {
                "positive": [d["id"] for d in concept_details if d["computed_category"] == "positive"],
                "neutral": [d["id"] for d in concept_details if d["computed_category"] == "neutral"],
                "negative": [d["id"] for d in concept_details if d["computed_category"] == "negative"]
            }
            
            debug_log["events"].append({
                "timestamp": datetime.now().isoformat(),
                "event_type": "categorization",
                "data": {
                    "round": r + 1,
                    "K": K,
                    "w_base": w_base,
                    "delta": delta,
                    "threshold_positive": threshold_positive,
                    "threshold_negative": threshold_negative,
                    "categorized_counts": {
                        "positive": len(categorized_ids["positive"]),
                        "neutral": len(categorized_ids["neutral"]),
                        "negative": len(categorized_ids["negative"])
                    },
                    "categorized_ids": categorized_ids,
                    "concept_details": concept_details,
                    "alignment_with_true": align,
                    "gp_fitted": pbo.fitted,
                    "num_candidates": len(pbo.candidates),
                    "num_duels": len(pbo.duels)
                }
            })
        else:
            logger.info(f"[Round {r+1}] No best() yet (GP not fitted?).")

    logger.info("=== PBO simulation complete ===")
    
    # Save debug log to file
    if save_debug:
        debug_logs_dir = Path("backend/sessions/debug_logs")
        debug_logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{session_id}_simulation_{timestamp_str}.json"
        log_path = debug_logs_dir / log_filename
        
        with open(log_path, "w") as f:
            json.dump(debug_log, f, indent=2)
        
        logger.info(f"\n✅ Debug log saved to: {log_path}")
        logger.info(f"   Run analyzer: python backend/analyze_pbo_debug.py {log_path}")


if __name__ == "__main__":
    # Basic logging setup for running as a script.
    # In your plugin, you can configure logging differently or ignore this.
    logging.basicConfig(
        level=logging.INFO,  # change to DEBUG for more detail
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _simulate_pbo()
