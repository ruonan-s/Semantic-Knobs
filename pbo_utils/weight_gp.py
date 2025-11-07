
# weight_gp.py
# Preference-based optimizer over weight vectors on the simplex.
# - Stores pairwise preferences between weight vectors
# - Derives per-point utilities via Copeland score (wins/total)
# - Fits a surrogate (Gaussian Process if sklearn available; otherwise RBF smoother)
# - Suggests new candidates by Expected Improvement on a Dirichlet sample pool

from __future__ import annotations
import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    HAS_SK = True
except Exception:
    HAS_SK = False

EPS = 1e-8

def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.maximum(0.0, np.asarray(x, dtype=np.float32))
    s = x.sum(axis=-1, keepdims=True)
    return x / (s + EPS)

class _RBFSmoother:
    """Fallback surrogate when sklearn GP is not available."""
    def __init__(self, length_scale: float = 0.3, ridge: float = 1e-3):
        self.l = length_scale
        self.ridge = ridge
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

    def predict(self, X: np.ndarray, return_std: bool = True):
        X = np.asarray(X, dtype=np.float32)
        if self.X is None or len(self.X) == 0:
            mu = np.zeros((len(X),), dtype=np.float32)
            std = np.ones((len(X),), dtype=np.float32)
            return (mu, std) if return_std else mu
        # RBF weights
        d2 = np.sum((X[:,None,:] - self.X[None,:,:])**2, axis=-1)  # (n, m)
        K = np.exp(-0.5 * d2 / (self.l**2)) + self.ridge
        w = K / (K.sum(axis=1, keepdims=True) + EPS)
        mu = (w @ self.y)
        # crude std proxy: distance to nearest neighbor
        nn = np.min(d2, axis=1)
        std = np.sqrt(nn + 1e-6)
        return (mu, std) if return_std else mu

class WeightPBO:
    def __init__(self, dim: int, dirichlet_alpha: float = 1.0, random_state: int = 42):
        self.dim = int(dim)
        self.rng = np.random.RandomState(random_state)
        self.dirichlet_alpha = float(dirichlet_alpha)

        # storage
        self._points: list[np.ndarray] = []    # unique weight vectors tried
        self._index = {}                       # quantized key -> idx
        self._wins  = []                       # wins count per point
        self._loss  = []                       # losses count per point

        # surrogate
        self._gp = None

    # ---- utilities ----
    def _key(self, w: np.ndarray) -> str:
        w = _normalize_simplex(np.asarray(w, dtype=np.float32))
        return "|".join([f"{v:.5f}" for v in w.tolist()])

    def _get_or_add_point(self, w: np.ndarray) -> int:
        k = self._key(w)
        if k in self._index:
            return self._index[k]
        idx = len(self._points)
        self._index[k] = idx
        self._points.append(_normalize_simplex(w))
        self._wins.append(0)
        self._loss.append(0)
        return idx

    # ---- API ----
    def add_preference(self, w_win: np.ndarray, w_lose: np.ndarray):
        """Record that w_win is preferred over w_lose."""
        i = self._get_or_add_point(w_win)
        j = self._get_or_add_point(w_lose)
        self._wins[i] += 1
        self._loss[j] += 1

    def _copeland_scores(self):
        m = len(self._points)
        if m == 0:
            return np.array([]), np.zeros((0,self.dim), dtype=np.float32)
        wins = np.asarray(self._wins, dtype=np.float32)
        loss = np.asarray(self._loss, dtype=np.float32)
        total = wins + loss
        frac = np.where(total > 0, wins / (total + EPS), 0.5)  # [0,1]
        # clip to avoid inf in logit
        frac = np.clip(frac, 0.05, 0.95)
        util = np.log(frac / (1.0 - frac))  # logit
        X = np.vstack(self._points).astype(np.float32)
        return util, X

    def fit(self):
        """Fit surrogate model over the simplex using current pairwise preferences."""
        y, X = self._copeland_scores()
        if len(X) == 0:
            # nothing to fit
            self._gp = None
            return

        if HAS_SK and len(X) >= 2:
            kernel = RBF(length_scale=0.4, length_scale_bounds=(1e-2, 5.0)) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=0)
            try:
                gp.fit(X, y)
                self._gp = gp
                return
            except Exception:
                pass

        # fallback
        smoother = _RBFSmoother(length_scale=0.4, ridge=1e-3)
        smoother.fit(X, y)
        self._gp = smoother

    def _predict(self, W: np.ndarray):
        if self._gp is None:
            mu = np.zeros((len(W),), dtype=np.float32)
            std = np.ones((len(W),), dtype=np.float32)
            return mu, std
        return self._gp.predict(W, return_std=True)

    def _expected_improvement(self, mu: np.ndarray, std: np.ndarray, best_mu: float, xi: float = 0.01):
        std = np.maximum(std, 1e-9)
        z = (mu - best_mu - xi) / std
        from math import sqrt, pi, exp
        # vectorized pdf and cdf of standard normal
        pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * z**2)
        cdf = 0.5 * (1 + erf(z / np.sqrt(2))) if False else 0  # not used; we implement via scipy? avoid
        # Without scipy, approximate CDF using error function series:
        # Use numpy.erf
        import numpy as _np
        cdf = 0.5 * (1.0 + _np.erf(z / np.sqrt(2.0)))
        ei = (mu - best_mu - xi) * cdf + std * pdf
        ei = np.where(std <= 1e-9, 0.0, ei)
        return ei

    def suggest(self, n: int = 4, pool: int = 128) -> np.ndarray:
        """Suggest n new weight vectors on the simplex using EI over a Dirichlet sample pool."""
        if n <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        # cold-start: corners + center until we collect enough data
        if len(self._points) < n:
            cold = np.vstack([np.eye(self.dim), np.ones((1,self.dim))/self.dim])
            return cold[:n].astype(np.float32)

        S = self.rng.dirichlet([self.dirichlet_alpha]*self.dim, size=pool).astype(np.float32)
        mu, std = self._predict(S)
        # best_mu from observed points
        if len(self._points) == 0:
            best_mu = 0.0
        else:
            obs_mu, _ = self._predict(np.vstack(self._points).astype(np.float32))
            best_mu = float(np.max(obs_mu))
        ei = self._expected_improvement(mu, std, best_mu, xi=0.01)
        idx = np.argsort(-ei)[:n]
        return S[idx]

    def best(self, pool: int = 512) -> np.ndarray:
        """Return the current best w according to the surrogate (argmax mu on a pool)."""
        if len(self._points) > 0:
            X = np.vstack(self._points).astype(np.float32)
            mu, _ = self._predict(X)
            return X[int(np.argmax(mu))]
        S = self.rng.dirichlet([self.dirichlet_alpha]*self.dim, size=pool).astype(np.float32)
        mu, _ = self._predict(S)
        return S[int(np.argmax(mu))]
