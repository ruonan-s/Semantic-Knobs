# PBO Coalescing Bug Fix

## 🐛 **The Bug**

After adding debug logging, we discovered that PBO was **merging all 4 candidates into a single candidate** in every round, preventing the GP from ever fitting.

### Evidence from Logs

```
[PBO] Added candidate round1_img0 (total: 1)
[PBO] Coalescing candidate (cos=0.9989) into round1_img0
[PBO] Coalescing candidate (cos=0.9983) into round1_img0
[PBO] Coalescing candidate (cos=0.9982) into round1_img0

[PBO STATE] After recording selection:
  candidates: 1  ← Should be 4!
  duels: 3

[PBO] Not enough data to fit (candidates=1, duels=3)  ← GP needs ≥2 candidates!
```

### Why It Happened

1. **Cold-start proposals are similar in embedding space:**
   - All 4 proposals based on same learned weights
   - Perturbations (top-heavy, diversified, smoothed) modify weights
   - But after projecting through 768-dim embedding matrix, they're very close
   - Cosine similarities: 0.9989, 0.9983, 0.9982

2. **Coalescing threshold was too aggressive:**
   - Threshold: `COALESCE_COSINE_THRESHOLD = 0.995`
   - All proposals above threshold → merged into one
   
3. **GP cannot fit with 1 candidate:**
   - Requires ≥2 candidates
   - PBO stuck in cold start forever
   - Same proposals every round

---

## ✅ **The Fix**

**Disable coalescing during early rounds** (when candidates < 10).

### Implementation

```python
# backend/pbo.py:331-340
allow_coalescing = len(self.candidates) >= 10  # Allow after ~2-3 rounds

if allow_coalescing:
    for cid, cand in self.candidates.items():
        cos_sim = cosine_similarity(z, cand.z)
        if cos_sim > COALESCE_COSINE_THRESHOLD:
            print(f"[PBO] Coalescing candidate (cos={cos_sim:.4f}) into {cid}")
            return cid
else:
    print(f"[PBO] Coalescing disabled (early rounds: {len(self.candidates)} candidates)")
```

### Why This Works

**Early Rounds (candidates < 10):**
- All proposals added as separate candidates
- Round 1: 4 candidates
- Round 2: 8 candidates (4 + 4)
- Round 3: 12 candidates (8 + 4) → threshold reached
- GP can fit with ≥2 candidates ✅

**Later Rounds (candidates ≥ 10):**
- Coalescing enabled
- True duplicates merged
- Keeps candidate set manageable

---

## 📊 **Expected Behavior After Fix**

### Round 1 (Cold Start)
```
[PBO STATE] Before:
  candidates: 0, duels: 0, fitted: False

[PBO] Added candidate round1_img0 (total: 1)
[PBO] Coalescing disabled (early rounds: 1 candidates)  ← NEW!
[PBO] Added candidate round1_img1 (total: 2)
[PBO] Coalescing disabled (early rounds: 2 candidates)  ← NEW!
[PBO] Added candidate round1_img2 (total: 3)
[PBO] Coalescing disabled (early rounds: 3 candidates)  ← NEW!
[PBO] Added candidate round1_img3 (total: 4)

[PBO] Added duel: round1_img0 ≻ round1_img1
[PBO] Added duel: round1_img0 ≻ round1_img2
[PBO] Added duel: round1_img0 ≻ round1_img3

[PBO STATE] After:
  candidates: 4  ← Now 4 instead of 1!
  duels: 3
```

### Round 2 (GP Can Fit!)
```
[PBO] Fitting GP with 4 candidates, 3 duels  ← Now possible!
[PBO FIT] Learned kernel: CosineRBFKernel(...)
[PBO FIT] Log-marginal-likelihood: -6.220

[PBO STATE] After propose:
  candidates: 8  ← Growing!
  duels: 6
  fitted: True  ← GP fitted!

[PBO PROPOSE] Round 2+ - Principled 4-candidate design
  [A] Anchor/Exploit: w_best from GP
  [B] Local Refinement: Dirichlet around w_best
  [C] Uncertainty-Diverse: High σ, far from A/B
  [D] Thompson Sampling: Posterior sample

[PBO PROPOSE] Generated 4 GP-driven proposals  ← DIFFERENT from Round 1!
```

### Round 3+
```
[PBO STATE]:
  candidates: 12, 16, 20...
  duels: 9, 12, 15...
  fitted: True

Proposals continue to evolve based on GP!
```

---

## 🧪 **Testing the Fix**

### 1. Run Your Session Again

```bash
cd backend
conda activate apl
python server.py 2>&1 | grep -E "\[PBO|candidates|duels|fitted"
```

### 2. Watch for These Logs

**Good (Fixed):**
```
[PBO] Coalescing disabled (early rounds: X candidates)
candidates: 4
candidates: 8
[PBO FIT] Fitting GP with 8 candidates, 6 duels
fitted: True
```

**Bad (Still Broken):**
```
[PBO] Coalescing candidate...
candidates: 1
[PBO] Not enough data to fit (candidates=1...)
fitted: False
```

### 3. Check Weight Evolution

```python
import json, numpy as np

with open("sessions/.../round_1/weights.json") as f:
    r1 = json.load(f)
with open("sessions/.../round_2/weights.json") as f:
    r2 = json.load(f)

# Should be DIFFERENT now!
for i in range(4):
    identical = np.allclose(r1['proposals'][i], r2['proposals'][i])
    print(f"Proposal {i}: {'❌ SAME' if identical else '✅ DIFFERENT'}")
```

---

## 📝 **Summary**

### Root Cause
- Cold-start proposals too similar in embedding space (cos > 0.995)
- All merged into 1 candidate
- GP requires ≥2 candidates to fit
- PBO stuck in cold start forever

### Fix
- Disable coalescing for first ~2-3 rounds (candidates < 10)
- Allows cold-start proposals to be treated as distinct
- GP can fit with multiple candidates
- Learning begins in Round 2

### Files Modified
- `backend/pbo.py:331-340` - Added early-round coalescing check

### Expected Outcome
- ✅ Round 1: 4 distinct candidates added
- ✅ Round 2: GP fits successfully (8 candidates, 6 duels)
- ✅ Round 2+: Proposals evolve based on preferences
- ✅ Each round has DIFFERENT weights

---

## 🚀 **Next Steps**

1. **Restart server** with the fix
2. **Run your session** again
3. **Check server logs** for:
   - `candidates: 4, 8, 12...` (growing)
   - `fitted: True` (after Round 1)
   - `[PBO PROPOSE] Round 2+ - Principled 4-candidate design`
4. **Verify weights differ** between rounds
5. **Enjoy functional PBO!** 🎉

